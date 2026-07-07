//! The LDE-streaming HOST STASH for the gate_air main tree (`results/LDE_STREAMING_SCOPE.md`,
//! route (c) shared bulk), gated behind `GATE_AIR_STREAM_COMMIT` / `GATE_AIR_FUSED_INTERP` and
//! DEFAULT OFF.
//!
//! When the flags are OFF, nothing here is touched and both `evaluate_polynomials` (poly.rs) and
//! `build_leaves` (blake2s.rs) run their exact current code paths — byte-for-byte unchanged.
//!
//! When streaming is ON, the CUDA `evaluate_polynomials` fused (interpolate-in-commit) group drives
//! extend+NTT ONE COLUMN AT A TIME and, right after each eval column is produced, HOST-STAGES it:
//! its bytes are D2H-copied into the host stash and the device buffer is freed
//! (`dehydrate_column`). This drops the commit peak below 40 GB so 2^24/2^25 base shards fit a
//! 40 GB A100. The (dehydrated) columns are still appended to `values_list`, so tree shape and
//! downstream indexing are unchanged. The leaf hashing is done later by `build_leaves`
//! (blake2s.rs), which REHYDRATES each staged column from this stash (`rehydrate_owned`) before
//! absorbing it — see the heterogeneous rehydrate path there. Post-commit readers (OODS/quotient,
//! decommit) likewise rehydrate/host-recover on demand.
//!
//! SOUNDNESS: the staged bytes are the exact committed bytes; a reader that rehydrates observes the
//! identical committed values, and `build_leaves`'s per-column absorb (rehydrate, absorb, free) is
//! byte-identical to the resident multi-column absorb by absorb-associativity. `is_staged` /
//! `rehydrate_*` FAIL LOUD on a missing key, never silently returning wrong data.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::core::fields::m31::BaseField;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::bindings;

/// Max CUDA devices for in-process multi-GPU base proving ("option A"). MUST match the C-side
/// `MAX_CUDA_DEVICES` in `cuda/cuda_mem_pool.cuh` (16), which sizes `g_mem_pool_table`. The
/// per-device streaming-globals tables below (`PINNED_POOL_TABLE`, `STASH_COPIES_PENDING`,
/// `DEFERRED_FREE`, `FREE_EVENT_RING`) are all sized by this and indexed by the caller's current
/// device ordinal (`cur_device()`), exactly like the mem pool. A device ordinal >= this const would
/// index out of bounds, but the C mem pool clamps to 0 on such a device, so `cur_device()` clamps
/// identically (they must agree so a thread's pool slot and its streaming-globals slot are the same
/// device).
const MAX_CUDA_DEVICES: usize = 16;

/// The calling thread's current CUDA device ordinal, clamped to `[0, MAX_CUDA_DEVICES)`. This is the
/// per-device table index for every streaming global below. It mirrors the C-side
/// `cuda_mem_pool_current_device()`: a producer thread has already bound its device via
/// `set_base_gpu(gpu)` (thread-local cudarc ordinal + `cudaSetDevice`), so `cuda_get_device()`
/// returns that ordinal with no extra plumbing. Clamps to 0 on error / out-of-range so it can never
/// index out of bounds and so the single-device path always lands on slot 0 (byte-identical).
///
/// BYTE-IDENTITY (N=1): one producer on device 0 => `cuda_get_device()` returns 0 => every table
/// access indexes slot [0], which holds the exact same lazily-created stream/pool/ring/flag the old
/// single global held. Pure "which device slot" plumbing; never changes WHAT is committed.
fn cur_device() -> usize {
    // SAFETY: FFI; `cuda_get_device` is a thin `cudaGetDevice` wrapper returning 0 on error.
    let ord = unsafe { bindings::cuda_get_device() };
    if ord < 0 || ord as usize >= MAX_CUDA_DEVICES {
        0
    } else {
        ord as usize
    }
}

/// PART B0: a stash entry's host bytes, either PAGEABLE (`Vec<u32>`, the default) or PINNED
/// (page-locked `cudaHostAlloc` buffer, behind `GATE_AIR_PIN_STASH`). Pinned host memory lets the
/// D2H (dehydrate) and H2D (rehydrate) copies run at PCIe DMA speed (~12 GB/s) instead of the
/// ~1.3 GB/s pageable staging copy, cutting the tree1 streaming-copy tax without any async logic.
///
/// Both variants expose the exact same `&[u32]` view (via `Deref`), so every reader
/// (`rehydrate_owned`/`rehydrate_block`/`staged_host_ptr`/`host_batch_get`/`is_staged`) is
/// byte-identical regardless of allocation kind — pinning changes only copy speed, not bytes.
enum StashEntry {
    Pageable(Vec<u32>),
    /// A `size`-long page-locked host buffer from `cuda_alloc_pinned_host_uint32_t`; freed with
    /// `cuda_free_pinned_host` (NOT dropped as a Vec) in `Drop`. (B0 per-column path.)
    Pinned {
        ptr: *mut u32,
        len: usize,
    },
    /// PART B1: a `len`-long page-locked buffer drawn from the persistent `PINNED_POOL`. On `Drop`
    /// it is RETURNED to the pool free-list (keyed by `cap`), NOT `cudaFreeHost`d, so the next run
    /// reuses the same page-locked pages (no per-run re-page-lock). `cap == len` always here.
    Pooled {
        ptr: *mut u32,
        len: usize,
        cap: usize,
    },
}

// SAFETY: the `Pinned`/`Pooled` raw pointer is a page-locked buffer owned solely by this entry
// (freed / recycled exactly once in `Drop`). The stash is a process-global `Mutex<HashMap>` read
// from rayon worker
// threads (see the PROCESS-GLOBAL note below), so entries must be `Send`/`Sync`. The buffer is
// only ever READ (as staged committed bytes) after `dehydrate_column` populates it and before
// `clear_stash`; never mutated concurrently. This matches the `Vec<u32>` variant's own Send/Sync.
unsafe impl Send for StashEntry {}
unsafe impl Sync for StashEntry {}

impl Deref for StashEntry {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        match self {
            StashEntry::Pageable(v) => v.as_slice(),
            // SAFETY: `ptr` is a valid page-locked allocation of exactly `len` u32s, live until
            // this entry drops; the slice is only read as committed bytes.
            StashEntry::Pinned { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            // SAFETY: same — a page-locked pool buffer of exactly `len` u32s, live until Drop.
            StashEntry::Pooled { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

impl Drop for StashEntry {
    fn drop(&mut self) {
        match self {
            StashEntry::Pinned { ptr, .. } => {
                // SAFETY: `ptr` came from `cuda_alloc_pinned_host_uint32_t` (cudaHostAlloc) and is
                // freed exactly once here; the `Vec` variant frees itself normally.
                unsafe {
                    bindings::cuda_free_pinned_host(*ptr as *const std::ffi::c_void);
                }
            }
            StashEntry::Pooled { ptr, cap, .. } => {
                // B1: return the page-locked buffer to the pool free-list (do NOT cudaFreeHost) so
                // the next run reuses it. Recycling, not freeing — zero steady-state page-lock.
                // Per-device: return to the CURRENT DEVICE's pool (Drop runs on the same device-bound
                // thread that took the buffer, so the ordinal matches the `take`).
                with_pool(|p| p.give(*ptr, *cap));
            }
            StashEntry::Pageable(_) => {}
        }
    }
}

/// Reads `GATE_AIR_PIN_STASH` (PART B0, opt-in, DEFAULT OFF). When set, `dehydrate_column` stages
/// each column into a PINNED (page-locked) host buffer instead of a pageable `Vec<u32>`, so the
/// dehydrate D2H and every rehydrate H2D run at DMA speed. Byte-identical either way (only copy
/// speed changes); default OFF keeps the current pageable behavior for safe A/B fallback.
///
/// PART B1 relationship: `GATE_AIR_ASYNC_STASH` (see `async_stash_enabled`) IMPLIES pinning — its
/// persistent recycled pinned pool + async copies subsume B0's per-column `cudaHostAlloc`. So B0
/// (pin only) and B1/B2 (pin-pool + async) are independently A/B-able: set `GATE_AIR_PIN_STASH`
/// alone for B0, `GATE_AIR_ASYNC_STASH` alone for B1/B2 (which also pins), both to compare against
/// the pageable baseline. B1/B2 supersede B0's re-page-lock-every-run behavior.
fn pin_stash_enabled() -> bool {
    std::env::var("GATE_AIR_PIN_STASH").is_ok() || async_stash_enabled()
}

/// Reads `GATE_AIR_ASYNC_STASH` (PART B1/B2, opt-in, DEFAULT OFF). When set, the streamed-commit
/// stash is served from a PERSISTENT, RECYCLED pinned-buffer pool (allocated once, reused across
/// the 188 columns AND across runs — no per-column `cudaHostAlloc`, no ~12 GB re-page-lock/run that
/// B0 paid), and the D2H (dehydrate, producer) is issued double-buffered on a dedicated copy stream
/// with CUDA events so copy N overlaps compute N+1 (the device free-after-D2H waits on the D2H
/// event, hidden behind the next column's compute). The rehydrate readers (OODS/quotient) H2D from
/// the pinned pool via a NO-REDUNDANT-MEMSET async copy on a copy stream (fixing OODS: B0 left it
/// at 10.7 s because the alloc+copy path issued TWO dead full-column device memsets serialized on
/// stream 0). Byte-identical: every consumer event-syncs before it reads. Default OFF for A/B vs.
/// B0.
fn async_stash_enabled() -> bool {
    std::env::var("GATE_AIR_ASYNC_STASH").is_ok() || async_stash_batched_enabled()
}

/// Reads `GATE_AIR_ASYNC_STASH_BATCHED` (PART B3, opt-in, DEFAULT OFF). When set, the dehydrate D2Hs
/// are PIPELINED so the 188 async copies run back-to-back at pinned bandwidth instead of serializing
/// on the host: each column's device-buffer free is enqueued STREAM-ORDERED on the copy stream after
/// its own D2H (`cuda_free_memory_on_stream`, no host block, replacing the per-column
/// `cuda_event_synchronize` of the one-behind B2 path), and there is NO per-column
/// `cudaStreamSynchronize(0)` (the B2/Part-A `reclaim_after_free` that foreclosed all overlap).
/// Device residency is instead bounded DEVICE-SIDE by a small event ring (`FREE_EVENT_RING`): before
/// stream 0 produces the next eval buffer, it waits on the free event of the column `RING` steps back,
/// so at most ~`RING` eval buffers are ever live on the device (see the ring below). BATCHED IMPLIES
/// the B1/B2 pinned async substrate (pinning + copy stream + `async_stash_enabled`), and composes
/// with `GATE_AIR_STREAM_MAIN_LOWMEM` / `GATE_AIR_BOUNDARY_TRIM` (it only changes the copy/free
/// SCHEDULE inside `dehydrate_column`, not the LOWMEM/free-after-k4 or boundary-trim work). Default
/// OFF for independent A/B vs. the one-behind B2 path (`GATE_AIR_ASYNC_STASH` alone). Byte-identical:
/// the reader-head fence `sync_stash_copies_if_pending` still drains the copy stream (and the ring)
/// before any consumer reads.
fn async_stash_batched_enabled() -> bool {
    std::env::var("GATE_AIR_ASYNC_STASH_BATCHED").is_ok()
}

/// Reads `GATE_AIR_FREE_ON_STREAM0` (refinement of PART B3, opt-in, DEFAULT OFF). Only meaningful on
/// the batched path (`GATE_AIR_ASYNC_STASH_BATCHED`). When set, each eval buffer's `cudaFreeAsync` is
/// enqueued on STREAM 0 (where the buffer was allocated by `cudaMallocFromPoolAsync(..., 0)`),
/// stream-ordered AFTER its D2H via `cudaStreamWaitEvent(stream0, e_d2h)`, INSTEAD of freeing on the
/// copy stream. Freeing on the copy stream interleaves 188 `cudaFreeAsync`s between the D2Hs and, being
/// a cross-stream free against the stream-0-owned pool, forces cross-stream pool ordering that
/// fragments the copy stream into non-contiguous DMA bursts (the ~2 GB/s dehydrate defect). Freeing on
/// stream 0 leaves the copy stream carrying ONLY the 188 back-to-back D2Hs (pinned DMA bandwidth),
/// while the device-residency ring bound is preserved by recording the ring event on stream 0 after
/// the free (so it still marks "D2H done AND free enqueued", and stream 0 waits the ring-old event
/// before the next eval alloc). Byte-identical: only WHICH stream frees and WHEN reuse happens changes,
/// never the committed bytes. Default OFF for A/B vs. the copy-stream-free batched path.
fn free_on_stream0_enabled() -> bool {
    std::env::var("GATE_AIR_FREE_ON_STREAM0").is_ok()
}

// ============================================================================
// PART B1: persistent, recycled PINNED host-buffer pool.
//
// B0 called `cudaHostAlloc` PER COLUMN and `cudaFreeHost` on drop, re-page-locking ~12 GB every run
// (the `dehydrate_reclaim` 1.9 s tax and the reason the D2H only hit ~40% of a pinned DMA's
// ceiling). B1 instead draws every stash buffer from this pool: `take(cap)` reuses a free-listed
// page-locked allocation of exactly `cap` u32s when one exists, else allocates ONE with
// `cudaHostAlloc`; `give` returns it to the free-list (NOT `cudaFreeHost`). `clear_stash` recycles
// a proof's buffers back to the free-list so the NEXT run/shard reuses the same page-locked pages —
// zero steady-state page-lock.
//
// SOUNDNESS: a pooled buffer holds one column's exact committed bytes for that column's lifetime;
// it is only recycled at `clear_stash` (the next commit boundary), after all readers are done,
// exactly like the per-entry lifetime B0 had. The bytes a reader sees are identical; only the
// allocation source (recycled vs. fresh `cudaHostAlloc`) changes.
struct PinnedPool {
    /// Free (recycled) page-locked buffers keyed by capacity in u32s -> list of (ptr, len).
    free: HashMap<usize, Vec<*mut u32>>,
    /// Dedicated non-blocking copy stream for async D2H/H2D (created lazily). Opaque
    /// `cudaStream_t`.
    copy_stream: *mut std::ffi::c_void,
}

// SAFETY: the raw pointers are process-global page-locked allocations / an opaque CUDA stream
// handle, guarded by the `Mutex` around `PINNED_POOL`. They are only ever used through the CUDA
// runtime.
unsafe impl Send for PinnedPool {}

impl PinnedPool {
    /// Take a page-locked buffer of exactly `cap` u32s: reuse a free-listed one if present, else
    /// `cudaHostAlloc` a new one. Never memset — the caller fully overwrites it via D2H.
    fn take(&mut self, cap: usize) -> *mut u32 {
        if let Some(list) = self.free.get_mut(&cap) {
            if let Some(ptr) = list.pop() {
                return ptr;
            }
        }
        let ptr = unsafe { bindings::cuda_alloc_pinned_host_uint32_t(cap as u32) };
        assert!(!ptr.is_null(), "GATE_AIR_ASYNC_STASH: cudaHostAlloc failed");
        ptr
    }

    /// Return a buffer to the free-list keyed by its capacity (does NOT `cudaFreeHost`).
    fn give(&mut self, ptr: *mut u32, cap: usize) {
        self.free.entry(cap).or_default().push(ptr);
    }

    /// The lazily-created dedicated copy stream (a non-blocking stream, so copies on it overlap
    /// default-stream compute). Created once and kept for the process lifetime.
    fn stream(&mut self) -> *mut std::ffi::c_void {
        if self.copy_stream.is_null() {
            self.copy_stream = unsafe { bindings::cuda_create_copy_stream() };
            assert!(
                !self.copy_stream.is_null(),
                "GATE_AIR_ASYNC_STASH: copy stream create failed"
            );
        }
        self.copy_stream
    }
}

/// PER-DEVICE pinned-pool table (multi-GPU "option A"). Each slot is an independently mutex-guarded,
/// lazily-created `PinnedPool` for one device, keyed by `cur_device()` — mirroring the C-side
/// `g_mem_pool_table[cudaGetDevice()]`. A slot's `copy_stream` is created (in `stream()`) while the
/// caller is bound to device N, so the stream lands on device N; that slot's recycled pinned host
/// buffers likewise serve only device-N D2H/H2D. Process-lifetime, exactly as the single global was.
///
/// Table type (mechanical choice): a fixed-size `[Mutex<Option<PinnedPool>>; MAX_CUDA_DEVICES]`
/// lazily filling each slot on first use — the direct analogue of the mem pool's fixed-size
/// `g_mem_pool_table` + `g_mem_pool_initialized_table`, and cheaper than a `Mutex<HashMap>` on the
/// hot per-column path (one array index + one mutex lock, no hashing). Const-initialized so it needs
/// no `LazyLock`.
///
/// BYTE-IDENTITY (N=1): device 0 => slot [0], created on first use exactly like the old single
/// `PINNED_POOL` (same `PinnedPool { free: empty, copy_stream: null }`), so `pool(0)` behaves
/// identically to the old global.
static PINNED_POOL_TABLE: [Mutex<Option<PinnedPool>>; MAX_CUDA_DEVICES] =
    [const { Mutex::new(None) }; MAX_CUDA_DEVICES];

/// Run `f` with a mutable borrow of the CURRENT DEVICE's `PinnedPool`, lazily creating it on first
/// use (empty free-list + null copy stream, identical to the old global's initial state). One array
/// index + one mutex lock, matching the mem pool's per-device lazy create.
fn with_pool<R>(f: impl FnOnce(&mut PinnedPool) -> R) -> R {
    let mut guard = PINNED_POOL_TABLE[cur_device()].lock().unwrap();
    let pool = guard.get_or_insert_with(|| PinnedPool {
        free: HashMap::new(),
        copy_stream: std::ptr::null_mut(),
    });
    f(pool)
}

/// The current device's dedicated copy stream handle (create-on-first-use, on THIS device because
/// the caller is bound to it). Cheap: one array index + one mutex lock + a null check.
fn copy_stream() -> *mut std::ffi::c_void {
    with_pool(|p| p.stream())
}

// ============================================================================
// STEP 2 host-staging stash — lives ENTIRELY in this streaming-commit layer.
//
// `BaseFieldVec` is a pure device type again (no `host_stage` field). When the streamed commit
// dehydrates an eval column, its exact bytes are D2H-copied here and the device buffer is freed;
// the (now-freed) device pointer VALUE is kept on the column as the stash KEY (`owns_memory` set
// false so Drop won't double-free). The ~3 post-commit reader sites (OODS barycentric, quotient,
// decommit) call the explicit rehydrate/host-recover helpers below — nothing is transparent.
//
// SOUNDNESS: the staged bytes are the exact committed bytes (same `to_vec()` D2H used everywhere),
// so any reader that rehydrates observes the identical committed values. The pointer key is stable
// for the lifetime of the commit: the entry pins the identity and the freed slot is not handed back
// to a live-keyed column during this window. `is_staged`/`stash_get_owned` FAIL LOUD if a key is
// missing, never silently returning wrong data.
// ============================================================================

// DEFERRED ASYNC (change 3): the stash holds pageable `Vec<u32>`. The pinned-host + async-copy
// primitives are landed (`cuda_alloc_pinned_host_uint32_t` / `cuda_free_pinned_host` /
// `copy_uint32_t_vec_*_async` in bindings.rs + utils.cu), but the stash is NOT yet wired onto them:
// to OVERLAP the D2H (at commit) / H2D (per reader) with GPU compute, `dehydrate_column` should D2H
// into a pinned buffer on a dedicated copy stream (not synchronize), and `rehydrate_owned` should
// H2D from that pinned buffer on the same stream, synchronizing only at the reader. Today's copies
// go through the pageable `to_vec()` / `copy_uint32_t_vec_from_host_to_device` path (correct, but
// serialized against compute on the default stream). This is the remaining async work; compute
// stays on GPU regardless (changes 1+2), so correctness and the residency win hold without it.
// PROCESS-GLOBAL (NOT thread-local) stash. The stash is populated by `dehydrate_column` on the
// commit thread, but the post-commit readers run on OTHER threads: with the `parallel` feature (the
// `cuda` feature enables it), OODS `eval_at_points` and the quotient accumulation run via rayon
// `par_map_cols` on WORKER threads. A thread_local stash is empty on those workers, so `is_staged`
// would return false for a genuinely-staged column and hand its (sentinel, unmapped) `device_ptr`
// straight to the barycentric/quotient kernel => illegal address. A global mutex-guarded map makes
// the staged bytes visible from every thread. Contention is negligible: reads are one HashMap
// lookup + a clone-free `rehydrate_*` H2D; the columns are large and GPU-bound.
static HOST_STASH: LazyLock<Mutex<HashMap<usize, StashEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Monotonic sentinel id source for stash keys. Each dehydrated column is keyed by a UNIQUE
/// sentinel derived from this counter (NOT its freed device address), so the mem-pool recycling a
/// freed address for a later column can never collide with an existing stash key. Starts at 1 (0
/// reserved) and only ever increments; the tagged sentinel (`SENTINEL_TAG | id`) is disjoint from
/// every real GPU device address (48-bit VA << 2^63). Global so ids are unique across threads too.
static STASH_NEXT_ID: AtomicUsize = AtomicUsize::new(1);
/// DIAGNOSTIC counters (printed via `diag_report_dehydrate`): total `dehydrate_column` calls, how
/// many EARLY-RETURNED on a colliding/already-present key (always 0 with the unique-sentinel key),
/// and how many actually stashed. Global atomics so a parallel run still aggregates correctly.
static DEHYDRATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static DEHYDRATE_EARLY_RETURNS: AtomicUsize = AtomicUsize::new(0);
static DEHYDRATE_STASHED: AtomicUsize = AtomicUsize::new(0);

/// PART B2: `true` whenever an ASYNC dehydrate D2H has been issued on the copy stream but not yet
/// awaited. The producer does NOT host-block per column (that would kill the overlap). Instead the
/// FIRST reader (`sync_stash_copies_if_pending`, called at the head of every rehydrate/host-recover
/// helper) host-blocks ONCE on the copy stream so every staged pinned buffer has fully arrived
/// before any consumer reads it — the async-ordering byte-identity guard. Reset in `clear_stash`.
///
/// A `Mutex<bool>` (not a bare atomic): the sync + flag-clear happen while holding the lock, and
/// every reader acquires the lock, so a reader arriving WHILE the sync is in flight BLOCKS on the
/// mutex until the sync completes, then observes `false` and proceeds — it can never read a
/// not-yet-arrived buffer. Uncontended fast path after the first sync is one lock + a bool check.
///
/// PER-DEVICE (multi-GPU): one flag per device, keyed by `cur_device()`. A dehydrate D2H on device N
/// runs on device N's copy stream; its pending flag and the reader-head sync that drains device N's
/// copy stream must be device N's, so a device-M reader never spuriously drains (or skips draining)
/// device N. Byte-identity (N=1): device 0 => slot [0], the exact old single flag.
static STASH_COPIES_PENDING: [Mutex<bool>; MAX_CUDA_DEVICES] =
    [const { Mutex::new(false) }; MAX_CUDA_DEVICES];

/// PART B2 deferred-free ring (single slot): the async producer defers each staged column's
/// device-buffer free behind its D2H event so the D2H overlaps the NEXT column's NTT. Holds
/// `(device_ptr, e_d2h)` for the ONE column whose free is still pending. `defer_free_after_event`
/// flushes the previous slot (host-wait its event, free, reclaim) before installing the new one, so
/// at most 2 eval buffers are live (this + next) — memory stays bounded exactly like the blocking
/// path. The trailing slot is flushed by `flush_deferred_free` (first reader / commit boundary).
// Device pointer and CUDA event handle both stored as `usize` so the static `Mutex` is `Sync`
// (raw pointers aren't `Send`). Both are only ever used through the CUDA runtime.
//
// PER-DEVICE (multi-GPU): one deferred-free slot per device, keyed by `cur_device()`. A column's
// device buffer + its D2H event belong to the device it was produced on; freeing it must happen
// while bound to that device. Byte-identity (N=1): device 0 => slot [0], the old single slot.
static DEFERRED_FREE: [Mutex<Option<(usize, usize)>>; MAX_CUDA_DEVICES] =
    [const { Mutex::new(None) }; MAX_CUDA_DEVICES];

/// Flush any pending deferred free: host-wait its D2H event, free the device buffer, reclaim,
/// destroy the event. No-op if nothing pending. Called before installing the next deferred slot,
/// and by the first reader / `clear_stash`.
fn flush_deferred_free() {
    let taken = DEFERRED_FREE[cur_device()].lock().unwrap().take();
    if let Some((ptr, event)) = taken {
        // SAFETY: FFI. Host-wait the D2H completes (so the bytes are captured) before freeing the
        // device buffer, then reclaim (bounds memory) and destroy the event.
        unsafe {
            bindings::cuda_event_synchronize(event as *mut std::ffi::c_void);
            bindings::cuda_destroy_event(event as *mut std::ffi::c_void);
            bindings::cuda_free_memory(ptr as *const std::ffi::c_void);
        }
        reclaim_after_free();
    }
}

/// Install `col`'s device-buffer free as the new deferred slot (behind `e_d2h`), flushing the
/// previous slot first. This one-behind pipeline hides the free's D2H-completion wait under the
/// next column's NTT compute — the B2 double-buffer.
fn defer_free_after_event(device_ptr: *const std::ffi::c_void, e_d2h: *mut std::ffi::c_void) {
    flush_deferred_free();
    *DEFERRED_FREE[cur_device()].lock().unwrap() = Some((device_ptr as usize, e_d2h as usize));
}

/// PART B3 (GATE_AIR_ASYNC_STASH_BATCHED) device-side residency bound. The bug in the B2 one-behind
/// path is that `flush_deferred_free` HOST-blocks on the previous column's D2H event and
/// `reclaim_after_free` does a `cudaStreamSynchronize(0)` per column, so with no compute queued the
/// 188 D2Hs serialize on the host at ~2 GB/s. B3 removes BOTH host stalls: each column's device free
/// is enqueued stream-ordered on the copy stream (after its D2H), and a device-side event records
/// "D2H done AND free enqueued" on the copy stream. This ring holds the last `FREE_RING_DEPTH` such
/// events. Before column N's D2H is issued, stream 0 (which allocates + NTTs the eval buffers) is made
/// to `cudaStreamWaitEvent` the ring-old (N-`FREE_RING_DEPTH`) event, so stream 0 cannot run more than
/// `FREE_RING_DEPTH` eval buffers ahead of a completed free — bounding LIVE device eval buffers to
/// ~`FREE_RING_DEPTH` WITHOUT any host block. Getting this wrong is the make-or-break: too large / no
/// ring OOMs the 40 GB card (188 x 256 MiB = 48 GiB at 2^25), a host-enforced ring kills the overlap.
///
/// Events stored as `usize` so the static `Mutex` is `Sync` (raw pointers aren't `Send`); only ever
/// used through the CUDA runtime, on the single producer thread that runs the dehydrate loop.
///
/// PER-DEVICE (multi-GPU): one ring per device, keyed by `cur_device()`. The ring's events are
/// recorded on device N's copy stream / stream 0 and bound device N's stream-0 run-ahead; a
/// device-M producer must not see device N's events (cross-device event waits are illegal). Each
/// producer thread runs its own device's dehydrate loop, so it only ever touches its own ring.
/// Byte-identity (N=1): device 0 => slot [0], the old single ring.
static FREE_EVENT_RING: [Mutex<Vec<usize>>; MAX_CUDA_DEVICES] =
    [const { Mutex::new(Vec::new()) }; MAX_CUDA_DEVICES];

/// Depth of the device-side free-event ring (max eval buffers allowed live on the device at once).
/// 3 keeps a healthy copy/compute pipeline (copy N overlaps NTT N+1 while N+2 is queued) while
/// capping residency at 3 x 256 MiB = 768 MiB at 2^25 — well within the 40 GB card and matching the
/// current "at most ~2 eval buffers live" invariant with one slot of slack for pipelining.
const FREE_RING_DEPTH: usize = 3;

/// Push a fresh "D2H-done + free-enqueued" copy-stream event into the ring. If the ring is already at
/// `FREE_RING_DEPTH`, pop the oldest event, make STREAM 0 wait on it (device-side — bounds the
/// producer's run-ahead so at most `FREE_RING_DEPTH` eval buffers are live), then destroy it. No host
/// block. `event` is a `cudaEvent_t` recorded on the copy stream (copy-stream-free path) or on stream
/// 0 (`GATE_AIR_FREE_ON_STREAM0` path — where the wait below is a same-stream, trivially-satisfied
/// order; residency is then bounded even tighter because alloc/free are serialized on stream 0).
fn ring_push_and_bound(event: *mut std::ffi::c_void) {
    let mut ring = FREE_EVENT_RING[cur_device()].lock().unwrap();
    if ring.len() >= FREE_RING_DEPTH {
        let old = ring.remove(0);
        // SAFETY: FFI. Order STREAM 0 (the alloc/NTT stream) after the ring-old column's
        // free-event so the producer cannot allocate the next eval buffer until that column's D2H
        // has completed and its device buffer has been freed back to the pool. Device-side wait —
        // the host is NOT blocked. Then destroy the consumed event.
        unsafe {
            bindings::cuda_stream_wait_event(std::ptr::null_mut(), old as *mut std::ffi::c_void);
            bindings::cuda_destroy_event(old as *mut std::ffi::c_void);
        }
    }
    ring.push(event as usize);
}

/// Drain the batched free-event ring: destroy every retained ring event. Called by
/// `sync_stash_copies_if_pending` / `clear_stash` AFTER the copy stream has been host-synced (so every
/// D2H — and, on the copy-stream-free path, every stream-ordered free — has completed). On the
/// `GATE_AIR_FREE_ON_STREAM0` refinement the frees are enqueued on STREAM 0 (ordered after the D2H
/// via `cudaStreamWaitEvent`), so a ring event recorded on stream 0 may still be pending here; that is
/// safe: `cudaEventDestroy` on an un-completed event returns immediately and releases the event's
/// resources automatically once the device reaches it (no corruption, no host block). The freed device
/// buffers return to the pool stream-ordered on stream 0, and the NEXT stream-0 allocation is naturally
/// ordered after them, so no reuse hazard. No host wait here. No-op when the ring is empty (B2 / off).
fn drain_free_event_ring() {
    let mut ring = FREE_EVENT_RING[cur_device()].lock().unwrap();
    for event in ring.drain(..) {
        // SAFETY: FFI. The copy stream was host-synced by the caller (all D2Hs done). Copy-stream-free
        // path: the event completed too. Stream-0-free path: destroying a still-pending event is safe
        // (CUDA defers release until the event completes).
        unsafe {
            bindings::cuda_destroy_event(event as *mut std::ffi::c_void);
        }
    }
}

/// PART B2 byte-identity guard: if any async dehydrate D2H is still in flight on the copy stream,
/// host-block ONCE on the copy stream (under the pending lock) so every staged pinned buffer has
/// fully arrived, then clear the flag. Every reader helper calls this BEFORE touching stash bytes,
/// so no consumer ever observes partial/stale data. Also flushes the trailing deferred free (its
/// D2H is part of the same copy stream we just drained). Cheap after the first call (lock + `false`
/// check).
fn sync_stash_copies_if_pending() {
    // Per-device: the caller's device flag + its own copy stream / deferred-free slot / ring. A
    // reader on device N drains device N's copy stream; a device-M dehydrate's pending state is
    // independent (its own slot), so this never over- or under-syncs across devices.
    let mut pending = STASH_COPIES_PENDING[cur_device()].lock().unwrap();
    if *pending {
        let stream = copy_stream();
        // Attribute the batched pipeline's FINAL host wait to the [T1] dehydrate_d2h accumulator so
        // it stays HONEST: the batched path issues D2Hs without per-column host blocks (the
        // per-column `t1_add(1, ..)` in `dehydrate_column` is then near-zero issue time), so the
        // real pinned-bandwidth wall is spent HERE draining the copy stream once. Adding it makes
        // [T1] dehydrate_d2h report the true reduced pipeline time (~1.6-2.0s @2^24) rather than
        // ~0. On the B2 / off path this sync is normally already-complete (readers ran later), so
        // the added time is negligible and dehydrate_d2h keeps its existing meaning.
        let timed = t1_timers_on();
        let sync_start = timed.then(std::time::Instant::now);
        // SAFETY: FFI host-barrier on the copy stream; after it returns every async D2H issued on
        // that stream has completed. Any other reader is blocked on this same lock meanwhile.
        unsafe {
            bindings::cuda_stream_synchronize(stream);
        }
        if let Some(s) = sync_start {
            t1_add(1, s.elapsed().as_nanos());
        }
        *pending = false;
        drop(pending);
        // The last column's device buffer is still in the deferred slot; free it now (its D2H just
        // completed with the stream sync above). (B2 one-behind path; empty on the B3 batched path,
        // which streams every free on the copy stream.)
        flush_deferred_free();
        // B3 batched path: the copy stream (which we just host-synced) carried every D2H (and, on the
        // copy-stream-free path, every stream-ordered free), so those ring events have completed and
        // their device buffers are freed. On the GATE_AIR_FREE_ON_STREAM0 refinement the frees run on
        // stream 0 instead; their ring events may still be pending, which drain_free_event_ring handles
        // safely (deferred event destroy). Either way, destroy the retained events. No-op on B2 / off.
        drain_free_event_ring();
    }
}

/// High-bit tag OR'd into the monotonic id to form a stash-key sentinel that can never equal a real
/// GPU device virtual address (GPU VAs are 48-bit, far below 2^63). A staged column's `device_ptr`
/// is OVERWRITTEN with this sentinel; it is NEVER dereferenced (the `owns_memory == false` guard +
/// every consumer routing staged columns through the host stash guarantee this — see the audit in
/// `dehydrate_column`).
const SENTINEL_TAG: usize = 1usize << 63;

/// Allocate the next unique stash-key sentinel (globally unique across threads).
fn next_sentinel() -> usize {
    let id = STASH_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    SENTINEL_TAG | id
}

/// DIAGNOSTIC: print + reset the `dehydrate_column` counters. Prints unconditionally so a single
/// both-flags run shows the split. Neutral: reading/resetting counters does not touch the stash or
/// any committed data.
pub fn diag_report_dehydrate(label: &str) {
    let calls = DEHYDRATE_CALLS.swap(0, Ordering::Relaxed);
    let early = DEHYDRATE_EARLY_RETURNS.swap(0, Ordering::Relaxed);
    let stashed = DEHYDRATE_STASHED.swap(0, Ordering::Relaxed);
    eprintln!(
        "[GATE_AIR_DIAG] dehydrate_column [{label}]: {calls} calls, {early} EARLY-RETURNED \
         (colliding/reused key), {stashed} stashed"
    );
}

/// Clear ALL stash entries. MUST be called at the start of each streamed commit (before staging the
/// new tree's columns) so a freed device-pointer key from a PRIOR proof/shard can never alias a
/// fresh live column's pointer and make `is_staged` return a false positive (which would serve
/// stale bytes — a soundness bug). Within a single proof the staged main-tree columns must survive
/// from commit through decommit, so this is called only at the NEXT streamed commit boundary, not
/// between a proof's readers.
pub(crate) fn clear_stash() {
    // Drain any still-in-flight async D2H before dropping the entries: an entry's Drop recycles its
    // pooled pinned buffer back to the free-list, and reusing that buffer for the next run's D2H
    // before the previous run's copy finished would corrupt it. This host-blocks only if a copy is
    // genuinely outstanding (normally already synced by the readers).
    //
    // MULTI-GPU: `HOST_STASH` is process-global (one shared map, unique-sentinel keys), but each
    // staged column's async D2H, deferred free, and free-event ring live on the DEVICE it was
    // produced on — in that device's per-device slot. A single caller-device drain would leave
    // OTHER devices' copies in flight and their trailing frees/rings unreclaimed; then dropping the
    // shared entries could recycle a pinned buffer whose device-N D2H hasn't landed. So bind each
    // device in turn and drain ITS copy stream + flush ITS deferred free + reset ITS pending flag,
    // THEN clear the shared host map. `sync_stash_copies_if_pending` is a cheap no-op on a device
    // whose flag is false (every device untouched this proof), so N=1 pays only the device-0 pass
    // (identical work to the old single drain) plus a per-device flag check.
    //
    // A device needs draining iff its pending flag is set: the flag is set whenever an async D2H is
    // issued on that device and cleared ONLY by a drain (`sync_stash_copies_if_pending`) or here, and
    // that drain also flushes the deferred-free slot and the ring. So a false flag means that
    // device's ring/deferred slot are already empty — nothing to do, and (crucially) NO
    // `cuda_set_device` on an absent/other device (which would print an error on a single-GPU box).
    //
    // BYTE-IDENTITY (N=1): only device 0 was ever touched, so only slot [0] has a set flag. We bind
    // device 0 (a no-op — it is already current) and run the exact old single drain; every other
    // slot's flag is false and is skipped. On the batched/async path where the readers already
    // synced, even slot [0]'s flag is false and the whole loop is a no-op, identical to before.
    //
    // The current device is saved and restored so `clear_stash` is transparent to the caller's
    // device binding (it is called at a commit boundary on the producer thread).
    let saved = cur_device();
    let mut rebound = false;
    for dev in 0..MAX_CUDA_DEVICES {
        if !*STASH_COPIES_PENDING[dev].lock().unwrap() {
            continue; // untouched (or already-synced) device — no in-flight copy, ring, or deferred
        }
        // SAFETY: FFI; bind this device so `sync_stash_copies_if_pending`/`flush_deferred_free`
        // (which index `cur_device()`) act on device `dev`'s slot and its stream sync targets
        // device `dev`. This device is known-touched, so the ordinal is valid.
        unsafe {
            bindings::cuda_set_device(dev as i32);
        }
        rebound = rebound || dev != saved;
        sync_stash_copies_if_pending(); // drains dev's copy stream + deferred + ring, clears flag
    }
    // SAFETY: FFI; restore the caller's original device (only if we actually rebound elsewhere).
    if rebound {
        unsafe {
            bindings::cuda_set_device(saved as i32);
        }
    }
    // All touched devices' copies have landed and their buffers freed; now drop the shared entries
    // (each entry's Drop recycles its pinned buffer to whatever device is current — safe: pinned host
    // memory is process-wide, not device-affine, and the D2H that filled it has completed).
    HOST_STASH.lock().unwrap().clear();
}

/// D2H-copy `col`'s bytes into the stash keyed by a UNIQUE monotonic sentinel, free the device
/// buffer, and OVERWRITE `col.device_ptr` with that same sentinel (leaving `owns_memory = false`).
///
/// STASH-ID KEY (fixes the 94/188 corruption): the stash was previously keyed by the FREED device
/// pointer. The CUDA mem-pool recycles freed addresses, so the next same-size allocation reused a
/// just-freed key; `contains_key` matched and the call early-returned, leaving that column NOT
/// stashed and TWO committed columns sharing one device_ptr — half the large tree1 columns
/// corrupted. Keying by a monotonic sentinel disjoint from any real device address (`SENTINEL_TAG |
/// id`, GPU VAs are 48-bit) makes every key unique: no pool reuse can ever collide, so all 188
/// large columns stash 1:1 (zero early-returns).
///
/// SAFETY — the sentinel `device_ptr` is NEVER dereferenced as a real address. A staged column has
/// `owns_memory = false`; `is_staged`/`staged_host_ptr` gate on `!owns_memory` and every consumer
/// routes a staged column through the HOST STASH: `build_leaves` rehydrates (blake2s.rs), OODS
/// `barycentric_eval_at_point` rehydrates (poly.rs), quotient `rehydrate_block` (quotient.rs),
/// decommit `host_batch_get` (column.rs), and the composition kernel supplies staged columns via
/// the per-column host table with `tiled_input` true (evaluate_gate_air.cu: staged entries use the
/// tile buffer, never `trace1_evaluations[c]`). Only RESIDENT columns (`is_staged == false`) ever
/// dereference `device_ptr` or do `device_ptr.add(off)`, and a resident column keeps its real live
/// address (it is never sentinel-tagged). Audited: no consumer dereferences a staged device_ptr.
pub(crate) fn dehydrate_column(col: &mut BaseFieldVec) {
    DEHYDRATE_CALLS.fetch_add(1, Ordering::Relaxed);
    DEHYDRATE_STASHED.fetch_add(1, Ordering::Relaxed);
    let timed = t1_timers_on();
    let d2h_start = timed.then(std::time::Instant::now);
    // PART B2 (GATE_AIR_ASYNC_STASH): async, double-buffered producer. Take a POOLED pinned buffer
    // (recycled — no per-column cudaHostAlloc), issue the D2H on the dedicated copy stream, and
    // order the device free AFTER that D2H device-side (via events) — NO host block, so this
    // column's D2H overlaps the NEXT column's NTT on the default stream. Falls back to the B0
    // blocking pinned path (per-column cudaHostAlloc) under `GATE_AIR_PIN_STASH` alone, and to
    // the pageable path otherwise.
    let host: StashEntry = if async_stash_enabled() {
        let len = col.size;
        let ptr = with_pool(|p| p.take(len));
        let stream = copy_stream();
        // SAFETY: FFI. `col.device_ptr` is the live just-produced eval column; `ptr` is a `len`-u32
        // pinned buffer. Ordering: the copy stream waits the default stream's current work (the NTT
        // that produced `col`) via `e_prod`, then D2Hs into `ptr`, then records `e_d2h`. The
        // default stream is NOT stalled on the D2H — it runs the next column's NTT
        // immediately (the overlap). The device buffer's free is DEFERRED (with `e_d2h`)
        // and flushed one column later, by which point the D2H has completed, so the free
        // never races the copy. Bytes captured before reuse.
        let e_d2h = unsafe {
            let e_prod = bindings::cuda_create_event();
            bindings::cuda_event_record(e_prod, std::ptr::null_mut()); // default stream = null
            bindings::cuda_stream_wait_event(stream, e_prod);
            bindings::copy_uint32_t_d2h_pinned_async(col.device_ptr, ptr, len as u32, stream);
            let e_d2h = bindings::cuda_create_event();
            bindings::cuda_event_record(e_d2h, stream);
            bindings::cuda_destroy_event(e_prod);
            e_d2h
        };
        *STASH_COPIES_PENDING[cur_device()].lock().unwrap() = true;
        if async_stash_batched_enabled() {
            // PART B3 (batched): NO host block. Free `col`'s device buffer STREAM-ORDERED on the
            // copy stream (after its D2H), re-record `e_d2h` so it now marks "D2H done AND free
            // enqueued", and push it into the device-side ring — which makes stream 0 wait on the
            // ring-old (N-FREE_RING_DEPTH) free event before producing the next eval buffer,
            // bounding live device eval buffers to ~FREE_RING_DEPTH without any host sync. The 188
            // D2Hs thus run back-to-back at pinned bandwidth (no per-column cuda_event_synchronize,
            // no cudaStreamSynchronize(0)). The trailing events are destroyed by the reader-head
            // fence after it host-syncs the copy stream.
            if col.owns_memory {
                if free_on_stream0_enabled() {
                    // REFINEMENT (GATE_AIR_FREE_ON_STREAM0): free on STREAM 0 (where the buffer was
                    // allocated), stream-ordered AFTER its D2H, so the copy stream carries ONLY the
                    // 188 back-to-back D2Hs (no interleaved cross-stream free fragmenting the DMA).
                    // Stream 0 waits `e_d2h` (recorded on the copy stream above) so the free never
                    // races the copy; then the free is enqueued on stream 0 and `e_d2h` is
                    // re-recorded on stream 0 so the ring event still marks "D2H done AND free
                    // enqueued" (stream 0 waits the ring-old event before the next eval alloc —
                    // residency bound intact). NO host block, NO cudaStreamSynchronize(0).
                    unsafe {
                        // stream 0 == null: order stream 0 after the copy-stream D2H, free on
                        // stream 0, then re-record the ring event on stream 0.
                        bindings::cuda_stream_wait_event(std::ptr::null_mut(), e_d2h);
                        bindings::cuda_free_memory_on_stream(
                            col.device_ptr as *const std::ffi::c_void,
                            std::ptr::null_mut(),
                        );
                        bindings::cuda_event_record(e_d2h, std::ptr::null_mut());
                    }
                    ring_push_and_bound(e_d2h);
                } else {
                    let stream = copy_stream();
                    unsafe {
                        bindings::cuda_free_memory_on_stream(
                            col.device_ptr as *const std::ffi::c_void,
                            stream,
                        );
                        // Re-record on the copy stream AFTER the free is enqueued, so the ring event
                        // marks both D2H completion and free execution.
                        bindings::cuda_event_record(e_d2h, stream);
                    }
                    ring_push_and_bound(e_d2h);
                }
            } else {
                unsafe { bindings::cuda_destroy_event(e_d2h) };
            }
        } else if col.owns_memory {
            // PART B2 (one-behind): defer this column's device-buffer free behind `e_d2h`; this also
            // FLUSHES the previous column's deferred free (host-waits its now-complete `e_d2h`,
            // frees, reclaims). This host stall per column is exactly what B3 removes.
            defer_free_after_event(col.device_ptr as *const std::ffi::c_void, e_d2h);
        } else {
            unsafe { bindings::cuda_destroy_event(e_d2h) };
        }
        StashEntry::Pooled { ptr, len, cap: len }
    } else if pin_stash_enabled() {
        // PART B0: stage into a per-column PINNED host buffer (DMA-speed D2H/H2D). Blocking
        // default-stream copy; re-page-locks every run (superseded by the async pool above).
        let len = col.size;
        let ptr = unsafe { bindings::cuda_alloc_pinned_host_uint32_t(len as u32) };
        assert!(!ptr.is_null(), "GATE_AIR_PIN_STASH: cudaHostAlloc failed");
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_host(col.device_ptr, ptr, len as u32);
        }
        StashEntry::Pinned { ptr, len }
    } else {
        StashEntry::Pageable(col.to_vec().into_iter().map(|v| v.0).collect())
    };
    if let Some(s) = d2h_start {
        t1_add(1, s.elapsed().as_nanos()); // dehydrate D2H (async path: issue time only,
                                           // overlapped)
    }
    // The async path already handed `col`'s device buffer to the deferred-free ring (freed one
    // column later, behind its D2H event); do NOT free here. The blocking paths (B0 / pageable)
    // free now.
    if col.owns_memory && !async_stash_enabled() {
        unsafe {
            bindings::cuda_free_memory(col.device_ptr as *const std::ffi::c_void);
        }
        // PART A reclaim: drain the stream so the freed block is reusable before the next alloc
        // (bounds memory — no OOM regression), but DON'T TrimTo/release-to-OS every column (the
        // ~22 s churn). `GATE_AIR_RECLAIM_TRIM` restores the legacy sync+trim for A/B measurement.
        reclaim_after_free();
    }
    // Re-key by a unique sentinel and overwrite the column's device_ptr so all staged consumers
    // (which look up by device_ptr) resolve THIS column's own entry. The freed real address is
    // never used as a key again, so mem-pool reuse cannot alias it.
    let sentinel = next_sentinel();
    col.owns_memory = false;
    col.device_ptr = sentinel as *const u32;
    HOST_STASH.lock().unwrap().insert(sentinel, host);
}

/// True iff `col` was host-staged by the streamed commit: it is a NON-OWNING column whose
/// (sentinel) `device_ptr` keys a stash entry AND the stashed length equals `col.size`.
///
/// Staged columns are keyed by a UNIQUE monotonic sentinel that `dehydrate_column` also writes into
/// `col.device_ptr` (never a real/freed device address), so there is no pointer-aliasing surface:
/// distinct columns always have distinct keys. The `!owns_memory` predicate is retained as
/// defense-in-depth (a staged column is always non-owning; a resident column keeps its real live
/// address and must take the resident path) and to short-circuit resident columns cheaply.
pub fn is_staged(col: &BaseFieldVec) -> bool {
    if col.owns_memory {
        return false;
    }
    let key = col.device_ptr as usize;
    HOST_STASH
        .lock()
        .unwrap()
        .get(&key)
        .is_some_and(|h| h.len() == col.size)
}

/// H2D `len` u32s from a (pinned) stash host pointer into a fresh device buffer, returning the
/// device pointer. Under `GATE_AIR_ASYNC_STASH` this uses the NO-REDUNDANT-MEMSET async copy on the
/// dedicated copy stream and SYNCS that stream before returning, so the buffer is fully populated
/// for the kernel (byte-identical) while skipping the two dead full-column device memsets the
/// default path issues (`cuda_malloc_uint32_t`'s zero + `copy_..._host_to_device`'s zero) — the
/// OODS 10.7 s root cause, which also serialized every parallel reader on stream 0. Off the flag,
/// the exact previous path.
fn rehydrate_h2d(host_ptr: *const u32, len: usize) -> *const u32 {
    if async_stash_enabled() {
        let stream = copy_stream();
        // SAFETY: FFI. `host_ptr` is a `len`-u32 pinned stash buffer whose D2H already completed
        // (`sync_stash_copies_if_pending` ran at the reader head). The async H2D fully overwrites
        // the fresh device buffer; syncing the copy stream before returning guarantees
        // arrival before any kernel reads it. Per-reader copy stream use (not stream 0)
        // removes the parallel-reader jam.
        unsafe {
            let device_ptr =
                bindings::copy_uint32_t_h2d_nomemset_async(host_ptr, len as u32, stream);
            bindings::cuda_stream_synchronize(stream);
            device_ptr
        }
    } else {
        // SAFETY: FFI; the existing blocking H2D (allocates + double-memsets + copies on stream 0).
        unsafe { bindings::copy_uint32_t_vec_from_host_to_device(host_ptr, len as u32) }
    }
}

/// Rehydrate the WHOLE staged column back to the device (H2D). Returns an OWNED `BaseFieldVec` that
/// owns fresh device memory; the stash entry is retained (the committed column may be read again).
/// Panics if `col` is not staged. Bytes are bit-identical to the committed column.
pub fn rehydrate_owned(col: &BaseFieldVec) -> BaseFieldVec {
    // B2: ensure every async dehydrate D2H has landed before reading the stash bytes (once, cheap).
    sync_stash_copies_if_pending();
    let key = col.device_ptr as usize;
    let guard = HOST_STASH.lock().unwrap();
    let host = guard
        .get(&key)
        .expect("rehydrate_owned on a column with no stash entry (staged bytes missing)");
    debug_assert_eq!(host.len(), col.size);
    let len = host.len();
    let host_ptr = host.as_ptr();
    let device_ptr = rehydrate_h2d(host_ptr, len);
    BaseFieldVec::new(device_ptr, len)
}

/// Rehydrate ONLY the row-block `[row_offset, row_offset + block_rows)` of a staged column back to
/// the device (H2D of a contiguous slice). Returns an OWNED tile-sized `BaseFieldVec` (length
/// `block_rows`) that owns fresh device memory. This is the composition/quotient ROW-TILING supply
/// primitive (COMPOSITION_TILING_SCOPE §2): because the stash holds the exact committed bytes in
/// the same eval-domain order the kernel indexes, and every tree0/tree1 read is a pure `[row]`
/// pointwise read, the physical slice `host[row_offset .. row_offset+block_rows]` IS the logical
/// row-block — no reindexing. Panics if `col` is not staged, if the block exceeds the staged length
/// (fail-loud: never silently H2D wrong bytes), or if `block_rows == 0`. Bytes are bit-identical to
/// the corresponding slice of the committed column.
pub fn rehydrate_block(col: &BaseFieldVec, row_offset: usize, block_rows: usize) -> BaseFieldVec {
    assert!(block_rows > 0, "rehydrate_block with zero block_rows");
    // B2: ensure every async dehydrate D2H has landed before reading the stash bytes (once, cheap).
    sync_stash_copies_if_pending();
    let key = col.device_ptr as usize;
    let guard = HOST_STASH.lock().unwrap();
    let host = guard
        .get(&key)
        .expect("rehydrate_block on a column with no stash entry (staged bytes missing)");
    let end = row_offset
        .checked_add(block_rows)
        .expect("rehydrate_block row range overflow");
    assert!(
        end <= host.len(),
        "rehydrate_block block [{row_offset}, {end}) exceeds staged column length {}",
        host.len()
    );
    let device_ptr = rehydrate_h2d(host[row_offset..end].as_ptr(), block_rows);
    BaseFieldVec::new(device_ptr, block_rows)
}

/// The staged host base pointer + length for a staged column, for callers (the CU per-block H2D
/// loop) that copy row-block slices directly from the stash without a Rust-side device round-trip.
/// The pointer is valid for the lifetime of the stash entry (through decommit, per `clear_stash`
/// contract). Returns `None` if `col` is not staged. SOUNDNESS: the bytes are the exact committed
/// bytes; the caller must only read `[0, len)` (u32 == M31 payload).
pub fn staged_host_ptr(col: &BaseFieldVec) -> Option<(*const u32, usize)> {
    // Same `!owns_memory` anti-aliasing guard as `is_staged`: a resident committed column that the
    // mem-pool aliased onto a freed tree1 stash key OWNS its live buffer, so it must resolve to the
    // resident (NULL host-table) path, never the stale staged bytes.
    if col.owns_memory {
        return None;
    }
    // B2: ensure every async dehydrate D2H has landed before exposing the host bytes (once, cheap).
    sync_stash_copies_if_pending();
    let key = col.device_ptr as usize;
    // The returned host pointer aliases the stash entry's `Vec<u32>`, which lives in the global map
    // (a stable heap allocation) for the rest of the proof — the stash is only mutated by
    // `dehydrate_column` (all done before any reader) and `clear_stash` (next commit boundary), and
    // never during a reader window. So the pointer stays valid after the lock guard drops, exactly
    // as with the previous thread-local map. The caller (composition per-column host table) reads
    // it only within the synchronous kernel dispatch.
    let guard = HOST_STASH.lock().unwrap();
    guard
        .get(&key)
        .filter(|h| h.len() == col.size)
        .map(|h| (h.as_ptr(), h.len()))
}

/// Read the values at `indices` DIRECTLY from the staged host bytes (no device round-trip). Used by
/// the streamed `decommit` to recover the ~70 sparse query openings. Byte-identical to a resident
/// `batch_get`. Panics if `col` is not staged.
pub(crate) fn host_batch_get(col: &BaseFieldVec, indices: &[usize]) -> Vec<BaseField> {
    // B2: ensure every async dehydrate D2H has landed before reading the stash bytes (once, cheap).
    sync_stash_copies_if_pending();
    let key = col.device_ptr as usize;
    let guard = HOST_STASH.lock().unwrap();
    let host = guard
        .get(&key)
        .expect("host_batch_get on a column with no stash entry (staged bytes missing)");
    indices
        .iter()
        .map(|&i| BaseField::from_u32_unchecked(host[i]))
        .collect()
}

// ============================================================================
// PART A reclaim + T1 sub-timers.
// ============================================================================

/// T1 sub-timer accumulators (nanoseconds), summed across the whole streamed tree1 commit and
/// printed once via `t1_report`. Behind `GATE_AIR_T1_TIMERS` (or PROVE_EX_TIMERS) — negligible when
/// off (a few atomic adds). These decompose the tree1 streaming tax (TBASE_DECOMP_ANALYSIS Q1b)
/// into NTT compute / dehydrate D2H / dehydrate reclaim barrier / build_leaves rehydrate H2D /
/// absorb.
static T1_NTT_NS: AtomicUsize = AtomicUsize::new(0);
static T1_DEHYDRATE_D2H_NS: AtomicUsize = AtomicUsize::new(0);
static T1_DEHYDRATE_RECLAIM_NS: AtomicUsize = AtomicUsize::new(0);
static T1_BUILD_LEAVES_H2D_NS: AtomicUsize = AtomicUsize::new(0);
static T1_ABSORB_NS: AtomicUsize = AtomicUsize::new(0);

/// Whether T1 sub-timers are collected/printed. Cheap gate; reads env once is fine (called rarely).
pub fn t1_timers_on() -> bool {
    std::env::var("GATE_AIR_T1_TIMERS").is_ok() || std::env::var("PROVE_EX_TIMERS").is_ok()
}

/// Add `ns` to a named T1 accumulator. `which`: 0=NTT, 1=dehydrate D2H, 2=dehydrate reclaim,
/// 3=build_leaves H2D, 4=absorb.
pub fn t1_add(which: u8, ns: u128) {
    let acc = match which {
        0 => &T1_NTT_NS,
        1 => &T1_DEHYDRATE_D2H_NS,
        2 => &T1_DEHYDRATE_RECLAIM_NS,
        3 => &T1_BUILD_LEAVES_H2D_NS,
        _ => &T1_ABSORB_NS,
    };
    acc.fetch_add(ns as usize, Ordering::Relaxed);
}

/// Print + reset the T1 accumulators (grep tag `[T1]`). Call after the tree1 commit.
pub fn t1_report(label: &str) {
    let ntt = T1_NTT_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let d2h = T1_DEHYDRATE_D2H_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let reclaim = T1_DEHYDRATE_RECLAIM_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let h2d = T1_BUILD_LEAVES_H2D_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let absorb = T1_ABSORB_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    eprintln!(
        "[T1] {label}: ntt {ntt:.3}s | dehydrate_d2h {d2h:.3}s | dehydrate_reclaim {reclaim:.3}s \
         | build_leaves_h2d {h2d:.3}s | absorb {absorb:.3}s"
    );
}

/// T2 sub-timer accumulators (nanoseconds): the prove_ex staged-consumer rehydrate H2D vs. kernel
/// compute split (TBASE_DECOMP_ANALYSIS Q1c), per consumer. Printed via `t2_report` (tag `[T2]`).
static T2_OODS_H2D_NS: AtomicUsize = AtomicUsize::new(0);
static T2_OODS_KERNEL_NS: AtomicUsize = AtomicUsize::new(0);
static T2_QUOTIENT_H2D_NS: AtomicUsize = AtomicUsize::new(0);
static T2_QUOTIENT_KERNEL_NS: AtomicUsize = AtomicUsize::new(0);

/// Add `ns` to a T2 accumulator. `which`: 0=OODS H2D, 1=OODS kernel, 2=quotient H2D, 3=quotient
/// kernel.
pub fn t2_add(which: u8, ns: u128) {
    let acc = match which {
        0 => &T2_OODS_H2D_NS,
        1 => &T2_OODS_KERNEL_NS,
        2 => &T2_QUOTIENT_H2D_NS,
        _ => &T2_QUOTIENT_KERNEL_NS,
    };
    acc.fetch_add(ns as usize, Ordering::Relaxed);
}

/// Print + reset the T2 accumulators (grep tag `[T2]`). Call after prove_values / at proof end.
pub fn t2_report(label: &str) {
    let oods_h2d = T2_OODS_H2D_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let oods_k = T2_OODS_KERNEL_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let quot_h2d = T2_QUOTIENT_H2D_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    let quot_k = T2_QUOTIENT_KERNEL_NS.swap(0, Ordering::Relaxed) as f64 / 1e9;
    eprintln!(
        "[T2] {label}: oods_h2d {oods_h2d:.3}s oods_kernel {oods_k:.3}s | quotient_h2d \
         {quot_h2d:.3}s quotient_kernel {quot_k:.3}s (composition tile H2D is inside \
         evaluate_gate_air.cu — see the .cu timer if wired)"
    );
}

/// Reads `GATE_AIR_RECLAIM_TRIM` — when set, use the LEGACY per-column `cudaStreamSynchronize(0) +
/// cudaMemPoolTrimTo(0)` reclaim (releases each freed segment to the OS and re-maps it, the ~22 s
/// tax). DEFAULT (unset) = PART A: `cudaStreamSynchronize(0)` only (no trim), so the pool caches +
/// reuses the freed segment. Both keep the memory bound (the sync gates live buffers before the
/// next alloc); Part A just removes the OS release/re-map churn. Byte-identical either way.
fn reclaim_trim_enabled() -> bool {
    std::env::var("GATE_AIR_RECLAIM_TRIM").is_ok()
}

/// OPTION-0: one-shot pool defrag for the tree1->interaction boundary. Releases ALL cached
/// already-freed segments (the ~23.5 GiB of dehydrated tree1 eval buffers held by the no-trim
/// Part-A path) back to the OS so a subsequent large CONTIGUOUS alloc (the 3 GiB `d_inter` at 2^25)
/// fits. No-op unless `GATE_AIR_BOUNDARY_TRIM` is set (default OFF, so the coordinator can A/B it).
///
/// Call EXACTLY ONCE per proof, after the streamed tree1 commit completes and BEFORE interaction/K4
/// allocation — NOT per column (the per-column trim is the ~22 s churn Part A removed; this is a
/// single sync+trim, negligible). Pure allocator hygiene: `cudaMemPoolTrimTo(0)` only returns
/// already-freed segments to the OS; live buffers (`d_cols`, tree0, twiddles) are untouched.
/// Byte-identical proof, no working-set change, no soundness surface.
pub fn boundary_trim_if_enabled() {
    if std::env::var("GATE_AIR_BOUNDARY_TRIM").is_ok() {
        unsafe {
            bindings::cuda_pool_trim();
        }
    }
}

/// The streamed-path reclaim after a `cuda_free_memory`: PART A no-trim by default (sync only),
/// legacy sync+trim under `GATE_AIR_RECLAIM_TRIM`. Times itself into the T1 dehydrate-reclaim
/// accumulator when timers are on. Shared by `dehydrate_column` and `build_leaves`.
pub fn reclaim_after_free() {
    let timed = t1_timers_on();
    let start = timed.then(std::time::Instant::now);
    unsafe {
        if reclaim_trim_enabled() {
            bindings::cuda_stream_reclaim_freed(0);
        } else {
            bindings::cuda_stream_reclaim_freed_notrim();
        }
    }
    if let Some(s) = start {
        t1_add(2, s.elapsed().as_nanos());
    }
}

/// Reads `GATE_AIR_FUSED_COMMIT`. Any set value (e.g. `1`) enables the fused per-column
/// LDE->absorb path. Unset (the default) => OFF => current behavior, byte-for-byte.
///
/// STEP 2 layering: `GATE_AIR_STREAM_COMMIT` (see `stream_commit_enabled`) IMPLIES the fused loop,
/// so this returns true whenever either flag is set. That keeps THREE A/B-comparable states:
///   * neither set  => legacy resident path (DEFAULT, byte-for-byte unchanged);
///   * FUSED only    => fused-but-resident (step 1, no free/host-stage, no ceiling move);
///   * STREAM (± FUSED) => fused + free-after-absorb + host-stage (step 2, lifts the ceiling).
pub(crate) fn fused_commit_enabled() -> bool {
    std::env::var("GATE_AIR_FUSED_COMMIT").is_ok() || stream_commit_enabled()
}

/// Reads `GATE_AIR_STREAM_COMMIT` (STEP 2, opt-in, DEFAULT OFF). When set, the fused per-column
/// commit ADDITIONALLY host-stages each eval column: after a column is absorbed, its bytes are
/// D2H-copied to a host buffer and the device buffer is freed (`BaseFieldVec::dehydrate_to_host`),
/// dropping the commit peak below 40 GB so 2^24/2^25 base shards fit a 40 GB A100. Post-commit
/// readers (OODS/quotient, decommit) rehydrate/host-recover on demand. Layered strictly ON TOP of
/// the step-1 fused loop; when this is set `fused_commit_enabled()` is also true. The default-flip
/// (streaming as default for >2^23 with a `CUDA_LEGACY_COMMIT` opt-out) is a SEPARATE follow-up.
pub(crate) fn stream_commit_enabled() -> bool {
    std::env::var("GATE_AIR_STREAM_COMMIT").is_ok()
}

/// Reads `GATE_AIR_FUSED_INTERP` (Fix (b): eliminate the ~2x main-trace device-memory duplication,
/// opt-in, DEFAULT OFF). When set, the gate_air driver hands the tree1 main columns to the commit
/// as BORROWED views into the K1 `d_cols` buffer (un-interpolated base-domain EVALS) via
/// `extend_polys` instead of materializing 188 D2D copies + a batched in-place interpolate. The
/// fused `evaluate_polynomials` loop then interpolates each column PER-COLUMN into a single reused
/// temp buffer (protecting the borrowed `d_cols` view from the in-place b2n NTT) before the
/// existing extend->n2b->absorb->dehydrate steps. This removes the second full main-trace resident
/// copy so a 2^25 base proof fits a 40 GB A100.
///
/// Scoping (byte-identity safety for the shared `PolyOps` methods): a global env flag alone cannot
/// distinguish tree1's un-interpolated main group from the already-interpolated tree0/tree2 groups.
/// So the per-column interpolate fires ONLY for a group that contains at least one BORROWED
/// (`owns_memory == false`) input column — the structural signature of the
/// deliberately-un-interpolated main views. Every normal already-interpolated coeff column
/// (tree0/tree2/small_main and the flag-OFF main path) OWNS its device buffer, so it is never
/// re-interpolated. This flag IMPLIES the fused loop for that group (the interpolate is folded into
/// it).
pub(crate) fn interpolate_in_commit_enabled() -> bool {
    std::env::var("GATE_AIR_FUSED_INTERP").is_ok()
}
