use core::alloc::Allocator;
use std::alloc::{AllocError, Global, Layout};
use std::ptr::NonNull;

/// 512-bit aligned allocator.
#[derive(Debug, Clone)]
pub struct Aligned512Bit;

unsafe impl Allocator for Aligned512Bit {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout.align_to(64).unwrap().pad_to_align())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout.align_to(64).unwrap().pad_to_align());
    }
}
