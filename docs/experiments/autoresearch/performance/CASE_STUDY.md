# Liquid Case Study Pattern

The Shopify Liquid case study is the reference pattern for this lab, not because the exact code
changes transfer, but because the experiment structure does.

## What Liquid added

From PR `https://github.com/Shopify/liquid/pull/2056`:

1. a machine-readable benchmark:
   - `performance/bench_quick.rb`
   - emitted `parse_us`, `render_us`, `combined_us`, and `allocations`
2. a correctness + performance contract:
   - `auto/bench.sh`
   - unit tests, `liquid-spec`, then the benchmark
3. an explicit agent charter:
   - `auto/autoresearch.md`
   - objective, files in scope, off-limits paths, constraints, baseline, progress log
4. a running experiment log:
   - `autoresearch.jsonl`
   - kept the search process inspectable instead of magical

## Why it worked

The PR did not ask the agent to "make Liquid faster" in the abstract. It gave the agent:

- one benchmark contract,
- a clear optimization target,
- known correctness gates,
- strong security constraints,
- a narrow search surface.

That caused the loop to converge on a series of small, measurable wins:

- regex removal in hot parsing paths,
- allocation avoidance,
- fast paths for common cases,
- progressively more direct byte-level scanning.

## Reported outcome

Karpathy summarized the run as:

- 53% faster combined parse + render time
- 61% fewer object allocations

The PR history shows the gains were incremental and benchmark-driven rather than a single large
rewrite.

## What transfers to STWO

Directly transferable:

- benchmark-first contract
- append-only experiment log
- one-experiment-at-a-time loop
- explicit scope and safety boundaries
- preference for simple, local wins before structural rewrites

Not directly transferable:

- Liquid's parser-centric fast paths
- Ruby object allocation heuristics
- its looser security boundary relative to cryptographic soundness work

## STWO-specific translation

For `stwo`, the core adaptations are:

- use Criterion JSON output as the machine-readable benchmark contract,
- keep autonomous edits out of soundness-critical code,
- reject any experiment that touches forbidden paths,
- reject any experiment that introduces or edits `unsafe`,
- gate improvements through targeted tests before accepting them.

This is stricter than Liquid because `stwo` performance work sits adjacent to proof-system logic.

