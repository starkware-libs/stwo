dev benchmark results can be seen at
https://starkware-libs.github.io/stwo/dev/bench/index.html

To run a bench:
```bash
cargo bench --features prover,parallel --bench [BENCH_NAME]
```
where `BENCH_NAME` is the name of the bench as it appears in `../Cargo.toml`.