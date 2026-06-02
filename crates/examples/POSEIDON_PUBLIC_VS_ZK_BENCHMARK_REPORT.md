# Poseidon public vs pre-target-bearing masking benchmark report

`log_n_instances`: 10

Pre-target-bearing masking mode: original trace and LogUp interaction running-sum columns are randomized, but `claimed_sum` remains public for lookup soundness. This is not a sound private LogUp ZK benchmark under the target-bearing signoff rule.

Metadata builder: `stwo::core::zk::build_stwo_zk_air_metadata`; Poseidon supplies AIR-specific tree policy and reviewed degree expansion.

Public metadata leakage: private column positions/counts, degree bounds, tree scopes, and stable metadata hashes are public circuit metadata and must not encode secrets.

| bucket | value |
|---|---:|
| public Poseidon prove | 14.419333ms |
| ZK original witness randomization | 24.510125ms |
| ZK LogUp interaction randomization | 3.585375ms |
| ZK quotient/composition masking + ZK FRI/proof generation | 429.509666ms |
| public verify | 1.154541ms |
| ZK verify | 130.80725ms |
| ZK trace domain log size | 7 |
| ZK randomized witness log degree | 8 |
| ZK FRI first layer log size | 13 |
| ZK quotient log degree bound | 11 |
| ZK trace tree scope hash | [179, 172, 152, 186, 159, 94, 32, 182, 136, 56, 101, 108, 185, 96, 7, 19, 246, 128, 193, 84, 189, 216, 208, 244, 112, 133, 56, 56, 3, 201, 230, 184] |
| ZK original private range count | 1264 |
| ZK LogUp interaction private range count | 32 |
| public proof size estimate bytes | 41024 |
| ZK proof size estimate bytes | 90404 |
| proof size delta bytes | 49380 |
| public sampled value count | 1308 |
| ZK sampled value count | 1308 |
| sampled value count delta | 0 |
| public queried/opened value count | 3912 |
| ZK queried/opened value count | 3912 |
| queried/opened value count delta | 0 |
