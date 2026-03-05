# LLM Distillation Index

## Paper Map
| Paper | 1-line contribution | Key parameters | Key invariants | Distilled file |
|---|---|---|---|---|
| Circle STARKs | Defines circle-curve STARK stack (codes, FFT, AIR IOP, circle FRI) with explicit handling of 1D FFT-space gap. | `p` CFFT-friendly, `N=2^n`, `rho=(N+1)/|D|`, `theta<1-sqrt(rho)`, AIR degree `d`. | `L_N = L'_N + <v_n>`, quotient decomposition includes scalar `lambda`, DEEP quotients must respect circle zerofier field model. | [`Circle_STARKs.llm.md`](./Circle_STARKs.llm.md) |
| S-two Whitepaper | Product-focused protocol spec for multi-table flat AIR + multi-domain circle FRI + NIROP + parameter sets. | `M31/QM31`, `B=2^beta`, `rho=2^{-beta}`, Johnson `theta` range, NIROP `lambda=256`, grinding bits. | Cross-domain correlated agreement over lifted domains, logUp multiplicity bounds, degree-corrected batch eval condition, ROM weakest-link soundness. | [`Stwo_Whitepaper.llm.md`](./Stwo_Whitepaper.llm.md) |

## Unified Glossary
| Symbol | Unified meaning |
|---|---|
| `q` / `p` | Base prime modulus (`q` in whitepaper, `p` in circle paper). |
| `C(F)` | Circle curve/group points satisfying `x^2+y^2=1` over field `F`. |
| `G_n` | Size-`2^n` subgroup of circle group. |
| `G_n'` / standard position coset | Canonical/twin FFT domain shift used for interpolation/evaluation. |
| `H` / `H_t` | Trace or witness domain(s). |
| `D` / `D_t` | Evaluation domain(s). |
| `L_N` / `\mathcal L_N` | Full circle polynomial space with degree bound `N/2`. |
| `L'_N` / `\mathcal L'_N` | FFT image subspace (codimension 1 in full space). |
| `v_D` | Vanishing polynomial (domain zerofier). |
| `rho` | Code rate. |
| `delta` | Minimum distance (`1-rho` up to tiny domain-size corrections). |
| `theta` | Proximity parameter (distance). |
| `ell(theta)` | List-size bound for proximate codewords. |

## Conflict Notes
- Notation clash: circle paper uses `p` for base prime and `N` for degree/size; whitepaper uses `q`, `n_t`, and script-indexed spaces (`L_n`). Treat as same objects after renaming.
- Space naming mismatch: whitepaper `\mathscr L_n` corresponds to circle paper `\mathcal L_N` with `N=2^n`.
- Protocol maturity mismatch: whitepaper includes implementation deviations and roadmap items; circle paper is mostly theory-first.
- Proven vs conjectured regimes are mixed in whitepaper parameter section; keep them explicitly separated in downstream prompting.
