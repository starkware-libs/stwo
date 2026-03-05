# Circle STARKs
Scope: Core math/algorithms/soundness for circle FFT, circle codes, circle FRI, and circle-STARK AIR IOP.

## 1. One-Screen Core
- Works over primes `p` where `p+1` has large `2`-adic factor; this gives large smooth subgroups on circle curve `x^2+y^2=1` and enables FFT-like recursion. [Def `def:prime:CFFT:friendly`]
- Circle curve points form a cyclic group of size `p+1` (for `p ≡ 3 mod 4`), with group square map `pi` and inversion `J` as core algebraic operators for FFT and FRI folding. [Sec `s:circle:curve`, Eq `e:pi`, Eq `e:J`]
- Polynomial space for degree bound `N/2` has dimension `N+1`; circle-FFT image space has dimension `N`. This 1D gap is the central anomaly vs univariate STARKs and must be handled explicitly. [Prop `prop:LN:properties`, Lem `lem:FFT:space:monomial:basis`]
- Circle codes are MDS and isomorphic to Reed-Solomon over a derived set; RS tools (list decoding/correlated agreement) transfer. [Thm `thm:circlecode:isomorphism`]
- Circle FFT/interpolation and inverse FFT on twin-cosets cost `O(N log N)` with Cooley-Tukey-like butterfly structure and precomputed twiddles. [Thm `thm:FFT`, Thm `thm:FFT:inverse`, Alg `alg:cfft`]
- AIR proving uses quotienting by trace-domain vanishing polynomial, then decomposes quotient into segment quotients plus one scalar `lambda` that fixes the 1D gap. [Eq `e:overall:identity`, Eq `e:quotient:decomposition`, Lem `lem:quotient:decomposition`]
- DEEP linking uses single-point quotient over `F(i)` and then batches real/imaginary parts for proximity testing. [Prop `prop:deep:quotients`, Rem `rem:DEEP:quotients:batching`]
- Circle FRI starts with decomposition `f = g + lambda * v_n` to enter FFT-space, then folds along projection chain; batching is via random linear combinations. [Prot `prot:IOP:proximity`, Prot `prot:IOP:proximity:batch`]
- Soundness: batch FRI gives correlated agreement guarantees; AIR IOP gives (knowledge) soundness with explicit error formula. [Thm `thm:FRI:soundness:round:by:round`, Thm `thm:AIR:soundness`, Thm `thm:IOP:AIR:extractor`]

## 2. Minimal Notation
| Symbol | Meaning | Constraints |
|---|---|---|
| `p` | Prime field modulus | `p ≡ 3 (mod 4)`; CFFT-friendly means `2^{n+1} | (p+1)` for needed `n` [Def `def:prime:CFFT:friendly`] |
| `C(F_p)` | Circle curve/group `x^2+y^2=1` over base field | cyclic of size `p+1` [Sec `s:circle:curve`] |
| `G_n` | Unique subgroup of circle group of size `2^n` | exists for supported order `n` |
| `D` | Twin-coset domain `Q*G_{n-1} ∪ Q^{-1}*G_{n-1}` | disjoint union required [Def `def:standard:twin:coset`] |
| `H` | Trace domain (standard position coset) | size `N=2^n` |
| `N` | Base code/order size | `N=2^n` |
| `L_N(F)` / `\mathcal L_N(F)` | Circle polynomial space of total degree `<= N/2` | dim `N+1` [Eq `e:LN:F`, Prop `prop:LN:properties`] |
| `L'_N(F)` / `\mathcal L'_N(F)` | FFT subspace | dim `N`; `L_N = L'_N + <v_n>` [Eq `e:LFFT:alternative`] |
| `v_D` | Vanishing polynomial of domain `D` | unique up to scalar in `L_N` [Lem `lem:vanishing:space`] |
| `v_P` | Single-point vanishing function for DEEP | quotient lives in `L_N(F(i))` [Eq `e:vanishing:function:singlepoint`, Prop `prop:deep:quotients`] |
| `rho` | Code rate | `rho=(N+1)/|D|` |
| `theta` | Proximity distance parameter | FRI uses `theta < 1 - sqrt(rho)` [Sec `s:FRI`] |
| `d` | AIR max constraint degree | usually `d <= |D|/|H| + 1`; theorem assumptions include `d*2^n < p+1` |

## 3. Model + Assumptions
- Algebraic setting:
  - Prime field with `p ≡ 3 mod 4`; for FFT/STARK use CFFT-friendly primes with enough supported order. [Def `def:prime:CFFT:friendly`]
  - Function space on circle curve; coding uses evaluations over circle subsets. [Sec `s:circle:curve`, Eq `e:LN:F`]
- Domain geometry:
  - FFT domains are twin-cosets; standard position cosets are a special case and are unique when supported. [Def `def:standard:twin:coset`, Prop `prop:standard:position:coset`]
  - Repeated squaring map halves domain size and preserves twin-coset form. [Lem `lem:twincosets:images`]
- Coding assumptions:
  - Circle codes are RS-isomorphic; list-decoding and correlated-agreement theorems from RS apply through isomorphism. [Thm `thm:circlecode:isomorphism`]
- AIR assumptions in STARK section:
  - Neighbor-row constraints with selector polynomials; max degree bounded so quotients fit available domains. [Sec `s:STARK`, Eq `e:AIR:constraints`]
  - DEEP query sampled outside committed/evaluation/trace domains. [Sec `s:STARK`, Prot `prot:IOP:AIR`]
- Soundness regime assumptions:
  - Johnson/list-decoding regime with multiplicity parameter `m>=3` in correlated agreement results. [Thm `thm:correlated:agreement`, Thm `thm:correlated:agreement:weighted`]

## 4. Algorithms / Constructions
- Circle FFT interpolation (twin-coset domain):
  - Step 1: split by involution `J` into `f0,f1` using `y` twiddle. [Eq `e:J:even`, Eq `e:J:odd`, Eq `e:J:combine`]
  - Step 2+: split over projected line domains via `x -> 2x^2-1` into even/odd parts using `x` twiddle. [Eq `e:T:even`, Eq `e:T:odd`, Eq `e:T:combine`]
  - Recurse until singleton; returned leaves are FFT-basis coefficients. [Thm `thm:FFT`]
- Inverse circle FFT:
  - Reverse recursion, recombining child values with same butterfly structure; outputs evaluations over chosen twin-coset. [Thm `thm:FFT:inverse`]
- AIR quotient construction:
  - Build combined identity with random `beta`: `sum beta^{i-1}*constraint_i = q*v_H`. [Eq `e:overall:identity`]
  - Decompose `q` over disjoint twin-cosets: `q = lambda*v_barH + sum (v_barH/v_Hk)*q_k`, `q_k in L'_N`. [Eq `e:quotient:decomposition`, Lem `lem:quotient:decomposition`]
  - Determine `lambda` by infinity-limit calculus/parity constraints when needed. [Eq `e:limit:infty`, Sec `s:computing:quotients`]
- Circle FRI (single domain):
  - Commit phase: decompose `f = g + lambda v_n`; fold `g` round-by-round using random linear combinations of even/odd parts. [Prot `prot:IOP:proximity`, Eq `e:FRI:folding`]
  - Query phase: spot-check fold identities along sampled trace. [Eq `e:FRI:g:even`, Eq `e:FRI:g:odd`]
- Batch circle FRI:
  - Randomly batch many functions into one linear combination, then run same FRI checks plus batch consistency checks. [Prot `prot:IOP:proximity:batch`, Eq `e:FRI:batching:step`]

## 5. Main Results (Theorems/Lemmas)
| ID | Statement (compressed) | Depends on | Anchor |
|---|---|---|---|
| `lem:isomorphism` | Circle curve isomorphic to `P^1`; `|C(F)|=|F|+1` | stereographic parametrization | [Lem `lem:isomorphism`] |
| `prop:LN:properties` | `L_N` rotation-invariant, dimension `N+1`, nonzero has <=`N` zeros | circle geometry | [Prop `prop:LN:properties`] |
| `thm:circlecode:isomorphism` | Circle code is RS-isomorphic; map/inverse preserve distance and are efficient | `P^1` isomorphism | [Thm `thm:circlecode:isomorphism`] |
| `prop:domain:quotients` | If `f` vanishes on domain `D`, then `f=q*v_D` with degree drop | vanishing-space lemma | [Prop `prop:domain:quotients`] |
| `prop:deep:quotients` | Single-point quotient over circle stays in `L_N(F(i))`; real/imag parts in `L_N(F)` | single-point zerofier | [Prop `prop:deep:quotients`] |
| `thm:FFT` | Circle FFT returns basis coefficients interpolating values on twin-coset | recursive even/odd decompositions | [Thm `thm:FFT`] |
| `thm:FFT:inverse` | Inverse transform evaluates FFT-basis polynomial on domain | reverse recursion | [Thm `thm:FFT:inverse`] |
| `prop:FFT:space:limits` | FFT space equals subspace with alternating infinity limits | limit lemmas + decomposition | [Prop `prop:FFT:space:limits`] |
| `lem:quotient:decomposition` | Quotient over union of twin-cosets decomposes into `lambda` + per-coset FFT-space segments | interpolation on disjoint union | [Lem `lem:quotient:decomposition`] |
| `thm:FRI:soundness:round:by:round` | Batch circle FRI enforces correlated agreement (round-by-round soundness) | correlated agreement theorems | [Thm `thm:FRI:soundness:round:by:round`] |
| `thm:AIR:soundness` | AIR IOP soundness with explicit error bound and proximity extraction target | FRI soundness + DEEP checks | [Thm `thm:AIR:soundness`] |
| `thm:IOP:AIR:extractor` | Knowledge soundness: extractor recovers AIR witness polynomials | correlated-agreement decoder | [Thm `thm:IOP:AIR:extractor`] |

## 6. Parameter Rules
- Prime/order rules:
  - Need supported order `n` so `2^{n+1} | (p+1)`; practical target `p=2^{31}-1`. [Def `def:prime:CFFT:friendly`, Sec `s:notation`]
- Domain size/rate rules:
  - Trace size `N=2^n`; evaluation domain typically `|D| = 2^B * N`; code rate `rho=(N+1)/|D|`. [Sec `s:STARK`, Sec `s:FRI`]
- FRI distance rule:
  - Run in Johnson regime `theta < 1 - sqrt(rho)` (or equivalent `alpha > sqrt(rho)` forms in soundness appendix). [Sec `s:FRI`, Lem `lem:FRI:soundness`]
- Quotient-degree rules:
  - AIR max degree `d` must keep quotient degrees/domain sizes feasible (`d*2^n < p+1` in theorem assumptions). [Summary theorem in `s:STARK`, Thm `thm:AIR:soundness` context]
- Correlated-agreement thresholds:
  - Uses explicit `epsilon` bounds depending on batch size, domain size, list/multiplicity parameters, and `|F|`. [Thm `thm:correlated:agreement`, Eq `e:epsilonJ`, Thm `thm:correlated:agreement:weighted`]

## 7. Complexity
| Component | Prover | Verifier | Memory | Notes |
|---|---|---|---|---|
| Circle FFT / inverse | `N log N` adds, `(N/2)log N` mults (precomputed twiddles) | same asymptotic when evaluating/checking polynomial data | in-place butterfly possible | [Thm `thm:FFT`, Thm `thm:FFT:inverse`, Rem `rem:FFT:scaled`, Alg `alg:cfft`] |
| RS/circle list decode (GS) | deterministic poly-time; bound given | n/a | poly-time | worst-case bound `O(|D|^{15})` | [Cor `cor:circlecode:GS`] |
| FRI query phase | `s` sampled traces and fold checks | `s` fold checks | oracle/Merkle model | error term `(1-theta)^s` | [Prot `prot:IOP:proximity`, Eq `e:FRI:soundness:error`] |
| AIR IOP extra rounds | quotient + DEEP claims + batch proximity | identity checks + OOD checks + FRI | depends on batching | explicit soundness formula provided | [Thm `thm:AIR:soundness`] |

## 8. Implementation-Critical Invariants
- Domain invariants:
  - Twin-coset domains must be `J`-stable and disjoint pair unions; projections by `pi` must follow the expected chain. [Def `def:standard:twin:coset`, Lem `lem:twincosets:images`]
- Polynomial-space invariants:
  - Witness/quotient segments that are assumed in FFT-space must satisfy alternating-limit condition (`L'_N = L^-_N`). [Prop `prop:FFT:space:limits`]
  - `L_N` decomposition into `L'_N + <v_n>` must be accounted for whenever interpolation domain has size `N`. [Sec `s:FFT:spaces`]
- Quotient invariants:
  - Overall quotient decomposition requires per-coset terms `q_k in L'_N` plus scalar `lambda`; dropping `lambda` without parity argument is unsound. [Eq `e:decomposition:q`, Sec `s:computing:quotients`]
- DEEP invariants:
  - Single-point quotient must use circle-compatible zerofier and correct field handling (`F(i)` or equivalent real/imag split). [Prop `prop:deep:quotients`]
- FRI invariants:
  - First decomposition (`f = g + lambda v_n`) is mandatory before folding to keep halving-compatible spaces. [Prot `prot:IOP:proximity`]
  - Fold spot-check equations must match exact twiddle/domain definitions. [Eq `e:FRI:g:even`, Eq `e:FRI:g:odd`]

## 9. Proof Skeletons (Only Essential Logic)
- Circle code is RS-isomorphic:
  - Translate circle points to projective-line coordinates.
  - Translate `L_N` to rational univariate form with denominator `(1+t^2)^{N/2}`.
  - Multiply by denominator to recover standard RS polynomial evaluation form.
  - Conclude distance-preserving linear isomorphism. [Thm `thm:circlecode:isomorphism`]
- Circle FFT correctness:
  - Induct on recursion depth.
  - At each step, decomposition equations define unique even/odd child functions on projected domain.
  - Basis functions align with recursive twiddle products.
  - Base case singleton gives constant coefficient; reconstruct full expansion. [Thm `thm:FFT`]
- Quotient decomposition lemma:
  - Build selector terms `v_barH/v_Hk` that isolate each twin-coset.
  - Interpolation over all cosets gives `|barH|`-dimensional image.
  - Add missing vanishing polynomial `v_barH` to close the final 1D gap.
  - Uniqueness from linear independence and dimension count. [Lem `lem:quotient:decomposition`]
- AIR soundness outline:
  - Random linear combinations reduce many constraints to one identity.
  - OOD point check reduces domain identity to polynomial identity with bounded degree.
  - Batch FRI enforces correlated proximity of all quotient-related codewords.
  - Extractor enumerates bounded candidate lists and recovers a valid witness tuple. [Thm `thm:AIR:soundness`, Thm `thm:IOP:AIR:extractor`]

## 10. Failure Modes / Edge Cases
- Using odd-sized domains for vanishing-polynomial logic over base field breaks uniqueness guarantees. [Lem `lem:vanishing:space` + remark below it]
- Ignoring the `+1` dimension gap when reconstructing quotients can produce underdetermined or inconsistent quotient representations. [Sec `s:computing:quotients`]
- Wrong parity handling at infinity (`L^+` vs `L^-`) can produce wrong `lambda` computation. [Sec `s:FFT:spaces`, Eq `e:L:minus`, Eq `e:L:plus`]
- DEEP quotient over wrong field model (or wrong zerofier) can leave target space and break FRI assumptions. [Prop `prop:deep:quotients`]
- Too-small extension field `F` inflates random-challenge collision/zero-test errors (`1/|F|` terms). [Thm `thm:AIR:soundness`, correlated-agreement theorems]
- Optimized non-ZK variant changes assumptions and is not zero-knowledge. [Sec `s:STARK:optimized`]

## 11. Ambiguities / Open Questions
- Non-interactive (ROM) formalization is postponed in this paper version; appendix focuses on oracle model. [Rem after AIR soundness summary]
- Full production implementation of complete circle STARK was not finished at paper time; only FFT benchmarks were reported. [Sec `s:implementation:remarks`]
- Some optimizations are sketched (e.g., punctured-domain selectors, non-ZK domain variant) but need separate security/engineering treatment before default use. [Rem `rem:transitional:constraints`, Sec `s:STARK:optimized`]

## 12. Source Anchor Map
- Field/order assumptions: [Def `def:prime:CFFT:friendly`], [Sec `s:notation`]
- Circle group operations: [Sec `s:circle:curve`], [Eq `e:pi`], [Eq `e:J`]
- Polynomial/coding foundations: [Eq `e:LN:F`], [Prop `prop:LN:properties`], [Def `def:circle:code`], [Thm `thm:circlecode:isomorphism`]
- Vanishing/quotients: [Lem `lem:vanishing:space`], [Prop `prop:domain:quotients`], [Prop `prop:deep:quotients`]
- FFT basis/algorithm: [Def `def:FFT:basis`], [Thm `thm:FFT`], [Thm `thm:FFT:inverse`], [Lem `lem:FFT:space:monomial:basis`]
- Dimension-gap handling: [Sec `s:FFT:spaces`], [Prop `prop:FFT:space:limits`], [Lem `lem:quotient:decomposition`], [Eq `e:decomposition:q`]
- AIR protocol core: [Eq `e:overall:identity`], [Eq `e:quotient:decomposition`], [Sec `s:computing:quotients`]
- FRI protocol/soundness: [Prot `prot:IOP:proximity`], [Prot `prot:IOP:proximity:batch`], [Thm `thm:FRI:soundness:round:by:round`]
- AIR soundness/extraction: [Thm `thm:AIR:soundness`], [Prop `prop:correlated:agreement:decoder`], [Thm `thm:IOP:AIR:extractor`]
- Implementation benchmark snapshot: [Sec `s:implementation:remarks`], [Alg `alg:cfft`]
