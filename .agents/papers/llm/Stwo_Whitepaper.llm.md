# S-two Whitepaper
Scope: Protocol-as-implemented math model for S-two: flat AIR + multi-domain circle FRI + non-interactive transform + soundness and parameterization.

## 1. One-Screen Core
- S-two is a circle-STARK over `M31 = F_q`, `q=2^31-1`, with extension-field challenges (current: `QM31 = F_{q^4}`). [Sec `s:preleminaries`, Sec `s:fields`]
- Witness polynomials are encoded in circle-FFT spaces on canonical cosets `G_n'`; each component table can live on its own domain height (`multi-table` setting). [Sec `s:cFFT`, Sec `s:IOPP:encoding`]
- Arithmetization target is flat AIR: many component tables + multiset bus consistency (`uses` == `yields` with multiplicities). [Def `def:flat:AIR`, Def `def:flat:AIR:solution`, Eq `e:flat:AIR:consistency`]
- Main interactive protocol (`IOPP for flat AIR`) has 3 core rounds:
  - logUp poles + running sums,
  - composition quotient + cross-domain linear combination,
  - out-of-domain query,
  then batch evaluation proof via multi-domain circle FRI. [Prot `prot:STARK:IOPP`]
- Multi-domain circle FRI is the central PCS/proximity layer: cross-domain batching + per-level folding-insertion + sampled query checks across projection trace. [Prot `prot:cFRI:multi`]
- Soundness in list-decoding (Johnson) regime is explicit and round-by-round, with cross-domain correlated agreement as key invariant to avoid combinatorial blow-up. [Thm `thm:IOPP:soundness`, Thm `thm:cFRI:multi:soundness`, Def `def:IOPP:relation`]
- Non-interactive STARK uses BCS+Fiat-Shamir with cross-domain Merkle commitments and optional grinding; ROM soundness depends on max round error and hash collision term. [Sec `s:STARK`, Thm `thm:NIROP:soundness:general`]
- Paper includes both proven and conjectural parameter regimes; conjectural regime relies on list/line-decodability assumptions up to Elias radius. [Sec `s:example:params`, Conj `conj:list:decodability`, Conj `conj:prox:gaps`]

## 2. Minimal Notation
| Symbol | Meaning | Constraints |
|---|---|---|
| `M31` / `F_q` | base field | `q = 2^31 - 1` [Sec `s:fields`] |
| `CM31` | quadratic extension | `F_{q^2} = F_q[i]/(i^2+1)` [Eq `e:complex:extension`] |
| `QM31` | quartic extension | `F_{q^4} = F_{q^2}[j]/(j^2-2-i)` [Eq `e:quartic:extension`] |
| `C(F)` | circle curve points `x^2+y^2=1` | group action by `SO(2,F)` [Eq `e:circle:curve:affine`, Eq `e:SO2:def`] |
| `G_n` / `G_n'` | subgroup / canonical coset | `|G_n|=2^n`, `G_n' = Q_{n+1} G_n` [Eq `e:subgroup:chain`, Eq `e:canonical:coset`] |
| `H_t` | witness domain of table `t` | `H_t = G_{n_t}'` [Eq `e:STARK:witness:domain`] |
| `D_t` | evaluation domain of table `t` | `D_t = G_{n_t+beta}'` [Eq `e:STARK:evaluation:domain`] |
| `B=2^beta` | blowup factor | `rho=1/B=2^{-beta}` [Eq `e:STARK:rate`] |
| `L_n(F)` / `L_n'(F)` | full / FFT circle spaces | `dim L_n = 2^n+1`, `L_n = L_n' + <v_n>` [Eq `e:circle:polynomials:N`, Eq `e:cFFT:space:decomposition`] |
| `theta` | proximity distance | Johnson regime: `theta in [delta/2, 1-sqrt(1-delta))` |
| `delta` | code min distance | for rate `rho`: `delta = 1-rho` (or `1-2^{-beta}` in IOPP notation) [Eq `e:min:distance:circle:codes`] |
| `ell(theta)` | list-size bound | combinatorial Johnson-like bound [Eq `e:list:bound:Johnson`] |
| `ell_GS(theta)` | GS list-size bound | multiplicity-based bound [Eq `e:GS:list:bound`, Eq `e:GS:multiplicity`] |

## 3. Model + Assumptions
- Field/model assumptions:
  - Base field is `M31`; protocol may generalize to other circle-FFT-friendly primes. [Sec `s:circle:IOPP`]
  - Extension-field challenges sampled from `F >= F_{q^2}`; current implementation uses `QM31`. [Prot `prot:STARK:IOPP`, Sec `s:implementation`]
- Domain/degree assumptions:
  - Each table height `N_t=2^{n_t}`; blowup `B=2^beta`; encoding rate `rho=2^{-beta}`. [Sec `s:circle:IOPP`, Eq `e:STARK:rate`]
  - AIR degree bound constrained by blowup and max domain capacity: `deg(A) <= B+1`, `N*B^2 <= 2^30`, `N=max_t N_t`. [Eq `e:STARK:degree:bound`, Eq `e:max:domain:size`]
- Flat AIR assumptions:
  - Components define row constraints + use/yield message maps + multiplicity/enabler semantics. [Def `def:flat:AIR`, Def `def:flat:AIR:solution`]
  - Channel consistency is multiset equality over all components plus public inputs/outputs. [Eq `e:flat:AIR:consistency`]
- Soundness assumptions:
  - Multiplicities for use messages must stay below characteristic to avoid modulo-`q` cancellation pathologies in logUp argument. [Thm `thm:IOPP:soundness`, Rem `rem:IOPP:soundness`]
  - Minimal table-height condition for degree-corrected evaluation reduction:
    `1-theta > (1-delta)*(1+2/N_t)`. [Eq `e:soundness:min:table:height`]

## 4. Algorithms / Constructions
- Circle FFT (foundation):
  - First split by `(x,y)` vs `(x,-y)` into line components; then recurse univariately with `x -> 2x^2-1`. [Eq `e:cFFT:step:1`, Eq `e:cFFT:step:2`]
  - Iterative in-place algorithms with precomputed twiddles are provided for evaluation/interpolation. [Sec `s:cFFT:pseudocode`, Eq `e:twiddles`, Alg `alg:IFFT`, Alg `alg:FFT`]
- IOPP for flat AIR (`prot:STARK:IOPP`):
  - Round 1 (`logUp`): build inverse-pole helper polynomials for uses/yields and centered running sums. [Eq `e:constraint:uses`, Eq `e:constraint:yields`, Eq `e:constraint:sum:increment`]
  - Round 2 (composition): build per-table quotient `q_t = p_t/v_{n_t}` then cross-domain combination `q = sum gamma^{alpha_t} q_t`. [Eq `e:crossdomain:quotient`]
  - Round 3 (OOD query): open values at random `Q`; verify algebraic identity at `Q`; then run batch evaluation proof.
- Multi-domain circle FRI (`prot:cFRI:multi`):
  - Cross-domain batch each domain with one random scalar. [Eq `e:cFRI:multi:batching`]
  - Per-level fold inserts batched domain function into FRI cascade: `g_i = g_even + lambda_i g_odd + lambda_i^2(h_even + lambda_i h_odd)`. [Eq `e:cFRI:multi:folding`]
  - Query checks validate both batching identity and fold identity on sampled projection traces. [Eq `e:cFRI:multi:batching:check`, Eq `e:cFRI:multi:folding:check`]
- Batch evaluation proof (`prot:cFRI:batch:eval:degree:corrected`):
  - Prove low degree for both non-quotients and single-point quotients `(f-v)/v_Q` over all domains. [Eq `e:cFRI:batch:eval:extended:ensemble`]
  - Degree correction avoids small-domain degeneration in quotient-only mode. [Eq `e:cFRI:batch:eval:condition:rho:plus`]
- Non-interactive transform:
  - Cross-domain Merkle tree shares one auth path along projection chain; transposed bit-reverse indexing gives group-squaring consistency in the abstract design. [Sec `s:cross:domain:merkle`, Eq `e:tranposed:bitreverse:index`, Eq `e:transposed:bitreverse:pi:consistent`, Alg `alg:merkle`]
  - Fiat-Shamir derives challenges from transcript hash chain; grinding multiplies down selected round errors by `2^{-z_i}`. [Eq `e:FiatShamir:randomness`]

## 5. Main Results (Theorems/Lemmas)
| ID | Statement (compressed) | Depends on | Anchor |
|---|---|---|---|
| `lem:circle:parametrization` | Circle curve is projectively parameterizable (`P^1 <-> C`) | stereographic map | [Lem `lem:circle:parametrization`] |
| `lem:SO2:cyclic` | Circle rotation group over `F_q` (`q≡3 mod 4`) is cyclic of order `q+1` | embedding into `F_{q^2}^*` | [Lem `lem:SO2:cyclic`] |
| `thm:circle:decoder` | Circle codes list-decodable to Johnson radius; GS decoder gives deterministic list output | RS equivalence | [Thm `thm:circle:decoder`] |
| `thm:IOPP:soundness` | Flat-AIR IOPP is round-by-round knowledge-sound with explicit per-round bounds | cross-domain CA + FRI soundness | [Thm `thm:IOPP:soundness`] |
| `thm:cFRI:multi:soundness` | Multi-domain circle FRI is round-by-round knowledge-sound for cross-domain relation | CA under constraints + folding reductions | [Thm `thm:cFRI:multi:soundness`] |
| `thm:cFRI:batch:eval:degree:corrected:soundness` | Batch OOD evaluation proof inherits FRI soundness bounds under degree-correction condition | multi-domain FRI theorem | [Thm `thm:cFRI:batch:eval:degree:corrected:soundness`] |
| `thm:NIROP:soundness:general` | BCS-transformed NIROP soundness: `(T+R)*max eps_i + 3*(T^2+1)/2^lambda` | IOP + ROM analysis | [Thm `thm:NIROP:soundness:general`, Eq `e:NIROP:soundness:error`] |
| `thm:CAT:subcodes` | Correlated agreement under linear constraints (key for constrained/protocol settings) | curve-decodability machinery | [Thm `thm:CAT:subcodes`] |
| `thm:cocurvilinear:proximates` | RS curve-decodability theorem with explicit threshold/concurrency structure | improved proximity-gap results | [Thm `thm:cocurvilinear:proximates`] |
| `conj:list:decodability` | Conjectured prime-field RS list-decodability up to Elias radius with controlled growth | random-code evidence + known counterexamples | [Conj `conj:list:decodability`] |
| `conj:prox:gaps` | Conjectured prime-field line-decodability (thus CA) up to Elias radius | list-decodability + amplification intuition | [Conj `conj:prox:gaps`] |

## 6. Parameter Rules
- Structural parameters:
  - `B=2^beta`, `rho=2^{-beta}`; max table size `N=max_t N_t`. [Eq `e:STARK:rate`]
  - Enforced setup: `deg(A)<=B+1` and `N*B^2<=2^30` for available circle domains. [Eq `e:STARK:degree:bound`, Eq `e:max:domain:size`]
- Proximity parameter:
  - Proven regime focuses on Johnson interval `theta in [delta/2, 1-sqrt(1-delta))`. [Eq `e:Johnson:radius`]
  - Additional requirement for small domains in batch evaluation reduction:
    `1-theta > (1-delta)*(1+2/N_t)`. [Eq `e:soundness:min:table:height`, Eq `e:cFRI:batch:eval:condition:rho:plus`]
- IOPP round errors:
  - `eps_1` (logUp), `eps_2` (composition), `eps_3` (OOD query) scale with list size bound `ell(theta)` and `1/|F|` terms. [Thm `thm:IOPP:soundness`]
- FRI round errors:
  - Batching: scales with `(M-1)*ell_GS(theta)*(...) * |D_0|/|F|`.
  - Folding step `k`: scales with `3*ell_k(theta)*(...) * |S_k|/|F|`.
  - Query: `(1-theta)^s`. [Thm `thm:cFRI:multi:soundness`]
- NIROP security rule:
  - Choose parameters so `eps(T)/T <= 2^{-sigma}` with
    `eps(T) <= (T+R) max_i eps_i + 3(T^2+1)/2^lambda`; paper examples use `lambda=256`, target `sigma=100`, and `T << 2^128`. [Eq `e:NIROP:soundness:error`, Eq `e:nirop:security`, Sec `s:example:params`]
- Example configurations:
  - Proven and conjectured tables provide concrete `n`, `beta`, list bounds, grind bits, query counts, and proof sizes for medium/large/CASM ensembles. [Tab `tab:example:params:LDR`, Tab `tab:example:params:conjectured`]

## 7. Complexity
| Component | Prover | Verifier | Memory | Notes |
|---|---|---|---|---|
| Circle FFT/eval FFT | `~O(N log N)` field ops; same counts as circle paper asymptotically | uses same transform for checks/derivations | in-place algorithms given | twiddle precompute critical | [Sec `s:cFFT`, Sec `s:cFFT:pseudocode`] |
| Flat AIR core rounds | logUp helpers + composition quotient + OOD openings | random point checks + identity checks | table/domain dependent | designed for many tables/heights | [Prot `prot:STARK:IOPP`] |
| Multi-domain FRI | batching + `r+1` fold commitments + `s` sampled queries | `s` spot-check traces across all levels | oracle/Merkle commitments | cross-domain correlated agreement target | [Prot `prot:cFRI:multi`] |
| Batch eval proof | run FRI on doubled ensemble (with degree correction) | same verifier pattern as FRI | doubled oracle set | protects small-domain correctness | [Prot `prot:cFRI:batch:eval:degree:corrected`] |
| NIROP hashing | one cross-domain Merkle root/path system | path verification along same trace | tree over largest active depth | auth path size same as single-domain tree | [Sec `s:cross:domain:merkle`, Alg `alg:merkle`] |

## 8. Implementation-Critical Invariants
- Flat AIR semantic invariants:
  - Every row satisfies component constraints.
  - Channel multiset consistency must hold with exact multiplicities (including public IO terms). [Eq `e:flat:AIR:consistency`]
- logUp invariants:
  - Pole identities (`u`, `v`) and running-sum increment relation must simultaneously hold over each table domain. [Eq `e:constraint:uses`, Eq `e:constraint:yields`, Eq `e:constraint:sum:increment`]
  - Use multiplicities must remain `< char(F_q)` for soundness argument assumptions. [Thm `thm:IOPP:soundness`]
- Cross-domain invariants:
  - Liftings to largest domain define the security relation; per-domain agreement is not enough in LDR. [Def `def:IOPP:relation`, Eq `e:cFRI:multi:relation`]
- FRI invariants:
  - Same batching challenge across domains in cross-domain step.
  - Fold equations must be checked with exact twiddles and projected trace points. [Eq `e:cFRI:multi:batching:check`, Eq `e:cFRI:multi:folding:check`]
- OOD evaluation invariants:
  - Degree-corrected proof requires both `f` and `(f-v)/v_Q` low degree; omitting non-quotient changes relation/risks. [Eq `e:cFRI:batch:eval:extended:ensemble`, Eq `e:cFRI:batch:eval:relation:plus`]
- NIROP invariants:
  - Fiat-Shamir transcript determinism and domain-separated hashing are mandatory.
  - Grinding only scales soundness if challenge-leading-zero condition is enforced exactly. [Eq `e:FiatShamir:randomness`]

## 9. Proof Skeletons (Only Essential Logic)
- IOPP soundness:
  - Define relation by existence of cross-domain proximate polynomial ensemble.
  - Show each main round is a randomized reduction with explicit failure probability.
  - Chain with FRI round reductions to get total round-by-round knowledge soundness.
  - Extraction uses list decoding on committed words with correlated-agreement constraints. [Thm `thm:IOPP:soundness`, Sec `s:IOPP:soundness:rbr`]
- Multi-domain FRI soundness:
  - Batching step: reduce many words to one random combination while preserving cross-domain structure.
  - Folding steps: inductively propagate relation through line-domain cascade.
  - Query phase: Monte Carlo check gives `(1-theta)^s` terminal error.
  - Core technical enabler: constrained/set-specific/weighted CA theorems. [Thm `thm:cFRI:multi:soundness`, Thm `thm:CAT:with:sets`, Thm `thm:CAT:weighted`, Thm `thm:CAT:subcodes`]
- Degree-corrected batch evaluation soundness:
  - Run FRI on paired ensemble `(f, (f-v)/v_Q)`.
  - Agreement + domain-size condition implies polynomial identity in slightly larger space, forcing OOD value consistency.
  - Inherits FRI soundness bounds unchanged. [Thm `thm:cFRI:batch:eval:degree:corrected:soundness`]
- NIROP soundness:
  - Apply BCS theorem over round-by-round IOP soundness.
  - First term from weakest reduction round under adaptive querying.
  - Second term from random-oracle collision/binding term. [Thm `thm:NIROP:soundness:general`]

## 10. Failure Modes / Edge Cases
- If only per-domain correlated agreement is tracked (without lifted cross-domain agreement), soundness can blow up combinatorially (`ell(theta)^r`). [Sec `s:IOPP:soundness`]
- If minimal table-height condition fails, quotient/non-quotient degree transition can break proof logic on small domains. [Eq `e:soundness:min:table:height`, Rem `rem:min:table:height`]
- Without degree correction, relation weakens to larger polynomial space (`L^+`), increasing effective list bound and risk near Johnson edge. [Eq `e:cFRI:batch:eval:relation:plus`, Eq `e:rho:plus:list:size`]
- Current implementation uses non-squaring-consistent circle-domain sampling in FRI queries; paper states full LDR proof for this variant is not provided in this writeup. [Sec `s:circle:FRI` implementation subsection]
- Multiplicity overflow (`>= char`) can invalidate logUp uniqueness intuition by modulo cancellation. [Thm `thm:IOPP:soundness`, Rem `rem:IOPP:soundness`]
- Conjectural parameter regime depends on unproven assumptions; should be treated as non-provable in current formal model. [Conj `conj:list:decodability`, Conj `conj:prox:gaps`]

## 11. Ambiguities / Open Questions
- Conjectured regime:
  - Two explicit conjectures (list-decodability and line-decodability up to Elias radius over prime fields) are not proven in this document. [Conj `conj:list:decodability`, Conj `conj:prox:gaps`]
- Implementation/proof gap:
  - Current release details differ from some idealized protocol choices (Merkle layout/query sampling; quotient style), and list-decoding-regime proof coverage is flagged as incomplete for that variant. [Circle FRI implementation subsection, Circle NIROP implementation subsection]
- Future roadmap items with security/perf impact:
  - lifted FRI, memoized Merkle tree, flat sumcheck variant, logUp-GKR integration, quantum-ROM analysis. [Sec `s:optimizations`]

## 12. Source Anchor Map
- Field/curve preliminaries: [Sec `s:fields`], [Eq `e:complex:extension`], [Eq `e:quartic:extension`], [Sec `s:circle:curve`], [Eq `e:group:law`]
- Circle polynomial/cFFT basics: [Sec `s:cFFT`], [Eq `e:cFFT:basis`], [Eq `e:cFFT:space:decomposition`], [Eq `e:cFFT:represention`]
- Flat AIR formalism: [Def `def:flat:AIR`], [Def `def:flat:AIR:solution`], [Def `def:flatAIR:degree`], [Eq `e:flat:AIR:consistency`]
- Main IOPP: [Prot `prot:STARK:IOPP`], [Eq `e:constraint:uses`], [Eq `e:constraint:yields`], [Eq `e:constraint:sum:increment`], [Eq `e:crossdomain:quotient`]
- IOPP soundness relation: [Def `def:IOPP:relation`], [Thm `thm:IOPP:soundness`]
- Multi-domain FRI: [Prot `prot:cFRI:multi`], [Eq `e:cFRI:multi:batching`], [Eq `e:cFRI:multi:folding`], [Thm `thm:cFRI:multi:soundness`]
- Batch eval proof: [Prot `prot:cFRI:batch:eval:degree:corrected`], [Eq `e:cFRI:batch:eval:extended:ensemble`], [Thm `thm:cFRI:batch:eval:degree:corrected:soundness`]
- CA machinery: [Sec `s:CATs`], [Thm `thm:CAT:with:sets`], [Thm `thm:CAT:weighted`], [Thm `thm:CAT:subcodes`]
- NIROP/ROM: [Sec `s:cross:domain:merkle`], [Alg `alg:merkle`], [Eq `e:FiatShamir:randomness`], [Thm `thm:NIROP:soundness:general`]
- Concrete parameters: [Sec `s:example:params`], [Tab `tab:soundness:errors`], [Tab `tab:example:params:LDR`], [Tab `tab:example:params:conjectured`]
- Conjectural appendix: [Sec `s:conjectures`], [Conj `conj:list:decodability`], [Conj `conj:prox:gaps`], [Thm `thm:Krachun`]
