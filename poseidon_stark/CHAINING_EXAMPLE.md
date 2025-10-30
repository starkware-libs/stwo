# Poseidon Chaining - Jak hashować długie wiadomości

## Problem z obecną implementacją

Obecny kod (`circuit/src/lib.rs` linia 235):
```rust
let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
    PackedBaseField::from_array(std::array::from_fn(|i| {
        BaseField::from_u32_unchecked((vec_index * 16 + i + state_i + rep_i) as u32)
        //                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //                             Input z WZORU, nie z poprzedniego outputu!
    }))
});
```

**Rezultat**: 8 niezależnych hashy, brak ciągłości.

---

## Rozwiązanie 1: Sequential Chaining (prosty chain)

### Jak to działa:
```
Hash 0: input [data_0..15]        → output_0 [16 elementów]
Hash 1: input [output_0[0..7], data_16..23] → output_1
Hash 2: input [output_1[0..7], data_24..31] → output_2
...
```

### Kod:

```rust
pub fn gen_trace_with_chaining(
    log_size: u32,
    initial_data: &[BaseField],  // Długa wiadomość do zhashowania
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    let mut lookup_data = LookupData { /* ... */ };

    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        let mut col_index = 0;

        // Zmienna do przechowywania poprzedniego outputu
        let mut previous_output: Option<[PackedBaseField; N_STATE]> = None;

        for rep_i in 0..N_INSTANCES_PER_ROW {
            // ========================================
            // CHAINING: Użyj poprzedniego outputu!
            // ========================================
            let mut state: [_; N_STATE] = if let Some(prev) = previous_output {
                // Użyj poprzedniego outputu jako części inputu
                std::array::from_fn(|state_i| {
                    if state_i < 8 {
                        // Pierwsze 8 elementów z poprzedniego outputu
                        prev[state_i]
                    } else {
                        // Pozostałe 8 z nowych danych
                        PackedBaseField::from_array(std::array::from_fn(|i| {
                            let data_idx = vec_index * 128 + rep_i * 16 + state_i;
                            if data_idx < initial_data.len() {
                                initial_data[data_idx]
                            } else {
                                BaseField::from_u32_unchecked(0)
                            }
                        }))
                    }
                })
            } else {
                // Pierwszy hash - użyj danych od początku
                std::array::from_fn(|state_i| {
                    PackedBaseField::from_array(std::array::from_fn(|i| {
                        let data_idx = vec_index * 128 + rep_i * 16 + state_i;
                        if data_idx < initial_data.len() {
                            initial_data[data_idx]
                        } else {
                            BaseField::from_u32_unchecked(0)
                        }
                    }))
                })
            };

            // Zapisz initial state do trace
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s;
                col_index += 1;
            });

            // ... (reszta obliczen Poseidon jak zwykle) ...

            // 4 full rounds
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
                });
                apply_external_round_matrix(&mut state);
                state = std::array::from_fn(|i| pow5(state[i]));
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
            });

            // Partial rounds
            (0..N_PARTIAL_ROUNDS).for_each(|round| {
                state[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);
                apply_internal_round_matrix(&mut state);
                state[0] = pow5(state[0]);
                trace[col_index].data[vec_index] = state[0];
                col_index += 1;
            });

            // 4 full rounds
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(
                        EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i],
                    );
                });
                apply_external_round_matrix(&mut state);
                state = std::array::from_fn(|i| pow5(state[i]));
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
            });

            // ⭐ Zapisz output jako poprzedni dla następnej iteracji
            previous_output = Some(state);

            lookup_data.final_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);
        }
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    (trace, lookup_data)
}
```

**Efekt**:
```
Hash 0: [data[0..15]]                    → output_0
Hash 1: [output_0[0..7], data[16..23]]   → output_1
Hash 2: [output_1[0..7], data[24..31]]   → output_2
...
Hash 7: [output_6[0..7], data[96..103]]  → output_7 = FINAL HASH
```

---

## Rozwiązanie 2: Poseidon Sponge (jak w Starknet!)

### Koncepcja:
```
State = [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]
        └─────rate──────┘ └────capacity───┘
         (8 elementów)     (8 elementów)

Absorb phase:
1. XOR chunk z rate częścią: state[0..8] ^= data[0..8]
2. Permutacja Poseidon: state = poseidon(state)
3. Powtórz dla każdego chunka

Squeeze phase:
Output = state[0..output_length]
```

### Kod conceptual:

```rust
struct PoseidonSponge {
    state: [BaseField; 16],
    rate: usize,  // 8 elementów
    capacity: usize,  // 8 elementów
}

impl PoseidonSponge {
    fn new() -> Self {
        Self {
            state: [BaseField::zero(); 16],
            rate: 8,
            capacity: 8,
        }
    }

    // Absorb (wczytaj dane)
    fn absorb(&mut self, data: &[BaseField]) {
        for chunk in data.chunks(self.rate) {
            // XOR chunk z rate częścią stanu
            for (i, &val) in chunk.iter().enumerate() {
                self.state[i] += val;  // W GF to jest XOR
            }

            // Permutacja Poseidon
            self.permute();
        }
    }

    // Squeeze (wyciągnij hash)
    fn squeeze(&mut self, output_len: usize) -> Vec<BaseField> {
        let mut output = Vec::new();

        while output.len() < output_len {
            // Weź rate elementów jako output
            output.extend_from_slice(&self.state[0..self.rate]);

            if output.len() < output_len {
                // Jeśli potrzeba więcej, permutuj znowu
                self.permute();
            }
        }

        output.truncate(output_len);
        output
    }

    fn permute(&mut self) {
        // To samo co w gen_trace - 22 rundy

        // 4 full rounds
        for round in 0..4 {
            for i in 0..16 {
                self.state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            }
            apply_external_round_matrix(&mut self.state);
            self.state = std::array::from_fn(|i| pow5(self.state[i]));
        }

        // 14 partial rounds
        for round in 0..14 {
            self.state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut self.state);
            self.state[0] = pow5(self.state[0]);
        }

        // 4 full rounds
        for round in 0..4 {
            for i in 0..16 {
                self.state[i] += EXTERNAL_ROUND_CONSTS[round + 4][i];
            }
            apply_external_round_matrix(&mut self.state);
            self.state = std::array::from_fn(|i| pow5(self.state[i]));
        }
    }
}

// Użycie:
fn hash_long_message(data: &[BaseField]) -> BaseField {
    let mut sponge = PoseidonSponge::new();
    sponge.absorb(data);  // Wczytaj całą wiadomość (dowolna długość!)
    let output = sponge.squeeze(1);  // Wyciągnij 1 element jako hash
    output[0]
}
```

---

## Porównanie:

| Metoda | Obecna impl | Sequential Chain | Sponge |
|--------|-------------|------------------|--------|
| Ciągłość | ✗ Brak | ✓ Tak | ✓ Tak |
| Dowolna długość | ✗ Nie | ~ Ograniczona | ✓ Tak |
| Bezpieczeństwo | - Demo only | ✓ Dobre | ✓✓ Najlepsze |
| Używane w praktyce | ✗ | ~ Rzadko | ✓ Starknet |

---

## Jak to działa w Starknet:

W Starknet używają **Poseidon Sponge** do hashowania:
- Długich wiadomości
- Merkle trees
- State commitments

Przykład: https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/crypto/signature/fast_pedersen_hash.py

```python
def hash_elements(elements):
    state = [0] * 16
    rate = 8

    for chunk in chunks(elements, rate):
        # Absorb
        for i, val in enumerate(chunk):
            state[i] += val

        # Permute
        state = poseidon_permutation(state)

    # Squeeze
    return state[0]
```

---

## Podsumowanie:

**Obecny kod**: Demonstracyjny, niezależne hashe
**Chain**: Możliwe do zrobienia, output → input
**Sponge**: Produkcyjny sposób, używany w Starknet

Chcesz żebym zrobił pełną implementację z chainingiem? 🎯
