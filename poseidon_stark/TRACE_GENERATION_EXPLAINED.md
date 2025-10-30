# Poseidon Trace Generation - Gdzie Co Jest

## Funkcja `gen_trace()` - linia 200-284

### 1. **INITIAL STATE** - Linie 222-234

```rust
// LINIA 222-226: Tworzenie INITIAL STATE
let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
    PackedBaseField::from_array(std::array::from_fn(|i| {
        BaseField::from_u32_unchecked((vec_index * 16 + i + state_i + rep_i) as u32)
    }))
});

// LINIA 227-230: ⭐ ZAPISANIE INITIAL STATE DO TRACE (kolumny 0-15)
state.iter().copied().for_each(|s| {
    trace[col_index].data[vec_index] = s;  // <-- TU zapisujemy do trace!
    col_index += 1;  // col_index idzie 0→1→2...→15
});

// LINIA 231-234: Kopia initial state do lookup_data (dla LogUp)
lookup_data.initial_state[rep_i]
    .iter_mut()
    .zip(state)
    .for_each(|(res, state_i)| res.data[vec_index] = state_i);
```

**Rezultat**: Kolumny 0-15 trace zawierają initial state

---

### 2. **PIERWSZE 4 FULL ROUNDS** - Linie 236-247

```rust
// LINIA 237-247: Loop przez 4 full rounds (round 0,1,2,3)
(0..N_HALF_FULL_ROUNDS).for_each(|round| {
    // LINIA 238-240: Dodaj stałe rundy
    (0..N_STATE).for_each(|i| {
        state[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
    });

    // LINIA 241: Zastosuj MDS matrix (miksowanie)
    apply_external_round_matrix(&mut state);

    // LINIA 242: S-box (x^5 dla wszystkich 16 elementów)
    state = std::array::from_fn(|i| pow5(state[i]));

    // LINIA 243-246: ⭐ ZAPISANIE STANU PO RUNDZIE DO TRACE
    state.iter().copied().for_each(|s| {
        trace[col_index].data[vec_index] = s;  // <-- TU zapisujemy!
        col_index += 1;  // col_index: 16→32→48→64→80
    });
});
```

**Rezultat**:
- Kolumny 16-31: Po Full Round 1
- Kolumny 32-47: Po Full Round 2
- Kolumny 48-63: Po Full Round 3
- Kolumny 64-79: Po Full Round 4

---

### 3. **14 PARTIAL ROUNDS** - Linie 249-256

```rust
// LINIA 250-256: Loop przez 14 partial rounds
(0..N_PARTIAL_ROUNDS).for_each(|round| {
    // LINIA 251: Dodaj stałą (tylko do pierwszego elementu!)
    state[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);

    // LINIA 252: MDS matrix (ale dla wszystkich elementów)
    apply_internal_round_matrix(&mut state);

    // LINIA 253: S-box TYLKO dla pierwszego elementu (oszczędność!)
    state[0] = pow5(state[0]);

    // LINIA 254-255: ⭐ ZAPISANIE TYLKO state[0] DO TRACE
    trace[col_index].data[vec_index] = state[0];  // <-- TU! Tylko jeden element
    col_index += 1;  // col_index: 80→81→82...→93
});
```

**Rezultat**:
- Kolumny 80-93: Partial rounds (tylko pierwszy element każdej rundy!)

---

### 4. **OSTATNIE 4 FULL ROUNDS** - Linie 258-271

```rust
// LINIA 259-271: Loop przez kolejne 4 full rounds (round 4,5,6,7)
(0..N_HALF_FULL_ROUNDS).for_each(|round| {
    // LINIA 260-264: Dodaj stałe (od round 4)
    (0..N_STATE).for_each(|i| {
        state[i] += PackedBaseField::broadcast(
            EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i],
        );
    });

    // LINIA 265: MDS matrix
    apply_external_round_matrix(&mut state);

    // LINIA 266: S-box dla wszystkich
    state = std::array::from_fn(|i| pow5(state[i]));

    // LINIA 267-270: ⭐ ZAPISANIE STANU PO RUNDZIE DO TRACE
    state.iter().copied().for_each(|s| {
        trace[col_index].data[vec_index] = s;  // <-- TU zapisujemy!
        col_index += 1;  // col_index: 94→110→126→142→158
    });
});
```

**Rezultat**:
- Kolumny 94-109:  Po Full Round 5
- Kolumny 110-125: Po Full Round 6
- Kolumny 126-141: Po Full Round 7
- Kolumny 142-157: Po Full Round 8 (**FINAL STATE!**)

---

### 5. **FINAL STATE** - Linie 273-276

```rust
// LINIA 273-276: Zapisanie final state do lookup_data (dla LogUp)
lookup_data.final_state[rep_i]
    .iter_mut()
    .zip(state)
    .for_each(|(res, state_i)| res.data[vec_index] = state_i);
```

**Rezultat**: Final state (16 elementów) zapisany do lookup_data

---

## Podsumowanie mapowania kolumn

```
col_index   Kolumny      Co zawiera
---------   --------     -----------
0-15        0-15         Initial state (16 elementów)
16-31       16-31        Po Full Round 1 (16 elementów)
32-47       32-47        Po Full Round 2 (16 elementów)
48-63       48-63        Po Full Round 3 (16 elementów)
64-79       64-79        Po Full Round 4 (16 elementów)
80          80           Po Partial Round 1 (1 element - state[0])
81          81           Po Partial Round 2 (1 element)
...         ...          ...
93          93           Po Partial Round 14 (1 element)
94-109      94-109       Po Full Round 5 (16 elementów)
110-125     110-125      Po Full Round 6 (16 elementów)
126-141     126-141      Po Full Round 7 (16 elementów)
142-157     142-157      Po Full Round 8 = FINAL STATE (16 elementów)
```

**Total: 158 kolumn na jeden hash instance**

---

## Kluczowa zmienna: `col_index`

```rust
let mut col_index = 0;  // Start na 0

// Po initial state:
col_index = 16

// Po każdej full round:
col_index += 16  // (bo zapisujemy 16 elementów)

// Po każdej partial round:
col_index += 1   // (bo zapisujemy 1 element)

// Na końcu:
col_index = 16 + 4*16 + 14 + 4*16 = 158
```

---

## Flow danych w pamięci

```
state (zmienna lokalna, 16 elementów)
  ↓ transformacja
  ↓ zapisz do: trace[col_index].data[vec_index]
  ↓ inkrementuj col_index
  ↓ następna transformacja
  ↓ zapisz do: trace[col_index].data[vec_index]
  ↓ ...
FINAL STATE
```

Każda linia w trace dump to `trace[col].data[row]` gdzie:
- `col` = numer kolumny (0-1263 dla 8 instancji)
- `row` = numer wiersza (0-15 dla 16 wierszy)
