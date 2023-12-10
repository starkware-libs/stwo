
# Example mixed degree merkle tree

## Trace

With `N=16`

```text
+=======+=======+=======+=======+=======+=======+=======+=======+
| col0  | col1  | col2  | col3  | col4  | col5  | col6  | col7  |
+=======+=======+=======+=======+=======+=======+=======+=======+
| c0r0  | c1r0  | c2r0  | c3r0  | c4r0  | c5r0  | c6r0  | c7r0  |
| c0r1  | c1r1  | c2r1  | c3r1  | c4r1  | c5r1  | c6r1  | c7r1  |
| c0r2  | c1r2  | c2r2  | c3r2  | c4r2  | c5r2  | c6r2  | c7r2  |
| c0r3  | c1r3  | c2r3  | c3r3  | c4r3  | c5r3  | c6r3  | c7r3  |
| c0r4  | c1r4  | c2r4  | c3r4  +-------+-------+-------+-------+
| c0r5  | c1r5  | c2r5  | c3r5  |
| c0r6  | c1r6  | c2r6  | c3r6  |
| c0r7  | c1r7  | c2r7  | c3r7  |
| c0r8  | c1r8  | c2r8  | c3r8  |
| c0r9  | c1r9  | c2r9  | c3r9  |
| c0r10 | c1r10 | c2r10 +-------+
| c0r11 | c1r11 | c2r11 |
| c0r12 | c1r12 | c2r12 |
| c0r13 | c1r13 | c2r13 |
| c0r14 | c1r14 | c2r14 |
| c0r15 | c1r15 | c2r15 |
| c0r16 | c1r16 | c2r16 |
| c0r17 | c1r17 | c2r17 |
| c0r18 | c1r18 | c2r18 |
| c0r19 | c1r19 | c2r19 |
+------+------+------+
```

## Tree

```text
                                                                                 root_hash                                                         
                                     ____________________________________________/       \________________________
                                    /                                                                             \
                                   /                                                                               \                             
                                  /                                                                                 \                            
                                ...                                                  ...                            ...
                   __________l2hash_0______________________                          ...                       _____l2hash_N/4-1______________________________             <--| layer2 hashes + 4 columns of size N/4:
                  /             \                          \                         ...                      /            \                                  \               | * l2row_hash_i = hash(c4ri || c5ri || c6ri || c7ri)
                 /          l2row_hash_0                    \                        ...                     /           l2row_hash_N/4-1                      \              | * l2hash_i = hash(l1hash_i*2 || l1hash_i*2+1 || l2row_hash_i)
                |                |                           |                       ...                    |                  |                                |
                |      [c4r0, c5r0, c6r0, c7r0]              |                       ...                    |  [c4rN/4-1, c5rN/4-1, c6rN/4-1, c7rN/4-1]         |
                |                                            |                       ...                    |                                                   |
           ____l1hash_0___                           ____l1hash_1____                ...             ____l1hash_N/2-2___                                  ___l1hash_N/2-1___            <--| layer1 hashes + 1 col of size N/2:
          /        |      \                         /        |       \               ...            /          |        \                                /           |      \              | * l1hash_i = hash(l1row_hash_i*2 || l1row_hash_i*2+1 || c3ri)
         /        c3r0     \                       /        c3r1      \              ...           /        c3rN/2-2     \                              /        c3rN/2-1    \             
    l0row_hash_0        l0row_hash_1           l0row_hash_2       l0row_hash_3       ...    l0row_hash_N-4             l0row_hash_N-3             l0row_hash_N-2           l0row_hash_N-1           <--| layer0 hashes 
        |                   |                      |                  |              ...          |                       |                            |                      |                        | * leaves (storing 3 col of size N)
[c0r0, c1r0, c2r0]   [c0r1, c1r1, c2r1]    [c0r2, c1r2, c2r2]   [c0r3, c1r3, c2r3]   ...   [c0rN-4, c1rN-4, c2rN-4]   [c0rN-3, c1rN-3, c2rN-3]   [c0rN-2, c1rN-2, c2rN-2]   [c0rN-1, c1rN-1, c2rN-1]   | * l0row_hash_i = hash(c0ri || c1ri || c2ri)
```

## Decommitment: Merkle path to c0r2

```text
[
    [c0r2, c1r2, c2r2],
    [l0row_hash_3, c3r1],
    [l1hash_0, l2row_hash_0],
    [l2hash_N/4-1],
    ...
]
```




+==========+==========+
| M31col0  | Q31col1  |
+==========+==========+
| c0r0     | c1r0     |
| c0r1     | c1r1     |
| c0r2     | c1r2     |
| c0r3     | c1r3     |
| c0r4     +----------+
| c0r5     |
| c0r6     |
| c0r7     |
| c0r8     |
| c0r9     |
| c0r10    |
| c0r11    |
| c0r12    |
| c0r13    |
| c0r14    |
| c0r15    |
+----------+

+==========+==========+
| M31col0  | Q31col1  |
+==========+==========+
| c0r0     | c1r0     |
| c0r1     | c1r1     |
| c0r2     | c1r2     |
| c0r3     | c1r3     |
| c0r4     +----------+
| c0r5     |
| c0r6     |
| c0r7     |
| c0r8     |
| c0r9     |
| c0r10    |
| c0r11    |
| c0r12    |
| c0r13    |
| c0r14    |
| c0r15    |
+----------+

+==========+==========+
| M31col0  | Q31col1  |
+==========+==========+
| c0r0     | c1r0     |
| c0r8     | c1r2     |
| c0r1     | c1r1     |
| c0r9     | c1r3     |
| c0r2     +----------+
| c0r10    |
| c0r3     |
| c0r11    |
| c0r4     |
| c0r12    |
| c0r5     |
| c0r13    |
| c0r6     |
| c0r14    |
| c0r7     |
| c0r15    |
+==========+

query_idx=10 - decommit to M31col0[10]
-> query_idx=10%8=2 - decommit to fold(M31col0)[2]
-> query_idx=10%4=2 - decommit to fold(fold(M31col0))[2] and Q31col1[2]
-> query_idx=10%2=0 - decommit to fold(fold(fold(M31col0)))[0] and fold(Q31col1)[0]
