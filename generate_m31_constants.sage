#!/usr/bin/env sage
# Script para generar constantes de Poseidon2 para el campo M31 con t=16
# Basado en: https://github.com/HorizenLabs/poseidon2

from sage.rings.polynomial.polynomial_gf2x import GF2X_BuildIrred_list
from math import *
import itertools

# Parámetros para M31
p = 2147483647  # M31 = 2^31 - 1
t = 16          # State size (matching stwo's implementation)
n = len(p.bits())  # bit length

FIELD = 1
SBOX = 0
FIELD_SIZE = n
NUM_CELLS = t

print(f"Generating Poseidon2 constants for M31")
print(f"Prime p = {p} (2^31 - 1)")
print(f"State size t = {t}")
print(f"Field size n = {n} bits")
print()

def get_alpha(p):
    """Find the smallest alpha > 2 coprime with p-1"""
    for alpha in range(3, p):
        if gcd(alpha, p-1) == 1:
            break
    return alpha

alpha = get_alpha(p)
print(f"Alpha (S-box degree) = {alpha}")
print()

# Security level (bits)
M = 128

def sat_inequiv_alpha(p, t, R_F, R_P, alpha, M):
    """Check if round numbers satisfy security inequalities"""
    N = int(FIELD_SIZE * NUM_CELLS)
    
    if alpha > 0:
        R_F_1 = 6 if M <= ((floor(log(p, 2) - ((alpha-1)/2.0))) * (t + 1)) else 10
        R_F_2 = 1 + ceil(log(2, alpha) * min(M, FIELD_SIZE)) + ceil(log(t, alpha)) - R_P
        R_F_3 = (log(2, alpha) * min(M, log(p, 2))) - R_P
        R_F_4 = t - 1 + log(2, alpha) * min(M / float(t + 1), log(p, 2) / float(2)) - R_P
        R_F_5 = (t - 2 + (M / float(2 * log(alpha, 2))) - R_P) / float(t - 1)
        R_F_max = max(ceil(R_F_1), ceil(R_F_2), ceil(R_F_3), ceil(R_F_4), ceil(R_F_5))
        
        # Addition due to https://eprint.iacr.org/2023/537.pdf
        r_temp = floor(t / 3.0)
        over = (R_F - 1) * t + R_P + r_temp + r_temp * (R_F / 2.0) + R_P + alpha
        under = r_temp * (R_F / 2.0) + R_P + alpha
        binom_log = log(binomial(over, under), 2)
        if binom_log == inf:
            binom_log = 10000000
        
        if binom_log < M:
            return False
        if R_F < R_F_max:
            return False
    return True

def find_round_numbers(p, t, alpha, M, security_margin=True):
    """Find optimal round numbers R_F and R_P"""
    n = ceil(log(p, 2))
    N = int(n * t)
    
    R_P = 0
    R_F = 0
    min_cost = float("inf")
    
    # Brute-force search
    for R_P_t in range(1, 100):
        for R_F_t in range(4, 50):
            if R_F_t % 2 == 0:
                if sat_inequiv_alpha(p, t, R_F_t, R_P_t, alpha, M):
                    if security_margin:
                        R_F_t += 2
                        R_P_t = int(ceil(float(R_P_t) * 1.075))
                    cost = t * R_F_t + R_P_t
                    if cost < min_cost:
                        R_P = int(R_P_t)
                        R_F = int(R_F_t)
                        min_cost = cost
    return (R_F, R_P)

# Calculate round numbers
R_F, R_P = find_round_numbers(p, t, alpha, M)
print(f"Round numbers:")
print(f"  R_F (full rounds) = {R_F}")
print(f"  R_P (partial rounds) = {R_P}")
print(f"  R_f (half full rounds) = {R_F // 2}")
print(f"  Total S-boxes = {t * R_F + R_P}")
print()

# Generate round constants using grain LFSR
def grain_sr_generator():
    """Grain LFSR for generating pseudo-random field elements"""
    bit_sequence = []
    state = [1] + [0] * 79  # Initial state
    
    for _ in range(160):
        new_bit = (state[62] + state[51] + state[38] + state[23] + state[13] + state[0]) % 2
        bit_sequence.append(state[0])
        state = [new_bit] + state[:-1]
    
    field_bits = ceil(log(p, 2))
    while True:
        # Generate field_bits bits
        bits = []
        for _ in range(field_bits):
            new_bit = (state[62] + state[51] + state[38] + state[23] + state[13] + state[0]) % 2
            bits.append(state[0])
            state = [new_bit] + state[:-1]
        
        # Convert to integer
        val = sum(b * (2**i) for i, b in enumerate(bits))
        if val < p:
            yield val

def generate_round_constants(p, t, R_F, R_P):
    """Generate round constants for Poseidon2"""
    gen = grain_sr_generator()
    F = GF(p)
    
    # External round constants (R_F rounds, t elements each)
    external_constants = []
    for _ in range(R_F):
        round_consts = [F(next(gen)) for _ in range(t)]
        external_constants.append(round_consts)
    
    # Internal round constants (R_P rounds, 1 element each)
    internal_constants = [F(next(gen)) for _ in range(R_P)]
    
    return external_constants, internal_constants

print("Generating round constants...")
external_consts, internal_consts = generate_round_constants(p, t, R_F, R_P)
print(f"Generated {len(external_consts)} external rounds with {t} constants each")
print(f"Generated {len(internal_consts)} internal round constants")
print()

# Generate internal matrix (with minpoly fix)
print("Generating internal matrix (with minpoly fix)...")
F = GF(p)

# Diagonal elements (mu_i)
# With fix: mu_0 = 4, mu_i = 2^{i+1} + 1 for i > 0
diagonal = [F(4)] + [F((1 << (i + 1)) + 1) for i in range(1, t)]

print("Internal matrix diagonal:")
print(f"  {[int(d) for d in diagonal]}")
print()

# Output as Rust code
print("=" * 80)
print("RUST CODE FOR CONSTANTS:")
print("=" * 80)
print()

# External constants
print("// External round constants")
print(f"const N_EXTERNAL_ROUNDS: usize = {R_F};")
print(f"const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; N_EXTERNAL_ROUNDS] = [")
for i, round_consts in enumerate(external_consts):
    consts_str = ", ".join(f"BaseField::from_u32_unchecked({int(c)})" for c in round_consts)
    print(f"    [{consts_str}],")
print("];")
print()

# Internal constants  
print("// Internal round constants")
print(f"const N_PARTIAL_ROUNDS: usize = {R_P};")
print("const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] = [")
for i, c in enumerate(internal_consts):
    print(f"    BaseField::from_u32_unchecked({int(c)}),")
print("];")
print()

# Diagonal
print("// Internal matrix diagonal (for apply_internal_round_matrix)")
print(f"const INTERNAL_MATRIX_DIAGONAL: [u32; N_STATE] = {[int(d) for d in diagonal]};")
print()

print("=" * 80)
print("Generation complete!")
print()
print(f"Copy the constants above into:")
print(f"  proof-of-burn-stwo/prover/src/utils/poseidon2_stwo.rs")

