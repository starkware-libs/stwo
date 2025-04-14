// This shader contains implementations for QM31/CM31/M31 operations.
// It is stateless, i.e. it does not contain any storage variables, and also it does not include
// any entrypoint functions, which means that it can be used as a library in other shaders.
// Note that the variable names that are used in this shader cannot be used in other shaders.
const P: u32 = 0x7FFFFFFF;  // 2^31 - 1
const MODULUS_BITS: u32 = 31u;
const HALF_BITS: u32 = 16u;

struct M31 {
    data: u32,
}

struct CM31 {
    a: M31,
    b: M31,
}

struct QM31 {
    a: CM31,
    b: CM31,
}

fn m31_add(a: M31, b: M31) -> M31 {
    return M31(partial_reduce(a.data + b.data));
}

fn m31_sub(a: M31, b: M31) -> M31 {
    return m31_add(a, m31_neg(b));
}

fn m31_mul(a: M31, b: M31) -> M31 {
    // Split into 16-bit parts
    let a1 = a.data >> HALF_BITS;
    let a0 = a.data & 0xFFFFu;
    let b1 = b.data >> HALF_BITS;
    let b0 = b.data & 0xFFFFu;

    // Compute partial products
    let m0 = partial_reduce(a0 * b0);
    let m1 = partial_reduce(a0 * b1);
    let m2 = partial_reduce(a1 * b0);
    let m3 = partial_reduce(a1 * b1);

    // Combine middle terms with reduction
    let mid = partial_reduce(m1 + m2);

    // Combine parts with partial reduction
    let shifted_mid = partial_reduce(mid << HALF_BITS);
    let low = partial_reduce(m0 + shifted_mid);

    let high_part = partial_reduce(m3 + (mid >> HALF_BITS));

    // Final combination using Mersenne prime property
    let result = partial_reduce(
        partial_reduce((high_part << 1u)) + 
        partial_reduce((low >> MODULUS_BITS)) + 
        partial_reduce(low & P)
    );
    return M31(result);
}

fn m31_neg(a: M31) -> M31 {
    return M31(partial_reduce(P - a.data));
}

fn m31_square(x: M31, n: u32) -> M31 {
    var result = x;
    for (var i = 0u; i < n; i += 1u) {
        result = m31_mul(result, result);
    }
    return result;
}

fn m31_pow5(x: M31) -> M31 {
    return m31_mul(m31_square(x, 2u), x);
}

fn m31_inverse(x: M31) -> M31 {
    // Computes x^(2^31-2) using the same sequence as pow2147483645
    // This is equivalent to x^(P-2) where P = 2^31-1
    
    // t0 = x^5
    let t0 = m31_mul(m31_square(x, 2u), x);
    
    // t1 = x^15
    let t1 = m31_mul(m31_square(t0, 1u), t0);
    
    // t2 = x^125
    let t2 = m31_mul(m31_square(t1, 3u), t0);
    
    // t3 = x^255
    let t3 = m31_mul(m31_square(t2, 1u), t0);
    
    // t4 = x^65535
    let t4 = m31_mul(m31_square(t3, 8u), t3);
    
    // t5 = x^16777215
    let t5 = m31_mul(m31_square(t4, 8u), t3);
    
    // result = x^2147483520
    var result = m31_square(t5, 7u);
    result = m31_mul(result, t2);
    
    return result;
}

// Complex field operations for CM31
fn cm31_add(a: CM31, b: CM31) -> CM31 {
    return CM31(
        m31_add(a.a, b.a),
        m31_add(a.b, b.b)
    );
}

fn cm31_sub(a: CM31, b: CM31) -> CM31 {
    return CM31(
        m31_sub(a.a, b.a),
        m31_sub(a.b, b.b)
    );
}

fn cm31_mul(a: CM31, b: CM31) -> CM31 {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let ac = m31_mul(a.a, b.a);
    let bd = m31_mul(a.b, b.b);
    let ad = m31_mul(a.a, b.b);
    let bc = m31_mul(a.b, b.a);

    return CM31(
        m31_sub(ac, bd),
        m31_add(ad, bc)
    );
}

fn cm31_neg(a: CM31) -> CM31 {
    return CM31(m31_neg(a.a), m31_neg(a.b));
}

fn cm31_square(x: CM31) -> CM31 {
    return cm31_mul(x, x);
}

fn cm31_inverse(x: CM31) -> CM31 {
    // 1/(a + bi) = (a - bi)/(a² + b²)
    let a_sq = m31_square(x.a, 1u);
    let b_sq = m31_square(x.b, 1u);
    let denom = m31_add(a_sq, b_sq);
    let denom_inv = m31_inverse(denom);

    // Multiply by conjugate and divide by norm
    return cm31_mul(
        CM31(x.a, m31_neg(x.b)),
        CM31(denom_inv, M31(0u))
    );
}

// Quadratic extension field operations for QM31
fn qm31_add(a: QM31, b: QM31) -> QM31 {
    return QM31(
        cm31_add(a.a, b.a),
        cm31_add(a.b, b.b)
    );
}

fn qm31_sub(a: QM31, b: QM31) -> QM31 {
    return QM31(
        cm31_sub(a.a, b.a),
        cm31_sub(a.b, b.b)
    );
}

fn qm31_mul(a: QM31, b: QM31) -> QM31 {
    // (a + bu)(c + du) = (ac + rbd) + (ad + bc)u
    // where r = 2 + i is the irreducible polynomial coefficient
    let ac = cm31_mul(a.a, b.a);
    let bd = cm31_mul(a.b, b.b);
    let ad = cm31_mul(a.a, b.b);
    let bc = cm31_mul(a.b, b.a);

    // r = 2 + i
    let r = CM31(M31(2u), M31(1u));
    let rbd = cm31_mul(r, bd);

    return QM31(
        cm31_add(ac, rbd),
        cm31_add(ad, bc)
    );
}

fn qm31_neg(a: QM31) -> QM31 {
    return QM31(cm31_neg(a.a), cm31_neg(a.b));
}

fn qm31_square(x: QM31) -> QM31 {
    return qm31_mul(x, x);
}

fn qm31_inverse(x: QM31) -> QM31 {
    // (a + bu)^-1 = (a - bu)/(a^2 - (2+i)b^2)
    let b2 = cm31_square(x.b);
    
    // Create 2+i
    let r = CM31(M31(2u), M31(1u));
    
    let rb2 = cm31_mul(r, b2);
    let a2 = cm31_square(x.a);
    let denom = cm31_sub(a2, rb2);
    let denom_inv = cm31_inverse(denom);
    
    // Compute (a - bu)
    let neg_b = cm31_neg(x.b);
    
    return QM31(
        cm31_mul(x.a, denom_inv),
        cm31_mul(neg_b, denom_inv)
    );
}

// Utility functions
fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
} 