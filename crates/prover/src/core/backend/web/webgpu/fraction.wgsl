// This shader contains implementations for fraction operations.
// It is stateless and can be used as a library in other shaders.

const ZERO_FRACTION: Fraction = Fraction(QM31(CM31(M31(0u), M31(0u)), CM31(M31(0u), M31(0u))), QM31(CM31(M31(1u), M31(0u)), CM31(M31(0u), M31(0u))));

struct Fraction {
    numerator: QM31,
    denominator: QM31,
}

// Add two fractions: (a/b + c/d) = (ad + bc)/(bd)
fn fraction_add(a: Fraction, b: Fraction) -> Fraction {
    let numerator = qm31_add(
        qm31_mul(a.numerator, b.denominator),
        qm31_mul(b.numerator, a.denominator)
    );
    let denominator = qm31_mul(a.denominator, b.denominator);
    return Fraction(numerator, denominator);
}

fn fraction_eq(a: Fraction, b: Fraction) -> bool {
    return a.numerator.a.a.data == b.numerator.a.a.data
        && a.numerator.a.b.data == b.numerator.a.b.data
        && a.numerator.b.a.data == b.numerator.b.a.data
        && a.numerator.b.b.data == b.numerator.b.b.data
        && a.denominator.a.a.data == b.denominator.a.a.data
        && a.denominator.a.b.data == b.denominator.a.b.data
        && a.denominator.b.a.data == b.denominator.b.a.data
        && a.denominator.b.b.data == b.denominator.b.b.data;
}
