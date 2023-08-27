use super::{circle::CirclePoint, field::Field, poly::circle::CircleDomain};

// Utils for computing constraints.

pub fn domain_poly_eval(domain: CircleDomain, mut p: CirclePoint) -> Field {
    p = p + domain.projection_shift.to_point();
    let mut x = p.x;
    for _ in 0..domain.n_bits() - 1 {
        x = x.square().double() - Field::one();
    }
    x
}

pub fn point_excluder(point: CirclePoint, excluded: CirclePoint) -> Field {
    (point - excluded).x - Field::one()
}

// pub fn subcoset_excluder(mut point: CirclePoint, coset: Coset) -> Field {
//     if coset.n_bits == 0 {
//         return point_excluder(point, coset.initial);
//     }
//     point = point - coset.initial;
//     for _ in 0..(coset.n_bits - 1) {
//         point = point.double();
//     }
//     point.y
// }
