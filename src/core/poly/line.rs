use crate::core::{
    circle::{CircleIndex, CirclePoint, Coset},
    fft::FFTree,
    field::m31::Field,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineDomain {
    pub n_bits: usize,
}
impl LineDomain {
    pub fn canonic(n_bits: usize) -> Self {
        assert!(n_bits > 0);
        Self { n_bits }
    }
    pub fn iter(&self) -> LineDomainIterator {
        LineDomainIterator {
            cur: self.initial_index().to_point(),
            step: CircleIndex::root(self.n_bits + 1).to_point(),
            remaining: 1 << self.n_bits,
        }
    }
    pub fn len(&self) -> usize {
        1 << self.n_bits
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }
    pub fn double(&self) -> Self {
        Self {
            n_bits: self.n_bits - 1,
        }
    }
    pub fn initial_index(&self) -> CircleIndex {
        CircleIndex::root(self.n_bits + 2)
    }
    pub fn associated_coset(&self) -> Coset {
        Coset::new(self.initial_index(), self.n_bits + 1)
    }
}
pub struct LineDomainIterator {
    cur: CirclePoint,
    step: CirclePoint,
    remaining: usize,
}
impl Iterator for LineDomainIterator {
    type Item = Field;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let res = self.cur;
        self.cur = self.cur + self.step;
        Some(res.x)
    }
}

#[derive(Clone, Debug)]
pub struct LineEvaluation {
    pub domain: LineDomain,
    pub values: Vec<Field>,
}
impl LineEvaluation {
    pub fn new(domain: LineDomain, values: Vec<Field>) -> Self {
        assert_eq!(domain.len(), values.len());
        Self { domain, values }
    }
    pub fn interpolate(self, tree: &FFTree) -> LinePoly {
        tree.ifft(self)
    }
}

#[derive(Clone, Debug)]
pub struct LinePoly {
    pub bound_bits: usize,
    pub coeffs: Vec<Field>,
}
impl LinePoly {
    pub fn new(bound_bits: usize, coeffs: Vec<Field>) -> Self {
        assert!(coeffs.len() == (1 << bound_bits));
        Self { bound_bits, coeffs }
    }
    pub fn eval_at_point(&self, mut point: Field) -> Field {
        let mut mults = vec![Field::one()];
        for _ in 0..self.bound_bits {
            mults.push(point);
            point = point.square().double() - Field::one();
        }
        mults.reverse();

        let mut sum = Field::zero();
        for (i, val) in self.coeffs.iter().enumerate() {
            let mut cur_mult = Field::one();
            for (j, mult) in mults.iter().enumerate() {
                if i & (1 << j) != 0 {
                    cur_mult *= *mult;
                }
            }
            sum += *val * cur_mult;
        }
        sum
    }
    pub fn extend(&self, line_domain: LineDomain) -> Self {
        let bound_bits = line_domain.n_bits();
        assert!(bound_bits >= self.bound_bits);
        let mut coeffs = vec![Field::zero(); 1 << bound_bits];
        let jump_bits = bound_bits - self.bound_bits;
        for (i, val) in self.coeffs.iter().enumerate() {
            coeffs[i << jump_bits] = *val;
        }
        Self { bound_bits, coeffs }
    }
    pub fn evaluate(self, tree: &FFTree) -> LineEvaluation {
        tree.fft(self)
    }
}

#[test]
fn test_canonic_domain() {
    let domain = LineDomain::canonic(3);
    assert_eq!(domain.len(), 8);
    assert_eq!(domain.n_bits(), 3);
    let xs = domain.iter().collect::<Vec<_>>();
    assert_eq!(xs[0], CircleIndex::root(5).to_point().x);
    assert_eq!(xs[1], CircleIndex::root(5).to_point().mul(3).x);
}

#[test]
fn test_associated_coset() {
    let domain = LineDomain::canonic(3);
    let coset = domain.associated_coset();
    assert_eq!(coset.n_bits, 4);
    assert_eq!(coset.initial_index, CircleIndex::root(5));
}

#[test]
fn test_extend() {
    let domain = LineDomain::canonic(3);
    let poly = LinePoly::new(3, (1..9).map(Field::from_u32_unchecked).collect::<Vec<_>>());
    let extended = poly.extend(domain);
    assert_eq!(
        poly.eval_at_point(Field::from_u32_unchecked(123)),
        extended.eval_at_point(Field::from_u32_unchecked(123))
    );
}

#[test]
fn test_interpolate() {
    let domain = LineDomain::canonic(3);
    let poly = LinePoly::new(3, (1..9).map(Field::from_u32_unchecked).collect::<Vec<_>>());
    let tree = FFTree::preprocess(domain);
    let evaluation = poly.clone().evaluate(&tree);
    let interpolated = evaluation.interpolate(&tree);
    assert_eq!(interpolated.coeffs, poly.coeffs);
}
