use crate::core::{
    circle::{CircleIndex, CirclePoint, Coset},
    fft::FFTree,
    field::Field,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineDomain {
    pub coset_up: Coset,
}
impl LineDomain {
    pub fn new(coset_up: Coset) -> Self {
        assert!(
            (coset_up.initial_index * 2).0 % (coset_up.step_size.0) != 0,
            "Coset is not disjoint to its minus."
        );
        Self { coset_up }
    }
    pub fn canonic(n_bits: usize) -> Self {
        assert!(n_bits > 0);
        Self::new(Coset::new(CircleIndex::root(n_bits + 2), n_bits))
    }
    pub fn iter(&self) -> LineDomainIterator {
        LineDomainIterator {
            cur: self.coset_up.initial,
            step: self.coset_up.step,
            remaining: self.coset_up.len(),
        }
    }
    pub fn len(&self) -> usize {
        self.coset_up.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn n_bits(&self) -> usize {
        self.coset_up.n_bits
    }
    pub fn double(&self) -> Self {
        Self {
            coset_up: self.coset_up.double(),
        }
    }
    pub fn repeated_double(&self, n_times: usize) -> Self {
        if n_times == 0 {
            return *self;
        }
        self.double().repeated_double(n_times - 1)
    }
    pub fn initial_index(&self) -> CircleIndex {
        self.coset_up.initial_index
    }
    pub fn at(&self, index: usize) -> Field {
        self.coset_up.at(index).x
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

    pub fn subeval(&self, _blowup_bit: usize) -> Self {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct LinePoly {
    // This affects the isogenies, and thus, the monom structure.
    pub domain: LineDomain,
    pub coeffs: Vec<Field>,
}
impl LinePoly {
    pub fn new(domain: LineDomain, coeffs: Vec<Field>) -> Self {
        assert!(domain.len() == coeffs.len());
        Self { domain, coeffs }
    }
    pub fn eval_at_point(&self, mut point: Field) -> Field {
        let mut mults = vec![Field::one()];
        for _ in 0..self.domain.n_bits() {
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
    pub fn extend(&self, extended_domain: LineDomain) -> Self {
        let jump_bits = extended_domain.n_bits() - self.domain.n_bits();
        assert_eq!(extended_domain.repeated_double(jump_bits), self.domain);
        let mut coeffs = vec![Field::zero(); extended_domain.len()];
        for (i, val) in self.coeffs.iter().enumerate() {
            coeffs[i << jump_bits] = *val;
        }
        Self {
            domain: extended_domain,
            coeffs,
        }
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
    assert_eq!(xs[1], CircleIndex::root(5).to_point().mul(5).x);
}

#[test]
fn test_extend() {
    let domain = LineDomain::canonic(3);
    let poly = LinePoly::new(
        domain,
        (1..9).map(Field::from_u32_unchecked).collect::<Vec<_>>(),
    );
    let extended = poly.extend(domain);
    assert_eq!(
        poly.eval_at_point(Field::from_u32_unchecked(123)),
        extended.eval_at_point(Field::from_u32_unchecked(123))
    );
}

#[test]
fn test_interpolate() {
    let domain = LineDomain::canonic(3);
    let poly = LinePoly::new(
        domain,
        (1..9).map(Field::from_u32_unchecked).collect::<Vec<_>>(),
    );
    let tree = FFTree::preprocess(domain);
    let evaluation = poly.clone().evaluate(&tree);
    let interpolated = evaluation.interpolate(&tree);
    assert_eq!(interpolated.coeffs, poly.coeffs);
}
