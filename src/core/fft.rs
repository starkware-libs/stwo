#[cfg(test)]
use crate::core::curve::CIRCLE_GEN;

use super::{
    curve::{CanonicCoset, CirclePoint},
    field::Field,
};

/// Evaluation on a canonic coset and its conjugate.
#[derive(Clone)]
pub struct Evaluation {
    pub coset: CanonicCoset,
    pub values: Vec<Field>,
}
impl Evaluation {
    pub fn new(coset: CanonicCoset, values: Vec<Field>) -> Self {
        // A canonic coset is one with only odds points.
        assert!(values.len() == coset.size());
        Self { coset, values }
    }
}

#[derive(Debug)]
pub struct Polynomial {
    pub coset: CanonicCoset,
    /// Coefficients of th polynomial basis.
    /// In regular polynomial representation, the i-th coefficient is the coefficient of
    ///   x^i = prod_{j bit is set in i} x^{2^i}
    ///   x^i = prod_{j bit is set in i} repeated squaring of x, j times.
    /// In our representation, instead of repeated squaring, we have repeated applicaiton of
    ///   2x^2-1.
    /// An exception is the msb, which has the `y` multiplier instead.
    coeffs: Vec<Field>,
}
impl Polynomial {
    pub fn eval(&self, point: CirclePoint) -> Field {
        let mut mults = vec![Field::one(), point.y];
        let mut cur = point.x;
        for _ in 0..self.coset.n_bits - 1 {
            mults.push(cur);
            cur = cur.square().double() - Field::one();
        }
        mults.reverse();
        let multi_inverse = mults.iter().map(|x| x.inverse()).collect::<Vec<_>>();

        let mut sum = Field::zero();
        let mut cur_mult = Field::one();
        for (i, val) in self.coeffs.iter().enumerate() {
            sum += *val * cur_mult;
            // Update cur_mult according to the flipped bits from i to i+1.
            let mut j = i;
            let mut bit_i = 0;
            while j & 1 == 1 {
                cur_mult *= multi_inverse[bit_i];
                j >>= 1;
                bit_i += 1;
            }
            cur_mult *= mults[bit_i];
        }
        sum
    }
    pub fn extend(&self, coset: CanonicCoset) -> Polynomial {
        assert!(coset.n_bits >= self.coset.n_bits);
        let mut coeffs = vec![Field::zero(); coset.size()];
        let jump_bits = coset.n_bits - self.coset.n_bits;
        for (i, val) in self.coeffs.iter().enumerate() {
            coeffs[i << jump_bits] = *val;
        }
        Polynomial { coset, coeffs }
    }
}

pub struct FFTree {
    coset: CanonicCoset,
    twiddle: Vec<Vec<Field>>,
    itwiddle: Vec<Vec<Field>>,
}
impl FFTree {
    pub fn preprocess(coset: CanonicCoset) -> FFTree {
        let mut twiddle = vec![];
        let mut itwiddle = vec![];
        // First twiddle layer.
        let mut layer = Vec::with_capacity(coset.size() / 2);
        let mut ilayer = Vec::with_capacity(coset.size() / 2);
        for point in coset.iter().take(coset.size() / 2) {
            layer.push(point.y);
            ilayer.push(point.y.inverse());
        }
        twiddle.push(layer);
        itwiddle.push(ilayer);

        // Next layers.
        let mut cur_coset = coset;
        while cur_coset.size() >= 4 {
            let mut layer = Vec::with_capacity(cur_coset.size() / 4);
            let mut ilayer = Vec::with_capacity(cur_coset.size() / 4);
            for point in cur_coset.iter().take(cur_coset.size() / 4) {
                layer.push(point.x);
                ilayer.push(point.x.inverse());
            }
            twiddle.push(layer);
            itwiddle.push(ilayer);
            cur_coset = cur_coset.double();
        }
        FFTree {
            coset,
            twiddle,
            itwiddle,
        }
    }
    pub fn ifft(&self, eval: Evaluation) -> Polynomial {
        assert!(eval.coset == self.coset);
        let mut data = eval.values;

        for layer in self.itwiddle.iter() {
            let len = layer.len() * 2;
            for chunk in data.chunks_mut(len) {
                chunk[len / 2..].reverse();
                for i in 0..(len / 2) {
                    let v0 = chunk[i];
                    let v1 = chunk[len / 2 + i];
                    chunk[i] = v0 + v1;
                    chunk[len / 2 + i] = (v0 - v1) * layer[i];
                }
            }
        }

        // Divide all values by 2^self.coset.n_bits.
        let n_bits = self.coset.n_bits;
        let inv = Field::from_u32_unchecked(1 << n_bits).inverse();
        for val in &mut data {
            *val *= inv;
        }

        Polynomial {
            coset: self.coset,
            coeffs: data,
        }
    }
    pub fn fft(&self, poly: Polynomial) -> Evaluation {
        assert!(poly.coset == self.coset);
        let mut data = poly.coeffs;

        // Bottom layers.
        for layer in self.twiddle.iter().rev() {
            let len = layer.len() * 2;
            for chunk in data.chunks_mut(len) {
                for i in 0..(len / 2) {
                    let v0 = chunk[i];
                    let v1 = chunk[len / 2 + i] * layer[i];
                    chunk[i] = v0 + v1;
                    chunk[len / 2 + i] = v0 - v1;
                }
                chunk[len / 2..].reverse();
            }
        }

        Evaluation::new(self.coset, data)
    }
}

#[cfg(test)]
fn get_trace() -> Evaluation {
    let trace_coset = CanonicCoset::new(10);
    let trace = (0..(trace_coset.size() as u32))
        .map(Field::from_u32_unchecked)
        .collect::<Vec<_>>();
    Evaluation::new(trace_coset, trace)
}

#[test]
fn test_ifft_eval() {
    let eval = get_trace();
    let fftree = FFTree::preprocess(eval.coset);
    let poly = fftree.ifft(eval.clone());
    for (point, val) in eval.coset.iter().zip(eval.values.iter()) {
        assert_eq!(poly.eval(point), *val);
    }
}

#[test]
fn test_ifft_fft() {
    let eval = get_trace();
    let fftree = FFTree::preprocess(eval.coset);
    let poly = fftree.ifft(eval.clone());
    let eval2 = fftree.fft(poly);
    for (val0, val1) in eval.values.iter().zip(eval2.values) {
        assert_eq!(*val0, val1);
    }
}

#[test]
fn test_extend() {
    let eval = get_trace();
    let fftree = FFTree::preprocess(eval.coset);
    let poly = fftree.ifft(eval.clone());
    let coset2 = CanonicCoset::new(poly.coset.n_bits + 2);
    let poly2 = poly.extend(coset2);

    assert_eq!(poly.eval(CIRCLE_GEN), poly2.eval(CIRCLE_GEN));
}
