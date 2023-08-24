use super::{
    field::Field,
    poly::line::{LineDomain, LineEvaluation, LinePoly},
};

pub struct FFTree {
    domain: LineDomain,
    twiddle: Vec<Vec<Field>>,
    itwiddle: Vec<Vec<Field>>,
}
impl FFTree {
    pub fn preprocess(domain: LineDomain) -> FFTree {
        let mut twiddle = vec![];
        let mut itwiddle = vec![];

        let mut cur_domain = domain;
        while cur_domain.len() > 1 {
            print!("{} ", cur_domain.len());
            let half_len = cur_domain.len() / 2;
            let mut layer = Vec::with_capacity(half_len);
            let mut ilayer = Vec::with_capacity(half_len);
            for x in cur_domain.iter().take(half_len) {
                layer.push(x);
                ilayer.push(x.inverse());
            }
            twiddle.push(layer);
            itwiddle.push(ilayer);
            cur_domain = cur_domain.double();
        }
        FFTree {
            domain,
            twiddle,
            itwiddle,
        }
    }
    pub fn ifft(&self, eval: LineEvaluation) -> LinePoly {
        assert!(eval.domain == self.domain); // TODO: How can we split the tree?
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

        // Divide all values by 2^n_bits.
        let inv = Field::from_u32_unchecked(self.domain.len() as u32).inverse();
        for val in &mut data {
            *val *= inv;
        }

        LinePoly::new(self.domain.n_bits(), data)
    }
    pub fn fft(&self, poly: LinePoly) -> LineEvaluation {
        assert!(poly.bound_bits == self.domain.n_bits());
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

        LineEvaluation::new(self.domain, data)
    }
}

#[cfg(test)]
fn get_trace() -> LineEvaluation {
    let domain = LineDomain::canonic(10);
    let trace = (0..(domain.len() as u32))
        .map(Field::from_u32_unchecked)
        .collect::<Vec<_>>();
    LineEvaluation::new(domain, trace)
}

#[test]
fn test_ifft_eval() {
    let eval = get_trace();
    let fftree = FFTree::preprocess(eval.domain);
    let poly = fftree.ifft(eval.clone());
    for (point, val) in eval.domain.iter().zip(eval.values.iter()) {
        assert_eq!(poly.eval_at_point(point), *val);
    }
}

#[test]
fn test_ifft_fft() {
    let eval = get_trace();
    let fftree = FFTree::preprocess(eval.domain);
    let poly = fftree.ifft(eval.clone());
    let eval2 = fftree.fft(poly);
    for (val0, val1) in eval.values.iter().zip(eval2.values) {
        assert_eq!(*val0, val1);
    }
}

#[test]
fn test_extend() {
    let eval = get_trace();
    let fftree = FFTree::preprocess(eval.domain);
    let poly = fftree.ifft(eval);
    let poly2 = poly.extend(LineDomain::canonic(poly.bound_bits + 2));

    assert_eq!(
        poly.eval_at_point(Field::one()),
        poly2.eval_at_point(Field::one())
    );
}
