use crate::core::{
    circle::CircleIndex,
    commitment::{Decommitment, LayerOracle, TableCommitmentProver},
    constraints::{domain_poly_eval, point_excluder},
    fft::FFTree,
    field::Field,
    poly::{
        circle::{CircleDomain, CircleEvaluation, CirclePoly},
        line::LineDomain,
        oracle::CircleFunctionOracle,
    },
};

#[cfg(test)]
use crate::core::circle::CIRCLE_GEN;
#[cfg(test)]
use crate::core::poly::oracle::{CircleEvalOracle, CirclePolyOracle};
// // #[cfg(test)]
// use crate::core::fft::FFTree;

// user.
pub struct TraceInfo {
    pub values_domain: CircleDomain,
    pub values_extension_domain: CircleDomain,
}
impl TraceInfo {
    pub fn new(n_bits: usize) -> Self {
        let values_extension_domain = CircleDomain::canonic_evaluation(n_bits + 1);
        let values_domain =
            CircleDomain::deduce_from_extension_domain(values_extension_domain, n_bits);
        Self {
            values_domain,
            values_extension_domain,
        }
    }
    pub fn get_trace(&self) -> TraceEvaluation {
        // Trace.
        let trace_n_bits = self.values_domain.n_bits();
        let n = 1 << trace_n_bits;
        let mut trace = vec![];
        trace.reserve(n);

        // Fill trace with fibonacci squared.
        let mut a = Field::one();
        let mut b = Field::one();
        for _ in 0..n {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        TraceEvaluation {
            values: CircleEvaluation::new(self.values_domain, trace),
        }
    }
    pub fn eval_constraint(&self, trace: impl CircleFunctionOracle) -> Field {
        let step = self.values_domain.coset.step_size;
        trace.get_at(CircleIndex::zero()).square() + trace.get_at(step * 1).square()
            - trace.get_at(step * 2)
    }
    pub fn eval_quotient(&self, trace: impl CircleFunctionOracle) -> Field {
        let excluded0 = self.values_domain.at(self.values_domain.len() - 2);
        let excluded1 = self.values_domain.at(self.values_domain.len() - 1);
        let num = self.eval_constraint(trace)
            * point_excluder(trace.point(), excluded0)
            * point_excluder(trace.point(), excluded1);
        let denom = domain_poly_eval(self.values_domain, trace.point());
        num / denom
    }
}

// autogen
pub struct TraceEvaluation {
    pub values: CircleEvaluation,
}
impl TraceEvaluation {
    pub fn interpolate(self) -> TracePolys {
        let tree = FFTree::preprocess(self.values.domain.projected_line_domain);
        let values = self.values.interpolate(&tree);

        TracePolys { values }
    }
}

pub struct TracePolys {
    values: CirclePoly,
}
impl TracePolys {
    pub fn extend(&self, trace_info: &TraceInfo) -> TraceExtension {
        let values = self.values.extend(trace_info.values_extension_domain);
        let tree = FFTree::preprocess(values.domain.projected_line_domain);
        let values = values.evaluate(&tree);
        TraceExtension { values }
    }
}

pub struct TraceExtension {
    pub values: CircleEvaluation,
}
impl TraceExtension {
    pub fn commit(&self, trace_info: &TraceInfo, blowup_bit: usize) -> TraceCommitmentProver {
        let values = self
            .values
            .skipped(self.values.domain.n_bits() - trace_info.values_domain.n_bits() + blowup_bit)
            .semi_interpolate();
        TraceCommitmentProver {
            poly0_domain: values.poly0_eval.domain,
            poly1_domain: values.poly1_eval.domain,
            values: TableCommitmentProver::new(&[
                &values.poly0_eval.values,
                &values.poly1_eval.values,
            ]),
        }
    }
}

pub struct TraceCommitmentProver {
    poly0_domain: LineDomain,
    poly1_domain: LineDomain,
    values: TableCommitmentProver,
}
impl TraceCommitmentProver {
    pub fn decommit(&self, trace_polys: &TracePolys, indices: Vec<usize>) -> Decommitment {
        self.values.decommit(
            indices,
            &[
                LayerOracle {
                    domain: self.poly0_domain,
                    poly: &trace_polys.values.poly0,
                },
                LayerOracle {
                    domain: self.poly1_domain,
                    poly: &trace_polys.values.poly1,
                },
            ],
        )
    }
}

pub struct TraceDecommitment {}

// Per constraint?
pub struct Mask {
    pub values0: Field,
    pub values1: Field,
    pub values2: Field,
}

#[test]
fn test_constraint_on_trace() {
    let trace_info = TraceInfo::new(3);
    let trace = trace_info.get_trace();
    let trace_domain = trace.values.domain;
    for (i, _point) in trace_domain.iter().enumerate().take(6) {
        let res = trace_info.eval_constraint(CircleEvalOracle {
            domain: trace_domain.coset,
            offset: i,
            eval: &trace.values,
        });
        assert_eq!(res, Field::zero());
    }
}

#[test]
fn test_quotient_is_low_degree() {
    let trace_info = TraceInfo::new(5);
    let trace = trace_info.get_trace();
    let trace_polys = trace.interpolate();
    let trace_extension = trace_polys.extend(&trace_info);

    // Compute quotient on other cosets.
    let mut quotient_values = Vec::with_capacity(trace_info.values_extension_domain.len());
    for (i, _point) in trace_info.values_extension_domain.iter().enumerate() {
        quotient_values.push(trace_info.eval_quotient(CircleEvalOracle {
            domain: trace_info.values_extension_domain.coset,
            offset: i,
            eval: &trace_extension.values,
        }));
    }
    let quotient_eval = CircleEvaluation::new(trace_info.values_extension_domain, quotient_values);
    let quotient_poly = quotient_eval.interpolate(&FFTree::preprocess(
        trace_info.values_extension_domain.projected_line_domain,
    ));
    let point = -CIRCLE_GEN;
    assert_eq!(
        quotient_poly.eval_at_point(point),
        trace_info.eval_quotient(CirclePolyOracle {
            point,
            poly: &trace_polys.values
        })
    );
}
