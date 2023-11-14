use crate::core::{
    air::{Mask, MaskItem},
    circle::Coset,
    constraints::{coset_vanishing, point_excluder, point_vanishing, PolyOracle},
    fields::m31::Field,
    poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation},
};
use num_traits::One;

pub struct Fibonacci {
    pub trace_coset: CanonicCoset,
    pub eval_domain: CircleDomain,
    pub constraint_coset: Coset,
    pub constraint_eval_domain: CircleDomain,
    pub claim: Field,
}
impl Fibonacci {
    pub fn new(n_bits: usize, claim: Field) -> Self {
        let trace_coset = CanonicCoset::new(n_bits);
        let eval_domain = trace_coset.eval_domain(n_bits + 1);
        let constraint_coset = Coset::subgroup(n_bits);
        let constraint_eval_domain = CircleDomain::constraint_domain(n_bits + 1);
        Self {
            trace_coset,
            eval_domain,
            constraint_coset,
            constraint_eval_domain,
            claim,
        }
    }
    pub fn get_trace(&self) -> CircleEvaluation {
        // Trace.
        let mut trace = Vec::with_capacity(self.trace_coset.len());

        // Fill trace with fibonacci squared.
        let mut a = Field::one();
        let mut b = Field::one();
        for _ in 0..self.trace_coset.len() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(self.trace_coset, trace)
    }
    pub fn eval_step_constraint(&self, trace: impl PolyOracle) -> Field {
        trace.get_at(self.trace_coset.index_at(0)).square()
            + trace.get_at(self.trace_coset.index_at(1)).square()
            - trace.get_at(self.trace_coset.index_at(2))
    }

    pub fn eval_step_quotient(&self, trace: impl PolyOracle) -> Field {
        let excluded0 = self.constraint_coset.at(self.constraint_coset.len() - 2);
        let excluded1 = self.constraint_coset.at(self.constraint_coset.len() - 1);
        let num = self.eval_step_constraint(trace)
            * point_excluder(excluded0, trace.point())
            * point_excluder(excluded1, trace.point());
        let denom = coset_vanishing(self.constraint_coset, trace.point());
        num / denom
    }

    pub fn eval_boundary_constraint(&self, trace: impl PolyOracle, value: Field) -> Field {
        trace.get_at(self.trace_coset.index_at(0)) - value
    }

    pub fn eval_boundary_quotient(
        &self,
        trace: impl PolyOracle,
        point_index: usize,
        value: Field,
    ) -> Field {
        let num = self.eval_boundary_constraint(trace, value);
        let denom = point_vanishing(self.constraint_coset.at(point_index), trace.point());
        num / denom
    }

    pub fn eval_quotient(&self, random_coeff: Field, trace: impl PolyOracle) -> Field {
        let mut quotient = random_coeff.pow(0) * self.eval_step_quotient(trace);
        quotient += random_coeff.pow(1) * self.eval_boundary_quotient(trace, 0, Field::one());
        quotient += random_coeff.pow(2)
            * self.eval_boundary_quotient(trace, self.constraint_coset.len() - 1, self.claim);
        quotient
    }

    pub fn get_mask(&self) -> Mask {
        Mask::new(
            (0..3)
                .map(|offset| MaskItem {
                    column_index: 0,
                    offset,
                })
                .collect(),
        )
    }
}

#[test]
fn test_constraint_on_trace() {
    use crate::core::constraints::EvalByEvaluation;
    use num_traits::Zero;

    let fib = Fibonacci::new(3, Field::from_u32_unchecked(1056169651));
    let trace = fib.get_trace();

    // Assert that the step constraint is satisfied on the trace.
    for p_ind in fib
        .constraint_coset
        .iter_indices()
        .take(fib.constraint_coset.len() - 2)
    {
        let res = fib.eval_step_constraint(EvalByEvaluation {
            offset: p_ind,
            eval: &trace,
        });
        assert_eq!(res, Field::zero());
    }

    // Assert that the first trace value is 1.
    assert_eq!(
        fib.eval_boundary_constraint(
            EvalByEvaluation {
                offset: fib.constraint_coset.index_at(0),
                eval: &trace,
            },
            Field::one()
        ),
        Field::zero()
    );

    // Assert that the last trace value is the fibonacci claim.
    assert_eq!(
        fib.eval_boundary_constraint(
            EvalByEvaluation {
                offset: fib
                    .constraint_coset
                    .index_at(fib.constraint_coset.len() - 1),
                eval: &trace,
            },
            fib.claim
        ),
        Field::zero()
    );
}

#[test]
fn test_quotient_is_low_degree() {
    use crate::core::circle::CirclePointIndex;
    use crate::core::constraints::EvalByEvaluation;
    use crate::core::constraints::EvalByPoly;
    use crate::core::poly::circle::PointSetEvaluation;

    let fib = Fibonacci::new(5, Field::from_u32_unchecked(443693538));
    let trace = fib.get_trace();
    let trace_poly = trace.interpolate();

    let extended_evaluation = trace_poly.clone().evaluate(fib.eval_domain);

    // TODO(ShaharS), Change to a channel implementation to retrieve the random coefficients from extension field.
    let random_coeff = Field::from_u32_unchecked(2213980);

    // Compute quotient on the evaluation domain.
    let mut quotient_values = Vec::with_capacity(fib.constraint_eval_domain.len());
    for p_ind in fib.constraint_eval_domain.iter_indices() {
        quotient_values.push(fib.eval_quotient(
            random_coeff,
            EvalByEvaluation {
                offset: p_ind,
                eval: &extended_evaluation,
            },
        ));
    }
    let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
    // Interpolate the poly. The the poly is indeed of degree lower than the size of eval_domain,
    // then it should interpolate correctly.
    let quotient_poly = quotient_eval.interpolate();

    // Evaluate this polynomial at another point, out of eval_domain and compare to what we expect.
    let oods_point_index = CirclePointIndex::generator() * 2;
    assert!(fib.constraint_eval_domain.find(oods_point_index).is_none());
    let oods_point = oods_point_index.to_point();

    let mask = fib.get_mask();
    let oods_values = mask.eval(
        &[fib.trace_coset],
        &[EvalByPoly {
            point: oods_point,
            poly: &trace_poly,
        }],
    );
    let point_domain = mask
        .get_point_indices(&[fib.trace_coset])
        .iter()
        .map(|p| (*p + oods_point_index).to_point())
        .collect();

    let oods_evaluation = EvalByEvaluation {
        offset: oods_point_index,
        eval: &PointSetEvaluation {
            domain: point_domain,
            values: oods_values,
        },
    };

    assert_eq!(
        quotient_poly.eval_at_point(oods_point),
        fib.eval_quotient(random_coeff, oods_evaluation)
    );
}

#[test]
fn test_mask() {
    use crate::core::circle::CirclePointIndex;
    use crate::core::constraints::EvalByPoly;

    let fib = Fibonacci::new(5, Field::from_u32_unchecked(443693538));
    let trace = fib.get_trace();
    let trace_poly = trace.interpolate();
    let z = (CirclePointIndex::generator() * 17).to_point();

    let mask = fib.get_mask();
    let mask_eval = mask.eval(
        &[fib.trace_coset],
        &[EvalByPoly {
            point: z,
            poly: &trace_poly,
        }],
    );

    assert_eq!(mask.items[0].column_index, 0);
    assert_eq!(
        mask_eval[0],
        trace_poly.eval_at_point(z + fib.trace_coset.coset.at(0))
    );
    assert_eq!(
        mask_eval[1],
        trace_poly.eval_at_point(z + fib.trace_coset.coset.at(1))
    );
    assert_eq!(
        mask_eval[2],
        trace_poly.eval_at_point(z + fib.trace_coset.coset.at(2))
    );
}
