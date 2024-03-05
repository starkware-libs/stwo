mod canonic;
mod domain;
mod evaluation;
mod ops;
mod poly;
mod secure_poly;

pub use canonic::CanonicCoset;
pub use domain::CircleDomain;
pub use evaluation::{CircleEvaluation, CosetSubEvaluation};
pub use ops::PolyOps;
pub use poly::CirclePoly;
pub use secure_poly::{SecureCircleEvaluation, SecureCirclePoly};

#[cfg(test)]
mod tests {
    use super::{CanonicCoset, CircleDomain};
    use crate::core::backend::cpu::CPUCircleEvaluation;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Field;
    use crate::core::poly::NaturalOrder;
    use crate::core::utils::bit_reverse_index;

    #[test]
    fn test_interpolate_and_eval() {
        let domain = CircleDomain::constraint_evaluation_domain(3);
        assert_eq!(domain.log_size(), 3);
        let evaluation =
            CPUCircleEvaluation::new(domain, (0..8).map(BaseField::from_u32_unchecked).collect());
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain);
        assert_eq!(evaluation.values, evaluation2.values);
    }

    #[test]
    fn test_mixed_degree_example() {
        let log_size = 4;

        // Compute domains.
        let domain0 = CanonicCoset::new(log_size);
        let eval_domain0 = domain0.evaluation_domain(log_size + 4);
        let domain1 = CanonicCoset::new(log_size + 2);
        let eval_domain1 = domain1.evaluation_domain(log_size + 3);
        let constraint_domain = CircleDomain::constraint_evaluation_domain(log_size + 1);

        // Compute values.
        let values1: Vec<_> = (0..(domain1.size() as u32))
            .map(BaseField::from_u32_unchecked)
            .collect();
        let values0: Vec<_> = values1[1..].iter().step_by(4).map(|x| *x * *x).collect();

        // Extend.
        let trace_eval0 = CPUCircleEvaluation::new_canonical_ordered(domain0, values0);
        let eval0 = trace_eval0.interpolate().evaluate(eval_domain0);
        let trace_eval1 = CPUCircleEvaluation::new_canonical_ordered(domain1, values1);
        let eval1 = trace_eval1.interpolate().evaluate(eval_domain1);

        // Compute constraint.
        let constraint_eval = CPUCircleEvaluation::<BaseField, NaturalOrder>::new(
            constraint_domain,
            constraint_domain
                .iter_indices()
                .map(|ind| {
                    // The constraint is poly0(x+off0)^2 = poly1(x+off1).
                    eval0.get_at(ind).square() - eval1.get_at(domain1.index_at(1) + ind).square()
                })
                .collect(),
        );
        // TODO(spapini): Check low degree.
        println!("{:?}", constraint_eval);
    }

    #[test]
    fn is_canonic_valid_domain() {
        let canonic_domain = CanonicCoset::new(4).circle_domain();

        assert!(canonic_domain.is_canonic());
    }

    #[test]
    pub fn test_bit_reverse_indices() {
        let log_domain_size = 7;
        let log_small_domain_size = 5;
        let domain = CanonicCoset::new(log_domain_size);
        let small_domain = CanonicCoset::new(log_small_domain_size);
        let n_folds = log_domain_size - log_small_domain_size;
        for i in 0..2usize.pow(log_domain_size) {
            let point = domain.at(bit_reverse_index(i, log_domain_size));
            let small_point = small_domain.at(bit_reverse_index(
                i / 2usize.pow(n_folds),
                log_small_domain_size,
            ));
            assert_eq!(point.repeated_double(n_folds), small_point);
        }
    }
}
