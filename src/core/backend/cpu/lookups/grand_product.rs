use num_traits::Zero;

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::CPUBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::grand_product::{GrandProductOps, GrandProductOracle, GrandProductTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::{SumcheckOracle, UnivariateEvals};

impl GrandProductOps for CPUBackend {
    fn next_layer(layer: &GrandProductTrace<Self>) -> GrandProductTrace<Self> {
        let half_n = layer.len() / 2;
        let mut next_layer = SecureColumn::default();

        for i in 0..half_n {
            let a = layer.at(i * 2);
            let b = layer.at(i * 2 + 1);
            next_layer.push(a * b);
        }

        GrandProductTrace::new(Mle::new(next_layer))
    }

    fn univariate_sum_evals(oracle: &GrandProductOracle<'_, Self>) -> UnivariateEvals {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();

        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        #[allow(clippy::needless_range_loop)]
        for i in 0..num_terms {
            let lhs0 = trace.at(i * 2);
            let lhs1 = trace.at(i * 2 + 1);

            let product0 = lhs0 * lhs1;

            let rhs0 = trace.at((num_terms + i) * 2);
            let rhs1 = trace.at((num_terms + i) * 2 + 1);

            let product2 = (rhs0.double() - lhs0) * (rhs1.double() - lhs1);

            let eq_eval = eq_evals[i];
            eval_at_0 += eq_eval * product0;
            eval_at_2 += eq_eval * product2;
        }

        UnivariateEvals {
            eval_at_0,
            eval_at_2,
        }
    }
}
