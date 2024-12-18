use std::sync::Arc;

use num_traits::Zero;

use super::{BaseExpr, ExtExpr};
use crate::constraint_framework::expr::ColumnExpr;
use crate::constraint_framework::preprocessed_columns::PreprocessedColumnOps;
use crate::constraint_framework::{EvalAtRow, Relation, RelationEntry, INTERACTION_TRACE_IDX};
use crate::core::fields::m31;
use crate::core::lookups::utils::Fraction;

pub struct FormalLogupAtRow {
    pub interaction: usize,
    pub total_sum: ExtExpr,
    pub claimed_sum: Option<(ExtExpr, usize)>,
    pub fracs: Vec<Fraction<ExtExpr, ExtExpr>>,
    pub is_finalized: bool,
    pub is_first: BaseExpr,
    pub log_size: u32,
}

// P is an offset no column can reach, it signifies the variable
// offset, which is an input to the verifier.
pub const CLAIMED_SUM_DUMMY_OFFSET: usize = m31::P as usize;

impl FormalLogupAtRow {
    pub fn new(interaction: usize, has_partial_sum: bool, log_size: u32) -> Self {
        let total_sum_name = "total_sum".to_string();
        let claimed_sum_name = "claimed_sum".to_string();

        Self {
            interaction,
            // TODO(alont): Should these be Expr::SecureField?
            total_sum: ExtExpr::Param(total_sum_name),
            claimed_sum: has_partial_sum
                .then_some((ExtExpr::Param(claimed_sum_name), CLAIMED_SUM_DUMMY_OFFSET)),
            fracs: vec![],
            is_finalized: true,
            is_first: BaseExpr::zero(),
            log_size,
        }
    }
}

/// Returns the expression
/// `value[0] * <relation>_alpha0 + value[1] * <relation>_alpha1 + ... - <relation>_z.`
fn combine_formal<R: Relation<BaseExpr, ExtExpr>>(relation: &R, values: &[BaseExpr]) -> ExtExpr {
    const Z_SUFFIX: &str = "_z";
    const ALPHA_SUFFIX: &str = "_alpha";

    let z = ExtExpr::Param(relation.get_name().to_owned() + Z_SUFFIX);
    let alpha_powers = (0..relation.get_size())
        .map(|i| ExtExpr::Param(relation.get_name().to_owned() + ALPHA_SUFFIX + &i.to_string()));
    values
        .iter()
        .zip(alpha_powers)
        .fold(ExtExpr::zero(), |acc, (value, power)| {
            acc + power * value.clone()
        })
        - z
}

/// An Evaluator that saves all constraint expressions.
pub struct ExprEvaluator {
    pub cur_var_index: usize,
    pub constraints: Vec<ExtExpr>,
    pub logup: FormalLogupAtRow,
    pub intermediates: Vec<(String, BaseExpr)>,
    pub ext_intermediates: Vec<(String, ExtExpr)>,
}

impl ExprEvaluator {
    pub fn new(log_size: u32, has_partial_sum: bool) -> Self {
        Self {
            cur_var_index: Default::default(),
            constraints: Default::default(),
            logup: FormalLogupAtRow::new(INTERACTION_TRACE_IDX, has_partial_sum, log_size),
            intermediates: vec![],
            ext_intermediates: vec![],
        }
    }

    pub fn format_constraints(&self) -> String {
        let lets_string = self
            .intermediates
            .iter()
            .map(|(name, expr)| format!("let {} = {};", name, expr.simplify_and_format()))
            .collect::<Vec<String>>()
            .join("\n\n");

        let secure_lets_string = self
            .ext_intermediates
            .iter()
            .map(|(name, expr)| format!("let {} = {};", name, expr.simplify_and_format()))
            .collect::<Vec<String>>()
            .join("\n\n");

        let constraints_str = self
            .constraints
            .iter()
            .enumerate()
            .map(|(i, c)| format!("let constraint_{i} = ") + &c.simplify_and_format() + ";")
            .collect::<Vec<String>>()
            .join("\n\n");

        [lets_string, secure_lets_string, constraints_str]
            .iter()
            .filter(|x| !x.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl EvalAtRow for ExprEvaluator {
    // TODO(alont): Should there be a version of this that disallows Secure fields for F?
    type F = BaseExpr;
    type EF = ExtExpr;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let res = std::array::from_fn(|i| {
            let col = ColumnExpr::from((interaction, self.cur_var_index, offsets[i]));
            BaseExpr::Col(col)
        });
        self.cur_var_index += 1;
        res
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: From<G>,
    {
        self.constraints.push(constraint.into());
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        ExtExpr::SecureCol([
            Box::new(values[0].clone()),
            Box::new(values[1].clone()),
            Box::new(values[2].clone()),
            Box::new(values[3].clone()),
        ])
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let intermediate =
            self.add_extension_intermediate(combine_formal(entry.relation, entry.values));
        let frac = Fraction::new(entry.multiplicity.clone(), intermediate);
        self.write_logup_frac(frac);
    }

    fn add_intermediate(&mut self, expr: Self::F) -> Self::F {
        let name = format!(
            "intermediate{}",
            self.intermediates.len() + self.ext_intermediates.len()
        );
        let intermediate = BaseExpr::Param(name.clone());
        self.intermediates.push((name, expr));
        intermediate
    }

    fn add_extension_intermediate(&mut self, expr: Self::EF) -> Self::EF {
        let name = format!(
            "intermediate{}",
            self.intermediates.len() + self.ext_intermediates.len()
        );
        let intermediate = ExtExpr::Param(name.clone());
        self.ext_intermediates.push((name, expr));
        intermediate
    }

    fn get_preprocessed_column(&mut self, column: Arc<dyn PreprocessedColumnOps>) -> Self::F {
        BaseExpr::Param(column.name().to_string())
    }

    crate::constraint_framework::logup_proxy!();
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::constraint_framework::expr::ExprEvaluator;
    use crate::constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry};
    use crate::core::fields::FieldExpOps;
    use crate::relation;

    #[test]
    fn test_expr_evaluator() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new(16, false));
        let expected = "let intermediate0 = (col_1_1[0]) * (col_1_2[0]);

\
        let intermediate1 = (TestRelation_alpha0) * (col_1_0[0]) \
            + (TestRelation_alpha1) * (col_1_1[0]) \
            + (TestRelation_alpha2) * (col_1_2[0]) \
            - (TestRelation_z);

\
        let constraint_0 = ((col_1_0[0]) * (intermediate0)) * (1 / (col_1_0[0] + col_1_1[0]));

\
        let constraint_1 = (SecureCol(col_2_3[0], col_2_4[0], col_2_5[0], col_2_6[0]) \
            - (SecureCol(col_2_3[-1], col_2_4[-1], col_2_5[-1], col_2_6[-1]) \
                - ((total_sum) * (preprocessed.is_first)))) \
            * (intermediate1) \
            - (1);"
            .to_string();

        assert_eq!(eval.format_constraints(), expected);
    }

    relation!(TestRelation, 3);

    struct TestStruct {}
    impl FrameworkEval for TestStruct {
        fn log_size(&self) -> u32 {
            0
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            0
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let x0 = eval.next_trace_mask();
            let x1 = eval.next_trace_mask();
            let x2 = eval.next_trace_mask();
            let intermediate = eval.add_intermediate(x1.clone() * x2.clone());
            eval.add_constraint(x0.clone() * intermediate * (x0.clone() + x1.clone()).inverse());
            eval.add_to_relation(RelationEntry::new(
                &TestRelation::dummy(),
                E::EF::one(),
                &[x0, x1, x2],
            ));
            eval.finalize_logup();
            eval
        }
    }
}
