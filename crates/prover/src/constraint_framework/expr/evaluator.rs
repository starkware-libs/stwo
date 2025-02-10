use num_traits::Zero;

use super::{BaseExpr, ExtExpr};
use crate::constraint_framework::expr::ColumnExpr;
use crate::constraint_framework::preprocessed_columns::PreProcessedColumnId;
use crate::constraint_framework::{EvalAtRow, Relation, RelationEntry, INTERACTION_TRACE_IDX};
use crate::core::lookups::utils::Fraction;

pub struct FormalLogupAtRow {
    pub interaction: usize,
    pub claimed_sum: ExtExpr,
    pub fracs: Vec<Fraction<ExtExpr, ExtExpr>>,
    pub is_finalized: bool,
    pub is_first: BaseExpr,
    pub cumsum_shift: ExtExpr,
}

impl FormalLogupAtRow {
    pub fn new(interaction: usize) -> Self {
        let claimed_sum_name = "claimed_sum".to_string();
        let column_size_name = "column_size".to_string();

        Self {
            interaction,
            // TODO(alont): Should these be Expr::SecureField?
            claimed_sum: ExtExpr::Param(claimed_sum_name.clone()),
            fracs: vec![],
            is_finalized: true,
            is_first: BaseExpr::zero(),
            cumsum_shift: ExtExpr::Param(claimed_sum_name)
                * BaseExpr::Inv(Box::new(BaseExpr::Param(column_size_name))),
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

impl Default for ExprEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprEvaluator {
    pub fn new() -> Self {
        Self {
            cur_var_index: Default::default(),
            constraints: Default::default(),
            logup: FormalLogupAtRow::new(INTERACTION_TRACE_IDX),
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

    fn get_preprocessed_column(&mut self, column: PreProcessedColumnId) -> Self::F {
        BaseExpr::Param(column.id)
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
        let eval = test_struct.evaluate(ExprEvaluator::new());
        let expected = "let intermediate0 = (trace_1_column_1_offset_0) * (trace_1_column_2_offset_0);

\
        let intermediate1 = (TestRelation_alpha0) * (trace_1_column_0_offset_0) \
            + (TestRelation_alpha1) * (trace_1_column_1_offset_0) \
            + (TestRelation_alpha2) * (trace_1_column_2_offset_0) \
            - (TestRelation_z);

\
        let constraint_0 = ((trace_1_column_0_offset_0) * (intermediate0)) * (1 / (trace_1_column_0_offset_0 + trace_1_column_1_offset_0));

\
        let constraint_1 = (QM31Impl::from_partial_evals([trace_2_column_3_offset_0, trace_2_column_4_offset_0, trace_2_column_5_offset_0, trace_2_column_6_offset_0]) \
            - (QM31Impl::from_partial_evals([trace_2_column_3_offset_neg_1, trace_2_column_4_offset_neg_1, trace_2_column_5_offset_neg_1, trace_2_column_6_offset_neg_1])) \
                + (claimed_sum) * (1 / (column_size))) \
            * (intermediate1) \
            - (qm31(1, 0, 0, 0));"
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
