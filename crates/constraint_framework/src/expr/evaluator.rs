use std::collections::HashMap;

use num_traits::Zero;
use stwo_prover::core::lookups::utils::Fraction;

use super::assignment::{ExprVarAssignment, ExprVariables};
use super::degree::NamedExprs;
use super::{BaseExpr, ExtExpr};
use crate::expr::ColumnExpr;
use crate::preprocessed_columns::PreProcessedColumnId;
use crate::{EvalAtRow, Relation, RelationEntry, INTERACTION_TRACE_IDX};

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
    assert!(
        relation.get_size() >= values.len(),
        "Not enough alpha powers to combine values"
    );
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
    pub intermediates: HashMap<String, BaseExpr>,
    pub ext_intermediates: HashMap<String, ExtExpr>,
    // Save all intermediate names by order they can be assigned.
    ordered_intermediates: Vec<String>,
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
            // TODO(alont) unify both intermediate types.
            intermediates: HashMap::new(),
            ext_intermediates: HashMap::new(),
            ordered_intermediates: vec![],
        }
    }

    pub fn format_constraints(&self) -> String {
        let intermediates_string = self
            .ordered_intermediates
            .iter()
            .map(|name| {
                if self.intermediates.contains_key(name) {
                    format!(
                        "let {} = {};",
                        name,
                        self.intermediates[name].simplify_and_format()
                    )
                } else if self.ext_intermediates.contains_key(name) {
                    format!(
                        "let {} = {};",
                        name,
                        self.ext_intermediates[name].simplify_and_format()
                    )
                } else {
                    panic!(
                        "Intermediate {} not found in intermediates or ext_intermediates",
                        name
                    )
                }
            })
            .collect::<Vec<String>>()
            .join("\n\n");

        let constraints_str = self
            .constraints
            .iter()
            .enumerate()
            .map(|(i, c)| format!("let constraint_{i} = ") + &c.simplify_and_format() + ";")
            .collect::<Vec<String>>()
            .join("\n\n");

        [intermediates_string, constraints_str]
            .iter()
            .filter(|x| !x.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn constraint_degree_bounds(&self) -> Vec<usize> {
        let named_exprs = NamedExprs::new(
            self.intermediates
                .iter()
                .map(|(name, expr)| (name.clone(), expr.clone()))
                .collect(),
            self.ext_intermediates
                .iter()
                .map(|(name, expr)| (name.clone(), expr.clone()))
                .collect(),
        );
        self.constraints
            .iter()
            .map(|c| c.degree_bound(&named_exprs))
            .collect()
    }

    /// Collects all the variables used in the constraints and intermediates. Excludes the
    /// intermediates themselves.
    fn collect_variables(&self) -> ExprVariables {
        let all_vars = self
            .constraints
            .iter()
            .map(|expr| expr.collect_variables())
            .chain(
                self.intermediates
                    .values()
                    .map(|expr| expr.collect_variables()),
            )
            .chain(
                self.ext_intermediates
                    .values()
                    .map(|expr| expr.collect_variables()),
            )
            .sum::<ExprVariables>();
        let intermediate_vars = self
            .ordered_intermediates
            .iter()
            .map(|name| ExprVariables::param(name.into()))
            .sum::<ExprVariables>();

        all_vars - intermediate_vars
    }

    /// Returns a random assignment to all the variables including intermediates, where the values
    /// for intermediates are consistent.
    pub fn random_assignment(&self) -> ExprVarAssignment {
        let mut assignment = self.collect_variables().random_assignment(0);
        for intermediate in self.ordered_intermediates.clone() {
            if let Some(expr) = self.intermediates.get(&intermediate) {
                assignment
                    .1
                    .insert(intermediate.clone(), expr.assign(&assignment));
            } else if let Some(expr) = self.ext_intermediates.get(&intermediate) {
                assignment
                    .2
                    .insert(intermediate.clone(), expr.assign(&assignment));
            } else {
                panic!(
                    "Intermediate {} not found in intermediates or ext_intermediates",
                    intermediate
                );
            }
        }
        assignment
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
        self.intermediates.insert(name.clone(), expr);
        self.ordered_intermediates.push(name);
        intermediate
    }

    fn add_extension_intermediate(&mut self, expr: Self::EF) -> Self::EF {
        let name = format!(
            "intermediate{}",
            self.intermediates.len() + self.ext_intermediates.len()
        );
        let intermediate = ExtExpr::Param(name.clone());
        self.ext_intermediates.insert(name.clone(), expr);
        self.ordered_intermediates.push(name);
        intermediate
    }

    fn get_preprocessed_column(&mut self, column: PreProcessedColumnId) -> Self::F {
        BaseExpr::Param(column.id)
    }

    crate::logup_proxy!();

    fn next_trace_mask(&mut self) -> Self::F {
        let [mask_item] = self.next_interaction_mask(crate::ORIGINAL_TRACE_IDX, [0]);
        mask_item
    }

    fn next_extension_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::EF; N] {
        let mut res_col_major =
            std::array::from_fn(|_| self.next_interaction_mask(interaction, offsets).into_iter());
        std::array::from_fn(|_| {
            Self::combine_ef(res_col_major.each_mut().map(|iter| iter.next().unwrap()))
        })
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use stwo_prover::core::fields::FieldExpOps;

    use crate::expr::{ExprEvaluator, ExtExpr};
    use crate::{relation, EvalAtRow, FrameworkEval, RelationEntry};

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

    #[test]
    fn test_constraint_regression() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new());

        let assignment = eval.random_assignment();
        let constraint_regression = eval
            .constraints
            .iter()
            .map(|c| c.assign(&assignment))
            .collect::<Vec<_>>();

        let equiv_struct = EquivTestStruct {};
        let eval = equiv_struct.evaluate(ExprEvaluator::new());

        let assignment = eval.random_assignment();
        assert_eq!(
            constraint_regression,
            eval.constraints
                .iter()
                .map(|c| c.assign(&assignment))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic]
    fn test_constraint_regression_fails() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new());

        let assignment = eval.random_assignment();
        let constraint_regression = eval
            .constraints
            .iter()
            .map(|c| c.assign(&assignment))
            .collect::<Vec<_>>();

        let other_struct = TestStructWithDiffLookup {};
        let eval = other_struct.evaluate(ExprEvaluator::new());

        let assignment = eval.random_assignment();
        assert_eq!(
            constraint_regression,
            eval.constraints
                .iter()
                .map(|c| c.assign(&assignment))
                .collect::<Vec<_>>()
        );
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

    struct TestStructWithDiffLookup {}
    impl FrameworkEval for TestStructWithDiffLookup {
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
                &[x0, x1],
            ));
            eval.finalize_logup();
            eval
        }
    }

    struct EquivTestStruct {}
    impl FrameworkEval for EquivTestStruct {
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
            eval.add_constraint(
                x0.clone() * (x1.clone() * x2.clone()) * (x0.clone() + x1.clone()).inverse(),
            );
            eval.add_to_relation(RelationEntry::new(
                &TestRelation::dummy(),
                E::EF::one(),
                &[x0, x1, x2],
            ));
            eval.finalize_logup();
            eval
        }
    }

    #[test]
    fn test_constraint_degree_bounds() {
        let mut eval = ExprEvaluator::new();
        let x0 = eval.next_trace_mask();
        let x1 = eval.next_trace_mask();
        let x2 = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &TestRelation::dummy(),
            ExtExpr::one(),
            &[x0.clone()],
        ));
        eval.add_to_relation(RelationEntry::new(
            &TestRelation::dummy(),
            ExtExpr::one(),
            &[x0.clone() * x1.clone()],
        ));
        eval.add_to_relation(RelationEntry::new(
            &TestRelation::dummy(),
            ExtExpr::one(),
            &[x1.clone() * x2.clone()],
        ));
        eval.finalize_logup_in_pairs();

        // 1st and 2nd entries are batched together to produce a denominator of degree 3, hence
        // prefix sum constraint is of degree 4. 3rd entry is of degree 3 and is not batched.
        let expected = vec![4, 3];

        assert_eq!(eval.constraint_degree_bounds(), expected);
    }
}
