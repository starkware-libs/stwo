use crate::constraint_framework::constant_columns::ConstantColumn;
use crate::constraint_framework::{EvalAtRow, FrameworkEval};

pub struct Add1Eval {
    pub log_size: u32,
}

impl FrameworkEval for Add1Eval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a = eval.next_trace_mask();
        let b = eval.next_trace_mask();
        let [one] = eval.constant_interaction_mask(ConstantColumn::One(self.log_size), [0]);
        eval.add_constraint(a + one - b);
        eval
    }
}

pub struct Add2Eval {
    pub log_size: u32,
}

impl FrameworkEval for Add2Eval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a = eval.next_trace_mask();
        let b = eval.next_trace_mask();
        let [one, _] = eval.constant_interaction_mask(ConstantColumn::One(self.log_size), [0, 1]);
        eval.add_constraint(a + one + one - b);
        eval
    }
}
