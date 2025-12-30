pub mod arena;
pub mod assignment;
pub mod degree;
pub mod evaluator;
pub mod format;
pub mod simplify;
pub mod utils;

pub use arena::{
    clear_arena, init_arena, set_arena, take_arena, with_arena, with_arena_ref, BaseExpr,
    BaseExprNode, ExprArena, ExtExpr, ExtExprNode, StringId,
};
pub use evaluator::ExprEvaluator;

/// A single base field column at index `idx` of interaction `interaction`, at mask offset `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnExpr {
    pub interaction: usize,
    pub idx: usize,
    pub offset: isize,
}

impl From<(usize, usize, isize)> for ColumnExpr {
    fn from((interaction, idx, offset): (usize, usize, isize)) -> Self {
        Self {
            interaction,
            idx,
            offset,
        }
    }
}
