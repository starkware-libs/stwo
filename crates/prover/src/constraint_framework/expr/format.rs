use num_traits::Zero;

use super::{BaseExpr, ColumnExpr, ExtExpr, CLAIMED_SUM_DUMMY_OFFSET};

impl BaseExpr {
    pub fn format_expr(&self) -> String {
        match self {
            BaseExpr::Col(ColumnExpr {
                interaction,
                idx,
                offset,
            }) => {
                let offset_str = if *offset == CLAIMED_SUM_DUMMY_OFFSET as isize {
                    "claimed_sum_offset".to_string()
                } else {
                    offset.to_string()
                };
                format!("col_{interaction}_{idx}[{offset_str}]")
            }
            BaseExpr::Const(c) => c.to_string(),
            BaseExpr::Param(v) => v.to_string(),
            BaseExpr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            BaseExpr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            BaseExpr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            BaseExpr::Neg(a) => format!("-({})", a.format_expr()),
            BaseExpr::Inv(a) => format!("1 / ({})", a.format_expr()),
        }
    }
}

impl ExtExpr {
    pub fn format_expr(&self) -> String {
        match self {
            ExtExpr::SecureCol([a, b, c, d]) => {
                // If the expression's non-base components are all constant zeroes, return the base
                // field representation of its first part.
                if **b == BaseExpr::zero() && **c == BaseExpr::zero() && **d == BaseExpr::zero() {
                    a.format_expr()
                } else {
                    format!(
                        "SecureCol({}, {}, {}, {})",
                        a.format_expr(),
                        b.format_expr(),
                        c.format_expr(),
                        d.format_expr()
                    )
                }
            }
            ExtExpr::Const(c) => {
                if c.0 .1.is_zero() && c.1 .0.is_zero() && c.1 .1.is_zero() {
                    // If the constant is in the base field, display it as such.
                    c.0 .0.to_string()
                } else {
                    c.to_string()
                }
            }
            ExtExpr::Param(v) => v.to_string(),
            ExtExpr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            ExtExpr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            ExtExpr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            ExtExpr::Neg(a) => format!("-({})", a.format_expr()),
        }
    }
}
