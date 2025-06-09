use num_traits::Zero;

use super::{BaseExpr, ColumnExpr, ExtExpr};

impl BaseExpr {
    pub fn format_expr(&self) -> String {
        match self {
            BaseExpr::Col(ColumnExpr {
                interaction,
                idx,
                offset,
            }) => {
                let offset_str = {
                    let offset_abs = offset.abs();
                    if *offset >= 0 {
                        offset.to_string()
                    } else {
                        format!("neg_{offset_abs}")
                    }
                };
                format!("trace_{interaction}_column_{idx}_offset_{offset_str}")
            }
            BaseExpr::Const(c) => format!("m31({c}).into()"),
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
                        "QM31Impl::from_partial_evals([{}, {}, {}, {}])",
                        a.format_expr(),
                        b.format_expr(),
                        c.format_expr(),
                        d.format_expr()
                    )
                }
            }
            ExtExpr::Const(c) => {
                let [v0, v1, v2, v3] = c.to_m31_array();
                format!("qm31({v0}, {v1}, {v2}, {v3})")
            }
            ExtExpr::Param(v) => v.to_string(),
            ExtExpr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            ExtExpr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            ExtExpr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            ExtExpr::Neg(a) => format!("-({})", a.format_expr()),
        }
    }
}
