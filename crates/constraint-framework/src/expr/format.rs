use num_traits::Zero;

use super::{BaseExpr, BaseExprNode, ColumnExpr, ExtExpr, ExtExprNode};

impl BaseExpr {
    pub fn format_expr(&self) -> String {
        let node = self.node();
        match node {
            BaseExprNode::Col(ColumnExpr {
                interaction,
                idx,
                offset,
            }) => {
                let offset_str = {
                    let offset_abs = offset.abs();
                    if offset >= 0 {
                        offset.to_string()
                    } else {
                        format!("neg_{offset_abs}")
                    }
                };
                format!("trace_{interaction}_column_{idx}_offset_{offset_str}")
            }
            BaseExprNode::Const(c) => format!("m31({c}).into()"),
            BaseExprNode::Param(v) => v.as_str(),
            BaseExprNode::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            BaseExprNode::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            BaseExprNode::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            BaseExprNode::Neg(a) => format!("-({})", a.format_expr()),
            BaseExprNode::Inv(a) => format!("1 / ({})", a.format_expr()),
        }
    }
}

impl ExtExpr {
    pub fn format_expr(&self) -> String {
        let node = self.node();
        match node {
            ExtExprNode::SecureCol([a, b, c, d]) => {
                // If the expression's non-base components are all constant zeroes, return the base
                // field representation of its first part.
                let b_node = b.node();
                let c_node = c.node();
                let d_node = d.node();
                let b_is_zero = matches!(b_node, BaseExprNode::Const(v) if v.is_zero());
                let c_is_zero = matches!(c_node, BaseExprNode::Const(v) if v.is_zero());
                let d_is_zero = matches!(d_node, BaseExprNode::Const(v) if v.is_zero());
                if b_is_zero && c_is_zero && d_is_zero {
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
            ExtExprNode::Const(c) => {
                let [v0, v1, v2, v3] = c.to_m31_array();
                format!("qm31({v0}, {v1}, {v2}, {v3})")
            }
            ExtExprNode::Param(v) => v.as_str(),
            ExtExprNode::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            ExtExprNode::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            ExtExprNode::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            ExtExprNode::Neg(a) => format!("-({})", a.format_expr()),
        }
    }
}
