use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use num_traits::{One, Zero};

use super::{EvalAtRow, Relation, RelationEntry, INTERACTION_TRACE_IDX};
use crate::core::fields::m31::{self, BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;

/// A single base field column at index `idx` of interaction `interaction`, at mask offset `offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnExpr {
    interaction: usize,
    idx: usize,
    offset: isize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Col(ColumnExpr),
    /// An atomic secure column constructed from 4 expressions.
    /// Expressions on the secure column are not reduced, i.e,
    /// if `a = SecureCol(a0, a1, a2, a3)`, `b = SecureCol(b0, b1, b2, b3)` then
    /// `a + b` evaluates to `Add(a, b)` rather than
    /// `SecureCol(Add(a0, b0), Add(a1, b1), Add(a2, b2), Add(a3, b3))`
    SecureCol([Box<Expr>; 4]),
    Const(BaseField),
    /// Formal parameter to the AIR, for example the interaction elements of a relation.
    Param(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Inv(Box<Expr>),
}

impl Expr {
    pub fn format_expr(&self) -> String {
        match self {
            Expr::Col(ColumnExpr {
                interaction,
                idx,
                offset,
            }) => {
                let offset_str = if *offset == CLAIMED_SUM_DUMMY_OFFSET.try_into().unwrap() {
                    "claimed_sum_offset".to_string()
                } else {
                    offset.to_string()
                };
                format!("col_{interaction}_{idx}[{offset_str}]")
            }
            Expr::SecureCol([a, b, c, d]) => format!(
                "SecureCol({}, {}, {}, {})",
                a.format_expr(),
                b.format_expr(),
                c.format_expr(),
                d.format_expr()
            ),
            Expr::Const(c) => c.0.to_string(),
            Expr::Param(v) => v.to_string(),
            Expr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            Expr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            Expr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            Expr::Neg(a) => format!("-({})", a.format_expr()),
            Expr::Inv(a) => format!("1 / ({})", a.format_expr()),
        }
    }

    pub fn simplify_and_format(&self) -> String {
        simplify(self.clone()).format_expr()
    }
}

impl From<BaseField> for Expr {
    fn from(val: BaseField) -> Self {
        Expr::Const(val)
    }
}

impl From<SecureField> for Expr {
    fn from(val: SecureField) -> Self {
        Expr::SecureCol([
            Box::new(val.0 .0.into()),
            Box::new(val.0 .1.into()),
            Box::new(val.1 .0.into()),
            Box::new(val.1 .1.into()),
        ])
    }
}

impl Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl AddAssign for Expr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs
    }
}

impl MulAssign for Expr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

impl Neg for Expr {
    type Output = Self;
    fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }
}

impl Zero for Expr {
    fn zero() -> Self {
        Expr::Const(BaseField::zero())
    }
    fn is_zero(&self) -> bool {
        // TODO(alont): consider replacing `Zero` in the trait bound with a custom trait
        // that only has `zero()`.
        panic!("Can't check if an expression is zero.");
    }
}

impl One for Expr {
    fn one() -> Self {
        Expr::Const(BaseField::one())
    }
}

impl FieldExpOps for Expr {
    fn inverse(&self) -> Self {
        Expr::Inv(Box::new(self.clone()))
    }
}

impl Add<BaseField> for Expr {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        self + Expr::from(rhs)
    }
}

impl Mul<BaseField> for Expr {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self * Expr::from(rhs)
    }
}

impl Mul<SecureField> for Expr {
    type Output = Self;
    fn mul(self, rhs: SecureField) -> Self {
        self * Expr::from(rhs)
    }
}

impl Add<SecureField> for Expr {
    type Output = Self;
    fn add(self, rhs: SecureField) -> Self {
        self + Expr::from(rhs)
    }
}

impl Sub<SecureField> for Expr {
    type Output = Self;
    fn sub(self, rhs: SecureField) -> Self {
        self - Expr::from(rhs)
    }
}

impl AddAssign<BaseField> for Expr {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = self.clone() + Expr::from(rhs)
    }
}

const ZERO: M31 = M31(0);
const ONE: M31 = M31(1);
const MINUS_ONE: M31 = M31(m31::P - 1);

// TODO(alont) Add random point assignment test.
pub fn simplify(expr: Expr) -> Expr {
    match expr {
        Expr::Add(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (a.clone(), b.clone()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                (Expr::Const(ZERO), _) => b, // 0 + b = b
                (_, Expr::Const(ZERO)) => a, // a + 0 = a
                // (-a + -b) = -(a + b)
                (Expr::Neg(minus_a), Expr::Neg(minus_b)) => -(*minus_a + *minus_b),
                (Expr::Neg(minus_a), _) => b - *minus_a, // -a + b = b - a
                (_, Expr::Neg(minus_b)) => a - *minus_b, // a + -b = a - b
                _ => Expr::Add(Box::new(a), Box::new(b)),
            }
        }
        Expr::Sub(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (a.clone(), b.clone()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
                (Expr::Const(ZERO), _) => -b, // 0 - b = -b
                (_, Expr::Const(ZERO)) => a,  // a - 0 = a
                // (-a - -b) = b - a
                (Expr::Neg(minus_a), Expr::Neg(minus_b)) => *minus_b - *minus_a,
                (Expr::Neg(minus_a), _) => -(*minus_a + b), // -a - b = -(a + b)
                (_, Expr::Neg(minus_b)) => a + *minus_b,    // a + -b = a - b
                _ => Expr::Sub(Box::new(a), Box::new(b)),
            }
        }
        Expr::Mul(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (a.clone(), b.clone()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                (Expr::Const(ZERO), _) => Expr::zero(), // 0 * b = 0
                (_, Expr::Const(ZERO)) => Expr::zero(), // a * 0 = 0
                (Expr::Const(ONE), _) => b,             // 1 * b = b
                (_, Expr::Const(ONE)) => a,             // a * 1 = a
                // (-a) * (-b) = a * b
                (Expr::Neg(minus_a), Expr::Neg(minus_b)) => *minus_a * *minus_b,
                (Expr::Neg(minus_a), _) => -(*minus_a * b), // (-a) * b = -(a * b)
                (_, Expr::Neg(minus_b)) => -(a * *minus_b), // a * (-b) = -(a * b)
                (Expr::Const(MINUS_ONE), _) => -b,          // -1 * b = -b
                (_, Expr::Const(MINUS_ONE)) => -a,          // a * -1 = -a
                _ => Expr::Mul(Box::new(a), Box::new(b)),
            }
        }
        Expr::Col(colexpr) => Expr::Col(colexpr),
        Expr::SecureCol([a, b, c, d]) => Expr::SecureCol([
            Box::new(simplify(*a)),
            Box::new(simplify(*b)),
            Box::new(simplify(*c)),
            Box::new(simplify(*d)),
        ]),
        Expr::Const(c) => Expr::Const(c),
        Expr::Param(x) => Expr::Param(x),
        Expr::Neg(a) => {
            let a = simplify(*a);
            match a {
                Expr::Const(c) => Expr::Const(-c),
                Expr::Neg(minus_a) => *minus_a,     // -(-a) = a
                Expr::Sub(a, b) => Expr::Sub(b, a), // -(a - b) = b - a
                _ => Expr::Neg(Box::new(a)),
            }
        }
        Expr::Inv(a) => {
            let a = simplify(*a);
            match a {
                Expr::Inv(inv_a) => *inv_a, // 1 / (1 / a) = a
                Expr::Const(c) => Expr::Const(c.inverse()),
                _ => Expr::Inv(Box::new(a)),
            }
        }
    }
}

/// Returns the expression
/// `value[0] * <relation>_alpha0 + value[1] * <relation>_alpha1 + ... - <relation>_z.`
fn combine_formal<R: Relation<Expr, Expr>>(relation: &R, values: &[Expr]) -> Expr {
    const Z_SUFFIX: &str = "_z";
    const ALPHA_SUFFIX: &str = "_alpha";

    let z = Expr::Param(relation.get_name().to_owned() + Z_SUFFIX);
    let alpha_powers = (0..relation.get_size())
        .map(|i| Expr::Param(relation.get_name().to_owned() + ALPHA_SUFFIX + &i.to_string()));
    values
        .iter()
        .zip(alpha_powers)
        .fold(Expr::zero(), |acc, (value, power)| {
            acc + power * value.clone()
        })
        - z
}

pub struct FormalLogupAtRow {
    pub interaction: usize,
    pub total_sum: Expr,
    pub claimed_sum: Option<(Expr, usize)>,
    pub prev_col_cumsum: Expr,
    pub cur_frac: Option<Fraction<Expr, Expr>>,
    pub is_finalized: bool,
    pub is_first: Expr,
    pub log_size: u32,
}

// P is an offset no column can reach, it signifies the variable
// offset, which is an input to the verifier.
const CLAIMED_SUM_DUMMY_OFFSET: usize = m31::P as usize;

impl FormalLogupAtRow {
    pub fn new(interaction: usize, has_partial_sum: bool, log_size: u32) -> Self {
        let total_sum_name = "total_sum".to_string();
        let claimed_sum_name = "claimed_sum".to_string();

        Self {
            interaction,
            // TODO(alont): Should these be Expr::SecureField?
            total_sum: Expr::Param(total_sum_name),
            claimed_sum: has_partial_sum
                .then_some((Expr::Param(claimed_sum_name), CLAIMED_SUM_DUMMY_OFFSET)),
            prev_col_cumsum: Expr::zero(),
            cur_frac: None,
            is_finalized: true,
            is_first: Expr::zero(),
            log_size,
        }
    }
}

/// An Evaluator that saves all constraint expressions.
pub struct ExprEvaluator {
    pub cur_var_index: usize,
    pub constraints: Vec<Expr>,
    pub logup: FormalLogupAtRow,
    pub intermediates: Vec<(String, Expr)>,
}

impl ExprEvaluator {
    #[allow(dead_code)]
    pub fn new(log_size: u32, has_partial_sum: bool) -> Self {
        Self {
            cur_var_index: Default::default(),
            constraints: Default::default(),
            logup: FormalLogupAtRow::new(INTERACTION_TRACE_IDX, has_partial_sum, log_size),
            intermediates: vec![],
        }
    }

    pub fn add_intermediate(&mut self, expr: Expr) -> Expr {
        let name = format!("intermediate{}", self.intermediates.len());
        let intermediate = Expr::Param(name.clone());
        self.intermediates.push((name, expr));
        intermediate
    }

    pub fn format_constraints(&self) -> String {
        let lets_string = self
            .intermediates
            .iter()
            .map(|(name, expr)| format!("let {} = {};", name, expr.simplify_and_format()))
            .collect::<Vec<String>>()
            .join("\n");

        let constraints_str = self
            .constraints
            .iter()
            .enumerate()
            .map(|(i, c)| format!("let constraint_{i} = ") + &c.simplify_and_format() + ";")
            .collect::<Vec<String>>()
            .join("\n\n");

        lets_string + "\n\n" + &constraints_str
    }
}

impl EvalAtRow for ExprEvaluator {
    // TODO(alont): Should there be a version of this that disallows Secure fields for F?
    type F = Expr;
    type EF = Expr;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        std::array::from_fn(|i| {
            let col = ColumnExpr {
                interaction,
                idx: self.cur_var_index,
                offset: offsets[i],
            };
            self.cur_var_index += 1;
            Expr::Col(col)
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        match Expr::one() * constraint {
            Expr::Mul(one, constraint) => {
                assert_eq!(*one, Expr::one());
                self.constraints.push(*constraint);
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        Expr::SecureCol([
            Box::new(values[0].clone()),
            Box::new(values[1].clone()),
            Box::new(values[2].clone()),
            Box::new(values[3].clone()),
        ])
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entries: &[RelationEntry<'_, Self::F, Self::EF, R>],
    ) {
        let fracs: Vec<Fraction<Self::EF, Self::EF>> = entries
            .iter()
            .map(
                |RelationEntry {
                     relation,
                     multiplicity,
                     values,
                 }| {
                    let intermediate = self.add_intermediate(combine_formal(*relation, values));
                    Fraction::new(multiplicity.clone(), intermediate)
                },
            )
            .collect();
        self.write_logup_frac(fracs.into_iter().sum());
    }

    super::logup_proxy!();
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::constraint_framework::expr::ExprEvaluator;
    use crate::constraint_framework::{relation, EvalAtRow, FrameworkEval, RelationEntry};
    use crate::core::fields::FieldExpOps;

    #[test]
    fn test_format_expr() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new(16, false));
        let expected = "let intermediate0 = (TestRelation_alpha0) * (col_1_0[0]) \
            + (TestRelation_alpha1) * (col_1_1[0]) \
            + (TestRelation_alpha2) * (col_1_2[0]) \
            - (TestRelation_z);

\
        let constraint_0 = \
            (((col_1_0[0]) * (col_1_1[0])) * (col_1_2[0])) * (1 / (col_1_0[0] + col_1_1[0]));

\
        let constraint_1 = (SecureCol(col_2_4[0], col_2_6[0], col_2_8[0], col_2_10[0]) \
            - (SecureCol(col_2_5[-1], col_2_7[-1], col_2_9[-1], col_2_11[-1]) \
                - ((col_0_3[0]) * (total_sum)))\
            ) \
            * (intermediate0) \
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
            eval.add_constraint(
                x0.clone() * x1.clone() * x2.clone() * (x0.clone() + x1.clone()).inverse(),
            );
            eval.add_to_relation(&[RelationEntry::new(
                &TestRelation::dummy(),
                E::EF::one(),
                &[x0, x1, x2],
            )]);
            eval.finalize_logup();
            eval
        }
    }
}
