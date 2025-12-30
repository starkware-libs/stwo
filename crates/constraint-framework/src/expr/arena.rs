//! Arena-based expression storage for Copy-able expression handles.
//!
//! This module provides an arena allocator for expression nodes, allowing expression
//! handles (`BaseExpr`, `ExtExpr`) to be `Copy` instead of requiring `Clone`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use num_traits::{One, Zero};
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, QM31};
use stwo::core::fields::FieldExpOps;

use super::ColumnExpr;

/// Interned string handle - allows Param to become Copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(u32);

/// Handle to a BaseExpr node in the arena - this is the Copy-able type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BaseExpr(u32);

/// Handle to an ExtExpr node in the arena - this is the Copy-able type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtExpr(u32);

/// Internal node representation for BaseExpr (stored in arena).
#[derive(Clone, Debug, PartialEq)]
pub enum BaseExprNode {
    Col(ColumnExpr),
    Const(BaseField),
    Param(StringId),
    Add(BaseExpr, BaseExpr),
    Sub(BaseExpr, BaseExpr),
    Mul(BaseExpr, BaseExpr),
    Neg(BaseExpr),
    Inv(BaseExpr),
}

/// Internal node representation for ExtExpr (stored in arena).
#[derive(Clone, Debug, PartialEq)]
pub enum ExtExprNode {
    SecureCol([BaseExpr; 4]),
    Const(SecureField),
    Param(StringId),
    Add(ExtExpr, ExtExpr),
    Sub(ExtExpr, ExtExpr),
    Mul(ExtExpr, ExtExpr),
    Neg(ExtExpr),
}

/// Central arena holding all expression nodes and interned strings.
#[derive(Default, Debug)]
pub struct ExprArena {
    base_nodes: Vec<BaseExprNode>,
    ext_nodes: Vec<ExtExprNode>,
    strings: Vec<String>,
    string_map: HashMap<String, StringId>,
}

impl ExprArena {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning its ID. Deduplicates identical strings.
    pub fn intern_string(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.string_map.get(s) {
            return id;
        }
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.string_map.insert(s.to_string(), id);
        id
    }

    /// Get the string for a StringId.
    pub fn get_string(&self, id: StringId) -> &str {
        &self.strings[id.0 as usize]
    }

    /// Allocate a BaseExpr node.
    pub fn alloc_base(&mut self, node: BaseExprNode) -> BaseExpr {
        let id = BaseExpr(self.base_nodes.len() as u32);
        self.base_nodes.push(node);
        id
    }

    /// Get a BaseExpr node by ID.
    pub fn get_base(&self, id: BaseExpr) -> &BaseExprNode {
        &self.base_nodes[id.0 as usize]
    }

    /// Allocate an ExtExpr node.
    pub fn alloc_ext(&mut self, node: ExtExprNode) -> ExtExpr {
        let id = ExtExpr(self.ext_nodes.len() as u32);
        self.ext_nodes.push(node);
        id
    }

    /// Get an ExtExpr node by ID.
    pub fn get_ext(&self, id: ExtExpr) -> &ExtExprNode {
        &self.ext_nodes[id.0 as usize]
    }

    // Convenience constructors for BaseExpr
    pub fn base_col(&mut self, col: ColumnExpr) -> BaseExpr {
        self.alloc_base(BaseExprNode::Col(col))
    }

    pub fn base_const(&mut self, val: BaseField) -> BaseExpr {
        self.alloc_base(BaseExprNode::Const(val))
    }

    pub fn base_param(&mut self, name: &str) -> BaseExpr {
        let id = self.intern_string(name);
        self.alloc_base(BaseExprNode::Param(id))
    }

    pub fn base_add(&mut self, a: BaseExpr, b: BaseExpr) -> BaseExpr {
        self.alloc_base(BaseExprNode::Add(a, b))
    }

    pub fn base_sub(&mut self, a: BaseExpr, b: BaseExpr) -> BaseExpr {
        self.alloc_base(BaseExprNode::Sub(a, b))
    }

    pub fn base_mul(&mut self, a: BaseExpr, b: BaseExpr) -> BaseExpr {
        self.alloc_base(BaseExprNode::Mul(a, b))
    }

    pub fn base_neg(&mut self, a: BaseExpr) -> BaseExpr {
        self.alloc_base(BaseExprNode::Neg(a))
    }

    pub fn base_inv(&mut self, a: BaseExpr) -> BaseExpr {
        self.alloc_base(BaseExprNode::Inv(a))
    }

    pub fn base_zero(&mut self) -> BaseExpr {
        self.base_const(BaseField::zero())
    }

    pub fn base_one(&mut self) -> BaseExpr {
        self.base_const(BaseField::one())
    }

    // Convenience constructors for ExtExpr
    pub fn ext_secure_col(&mut self, cols: [BaseExpr; 4]) -> ExtExpr {
        self.alloc_ext(ExtExprNode::SecureCol(cols))
    }

    pub fn ext_const(&mut self, val: SecureField) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Const(val))
    }

    pub fn ext_param(&mut self, name: &str) -> ExtExpr {
        let id = self.intern_string(name);
        self.alloc_ext(ExtExprNode::Param(id))
    }

    pub fn ext_add(&mut self, a: ExtExpr, b: ExtExpr) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Add(a, b))
    }

    pub fn ext_sub(&mut self, a: ExtExpr, b: ExtExpr) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Sub(a, b))
    }

    pub fn ext_mul(&mut self, a: ExtExpr, b: ExtExpr) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Mul(a, b))
    }

    pub fn ext_neg(&mut self, a: ExtExpr) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Neg(a))
    }

    pub fn ext_zero(&mut self) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Const(SecureField::zero()))
    }

    pub fn ext_one(&mut self) -> ExtExpr {
        self.alloc_ext(ExtExprNode::Const(SecureField::one()))
    }

    /// Convert a BaseExpr to ExtExpr (embed in first component).
    pub fn base_to_ext(&mut self, base: BaseExpr) -> ExtExpr {
        let zero = self.base_zero();
        self.ext_secure_col([base, zero, zero, zero])
    }

    /// Convert a BaseField to ExtExpr.
    pub fn base_field_to_ext(&mut self, val: BaseField) -> ExtExpr {
        let base = self.base_const(val);
        self.base_to_ext(base)
    }

    /// Convert a SecureField to ExtExpr.
    pub fn secure_field_to_ext(&mut self, QM31(CM31(a, b), CM31(c, d)): SecureField) -> ExtExpr {
        let a = self.base_const(a);
        let b = self.base_const(b);
        let c = self.base_const(c);
        let d = self.base_const(d);
        self.ext_secure_col([a, b, c, d])
    }
}

// Thread-local arena for operator ergonomics
thread_local! {
    static EXPR_ARENA: RefCell<Option<ExprArena>> = const { RefCell::new(None) };
}

/// Initialize the thread-local arena. Must be called before using expression operators.
pub fn init_arena() {
    EXPR_ARENA.with(|arena| {
        *arena.borrow_mut() = Some(ExprArena::new());
    });
}

/// Clear the thread-local arena.
pub fn clear_arena() {
    EXPR_ARENA.with(|arena| {
        *arena.borrow_mut() = None;
    });
}

/// Execute a closure with access to the arena, initializing if needed.
pub fn with_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut ExprArena) -> R,
{
    EXPR_ARENA.with(|arena| {
        let mut borrowed = arena.borrow_mut();
        if borrowed.is_none() {
            *borrowed = Some(ExprArena::new());
        }
        f(borrowed.as_mut().unwrap())
    })
}

/// Get a reference to the arena for read-only operations.
pub fn with_arena_ref<F, R>(f: F) -> R
where
    F: FnOnce(&ExprArena) -> R,
{
    EXPR_ARENA.with(|arena| {
        let borrowed = arena.borrow();
        f(borrowed.as_ref().expect("Arena not initialized"))
    })
}

/// Take ownership of the arena (for final processing).
pub fn take_arena() -> Option<ExprArena> {
    EXPR_ARENA.with(|arena| arena.borrow_mut().take())
}

/// Replace the arena with a new one.
pub fn set_arena(new_arena: ExprArena) {
    EXPR_ARENA.with(|arena| {
        *arena.borrow_mut() = Some(new_arena);
    });
}

// Helper functions to allocate nodes using the thread-local arena
fn alloc_base(node: BaseExprNode) -> BaseExpr {
    with_arena(|arena| arena.alloc_base(node))
}

fn alloc_ext(node: ExtExprNode) -> ExtExpr {
    with_arena(|arena| arena.alloc_ext(node))
}

// ============================================================================
// Implement operators for BaseExpr
// ============================================================================

impl Add for BaseExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        alloc_base(BaseExprNode::Add(self, rhs))
    }
}

impl Sub for BaseExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        alloc_base(BaseExprNode::Sub(self, rhs))
    }
}

impl Mul for BaseExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        alloc_base(BaseExprNode::Mul(self, rhs))
    }
}

impl AddAssign for BaseExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for BaseExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for BaseExpr {
    type Output = Self;
    fn neg(self) -> Self {
        alloc_base(BaseExprNode::Neg(self))
    }
}

impl Zero for BaseExpr {
    fn zero() -> Self {
        with_arena(|arena| arena.base_zero())
    }
    fn is_zero(&self) -> bool {
        panic!("Can't check if an expression is zero.");
    }
}

impl One for BaseExpr {
    fn one() -> Self {
        with_arena(|arena| arena.base_one())
    }
}

impl FieldExpOps for BaseExpr {
    fn inverse(&self) -> Self {
        alloc_base(BaseExprNode::Inv(*self))
    }
}

impl From<BaseField> for BaseExpr {
    fn from(val: BaseField) -> Self {
        with_arena(|arena| arena.base_const(val))
    }
}

impl Add<BaseField> for BaseExpr {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        self + BaseExpr::from(rhs)
    }
}

impl AddAssign<BaseField> for BaseExpr {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = *self + BaseExpr::from(rhs);
    }
}

impl Mul<BaseField> for BaseExpr {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self * BaseExpr::from(rhs)
    }
}

impl Mul<SecureField> for BaseExpr {
    type Output = ExtExpr;
    fn mul(self, rhs: SecureField) -> ExtExpr {
        ExtExpr::from(self) * ExtExpr::from(rhs)
    }
}

impl Add<SecureField> for BaseExpr {
    type Output = ExtExpr;
    fn add(self, rhs: SecureField) -> ExtExpr {
        ExtExpr::from(self) + ExtExpr::from(rhs)
    }
}

impl Sub<SecureField> for BaseExpr {
    type Output = ExtExpr;
    fn sub(self, rhs: SecureField) -> ExtExpr {
        ExtExpr::from(self) - ExtExpr::from(rhs)
    }
}

// ============================================================================
// Implement operators for ExtExpr
// ============================================================================

impl Add for ExtExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        alloc_ext(ExtExprNode::Add(self, rhs))
    }
}

impl Sub for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        alloc_ext(ExtExprNode::Sub(self, rhs))
    }
}

impl Mul for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        alloc_ext(ExtExprNode::Mul(self, rhs))
    }
}

impl AddAssign for ExtExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for ExtExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for ExtExpr {
    type Output = Self;
    fn neg(self) -> Self {
        alloc_ext(ExtExprNode::Neg(self))
    }
}

impl Zero for ExtExpr {
    fn zero() -> Self {
        with_arena(|arena| arena.ext_zero())
    }
    fn is_zero(&self) -> bool {
        panic!("Can't check if an expression is zero.");
    }
}

impl One for ExtExpr {
    fn one() -> Self {
        with_arena(|arena| arena.ext_one())
    }
}

impl From<BaseField> for ExtExpr {
    fn from(val: BaseField) -> Self {
        with_arena(|arena| arena.base_field_to_ext(val))
    }
}

impl From<SecureField> for ExtExpr {
    fn from(val: SecureField) -> Self {
        with_arena(|arena| arena.secure_field_to_ext(val))
    }
}

impl From<BaseExpr> for ExtExpr {
    fn from(expr: BaseExpr) -> Self {
        with_arena(|arena| arena.base_to_ext(expr))
    }
}

impl Add<BaseField> for ExtExpr {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        self + ExtExpr::from(rhs)
    }
}

impl AddAssign<BaseField> for ExtExpr {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = *self + ExtExpr::from(rhs);
    }
}

impl Mul<BaseField> for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self * ExtExpr::from(rhs)
    }
}

impl Mul<SecureField> for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: SecureField) -> Self {
        self * ExtExpr::from(rhs)
    }
}

impl Add<SecureField> for ExtExpr {
    type Output = Self;
    fn add(self, rhs: SecureField) -> Self {
        self + ExtExpr::from(rhs)
    }
}

impl Sub<SecureField> for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: SecureField) -> Self {
        self - ExtExpr::from(rhs)
    }
}

impl Add<BaseExpr> for ExtExpr {
    type Output = Self;
    fn add(self, rhs: BaseExpr) -> Self {
        self + ExtExpr::from(rhs)
    }
}

impl Mul<BaseExpr> for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: BaseExpr) -> Self {
        self * ExtExpr::from(rhs)
    }
}

impl Mul<ExtExpr> for BaseExpr {
    type Output = ExtExpr;
    fn mul(self, rhs: ExtExpr) -> ExtExpr {
        rhs * self
    }
}

impl Sub<BaseExpr> for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: BaseExpr) -> Self {
        self - ExtExpr::from(rhs)
    }
}

// ============================================================================
// Helper methods on BaseExpr and ExtExpr for accessing nodes
// ============================================================================

impl BaseExpr {
    /// Create a column expression.
    pub fn col(col: ColumnExpr) -> Self {
        with_arena(|arena| arena.base_col(col))
    }

    /// Create a parameter expression.
    pub fn param(name: &str) -> Self {
        with_arena(|arena| arena.base_param(name))
    }

    /// Get the node for this expression.
    pub fn node(&self) -> BaseExprNode {
        with_arena_ref(|arena| arena.get_base(*self).clone())
    }

    /// Get a reference to the node (requires arena reference).
    pub fn node_ref<'a>(&self, arena: &'a ExprArena) -> &'a BaseExprNode {
        arena.get_base(*self)
    }
}

impl ExtExpr {
    /// Create a secure column expression.
    pub fn secure_col(cols: [BaseExpr; 4]) -> Self {
        with_arena(|arena| arena.ext_secure_col(cols))
    }

    /// Create a parameter expression.
    pub fn param(name: &str) -> Self {
        with_arena(|arena| arena.ext_param(name))
    }

    /// Get the node for this expression.
    pub fn node(&self) -> ExtExprNode {
        with_arena_ref(|arena| arena.get_ext(*self).clone())
    }

    /// Get a reference to the node (requires arena reference).
    pub fn node_ref<'a>(&self, arena: &'a ExprArena) -> &'a ExtExprNode {
        arena.get_ext(*self)
    }
}

impl StringId {
    /// Get the string for this ID.
    pub fn as_str(&self) -> String {
        with_arena_ref(|arena| arena.get_string(*self).to_string())
    }

    /// Get the string with an arena reference.
    pub fn as_str_ref<'a>(&self, arena: &'a ExprArena) -> &'a str {
        arena.get_string(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        init_arena();

        let a = BaseExpr::from(BaseField::from(5u32));
        let b = BaseExpr::from(BaseField::from(3u32));
        let c = a + b;

        match c.node() {
            BaseExprNode::Add(left, right) => {
                assert!(matches!(left.node(), BaseExprNode::Const(_)));
                assert!(matches!(right.node(), BaseExprNode::Const(_)));
            }
            _ => panic!("Expected Add node"),
        }

        clear_arena();
    }

    #[test]
    fn test_copy_semantics() {
        init_arena();

        let a = BaseExpr::from(BaseField::from(5u32));
        let b = a; // Copy, not move
        let c = a + b; // Both a and b are still valid

        assert!(matches!(c.node(), BaseExprNode::Add(_, _)));

        clear_arena();
    }

    #[test]
    fn test_string_interning() {
        init_arena();

        let a = BaseExpr::param("test_param");
        let b = BaseExpr::param("test_param");

        // Both should reference the same interned string
        match (a.node(), b.node()) {
            (BaseExprNode::Param(id_a), BaseExprNode::Param(id_b)) => {
                assert_eq!(id_a, id_b);
            }
            _ => panic!("Expected Param nodes"),
        }

        clear_arena();
    }
}
