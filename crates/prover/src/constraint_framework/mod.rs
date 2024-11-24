/// ! This module contains helpers to express and use constraints for components.
mod assert;
mod component;
mod cpu_domain;
pub mod expr;
mod info;
pub mod logup;
mod point;
pub mod preprocessed_columns;
mod simd_domain;

use std::array;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

pub use assert::{assert_constraints, AssertEvaluator};
pub use component::{FrameworkComponent, FrameworkEval, TraceLocationAllocator};
pub use info::InfoEvaluator;
use num_traits::{One, Zero};
pub use point::PointEvaluator;
use preprocessed_columns::PreprocessedColumn;
pub use simd_domain::SimdDomainEvaluator;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;

pub const PREPROCESSED_TRACE_IDX: usize = 0;
pub const ORIGINAL_TRACE_IDX: usize = 1;
pub const INTERACTION_TRACE_IDX: usize = 2;

/// A trait for evaluating expressions at some point or row.
pub trait EvalAtRow {
    // TODO(Ohad): Use a better trait for these, like 'Algebra' or something.
    /// The field type holding values of columns for the component. These are the inputs to the
    /// constraints. It might be [BaseField] packed types, or even [SecureField], when evaluating
    /// the columns out of domain.
    type F: FieldExpOps
        + Clone
        + Debug
        + Zero
        + Neg<Output = Self::F>
        + AddAssign
        + AddAssign<BaseField>
        + Add<Self::F, Output = Self::F>
        + Sub<Self::F, Output = Self::F>
        + Mul<BaseField, Output = Self::F>
        + Add<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Neg<Output = Self::F>
        + From<BaseField>;

    /// A field type representing the closure of `F` with multiplying by [SecureField]. Constraints
    /// usually get multiplied by [SecureField] values for security.
    type EF: One
        + Clone
        + Debug
        + Zero
        + From<Self::F>
        + Neg<Output = Self::EF>
        + AddAssign
        + Add<BaseField, Output = Self::EF>
        + Mul<BaseField, Output = Self::EF>
        + Add<SecureField, Output = Self::EF>
        + Sub<SecureField, Output = Self::EF>
        + Mul<SecureField, Output = Self::EF>
        + Add<Self::F, Output = Self::EF>
        + Mul<Self::F, Output = Self::EF>
        + Sub<Self::EF, Output = Self::EF>
        + Mul<Self::EF, Output = Self::EF>
        + From<SecureField>
        + From<Self::F>;

    /// Returns the next mask value for the first interaction at offset 0.
    fn next_trace_mask(&mut self) -> Self::F {
        let [mask_item] = self.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        mask_item
    }

    fn get_preprocessed_column(&mut self, _column: PreprocessedColumn) -> Self::F {
        let [mask_item] = self.next_interaction_mask(PREPROCESSED_TRACE_IDX, [0]);
        mask_item
    }

    /// Returns the mask values of the given offsets for the next column in the interaction.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N];

    /// Returns the extension mask values of the given offsets for the next extension degree many
    /// columns in the interaction.
    fn next_extension_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::EF; N] {
        let mut res_col_major =
            array::from_fn(|_| self.next_interaction_mask(interaction, offsets).into_iter());
        array::from_fn(|_| {
            Self::combine_ef(res_col_major.each_mut().map(|iter| iter.next().unwrap()))
        })
    }

    /// Adds a constraint to the component.
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>;

    /// Combines 4 base field values into a single extension field value.
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF;

    /// Adds `entry.values` to `entry.relation` with `entry.multiplicity` for all 'entry' in
    /// 'entries', batched together.
    /// Constraint degree increases with number of batched constraints as the denominators are
    /// multiplied.
    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entries: &[RelationEntry<'_, Self::F, Self::EF, R>],
    ) {
        let fracs = entries.iter().map(
            |RelationEntry {
                 relation,
                 multiplicity,
                 values,
             }| { Fraction::new(multiplicity.clone(), relation.combine(values)) },
        );
        self.write_logup_frac(fracs.sum());
    }

    // TODO(alont): Remove these once LogupAtRow is no longer used.
    fn write_logup_frac(&mut self, _fraction: Fraction<Self::EF, Self::EF>) {
        unimplemented!()
    }
    fn finalize_logup(&mut self) {
        unimplemented!()
    }
}

/// Default implementation for evaluators that have an element called "logup" that works like a
/// LogupAtRow, where the logup functionality can be proxied.
/// TODO(alont): Remove once LogupAtRow is no longer used.
macro_rules! logup_proxy {
    () => {
        fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
            // Add a constraint that num / denom = diff.
            if let Some(cur_frac) = self.logup.cur_frac.clone() {
                let [cur_cumsum] =
                    self.next_extension_interaction_mask(self.logup.interaction, [0]);
                let diff = cur_cumsum.clone() - self.logup.prev_col_cumsum.clone();
                self.logup.prev_col_cumsum = cur_cumsum;
                self.add_constraint(diff * cur_frac.denominator - cur_frac.numerator);
            } else {
                self.logup.is_first = self.get_preprocessed_column(
                    super::preprocessed_columns::PreprocessedColumn::IsFirst(self.logup.log_size),
                );
                self.logup.is_finalized = false;
            }
            self.logup.cur_frac = Some(fraction);
        }

        fn finalize_logup(&mut self) {
            assert!(!self.logup.is_finalized, "LogupAtRow was already finalized");

            let frac = self.logup.cur_frac.clone().unwrap();

            // TODO(ShaharS): remove `claimed_row_index` interaction value and get the shifted
            // offset from the is_first column when constant columns are supported.
            let (cur_cumsum, prev_row_cumsum) = match self.logup.claimed_sum {
                Some((claimed_sum, claimed_row_index)) => {
                    let [cur_cumsum, prev_row_cumsum, claimed_cumsum] = self
                        .next_extension_interaction_mask(
                            self.logup.interaction,
                            [0, -1, claimed_row_index as isize],
                        );

                    // Constrain that the claimed_sum in case that it is not equal to the total_sum.
                    self.add_constraint(
                        (claimed_cumsum - claimed_sum) * self.logup.is_first.clone(),
                    );
                    (cur_cumsum, prev_row_cumsum)
                }
                None => {
                    let [cur_cumsum, prev_row_cumsum] =
                        self.next_extension_interaction_mask(self.logup.interaction, [0, -1]);
                    (cur_cumsum, prev_row_cumsum)
                }
            };
            // Fix `prev_row_cumsum` by subtracting `total_sum` if this is the first row.
            let fixed_prev_row_cumsum =
                prev_row_cumsum - self.logup.is_first.clone() * self.logup.total_sum;
            let diff = cur_cumsum - fixed_prev_row_cumsum - self.logup.prev_col_cumsum.clone();

            self.add_constraint(diff * frac.denominator - frac.numerator);

            self.logup.is_finalized = true;
        }
    };
}
pub(crate) use logup_proxy;

pub trait RelationEFTraitBound<F: Clone>:
    Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = Self> + Sub<Self, Output = Self>
{
}

impl<F, EF> RelationEFTraitBound<F> for EF
where
    F: Clone,
    EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
{
}

/// A trait for defining a logup relation type.
pub trait Relation<F: Clone, EF: RelationEFTraitBound<F>>: Sized {
    fn combine(&self, values: &[F]) -> EF;

    fn get_name(&self) -> &str;
    fn get_size(&self) -> usize;
}

/// A struct representing a relation entry.
/// `relation` is the relation into which elements are entered.
/// `multiplicity` is the multiplicity of the elements.
///     A positive multiplicity is used to signify a "use", while a negative multiplicity
///     signifies a "yield".
/// `values` are elements in the base field that are entered into the relation.
pub struct RelationEntry<'a, F: Clone, EF: RelationEFTraitBound<F>, R: Relation<F, EF>> {
    relation: &'a R,
    multiplicity: EF,
    values: &'a [F],
}
impl<'a, F: Clone, EF: RelationEFTraitBound<F>, R: Relation<F, EF>> RelationEntry<'a, F, EF, R> {
    pub fn new(relation: &'a R, multiplicity: EF, values: &'a [F]) -> Self {
        Self {
            relation,
            multiplicity,
            values,
        }
    }
}

macro_rules! relation {
    ($name:tt, $size:tt) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(crate::constraint_framework::logup::LookupElements<$size>);

        #[allow(dead_code)]
        impl $name {
            pub fn dummy() -> Self {
                Self(crate::constraint_framework::logup::LookupElements::dummy())
            }
            pub fn draw(channel: &mut impl crate::core::channel::Channel) -> Self {
                Self(crate::constraint_framework::logup::LookupElements::draw(
                    channel,
                ))
            }
        }

        impl<F: Clone, EF: crate::constraint_framework::RelationEFTraitBound<F>>
            crate::constraint_framework::Relation<F, EF> for $name
        {
            fn combine(&self, values: &[F]) -> EF {
                values
                    .iter()
                    .zip(self.0.alpha_powers)
                    .fold(EF::zero(), |acc, (value, power)| {
                        acc + EF::from(power) * value.clone()
                    })
                    - self.0.z.into()
            }

            fn get_name(&self) -> &str {
                stringify!($name)
            }

            fn get_size(&self) -> usize {
                $size
            }
        }
    };
}
pub(crate) use relation;
