/// ! This module contains helpers to express and use constraints for components.
mod assert;
mod component;
mod cpu_domain;
pub mod expr;
mod info;
pub mod logup;
mod point;
pub mod preprocessed_columns;
pub mod relation_tracker;
mod simd_domain;

use std::array;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

pub use assert::{assert_constraints, AssertEvaluator};
pub use component::{FrameworkComponent, FrameworkEval, TraceLocationAllocator};
pub use info::InfoEvaluator;
use num_traits::{One, Zero};
pub use point::PointEvaluator;
use preprocessed_columns::PreProcessedColumnId;
pub use simd_domain::SimdDomainEvaluator;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;

pub const PREPROCESSED_TRACE_IDX: usize = 0;
pub const ORIGINAL_TRACE_IDX: usize = 1;
pub const INTERACTION_TRACE_IDX: usize = 2;

/// A vector that describes the batching of logup entries.
/// Each vector member corresponds to a logup entry, and contains the batch number to which the
/// entry should be added.
/// Note that the batch numbers should be consecutive and start from 0, and that the vector's
/// length should be equal to the number of logup entries.
type Batching = Vec<usize>;

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

    fn get_preprocessed_column(&mut self, _column: PreProcessedColumnId) -> Self::F {
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
        Self::EF: Mul<G, Output = Self::EF> + From<G>;

    /// Adds an intermediate value in the base field to the component and returns its value.
    /// Does nothing by default.
    fn add_intermediate(&mut self, val: Self::F) -> Self::F {
        val
    }

    /// Adds an intermediate value in the extension field to the component and returns its value.
    /// Does nothing by default.
    fn add_extension_intermediate(&mut self, val: Self::EF) -> Self::EF {
        val
    }

    /// Combines 4 base field values into a single extension field value.
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF;

    /// Adds `entry.values` to `entry.relation` with `entry.multiplicity` for all 'entry' in
    /// 'entries', batched together.
    /// Constraint degree increases with number of batched constraints as the denominators are
    /// multiplied.
    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let frac = Fraction::new(
            entry.multiplicity.clone(),
            entry.relation.combine(entry.values),
        );
        self.write_logup_frac(frac);
    }

    // TODO(alont): Remove these once LogupAtRow is no longer used.
    fn write_logup_frac(&mut self, _fraction: Fraction<Self::EF, Self::EF>) {
        unimplemented!()
    }
    fn finalize_logup_batched(&mut self, _batching: &Batching) {
        unimplemented!()
    }

    fn finalize_logup(&mut self) {
        unimplemented!();
    }

    fn finalize_logup_in_pairs(&mut self) {
        unimplemented!();
    }
}

/// Default implementation for evaluators that have an element called "logup" that works like a
/// LogupAtRow, where the logup functionality can be proxied.
/// TODO(alont): Remove once LogupAtRow is no longer used.
macro_rules! logup_proxy {
    () => {
        fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
            if self.logup.fracs.is_empty() {
                self.logup.is_finalized = false;
            }
            self.logup.fracs.push(fraction.clone());
        }

        /// Finalize the logup by adding the constraints for the fractions, batched by
        /// the given `batching`.
        /// `batching` should contain the batch into which every logup entry should be inserted.
        fn finalize_logup_batched(&mut self, batching: &crate::constraint_framework::Batching) {
            assert!(!self.logup.is_finalized, "LogupAtRow was already finalized");
            assert_eq!(
                batching.len(),
                self.logup.fracs.len(),
                "Batching must be of the same length as the number of entries"
            );

            let last_batch = *batching.iter().max().unwrap();

            let mut fracs_by_batch =
                std::collections::HashMap::<usize, Vec<Fraction<Self::EF, Self::EF>>>::new();

            for (batch, frac) in batching.iter().zip(self.logup.fracs.iter()) {
                fracs_by_batch
                    .entry(*batch)
                    .or_insert_with(Vec::new)
                    .push(frac.clone());
            }

            let keys_set: std::collections::HashSet<_> = fracs_by_batch.keys().cloned().collect();
            let all_batches_set: std::collections::HashSet<_> = (0..last_batch + 1).collect();

            assert_eq!(
                keys_set, all_batches_set,
                "Batching must contain all consecutive batches"
            );

            let mut prev_col_cumsum = <Self::EF as num_traits::Zero>::zero();

            // All batches except the last are cumulatively summed in new interaction columns.
            for batch_id in (0..last_batch) {
                let cur_frac: Fraction<_, _> = fracs_by_batch[&batch_id].iter().cloned().sum();
                let [cur_cumsum] =
                    self.next_extension_interaction_mask(self.logup.interaction, [0]);
                let diff = cur_cumsum.clone() - prev_col_cumsum.clone();
                prev_col_cumsum = cur_cumsum;
                self.add_constraint(diff * cur_frac.denominator - cur_frac.numerator);
            }

            let frac: Fraction<_, _> = fracs_by_batch[&last_batch].clone().into_iter().sum();
            let [prev_row_cumsum, cur_cumsum] =
                self.next_extension_interaction_mask(self.logup.interaction, [-1, 0]);

            let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum.clone();
            // Instead of checking diff = num / denom, check diff = num / denom - cumsum_shift.
            // This makes (num / denom - cumsum_shift) have sum zero, which makes the constraint
            // uniform - apply on all rows.
            let fixed_diff = diff + self.logup.cumsum_shift.clone();

            self.add_constraint(fixed_diff * frac.denominator - frac.numerator);

            self.logup.is_finalized = true;
        }

        /// Finalizes the row's logup in the default way. Currently, this means no batching.
        fn finalize_logup(&mut self) {
            let batches = (0..self.logup.fracs.len()).collect();
            self.finalize_logup_batched(&batches)
        }

        /// Finalizes the row's logup, batched in pairs.
        /// TODO(alont) Remove this once a better batching mechanism is implemented.
        fn finalize_logup_in_pairs(&mut self) {
            let batches = (0..self.logup.fracs.len()).map(|n| n / 2).collect();
            self.finalize_logup_batched(&batches)
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
    pub const fn new(relation: &'a R, multiplicity: EF, values: &'a [F]) -> Self {
        Self {
            relation,
            multiplicity,
            values,
        }
    }
}

#[macro_export]
macro_rules! relation {
    ($name:tt, $size:tt) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name($crate::constraint_framework::logup::LookupElements<$size>);

        #[allow(dead_code)]
        impl $name {
            pub fn dummy() -> Self {
                Self($crate::constraint_framework::logup::LookupElements::dummy())
            }
            pub fn draw(channel: &mut impl $crate::core::channel::Channel) -> Self {
                Self($crate::constraint_framework::logup::LookupElements::draw(
                    channel,
                ))
            }
        }

        impl<F: Clone, EF: $crate::constraint_framework::RelationEFTraitBound<F>>
            $crate::constraint_framework::Relation<F, EF> for $name
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
