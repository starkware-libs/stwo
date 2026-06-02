use core::array;
use core::ops::{Mul, Sub};

use num_traits::{One, Zero};
use std_shims::{vec, Vec};
use stwo::core::channel::Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::Fraction;

use super::EvalAtRow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogupClaim {
    Public(SecureField),
    MaskedWithPrivateCorrection(SecureField),
}

impl LogupClaim {
    #[must_use]
    pub const fn value(self) -> SecureField {
        match self {
            Self::Public(value) | Self::MaskedWithPrivateCorrection(value) => value,
        }
    }

    #[must_use]
    pub const fn uses_private_correction(self) -> bool {
        matches!(self, Self::MaskedWithPrivateCorrection(_))
    }
}

/// Evaluates constraints for batched logups.
/// These constraint enforce the sum of multiplicity_i / (z + sum_j alpha^j * x_j) = claimed_sum.
pub struct LogupAtRow<E: EvalAtRow> {
    /// The index of the interaction used for the cumulative sum columns.
    pub interaction: usize,
    pub claim: LogupClaim,
    /// The total sum of all the fractions divided by n_rows.
    pub cumsum_shift: SecureField,
    /// The evaluation of the last cumulative sum column.
    pub fracs: Vec<Fraction<E::EF, E::EF>>,
    pub is_finalized: bool,
    pub log_size: u32,
}

impl<E: EvalAtRow> Default for LogupAtRow<E> {
    fn default() -> Self {
        Self::dummy()
    }
}
impl<E: EvalAtRow> LogupAtRow<E> {
    pub fn new(interaction: usize, claimed_sum: SecureField, log_size: u32) -> Self {
        Self::new_with_claim(interaction, LogupClaim::Public(claimed_sum), log_size)
    }

    pub fn new_masked_with_private_correction(
        interaction: usize,
        masked_claim: SecureField,
        log_size: u32,
    ) -> Self {
        Self::new_with_claim(
            interaction,
            LogupClaim::MaskedWithPrivateCorrection(masked_claim),
            log_size,
        )
    }

    pub fn new_with_claim(interaction: usize, claim: LogupClaim, log_size: u32) -> Self {
        Self {
            interaction,
            claim,
            cumsum_shift: claim.value() / BaseField::from_u32_unchecked(1 << log_size),
            fracs: vec![],
            is_finalized: true,
            log_size,
        }
    }

    // TODO(alont): Remove this once unnecessary LogupAtRows are gone.
    pub fn dummy() -> Self {
        Self {
            interaction: 100,
            claim: LogupClaim::Public(SecureField::one()),
            cumsum_shift: SecureField::one(),
            fracs: vec![],
            is_finalized: true,
            log_size: 10,
        }
    }
}

/// Ensures that the LogupAtRow is finalized.
/// LogupAtRow should be finalized exactly once.
impl<E: EvalAtRow> Drop for LogupAtRow<E> {
    fn drop(&mut self) {
        assert!(self.is_finalized, "LogupAtRow was not finalized");
    }
}

/// Interaction elements for the logup protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupElements<const N: usize> {
    pub z: SecureField,
    pub alpha: SecureField,
    pub alpha_powers: [SecureField; N],
}
impl<const N: usize> LookupElements<N> {
    pub fn draw(channel: &mut impl Channel) -> Self {
        let [z, alpha] = channel.draw_secure_felts(2).try_into().unwrap();
        let mut cur = SecureField::one();
        let alpha_powers = array::from_fn(|_| {
            let res = cur;
            cur *= alpha;
            res
        });
        Self {
            z,
            alpha,
            alpha_powers,
        }
    }
    pub fn combine<F: Clone, EF>(&self, values: &[F]) -> EF
    where
        EF: Clone + Zero + From<F> + From<SecureField> + Mul<F, Output = EF> + Sub<EF, Output = EF>,
    {
        assert!(
            self.alpha_powers.len() >= values.len(),
            "Not enough alpha powers to combine values"
        );
        values
            .iter()
            .zip(self.alpha_powers)
            .fold(EF::zero(), |acc, (value, power)| {
                acc + EF::from(power) * value.clone()
            })
            - EF::from(self.z)
    }

    pub fn dummy() -> Self {
        let z = SecureField::from_u32_unchecked(1, 2, 3, 4);
        let alpha = SecureField::from_u32_unchecked(4, 3, 2, 1);
        let mut cur = SecureField::one();
        let alpha_powers = array::from_fn(|_| {
            let res = cur;
            cur *= alpha;
            res
        });
        Self {
            z,
            alpha,
            alpha_powers,
        }
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::FieldExpOps;

    use super::LookupElements;

    #[test]
    fn test_lookup_elements_combine() {
        let mut channel = Blake2sChannel::default();
        let lookup_elements = LookupElements::<3>::draw(&mut channel);
        let values = [
            BaseField::from_u32_unchecked(123),
            BaseField::from_u32_unchecked(456),
            BaseField::from_u32_unchecked(789),
        ];

        assert_eq!(
            lookup_elements.combine::<BaseField, SecureField>(&values),
            BaseField::from_u32_unchecked(123)
                + BaseField::from_u32_unchecked(456) * lookup_elements.alpha
                + BaseField::from_u32_unchecked(789) * lookup_elements.alpha.pow(2)
                - lookup_elements.z
        );
    }
}
