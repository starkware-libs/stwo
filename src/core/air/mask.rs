use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CanonicCoset;
use crate::core::ColumnVec;

/// Mask holds a vector with an entry for each column.
/// Each entry holds a list of mask items, which are the offsets of the mask at that column.
type Mask = ColumnVec<Vec<usize>>;

/// Returns the same point for each mask item.
/// Should be used where all mask items has the same offset.
pub fn fixed_mask_points(
    mask: &Mask,
    point: CirclePoint<SecureField>,
) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
    mask.iter()
        .map(|mask_entry| mask_entry.iter().map(|_| point).collect())
        .collect()
}

/// For each mask item returns the point shifted by the domain initial point of the column.
pub fn shifted_mask_points(
    mask: &Mask,
    domains: &[CanonicCoset],
    point: CirclePoint<SecureField>,
) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
    mask.iter()
        .zip(domains.iter())
        .map(|(mask_entry, domain)| {
            mask_entry
                .iter()
                .map(|mask_item| point + domain.at(*mask_item).into_ef())
                .collect()
        })
        .collect()
}
