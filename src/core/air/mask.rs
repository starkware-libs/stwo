use std::collections::HashSet;
use std::vec;

use itertools::Itertools;

use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CanonicCoset;
use crate::core::ColumnVec;

/// Mask holds a vector with an entry for each column.
/// Each entry holds a list of mask items, which are the offsets of the mask at that column.
type Mask = ColumnVec<Vec<usize>>;

/// Returns the same point for each mask item.
/// Should be used where all the mask items has no shift from the constraint point.
pub fn fixed_mask_points(
    mask: &Mask,
    point: CirclePoint<SecureField>,
) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
    assert_eq!(
        mask.iter()
            .flat_map(|mask_entry| mask_entry.iter().collect::<HashSet<_>>())
            .collect::<HashSet<&usize>>()
            .into_iter()
            .collect_vec(),
        vec![&0]
    );
    mask.iter()
        .map(|mask_entry| mask_entry.iter().map(|_| point).collect())
        .collect()
}

/// For each mask item returns the point shifted by the domain initial point of the column.
/// Should be used where the mask items are shifted from the constraint point.
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

#[cfg(test)]
mod tests {
    use crate::core::air::mask::{fixed_mask_points, shifted_mask_points};
    use crate::core::circle::CirclePoint;
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_mask_fixed_points() {
        let mask = vec![vec![0], vec![0]];
        let constraint_point = CirclePoint::get_point(1234);

        let points = fixed_mask_points(&mask, constraint_point);

        assert_eq!(points.len(), 2);
        assert_eq!(points[0].len(), 1);
        assert_eq!(points[1].len(), 1);
        assert_eq!(points[0][0], constraint_point);
        assert_eq!(points[1][0], constraint_point);
    }

    #[test]
    fn test_mask_shifted_points() {
        let mask = vec![vec![0, 1], vec![0, 1, 2]];
        let constraint_point = CirclePoint::get_point(1234);
        let domains = (0..mask.len() as u32)
            .map(|i| CanonicCoset::new(7 + i))
            .collect::<Vec<_>>();

        let points = shifted_mask_points(&mask, &domains, constraint_point);

        assert_eq!(points.len(), 2);
        assert_eq!(points[0].len(), 2);
        assert_eq!(points[1].len(), 3);
        assert_eq!(points[0][0], constraint_point + domains[0].at(0).into_ef());
        assert_eq!(points[0][1], constraint_point + domains[0].at(1).into_ef());
        assert_eq!(points[1][0], constraint_point + domains[1].at(0).into_ef());
        assert_eq!(points[1][1], constraint_point + domains[1].at(1).into_ef());
        assert_eq!(points[1][2], constraint_point + domains[1].at(2).into_ef());
    }
}
