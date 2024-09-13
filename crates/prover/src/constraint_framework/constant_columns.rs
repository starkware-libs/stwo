use std::collections::HashMap;

use num_traits::One;

use crate::core::air::CONST_INTERACTION;
use crate::core::backend::{Backend, Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::pcs::TreeLocation;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use crate::core::vcs::blake2_hash::Blake2sHash;

/// Generates a column with a single one at the first position, and zeros elsewhere.
pub fn gen_is_first<B: Backend>(log_size: u32) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);
    col.set(0, BaseField::one());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

/// Generates a column with `1` at every `2^log_step` positions, `0` elsewhere, shifted by offset.
// TODO(andrew): Consider optimizing. Is a quotients of two coset_vanishing (use succinct rep for
// verifier).
pub fn gen_is_step_with_offset<B: Backend>(
    log_size: u32,
    log_step: u32,
    offset: usize,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);

    let size = 1 << log_size;
    let step = 1 << log_step;
    let step_offset = offset % step;

    for i in (step_offset..size).step_by(step) {
        let circle_domain_index = coset_index_to_circle_domain_index(i, log_size);
        let circle_domain_index_bit_rev = bit_reverse_index(circle_domain_index, log_size);
        col.set(circle_domain_index_bit_rev, BaseField::one());
    }

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstantColumn {
    XorTable(u32, u32, usize),
    One(u32),
}

#[derive(Debug, Default, Clone)]
pub struct ConstantTableLocation {
    locations: HashMap<ConstantColumn, usize>,
}

impl ConstantTableLocation {
    pub fn add(&mut self, column: ConstantColumn, location: usize) {
        if self.locations.contains_key(&column) {
            panic!("Type already exists.");
        }
        self.locations.insert(column, location);
    }

    pub fn get_location(&self, column: ConstantColumn) -> Option<TreeLocation> {
        self.locations.get(&column).map(|&col_index| TreeLocation {
            tree_index: CONST_INTERACTION,
            col_index,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct StaticTree {
    pub root: Blake2sHash,
    pub locations: ConstantTableLocation,
}

impl StaticTree {
    pub fn blake_tree() -> Self {
        let root = Blake2sHash::default();
        let mut locations = ConstantTableLocation::default();
        locations.add(ConstantColumn::XorTable(12, 4, 0), 0);
        locations.add(ConstantColumn::XorTable(12, 4, 1), 1);
        locations.add(ConstantColumn::XorTable(12, 4, 2), 2);
        locations.add(ConstantColumn::XorTable(9, 2, 0), 3);
        locations.add(ConstantColumn::XorTable(9, 2, 1), 4);
        locations.add(ConstantColumn::XorTable(9, 2, 2), 5);
        locations.add(ConstantColumn::XorTable(8, 2, 0), 6);
        locations.add(ConstantColumn::XorTable(8, 2, 1), 7);
        locations.add(ConstantColumn::XorTable(8, 2, 2), 8);

        locations.add(ConstantColumn::XorTable(7, 2, 0), 12);
        locations.add(ConstantColumn::XorTable(7, 2, 1), 13);
        locations.add(ConstantColumn::XorTable(7, 2, 2), 14);

        locations.add(ConstantColumn::XorTable(4, 0, 0), 9);
        locations.add(ConstantColumn::XorTable(4, 0, 1), 10);
        locations.add(ConstantColumn::XorTable(4, 0, 2), 11);

        StaticTree { root, locations }
    }

    pub fn add1(log_size: u32) -> Self {
        let root = Blake2sHash::default();
        let mut locations = ConstantTableLocation::default();
        locations.add(ConstantColumn::One(log_size), 0);
        StaticTree { root, locations }
    }

    pub fn get_location(&self, column: ConstantColumn) -> TreeLocation {
        self.locations
            .get_location(column)
            .unwrap_or_else(|| panic!("{:?} column does not exist in the chosen tree!", column))
    }

    pub fn n_columns(&self) -> usize {
        self.locations.locations.len()
    }

    pub fn log_sizes(&self) -> Vec<u32> {
        self.locations
            .locations
            .iter()
            .map(|(_, &log_size)| log_size as u32)
            .collect()
    }
}
