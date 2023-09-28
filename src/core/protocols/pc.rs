use crate::core::{circle::CircleIndex, field::m31::Field};

use super::{
    channel::{Challenge, SentValue},
    col::Column,
};

pub struct PC {
    columns: Vec<Column>,
    n_blowup: usize,
    n_heavy_bits: usize,
    commit_time: SentValue<Field>,
}

// TODO: How to track the mask to each PC?
pub struct MaskItem {
    col: Column,
    offset: CircleIndex,
}

pub struct Mask {
    items: Vec<MaskItem>,
}

pub struct BatchPC {
    pcs: Vec<PC>,
    opening_air: PCOpeningAIR,
    ldt: LowDegreeTest,
    pc_checks: Vec<PCCheck>,
}

pub struct PCOpeningAIR {
    pcs: Vec<PC>,
    values: Vec<PCOpeningValue>,
    mask: Mask,
}

pub struct PCOpeningValue {
    col: Column,
    value: SentValue<Field>,
}

pub struct PCCheck {
    queries: Vec<Vec<CircleIndex>>,
    leaf_responses: Vec<Vec<SentValue<Field>>>,
    inner_responses: Vec<Vec<SentValue<Field>>>, // TODO: Hash.
}

pub struct LowDegreeTest {
    n_bits: usize,
    layers: Vec<PC>,
    queries: Vec<Challenge<CircleIndex>>,
    leaf_responses: Vec<Vec<SentValue<Field>>>,
    inner_responses: Vec<Vec<SentValue<Field>>>, // TODO: Hash.
}
