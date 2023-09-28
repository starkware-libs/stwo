pub mod channel;
pub mod col;
pub mod pc;

use self::{
    channel::{Challenge, Channel},
    col::Column,
    pc::{BatchPC, PC},
};

use super::{circle::CirclePoint, field::m31::Field};

// Example.
struct Fibonacci {
    final_channel: Channel,
    fib_col: Column,
    trace_pc: PC,
    constraint_coefs: Vec<Challenge<Field>>, // TODO(spapini): extension field.
    composition_pc: PC,
    deep_point: Vec<Challenge<CirclePoint>>, // TODO(spapini): extension field circle point.
    batch_pc: BatchPC,
}
