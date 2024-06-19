pub mod registry;

use downcast_rs::{impl_downcast, Downcast};

pub trait ComponentGen: Downcast {}
impl_downcast!(ComponentGen);
