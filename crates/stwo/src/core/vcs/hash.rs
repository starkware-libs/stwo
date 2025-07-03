use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

pub trait Hash:
    Copy
    + Default
    + Display
    + Debug
    + Eq
    + Send
    + Sync
    + 'static
    + Serialize
    + for<'de> Deserialize<'de>
{
}
