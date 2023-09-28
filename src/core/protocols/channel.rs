use std::{borrow::Cow, marker::PhantomData};

use crate::core::{
    circle::{CircleIndex, CirclePoint},
    field::m31::Field,
};

pub struct Channel {
    time: ChannelTime,
}
pub struct ChannelTime {
    n_sent: usize,
    n_challenges: usize,
}

pub trait ChannelValue {
    const N_ELS: usize;
    fn from_els(els: &[Field]) -> Self;
    fn to_els(&self) -> Cow<'_, Field>;
}
pub struct Challenge<T: ChannelValue> {
    time: ChannelTime,
    phantom: PhantomData<T>,
}
pub struct SentValue<T: ChannelValue> {
    time: ChannelTime,
    phantom: PhantomData<T>,
}

impl ChannelValue for Field {
    const N_ELS: usize = 1;

    fn from_els(els: &[Field]) -> Self {
        todo!()
    }

    fn to_els(&self) -> Cow<'_, Field> {
        todo!()
    }
}

impl ChannelValue for CircleIndex {
    const N_ELS: usize = 2;

    fn from_els(els: &[Field]) -> Self {
        todo!()
    }

    fn to_els(&self) -> Cow<'_, Field> {
        todo!()
    }
}

impl ChannelValue for CirclePoint {
    const N_ELS: usize = 2;

    fn from_els(els: &[Field]) -> Self {
        todo!()
    }

    fn to_els(&self) -> Cow<'_, Field> {
        todo!()
    }
}
