use std::sync::OnceLock; // ← 표준 라이브러리만!

use flume::{Receiver, Sender};

use crate::poseidon::web::{ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput};

#[derive(Debug)]
pub struct Channels {
    pub tx: Sender<Box<ComputeCompositionPolynomialInput>>,
    pub rx: Receiver<Box<ComputeCompositionPolynomialOutput>>,
}

thread_local! {
    static CHANNELS: OnceLock<Channels> = OnceLock::new();
}

#[allow(dead_code)]
pub fn init_gpu_channels(
    tx: Sender<Box<ComputeCompositionPolynomialInput>>,
    rx: Receiver<Box<ComputeCompositionPolynomialOutput>>,
) {
    CHANNELS.with(|cell| {
        cell.set(Channels { tx, rx })
            .expect("gpu_channels::init called twice");
    });
}

#[allow(dead_code)]
pub fn with<F, R>(f: F) -> R
where
    F: FnOnce(&Channels) -> R,
{
    CHANNELS.with(|cell| {
        let chans = cell.get().expect("gpu channels not initialised");
        f(chans)
    })
}
