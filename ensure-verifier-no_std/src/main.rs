#![no_std]
#![allow(unused_imports)]
#![no_main]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

use stwo::core::verifier::verify;
use stwo_constraint_framework::PointEvaluator;
