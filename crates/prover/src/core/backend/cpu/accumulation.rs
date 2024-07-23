use super::CpuBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for CpuBackend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {
        use std::mem::transmute;

        use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
        use icicle_cuda_runtime::memory::HostSlice;
        use icicle_m31::field::ExtensionField;

        use crate::core::SecureField;

        let mut a: Vec<ExtensionField> = vec![];
        let mut b: Vec<ExtensionField> = vec![];
        let len = column.len();
        for i in 0..len {
            // TODO: just for the sake of correctness check - perf optimisation can be done without
            // data conversion
            let ci = column.at(i);
            let oi = other.at(i);

            let aa = ci.to_m31_array();
            let bb = oi.to_m31_array();

            let aa: ExtensionField = unsafe { transmute(aa) };
            let bb: ExtensionField = unsafe { transmute(bb) };

            a.push(aa);
            b.push(bb);
        }

        let a = HostSlice::from_mut_slice(&mut a);
        let b = HostSlice::from_slice(&b);

        let cfg = VecOpsConfig::default();

        accumulate_scalars(a, b, &cfg).unwrap();

        a.iter().enumerate().for_each(|(i, &item)| {
            column.set(
                i,
                SecureField::from_m31_array(unsafe { std::mem::transmute(item) }),
            )
        });
    }
}
