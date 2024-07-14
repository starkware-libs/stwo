use super::CpuBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::secure_column::SecureColumn;

impl AccumulationOps for CpuBackend {
    fn accumulate(column: &mut SecureColumn<Self>, other: &SecureColumn<Self>) {

        use std::mem::transmute;

        use icicle_m31::field::{
            // ExtensionCfg, 
            ExtensionField};

        let mut a : Vec<ExtensionField> = vec![];
        let mut b : Vec<ExtensionField> = vec![];
        let len = column.len();
        for i in 0..len {
            // let res_coeff = column.at(i) + other.at(i);
            // column.set(i, res_coeff);

            //

            let ci = column.at(i);
            let oi = other.at(i);

            let aa = ci.to_m31_array();
            let bb = oi.to_m31_array();

            let aa: ExtensionField = unsafe { transmute(aa)  };
            let bb: ExtensionField = unsafe { transmute(bb)  };

            a.push(aa);
            b.push(bb);

            // if ci.0.0.0 == 0 && ci.1.0.0 == 0 {
            //     if oi.0.0.0 == 0 && oi.1.0.0 == 0 { continue; }
            //     column.set(i, oi);
            // } else if oi.0.0.0 == 0 && oi.1.0.0 == 0 {
            //     column.set(i, ci);
            // } else { column.set(i, ci + oi); }
        }

        // use std::mem::transmute;

        // // use icicle_bls12_381::curve::ScalarField;
        // use icicle_m31::field::{
        //     // ExtensionCfg, 
        //     ExtensionField};
        // // use icicle_core::traits::GenerateRandom;
        use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
        use icicle_cuda_runtime::memory::HostSlice;
        use crate::core::SecureField;
        // // let mut a = ExtensionCfg::generate_random(test_size);
        // // let b = ExtensionCfg::generate_random(test_size);
        let a = HostSlice::from_mut_slice(&mut a);
        let b = HostSlice::from_slice(&b);

        let cfg = VecOpsConfig::default();

        accumulate_scalars(a, b, &cfg).unwrap();


        for i in 0..len {
            // let res_coeff = column.at(i) + other.at(i);
            // column.set(i, res_coeff);

            //
            let arr = unsafe{ transmute(a[i]) };
            let ci = SecureField::from_m31_array(arr);
            column.set(i, ci)

 

            // if ci.0.0.0 == 0 && ci.1.0.0 == 0 {
            //     if oi.0.0.0 == 0 && oi.1.0.0 == 0 { continue; }
            //     column.set(i, oi);
            // } else if oi.0.0.0 == 0 && oi.1.0.0 == 0 {
            //     column.set(i, ci);
            // } else { column.set(i, ci + oi); }
        }

        // panic!();
    }
}
