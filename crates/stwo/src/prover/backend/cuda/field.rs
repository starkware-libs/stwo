// use crate::core::{
//     backend::Column,
//     fields::{m31::BaseField, qm31::SecureField, FieldOps},
// };
// use crate::stwo_cuda as interface;
// use crate::prover::backend::cuda::CudaBackend;

// impl FieldOps<BaseField> for CudaBackend {
//     fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
//         unsafe {
//             interface::bindings::batch_inverse_base_field(
//                 column.device_ptr,
//                 dst.device_ptr,
//                 column.len(),
//             );
//         }
//     }
// }

// impl FieldOps<SecureField> for CudaBackend {
//     fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
//         unsafe {
//             interface::bindings::batch_inverse_secure_field(
//                 column.device_ptr,
//                 dst.device_ptr,
//                 column.len(),
//             );
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use stwo::core::{
//         backend::{Column, CpuBackend},
//         fields::{m31::BaseField, qm31::SecureField, FieldOps},
//     };

//     use crate::{backend::CudaBackend, cuda};

//     #[test]
//     fn test_batch_inverse_basefield() {
//         let size: usize = 1 << 25;
//         let from = (1..(size + 1) as u32)
//             .map(BaseField::from)
//             .collect::<Vec<_>>();
//         let dst = from.clone();
//         let mut dst_expected = dst.clone();
//         CpuBackend::batch_inverse(&from, &mut dst_expected);

//         let from_device = cuda::BaseFieldVec::from_vec(from);
//         let mut dst_device = cuda::BaseFieldVec::new_uninitialized(size);
//         <CudaBackend as FieldOps<BaseField>>::batch_inverse(&from_device, &mut dst_device);

//         assert_eq!(dst_device.to_cpu(), dst_expected.to_cpu());
//     }

//     #[test]
//     fn test_batch_inverse_secure_field() {
//         let size: usize = 1 << 25;

//         let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();

//         let from_cpu = from_raw
//             .chunks(4)
//             .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
//             .collect::<Vec<_>>();
//         let mut dst_expected_cpu = from_cpu.clone();

//         CpuBackend::batch_inverse(&from_cpu, &mut dst_expected_cpu);

//         let from_device = cuda::SecureFieldVec::from_vec(from_cpu.clone());
//         let mut dst_device = cuda::SecureFieldVec::from_vec(from_cpu.clone());

//         <CudaBackend as FieldOps<SecureField>>::batch_inverse(&from_device, &mut dst_device);

//         assert_eq!(dst_device.to_cpu(), dst_expected_cpu);
//     }
// }
