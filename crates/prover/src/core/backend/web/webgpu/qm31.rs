use std::borrow::Cow;

use super::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuM31 {
    pub data: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuCM31 {
    pub a: GpuM31,
    pub b: GpuM31,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GpuQM31 {
    pub a: GpuCM31,
    pub b: GpuCM31,
}

impl From<QM31> for GpuQM31 {
    fn from(value: QM31) -> Self {
        GpuQM31 {
            a: GpuCM31 {
                a: GpuM31 {
                    data: value.0 .0.into(),
                },
                b: GpuM31 {
                    data: value.0 .1.into(),
                },
            },
            b: GpuCM31 {
                a: GpuM31 {
                    data: value.1 .0.into(),
                },
                b: GpuM31 {
                    data: value.1 .1.into(),
                },
            },
        }
    }
}

impl From<GpuQM31> for QM31 {
    fn from(value: GpuQM31) -> Self {
        QM31(
            CM31::from_m31(value.a.a.data.into(), value.a.b.data.into()),
            CM31::from_m31(value.b.a.data.into(), value.b.b.data.into()),
        )
    }
}

impl From<M31> for GpuM31 {
    fn from(value: M31) -> Self {
        GpuM31 { data: value.into() }
    }
}

impl From<GpuM31> for M31 {
    fn from(value: GpuM31) -> Self {
        M31::from(value.data)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComputeInput {
    pub first: GpuQM31,
    pub second: GpuQM31,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComputeOutput {
    pub result: GpuQM31,
}

impl ByteSerialize for ComputeInput {}
impl ByteSerialize for ComputeOutput {}

pub enum QM31Operation {
    Add,
    Subtract,
    Multiply,
    Negate,
    Inverse,
}

impl GpuOperation for QM31Operation {
    fn shader_source(&self) -> Cow<'static, str> {
        let base_source = include_str!("qm31.wgsl");

        let inputs = r#"
            struct ComputeInput {
                first: QM31,
                second: QM31,
            }

            @group(0) @binding(0) var<storage, read> input: ComputeInput;
        "#;

        let output = r#"
            struct ComputeOutput {
                result: QM31,
            }

            @group(0) @binding(1) var<storage, read_write> output: ComputeOutput;
        "#;

        let operation = match self {
            QM31Operation::Add => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = qm31_add(input.first, input.second);
                }
            "#
            }
            QM31Operation::Multiply => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = qm31_mul(input.first, input.second);
                }
            "#
            }
            QM31Operation::Subtract => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = qm31_sub(input.first, input.second);
                }
            "#
            }
            QM31Operation::Negate => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = qm31_neg(input.first);
                }
            "#
            }
            QM31Operation::Inverse => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = qm31_inverse(input.first);
                }
            "#
            }
        };

        format!("{base_source}\n{inputs}\n{output}\n{operation}").into()
    }
}

pub async fn compute_field_operation(operation: QM31Operation, first: QM31, second: QM31) -> QM31 {
    let input = ComputeInput {
        first: first.into(),
        second: second.into(),
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<ComputeOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<ComputeOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output.result.into()
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::FieldExpOps;
    use crate::{cm31, qm31};

    #[test]
    fn test_gpu_field_values() {
        let qm0 = qm31!(1, 2, 3, 4);
        let qm1 = qm31!(4, 5, 6, 7);

        // Test round-trip conversion CPU -> GPU -> CPU
        let gpu_qm0 = GpuQM31::from(qm0);
        let gpu_qm1 = GpuQM31::from(qm1);

        let cpu_qm0 = QM31(
            CM31(gpu_qm0.a.a.data.into(), gpu_qm0.a.b.data.into()),
            CM31(gpu_qm0.b.a.data.into(), gpu_qm0.b.b.data.into()),
        );

        let cpu_qm1 = QM31(
            CM31(gpu_qm1.a.a.data.into(), gpu_qm1.a.b.data.into()),
            CM31(gpu_qm1.b.a.data.into(), gpu_qm1.b.b.data.into()),
        );

        assert_eq!(
            qm0, cpu_qm0,
            "Round-trip conversion should preserve values for qm0"
        );
        assert_eq!(
            qm1, cpu_qm1,
            "Round-trip conversion should preserve values for qm1"
        );
    }

    #[test]
    fn test_gpu_m31_field_arithmetic() {
        // Test M31 field operations
        let m = M31::from(19u32);
        let one = M31::from(1u32);
        let zero = M31::zero();

        // Create QM31 values for GPU computation
        let m_qm = QM31(CM31(m, zero), CM31::zero());
        let one_qm = QM31(CM31(one, zero), CM31::zero());
        let zero_qm = QM31(CM31(zero, zero), CM31::zero());

        // Test addition
        let cpu_add = m + one;
        let gpu_add = pollster::block_on(compute_field_operation(QM31Operation::Add, m_qm, one_qm));
        assert_eq!(gpu_add.0 .0, cpu_add, "M31 addition failed");

        // Test subtraction
        let cpu_sub = m - one;
        let gpu_sub = pollster::block_on(compute_field_operation(
            QM31Operation::Subtract,
            m_qm,
            one_qm,
        ));
        assert_eq!(gpu_sub.0 .0, cpu_sub, "M31 subtraction failed");

        // Test multiplication
        let cpu_mul = m * one;
        let gpu_mul = pollster::block_on(compute_field_operation(
            QM31Operation::Multiply,
            m_qm,
            one_qm,
        ));
        assert_eq!(gpu_mul.0 .0, cpu_mul, "M31 multiplication failed");

        // Test negation
        let cpu_neg = -m;
        let gpu_neg = pollster::block_on(compute_field_operation(
            QM31Operation::Negate,
            m_qm,
            zero_qm,
        ));
        assert_eq!(gpu_neg.0 .0, cpu_neg, "M31 negation failed");

        // Test inverse
        let cpu_inv = m.inverse();
        let gpu_inv = pollster::block_on(compute_field_operation(
            QM31Operation::Inverse,
            m_qm,
            zero_qm,
        ));
        assert_eq!(gpu_inv.0 .0, cpu_inv, "M31 inverse failed");

        // Test with large numbers (near P)
        let large = M31::from(P - 1);
        let large_qm = QM31(CM31(large, zero), CM31::zero());

        // Test large number multiplication
        let cpu_large_mul = large * m;
        let gpu_large_mul = pollster::block_on(compute_field_operation(
            QM31Operation::Multiply,
            large_qm,
            m_qm,
        ));
        assert_eq!(
            gpu_large_mul.0 .0, cpu_large_mul,
            "M31 large number multiplication failed"
        );

        // Test large number inverse
        let cpu_large_inv = one / large;
        let gpu_large_inv = pollster::block_on(compute_field_operation(
            QM31Operation::Inverse,
            large_qm,
            zero_qm,
        ));
        assert_eq!(
            gpu_large_inv.0 .0, cpu_large_inv,
            "M31 large number inverse failed"
        );
    }

    #[test]
    fn test_gpu_cm31_field_arithmetic() {
        let cm0 = cm31!(1, 2);
        let cm1 = cm31!(4, 5);
        let zero = CM31::zero();

        // Test addition
        let cpu_add = cm0 + cm1;
        let gpu_add = pollster::block_on(compute_field_operation(
            QM31Operation::Add,
            QM31(cm0, zero),
            QM31(cm1, zero),
        ));
        assert_eq!(gpu_add.0, cpu_add, "CM31 addition failed");

        // Test subtraction
        let cpu_sub = cm0 - cm1;
        let gpu_sub = pollster::block_on(compute_field_operation(
            QM31Operation::Subtract,
            QM31(cm0, zero),
            QM31(cm1, zero),
        ));
        assert_eq!(gpu_sub.0, cpu_sub, "CM31 subtraction failed");

        // Test multiplication
        let cpu_mul = cm0 * cm1;
        let gpu_mul = pollster::block_on(compute_field_operation(
            QM31Operation::Multiply,
            QM31(cm0, zero),
            QM31(cm1, zero),
        ));
        assert_eq!(gpu_mul.0, cpu_mul, "CM31 multiplication failed");

        // Test negation
        let cpu_neg = -cm0;
        let gpu_neg = pollster::block_on(compute_field_operation(
            QM31Operation::Negate,
            QM31(cm0, zero),
            QM31(zero, zero),
        ));
        assert_eq!(gpu_neg.0, cpu_neg, "CM31 negation failed");

        // Test inverse
        let cpu_inv = cm0.inverse();
        let gpu_inv = pollster::block_on(compute_field_operation(
            QM31Operation::Inverse,
            QM31(cm0, zero),
            QM31(zero, zero),
        ));
        assert_eq!(gpu_inv.0, cpu_inv, "CM31 inverse failed");

        // Test with large numbers (near P)
        let large = cm31!(P - 1, P - 2);
        let large_qm = QM31(large, zero);

        // Test large number multiplication
        let cpu_large_mul = large * cm1;
        let gpu_large_mul = pollster::block_on(compute_field_operation(
            QM31Operation::Multiply,
            large_qm,
            QM31(cm1, zero),
        ));
        assert_eq!(
            gpu_large_mul.0, cpu_large_mul,
            "CM31 large number multiplication failed"
        );

        // Test large number inverse
        let cpu_large_inv = large.inverse();
        let gpu_large_inv = pollster::block_on(compute_field_operation(
            QM31Operation::Inverse,
            large_qm,
            QM31(zero, zero),
        ));
        assert_eq!(
            gpu_large_inv.0, cpu_large_inv,
            "CM31 large number inverse failed"
        );
    }

    #[test]
    fn test_gpu_qm31_field_arithmetic() {
        let qm0 = qm31!(1, 2, 3, 4);
        let qm1 = qm31!(4, 5, 6, 7);
        let zero = QM31::zero();

        // Test addition
        let cpu_add = qm0 + qm1;
        let gpu_add = pollster::block_on(compute_field_operation(QM31Operation::Add, qm0, qm1));
        assert_eq!(gpu_add, cpu_add, "QM31 addition failed");

        // Test subtraction
        let cpu_sub = qm0 - qm1;
        let gpu_sub =
            pollster::block_on(compute_field_operation(QM31Operation::Subtract, qm0, qm1));
        assert_eq!(gpu_sub, cpu_sub, "QM31 subtraction failed");

        // Test multiplication
        let cpu_mul = qm0 * qm1;
        let gpu_mul =
            pollster::block_on(compute_field_operation(QM31Operation::Multiply, qm0, qm1));
        assert_eq!(gpu_mul, cpu_mul, "QM31 multiplication failed");

        // Test negation
        let cpu_neg = -qm0;
        let gpu_neg = pollster::block_on(compute_field_operation(QM31Operation::Negate, qm0, zero));
        assert_eq!(gpu_neg, cpu_neg, "QM31 negation failed");

        // Test inverse
        let cpu_inv = qm0.inverse();
        let gpu_inv = pollster::block_on(compute_field_operation(QM31Operation::Inverse, qm0, qm1));
        assert_eq!(gpu_inv, cpu_inv, "QM31 inverse failed");

        // Test with large numbers (near P)
        let large = qm31!(P - 1, P - 2, P - 3, P - 4);

        // Test large number multiplication
        let cpu_large_mul = large * qm1;
        let gpu_large_mul =
            pollster::block_on(compute_field_operation(QM31Operation::Multiply, large, qm1));
        assert_eq!(
            gpu_large_mul, cpu_large_mul,
            "QM31 large number multiplication failed"
        );

        // Test large number inverse
        let cpu_large_inv = qm1.inverse();
        let gpu_large_inv =
            pollster::block_on(compute_field_operation(QM31Operation::Inverse, qm1, zero));
        assert_eq!(
            gpu_large_inv, cpu_large_inv,
            "QM31 large number inverse failed"
        );
    }
}
