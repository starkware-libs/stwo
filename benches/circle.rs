use std::iter::zip;

use criterion::black_box;
use prover_research::core::circle::{CirclePointIndex, M31_CIRCLE_ORDER_BITS};
use rand::Rng;

pub const N_ELEMENTS: usize = 1 << 10;
pub const N_STATE_ELEMENTS: usize = 8;

pub fn circle_point_bench(c: &mut criterion::Criterion) {
    let mut rng = rand::thread_rng();
    let mut circle_indices: Vec<CirclePointIndex> = Vec::new();

    let mut index_state: [CirclePointIndex; N_STATE_ELEMENTS] = core::array::from_fn(|_| {
        CirclePointIndex(rng.gen::<usize>() % (1 << M31_CIRCLE_ORDER_BITS))
    });

    let mut point_state = index_state
        .iter()
        .map(|index| index.to_point())
        .collect::<Vec<_>>();

    for _ in 0..N_ELEMENTS {
        let point_index = CirclePointIndex(rng.gen::<usize>() % (1 << M31_CIRCLE_ORDER_BITS));
        circle_indices.push(point_index);
    }

    c.bench_function("CirclePoint addition", |b| {
        b.iter(|| {
            for index in black_box(&circle_indices) {
                for _ in 0..128 {
                    for point in &mut point_state {
                        *point = *point + index.to_point();
                    }
                }
            }
        })
    });

    c.bench_function("CirclePointIndex addition", |b| {
        b.iter(|| {
            for index in black_box(&circle_indices) {
                for _ in 0..128 {
                    for index2 in &mut index_state {
                        let _p = black_box((*index2 + *index).to_point());
                        *index2 = *index2 + *index;
                    }
                }
            }
        })
    });
    for (index, point) in zip(index_state, point_state) {
        assert_eq!(index.to_point(), point);
    }
}

criterion::criterion_group!(circle_benches, circle_point_bench,);
criterion::criterion_main!(circle_benches);
