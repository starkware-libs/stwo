

use std::fmt::Debug;

use super::{hasher::Hasher, mdconfig::MDConfig};
use crate::core::fields::{Field, IntoSlice};

pub const DEFAULT_TRAPZOID_HEIGHT: usize = 8; // Optimize for L1 cache size.
pub const MAXIMAL_TRAPZOID_TOP_LAYER_LENGTH: usize = 8; // Optimize for avx512.
pub const MAX_FAN_OUT: usize = 32;
pub struct Trapezoid<'a, F: Field, H: Hasher> {
    pub hash_layers: Vec<Vec<H::Hash>>,
    matrices_to_inject: Option<MDConfig<'a,F>>,
    sub_trapezoids: Option<Vec<Trapezoid<'a, F, H>>>,
    depth_in_containing_tree: usize, /* index of the top layer of the trapzoid in the
                                      * containing tree. */
    trapezoids_in_containing_tree_level: usize,
    index_in_containing_tree_level: usize,
    n_nodes_top_layer: usize,
    height: usize,
}

impl<'a, F: Field + Debug, H: Hasher> Trapezoid<'a, F, H>
where
    F: IntoSlice<H::NativeType>,
{
    pub fn construct_pass_commit(
        passed_columns: MDConfig<'a,F>,
        height: usize,
        depth_in_containing_tree: usize, // top layer index in the whole tree.
        n_nodes_top_layer: usize,
        trapezoids_in_containing_tree_level: usize,
        index_in_containing_tree_level: usize,
    ) -> Self {
        assert!(passed_columns.max_depth >= depth_in_containing_tree + height);
        let key_to_split_at = Self::find_split_key(&passed_columns, depth_in_containing_tree + height);
        let matrices_to_pass = match key_to_split_at {
            Some(key) => Self::split_trace(&mut (passed_columns.clone()), key),
            None =>  None,
        };
        let sub_trapezoids = matrices_to_pass.map(|matrices_to_pass| {
            Self::get_sub_trapezoids(
                matrices_to_pass.clone(),
                n_nodes_top_layer << (height - 1),
                depth_in_containing_tree + height,
                trapezoids_in_containing_tree_level,
                index_in_containing_tree_level,
            )
        });

        let mut trap = Self {
            hash_layers: Vec::with_capacity(height),
            matrices_to_inject: Some(passed_columns),
            sub_trapezoids,
            depth_in_containing_tree,
            trapezoids_in_containing_tree_level,
            index_in_containing_tree_level,
            n_nodes_top_layer,
            height,
        };
        trap.commit();
        trap
    }

    fn commit(&mut self) {
        // Bottom layer gets values from sub-trapezoids
        let n_nodes_sub_trap_layer = self.n_nodes_top_layer << (self.height);
        let mut inputs = std::iter::repeat(Vec::new())
            .take(n_nodes_sub_trap_layer >> 1)
            .collect::<Vec<Vec<&[H::NativeType]>>>();
        if let Some(sub_trapezoids) = self.sub_trapezoids.as_ref() {
            let n_nodes_in_sub_trapezoid = sub_trapezoids[0].hash_layers.last().unwrap().len();
            for (j, trap) in sub_trapezoids.iter().enumerate() {
                trap.get_roots_ref().iter().enumerate().for_each(|(i, h)| {
                    inputs[(j * n_nodes_in_sub_trapezoid + i) / 2].push(h.as_ref())
                });
            }
        }

        for i in 0..self.height {
            if let Some(prev_hashes) = self.hash_layers.last() {
                for (j, hashes) in prev_hashes.chunks(2).enumerate() {
                    inputs[j].push(hashes[0].as_ref());
                    inputs[j].push(hashes[1].as_ref());
                }
            }
            if let Some(matrices_to_inject) = self.matrices_to_inject.as_ref() {
                if let Some(matrices) =
                    matrices_to_inject.get(self.depth_in_containing_tree + self.height - i)
                {
                    for column in matrices.iter() {
                        let n_rows_in_inject = (column.len()
                            / self.trapezoids_in_containing_tree_level)
                            / (n_nodes_sub_trap_layer >> (i + 1));
                        (0..inputs.len()).for_each(|m| {
                            let idx = (column.len() / self.trapezoids_in_containing_tree_level)
                                * self.index_in_containing_tree_level
                                + m * n_rows_in_inject;
                            inputs[m].push(<F as IntoSlice<H::NativeType>>::into_slice(
                                &column[idx..idx+n_rows_in_inject],
                            ));
                        });
                    }
                }
            }
            self.hash_layers.push(H::hash_many_multi_src(&inputs));
            inputs = std::iter::repeat(Vec::new())
                .take(n_nodes_sub_trap_layer >> (i + 2))
                .collect::<Vec<Vec<&[H::NativeType]>>>();
        }
    }

    pub fn get_roots_ref(&self) -> &[H::Hash] {
        self.hash_layers.last().unwrap()
    }

    fn get_sub_trapezoids(
        matrices_to_pass: MDConfig<'a, F>,
        n_nodes_bottom_layer: usize,
        depth_in_containing_tree: usize,
        trapezoids_in_containing_tree_level: usize,
        index_in_containing_tree_level: usize,
    ) -> Vec<Trapezoid<'a, F, H>> {
        let sub_trapezoids_height = std::cmp::min(
            matrices_to_pass.keys().max().unwrap_or(&0) - (depth_in_containing_tree),
            DEFAULT_TRAPZOID_HEIGHT,
        );
        let n_sub_trapezoids = Self::decide_fan_out(n_nodes_bottom_layer);
        let sub_trapezoids_width = (n_nodes_bottom_layer << 1) / n_sub_trapezoids;
        (0..n_sub_trapezoids)
            .map(|i| {
                Trapezoid::construct_pass_commit(
                    matrices_to_pass.clone(),
                    sub_trapezoids_height,
                    depth_in_containing_tree,
                    sub_trapezoids_width,
                    trapezoids_in_containing_tree_level * n_sub_trapezoids,
                    index_in_containing_tree_level * n_sub_trapezoids + i,
                )
            })
            .collect::<Vec<Trapezoid<'a, F, H>>>()
    }

    fn find_split_key(
        matrices_to_inject: &MDConfig<'a, F>,
        max_depth: usize,
    ) -> Option<usize> {
        for key in matrices_to_inject.keys() {
            if *key > max_depth {
                return Some(*key);
            }
        }
        None
    }

    fn split_trace(
        passed_columns: &mut MDConfig<'a, F>,
        key_to_split_at: usize,
    ) -> Option<MDConfig<'a, F>> {
            let matrices_to_pass = passed_columns.split_off(&key_to_split_at);
            Some(matrices_to_pass)
        }

    fn decide_fan_out(n_nodes_bottom_layer: usize) -> usize {
        if n_nodes_bottom_layer << 1 > MAXIMAL_TRAPZOID_TOP_LAYER_LENGTH * MAX_FAN_OUT {
            MAX_FAN_OUT
        } else {
            2
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::merkle_tree::MerkleTree;
    use crate::core::fields::m31::M31;

    #[test]
    pub fn debug_trapezoid() {
        // let trace_column = [M31::from_u32_unchecked(0); 1024].to_vec();
        let trace_column = (0..4096).map(M31::from_u32_unchecked).collect::<Vec<M31>>();

        let mut config = super::MDConfig::<M31>::default();
        config.insert(9, &trace_column).unwrap();
        // config.insert(4, &trace_column).unwrap();
        // config.insert(3, &trace_column).unwrap();
        // config.insert(2, &trace_column).unwrap();
        // config.insert(1, &trace_column).unwrap();

        let trap = super::Trapezoid::<M31, Blake3Hasher>::construct_pass_commit(config, 7, 0, 1, 1, 0);
        if let Some(sub_trapezoides) = trap.sub_trapezoids.as_ref() {
            sub_trapezoides.iter().for_each(|trap| {
                trap.hash_layers.iter().for_each(|layer| {
                    println!("{}, *:{}", hex::encode(layer[0].as_ref()), layer.len());
                });
                println!("---");
            });
            println!("---");
        }
        trap.hash_layers.iter().for_each(|layer| {
            println!("{}, *{}", hex::encode(layer[0].as_ref()), layer.len());
        });
    }

    #[test]
    pub fn test_trapezoid() {
        let trace_column = (0..4096).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let tree = MerkleTree::<M31, Blake3Hasher>::commit(vec![trace_column.clone()]);
        let mut config = super::MDConfig::<M31>::default();
        config.insert(9, &trace_column).unwrap();

        let trap = super::Trapezoid::<M31, Blake3Hasher>::construct_pass_commit(config, 1, 0, 1, 1, 0);

        assert_eq!(trap.get_roots_ref()[0], tree.root());
    }
}

// 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
// 00000000000000000000000000000000
// 01000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000
