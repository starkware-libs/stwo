use hashbrown::HashMap;
use itertools::Itertools;
use rand::{CryptoRng, RngCore};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{info, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::CanonicCoset;
use crate::core::queries::Queries;
use crate::core::utils::MaybeOwned;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::ExtendedMerkleDecommitmentLifted;
use crate::core::zk::{
    mix_zk_public_metadata, validate_zk_witness_metadata, zk_fri_batch_mask_fri_config,
    zk_fri_batch_mask_query_positions, ExtendedZkCommitmentSchemeProof, ZkColumnRange,
    ZkCommitmentSchemeProof, ZkCommitmentSchemeProofAux,
};
use crate::core::ColumnVec;
use crate::prover::air::component_prover::{Poly, Trace, WeightsHashMap};
use crate::prover::backend::{BackendForChannel, Col};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::pcs::quotient_ops::compute_fri_quotients;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;
use crate::prover::zk::{
    add_fri_batch_mask, sample_fri_batch_mask_evaluation, PrecommitZkWitnessRandomizationContext,
    ZkFriBatchMaskOracleProver, ZkProvingConfig, ZkProvingConfigError, ZkWitnessRandomizationError,
};

pub mod quotient_ops;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<MaybeOwned<'a, CommitmentTreeProver<B, MC>>>,
    pub config: PcsConfig,
    pub twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    /// Pre-allocated base field column pool for polynomial evaluation during commit.
    pub base_column_pool: MaybeOwned<'a, BaseColumnPool<B>>,
    zk_witness_randomization_contexts: Vec<PrecommitZkWitnessRandomizationContext<B>>,
    zk_witness_randomized_ranges: Vec<ZkColumnRange>,
}
impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    /// Creates a new empty commitment scheme prover with the given configuration and twiddles. The
    /// commitment scheme does not store the polynomials coefficients by default.
    pub fn new(config: PcsConfig, twiddles: &'a TwiddleTree<B>) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            base_column_pool: MaybeOwned::Owned(BaseColumnPool::new()),
            zk_witness_randomization_contexts: Vec::new(),
            zk_witness_randomized_ranges: Vec::new(),
        }
    }

    pub fn with_memory_pool(
        config: PcsConfig,
        twiddles: &'a TwiddleTree<B>,
        base_column_pool: &'a BaseColumnPool<B>,
    ) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            base_column_pool: MaybeOwned::Borrowed(base_column_pool),
            zk_witness_randomization_contexts: Vec::new(),
            zk_witness_randomized_ranges: Vec::new(),
        }
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Evaluates the given polynomials, commits them into a Merkle tree, mixes the root into
    /// the channel, and appends the resulting tree to the scheme.
    fn commit(&mut self, polynomials: ColumnVec<CircleCoefficients<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let tree = CommitmentTreeProver::new(
            polynomials,
            self.config.fri_config.log_blowup_factor,
            self.twiddles,
            self.store_polynomials_coefficients,
            self.config.lifting_log_size,
            &self.base_column_pool,
        );
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(MaybeOwned::Owned(tree));
    }

    /// Appends an externally constructed [`CommitmentTreeProver`] to the scheme and mixes its
    /// Merkle root into the channel. Accepts both owned and borrowed trees.
    pub fn commit_tree(
        &mut self,
        tree: MaybeOwned<'a, CommitmentTreeProver<B, MC>>,
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(tree);
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder {
            tree_index: self.trees.len(),
            commitment_scheme: self,
            polys: Vec::default(),
        }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasherLifted>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    pub fn evaluations(
        &self,
    ) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().map(|poly| &poly.evals).collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        let polys = self.polynomials();
        Trace { polys }
    }

    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        let weights_dashmap = WeightsHashMap::<B>::new();

        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                let compute_weights = |(log_size, point): (u32, CirclePoint<SecureField>)| {
                    weights_dashmap.entry((log_size, point)).or_insert_with(|| {
                        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                            CanonicCoset::new(log_size),
                            point,
                        )
                    });
                };

                let log_size = poly.evals.domain.log_size();
                // For each sample point, compute the weights needed to evaluate the polynomial at
                // the folded sample point.
                // TODO(Leo): the computation `point.repeated_double(max_log_size - log_size)` is
                // likely repeated a bunch of times in a typical flat air. Consider moving it
                // outside the loop.
                #[cfg(not(feature = "parallel"))]
                points.iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });

                #[cfg(feature = "parallel")]
                points.par_iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });
            });

        weights_dashmap
    }

    pub fn prove_values(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        // Evaluate polynomials on open points.
        let span = span!(
            Level::INFO,
            "Evaluate columns out of domain",
            class = "EvaluateOutOfDomain"
        )
        .entered();

        let lifting_log_size = self.trees.last().unwrap().commitment.layers.len() as u32 - 1;
        let weights_hash_map = if self.store_polynomials_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };

        // Lambda that evaluates a polynomial on a collection of circle points and returns a vector
        // of point samples.
        let eval_at_points = |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(
                        point.repeated_double(lifting_log_size - poly.evals.domain.log_size()),
                        weights_hash_map.as_ref(),
                    ),
                })
                .collect_vec()
        };

        #[cfg(not(feature = "parallel"))]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(eval_at_points);
        #[cfg(feature = "parallel")]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .par_map_cols(eval_at_points);

        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        let columns = self.evaluations();
        print_column_size_histogram::<B, MC>(&columns);
        // Compute oods quotients for boundary constraints on the sampled points.
        let quotients = compute_fri_quotients(
            &columns,
            &samples,
            channel.draw_secure_felt(),
            lifting_log_size,
            self.twiddles,
            self.config.fri_config.log_blowup_factor,
        );

        // Run FRI commitment phase on the oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        // Build the query position tree.
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].commitment.layers.len() as u32 - 1,
        );
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let commitments = self.roots();
        let (queried_values, decommitments, aux): (Vec<_>, Vec<_>, Vec<_>) = self
            .trees
            .as_ref()
            .zip_eq(query_positions_tree)
            .map(|(tree, query_positions)| tree.decommit(query_positions))
            .0
            .into_iter()
            .map(|(v, x)| (v, x.decommitment, x.aux))
            .multiunzip();

        // Return evaluation buffers to the memory pool for reuse (owned trees only).
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                for poly in tree.polynomials.drain(..) {
                    let log_size = poly.evals.domain.log_size();
                    self.base_column_pool.give_back(log_size, poly.evals.values);
                }
            }
        }

        ExtendedCommitmentSchemeProof {
            proof: CommitmentSchemeProof {
                commitments,
                sampled_values,
                decommitments: TreeVec(decommitments),
                queried_values: TreeVec(queried_values),
                proof_of_work,
                fri_proof: fri_proof.proof,
                config: self.config,
            },
            aux: CommitmentSchemeProofAux {
                unsorted_query_locations,
                trace_decommitment: TreeVec(aux),
                fri: fri_proof.aux,
            },
        }
    }

    pub fn prove_values_zk<R>(
        self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        zk_config: &ZkProvingConfig,
        rng: &mut R,
        channel: &mut MC::C,
    ) -> Result<ExtendedZkCommitmentSchemeProof<MC::H>, ZkProvingConfigError>
    where
        R: RngCore + CryptoRng + ?Sized,
    {
        self.prove_values_zk_with_fri_batch_mask(sampled_points, zk_config, rng, channel)
    }

    pub fn prove_values_zk_with_fri_batch_mask<R>(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        zk_config: &ZkProvingConfig,
        fri_batch_mask_rng: &mut R,
        channel: &mut MC::C,
    ) -> Result<ExtendedZkCommitmentSchemeProof<MC::H>, ZkProvingConfigError>
    where
        R: RngCore + CryptoRng + ?Sized,
    {
        let span = span!(
            Level::INFO,
            "Evaluate ZK columns out of domain",
            class = "EvaluateOutOfDomain"
        )
        .entered();

        let lifting_log_size = self.trees.last().unwrap().commitment.layers.len() as u32 - 1;
        let has_private_witness_columns = !zk_config.privacy_map.private_columns.is_empty();
        if has_private_witness_columns {
            if self.zk_witness_randomization_contexts.is_empty() {
                return Err(ZkProvingConfigError::MissingWitnessRandomizationContext);
            }
            validate_zk_witness_metadata(
                &zk_config.metadata,
                lifting_log_size,
                self.config.fri_config.log_blowup_factor,
            )
            .map_err(ZkProvingConfigError::Metadata)?;
            let mut expected_ranges = zk_config.privacy_map.private_columns.clone();
            expected_ranges.sort_unstable();
            let mut randomized_ranges = self.zk_witness_randomized_ranges.clone();
            randomized_ranges.sort_unstable();
            if randomized_ranges != expected_ranges {
                return Err(ZkProvingConfigError::WitnessRandomizationRangeMismatch);
            }
        } else {
            zk_config.validate_for_fri_batch_mask_only(
                lifting_log_size,
                self.config.fri_config.log_blowup_factor,
            )?;
        }
        let weights_hash_map = if self.store_polynomials_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };

        let eval_at_points = |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(
                        point.repeated_double(lifting_log_size - poly.evals.domain.log_size()),
                        weights_hash_map.as_ref(),
                    ),
                })
                .collect_vec()
        };

        #[cfg(not(feature = "parallel"))]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .map_cols(eval_at_points);
        #[cfg(feature = "parallel")]
        let samples: TreeVec<Vec<Vec<PointSample>>> = self
            .polynomials()
            .zip_cols(&sampled_points)
            .par_map_cols(eval_at_points);

        span.exit();
        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        mix_zk_public_metadata(
            channel,
            &zk_config.metadata,
            &zk_config.column_degree_bounds,
        );
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        let fri_batch_log_degree_bound =
            lifting_log_size - self.config.fri_config.log_blowup_factor;
        let fri_batch_mask = sample_fri_batch_mask_evaluation(
            CanonicCoset::new(lifting_log_size).circle_domain(),
            fri_batch_log_degree_bound,
            self.twiddles,
            fri_batch_mask_rng,
        );
        let fri_batch_mask_oracle = ZkFriBatchMaskOracleProver::<B, MC::H>::new(fri_batch_mask);
        assert_eq!(
            fri_batch_mask_oracle.log_size(),
            lifting_log_size,
            "FRI batch mask must use the first FRI layer domain"
        );
        let fri_batch_mask_fri_prover = FriProver::<B, MC>::commit(
            channel,
            zk_fri_batch_mask_fri_config(self.config.fri_config),
            fri_batch_mask_oracle.evaluation(),
            self.twiddles,
        );

        let columns = self.evaluations();
        print_column_size_histogram::<B, MC>(&columns);
        let random_coeff = channel.draw_secure_felt();
        let quotients = compute_fri_quotients(
            &columns,
            &samples,
            random_coeff,
            lifting_log_size,
            self.twiddles,
            self.config.fri_config.log_blowup_factor,
        );
        let h_batch = add_fri_batch_mask(quotients, fri_batch_mask_oracle.evaluation());

        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &h_batch, self.twiddles);

        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].commitment.layers.len() as u32 - 1,
        );
        if has_private_witness_columns {
            if zk_config
                .privacy_map
                .private_columns
                .iter()
                .any(|range| range.tree_index == 0)
                && preprocessed_query_positions != query_positions
            {
                return Err(ZkProvingConfigError::WitnessRandomization);
            }
            let mut audit_config = zk_config.clone();
            audit_config.derive_randomizer_metadata_from_stwo_samples(
                CanonicCoset::new(audit_config.metadata.degree_profile.trace_domain_log_size).coset,
                &sampled_points,
                &query_positions,
                lifting_log_size,
            )?;
            for context in &self.zk_witness_randomization_contexts {
                context
                    .validate_post_sampling_audit(&audit_config, &sampled_points, &query_positions)
                    .map_err(|_| ZkProvingConfigError::WitnessRandomization)?;
            }
        }
        let fri_batch_mask_query_positions = zk_fri_batch_mask_query_positions(
            channel,
            lifting_log_size,
            self.config.fri_config.n_queries,
            &query_positions,
            self.config.fri_config,
        )
        .map_err(|_| ZkProvingConfigError::InsufficientFriBatchMaskQueryDomain)?;
        let fri_batch_mask_queries =
            Queries::new(&fri_batch_mask_query_positions, lifting_log_size);
        let fri_batch_mask_fri_proof =
            fri_batch_mask_fri_prover.decommit_on_queries(&fri_batch_mask_queries);
        assert_eq!(
            fri_batch_mask_oracle.root(),
            fri_batch_mask_fri_proof.proof.first_layer.commitment,
            "FRI batch mask opening commitment must match its low-degree FRI commitment"
        );
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let commitments = self.roots();
        let (queried_values, decommitments, aux): (Vec<_>, Vec<_>, Vec<_>) = self
            .trees
            .as_ref()
            .zip_eq(query_positions_tree)
            .map(|(tree, query_positions)| tree.decommit(query_positions))
            .0
            .into_iter()
            .map(|(v, x)| (v, x.decommitment, x.aux))
            .multiunzip();

        let (fri_batch_mask, fri_batch_mask_decommitment_aux) = fri_batch_mask_oracle.decommit(
            &query_positions,
            &fri_batch_mask_query_positions,
            fri_batch_mask_fri_proof.proof,
        );

        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                for poly in tree.polynomials.drain(..) {
                    let log_size = poly.evals.domain.log_size();
                    self.base_column_pool.give_back(log_size, poly.evals.values);
                }
            }
        }

        Ok(ExtendedZkCommitmentSchemeProof {
            proof: ZkCommitmentSchemeProof {
                version: zk_config.metadata.version,
                randomized_pcs_proof: CommitmentSchemeProof {
                    commitments,
                    sampled_values,
                    decommitments: TreeVec(decommitments),
                    queried_values: TreeVec(queried_values),
                    proof_of_work,
                    fri_proof: fri_proof.proof,
                    config: self.config,
                },
                fri_batch_mask,
                public_metadata: zk_config.metadata.clone(),
            },
            aux: ZkCommitmentSchemeProofAux {
                randomized_pcs_aux: CommitmentSchemeProofAux {
                    unsorted_query_locations,
                    trace_decommitment: TreeVec(aux),
                    fri: fri_proof.aux,
                },
                fri_batch_mask_fri_aux: fri_batch_mask_fri_proof.aux,
                fri_batch_mask_decommitment_aux,
            },
        })
    }
}

/// Helper struct for aggregating polynomials and evaluations for a commitment tree.
pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CircleCoefficients<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns(columns, self.commitment_scheme.twiddles);
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        self.commitment_scheme.commit(self.polys, channel);
    }

    #[allow(dead_code)]
    pub(crate) fn commit_zk_witness_randomized<R>(
        mut self,
        zk_config: &ZkProvingConfig,
        rng: &mut R,
        channel: &mut MC::C,
    ) -> Result<(), ZkWitnessRandomizationError>
    where
        R: RngCore + CryptoRng + ?Sized,
    {
        let private_ranges = zk_config
            .privacy_map
            .private_columns
            .iter()
            .copied()
            .filter(|range| range.tree_index == self.tree_index)
            .collect_vec();
        if private_ranges.is_empty() {
            self.commit(channel);
            return Ok(());
        }

        let Some(randomized_log_degree) = zk_config
            .metadata
            .witness_randomization
            .private_column_degree_bounds
            .iter()
            .find(|bound| private_ranges.contains(&bound.range))
            .map(|bound| bound.log_degree_bound)
        else {
            return Err(ZkWitnessRandomizationError::Config(
                ZkProvingConfigError::MissingPrivateColumnDegreeBounds,
            ));
        };
        let trace_domain =
            CanonicCoset::new(zk_config.metadata.degree_profile.trace_domain_log_size)
                .circle_domain();
        let randomized_domain = CanonicCoset::new(randomized_log_degree).circle_domain();
        let context = PrecommitZkWitnessRandomizationContext::<B>::new(
            zk_config,
            trace_domain,
            randomized_domain,
        )?;

        for &range in &private_ranges {
            if range.column_end > self.polys.len() {
                return Err(ZkWitnessRandomizationError::PrivateRangeOutOfBounds {
                    range,
                    column_count: self.polys.len(),
                });
            }
        }

        let mut randomized_polys = core::mem::take(&mut self.polys);
        for &range in &private_ranges {
            for column_index in range.column_start..range.column_end {
                randomized_polys[column_index] =
                    context.randomize_range(range, &randomized_polys[column_index], rng)?;
            }
        }
        self.polys = randomized_polys;
        self.commitment_scheme
            .zk_witness_randomized_ranges
            .extend(private_ranges.iter().copied());
        self.commitment_scheme
            .zk_witness_randomization_contexts
            .push(context);

        self.commit(channel);
        Ok(())
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: MerkleProverLifted<B, MC::H>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            base_column_pool,
        );
        span.exit();

        let _span = span!(Level::INFO, "Merkle").entered();
        let max_log_domain_size = polynomials
            .iter()
            .map(|poly| poly.evals.domain.log_size())
            .max()
            .unwrap_or_default();
        let lifting_log_size = lifting_log_size.unwrap_or(max_log_domain_size);
        let tree = MerkleProverLifted::commit(
            polynomials
                .iter()
                .map(|poly: &Poly<B>| &poly.evals.values)
                .collect(),
            lifting_log_size,
            0,
        );

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    fn decommit(
        &self,
        queries: &[usize],
    ) -> (
        ColumnVec<Vec<BaseField>>,
        ExtendedMerkleDecommitmentLifted<MC::H>,
    ) {
        let eval_vec = self
            .polynomials
            .iter()
            .map(|poly| &poly.evals.values)
            .collect_vec();
        self.commitment.decommit(queries, eval_vec)
    }
}

fn print_column_size_histogram<B: BackendForChannel<MC>, MC: MerkleChannel>(
    columns_per_tree: &TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
) {
    let mut log_size_histogram = HashMap::new();
    for columns in columns_per_tree.iter() {
        for column in columns {
            *log_size_histogram
                .entry(column.domain.log_size())
                .or_insert(0) += 1;
        }
    }
    for (log_size, count) in log_size_histogram {
        info!("Log size {log_size}: {count}");
    }
}
