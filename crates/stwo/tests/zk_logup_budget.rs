use stwo::core::circle::CirclePoint;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::zk::{
    canonical_zk_logup_statistical_security_budget_hash, canonical_zk_privacy_map_hash,
    canonical_zk_private_column_scope_hash, expected_zk_column_degree_bounds,
    reject_zk_logup_statistical_security_budget_activation,
    validate_zk_public_metadata_against_verifier_config,
    validate_zk_witness_randomization_audit_for_verifier, ZkColumnDegreeBound, ZkColumnRange,
    ZkDegreeProfile, ZkLogupStatisticalAggregatePolicyError, ZkLogupStatisticalSecurityBudget,
    ZkPrivacyMap, ZkPrivacyMapHash, ZkPrivateColumnScope, ZkPrivateColumnScopeEntry,
    ZkPrivateColumnUsage, ZkProofVersion, ZkPublicMetadata, ZkPublicStatementHash,
    ZkQuotientIntegrationProfile, ZkVerificationConfig, ZkVerificationConfigValidationError,
    ZkWitnessRandomizationProfile, ZkWitnessRandomizationVerifierAudit,
    ZkWitnessRandomizationVerifierAuditError, ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH,
    ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS,
};

fn logup_budget() -> ZkLogupStatisticalSecurityBudget {
    ZkLogupStatisticalSecurityBudget {
        aggregate_id: 0,
        extension_field_bits: ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS,
        private_lookup_term_count_bound: 1,
        lookup_challenge_count: 1,
        expected_proof_volume: 1,
        safety_margin_bits: 0,
        computed_security_bits: ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS,
        min_statistical_security_bits: 80,
    }
}

fn private_logup_policy() -> (
    ZkPrivacyMap,
    ZkPrivateColumnScope,
    ZkColumnDegreeBound,
    ZkColumnDegreeBound,
) {
    let private_range = ZkColumnRange::new(1, 0, 1);
    let mut privacy_map = ZkPrivacyMap {
        version: ZkProofVersion::V1,
        private_columns: vec![private_range],
        hash: ZkPrivacyMapHash([0; 32]),
    };
    privacy_map.hash = canonical_zk_privacy_map_hash(&privacy_map);

    let mut private_column_scope = ZkPrivateColumnScope {
        version: ZkProofVersion::V1,
        entries: vec![ZkPrivateColumnScopeEntry {
            range: private_range,
            usage: ZkPrivateColumnUsage::LogUp,
            trace_domain_log_size: 4,
            semantic_trace_domain_log_sizes: vec![4],
        }],
        hash: [0; 32],
    };
    private_column_scope.hash = canonical_zk_private_column_scope_hash(&private_column_scope);

    let private_bound = ZkColumnDegreeBound {
        range: private_range,
        log_degree_bound: 5,
    };
    let quotient_bound = ZkColumnDegreeBound {
        range: ZkColumnRange::new(2, 0, 1),
        log_degree_bound: 5,
    };

    (
        privacy_map,
        private_column_scope,
        private_bound,
        quotient_bound,
    )
}

fn metadata(
    privacy_map: &ZkPrivacyMap,
    private_column_scope: &ZkPrivateColumnScope,
    private_bound: ZkColumnDegreeBound,
    quotient_bound: ZkColumnDegreeBound,
    budget_hash: [u8; 32],
) -> ZkPublicMetadata {
    ZkPublicMetadata {
        version: ZkProofVersion::V1,
        privacy_map_hash: privacy_map.hash.clone(),
        public_statement_hash: ZkPublicStatementHash([7; 32]),
        logup_statistical_security_budget_hash: budget_hash,
        degree_profile: ZkDegreeProfile {
            trace_domain_log_size: 4,
            h_witness: 5,
            h_batch: 5,
            fri_first_layer_log_size: 7,
        },
        witness_randomization: ZkWitnessRandomizationProfile {
            h_witness: 5,
            randomizer_space_hash: [11; 32],
            private_column_scope_hash: private_column_scope.hash,
            private_column_degree_bounds: vec![private_bound],
        },
        quotient_integration: ZkQuotientIntegrationProfile {
            h_batch: 5,
            fri_first_layer_log_size: 7,
            split_derivation_hash: [13; 32],
            quotient_degree_bounds: vec![quotient_bound],
        },
    }
}

#[test]
fn logup_statistical_activation_policy_accepts_empty_budget_state() {
    let (privacy_map, private_column_scope, private_bound, quotient_bound) = private_logup_policy();
    let metadata = metadata(
        &privacy_map,
        &private_column_scope,
        private_bound,
        quotient_bound,
        ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH,
    );

    assert_eq!(
        reject_zk_logup_statistical_security_budget_activation(&metadata, &[]),
        Ok(())
    );
}

#[test]
fn logup_statistical_activation_policy_rejects_nonempty_budgets() {
    let (privacy_map, private_column_scope, private_bound, quotient_bound) = private_logup_policy();
    let metadata = metadata(
        &privacy_map,
        &private_column_scope,
        private_bound,
        quotient_bound,
        ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH,
    );

    assert_eq!(
        reject_zk_logup_statistical_security_budget_activation(&metadata, &[logup_budget()]),
        Err(ZkLogupStatisticalAggregatePolicyError::NonEmptySecurityBudgets)
    );
}

#[test]
fn logup_statistical_activation_policy_rejects_nondefault_budget_hash() {
    let (privacy_map, private_column_scope, private_bound, quotient_bound) = private_logup_policy();
    let metadata = metadata(
        &privacy_map,
        &private_column_scope,
        private_bound,
        quotient_bound,
        canonical_zk_logup_statistical_security_budget_hash(&[logup_budget()]),
    );

    assert_eq!(
        reject_zk_logup_statistical_security_budget_activation(&metadata, &[]),
        Err(ZkLogupStatisticalAggregatePolicyError::NonEmptySecurityBudgetHash)
    );
}

#[test]
fn verifier_config_rejects_logup_budget_hash_mismatch() {
    let (privacy_map, private_column_scope, private_bound, quotient_bound) = private_logup_policy();
    let metadata = metadata(
        &privacy_map,
        &private_column_scope,
        private_bound,
        quotient_bound,
        ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH,
    );
    let verifier_config = ZkVerificationConfig {
        column_degree_bounds: expected_zk_column_degree_bounds(&metadata),
        metadata: metadata.clone(),
        quotient_split_mask_profile: None,
        logup_statistical_security_budgets: vec![logup_budget()],
    };

    assert_eq!(
        validate_zk_public_metadata_against_verifier_config(&metadata, &verifier_config),
        Err(ZkVerificationConfigValidationError::LogupStatisticalSecurityBudgetHashMismatch)
    );
}

#[test]
fn verifier_audit_rejects_private_logup_without_budget() {
    let (privacy_map, private_column_scope, private_bound, quotient_bound) = private_logup_policy();
    let metadata = metadata(
        &privacy_map,
        &private_column_scope,
        private_bound,
        quotient_bound,
        canonical_zk_logup_statistical_security_budget_hash(&[]),
    );
    let verifier_config = ZkVerificationConfig {
        column_degree_bounds: expected_zk_column_degree_bounds(&metadata),
        metadata,
        quotient_split_mask_profile: None,
        logup_statistical_security_budgets: vec![],
    };
    let audit = ZkWitnessRandomizationVerifierAudit {
        privacy_map,
        private_column_scope,
    };
    let sampled_points: TreeVec<Vec<Vec<CirclePoint<SecureField>>>> = TreeVec(vec![]);

    assert_eq!(
        validate_zk_witness_randomization_audit_for_verifier(
            &verifier_config,
            Some(&audit),
            &sampled_points,
            &[],
            7,
        ),
        Err(ZkWitnessRandomizationVerifierAuditError::MissingPrivateLogupStatisticalSecurityBudget)
    );
}
