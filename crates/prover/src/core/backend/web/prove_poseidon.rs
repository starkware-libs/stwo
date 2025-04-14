use std::any::Any;
use std::borrow::Cow;

use itertools::Itertools;
use tracing::{span, Level};

use super::WebBackend;
use crate::constraint_framework::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};
use crate::core::air::accumulation::DomainEvaluationAccumulator;
use crate::core::air::{Component, Trace};
use crate::core::backend::web::webgpu::compute_composition_polynomial_original_trace::compute_composition_polynomial_original_trace_gpu;
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::utils;
use crate::examples::poseidon::PoseidonComponent;

#[allow(unused_mut)]
#[allow(unused_variables)]
pub async fn prove_gpu<MC: MerkleChannel, E: FrameworkEval + 'static>(
    components: &[&FrameworkComponent<E>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, WebBackend, MC>,
) where
    WebBackend: BackendForChannel<MC>,
{
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let trace = commitment_scheme.trace();

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition").entered();
    let span1 = span!(Level::INFO, "Generation").entered();
    // let composition_poly =
    compute_composition_polynomial_gpu(components, random_coeff, &trace).await;
    span1.exit();

    // let mut tree_builder = commitment_scheme.tree_builder();
    // tree_builder.extend_polys(composition_poly.into_coordinate_polys());
    // tree_builder.commit(channel);
    // span.exit();
}

#[allow(unused_variables)]
async fn compute_composition_polynomial_gpu<'a, MC: MerkleChannel, E: FrameworkEval + 'static>(
    components: &'a [&'a FrameworkComponent<E>],
    random_coeff: SecureField,
    trace: &'a Trace<'a, WebBackend>,
)
// -> SecureCirclePoly<SimdBackend>
where
    WebBackend: BackendForChannel<MC>,
{
    let total_constraints: usize = components.iter().map(|c| c.n_constraints()).sum();
    let composition_log_degree_bound = components
        .iter()
        .map(|c| c.max_constraint_log_degree_bound())
        .max()
        .unwrap();
    #[allow(unused_mut)]
    let mut accumulator = DomainEvaluationAccumulator::new(
        random_coeff,
        composition_log_degree_bound,
        total_constraints,
    );

    for &component in components {
        evaluate_constraint_quotients_on_domain_gpu(component, trace, &mut accumulator).await;
    }
    // accumulator.finalize()
}

#[allow(unused_variables)]
async fn evaluate_constraint_quotients_on_domain_gpu<'a, E: FrameworkEval + Any + 'static>(
    component: &'a FrameworkComponent<E>,
    trace: &'a Trace<'a, WebBackend>,
    evaluation_accumulator: &mut DomainEvaluationAccumulator<WebBackend>,
) {
    if component.n_constraints() == 0 {
        return;
    }

    let eval_domain =
        CanonicCoset::new(component.max_constraint_log_degree_bound()).circle_domain();
    let trace_domain = CanonicCoset::new(component.eval().log_size());

    let mut component_polys = trace.polys.sub_tree(&component.trace_locations());
    component_polys[PREPROCESSED_TRACE_IDX] = component
        .preproccessed_column_indices()
        .iter()
        .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
        .collect();

    let mut component_evals = trace.evals.sub_tree(&component.trace_locations());
    component_evals[PREPROCESSED_TRACE_IDX] = component
        .preproccessed_column_indices()
        .iter()
        .map(|idx| &trace.evals[PREPROCESSED_TRACE_IDX][*idx])
        .collect();

    // Extend trace if necessary.
    // TODO: Don't extend when eval_size < committed_size. Instead, pick a good
    // subdomain. (For larger blowup factors).
    let need_to_extend = component_evals
        .iter()
        .flatten()
        .any(|c| c.domain != eval_domain);
    #[cfg(target_family = "wasm")]
    let cpu_trace_extend_start = web_sys::window().unwrap().performance().unwrap().now();
    let trace: TreeVec<Vec<Cow<'_, CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>> =
        if need_to_extend {
            let _span = span!(Level::INFO, "Extension").entered();
            let twiddles = WebBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.evaluate_with_twiddles(eval_domain, &twiddles)))
        } else {
            component_evals.clone().map_cols(|c| Cow::Borrowed(*c))
        };
    #[cfg(target_family = "wasm")]
    let cpu_trace_extend_end = web_sys::window().unwrap().performance().unwrap().now();
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(
        &format!(
            "Time spent on CPU trace extend: {:?}ms",
            cpu_trace_extend_end - cpu_trace_extend_start
        )
        .into(),
    );

    // Denom inverses.
    let log_expand = eval_domain.log_size() - trace_domain.log_size();
    let mut denom_inv = (0..1 << log_expand)
        .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
        .collect_vec();

    pub fn bit_reverse<T>(v: &mut [T]) {
        let n = v.len();
        assert!(n.is_power_of_two());
        let log_n = n.ilog2();
        for i in 0..n {
            let j = utils::bit_reverse_index(i, log_n);
            if j > i {
                v.swap(i, j);
            }
        }
    }

    bit_reverse(&mut denom_inv);

    // Accumulator.
    let [mut accum] =
        evaluation_accumulator.columns([(eval_domain.log_size(), component.n_constraints())]);
    accum.random_coeff_powers.reverse();

    let trace_cols = trace.as_cols_ref().map_cols(|c| c.to_cpu());
    let trace_cols = trace_cols.as_cols_ref();

    #[cfg(target_family = "wasm")]
    let gpu_start = web_sys::window().unwrap().performance().unwrap().now();

    // if let Some(poseidon_component) = (component as &dyn Any).downcast_ref::<PoseidonComponent>()
    // {     let gpu_results = compute_composition_polynomial_gpu_poseidon(
    //         trace_cols,
    //         denom_inv.clone(),
    //         accum.random_coeff_powers.clone(),
    //         poseidon_component.eval().lookup_elements.clone(),
    //         trace_domain.log_size(),
    //         eval_domain.log_size(),
    //         poseidon_component.eval().total_sum,
    //     )
    //     .await;
    // }

    if let Some(poseidon_component) = (component as &dyn Any).downcast_ref::<PoseidonComponent>() {
        let gpu_results = compute_composition_polynomial_original_trace_gpu(
            component_polys,
            eval_domain,
            denom_inv.clone(),
            accum.random_coeff_powers.clone(),
            poseidon_component.eval().lookup_elements.clone(),
            trace_domain.log_size(),
            eval_domain.log_size(),
            poseidon_component.eval().claimed_sum,
        )
        .await;
    }

    #[cfg(target_family = "wasm")]
    let gpu_end = web_sys::window().unwrap().performance().unwrap().now();
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&format!("Time spent on GPU: {:?}ms", gpu_end - gpu_start).into());
}
