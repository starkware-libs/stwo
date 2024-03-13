use core::slice;
use std::collections::BTreeMap;
use std::iter::zip;

use itertools::Itertools;

use super::evaluation::SECURE_EXTENSION_DEGREE;
use super::{Air, Component, ComponentTrace, ComponentVisitor};
use crate::core::air::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::backend::{Backend, CPUBackend};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::CirclePolyDegreeBound;
use crate::core::poly::circle::{CanonicCoset, CirclePoly, SecureCirclePoly};
use crate::core::prover::LOG_BLOWUP_FACTOR;
use crate::core::{ColumnVec, ComponentVec};

pub trait AirExt: Air<CPUBackend> {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        struct MaxConstraintLogDegreeBoundVisitor(u32);

        impl<B: Backend> ComponentVisitor<B> for MaxConstraintLogDegreeBoundVisitor {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                self.0 = self.0.max(component.max_constraint_log_degree_bound());
            }
        }

        let mut visitor = MaxConstraintLogDegreeBoundVisitor(0);
        self.visit_components(&mut visitor);
        visitor.0
    }

    #[inline(never)]
    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, CPUBackend>],
    ) -> SecureCirclePoly {
        pub struct ConstraintEvaluator<'a, B: Backend> {
            component_traces: slice::Iter<'a, ComponentTrace<'a, B>>,
            evaluation_accumulator: DomainEvaluationAccumulator<B>,
        }

        impl<'a, B: Backend> ComponentVisitor<B> for ConstraintEvaluator<'a, B> {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                component.evaluate_constraint_quotients_on_domain(
                    self.component_traces
                        .next()
                        .expect("no more component traces"),
                    &mut self.evaluation_accumulator,
                )
            }
        }

        let mut evaluator = {
            let max_log_size = self.max_constraint_log_degree_bound();
            ConstraintEvaluator::<CPUBackend> {
                component_traces: component_traces.iter(),
                evaluation_accumulator: DomainEvaluationAccumulator::new(
                    random_coeff,
                    max_log_size,
                ),
            }
        };
        self.visit_components(&mut evaluator);
        evaluator.evaluation_accumulator.finalize()
    }

    fn mask_points_and_values(
        &self,
        point: CirclePoint<SecureField>,
        component_traces: &[ComponentTrace<'_, CPUBackend>],
    ) -> (
        ComponentVec<Vec<CirclePoint<SecureField>>>,
        ComponentVec<Vec<SecureField>>,
    ) {
        struct MaskEvaluator<'a, B: Backend> {
            point: CirclePoint<SecureField>,
            component_traces: slice::Iter<'a, ComponentTrace<'a, B>>,
            component_points: ComponentVec<Vec<CirclePoint<SecureField>>>,
            component_values: ComponentVec<Vec<SecureField>>,
        }

        impl<'a, B: Backend> ComponentVisitor<B> for MaskEvaluator<'a, B> {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                let trace = self.component_traces.next().unwrap();
                let (points, values) = component.mask_points_and_values(self.point, trace);
                self.component_points.push(points);
                self.component_values.push(values);
            }
        }

        let mut visitor = MaskEvaluator::<CPUBackend> {
            point,
            component_traces: component_traces.iter(),
            component_points: Vec::new(),
            component_values: Vec::new(),
        };
        self.visit_components(&mut visitor);
        (visitor.component_points, visitor.component_values)
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ComponentVec<Vec<CirclePoint<SecureField>>> {
        struct MaskPointsEvaluator {
            point: CirclePoint<SecureField>,
            points: ComponentVec<Vec<CirclePoint<SecureField>>>,
        }

        impl<B: Backend> ComponentVisitor<B> for MaskPointsEvaluator {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                let domains = component
                    .trace_log_degree_bounds()
                    .iter()
                    .map(|&log_size| CanonicCoset::new(log_size))
                    .collect_vec();
                self.points
                    .push(component.mask().to_points(&domains, self.point));
            }
        }

        let mut visitor = MaskPointsEvaluator {
            point,
            points: Vec::new(),
        };
        self.visit_components(&mut visitor);
        visitor.points
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &ComponentVec<Vec<SecureField>>,
        random_coeff: SecureField,
    ) -> SecureField {
        struct ConstraintPointEvaluator<'a> {
            point: CirclePoint<SecureField>,
            mask_values: slice::Iter<'a, ColumnVec<Vec<SecureField>>>,
            evaluation_accumulator: PointEvaluationAccumulator,
        }

        impl<'a, B: Backend> ComponentVisitor<B> for ConstraintPointEvaluator<'a> {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                component.evaluate_quotients_by_mask(
                    self.point,
                    self.mask_values
                        .next()
                        .expect("no more component mask values"),
                    &mut self.evaluation_accumulator,
                )
            }
        }

        let mut evaluator = {
            ConstraintPointEvaluator {
                point,
                mask_values: mask_values.iter(),
                evaluation_accumulator: PointEvaluationAccumulator::new(
                    random_coeff,
                    self.max_constraint_log_degree_bound(),
                ),
            }
        };
        self.visit_components(&mut evaluator);
        evaluator.evaluation_accumulator.finalize()
    }

    fn quotient_log_bounds(&self) -> Vec<CirclePolyDegreeBound> {
        struct QuotientLogBoundsVisitor {
            // Maps the log degree bound to the number of quotients with that bound.
            bounds: BTreeMap<u32, usize>,
        }

        impl<B: Backend> ComponentVisitor<B> for QuotientLogBoundsVisitor {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                for (mask_points, trace_bound) in zip(
                    component.mask().iter(),
                    &component.trace_log_degree_bounds(),
                ) {
                    let n = self.bounds.entry(*trace_bound);
                    *n.or_default() += mask_points.len();
                }
            }
        }

        let mut bounds_visitor = QuotientLogBoundsVisitor {
            bounds: BTreeMap::new(),
        };
        self.visit_components(&mut bounds_visitor);
        let mut bounds = bounds_visitor
            .bounds
            .into_iter()
            .flat_map(|(bound, n)| (0..n).map(|_| bound).collect_vec())
            .collect_vec();
        // Add the composition polynomial's log degree bounds.
        bounds.extend([self.max_constraint_log_degree_bound(); SECURE_EXTENSION_DEGREE]);
        bounds
            .into_iter()
            .rev()
            .map(CirclePolyDegreeBound::new)
            .collect()
    }

    /// Returns the log degree bounds of the quotient polynomials in descending order.
    fn commitment_domains(&self) -> Vec<CanonicCoset> {
        struct DomainsVisitor {
            domains: Vec<CanonicCoset>,
        }

        impl<B: Backend> ComponentVisitor<B> for DomainsVisitor {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                self.domains.extend(
                    component
                        .trace_log_degree_bounds()
                        .iter()
                        .map(|&log_size| CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR)),
                );
            }
        }

        let mut domains_visitor = DomainsVisitor {
            domains: Vec::new(),
        };
        self.visit_components(&mut domains_visitor);
        // Add the composition polynomial's domain.
        domains_visitor.domains.push(CanonicCoset::new(
            self.max_constraint_log_degree_bound() + LOG_BLOWUP_FACTOR,
        ));
        domains_visitor.domains
    }

    fn component_traces<'a>(
        &'a self,
        polynomials: &'a [CirclePoly<CPUBackend>],
    ) -> Vec<ComponentTrace<'_, CPUBackend>> {
        struct ComponentTracesVisitor<'a, B: Backend> {
            polynomials: slice::Iter<'a, CirclePoly<B>>,
            component_traces: Vec<ComponentTrace<'a, B>>,
        }

        impl<'a, B: Backend> ComponentVisitor<B> for ComponentTracesVisitor<'a, B> {
            fn visit<C: Component<B>>(&mut self, component: &C) {
                let n_columns = component.trace_log_degree_bounds().len();
                let columns = (&mut self.polynomials).take(n_columns).collect();
                self.component_traces.push(ComponentTrace::new(columns));
            }
        }

        let mut component_traces_visitor = ComponentTracesVisitor::<CPUBackend> {
            polynomials: polynomials.iter(),
            component_traces: Vec::new(),
        };
        self.visit_components(&mut component_traces_visitor);
        component_traces_visitor.component_traces
    }
}

impl<A: Air<CPUBackend>> AirExt for A {}
