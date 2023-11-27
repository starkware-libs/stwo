use serde::{Deserialize, Serialize};

use super::expr::{MultiVariatePolyExpression, UnivariatePolyExpression};
use super::generation::GenerationSchedule;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Component {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub instances: Vec<ComponentInstance>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentInstance {
    pub n_bits: u32,
    pub columns: Vec<Column>,
    pub constraints: Vec<UnivariateConstraint>,
    pub interaction_elements: Vec<InteractionElement>,
    pub trace_generation: GenerationSchedule,
    pub outputs: Vec<ComponentOutput>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub description: String,
    pub n_bits: u32,
    pub kind: ColumnKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ColumnKind {
    Witness,
    GKR(MultiVariatePolyExpression),
    IntermediateUnivariate(UnivariatePolyExpression),
    IntermediateMultivariate(MultiVariatePolyExpression),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateConstraint {
    pub name: String,
    pub description: String,
    pub domain: UnivariateDomain,
    pub expr: UnivariatePolyExpression,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateDomain;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractionElement {
    pub name: String,
    pub description: String,
    pub column_dependencies: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentOutput {
    pub name: String,
    pub description: String,
    pub column: String,
    pub offset: u64,
    pub step: u64,
}
