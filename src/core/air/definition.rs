use serde::{Deserialize, Serialize};

use super::expr::{MultiVariatePolyExpression, UnivariatePolyExpression};
use super::generation::SubcolumnGeneration;
use super::slice::SliceDomain;

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
    /// A constant column. A special case is a scalar public input, which is the same as a constant
    /// column with the same value everywhere.
    /// Valid in any expression.
    Constant,

    /// A column that gets committed.
    /// Valid in any expression.
    Witness {
        generation: Vec<SubcolumnGeneration>,
    },

    /// A intermediate value in the computation. This isn't saved in memory as an entire column,
    /// only as a "temporary" cell expression.
    /// Valid in any univariate or generation expression.
    IntermediateUnivariate(UnivariatePolyExpression),

    /// Ephemeral values used only as inputs for the trace generation of the current component.
    /// Valid only in generation expressions.
    GenerationInput,

    /// A column proved using the GKR protocol
    /// Valid in GKR expressions.
    GKR(MultiVariatePolyExpression),

    /// An intermediate value in a GKR computaion.
    /// Valid in GKR expressions.
    IntermediateMultivariate(MultiVariatePolyExpression),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateConstraint {
    pub name: String,
    pub description: String,
    pub domain: SliceDomain,
    pub expr: UnivariatePolyExpression,
}

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
