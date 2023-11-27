use serde::{Deserialize, Serialize};

use super::expr::{UnivariateMaskItem, UnivariatePolyExpression};
use super::slice::SliceDomain;

/// Instructions for generating the component trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubcolumnGeneration {
    pub column: String,
    pub domain: SliceDomain,

    pub kind: GenerationFormula,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Subcolumn {
    pub offset: u64,
    pub log_steps: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GenerationFormula {
    Explicit(UnivariatePolyExpression),
    External {
        name: String,
        mask: Vec<UnivariateMaskItem>,
    },
}
