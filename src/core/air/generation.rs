use serde::{Deserialize, Serialize};

use super::expr::UnivariatePolyExpression;

/// Instructions for generating the component trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationSchedule {
    pub kind: GenerationScheduleKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GenerationScheduleKind {
    Cell(CellGeneration),
    Repeat {
        n_repeats: u64,
        subschedule: Box<GenerationSchedule>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellGeneration {
    pub column: String,
    pub offset: u64,
    pub kind: TraceGenerationStepKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TraceGenerationStepKind {
    Explicit(UnivariatePolyExpression),
    Input {
        name: String,
    },
    FirstOr {
        on_first: Box<TraceGenerationStepKind>,
        on_rest: Box<TraceGenerationStepKind>,
    },
    External {
        name: String,
    },
}
