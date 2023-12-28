use serde::{Deserialize, Serialize};

use super::graph::GraphNode;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Component {
    // TODO: Share inputs and outputs between instances.
    pub name: String,
    pub version: String,
    pub description: String,
    pub instances: Vec<ComponentInstance>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentInstance {
    pub n_bits: u32,

    // Generation.
    pub generation_graph: Vec<GraphNode>,
    pub columns: Vec<Column>,
    pub outputs: Vec<String>,

    // Constraints.
    pub constraint_graph: Vec<GraphNode>,
    pub constraints: Vec<Constraint>,

    pub interaction_elements: Vec<InteractionElement>,
}

impl ComponentInstance {
    pub fn get_generation_node(&self, name: &str) -> (usize, &GraphNode) {
        self.generation_graph
            .iter()
            .enumerate()
            .find(|(_index, n)| n.name == name)
            .unwrap()
    }

    pub fn get_node_index(self, name: &str) -> usize {
        self.generation_graph
            .iter()
            .enumerate()
            .find(|(_index, n)| n.name == name)
            .unwrap()
            .0.clone()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub description: String,
    /// Name of the node in the generation graph that generates this column.
    pub generation_node: String,
    pub kind: ColumnKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ColumnKind {
    Precomputed,
    Witness,
    GKR,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub description: String,
    /// Name of the node in the constraint graph that evaluates this constraint.
    /// Must be of size that is a power of 2.
    pub constraint_node: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractionElement {
    pub name: String,
    pub description: String,
    // TODO(spapini): Dependencies on GKR rounds.
    pub witness_dependencies: Vec<String>,
}
