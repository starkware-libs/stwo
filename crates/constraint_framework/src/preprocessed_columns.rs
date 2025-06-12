/// Used for comparing preprocessed columns.
/// Column IDs must be unique in a given context.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreProcessedColumnId {
    pub id: String,
}
