//! Tracing module for performance monitoring and span tracking.
//!
//! This module provides functionality to collect and analyze timing data for various
//! operations within the prover.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use itertools::Itertools;
use tracing::span::Attributes;
use tracing::{Id, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

pub struct SpanData {
    label: String,
    start: Instant,
}

#[derive(Clone, Default)]
pub struct SpanAccumulator {
    /// Active spans being tracked.
    spans: Arc<Mutex<HashMap<Id, SpanData>>>,
    /// Accumulated timing results.
    results: Arc<Mutex<HashMap<String, Duration>>>,
}
impl SpanAccumulator {
    /// Exports the collected timing data as a CSV string.
    ///
    /// # Returns
    ///
    /// A formatted string with two columns:
    /// - Label: The name of the span
    /// - Duration_ms: The total time spent in spans with that label, in milliseconds
    pub fn export_csv(&self) -> String {
        let mut out = String::from("Label,Duration_ms\n");
        for (label, duration) in self
            .results
            .lock()
            .unwrap()
            .iter()
            .sorted_by_key(|(label, _)| *label)
        {
            out.push_str(&format!("{},{}\n", label, duration.as_secs_f64() * 1000.0));
        }
        out
    }
}

impl<S> Layer<S> for SpanAccumulator
where
    S: Subscriber,
    S: for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let meta = ctx.metadata(id).unwrap();
        let label = meta.name().to_string();

        let span_data = SpanData {
            label,
            // Start timing the span.
            start: Instant::now(),
        };

        // Assumes Ids are unique.
        self.spans.lock().unwrap().insert(id.clone(), span_data);
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        // Add the elapsed time to the accumulated duration for this label.
        let mut spans = self.spans.lock().unwrap();
        if let Some(span) = spans.remove(&id) {
            let mut results = self.results.lock().unwrap();
            let key = span.label;
            let entry = results.entry(key).or_insert(Duration::ZERO);

            *entry += span.start.elapsed();
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Registry;

    use super::*;

    #[test]
    fn test_span_accumulator() {
        let collector = SpanAccumulator::default();
        let layer = collector.clone();
        let subscriber = Registry::default().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let span1 = tracing::span!(tracing::Level::INFO, "span1").entered();
        let span2 = tracing::span!(tracing::Level::INFO, "span2").entered();
        drop(span2);
        drop(span1);

        let results = collector.results.lock().unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains_key("span1"));
        assert!(results.contains_key("span2"));
    }
}
