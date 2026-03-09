//! ShortcutMatcher — matches partial patterns to cognitive shortcuts (DDD-005).
//!
//! Uses VectorMemory KNN search and the ShortcutRegistry to find the most
//! relevant shortcuts for a given input pattern.

use synapse_graph_cognitive::graph::ShortcutRegistry;
use synapse_graph_cognitive::hypergraph::HypergraphStore;
use synapse_graph_memory::store::VectorMemory;
use synapse_graph_types::{GatedEmbedding, MatchedShortcut};

/// Configuration for the ShortcutMatcher.
#[derive(Clone, Debug)]
pub struct MatcherConfig {
    /// Number of KNN results to retrieve from VectorMemory. Default: 10.
    pub top_k: usize,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self { top_k: 10 }
    }
}

/// Matches partial patterns to cognitive shortcuts via KNN search.
pub struct ShortcutMatcher {
    config: MatcherConfig,
}

impl ShortcutMatcher {
    pub fn new(config: MatcherConfig) -> Self {
        Self { config }
    }

    /// Match a partial pattern to shortcuts.
    ///
    /// Pipeline:
    /// a) Search KNN in VectorMemory for nearest embeddings.
    /// b) Map those embeddings to shortcuts via the registry.
    /// c) Sort by distance ascending, return top matches.
    pub fn match_shortcuts(
        &self,
        partial_pattern: &GatedEmbedding,
        memory: &VectorMemory,
        registry: &ShortcutRegistry,
        hypergraph: &HypergraphStore,
    ) -> Vec<MatchedShortcut> {
        // (a) KNN search.
        let candidates = memory.search_knn(partial_pattern.vector(), self.config.top_k);

        // (b) Map candidates to shortcuts.
        let matched = registry.find_matching(&candidates, hypergraph);

        // (c) Convert to MatchedShortcut and sort by distance.
        let mut results: Vec<MatchedShortcut> = matched
            .into_iter()
            .map(|(shortcut, score)| MatchedShortcut {
                shortcut,
                distance: 1.0 - score, // convert similarity to distance
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}
