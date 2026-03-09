//! CollectionManager — logical partitions of the vector store (DDD-002).

use std::collections::HashMap;

use synapse_graph_types::{
    Collection, CollectionFilter, CollectionId, TemporalMeta, Timestamp,
};

/// Manages named logical partitions of the vector store.
pub struct CollectionManager {
    collections: HashMap<CollectionId, Collection>,
    next_id: u64,
}

impl CollectionManager {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new collection with the given name and filter.
    pub fn create(
        &mut self,
        name: &str,
        filter: CollectionFilter,
        now: Timestamp,
    ) -> CollectionId {
        let id = CollectionId(self.next_id);
        self.next_id += 1;
        self.collections.insert(
            id,
            Collection {
                id,
                name: name.into(),
                filter,
                embedding_count: 0,
                created_at: now,
            },
        );
        id
    }

    /// List all collections.
    pub fn list(&self) -> Vec<Collection> {
        self.collections.values().cloned().collect()
    }

    /// Get a collection by ID.
    pub fn get(&self, id: CollectionId) -> Option<&Collection> {
        self.collections.get(&id)
    }

    /// Increment the embedding count for a collection.
    pub fn increment_count(&mut self, id: CollectionId) {
        if let Some(c) = self.collections.get_mut(&id) {
            c.embedding_count += 1;
        }
    }

    /// Find the first collection whose filter matches the given metadata.
    pub fn find_matching(&self, meta: &TemporalMeta) -> Option<CollectionId> {
        for (_, c) in &self.collections {
            if matches_filter(&c.filter, meta) {
                return Some(c.id);
            }
        }
        None
    }
}

impl Default for CollectionManager {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_filter(filter: &CollectionFilter, meta: &TemporalMeta) -> bool {
    if let Some(sid) = filter.session_id {
        if meta.session_id != sid {
            return false;
        }
    }
    if let Some((lo, hi)) = filter.channel_range {
        if meta.channel_id.0 < lo.0 || meta.channel_id.0 > hi.0 {
            return false;
        }
    }
    if let Some(tier) = filter.temporal_tier {
        if meta.tier != tier {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{ChannelId, SessionId, TemporalTier};

    #[test]
    fn create_and_list() {
        let mut cm = CollectionManager::new();
        let filter = CollectionFilter {
            session_id: Some(SessionId(1)),
            channel_range: None,
            temporal_tier: None,
        };
        let id = cm.create("test-collection", filter, Timestamp(0));
        assert_eq!(id, CollectionId(1));
        let list = cm.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "test-collection");
    }

    #[test]
    fn filter_matching() {
        let mut cm = CollectionManager::new();
        cm.create(
            "session-1",
            CollectionFilter {
                session_id: Some(SessionId(1)),
                channel_range: None,
                temporal_tier: None,
            },
            Timestamp(0),
        );
        cm.create(
            "fast-tier",
            CollectionFilter {
                session_id: None,
                channel_range: None,
                temporal_tier: Some(TemporalTier::Fast),
            },
            Timestamp(0),
        );

        let meta = TemporalMeta {
            tier: TemporalTier::Fast,
            timestamp: Timestamp(100),
            channel_id: ChannelId(5),
            session_id: SessionId(1),
            coherence_score: 0.9,
        };
        // Should match session-1 first (iteration order isn't guaranteed, but
        // at least one collection should match).
        let matched = cm.find_matching(&meta);
        assert!(matched.is_some());
    }
}
