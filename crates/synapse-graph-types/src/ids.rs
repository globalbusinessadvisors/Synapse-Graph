use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
            pub struct $name(pub u64);
        )+
    };
}

define_id! {
    /// Unique identifier for an embedding vector.
    EmbeddingId,
    /// Unique identifier for an input channel.
    ChannelId,
    /// Unique identifier for a processing session.
    SessionId,
    /// Unique identifier for a graph node.
    NodeId,
    /// Unique identifier for a cluster.
    ClusterId,
    /// Unique identifier for a shortcut edge.
    ShortcutId,
    /// Unique identifier for a prediction.
    PredictionId,
    /// Unique identifier for an event.
    EventId,
    /// Unique identifier for a temporal pattern.
    PatternId,
    /// Unique identifier for an intent.
    IntentId,
    /// Unique identifier for a graph layer.
    LayerId,
    /// Unique identifier for a vector memory collection.
    CollectionId,
}
