#![no_std]

extern crate alloc;

mod ids;
mod temporal;
mod spike;
mod coherence;
mod gated;
mod channel;
mod events;
mod processing;
mod adapter;
mod memory;
mod temporal_types;
mod sequences;
mod slow_types;
mod cognitive;
mod prediction;
mod adaptation;
mod provenance;

pub use ids::*;
pub use temporal::*;
pub use spike::*;
pub use coherence::*;
pub use gated::*;
pub use channel::*;
pub use events::*;
pub use processing::*;
pub use adapter::*;
pub use memory::*;
pub use temporal_types::*;
pub use sequences::*;
pub use slow_types::*;
pub use cognitive::*;
pub use prediction::*;
pub use adaptation::*;
pub use provenance::*;

use serde::{Deserialize, Serialize};

/// Microseconds since epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

/// Position in 3-D embedding space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cfg(test)]
mod tests;
