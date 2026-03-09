use core::fmt;
use serde::{Deserialize, Serialize};

/// Error produced by a [`BciAdapter`](crate) when translating raw bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterError {
    /// The input buffer was too short or malformed.
    InvalidFormat(alloc::string::String),
    /// A required header field was missing.
    MissingHeader,
    /// CRC / checksum mismatch.
    ChecksumMismatch,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::MissingHeader => write!(f, "missing header"),
            Self::ChecksumMismatch => write!(f, "checksum mismatch"),
        }
    }
}
