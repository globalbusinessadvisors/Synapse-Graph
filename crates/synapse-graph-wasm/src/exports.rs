//! WASM-exported functions for the SynapseGraph module (ADR-006).
//!
//! These functions use C ABI exports compatible with both wasm32-wasi
//! and wasm32-unknown-unknown targets. All data crosses the WASM
//! boundary as JSON-encoded byte arrays for maximum interoperability.
//!
//! # Safety
//!
//! All exported functions that take `*mut SynapseGraph` require a valid,
//! non-null pointer obtained from `init_synapse_graph`. The pointer must
//! not be used after calling `free_synapse_graph`.

use synapse_graph_core::orchestrator::SynapseGraph;
use synapse_graph_types::{IntentId, PredictionFeedback, PredictionId, SessionId, SpikeBatch};

use crate::config::WasmConfig;

// ---------------------------------------------------------------------------
// WASM memory helpers
// ---------------------------------------------------------------------------

/// Allocate `size` bytes of memory for the host to write into.
///
/// Returns a pointer to the allocated buffer. The host must write data
/// into this buffer before calling functions that consume it.
#[no_mangle]
pub extern "C" fn wasm_alloc(size: u32) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 8)
        .expect("invalid allocation size");
    unsafe { std::alloc::alloc(layout) }
}

/// Free a buffer previously allocated by `wasm_alloc`.
#[no_mangle]
pub extern "C" fn wasm_dealloc(ptr: *mut u8, size: u32) {
    if ptr.is_null() {
        return;
    }
    let layout = std::alloc::Layout::from_size_align(size as usize, 8)
        .expect("invalid allocation size");
    unsafe { std::alloc::dealloc(ptr, layout) }
}

// ---------------------------------------------------------------------------
// Result buffer management
// ---------------------------------------------------------------------------

/// A result buffer that holds serialized output data.
///
/// The WASM host reads from this after each call, then frees it.
#[repr(C)]
pub struct WasmResult {
    pub ptr: *mut u8,
    pub len: u32,
}

/// Serialize a value to JSON and return it as a WasmResult.
fn to_result<T: serde::Serialize>(value: &T) -> WasmResult {
    match serde_json::to_vec(value) {
        Ok(bytes) => {
            let len = bytes.len() as u32;
            let ptr = bytes.as_ptr() as *mut u8;
            std::mem::forget(bytes); // host will free via wasm_dealloc
            WasmResult { ptr, len }
        }
        Err(_) => WasmResult {
            ptr: std::ptr::null_mut(),
            len: 0,
        },
    }
}

/// Free a WasmResult buffer.
#[no_mangle]
pub extern "C" fn wasm_free_result(result: WasmResult) {
    if !result.ptr.is_null() && result.len > 0 {
        let bytes = unsafe {
            Vec::from_raw_parts(result.ptr, result.len as usize, result.len as usize)
        };
        drop(bytes);
    }
}

// ---------------------------------------------------------------------------
// Lifecycle functions
// ---------------------------------------------------------------------------

/// Initialize a new SynapseGraph instance from a JSON configuration.
///
/// The `config_ptr` and `config_len` point to a UTF-8 JSON string.
/// Returns a pointer to the heap-allocated SynapseGraph.
/// Returns null on parse failure.
#[no_mangle]
pub extern "C" fn init_synapse_graph(config_ptr: *const u8, config_len: u32) -> *mut SynapseGraph {
    let config_bytes = unsafe { std::slice::from_raw_parts(config_ptr, config_len as usize) };
    let config_str = match std::str::from_utf8(config_bytes) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    init_from_json(config_str)
}

/// Internal: parse JSON config and create SynapseGraph.
pub(crate) fn init_from_json(config_json: &str) -> *mut SynapseGraph {
    let wasm_config: WasmConfig = match serde_json::from_str(config_json) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };

    let sg_config = wasm_config.into_synapse_config();
    let sg = SynapseGraph::new(sg_config);
    Box::into_raw(Box::new(sg))
}

/// Free a SynapseGraph instance.
///
/// # Safety
///
/// `sg` must be a valid pointer from `init_synapse_graph` and must not
/// be used after this call.
#[no_mangle]
pub extern "C" fn free_synapse_graph(sg: *mut SynapseGraph) {
    if !sg.is_null() {
        unsafe {
            drop(Box::from_raw(sg));
        }
    }
}

// ---------------------------------------------------------------------------
// Operation exports
// ---------------------------------------------------------------------------

/// Ingest a spike batch.
///
/// `batch_ptr` / `batch_len` point to a JSON-encoded SpikeBatch.
/// Returns a WasmResult containing JSON-encoded IngestResult.
#[no_mangle]
pub extern "C" fn wasm_ingest(
    sg: *mut SynapseGraph,
    batch_ptr: *const u8,
    batch_len: u32,
) -> WasmResult {
    let sg = unsafe { &mut *sg };
    let batch_bytes = unsafe { std::slice::from_raw_parts(batch_ptr, batch_len as usize) };

    let batch: SpikeBatch = match serde_json::from_slice(batch_bytes) {
        Ok(b) => b,
        Err(_) => return WasmResult { ptr: std::ptr::null_mut(), len: 0 },
    };

    let result = sg.ingest(batch);
    to_result(&result)
}

/// Run intent prediction on a partial pattern.
///
/// `pattern_ptr` / `pattern_len` point to a JSON-encoded GatedEmbedding.
/// Returns a WasmResult containing JSON-encoded IntentPrediction.
#[no_mangle]
pub extern "C" fn wasm_predict(
    sg: *mut SynapseGraph,
    pattern_ptr: *const u8,
    pattern_len: u32,
) -> WasmResult {
    let sg = unsafe { &mut *sg };
    let pattern_bytes = unsafe { std::slice::from_raw_parts(pattern_ptr, pattern_len as usize) };

    let gated: synapse_graph_types::GatedEmbedding = match serde_json::from_slice(pattern_bytes) {
        Ok(g) => g,
        Err(_) => return WasmResult { ptr: std::ptr::null_mut(), len: 0 },
    };

    let prediction = sg.predict_intent(&gated);
    to_result(&prediction)
}

/// Provide feedback on a prediction.
///
/// `feedback_type`: 0 = Confirmed, 1 = Corrected, 2 = Rejected, 3 = Timeout.
#[no_mangle]
pub extern "C" fn wasm_feedback(
    sg: *mut SynapseGraph,
    prediction_id: u64,
    feedback_type: u8,
) {
    let sg = unsafe { &mut *sg };
    let feedback = match feedback_type {
        0 => PredictionFeedback::Confirmed,
        1 => PredictionFeedback::Corrected { actual: IntentId(0) },
        2 => PredictionFeedback::Rejected,
        _ => PredictionFeedback::Timeout,
    };
    sg.provide_feedback(PredictionId(prediction_id), feedback);
}

/// Run the maintenance cycle.
///
/// Returns a WasmResult containing JSON-encoded MaintenanceSummary.
#[no_mangle]
pub extern "C" fn wasm_maintain(sg: *mut SynapseGraph) -> WasmResult {
    let sg = unsafe { &mut *sg };
    let summary = sg.maintain();
    to_result(&summary)
}

/// Update the cognitive graph from recent memory state.
///
/// Returns a WasmResult containing JSON-encoded GraphUpdateResult.
#[no_mangle]
pub extern "C" fn wasm_update_cognitive_graph(sg: *mut SynapseGraph) -> WasmResult {
    let sg = unsafe { &mut *sg };
    let result = sg.update_cognitive_graph();
    to_result(&result)
}

/// Discover cognitive shortcuts via GNN engine.
///
/// Returns a WasmResult containing JSON-encoded Vec<CognitiveShortcut>.
#[no_mangle]
pub extern "C" fn wasm_discover_shortcuts(sg: *mut SynapseGraph) -> WasmResult {
    let sg = unsafe { &mut *sg };
    let shortcuts = sg.discover_shortcuts();
    to_result(&shortcuts)
}

/// Start a new session.
#[no_mangle]
pub extern "C" fn wasm_start_session(sg: *mut SynapseGraph, session_id: u64) {
    let sg = unsafe { &mut *sg };
    sg.start_session(SessionId(session_id));
}

/// End the current session.
#[no_mangle]
pub extern "C" fn wasm_end_session(sg: *mut SynapseGraph) {
    let sg = unsafe { &mut *sg };
    sg.end_session();
}

// ---------------------------------------------------------------------------
// Safe Rust wrapper (used by native tests)
// ---------------------------------------------------------------------------

/// A safe Rust wrapper around the raw WASM exports.
///
/// Owns the SynapseGraph pointer and drops it on destruction.
/// Used for testing the WASM API surface from native Rust.
pub struct WasmSynapseGraph {
    ptr: *mut SynapseGraph,
}

impl WasmSynapseGraph {
    /// Create a new instance from a JSON config string.
    pub fn new(config_json: &str) -> Option<Self> {
        let ptr = init_from_json(config_json);
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Get a reference to the underlying SynapseGraph.
    pub fn inner(&self) -> &SynapseGraph {
        unsafe { &*self.ptr }
    }

    /// Get a mutable reference to the underlying SynapseGraph.
    pub fn inner_mut(&mut self) -> &mut SynapseGraph {
        unsafe { &mut *self.ptr }
    }

    /// Raw pointer for FFI-style calls.
    pub fn as_ptr(&mut self) -> *mut SynapseGraph {
        self.ptr
    }

    /// Ingest a SpikeBatch via the WASM serialization path.
    pub fn ingest_json(&mut self, batch: &SpikeBatch) -> Option<synapse_graph_core::orchestrator::IngestResult> {
        let batch_json = serde_json::to_vec(batch).ok()?;
        let result = wasm_ingest(self.ptr, batch_json.as_ptr(), batch_json.len() as u32);
        if result.ptr.is_null() {
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(result.ptr, result.len as usize) };
        let parsed = serde_json::from_slice(bytes).ok();
        wasm_free_result(result);
        parsed
    }

    /// Provide feedback via the WASM API.
    pub fn feedback(&mut self, prediction_id: u64, feedback_type: u8) {
        wasm_feedback(self.ptr, prediction_id, feedback_type);
    }

    /// Run maintenance via the WASM API.
    pub fn maintain_json(&mut self) -> Option<synapse_graph_core::orchestrator::MaintenanceSummary> {
        let result = wasm_maintain(self.ptr);
        if result.ptr.is_null() {
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(result.ptr, result.len as usize) };
        let parsed = serde_json::from_slice(bytes).ok();
        wasm_free_result(result);
        parsed
    }

    /// Discover shortcuts via the WASM API.
    pub fn discover_shortcuts_json(&mut self) -> Option<Vec<synapse_graph_types::CognitiveShortcut>> {
        let result = wasm_discover_shortcuts(self.ptr);
        if result.ptr.is_null() {
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(result.ptr, result.len as usize) };
        let parsed = serde_json::from_slice(bytes).ok();
        wasm_free_result(result);
        parsed
    }

    /// Start a session via the WASM API.
    pub fn start_session(&mut self, session_id: u64) {
        wasm_start_session(self.ptr, session_id);
    }

    /// End the current session via the WASM API.
    pub fn end_session(&mut self) {
        wasm_end_session(self.ptr);
    }
}

impl Drop for WasmSynapseGraph {
    fn drop(&mut self) {
        free_synapse_graph(self.ptr);
    }
}
