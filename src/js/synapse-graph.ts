/**
 * SynapseGraph TypeScript Bridge (ADR-006)
 *
 * Loads the WASM module and provides a typed TypeScript API wrapping
 * the WASM exports. Uses JSON serialization over WASM linear memory
 * for data exchange.
 *
 * @example
 * ```typescript
 * import { SynapseGraph } from './synapse-graph';
 *
 * const sg = await SynapseGraph.init({ embedding_dim: 64 });
 * const result = sg.ingest(spikeBatch);
 * const prediction = sg.predict(pattern);
 * sg.feedback(prediction.id, FeedbackType.Confirmed);
 * sg.free();
 * ```
 */

// ---------------------------------------------------------------------------
// Type definitions matching Rust types in synapse-graph-types
// ---------------------------------------------------------------------------

export interface SynapseGraphConfig {
  embedding_dim?: number;
  cluster_threshold?: number;
  temporal_window_us?: number;
  coherence_threshold?: number;
  fast_rank?: number;
  slow_rank?: number;
  fast_tier_capacity?: number;
  feedback_window_ms?: number;
  arena_size?: number;
}

export interface SpikeEvent {
  channel_id: number;
  amplitude: number;
  sub_timestamp: number;
}

export interface SpikeBatch {
  timestamp: number;
  session_id: number;
  spikes: SpikeEvent[];
  duration_us: number;
}

export interface IngestResult {
  ingested: number;
  denied: number;
  stored: number;
}

export interface ScoredIntent {
  intent_id: number;
  confidence: number;
  shortcut_id: number | null;
  match_distance: number;
}

export interface IntentPrediction {
  id: number;
  timestamp: number;
  ranked_intents: ScoredIntent[];
  latency_us: number;
}

export interface MaintenanceSummary {
  patterns_promoted: number;
  weak_points_healed: number;
  deltas_consolidated: number;
}

export interface GraphUpdateResult {
  nodes_created: number;
  nodes_updated: number;
  edges_created: number;
  edges_updated: number;
}

export interface CognitiveShortcut {
  id: number;
  nodes: number[];
  activation_frequency: number;
  predictive_power: number;
}

export enum FeedbackType {
  Confirmed = 0,
  Corrected = 1,
  Rejected = 2,
  Timeout = 3,
}

// ---------------------------------------------------------------------------
// WASM module interface (raw exports)
// ---------------------------------------------------------------------------

interface WasmExports {
  memory: WebAssembly.Memory;
  wasm_alloc(size: number): number;
  wasm_dealloc(ptr: number, size: number): void;
  init_synapse_graph(config_ptr: number, config_len: number): number;
  free_synapse_graph(sg: number): void;
  wasm_ingest(sg: number, batch_ptr: number, batch_len: number): number;
  wasm_predict(sg: number, pattern_ptr: number, pattern_len: number): number;
  wasm_feedback(sg: number, prediction_id: number, feedback_type: number): void;
  wasm_maintain(sg: number): number;
  wasm_update_cognitive_graph(sg: number): number;
  wasm_discover_shortcuts(sg: number): number;
  wasm_start_session(sg: number, session_id: number): void;
  wasm_end_session(sg: number): void;
  wasm_free_result(result_ptr: number): void;
}

// ---------------------------------------------------------------------------
// Helper: read/write JSON through WASM linear memory
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function writeToMemory(
  exports: WasmExports,
  data: Uint8Array
): { ptr: number; len: number } {
  const ptr = exports.wasm_alloc(data.length);
  if (ptr === 0) {
    throw new Error("WASM allocation failed");
  }
  const mem = new Uint8Array(exports.memory.buffer, ptr, data.length);
  mem.set(data);
  return { ptr, len: data.length };
}

function writeJsonToMemory(
  exports: WasmExports,
  obj: unknown
): { ptr: number; len: number } {
  const json = JSON.stringify(obj);
  const bytes = encoder.encode(json);
  return writeToMemory(exports, bytes);
}

/**
 * Read a WasmResult from linear memory.
 *
 * The WASM functions return a pair (ptr, len) packed into a single i64
 * or as two i32 values depending on the ABI. For wasm32, results are
 * returned via a pointer to a (ptr: u32, len: u32) struct in memory.
 */
function readResult(
  exports: WasmExports,
  resultPtr: number
): Uint8Array | null {
  // The WasmResult struct is { ptr: *mut u8, len: u32 } = 8 bytes
  const view = new DataView(exports.memory.buffer);
  const dataPtr = view.getUint32(resultPtr, true);
  const dataLen = view.getUint32(resultPtr + 4, true);

  if (dataPtr === 0 || dataLen === 0) {
    return null;
  }

  // Copy data out before it might be freed
  const slice = new Uint8Array(exports.memory.buffer, dataPtr, dataLen);
  const copy = new Uint8Array(dataLen);
  copy.set(slice);

  return copy;
}

function readJsonResult<T>(exports: WasmExports, resultPtr: number): T | null {
  const bytes = readResult(exports, resultPtr);
  if (!bytes) return null;
  const json = decoder.decode(bytes);
  return JSON.parse(json) as T;
}

// ---------------------------------------------------------------------------
// SynapseGraph class — typed TypeScript API
// ---------------------------------------------------------------------------

/**
 * High-level TypeScript wrapper for the SynapseGraph WASM module.
 *
 * Provides a typed API for all neural processing operations:
 * ingest, predict, feedback, maintenance, shortcuts, and sessions.
 */
export class SynapseGraph {
  private exports: WasmExports;
  private ptr: number;
  private freed = false;

  private constructor(exports: WasmExports, ptr: number) {
    this.exports = exports;
    this.ptr = ptr;
  }

  /**
   * Initialize a SynapseGraph instance from a WASM module.
   *
   * @param config - Configuration options (all optional, with defaults)
   * @param wasmPath - Path to the .wasm file (default: synapse_graph_wasm.wasm)
   * @returns Initialized SynapseGraph instance
   */
  static async init(
    config: SynapseGraphConfig = {},
    wasmPath = "synapse_graph_wasm.wasm"
  ): Promise<SynapseGraph> {
    const wasmBytes = await loadWasmBytes(wasmPath);
    const module = await WebAssembly.compile(wasmBytes);
    const instance = await WebAssembly.instantiate(module, {
      wasi_snapshot_preview1: createWasiStubs(),
    });

    const exports = instance.exports as unknown as WasmExports;
    const configJson = JSON.stringify(config);
    const { ptr: configPtr, len: configLen } = writeJsonToMemory(
      exports,
      config
    );

    const sgPtr = exports.init_synapse_graph(configPtr, configLen);
    exports.wasm_dealloc(configPtr, configLen);

    if (sgPtr === 0) {
      throw new Error("Failed to initialize SynapseGraph: invalid config");
    }

    return new SynapseGraph(exports, sgPtr);
  }

  /**
   * Initialize from an already-instantiated WASM module.
   */
  static fromInstance(
    instance: WebAssembly.Instance,
    config: SynapseGraphConfig = {}
  ): SynapseGraph {
    const exports = instance.exports as unknown as WasmExports;
    const { ptr: configPtr, len: configLen } = writeJsonToMemory(
      exports,
      config
    );

    const sgPtr = exports.init_synapse_graph(configPtr, configLen);
    exports.wasm_dealloc(configPtr, configLen);

    if (sgPtr === 0) {
      throw new Error("Failed to initialize SynapseGraph");
    }

    return new SynapseGraph(exports, sgPtr);
  }

  /**
   * Ingest a spike batch through the neural processing pipeline.
   */
  ingest(batch: SpikeBatch): IngestResult {
    this.ensureNotFreed();
    const { ptr, len } = writeJsonToMemory(this.exports, batch);
    const resultPtr = this.exports.wasm_ingest(this.ptr, ptr, len);
    this.exports.wasm_dealloc(ptr, len);

    const result = readJsonResult<IngestResult>(this.exports, resultPtr);
    if (!result) {
      throw new Error("Ingest failed: could not parse result");
    }
    return result;
  }

  /**
   * Run intent prediction on a partial neural pattern.
   *
   * Note: In production, patterns are generated internally by the
   * ingestion pipeline. This method accepts a raw pattern for testing.
   */
  predict(patternJson: unknown): IntentPrediction {
    this.ensureNotFreed();
    const { ptr, len } = writeJsonToMemory(this.exports, patternJson);
    const resultPtr = this.exports.wasm_predict(this.ptr, ptr, len);
    this.exports.wasm_dealloc(ptr, len);

    const result = readJsonResult<IntentPrediction>(this.exports, resultPtr);
    if (!result) {
      throw new Error("Predict failed: could not parse result");
    }
    return result;
  }

  /**
   * Provide feedback on a prediction to trigger adaptation.
   */
  feedback(predictionId: number, type_: FeedbackType): void {
    this.ensureNotFreed();
    this.exports.wasm_feedback(this.ptr, predictionId, type_);
  }

  /**
   * Run the maintenance cycle (temporal promotions, graph healing,
   * LoRA consolidation).
   */
  maintain(): MaintenanceSummary {
    this.ensureNotFreed();
    const resultPtr = this.exports.wasm_maintain(this.ptr);
    const result = readJsonResult<MaintenanceSummary>(
      this.exports,
      resultPtr
    );
    if (!result) {
      throw new Error("Maintain failed: could not parse result");
    }
    return result;
  }

  /**
   * Update the cognitive graph from recent memory state.
   */
  updateCognitiveGraph(): GraphUpdateResult {
    this.ensureNotFreed();
    const resultPtr = this.exports.wasm_update_cognitive_graph(this.ptr);
    const result = readJsonResult<GraphUpdateResult>(
      this.exports,
      resultPtr
    );
    if (!result) {
      throw new Error("Update cognitive graph failed");
    }
    return result;
  }

  /**
   * Discover cognitive shortcuts via GNN engine.
   */
  discoverShortcuts(): CognitiveShortcut[] {
    this.ensureNotFreed();
    const resultPtr = this.exports.wasm_discover_shortcuts(this.ptr);
    const result = readJsonResult<CognitiveShortcut[]>(
      this.exports,
      resultPtr
    );
    return result ?? [];
  }

  /**
   * Start a new session.
   */
  startSession(sessionId: number): void {
    this.ensureNotFreed();
    this.exports.wasm_start_session(this.ptr, sessionId);
  }

  /**
   * End the current session.
   */
  endSession(): void {
    this.ensureNotFreed();
    this.exports.wasm_end_session(this.ptr);
  }

  /**
   * Get current WASM linear memory usage in bytes.
   */
  memoryUsage(): number {
    this.ensureNotFreed();
    return this.exports.memory.buffer.byteLength;
  }

  /**
   * Free the SynapseGraph instance.
   *
   * Must be called when done to prevent memory leaks.
   * The instance cannot be used after calling free().
   */
  free(): void {
    if (!this.freed) {
      this.exports.free_synapse_graph(this.ptr);
      this.freed = true;
    }
  }

  private ensureNotFreed(): void {
    if (this.freed) {
      throw new Error("SynapseGraph instance has been freed");
    }
  }
}

// ---------------------------------------------------------------------------
// WASM loading helpers
// ---------------------------------------------------------------------------

/**
 * Load WASM bytes from a file path or URL.
 *
 * Works in both Node.js (via fs) and browser (via fetch) environments.
 */
async function loadWasmBytes(path: string): Promise<ArrayBuffer> {
  // Node.js environment
  if (typeof globalThis.process !== "undefined" && globalThis.process.versions?.node) {
    const fs = await import("fs/promises");
    const buffer = await fs.readFile(path);
    return buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength
    );
  }

  // Browser environment
  const response = await fetch(path);
  return response.arrayBuffer();
}

/**
 * Create minimal WASI stubs for running the WASM module.
 *
 * The SynapseGraph WASM module targets wasi_snapshot_preview1 but
 * only needs a minimal set of WASI functions (fd_write for debugging,
 * proc_exit for clean shutdown).
 */
function createWasiStubs(): Record<string, (...args: unknown[]) => unknown> {
  return {
    fd_write(
      _fd: number,
      _iovs: number,
      _iovs_len: number,
      _nwritten: number
    ): number {
      return 0; // success, but discard output
    },
    fd_seek(): number {
      return 0;
    },
    fd_close(): number {
      return 0;
    },
    fd_read(): number {
      return 0;
    },
    fd_fdstat_get(): number {
      return 0;
    },
    fd_prestat_get(): number {
      return 8; // EBADF - no preopens
    },
    fd_prestat_dir_name(): number {
      return 8;
    },
    environ_sizes_get(
      _count_ptr: number,
      _buf_size_ptr: number
    ): number {
      return 0;
    },
    environ_get(): number {
      return 0;
    },
    args_sizes_get(): number {
      return 0;
    },
    args_get(): number {
      return 0;
    },
    clock_time_get(
      _clock_id: number,
      _precision: bigint,
      _time_ptr: number
    ): number {
      return 0;
    },
    proc_exit(_code: number): void {
      // no-op in embedded context
    },
    random_get(_buf: number, _len: number): number {
      return 0;
    },
  };
}
