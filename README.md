# NvTrace Client

NvTrace Client provides header-based interfaces for collecting data from a running target and consuming trace data in-process or offline.

## Contents

- `include/NvTrace/Target` contains target-side routers, observers, consumers, and service interfaces.
- `include/NvTrace/Offline` contains offline data-model, database-reader, and exporter interfaces.
- [`include/NvTrace/DataFormats`](include/NvTrace/DataFormats/README.md) contains versioned provider-buffer data formats.
- [`schemas/NvTrace`](schemas/NvTrace/README.md) contains the FlatBuffers schema sources for generated client headers.

## Architecture

The client connects to the NVIDIA Nsight Perf service and reads the shared-memory buffers published by profiling providers. Each data source has a corresponding observer that interprets its provider-buffer records.

Consumers coordinate observers and persist the resulting event stream as a Trace Stream Database:

- `nv::trace::StreamingConsumer` writes data incrementally from start to stop.
- `nv::trace::TsfSnapshotConsumer` retains a rolling history and writes a lookback snapshot when triggered.
- `nv::trace::UptimeValidationConsumer` checks service and provider liveness.

Offline readers load Trace Stream Database blocks into `nv::trace::Codec::DataModel`. Exporters transform the data model into outputs such as Perfetto traces.

## Choosing a consumer

Use `StreamingConsumer` for:

- captures with a known start and stop;
- extended sessions where data should be written incrementally;
- continuous monitoring that persists all observed data.

Use `SnapshotConsumer` for:

- flight-recorder and lookback profiling;
- anomaly-triggered capture;
- retaining bounded history without persisting every interval.

The command-line counterparts are documented under [`nvtrace-tools`](https://github.com/NVIDIA/nvtrace-tools/blob/main/README.md).

## Time windows

- The **observation window** is the interval during which a consumer observes provider data.
- The **analysis window** is the interval selected when the offline data-model builder loads a Trace Stream Database.

An analysis window can be narrower than its source observation window. Events requiring state from before the analysis window may be incomplete or absent; see [Known limitations](#known-limitations).

## Data flow

### Streaming

1. The service collects provider data in shared-memory buffers.
2. A client application initializes `StreamingConsumer` with a database writer.
3. Observers read provider data while the consumer is active.
4. A background writer stores numbered `.nvtsf` data blocks in a Trace Stream Database.
5. An offline reader builds a data model that an exporter can convert to another format.

### Snapshot

1. The service and observers run continuously.
2. `SnapshotConsumer` retains a bounded rolling history.
3. A trigger selects the requested lookback and optional future interval.
4. The consumer writes that interval as a new Trace Stream Database.
5. An offline exporter converts the snapshot database.

## Compatibility

Service-connected consumers and the NVIDIA Nsight Perf service are compatible across SDK releases in both directions: newer consumers can connect to older services, and older consumers can connect to newer services. A cross-version combination exposes only features understood by the participating versions.

### FlatBuffers dependency boundary

The generated C++ headers compile only with the FlatBuffers runtime version that generated them. Source that includes an NvTrace header which reaches a `*_generated.h` file will fail at compile time when used with a different FlatBuffers runtime version:

```text
static assertion failed: Non-compatible flatbuffers version included
```

This assertion is a source-level dependency check; it does not establish wire-format compatibility with another FlatBuffers version. Code that does not compile or expose the generated headers is unaffected by the assertion.

To use another FlatBuffers runtime, either [regenerate all NvTrace generated headers](schemas/NvTrace/README.md#regenerating-headers) with the matching `flatc`, or isolate NvTrace in a separately compiled library that uses the shipped FlatBuffers version.

For isolation to work, the library's public interface must not:

- include an NvTrace `*_generated.h` file;
- expose a `flatbuffers::` type in a signature;
- expose inline or template code that depends on either.

A C-style or PIMPL boundary prevents the generated FlatBuffers dependency from entering application translation units. Allowing different FlatBuffers versions to reach the same translation unit or inline symbol can create One Definition Rule violations instead of a clear compile-time error.

## Modification guidance

- Do not directly edit `*_generated.h` files. Change the corresponding `.fbs` schema and regenerate the headers.
- Do not reorder, remove, or repurpose fields or enum values in versioned data-format headers. Some base structures are explicitly frozen and must not change.
- Treat a new format version and its reader/writer handling as one change. A version declaration alone does not provide compatibility.

## Known limitations

- Kernels that begin before the observation window may not be displayed.
- Loading a partial analysis window can omit an event that begins before and ends after the window.
- A long-running CUDA API call whose entry is before the analysis window may be absent even when its exit is inside the window.
- An NVTX push-pop range that starts before the analysis window may appear with a placeholder name when its pop is inside the window.
- A GPU context-switch interval enclosing the entire analysis window may be attributed to an inferred or unknown process.
- NVTX start-end ranges are not supported.

## Contributions

Contributions require signed-off commits and the Developer Certificate of Origin. See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

## Security

Do not report potential security vulnerabilities through public GitHub issues. Follow the private reporting process in [SECURITY.md](SECURITY.md).

## License

NVIDIA-authored source is licensed under Apache-2.0; see [LICENSE](LICENSE). Dependencies and separately supplied proprietary runtimes retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
