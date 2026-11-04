/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_CUDA_ApiTrace_ProviderTypeUuid = NVTRACE_UUID_VALUE(0x4a8e1a64, 0x4565, 0x4b4b, 0xa1, 0xb0, 0xdb, 0x69, 0x53, 0xb4, 0xf1, 0x62);

enum NvTrace_CUDA_ApiTrace_SectionTag
{
    NvTrace_CUDA_ApiTrace_SectionTag_INVALID = 0,

    // Data blobs
    NvTrace_CUDA_ApiTrace_SectionTag_RecordVersionInfo = 1,
    NvTrace_CUDA_ApiTrace_SectionTag_Configuration = 2,
    NvTrace_CUDA_ApiTrace_SectionTag_ProviderStatistics = 3,
    NvTrace_CUDA_ApiTrace_SectionTag_Device = 4, // one blob per device, all under this tag: devices predate any consumer and are never destroyed

    // State tables
    NvTrace_CUDA_ApiTrace_SectionTag_Contexts = 5,
    NvTrace_CUDA_ApiTrace_SectionTag_Streams = 6,
    NvTrace_CUDA_ApiTrace_SectionTag_Modules = 7, // chain-extended: a module's functions are its chain records
    NvTrace_CUDA_ApiTrace_SectionTag_GraphExecs = 8,
    NvTrace_CUDA_ApiTrace_SectionTag_GraphNodes = 9,
    NvTrace_CUDA_ApiTrace_SectionTag_Channels = 10,

    // Chunk pools
    NvTrace_CUDA_ApiTrace_SectionTag_StateChunkPool = 11,     // backs the Modules entries' function chains
    NvTrace_CUDA_ApiTrace_SectionTag_GpuReportChunkPool = 12, // backs the GPU timestamp report runs

    // GPU report buffers
    NvTrace_CUDA_ApiTrace_SectionTag_GpuReportBuffer = 13,

    NvTrace_CUDA_ApiTrace_SectionTag_FORCE_INT = 0x7fffffff
};

/// The provider's one record-type space: every record in any section carries one of these values.
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_CUDA_ApiTrace_RecordType
{
    NvTrace_CUDA_ApiTrace_RecordType_INVALID = 0,

    NvTrace_CUDA_ApiTrace_RecordType_RecordVersionInfo,
    NvTrace_CUDA_ApiTrace_RecordType_Configuration,
    NvTrace_CUDA_ApiTrace_RecordType_ProviderStatistics,

    // The object records: each created record directly followed by its destroyed record, for the
    // objects that have one, and by the continuation record that extends it -- which names its
    // owner in its own payload. Devices are never destroyed; functions and graph nodes go away with
    // their module and graph exec, and channels with their context.
    NvTrace_CUDA_ApiTrace_RecordType_DeviceCreated,
    NvTrace_CUDA_ApiTrace_RecordType_ContextCreated,
    NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed,
    NvTrace_CUDA_ApiTrace_RecordType_StreamCreated,
    NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed,
    NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded,
    NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded,
    NvTrace_CUDA_ApiTrace_RecordType_FunctionCreated,
    NvTrace_CUDA_ApiTrace_RecordType_FunctionNameSegment,
    NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated,
    NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed,
    NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated,
    NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated,

    NvTrace_CUDA_ApiTrace_RecordType_GpuTimestampReport,

    // The API thread events. Nothing emits MemcpyEnd, MemsetEnd, GraphMemcpyEnd, GraphMemsetEnd or MemOpEnd; those values stay reserved for streams that do split the operation and join its end to its begin by correlation id.
    NvTrace_CUDA_ApiTrace_RecordType_CudaDriverApiEnter,
    NvTrace_CUDA_ApiTrace_RecordType_CudaDriverApiExit,
    NvTrace_CUDA_ApiTrace_RecordType_CudaRuntimeApiEnter,
    NvTrace_CUDA_ApiTrace_RecordType_CudaRuntimeApiExit,
    NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch,
    NvTrace_CUDA_ApiTrace_RecordType_GraphLaunch,
    NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin,
    NvTrace_CUDA_ApiTrace_RecordType_MemcpyEnd,
    NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin,
    NvTrace_CUDA_ApiTrace_RecordType_MemsetEnd,
    NvTrace_CUDA_ApiTrace_RecordType_HostCallbackLaunch,
    NvTrace_CUDA_ApiTrace_RecordType_HostCallbackExecBegin,
    NvTrace_CUDA_ApiTrace_RecordType_HostCallbackExecEnd,
    NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyBegin,
    NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyEnd,
    NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetBegin,
    NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetEnd,
    NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackLaunch,
    NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackExecBegin,
    NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackExecEnd,
    NvTrace_CUDA_ApiTrace_RecordType_MemOpBegin,
    NvTrace_CUDA_ApiTrace_RecordType_MemOpEnd,

    NvTrace_CUDA_ApiTrace_RecordType_COUNT,
    NvTrace_CUDA_ApiTrace_RecordType_FORCE_INT = 0x7fffffff
};

#define NVTRACE_CUDA_THREAD_EVENT_RECORD_IDENTIFIER(recordType) NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER(NvTrace_ThreadEvent_StaticSourceProviderIndex_CUDA, (recordType))

// Self-describing record-format descriptor: recordVersionInfoSize and recordVersionInfoMinorVersion
// describe this struct itself, so a consumer can version it from the payload alone.
struct NvTrace_CUDA_ApiTrace_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;
    uint32_t providerStatisticsSize;
    uint32_t providerStatisticsMinorVersion;

    uint32_t deviceCreatedSize;
    uint32_t deviceCreatedMinorVersion;
    uint32_t contextCreatedSize;
    uint32_t contextCreatedMinorVersion;
    uint32_t contextDestroyedSize;
    uint32_t contextDestroyedMinorVersion;
    uint32_t streamCreatedSize;
    uint32_t streamCreatedMinorVersion;
    uint32_t streamDestroyedSize;
    uint32_t streamDestroyedMinorVersion;
    uint32_t moduleLoadedSize;
    uint32_t moduleLoadedMinorVersion;
    uint32_t moduleUnloadedSize;
    uint32_t moduleUnloadedMinorVersion;
    uint32_t functionCreatedSize;
    uint32_t functionCreatedMinorVersion;
    uint32_t functionNameSegmentSize;
    uint32_t functionNameSegmentMinorVersion;
    uint32_t graphExecCreatedSize;
    uint32_t graphExecCreatedMinorVersion;
    uint32_t graphExecDestroyedSize;
    uint32_t graphExecDestroyedMinorVersion;
    uint32_t graphNodeCreatedSize;
    uint32_t graphNodeCreatedMinorVersion;
    uint32_t channelCreatedSize;
    uint32_t channelCreatedMinorVersion;

    uint32_t gpuTimestampReportSize;
    uint32_t gpuTimestampReportMinorVersion;

    uint32_t cudaDriverApiEnterSize;
    uint32_t cudaDriverApiEnterMinorVersion;
    uint32_t cudaDriverApiExitSize;
    uint32_t cudaDriverApiExitMinorVersion;
    uint32_t cudaRuntimeApiEnterSize;
    uint32_t cudaRuntimeApiEnterMinorVersion;
    uint32_t cudaRuntimeApiExitSize;
    uint32_t cudaRuntimeApiExitMinorVersion;
    uint32_t immediateKernelLaunchSize;
    uint32_t immediateKernelLaunchMinorVersion;
    uint32_t graphLaunchSize;
    uint32_t graphLaunchMinorVersion;
    uint32_t memcpyBeginSize;
    uint32_t memcpyBeginMinorVersion;
    uint32_t memsetBeginSize;
    uint32_t memsetBeginMinorVersion;
    uint32_t hostCallbackLaunchSize;
    uint32_t hostCallbackLaunchMinorVersion;
    uint32_t hostCallbackExecBeginSize;
    uint32_t hostCallbackExecBeginMinorVersion;
    uint32_t hostCallbackExecEndSize;
    uint32_t hostCallbackExecEndMinorVersion;
    uint32_t graphMemcpyBeginSize;
    uint32_t graphMemcpyBeginMinorVersion;
    uint32_t graphMemsetBeginSize;
    uint32_t graphMemsetBeginMinorVersion;
    uint32_t graphHostCallbackLaunchSize;
    uint32_t graphHostCallbackLaunchMinorVersion;
    uint32_t graphHostCallbackExecBeginSize;
    uint32_t graphHostCallbackExecBeginMinorVersion;
    uint32_t graphHostCallbackExecEndSize;
    uint32_t graphHostCallbackExecEndMinorVersion;
    uint32_t memOpBeginSize;
    uint32_t memOpBeginMinorVersion;

    uint32_t graphNodeKernelParamsSize;
    uint32_t graphNodeKernelParamsMinorVersion;
    uint32_t graphNodeMemcpyParamsSize;
    uint32_t graphNodeMemcpyParamsMinorVersion;
    uint32_t graphNodeMemsetParamsSize;
    uint32_t graphNodeMemsetParamsMinorVersion;
    uint32_t graphNodeHostParamsSize;
    uint32_t graphNodeHostParamsMinorVersion;
    uint32_t graphNodeChildGraphParamsSize;
    uint32_t graphNodeChildGraphParamsMinorVersion;
    uint32_t graphNodeWaitEventParamsSize;
    uint32_t graphNodeWaitEventParamsMinorVersion;
    uint32_t graphNodeEventRecordParamsSize;
    uint32_t graphNodeEventRecordParamsMinorVersion;
    uint32_t graphNodeExtSemasSignalParamsSize;
    uint32_t graphNodeExtSemasSignalParamsMinorVersion;
    uint32_t graphNodeExtSemasWaitParamsSize;
    uint32_t graphNodeExtSemasWaitParamsMinorVersion;
    uint32_t graphNodeMemAllocParamsSize;
    uint32_t graphNodeMemAllocParamsMinorVersion;
    uint32_t graphNodeMemFreeParamsSize;
    uint32_t graphNodeMemFreeParamsMinorVersion;
    uint32_t graphNodeBatchMemOpParamsSize;
    uint32_t graphNodeBatchMemOpParamsMinorVersion;
    uint32_t graphNodeConditionalParamsSize;
    uint32_t graphNodeConditionalParamsMinorVersion;
    uint32_t graphNodeMemcpyBatchParamsSize;
    uint32_t graphNodeMemcpyBatchParamsMinorVersion;
    uint32_t graphNodeMemcpyBatch3DParamsSize;
    uint32_t graphNodeMemcpyBatch3DParamsMinorVersion;

    uint32_t memOpWriteValueParamsSize;
    uint32_t memOpWriteValueParamsMinorVersion;
    uint32_t memOpWaitValueParamsSize;
    uint32_t memOpWaitValueParamsMinorVersion;
    uint32_t memOpBarrierParamsSize;
    uint32_t memOpBarrierParamsMinorVersion;
    uint32_t memOpAtomicReductionParamsSize;
    uint32_t memOpAtomicReductionParamsMinorVersion;
};
#define NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_RecordVersionInfo, memOpAtomicReductionParamsMinorVersion)

// Independently meaningful 64-bit fields, sampled best-effort: there is no consistency protocol
struct NvTrace_CUDA_ApiTrace_ProviderStatistics
{
    uint64_t contextCount;
    uint64_t streamCount;
    uint64_t moduleCount;
    uint64_t functionCount;
    uint64_t graphExecCount;
    uint64_t graphNodeCount;
    uint64_t channelCount;

    uint64_t peakContextCount;
    uint64_t peakStreamCount;
    uint64_t peakModuleCount;
    uint64_t peakFunctionCount;
    uint64_t peakGraphExecCount;
    uint64_t peakGraphNodeCount;
    uint64_t peakChannelCount;

    // Functions whose record found no room in their module's chain: the module is published without them.
    uint64_t droppedFunctionCount;
    // Function names cut by state chunk pool exhaustion: the function's record was published with its
    // inline bytes but the rest of its name was not.
    uint64_t droppedFunctionNameCount;

    // Streams destroyed with no way to pad their in-flight GPU timestamp chunk (no null stream and
    // no host mapping): each wedges one chunk and one stream-table slot until process teardown.
    uint64_t wedgedStreamCount;
};
#define NvTrace_CUDA_ApiTrace_ProviderStatistics_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ProviderStatistics, wedgedStreamCount)

/// Represents the static properties of a GPU device, published as its own data blob. Devices predate
/// any consumer and are never destroyed.
struct NvTrace_CUDA_ApiTrace_DeviceCreated
{
    uint64_t creationTimestampTsc;

    /// GPU UUID, usable to correlate with other device lists.
    NVTRACE_UUID gpuUuid;
    /// cudaDeviceOrdinal is also the device blob's position among the published device blobs.
    uint32_t cudaDeviceOrdinal;
    uint32_t cudeviceHandle;
};
#define NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_DeviceCreated, cudeviceHandle)

struct NvTrace_CUDA_ApiTrace_ContextCreated
{
    uint64_t creationTimestampTsc;
    uint64_t destructionTimestampTsc; // Every destructionTimestampTsc in this header: 0 while alive, stamped in place, moving no revision.

    uint64_t contextId;
    uint64_t cucontextHandle;
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ContextCreated, cudaDeviceOrdinal)

struct NvTrace_CUDA_ApiTrace_StreamCreated
{
    uint64_t creationTimestampTsc;
    uint64_t destructionTimestampTsc;

    uint64_t streamId;
    uint64_t custreamHandle;
    /// The context to which this stream belongs.
    uint64_t contextId;
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_StreamCreated, cudaDeviceOrdinal)

struct NvTrace_CUDA_ApiTrace_ModuleLoaded
{
    uint64_t creationTimestampTsc;
    uint64_t destructionTimestampTsc;

    uint64_t moduleId;
    uint64_t cumoduleHandle;
    /// The context to which this module belongs.
    uint64_t contextId;
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ModuleLoaded, cudaDeviceOrdinal)

/// A chain record of the owning module's entry: a module's functions are its chain, in the order
/// FunctionCreated_1, FunctionNameSegment_1..., FunctionCreated_2, FunctionNameSegment_2..., so a module is read with
/// its functions wherever its entry is read. A function is created and destroyed only with its module,
/// so it carries no timestamps of its own; ModuleLoaded's are the function's.
///
/// The mangled name, not NUL-terminated, leads with the inlineNameSegmentLength bytes that follow this
/// struct at the published functionCreatedSize, as far as the chunk record allows; the rest follows in
/// the chain as NvTrace_CUDA_ApiTrace_FunctionNameSegment records.
struct NvTrace_CUDA_ApiTrace_FunctionCreated
{
    uint64_t cufunctionId;
    uint64_t cufunctionHandle;
    /// The unique ID of the CUmodule that this function belongs to.
    uint64_t moduleId;

    uint32_t totalNameLength;
    uint32_t inlineNameSegmentLength;
};
#define NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_FunctionCreated, inlineNameSegmentLength)

/// One segment of a function's mangled name, for the part its FunctionCreated record did not carry
/// inline: nameSegmentLength bytes following this struct at the published functionNameSegmentSize,
/// placed at nameOffset into the name. A name longer than one chain record travels as several such
/// records; a reader reassembles the name from the records in any order, keyed on cufunctionId.
struct NvTrace_CUDA_ApiTrace_FunctionNameSegment
{
    uint64_t cufunctionId;
    uint32_t nameOffset;
    uint32_t nameSegmentLength;
};
#define NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_FunctionNameSegment, nameSegmentLength)

/// A consumer enumerates a graph exec's nodes by cugraphExecId over the attach snapshot.
struct NvTrace_CUDA_ApiTrace_GraphExecCreated
{
    uint64_t creationTimestampTsc;
    uint64_t destructionTimestampTsc;

    uint64_t cugraphExecId;
    uint64_t cugraphExecHandle;
    /// The CUgraph from which this CUgraphExec was created.
    uint64_t cugraphId;
};
#define NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphExecCreated, cugraphId)


enum NvTrace_CUDA_ApiTrace_AddressKind
{
    NvTrace_CUDA_ApiTrace_AddressKind_Invalid = 0,
    NvTrace_CUDA_ApiTrace_AddressKind_Host,
    NvTrace_CUDA_ApiTrace_AddressKind_Device,
    NvTrace_CUDA_ApiTrace_AddressKind_Array,
    NvTrace_CUDA_ApiTrace_AddressKind_COUNT,
    NvTrace_CUDA_ApiTrace_AddressKind_FORCE_INT = 0x7fffffff
};

enum NvTrace_CUDA_ApiTrace_MemcpyFlags
{
    NvTrace_CUDA_ApiTrace_MemcpyFlags_IsAsync = 0x1,
    NvTrace_CUDA_ApiTrace_MemcpyFlags_IsP2P = 0x2,
};

enum NvTrace_CUDA_ApiTrace_CommandType
{
    NvTrace_CUDA_ApiTrace_CommandType_Invalid = 0,
    NvTrace_CUDA_ApiTrace_CommandType_Kernel = 1,
    NvTrace_CUDA_ApiTrace_CommandType_InlineToMemory = 2,
    NvTrace_CUDA_ApiTrace_CommandType_ComputeDma = 3,
    NvTrace_CUDA_ApiTrace_CommandType_AsyncDma = 4,
    NvTrace_CUDA_ApiTrace_CommandType_COUNT,
    NvTrace_CUDA_ApiTrace_CommandType_FORCE_INT = 0x7fffffff
};

enum NvTrace_CUDA_ApiTrace_HostTaskSyncMode
{
    NvTrace_CUDA_ApiTrace_HostTaskSyncMode_Blocking = 0,
    NvTrace_CUDA_ApiTrace_HostTaskSyncMode_SpinWait = 1,
    NvTrace_CUDA_ApiTrace_HostTaskSyncMode_FORCE_INT = 0x7fffffff
};

enum NvTrace_CUDA_ApiTrace_GraphChildGraphNodeOwnership
{
    NvTrace_CUDA_ApiTrace_GraphChildGraphNodeOwnership_Clone = 0,
    NvTrace_CUDA_ApiTrace_GraphChildGraphNodeOwnership_Move = 1,
    NvTrace_CUDA_ApiTrace_GraphChildGraphNodeOwnership_Invalid = -1,
    NvTrace_CUDA_ApiTrace_GraphChildGraphNodeOwnership_FORCE_INT = 0x7fffffff
};

enum NvTrace_CUDA_ApiTrace_MemLocationType
{
    NvTrace_CUDA_ApiTrace_MemLocationType_Invalid = 0,
    NvTrace_CUDA_ApiTrace_MemLocationType_Device = 1,
    NvTrace_CUDA_ApiTrace_MemLocationType_Host = 2,
    NvTrace_CUDA_ApiTrace_MemLocationType_HostNuma = 3,
    NvTrace_CUDA_ApiTrace_MemLocationType_HostNumaCurrent = 4,
    NvTrace_CUDA_ApiTrace_MemLocationType_Invisible = 5,
    NvTrace_CUDA_ApiTrace_MemLocationType_DeviceLocalityDomain = 6,
    NvTrace_CUDA_ApiTrace_MemLocationType_FORCE_INT = 0x7fffffff
};

enum NvTrace_CUDA_ApiTrace_GraphConditionalNodeType
{
    NvTrace_CUDA_ApiTrace_GraphConditionalNodeType_If = 0,
    NvTrace_CUDA_ApiTrace_GraphConditionalNodeType_While = 1,
    NvTrace_CUDA_ApiTrace_GraphConditionalNodeType_Switch = 2,
    NvTrace_CUDA_ApiTrace_GraphConditionalNodeType_FORCE_INT = 0x7fffffff
};

enum NvTrace_CUDA_ApiTrace_ExtSemSignalFlags
{
    NvTrace_CUDA_ApiTrace_ExtSemSignalFlags_SkipNvSciBufMemsync = 0x01,
};

enum NvTrace_CUDA_ApiTrace_ExtSemWaitFlags
{
    NvTrace_CUDA_ApiTrace_ExtSemWaitFlags_SkipNvSciBufMemsync = 0x02,
};

/// The type of a CUgraphNode, matching the sequence of CUgraphNodeType from Kernel onwards. Other
/// is the catch-all for node types NvTrace supplies no additional semantic info for; Kernel is the
/// launch that generates HW trace events. Not all types may be seen in executable nodes.
enum NvTrace_CUDA_ApiTrace_GraphNodeType
{
    NvTrace_CUDA_ApiTrace_GraphNodeType_Invalid = 0,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Other,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Memset,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Host,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Graph,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Empty,
    NvTrace_CUDA_ApiTrace_GraphNodeType_WaitEvent,
    NvTrace_CUDA_ApiTrace_GraphNodeType_EventRecord,
    NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasSignal,
    NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasWait,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemAlloc,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemFree,
    NvTrace_CUDA_ApiTrace_GraphNodeType_BatchMemOp,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Conditional,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch3D,
    NvTrace_CUDA_ApiTrace_GraphNodeType_SubgridSchedule,
    NvTrace_CUDA_ApiTrace_GraphNodeType_FORCE_INT = 0x7fffffff
};

/// The common segment of every graph-node created-record. The record is a composite: this segment,
/// then the params segment graphNodeType selects, beginning at the published graphNodeCreatedSize
/// per NvTrace_CUDA_ApiTrace_RecordVersionInfo -- never at an offsetof or sizeof, which describe only
/// the build they came from. A consumer that does not know a graphNodeType still reads this
/// segment; only the params segment is opaque to it.
struct NvTrace_CUDA_ApiTrace_GraphNodeCreated
{
    uint64_t creationTimestampTsc;
    uint64_t destructionTimestampTsc;

    uint64_t cugraphNodeId;
    uint64_t cugraphNodeHandle;
    /// The CUgraphExec into which this CUgraphNode was created.
    uint64_t cugraphExecId;
    /// Entry ref of the NVTX stack group capturing the ranges open at the node's creation, or
    /// NVTRACE_INVALID_REVISIONED_REF if none.
    NvTrace_RevisionedRef nvtxTraceStackGroupRef;

    uint32_t cudaDeviceOrdinal;

    enum NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType;
};
#define NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNodeCreated, graphNodeType)

/// The type-specific params segment of a graph-node created-record (see
/// NvTrace_CUDA_ApiTrace_GraphNodeCreated for the record layout).
/// The Other, Empty and SubgridSchedule node types carry no params, so they have no struct here.
/// For version safety, structs must be 1:1 with the NvTrace_CUDA_ApiTrace_GraphNodeType values and
/// the sizeof each struct must be reflected in NvTrace_CUDA_ApiTrace_RecordVersionInfo.
struct NvTrace_CUDA_ApiTrace_GraphNode_KernelParams
{
    uint64_t cufunctionId;
    uint32_t gridDimX;
    uint32_t gridDimY;
    uint32_t gridDimZ;
    uint32_t blockDimX;
    uint32_t blockDimY;
    uint32_t blockDimZ;
    uint32_t clusterDimX;
    uint32_t clusterDimY;
    uint32_t clusterDimZ;
    uint32_t totalSharedMemPerBlock;
    /// When this graph node is a converted memcpy/memset kernel follower, the graph-node correlation id of the
    /// logical head node (0 for ordinary kernel nodes).
    uint64_t enclosingGraphNodeCorrelationId;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_KernelParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_KernelParams, enclosingGraphNodeCorrelationId)

struct NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams
{
    uint64_t byteCount;
    enum NvTrace_CUDA_ApiTrace_AddressKind srcAddressKind;
    enum NvTrace_CUDA_ApiTrace_AddressKind dstAddressKind;
    uint64_t srcDevPtr;
    uint64_t dstDevPtr;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint32_t memcpyFlags; // mask of NvTrace_CUDA_ApiTrace_MemcpyFlags
};
#define NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams, memcpyFlags)

struct NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams
{
    uint64_t devPtr;
    uint64_t byteCount;
    uint32_t value;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams, value)

struct NvTrace_CUDA_ApiTrace_GraphNode_HostParams
{
    uint64_t callback;
    uint64_t userData;
    enum NvTrace_CUDA_ApiTrace_HostTaskSyncMode syncMode;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_HostParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_HostParams, syncMode)

struct NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams
{
    uint64_t cugraphId;
    enum NvTrace_CUDA_ApiTrace_GraphChildGraphNodeOwnership ownership;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams, ownership)

struct NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams
{
    uint64_t eventHandle;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams, eventHandle)

struct NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams
{
    uint64_t eventHandle;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams, eventHandle)

struct NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams
{
    uint32_t numExtSems;
    uint32_t rsvd0004;
    uint64_t extSemHandle;
    uint64_t signalValue;
    uint32_t flags; // mask of NvTrace_CUDA_ApiTrace_ExtSemSignalFlags
};
#define NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams, flags)

struct NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams
{
    uint32_t numExtSems;
    uint32_t rsvd0004;
    uint64_t extSemHandle;
    uint64_t waitValue;
    uint32_t flags; // mask of NvTrace_CUDA_ApiTrace_ExtSemWaitFlags
};
#define NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams, flags)

struct NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams
{
    uint64_t bytesize;
    uint64_t dptr;
    enum NvTrace_CUDA_ApiTrace_MemLocationType poolLocationType;
    int32_t poolLocationId;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams, poolLocationId)

struct NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams
{
    uint64_t dptr;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams, dptr)

struct NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams
{
    uint32_t count;
    uint32_t flags;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams, flags)

struct NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams
{
    uint64_t conditionalHandle;
    enum NvTrace_CUDA_ApiTrace_GraphConditionalNodeType conditionalNodeType;
    uint32_t size;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams, size)

struct NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams
{
    uint64_t numOps;
    uint64_t totalByteCount;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams, totalByteCount)

struct NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams
{
    uint64_t numOps;
    uint64_t flags;
};
#define NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams, flags)

/// A compute (GR) channel owned by a CUcontext. A context owns several compute channels (8 by default),
/// each with its own RM CID, and a FECS GrCtx context-switch event reports whichever of them was
/// switched. These entries map any of a context's CIDs back to the owning context -- and thereby to the
/// process -- independent of PID namespaces. Entry lifetimes are tied to the owning context.
struct NvTrace_CUDA_ApiTrace_ChannelCreated
{
    uint64_t creationTimestampTsc;
    uint64_t destructionTimestampTsc;

    /// NvTrace_CUDA_ApiTrace_ContextCreated::contextId of the owning context.
    uint64_t cucontextId;
    /// RM channel CID (system-globally unique, monotonically allocated). Matches the FECS GrCtx context_id.
    uint32_t rmCid;
};
#define NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ChannelCreated, rmCid)

/// A GPU report record (see NvTrace_GpuReportBufferSectionHeader): the GPU's semaphore release writes
/// the CPU-programmed reportRef payload and its own PTIMER timestamp back to back.
///
/// reportRef = (chunkRevision << gpuTimestampReportsLog2) | (gpuTimestampChunkIndex << gpuTimestampReportsPerChunkLog2) | reportIndexInChunk;
/// where gpuTimestampReportsPerChunkLog2 = CeilLog2(recordsPerChunk) and
/// gpuTimestampReportsLog2 = CeilLog2(chunkCount) + gpuTimestampReportsPerChunkLog2, both from the
/// GPU-report chunk pool's section header; the same currency the *Begin records' gpuTimestampRef
/// fields carry. chunkRevision is the chunk's recycling revision, so a ref from newer work over the
/// same chunk compares greater by construction.
///
/// A producer pads a chunk's unused tail with the refs the consumer expects over a zeroed payload:
/// ptimer == 0 marks a pad, never a timestamp -- PTIMER reads 0 only at cold reset, so a real
/// report cannot carry it -- and readers drop such reports rather than joining them.
struct NvTrace_CUDA_ApiTrace_GpuTimestampReport
{
    NvTrace_RevisionedRef reportRef;
    uint64_t ptimer;
};
#define NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GpuTimestampReport, ptimer)

struct NvTrace_CUDA_ApiTrace_CudaDriverApiEnter
{
    uint64_t timestampTsc;
    uint32_t driverApiFunctionId;
};
#define NvTrace_CUDA_ApiTrace_CudaDriverApiEnter_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_CudaDriverApiEnter, driverApiFunctionId)

struct NvTrace_CUDA_ApiTrace_CudaDriverApiExit
{
    uint64_t timestampTsc;
    uint32_t driverApiFunctionId;
};
#define NvTrace_CUDA_ApiTrace_CudaDriverApiExit_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_CudaDriverApiExit, driverApiFunctionId)

struct NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter
{
    uint64_t timestampTsc;
    uint32_t runtimeApiFunctionId;
};
#define NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter, runtimeApiFunctionId)

struct NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit
{
    uint64_t timestampTsc;
    uint32_t runtimeApiFunctionId;
};
#define NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit, runtimeApiFunctionId)

struct NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch
{
    uint32_t cudaDeviceOrdinal;
    uint64_t custreamId;
    uint64_t cufunctionId;
    uint64_t gridId;
    uint32_t gridDim_x;
    uint32_t gridDim_y;
    uint32_t gridDim_z;
    uint32_t blockDim_x;
    uint32_t blockDim_y;
    uint32_t blockDim_z;
    uint32_t clusterDim_x;
    uint32_t clusterDim_y;
    uint32_t clusterDim_z;
    uint32_t totalSharedMemPerBlock;
    uint64_t enclosingCorrelationId; // Every enclosingCorrelationId in this header points at the DriverApiCall, or -- for graph node work -- the GraphLaunch, with DriverApiCall as the ultimate fallback.
};
#define NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch, enclosingCorrelationId)

struct NvTrace_CUDA_ApiTrace_GraphLaunch
{
    uint64_t custreamId;
    uint64_t cugraphExecId;
    uint64_t enclosingCorrelationId;
};
#define NvTrace_CUDA_ApiTrace_GraphLaunch_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphLaunch, enclosingCorrelationId)

struct NvTrace_CUDA_ApiTrace_MemcpyBegin
{
    uint64_t custreamId;

    uint64_t byteCount;
    enum NvTrace_CUDA_ApiTrace_AddressKind srcAddressKind;
    enum NvTrace_CUDA_ApiTrace_AddressKind dstAddressKind;
    uint32_t isAsync;
    uint64_t srcDevPtr;
    uint64_t dstDevPtr;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint32_t isP2P;
    uint32_t memcpyFlags; // mask of NvTrace_CUDA_ApiTrace_MemcpyFlags
    enum NvTrace_CUDA_ApiTrace_CommandType commandType;
    uint64_t gpuTimestampRefStart;
    uint64_t gpuTimestampRefEnd;
    uint64_t enclosingCorrelationId;
};
#define NvTrace_CUDA_ApiTrace_MemcpyBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemcpyBegin, enclosingCorrelationId)

struct NvTrace_CUDA_ApiTrace_MemsetBegin
{
    uint64_t custreamId;
    uint64_t devPtr;
    uint64_t byteCount;
    uint32_t value;
    enum NvTrace_CUDA_ApiTrace_CommandType commandType;
    uint64_t enclosingCorrelationId;
    uint64_t gpuTimestampRefStart;
    uint64_t gpuTimestampRefEnd;
};
#define NvTrace_CUDA_ApiTrace_MemsetBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemsetBegin, gpuTimestampRefEnd)

struct NvTrace_CUDA_ApiTrace_HostCallbackLaunch
{
    uint64_t custreamId;
    uint64_t callback;
    uint64_t userData;
};
#define NvTrace_CUDA_ApiTrace_HostCallbackLaunch_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_HostCallbackLaunch, userData)

struct NvTrace_CUDA_ApiTrace_HostCallbackExecBegin
{
    uint64_t timestampTsc;
    uint64_t custreamId;
    uint64_t launchApiCorrelationId;
    uint64_t enclosingCorrelationId;
};
#define NvTrace_CUDA_ApiTrace_HostCallbackExecBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_HostCallbackExecBegin, enclosingCorrelationId)

struct NvTrace_CUDA_ApiTrace_HostCallbackExecEnd
{
    uint64_t timestampTsc;
    uint64_t custreamId;
    uint64_t launchApiCorrelationId;
};
#define NvTrace_CUDA_ApiTrace_HostCallbackExecEnd_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_HostCallbackExecEnd, launchApiCorrelationId)

struct NvTrace_CUDA_ApiTrace_GraphMemcpyBegin
{
    uint64_t custreamId;
    uint64_t graphNodeCorrelationId;
    enum NvTrace_CUDA_ApiTrace_CommandType commandType;
    uint64_t enclosingCorrelationId;
    uint64_t gpuTimestampRefStart;
    uint64_t gpuTimestampRefEnd;
};
#define NvTrace_CUDA_ApiTrace_GraphMemcpyBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphMemcpyBegin, gpuTimestampRefEnd)

struct NvTrace_CUDA_ApiTrace_GraphMemsetBegin
{
    uint64_t custreamId;
    uint64_t graphNodeCorrelationId;
    enum NvTrace_CUDA_ApiTrace_CommandType commandType;
    uint64_t enclosingCorrelationId;
    uint64_t gpuTimestampRefStart;
    uint64_t gpuTimestampRefEnd;
};
#define NvTrace_CUDA_ApiTrace_GraphMemsetBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphMemsetBegin, gpuTimestampRefEnd)

struct NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch
{
    uint64_t custreamId;
    uint64_t graphNodeCorrelationId;
    uint64_t enclosingCorrelationId; // Points at the GraphLaunch event, or DriverApiCall as a fallback.
};
#define NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch, graphNodeCorrelationId)
#define NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_B NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch, enclosingCorrelationId)

struct NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin
{
    uint64_t timestampTsc;
    uint64_t custreamId;
    uint64_t graphNodeCorrelationId;
    uint64_t enclosingCorrelationId;
};
#define NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin, enclosingCorrelationId)

struct NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd
{
    uint64_t timestampTsc;
    uint64_t custreamId;
    uint64_t graphNodeCorrelationId;
    uint64_t enclosingCorrelationId; // Points at the GraphLaunch event, or DriverApiCall as a fallback.
};
#define NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd, graphNodeCorrelationId)
#define NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_B NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd, enclosingCorrelationId)

/// The same as CUstreamBatchMemOpType.
enum NvTrace_CUDA_ApiTrace_MemOpType
{
    NvTrace_CUDA_ApiTrace_MemOpType_Invalid = 0,
    NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32 = 1,       // CU_STREAM_MEM_OP_WAIT_VALUE_32
    NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32 = 2,      // CU_STREAM_MEM_OP_WRITE_VALUE_32
    NvTrace_CUDA_ApiTrace_MemOpType_FlushRemoteWrites = 3, // CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES
    NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64 = 4,       // CU_STREAM_MEM_OP_WAIT_VALUE_64
    NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64 = 5,      // CU_STREAM_MEM_OP_WRITE_VALUE_64
    NvTrace_CUDA_ApiTrace_MemOpType_Barrier = 6,           // CU_STREAM_MEM_OP_BARRIER
    NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction = 8    // CU_STREAM_MEM_OP_ATOMIC_REDUCTION
};

/// The same as CUstreamWaitValue_flags.
enum NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags
{
    NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags_GEQ = 0x0,          // CU_STREAM_WAIT_VALUE_GEQ
    NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags_EQ = 0x1,           // CU_STREAM_WAIT_VALUE_EQ
    NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags_AND = 0x2,          // CU_STREAM_WAIT_VALUE_AND
    NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags_NOR = 0x3,          // CU_STREAM_WAIT_VALUE_NOR
    NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags_FLUSH = 0x40000000, // CU_STREAM_WAIT_VALUE_FLUSH
    NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags_FORCE_INT = 0x7fffffff
};

/// The same as CUstreamWriteValue_flags.
enum NvTrace_CUDA_ApiTrace_MemOpWriteValueFlags
{
    NvTrace_CUDA_ApiTrace_MemOpWriteValueFlags_Default = 0x0,         // CU_STREAM_WRITE_VALUE_DEFAULT
    NvTrace_CUDA_ApiTrace_MemOpWriteValueFlags_NoMemoryBarrier = 0x1, // CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
    NvTrace_CUDA_ApiTrace_MemOpWriteValueFlags_FORCE_INT = 0x7fffffff
};

/// The same as CUstreamMemoryBarrier_flags.
enum NvTrace_CUDA_ApiTrace_MemOpMemoryBarrierFlags
{
    NvTrace_CUDA_ApiTrace_MemOpMemoryBarrierFlags_Sys = 0x0, // CU_STREAM_MEMORY_BARRIER_TYPE_SYS
    NvTrace_CUDA_ApiTrace_MemOpMemoryBarrierFlags_Gpu = 0x1, // CU_STREAM_MEMORY_BARRIER_TYPE_GPU
    NvTrace_CUDA_ApiTrace_MemOpMemoryBarrierFlags_FORCE_INT = 0x7fffffff
};

/// The same as CUstreamAtomicReductionOpType.
enum NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType
{
    NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType_OR = 6,  // CU_STREAM_ATOMIC_REDUCTION_OP_OR
    NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType_AND = 5, // CU_STREAM_ATOMIC_REDUCTION_OP_AND
    NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType_ADD = 0  // CU_STREAM_ATOMIC_REDUCTION_OP_ADD
};

/// The same as CUstreamAtomicReductionDataType.
enum NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType
{
    NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType_Unsigned32 = 14, // CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_32
    NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType_Unsigned64 = 22  // CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_64
};

/// The fields common to every memory operation. Applies to eager and graphNode invocations.
/// The record is this struct followed by the type-specific params selected by memOpType; see
/// the NvTrace_CUDA_ApiTrace_MemOp_*Params structs below for the record layout.
struct NvTrace_CUDA_ApiTrace_MemOpBegin
{
    uint64_t custreamId;
    uint64_t graphNodeCorrelationId; // = 0 for eager invocations
    uint64_t enclosingCorrelationId;
    uint64_t gpuTimestampRefStart;
    uint64_t gpuTimestampRefEnd;

    uint16_t memOpType;  // NvTrace_CUDA_ApiTrace_MemOpType
    uint16_t memOpIndex; // cuStreamBatchMemOp() states that count must be less than 256
};
#define NvTrace_CUDA_ApiTrace_MemOpBegin_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemOpBegin, memOpIndex)

/// The type-specific half of a MemOpBegin record: the params memOpType selects, sized exactly at the
/// published memOpBeginSize, so the record length varies by memOpType (FlushRemoteWrites carries none).
/// Every NvTrace_CUDA_ApiTrace_MemOpType value must select exactly one struct here -- WriteValue32/64 and
/// WaitValue32/64 each share one -- with its sizeof reflected in NvTrace_CUDA_ApiTrace_RecordVersionInfo.

/// NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32 or NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64
struct NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams
{
    enum NvTrace_CUDA_ApiTrace_MemOpWriteValueFlags flags;
    uint32_t rsvd0004;
    uint64_t address;
    uint64_t value;
};
#define NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams, value)

/// NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32 or NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64
struct NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams
{
    enum NvTrace_CUDA_ApiTrace_MemOpWaitValueFlags flags;
    uint32_t rsvd0004;
    uint64_t address;
    uint64_t value;
};
#define NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams, value)

/// NvTrace_CUDA_ApiTrace_MemOpType_Barrier
struct NvTrace_CUDA_ApiTrace_MemOp_BarrierParams
{
    enum NvTrace_CUDA_ApiTrace_MemOpMemoryBarrierFlags flags;
    // Pads the params to 8 so that a MemOpBegin record, which is placed at an 8-aligned end address
    // minus its length, keeps its leading uint64_t naturally aligned.
    uint32_t rsvd0004;
};
#define NvTrace_CUDA_ApiTrace_MemOp_BarrierParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemOp_BarrierParams, flags)

/// NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction
struct NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams
{
    uint8_t reductionOp; // NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType
    uint8_t dataType;    // NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType
    uint8_t rsvd0002[6];
    uint64_t address;
    uint64_t value;
};
#define NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams, value)

/// The params trailing a MemOpBegin record, as one type. Only the arm memOpType selects is live.
/// Unlike the graph node params slot, a record is sized exactly for that arm, so the bytes past it
/// are not part of the record -- read no more of an arm than the matching memOp*ParamsSize says.
union NvTrace_CUDA_ApiTrace_MemOpParams
{
    struct NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams writeValue;
    struct NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams waitValue;
    struct NvTrace_CUDA_ApiTrace_MemOp_BarrierParams barrier;
    struct NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams atomicReduction;
};

// The object-destruction thread events. Each is self-contained: the object id and timestamp stay
// meaningful after the object's state-table slot is recycled, and a consumer joins the destroy to
// the created record it holds by the object id alone.

struct NvTrace_CUDA_ApiTrace_ContextDestroyed
{
    uint64_t timestampTsc;
    uint64_t contextId;
};
#define NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ContextDestroyed, contextId)

struct NvTrace_CUDA_ApiTrace_StreamDestroyed
{
    uint64_t timestampTsc;
    uint64_t streamId;
};
#define NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_StreamDestroyed, streamId)

struct NvTrace_CUDA_ApiTrace_ModuleUnloaded
{
    uint64_t timestampTsc;
    uint64_t moduleId;
};
#define NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ModuleUnloaded, moduleId)

struct NvTrace_CUDA_ApiTrace_GraphExecDestroyed
{
    uint64_t timestampTsc;
    uint64_t cugraphExecId;
};
#define NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphExecDestroyed, cugraphExecId)

#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
