/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer_v1.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_CUDA_ApiTrace_ProviderTypeUuid_V1 = NVTRACE_UUID_VALUE(0xba0f13bb, 0x6c6, 0x4b4f, 0xb8, 0x90, 0xac, 0xe6, 0x2e, 0x53, 0x8e, 0x89);

/// Buffer layout structure for the V1 CUDA API Trace provider.
/// Note: this provider will allocate one giant buffer for all sub-buffers. Thus, offsets are sufficient to discover sub-buffers.
struct NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1
{
    /// This is a COFF-style scheme, where each offset is relative to the pointer of the provider buffer header struct.

    /// points at NvTrace_CUDA_ApiTrace_EventVersionInfo_V1
    uint64_t eventVersionInfoOffset;
    uint32_t eventVersionInfoSize;
    uint32_t eventVersionInfoMinorVersion;

    /// points at a flatbuffer of type nv::trace::FB::Provider::CudaApiTrace::ProviderConfiguration
    uint64_t configurationOffset;
    uint32_t configurationSize;

    /// points at NvTrace_CUDA_ApiTrace_ContextState_V1
    uint64_t contextStatesOffset;
    uint32_t contextStatesCount;

    /// points at NvTrace_CUDA_ApiTrace_StreamState_V1
    uint64_t streamStatesOffset;
    uint32_t streamStatesCount;

    /// points at NvTrace_CUDA_ApiTrace_ModuleState_V1
    uint64_t moduleStatesOffset;
    uint32_t moduleStatesCount;

    /// points at NvTrace_CUDA_ApiTrace_FunctionState_V1
    uint64_t functionStatesOffset;
    uint32_t functionStatesCount;

    /// points at NvTrace_CUDA_ApiTrace_GraphExecState_V1
    uint64_t graphExecStatesOffset;
    uint32_t graphExecStatesCount;

    /// points at entries of NvTrace_CUDA_ApiTrace_GraphNodeState_V1 followed by the params graphNodeType selects,
    /// strided by graphNodeStateSize + graphNodeParamsSize
    uint64_t graphNodeStatesOffset;
    uint32_t graphNodeStatesCount;

    /// points at an array of NvTrace_CUDA_ApiTrace_StringTableEntry_V1
    uint64_t stringTableEntriesOffset;
    uint32_t stringTableEntriesCount;
    /// points at char[]; substrings are pointed at by NvTrace_CUDA_ApiTrace_StringTableEntry_V1
    uint64_t stringTableDataOffset;
    uint32_t stringTableDataSize;

    /// points at NvTrace_CUDA_ApiTrace_StateTableStats_V1; represents current values
    uint64_t currentStateTableStatsOffset;
    /// points at NvTrace_CUDA_ApiTrace_StateTableStats_V1; represents max values per-stat across all time
    uint64_t maxStateTableStatsOffset;

    /// points at an array of NvTrace_CUDA_ApiTrace_DeviceState_V1
    uint64_t deviceStatesOffset;
    uint32_t deviceStatesCount; // the exact number of CUDA devices enumerated in this process

    /// points at an array of NvTrace_CpuTimingDescriptor
    uint64_t cpuTimingDescriptorsOffset;
    /// points at an array of NvTrace_CpuTimingEntry
    uint64_t cpuTimingEntriesOffset;
    uint32_t cpuTimingEntriesCount;

    /// points at NvTrace_CUDA_ApiTrace_ChannelState_V1
    uint64_t channelStatesOffset;
    uint32_t channelStatesCount;

    /// points at an array of NvTrace_CircularBufferState; these are 1:1 with CUstream table entriess
    uint64_t gpuTimestampFifoCircularBufferStateOffset;
    uint32_t gpuTimestampFifoCircularBufferStateSize;
    uint32_t gpuTimestampFifoCircularBufferStateMinorVersion;

    /// points at an array of NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1; each block of these is 1:1 with CUstream table entriess
    uint64_t gpuTimestampFifoCellsOffset;
    uint32_t gpuTimestampFifoCellSize;
    uint32_t gpuTimestampFifoCellMinorVersion;
    uint32_t gpuTimestampFifoCellsPerStream;

    /// points at an array of NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1
    uint64_t gpuTimestampChunksOffset;
    uint32_t gpuTimestampChunksCount;
    uint32_t gpuTimestampReportsPerChunk;
};
#define NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1, gpuTimestampReportsPerChunk)

struct NvTrace_CUDA_ApiTrace_EventVersionInfo_V1
{
    uint32_t contextStateSize;
    uint32_t contextStateMinorVersion;

    uint32_t streamStateSize;
    uint32_t streamStateMinorVersion;

    uint32_t moduleStateSize;
    uint32_t moduleStateMinorVersion;

    uint32_t functionStateSize;
    uint32_t functionStateMinorVersion;

    uint32_t graphExecStateSize;
    uint32_t graphExecStateMinorVersion;

    // Describes only the common part of a graph node table entry; the params slot that follows it is
    // described by graphNodeParamsSize and the per-type sizes below, and begins at graphNodeStateSize.
    uint32_t graphNodeStateSize;
    uint32_t graphNodeStateMinorVersion;

    uint32_t stringTableEntrySize;
    uint32_t stringTableEntryMinorVersion;

    uint32_t objectLifetimeEventHeaderSize;
    uint32_t objectLifetimeEventHeaderMinorVersion;
    // The V1 object lifetime event payload is a union so it does not itself have a minor version;
    // but each individual struct within the union must have a minor version for extensibility.
    uint32_t objectLifetimeEventPayloadSize;

    // lifetime events
    uint32_t objectLifetimeContextCreatedSize;
    uint32_t objectLifetimeContextCreatedMinorVersion;
    uint32_t objectLifetimeContextDestroyedSize;
    uint32_t objectLifetimeContextDestroyedMinorVersion;
    uint32_t objectLifetimeStreamCreatedSize;
    uint32_t objectLifetimeStreamCreatedMinorVersion;
    uint32_t objectLifetimeStreamDestroyedSize;
    uint32_t objectLifetimeStreamDestroyedMinorVersion;
    uint32_t objectLifetimeModuleLoadedSize;
    uint32_t objectLifetimeModuleLoadedMinorVersion;
    uint32_t objectLifetimeModuleUnloadedSize;
    uint32_t objectLifetimeModuleUnloadedMinorVersion;
    uint32_t objectLifetimeGraphExecCreatedSize;
    uint32_t objectLifetimeGraphExecCreatedMinorVersion;
    uint32_t objectLifetimeGraphExecDestroyedSize;
    uint32_t objectLifetimeGraphExecDestroyedMinorVersion;

    uint32_t stateTableStatsSize;
    uint32_t stateTableStatsMinorVersion;

    uint32_t deviceStateSize;
    uint32_t deviceStateMinorVersion;

    uint32_t cpuTimingDescriptorSize;
    uint32_t cpuTimingDescriptorMinorVersion;
    uint32_t cpuTimingEntrySize;
    uint32_t cpuTimingEntryMinorVersion;

    // thread events
    uint32_t threadEventCudaDriverApiEnterSize;
    uint32_t threadEventCudaDriverApiEnterMinorVersion;
    uint32_t threadEventCudaDriverApiExitSize;
    uint32_t threadEventCudaDriverApiExitMinorVersion;
    uint32_t threadEventCudaRuntimeApiEnterSize;
    uint32_t threadEventCudaRuntimeApiEnterMinorVersion;
    uint32_t threadEventCudaRuntimeApiExitSize;
    uint32_t threadEventCudaRuntimeApiExitMinorVersion;
    uint32_t threadEventImmediateKernelLaunchSize;
    uint32_t threadEventImmediateKernelLaunchMinorVersion;
    uint32_t threadEventGraphLaunchSize;
    uint32_t threadEventGraphLaunchMinorVersion;
    uint32_t threadEventMemcpyBeginSize;
    uint32_t threadEventMemcpyBeginMinorVersion;
    uint32_t threadEventMemsetBeginSize;
    uint32_t threadEventMemsetBeginMinorVersion;

    uint32_t channelStateSize;
    uint32_t channelStateMinorVersion;
    uint32_t objectLifetimeChannelCreatedSize;
    uint32_t objectLifetimeChannelCreatedMinorVersion;

    uint32_t threadEventGraphMemcpyBeginSize;
    uint32_t threadEventGraphMemcpyBeginMinorVersion;
    uint32_t threadEventGraphMemsetBeginSize;
    uint32_t threadEventGraphMemsetBeginMinorVersion;
    uint32_t threadEventMemOpBeginSize;
    uint32_t threadEventMemOpBeginMinorVersion;

    // host callback thread events
    uint32_t threadEventHostCallbackLaunchSize;
    uint32_t threadEventHostCallbackLaunchMinorVersion;
    uint32_t threadEventHostCallbackExecBeginSize;
    uint32_t threadEventHostCallbackExecBeginMinorVersion;
    uint32_t threadEventHostCallbackExecEndSize;
    uint32_t threadEventHostCallbackExecEndMinorVersion;
    uint32_t threadEventGraphHostCallbackLaunchSize;
    uint32_t threadEventGraphHostCallbackLaunchMinorVersion;
    uint32_t threadEventGraphHostCallbackExecBeginSize;
    uint32_t threadEventGraphHostCallbackExecBeginMinorVersion;
    uint32_t threadEventGraphHostCallbackExecEndSize;
    uint32_t threadEventGraphHostCallbackExecEndMinorVersion;

    // The params slot of a graph node table entry. The slot itself has no minor version, since what
    // occupies it varies by graphNodeType; each params struct carries its own for extensibility.
    uint32_t graphNodeParamsSize;
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

    // The params trailing a MemOpBegin record: unlike the shared graphNodeParamsSize above, each arm
    // here carries its own size and minor version.
    uint32_t memOpWriteValueParamsSize;
    uint32_t memOpWriteValueParamsMinorVersion;
    uint32_t memOpWaitValueParamsSize;
    uint32_t memOpWaitValueParamsMinorVersion;
    uint32_t memOpBarrierParamsSize;
    uint32_t memOpBarrierParamsMinorVersion;
    uint32_t memOpAtomicReductionParamsSize;
    uint32_t memOpAtomicReductionParamsMinorVersion;
};
#define NvTrace_CUDA_ApiTrace_EventVersionInfo_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_EventVersionInfo_V1, memOpAtomicReductionParamsMinorVersion)

struct NvTrace_CUDA_ApiTrace_StateTableStats_V1
{
    uint64_t contextCount;
    uint64_t streamCount;
    uint64_t moduleCount;
    uint64_t functionCount;
    uint64_t graphCount;
    uint64_t graphNodeCount;
    uint64_t graphExecCount;
    uint64_t stringTableEntryCount;
    uint64_t stringTableDataAllocated; // total bytes allocated
    uint64_t stringTableDataUsed;      // total bytes used <= allocated; this is the actual character count
    uint64_t channelCount;
};
#define NvTrace_CUDA_ApiTrace_StateTableStats_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_StateTableStats_V1, channelCount)

/// Represents the static properties of a GPU device.
struct NvTrace_CUDA_ApiTrace_DeviceState_V1
{
    /// cudaDeviceOrdinal is also the array index of this DeviceState in the provider buffer.
    uint32_t cudaDeviceOrdinal;
    /// CUdevice handle.
    uint32_t cudeviceHandle;
    /// GPU UUID, usable to correlate with other device lists.
    NVTRACE_UUID gpuUuid;
};
#define NvTrace_CUDA_ApiTrace_DeviceState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_DeviceState_V1, gpuUuid)

struct NvTrace_CUDA_ApiTrace_ContextState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// Unique ID for the context, defined by the CUDA driver.
    uint64_t contextId;
    /// CUcontext handle.
    uint64_t cucontextHandle;
    /// The device in which this object was created.
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_ContextState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ContextState_V1, cudaDeviceOrdinal)

/// A compute (GR) channel owned by a CUcontext; see NvTrace_CUDA_ApiTrace_ChannelCreated for the field
/// semantics.
struct NvTrace_CUDA_ApiTrace_ChannelState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// NvTrace_CUDA_ApiTrace_ContextState_V1::contextId of the owning context.
    uint64_t cucontextId;
    /// RM channel CID (system-globally unique, monotonically allocated). Matches the FECS GrCtx context_id.
    uint32_t rmCid;
};
#define NvTrace_CUDA_ApiTrace_ChannelState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ChannelState_V1, rmCid)

struct NvTrace_CUDA_ApiTrace_StreamState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// Unique ID for the stream, defined by the CUDA driver.
    uint64_t streamId;
    /// CUstream handle.
    uint64_t custreamHandle;
    /// The context to which this stream belongs.
    uint64_t contextId;
    /// The device in which this object was created.
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_StreamState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_StreamState_V1, cudaDeviceOrdinal)

struct NvTrace_CUDA_ApiTrace_ModuleState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// Unique ID for the CUmodule, defined by the CUDA driver.
    uint64_t moduleId;
    /// CUmodule handle.
    uint64_t cumoduleHandle;

    /// The first function belonging to this module, for the purpose of enumeration.
    uint64_t firstFunctionTableIndex;

    /// The device in which this object was created.
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_ModuleState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ModuleState_V1, cudaDeviceOrdinal)

struct NvTrace_CUDA_ApiTrace_FunctionState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// Unique ID for the CUfunction, defined by the CUDA driver.
    uint64_t cufunctionId;
    /// CUfunction handle.
    uint64_t cufunctionHandle;
    /// The unique ID of the CUmodule that this function belongs to.
    uint64_t moduleId;
    /// The next function belonging to the same module, for the purpose of enumeration.
    uint64_t nextFunctionTableIndex;
    /// Index into the NvTrace_CUDA_ApiTrace_StringTableEntry_V1 array, for the name of this function.
    uint64_t nameStringTableEntryIndex;
    /// The length of the name string, in bytes (characters), not including the NUL terminator.
    uint32_t nameStringLength;
    /// The device in which this object was created.
    uint32_t cudaDeviceOrdinal;
};
#define NvTrace_CUDA_ApiTrace_FunctionState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_FunctionState_V1, cudaDeviceOrdinal)

struct NvTrace_CUDA_ApiTrace_GraphExecState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// Unique ID for the CUgraphExec, defined by the CUDA driver.
    uint64_t cugraphExecId;
    /// CUgraphExec handle.
    uint64_t cugraphExecHandle;
    /// The device in which this object was created.
    uint32_t cudaDeviceOrdinal;

    /// The CUgraph from which this CUgraphExec was created.
    uint64_t cugraphId;

    /// The first graph node belonging to this graph, for the purpose of enumeration.
    uint64_t firstGraphNodeTableIndex;

    /// The number of graph nodes that appear in this provider buffer.
    uint64_t tracedGraphNodeCount;
};
#define NvTrace_CUDA_ApiTrace_GraphExecState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphExecState_V1, tracedGraphNodeCount)

enum NvTrace_CUDA_ApiTrace_GraphNodeType_V1
{
    NvTrace_CUDA_ApiTrace_GraphNodeType_Invalid_V1 = 0,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Other_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Memset_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Host_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Graph_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Empty_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_WaitEvent_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_EventRecord_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasSignal_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasWait_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemAlloc_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemFree_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_BatchMemOp_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_Conditional_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch3D_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_SubgridSchedule_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_COUNT_V1,
    NvTrace_CUDA_ApiTrace_GraphNodeType_FORCE_INT_V1 = 0x7fffffff
};

/// The fields common to every V1 graph node table entry, regardless of type.
/// Each entry is this struct followed by the type-specific params selected by graphNodeType, with an
/// entry stride of graphNodeStateSize + graphNodeParamsSize taken from NvTrace_CUDA_ApiTrace_EventVersionInfo_V1.
struct NvTrace_CUDA_ApiTrace_GraphNodeState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks on the remaining fields.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// Unique ID for the CUgraphNode, defined by the CUDA driver.
    uint64_t cugraphNodeId;
    /// CUgraphNode handle.
    uint64_t cugraphNodeHandle;
    /// The CUgraphExec into which this CUgraphNode was created.
    uint64_t cugraphExecId;
    /// The next function belonging to the same module, for the purpose of enumeration.
    uint64_t nextGraphNodeTableIndex;

    /// The device in which this object was created.
    uint32_t cudaDeviceOrdinal;
    /// Selects which NvTrace_CUDA_ApiTrace_GraphNode*Params struct occupies the entry's params slot.
    enum NvTrace_CUDA_ApiTrace_GraphNodeType_V1 graphNodeType;

    /// V1 ref of the NVTX stack group capturing the ranges open at the node's creation, or
    /// NvTrace_NvtxTrace_INVALID_REF_V1 if none.
    uint64_t nvtxTraceStackGroupRef;
};
#define NvTrace_CUDA_ApiTrace_GraphNodeState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GraphNodeState_V1, nvtxTraceStackGroupRef)

// The V1 wire's object-lifetime record types: the Created announces and Destroyed events the
// consumer-side shim translates into the V2 vocabulary (a record reference, or a self-contained
// NvTrace_CUDA_ApiTrace_*Destroyed record). Only V1 buffers carry these values; they are frozen wire
// facts, deliberately outside NvTrace_CUDA_ApiTrace_RecordType's continuous range.
enum NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_V1
{
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_LO_V1 = 0x10000,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ContextCreated_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ContextDestroyed_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_StreamCreated_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_StreamDestroyed_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ModuleLoaded_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ModuleUnloaded_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_GraphExecCreated_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_GraphExecDestroyed_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ChannelCreated_V1,
    NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_MAX_V1 = 0x1FFFF,
};

enum NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_V1
{
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaDriverApiEnter_V1 = 1,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaDriverApiExit_V1 = 2,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaRuntimeApiEnter_V1 = 3,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaRuntimeApiExit_V1 = 4,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_ImmediateKernelLaunch_V1 = 5,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphLaunch_V1 = 6,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemcpyBegin_V1 = 7,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemcpyEnd_V1 = 8,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemsetBegin_V1 = 9,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemsetEnd_V1 = 10,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_HostCallbackLaunch_V1 = 11,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_HostCallbackExecBegin_V1 = 12,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_HostCallbackExecEnd_V1 = 13,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemcpyBegin_V1 = 14,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemcpyEnd_V1 = 15,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemsetBegin_V1 = 16,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemsetEnd_V1 = 17,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphHostCallbackLaunch_V1 = 18,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphHostCallbackExecBegin_V1 = 19,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphHostCallbackExecEnd_V1 = 20,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemOpBegin_V1 = 21,
    NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemOpEnd_V1 = 22,
};

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextCreated_V1
{
    uint64_t contextId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextCreated_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextCreated_V1, contextId)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamCreated_V1
{
    uint64_t streamId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamCreated_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamCreated_V1, streamId)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleLoaded_V1
{
    uint64_t moduleId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleLoaded_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleLoaded_V1, moduleId)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecCreated_V1
{
    uint64_t cugraphExecId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecCreated_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecCreated_V1, cugraphExecId)

/// The event's data (cucontextId, rmCid) resides in its NvTrace_CUDA_ApiTrace_ChannelState_V1 table entry,
/// referenced by the event header's tableEntryIndex/tableEntryRevision; the payload carries nothing.
struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ChannelCreated_V1
{
    uint8_t rsvd0000; // C requires at least one member
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_ChannelCreated_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ChannelCreated_V1, rsvd0000)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextDestroyed_V1
{
    uint64_t contextId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextDestroyed_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextDestroyed_V1, contextId)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamDestroyed_V1
{
    uint64_t streamId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamDestroyed_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamDestroyed_V1, streamId)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleUnloaded_V1
{
    uint64_t moduleId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleUnloaded_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleUnloaded_V1, moduleId)

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecDestroyed_V1
{
    uint64_t cugraphExecId;
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecDestroyed_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecDestroyed_V1, cugraphExecId)

union NvTrace_CUDA_ApiTrace_ObjectLifetime_EventPayload_V1
{
    struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextDestroyed_V1 contextDestroyed;
    struct NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamDestroyed_V1 streamDestroyed;
    struct NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleUnloaded_V1 moduleUnloaded;
    struct NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecDestroyed_V1 graphExecDestroyed;
};

struct NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1
{
    uint64_t timestampTsc;
    uint64_t tableEntryIndex;
    uint64_t tableEntryRevision;
    // A reader locates the payload at the published objectLifetimeEventHeaderSize.
};
#define NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1, tableEntryRevision)

struct NvTrace_CUDA_ApiTrace_StringTableEntry_V1
{
    /// Encoded {offset, length, terminator}.
    int64_t value;
};
#define NvTrace_CUDA_ApiTrace_StringTableEntry_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_StringTableEntry_V1, value)

// clang-format off
#define NVTRACE_STRING_TABLE_ENTRY_OFFSET     47: 0
#define NVTRACE_STRING_TABLE_ENTRY_LENGTH     62:48
#define NVTRACE_STRING_TABLE_ENTRY_ISLAST     63:63
// clang-format on

struct NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1
{
    // gpuTimestampref = (streamSequenceId << gpuTimestampReportsLog2) | (gpuTimestampChunkIndex << gpuTimestampReportsPerChunkLog2);
    // where gpuTimestampReportsPerChunkLog2 = CeilLog2(gpuTimestampReportsPerChunk);
    // and gpuTimestampReportsLog2 = CeilLog2(gpuTimestampChunksCount) + gpuTimestampReportsPerChunkLog2;
    uint64_t gpuTimestampRef;
};
#define NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1, gpuTimestampRef)

struct NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1
{
    uint64_t streamSequenceId;
    uint64_t ptimer;
};
#define NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1, ptimer)

#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
