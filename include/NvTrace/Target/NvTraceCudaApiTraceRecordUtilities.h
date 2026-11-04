/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>

namespace nv::trace {

    // The CudaApiTrace wire format's record-handling conventions, shared by the provider encoding
    // records and the consumers decoding them.

    inline constexpr uint32_t kCudaApiTraceRecordTypeCount = NvTrace_CUDA_ApiTrace_RecordType_COUNT;

    /// A reader locates and sizes the arm from RecordVersionInfo, never from this union.
    union CudaApiTraceGraphNodeParams
    {
        NvTrace_CUDA_ApiTrace_GraphNode_KernelParams kernel;
        NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams memcpy;
        NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams memset;
        NvTrace_CUDA_ApiTrace_GraphNode_HostParams host;
        NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams childGraph;
        NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams waitEvent;
        NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams eventRecord;
        NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams extSemasSignal;
        NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams extSemasWait;
        NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams memAlloc;
        NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams memFree;
        NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams batchMemOp;
        NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams conditional;
        NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams memcpyBatch;
        NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams memcpyBatch3D;
    };

    inline uint32_t CudaApiTraceGraphNodeParamsMinorVersion(const NvTrace_CUDA_ApiTrace_RecordVersionInfo& recordVersionInfo, NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType)
    {
        switch (graphNodeType)
        {
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel: return recordVersionInfo.graphNodeKernelParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy: return recordVersionInfo.graphNodeMemcpyParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Memset: return recordVersionInfo.graphNodeMemsetParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Host: return recordVersionInfo.graphNodeHostParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Graph: return recordVersionInfo.graphNodeChildGraphParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_WaitEvent: return recordVersionInfo.graphNodeWaitEventParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_EventRecord: return recordVersionInfo.graphNodeEventRecordParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasSignal: return recordVersionInfo.graphNodeExtSemasSignalParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasWait: return recordVersionInfo.graphNodeExtSemasWaitParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemAlloc: return recordVersionInfo.graphNodeMemAllocParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemFree: return recordVersionInfo.graphNodeMemFreeParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_BatchMemOp: return recordVersionInfo.graphNodeBatchMemOpParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Conditional: return recordVersionInfo.graphNodeConditionalParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch: return recordVersionInfo.graphNodeMemcpyBatchParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch3D: return recordVersionInfo.graphNodeMemcpyBatch3DParamsMinorVersion;
            default: return 0;
        }
    }

    inline uint32_t CudaApiTraceGraphNodeParamsFloorMinorVersion(NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType)
    {
        switch (graphNodeType)
        {
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel: return NvTrace_CUDA_ApiTrace_GraphNode_KernelParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy: return NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Memset: return NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Host: return NvTrace_CUDA_ApiTrace_GraphNode_HostParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Graph: return NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_WaitEvent: return NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_EventRecord: return NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasSignal: return NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasWait: return NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemAlloc: return NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemFree: return NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_BatchMemOp: return NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Conditional: return NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch: return NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch3D: return NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams_MINOR_VERSION_A;
            default: return 0;
        }
    }

    /// This build's size for the params arm a graphNodeType selects, or 0 when the type carries no params.
    inline size_t CudaApiTraceGraphNodeParamsFloorSize(NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType)
    {
        switch (graphNodeType)
        {
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_KernelParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Memset: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Host: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_HostParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Graph: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_WaitEvent: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_EventRecord: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasSignal: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_ExtSemasWait: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemAlloc: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemFree: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_BatchMemOp: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_Conditional: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams);
            case NvTrace_CUDA_ApiTrace_GraphNodeType_MemcpyBatch3D: return sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams);
            default: return 0;
        }
    }

    inline uint32_t CudaApiTraceMemOpParamsMinorVersion(const NvTrace_CUDA_ApiTrace_RecordVersionInfo& recordVersionInfo, NvTrace_CUDA_ApiTrace_MemOpType memOpType)
    {
        switch (memOpType)
        {
            case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32:
            case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64: return recordVersionInfo.memOpWriteValueParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32:
            case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64: return recordVersionInfo.memOpWaitValueParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_MemOpType_Barrier: return recordVersionInfo.memOpBarrierParamsMinorVersion;
            case NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction: return recordVersionInfo.memOpAtomicReductionParamsMinorVersion;
            default: return 0;
        }
    }

    inline uint32_t CudaApiTraceMemOpParamsFloorMinorVersion(NvTrace_CUDA_ApiTrace_MemOpType memOpType)
    {
        switch (memOpType)
        {
            case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32:
            case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64: return NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32:
            case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64: return NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_MemOpType_Barrier: return NvTrace_CUDA_ApiTrace_MemOp_BarrierParams_MINOR_VERSION_A;
            case NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction: return NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams_MINOR_VERSION_A;
            default: return 0;
        }
    }

    // This build's producer publishes sizeof as the leading segment's size, so its leading structs
    // must be sized to the alignment of the params that follow -- every arm's, since one published
    // size serves them all. FunctionNameSegment is not here: the name bytes it leads are raw, so a producer
    // owes them no alignment.
    static_assert(sizeof(NvTrace_CUDA_ApiTrace_GraphNodeCreated) % alignof(CudaApiTraceGraphNodeParams) == 0);
    static_assert(sizeof(NvTrace_CUDA_ApiTrace_MemOpBegin) % alignof(NvTrace_CUDA_ApiTrace_MemOpParams) == 0);

    /// A pure function of the wire header; a producer publishes the result as is.
    inline void FillCudaApiTraceRecordVersionInfo(NvTrace_CUDA_ApiTrace_RecordVersionInfo& recordVersionInfo)
    {
        recordVersionInfo.recordVersionInfoSize = sizeof(NvTrace_CUDA_ApiTrace_RecordVersionInfo);
        recordVersionInfo.recordVersionInfoMinorVersion = NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A;

        recordVersionInfo.providerStatisticsSize = sizeof(NvTrace_CUDA_ApiTrace_ProviderStatistics);
        recordVersionInfo.providerStatisticsMinorVersion = NvTrace_CUDA_ApiTrace_ProviderStatistics_MINOR_VERSION_A;

        recordVersionInfo.deviceCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_DeviceCreated);
        recordVersionInfo.deviceCreatedMinorVersion = NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A;
        recordVersionInfo.contextCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_ContextCreated);
        recordVersionInfo.contextCreatedMinorVersion = NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A;
        recordVersionInfo.streamCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_StreamCreated);
        recordVersionInfo.streamCreatedMinorVersion = NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A;
        recordVersionInfo.moduleLoadedSize = sizeof(NvTrace_CUDA_ApiTrace_ModuleLoaded);
        recordVersionInfo.moduleLoadedMinorVersion = NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A;
        recordVersionInfo.functionCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_FunctionCreated);
        recordVersionInfo.functionCreatedMinorVersion = NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A;
        recordVersionInfo.functionNameSegmentSize = sizeof(NvTrace_CUDA_ApiTrace_FunctionNameSegment);
        recordVersionInfo.functionNameSegmentMinorVersion = NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A;
        recordVersionInfo.graphExecCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_GraphExecCreated);
        recordVersionInfo.graphExecCreatedMinorVersion = NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A;
        recordVersionInfo.graphNodeCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNodeCreated);
        recordVersionInfo.graphNodeCreatedMinorVersion = NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A;
        recordVersionInfo.channelCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_ChannelCreated);
        recordVersionInfo.channelCreatedMinorVersion = NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A;

        recordVersionInfo.gpuTimestampReportSize = sizeof(NvTrace_CUDA_ApiTrace_GpuTimestampReport);
        recordVersionInfo.gpuTimestampReportMinorVersion = NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A;
        recordVersionInfo.cudaDriverApiEnterSize = sizeof(NvTrace_CUDA_ApiTrace_CudaDriverApiEnter);
        recordVersionInfo.cudaDriverApiEnterMinorVersion = NvTrace_CUDA_ApiTrace_CudaDriverApiEnter_MINOR_VERSION_A;
        recordVersionInfo.cudaDriverApiExitSize = sizeof(NvTrace_CUDA_ApiTrace_CudaDriverApiExit);
        recordVersionInfo.cudaDriverApiExitMinorVersion = NvTrace_CUDA_ApiTrace_CudaDriverApiExit_MINOR_VERSION_A;
        recordVersionInfo.cudaRuntimeApiEnterSize = sizeof(NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter);
        recordVersionInfo.cudaRuntimeApiEnterMinorVersion = NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter_MINOR_VERSION_A;
        recordVersionInfo.cudaRuntimeApiExitSize = sizeof(NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit);
        recordVersionInfo.cudaRuntimeApiExitMinorVersion = NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit_MINOR_VERSION_A;
        recordVersionInfo.immediateKernelLaunchSize = sizeof(NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch);
        recordVersionInfo.immediateKernelLaunchMinorVersion = NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch_MINOR_VERSION_A;
        recordVersionInfo.graphLaunchSize = sizeof(NvTrace_CUDA_ApiTrace_GraphLaunch);
        recordVersionInfo.graphLaunchMinorVersion = NvTrace_CUDA_ApiTrace_GraphLaunch_MINOR_VERSION_A;
        recordVersionInfo.memcpyBeginSize = sizeof(NvTrace_CUDA_ApiTrace_MemcpyBegin);
        recordVersionInfo.memcpyBeginMinorVersion = NvTrace_CUDA_ApiTrace_MemcpyBegin_MINOR_VERSION_A;
        recordVersionInfo.memsetBeginSize = sizeof(NvTrace_CUDA_ApiTrace_MemsetBegin);
        recordVersionInfo.memsetBeginMinorVersion = NvTrace_CUDA_ApiTrace_MemsetBegin_MINOR_VERSION_A;
        recordVersionInfo.hostCallbackLaunchSize = sizeof(NvTrace_CUDA_ApiTrace_HostCallbackLaunch);
        recordVersionInfo.hostCallbackLaunchMinorVersion = NvTrace_CUDA_ApiTrace_HostCallbackLaunch_MINOR_VERSION_A;
        recordVersionInfo.hostCallbackExecBeginSize = sizeof(NvTrace_CUDA_ApiTrace_HostCallbackExecBegin);
        recordVersionInfo.hostCallbackExecBeginMinorVersion = NvTrace_CUDA_ApiTrace_HostCallbackExecBegin_MINOR_VERSION_A;
        recordVersionInfo.hostCallbackExecEndSize = sizeof(NvTrace_CUDA_ApiTrace_HostCallbackExecEnd);
        recordVersionInfo.hostCallbackExecEndMinorVersion = NvTrace_CUDA_ApiTrace_HostCallbackExecEnd_MINOR_VERSION_A;
        recordVersionInfo.graphMemcpyBeginSize = sizeof(NvTrace_CUDA_ApiTrace_GraphMemcpyBegin);
        recordVersionInfo.graphMemcpyBeginMinorVersion = NvTrace_CUDA_ApiTrace_GraphMemcpyBegin_MINOR_VERSION_A;
        recordVersionInfo.graphMemsetBeginSize = sizeof(NvTrace_CUDA_ApiTrace_GraphMemsetBegin);
        recordVersionInfo.graphMemsetBeginMinorVersion = NvTrace_CUDA_ApiTrace_GraphMemsetBegin_MINOR_VERSION_A;
        recordVersionInfo.graphHostCallbackLaunchSize = sizeof(NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch);
        recordVersionInfo.graphHostCallbackLaunchMinorVersion = NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_B;
        recordVersionInfo.graphHostCallbackExecBeginSize = sizeof(NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin);
        recordVersionInfo.graphHostCallbackExecBeginMinorVersion = NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin_MINOR_VERSION_A;
        recordVersionInfo.graphHostCallbackExecEndSize = sizeof(NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd);
        recordVersionInfo.graphHostCallbackExecEndMinorVersion = NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_B;
        recordVersionInfo.memOpBeginSize = sizeof(NvTrace_CUDA_ApiTrace_MemOpBegin);
        recordVersionInfo.memOpBeginMinorVersion = NvTrace_CUDA_ApiTrace_MemOpBegin_MINOR_VERSION_A;
        recordVersionInfo.contextDestroyedSize = sizeof(NvTrace_CUDA_ApiTrace_ContextDestroyed);
        recordVersionInfo.contextDestroyedMinorVersion = NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A;
        recordVersionInfo.streamDestroyedSize = sizeof(NvTrace_CUDA_ApiTrace_StreamDestroyed);
        recordVersionInfo.streamDestroyedMinorVersion = NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A;
        recordVersionInfo.moduleUnloadedSize = sizeof(NvTrace_CUDA_ApiTrace_ModuleUnloaded);
        recordVersionInfo.moduleUnloadedMinorVersion = NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A;
        recordVersionInfo.graphExecDestroyedSize = sizeof(NvTrace_CUDA_ApiTrace_GraphExecDestroyed);
        recordVersionInfo.graphExecDestroyedMinorVersion = NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A;

        recordVersionInfo.graphNodeKernelParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_KernelParams);
        recordVersionInfo.graphNodeKernelParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_KernelParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeMemcpyParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams);
        recordVersionInfo.graphNodeMemcpyParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeMemsetParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams);
        recordVersionInfo.graphNodeMemsetParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeHostParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_HostParams);
        recordVersionInfo.graphNodeHostParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_HostParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeChildGraphParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams);
        recordVersionInfo.graphNodeChildGraphParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeWaitEventParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams);
        recordVersionInfo.graphNodeWaitEventParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeEventRecordParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams);
        recordVersionInfo.graphNodeEventRecordParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeExtSemasSignalParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams);
        recordVersionInfo.graphNodeExtSemasSignalParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeExtSemasWaitParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams);
        recordVersionInfo.graphNodeExtSemasWaitParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeMemAllocParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams);
        recordVersionInfo.graphNodeMemAllocParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeMemFreeParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams);
        recordVersionInfo.graphNodeMemFreeParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeBatchMemOpParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams);
        recordVersionInfo.graphNodeBatchMemOpParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeConditionalParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams);
        recordVersionInfo.graphNodeConditionalParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeMemcpyBatchParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams);
        recordVersionInfo.graphNodeMemcpyBatchParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams_MINOR_VERSION_A;
        recordVersionInfo.graphNodeMemcpyBatch3DParamsSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams);
        recordVersionInfo.graphNodeMemcpyBatch3DParamsMinorVersion = NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams_MINOR_VERSION_A;

        recordVersionInfo.memOpWriteValueParamsSize = sizeof(NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams);
        recordVersionInfo.memOpWriteValueParamsMinorVersion = NvTrace_CUDA_ApiTrace_MemOp_WriteValueParams_MINOR_VERSION_A;
        recordVersionInfo.memOpWaitValueParamsSize = sizeof(NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams);
        recordVersionInfo.memOpWaitValueParamsMinorVersion = NvTrace_CUDA_ApiTrace_MemOp_WaitValueParams_MINOR_VERSION_A;
        recordVersionInfo.memOpBarrierParamsSize = sizeof(NvTrace_CUDA_ApiTrace_MemOp_BarrierParams);
        recordVersionInfo.memOpBarrierParamsMinorVersion = NvTrace_CUDA_ApiTrace_MemOp_BarrierParams_MINOR_VERSION_A;
        recordVersionInfo.memOpAtomicReductionParamsSize = sizeof(NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams);
        recordVersionInfo.memOpAtomicReductionParamsMinorVersion = NvTrace_CUDA_ApiTrace_MemOp_AtomicReductionParams_MINOR_VERSION_A;
    }

} // namespace nv::trace
