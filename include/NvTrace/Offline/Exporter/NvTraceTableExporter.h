/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NvTraceExporterBase.h"
#include "NvTraceGpuPmOfflineMetricsEvaluator.h"
#include "NvTraceTableSchema.h"

namespace nv::trace {

    // Identifies the capture a snapshot belongs to. The traces row is keyed by the capture start, so it must
    // stay the same for every snapshot of one capture.
    struct TraceExportInfo
    {
        std::string_view traceDir;
        uint64_t captureStartUtcNsec = Codec::InvalidTimestamp;
        uint64_t captureEndUtcNsec = Codec::InvalidTimestamp;
    };

    // Emits DataModel contents as database-agnostic table rows into TTableSink.
    // See NvTraceTableSchema.h for the sink contract.
    // A sink may reject a primary key appended twice in one snapshot, so each table emits a key at most once.
    template <typename TTraceReader, typename TTableSink>
    class TableExporter : public TraceExporterBase<TTraceReader>
    {
        using Base = TraceExporterBase<TTraceReader>;
        using CudaApiTraceData = Codec::CudaApiTraceData;
        using NvtxTraceData = Codec::NvtxTraceData;

    public:
        explicit TableExporter(Codec::DataModel<TTraceReader>& dataModel)
            : Base(dataModel)
        {
        }

        // snapshotNumber identifies the snapshot within its capture run, when the source has one.
        bool Export(TTableSink& sink, std::optional<uint64_t> snapshotNumber = std::nullopt, const TraceExportInfo& traceExportInfo = {})
        {
            BuildProcessContexts();
            return ExportTraces(sink, traceExportInfo) && ExportSnapshotMetadata(sink, snapshotNumber, traceExportInfo) && ExportProcessesAndThreads(sink) && ExportCudaResources(sink) && ExportCudaApiCalls(sink) && ExportCudaStreamApiCallRefs(sink) && ExportCudaGraphLaunches(sink) && ExportCudaImmediateKernelLaunches(sink) && ExportCudaGraphNodeLaunches(sink) && ExportCudaGpuCommands(sink) && ExportCudaHostCallbackInstances(sink) && ExportNvtxDomains(sink) && ExportNvtxMarks(sink) && ExportNvtxPushPopRanges(sink) && ExportGpuPartitions(sink) && ExportGpuGrCtxResidentContexts(sink) && ExportGpuPmSamples(sink) && ExportNvtxGpuProjectionSegments(sink);
        }

        static bool IsKnownCpuTimestamp(uint64_t timestampTsc)
        {
            return timestampTsc != Codec::InvalidTimestamp && timestampTsc != Codec::MinTimestampTsc && timestampTsc != Codec::MaxTimestampTsc;
        }

        // Converts a CPU timestamp to UTC, or NULL when it is unknown: invalid, or an unbounded lifetime sentinel.
        TableValue CpuTimestampToUtcValue(uint64_t timestampTsc) const
        {
            if (!IsKnownCpuTimestamp(timestampTsc))
            {
                return {};
            }

            const uint64_t timestampUtcNsec = this->m_dataModel.ConvertTscTickToUtcNsec(timestampTsc);
            if (timestampUtcNsec == Codec::InvalidTimestamp)
            {
                return {};
            }
            return timestampUtcNsec;
        }

        TableValue GpuTimestampToUtcValue(const Codec::GpuPartitionData* pGpuPartition, uint64_t timestampGpuNsec) const
        {
            if (!pGpuPartition || timestampGpuNsec == Codec::InvalidTimestamp)
            {
                return {};
            }

            const uint64_t timestampUtcNsec = this->m_dataModel.ConvertGpuNsecToUtcNsec(*pGpuPartition, timestampGpuNsec);
            if (timestampUtcNsec == Codec::InvalidTimestamp)
            {
                return {};
            }
            return timestampUtcNsec;
        }

        // Every graph launch has an exec ID; only host launches have a launching API call.
        static bool IsValidGraphLaunchIdentity(uint64_t graphLaunchApiCorrelationId, uint64_t graphLaunchCugraphExecId, bool isHostLaunch)
        {
            return graphLaunchCugraphExecId != 0 && (graphLaunchApiCorrelationId != 0) == isHostLaunch;
        }

        // "ns::Kernel<int>(float*)" -> "Kernel". Names that do not parse as a C++ signature are returned whole.
        static std::string_view GetUnqualifiedFunctionName(std::string_view name)
        {
            // The parameter list is the last top-level parenthesized group; earlier ones are qualifiers
            // such as "(anonymous namespace)".
            size_t end = name.size();
            int depth = 0;
            int parenDepth = 0;
            for (size_t i = 0; i < name.size(); ++i)
            {
                const char c = name[i];
                if (c == '<')
                {
                    ++depth;
                }
                else if (c == '>' && depth > 0)
                {
                    --depth;
                }
                else if (c == '(')
                {
                    if (depth == 0 && parenDepth == 0)
                    {
                        end = i;
                    }
                    ++parenDepth;
                }
                else if (c == ')' && parenDepth > 0)
                {
                    --parenDepth;
                }
            }

            if (end > 0 && name[end - 1] == '>')
            {
                depth = 0;
                for (size_t i = end; i > 0; --i)
                {
                    if (name[i - 1] == '>')
                    {
                        ++depth;
                    }
                    else if (name[i - 1] == '<' && --depth == 0)
                    {
                        end = i - 1;
                        break;
                    }
                }
            }

            size_t begin = 0;
            depth = 0;
            for (size_t i = end; i > 0; --i)
            {
                const char c = name[i - 1];
                if (c == '>' || c == ')')
                {
                    ++depth;
                }
                else if ((c == '<' || c == '(') && depth > 0)
                {
                    --depth;
                }
                else if (depth == 0 && (c == ' ' || (c == ':' && i >= 2 && name[i - 2] == ':')))
                {
                    begin = i;
                    break;
                }
            }

            const std::string_view unqualifiedName = name.substr(begin, end - begin);
            return unqualifiedName.empty() ? name : unqualifiedName;
        }

    private:
        // Process-scoped lookups shared by the per-process exporters.
        struct ProcessContext
        {
            std::string_view nodeId;
            uint64_t serviceStartTsc = 0;
            NVTRACE_LUID processLuid = NVTRACE_NIL_LUID;
            const Codec::ProcessData* pProcessData = nullptr;
            std::unordered_map<uint64_t, uint64_t> cucontextIdsByCustreamId;

            const CudaApiTraceData& CudaApiTrace() const
            {
                return pProcessData->cudaApiTrace;
            }

            TableValue CuContextIdValue(uint64_t custreamId) const
            {
                const auto itr = cucontextIdsByCustreamId.find(custreamId);
                return itr != cucontextIdsByCustreamId.end() ? TableValue(itr->second) : TableValue();
            }
        };

        // Device graph launches have no launching API call or stream, so their correlation ID and stream are 0, and
        // their timestamp keys them instead. A host launch's timestamp is unknown until its GridNullEnd.
        struct GraphLaunchKey
        {
            uint64_t graphLaunchCorrelationId = 0;
            uint64_t deviceLaunchTimestampGpuNsec = 0;
            uint64_t timestampGpuNsec = Codec::InvalidTimestamp;
            uint32_t cudaDeviceOrdinal = UINT32_MAX;
            uint64_t custreamId = 0;
            uint64_t uniqueThreadId = 0;
        };

        // Wire fields every non-kernel GPU command carries, whichever begin record it came from.
        struct GpuCommandPayload
        {
            uint64_t custreamId = 0;
            uint64_t startGpuReportRef = 0;
            uint64_t endGpuReportRef = 0;
            uint64_t enclosingCorrelationId = 0;
            NvTrace_CUDA_ApiTrace_CommandType payloadCommandType = NvTrace_CUDA_ApiTrace_CommandType_Invalid;
        };

        // A GPU operation's key in its operation table and in cuda_gpu_operations. Columns that do not apply to
        // operationSource are 0, as nvtx_gpu_projection_segments stores them.
        struct GpuOperationKey
        {
            std::string_view operationSource;
            uint64_t uniqueThreadId = 0;
            uint64_t kernelLaunchCorrelationId = 0;
            uint64_t graphLaunchApiCorrelationId = 0;
            uint64_t graphLaunchCugraphExecId = 0;
            uint32_t graphLaunchHwCorrelationId = 0;
            uint64_t graphLaunchDeviceTimestampGpuNsec = 0;
            uint64_t graphNodeCorrelationId = 0;
            uint64_t startGpuReportRef = 0;
        };

        // Keyed by {launching thread, driver API call index, graph node correlation ID, domain clique index}.
        using NvtxProjectionStackCache = std::map<std::tuple<uint64_t, size_t, uint64_t, uint32_t>, std::vector<Codec::NvtxAssociationStackEntry>>;

        template <size_t ValueCount>
        static bool AppendRow(TTableSink& sink, TableIndex tableIndex, const TableValue (&values)[ValueCount])
        {
            if (!sink.AppendRow(tableIndex, values, ValueCount))
            {
                NVTRACE_LOG_ERR(50, "Failed to append %s row", GetTableDescs()[tableIndex].pName);
                return false;
            }
            return true;
        }

        template <size_t ValueCount>
        static bool QueueUpdate(TTableSink& sink, TableUpdateIndex updateIndex, const TableValue (&values)[ValueCount])
        {
            if (!sink.QueueUpdate(updateIndex, values, ValueCount))
            {
                NVTRACE_LOG_ERR(50, "Failed to queue update %s", GetTableUpdateDescs()[updateIndex].pName);
                return false;
            }
            return true;
        }

        static TableValue NonZeroValue(uint64_t value)
        {
            return value != 0 ? TableValue(value) : TableValue();
        }

        void BuildProcessContexts()
        {
            m_processContexts.clear();
            m_processContexts.reserve(this->m_dataModel.processes.size());
            for (const auto& [processLuid, processData] : this->m_dataModel.processes)
            {
                ProcessContext& context = m_processContexts.emplace_back();
                context.nodeId = this->m_dataModel.systemInfo.hostname;
                context.serviceStartTsc = this->m_dataModel.systemInfo.serviceStartTimestampTsc;
                context.processLuid = processLuid;
                context.pProcessData = &processData;
                for (const auto& [cucontextId, contextData] : processData.cudaApiTrace.contexts)
                {
                    for (const uint64_t custreamId : contextData.custreamIds)
                    {
                        context.cucontextIdsByCustreamId.emplace(custreamId, cucontextId);
                    }
                }
            }
        }

        template <typename TOnProcess>
        bool ForEachProcess(TOnProcess&& onProcess)
        {
            for (const ProcessContext& context : m_processContexts)
            {
                if (!onProcess(context))
                {
                    return false;
                }
            }
            return true;
        }

        // Enter and exit records carry the same correlation ID, so either one keys the call.
        bool GetApiCallCorrelationId(const CudaApiTraceData::ApiCall& apiCall, uint64_t& correlationId) const
        {
            const uint64_t keyRecordLocation = apiCall.startRecordLocation != Codec::InvalidRecordLocation ? apiCall.startRecordLocation : apiCall.endRecordLocation;
            if (keyRecordLocation == Codec::InvalidRecordLocation)
            {
                return false;
            }
            correlationId = this->m_dataModel.GetRecordMetadata(keyRecordLocation).correlationId;
            return true;
        }

        bool GetDriverApiCorrelationId(
            const CudaApiTraceData& cudaApiTrace,
            const CudaApiTraceData::DriverApiCallRef& driverApiCallRef,
            uint64_t& correlationId) const
        {
            const auto threadItr = cudaApiTrace.threads.find(driverApiCallRef.uniqueThreadId);
            if (threadItr == cudaApiTrace.threads.end() || driverApiCallRef.driverApiCallIdx >= threadItr->second.driverApiCalls.size())
            {
                return false;
            }
            return GetApiCallCorrelationId(threadItr->second.driverApiCalls[driverApiCallRef.driverApiCallIdx], correlationId);
        }

        TableValue GetDriverApiCorrelationIdValue(
            const CudaApiTraceData& cudaApiTrace,
            const CudaApiTraceData::DriverApiCallRef& driverApiCallRef) const
        {
            uint64_t correlationId = 0;
            return GetDriverApiCorrelationId(cudaApiTrace, driverApiCallRef, correlationId) ? TableValue(correlationId) : TableValue();
        }

        static const CudaApiTraceData::CuDeviceData* FindCuDevice(const CudaApiTraceData& cudaApiTrace, uint64_t cudaDeviceOrdinal)
        {
            if (cudaDeviceOrdinal >= cudaApiTrace.devices.size() || cudaApiTrace.devices[cudaDeviceOrdinal].gpuUuid == NVTRACE_NIL_UUID)
            {
                return nullptr;
            }
            return &cudaApiTrace.devices[cudaDeviceOrdinal];
        }

        static bool GetGraphLaunchKey(const CudaApiTraceData::GraphLaunch& graphLaunch, GraphLaunchKey& key)
        {
            key = {};
            switch (graphLaunch.source)
            {
                case CudaApiTraceData::GraphLaunchSource::Host:
                    key.graphLaunchCorrelationId = graphLaunch.payload.host.apiCorrelationId;
                    key.timestampGpuNsec = graphLaunch.payload.host.timestampGpuNsec;
                    key.cudaDeviceOrdinal = graphLaunch.payload.host.cudaDeviceOrdinal;
                    key.custreamId = graphLaunch.payload.host.custreamId;
                    key.uniqueThreadId = graphLaunch.payload.host.driverApiCallRef.uniqueThreadId;
                    return true;
                case CudaApiTraceData::GraphLaunchSource::Device:
                    key.deviceLaunchTimestampGpuNsec = graphLaunch.payload.device.timestampGpuNsec;
                    key.timestampGpuNsec = graphLaunch.payload.device.timestampGpuNsec;
                    key.cudaDeviceOrdinal = graphLaunch.payload.device.cudaDeviceOrdinal;
                    return key.timestampGpuNsec != Codec::InvalidTimestamp;
                case CudaApiTraceData::GraphLaunchSource::Unknown:
                    break;
            }
            return false;
        }

        // An ImmediateKernelLaunch entry can also stand for an internal kernel of a memcpy or memset.
        bool IsKernelLaunchRecord(const CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch) const
        {
            return immediateKernelLaunch.recordLocation != Codec::InvalidRecordLocation
                && this->m_dataModel.GetCudaApiTraceRecordType(immediateKernelLaunch.recordLocation) == NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch;
        }

        // Null for entries that get no cuda_immediate_kernel_launches row.
        const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch* GetExportedKernelLaunchPayload(const CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch) const
        {
            return IsKernelLaunchRecord(immediateKernelLaunch) ? this->m_dataModel.GetImmediateKernelLaunchPayload(immediateKernelLaunch.recordLocation) : nullptr;
        }

        static bool HasDriverApiCall(const CudaApiTraceData& cudaApiTrace, const CudaApiTraceData::DriverApiCallRef& driverApiCallRef)
        {
            const auto threadItr = cudaApiTrace.threads.find(driverApiCallRef.uniqueThreadId);
            return threadItr != cudaApiTrace.threads.end() && driverApiCallRef.driverApiCallIdx < threadItr->second.driverApiCalls.size();
        }

        // Host callbacks have no GPU report refs; cuda_host_callback_instances carries them instead.
        static bool IsReportedGpuCommand(const CudaApiTraceData::GpuCommand& gpuCommand)
        {
            return gpuCommand.commandType != CudaApiTraceData::GpuCommandType::Invalid && gpuCommand.commandType != CudaApiTraceData::GpuCommandType::HostCallback;
        }

        bool GetGpuCommandPayload(const CudaApiTraceData& cudaApiTrace, const CudaApiTraceData::GpuCommand& gpuCommand, GpuCommandPayload& payload) const
        {
            const auto fnCopy = [&payload](const auto* pBegin) {
                if (!pBegin)
                {
                    return false;
                }
                payload.custreamId = pBegin->custreamId;
                payload.startGpuReportRef = pBegin->gpuTimestampRefStart;
                payload.endGpuReportRef = pBegin->gpuTimestampRefEnd;
                payload.enclosingCorrelationId = pBegin->enclosingCorrelationId;
                return true;
            };
            const auto fnCopyWithCommandType = [&](const auto* pBegin) {
                if (!fnCopy(pBegin))
                {
                    return false;
                }
                payload.payloadCommandType = pBegin->commandType;
                return true;
            };

            payload = {};
            const uint64_t recordLocation = gpuCommand.recordLocation;
            switch (gpuCommand.commandType)
            {
                case CudaApiTraceData::GpuCommandType::EagerMemcpy:
                    return fnCopyWithCommandType(this->m_dataModel.GetMemcpyBeginPayload(recordLocation));
                case CudaApiTraceData::GpuCommandType::EagerMemset:
                    return fnCopyWithCommandType(this->m_dataModel.GetMemsetBeginPayload(recordLocation));
                case CudaApiTraceData::GpuCommandType::GraphMemcpy:
                    return fnCopyWithCommandType(this->m_dataModel.GetGraphMemcpyBeginPayload(recordLocation));
                case CudaApiTraceData::GpuCommandType::GraphMemset:
                    return fnCopyWithCommandType(this->m_dataModel.GetGraphMemsetBeginPayload(recordLocation));
                case CudaApiTraceData::GpuCommandType::MemOp:
                    return fnCopy(this->m_dataModel.GetMemOpBeginPayload(recordLocation, cudaApiTrace));
                case CudaApiTraceData::GpuCommandType::Invalid:
                case CudaApiTraceData::GpuCommandType::HostCallback:
                    break;
            }
            return false;
        }

        const CudaApiTraceGraphNodeParams* GetGraphNodeParams(
            const CudaApiTraceData& cudaApiTrace,
            uint64_t graphNodeCorrelationId,
            NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType) const
        {
            const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = this->m_dataModel.FindCuGraphNode(cudaApiTrace, graphNodeCorrelationId);
            if (!pGraphNodeData || this->m_dataModel.GetCudaApiTraceGraphNodeType(pGraphNodeData->recordLocation, cudaApiTrace) != graphNodeType)
            {
                return nullptr;
            }
            return this->m_dataModel.GetCudaApiTraceGraphNodeParams(pGraphNodeData->recordLocation, cudaApiTrace, graphNodeType);
        }

        bool IsKernelGraphNode(const CudaApiTraceData& cudaApiTrace, uint64_t graphNodeCorrelationId) const
        {
            return GetGraphNodeParams(cudaApiTrace, graphNodeCorrelationId, NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel) != nullptr;
        }

        static std::string_view GetCopyDirection(NvTrace_CUDA_ApiTrace_AddressKind srcAddressKind, NvTrace_CUDA_ApiTrace_AddressKind dstAddressKind)
        {
            // Indexed by NvTrace_CUDA_ApiTrace_AddressKind, initials as in GetAddressKindName
            static constexpr std::string_view kCopyDirections[NvTrace_CUDA_ApiTrace_AddressKind_COUNT][NvTrace_CUDA_ApiTrace_AddressKind_COUNT] = {
                {"I2I", "I2H", "I2D", "I2A"},
                {"H2I", "H2H", "H2D", "H2A"},
                {"D2I", "D2H", "D2D", "D2A"},
                {"A2I", "A2H", "A2D", "A2A"},
            };
            if (srcAddressKind >= NvTrace_CUDA_ApiTrace_AddressKind_COUNT || dstAddressKind >= NvTrace_CUDA_ApiTrace_AddressKind_COUNT)
            {
                return Base::kUnknownData;
            }
            return kCopyDirections[srcAddressKind][dstAddressKind];
        }

        static std::string_view GetGpuCommandName(CudaApiTraceData::GpuCommandType commandType, const NvTrace_CUDA_ApiTrace_MemOpBegin* pMemOpBegin)
        {
            switch (commandType)
            {
                case CudaApiTraceData::GpuCommandType::EagerMemcpy:
                case CudaApiTraceData::GpuCommandType::GraphMemcpy:
                    return "Memcpy";
                case CudaApiTraceData::GpuCommandType::EagerMemset:
                case CudaApiTraceData::GpuCommandType::GraphMemset:
                    return "Memset";
                case CudaApiTraceData::GpuCommandType::MemOp:
                    switch (pMemOpBegin ? static_cast<NvTrace_CUDA_ApiTrace_MemOpType>(pMemOpBegin->memOpType) : NvTrace_CUDA_ApiTrace_MemOpType_Invalid)
                    {
                        case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32: return "MemOp(WAIT_VALUE_32)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32: return "MemOp(WRITE_VALUE_32)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_FlushRemoteWrites: return "MemOp(FLUSH_REMOTE_WRITES)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64: return "MemOp(WAIT_VALUE_64)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64: return "MemOp(WRITE_VALUE_64)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_Barrier: return "MemOp(BARRIER)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction: return "MemOp(ATOMIC_REDUCTION)";
                        case NvTrace_CUDA_ApiTrace_MemOpType_Invalid: break;
                    }
                    return "MemOp(?\?)";
                case CudaApiTraceData::GpuCommandType::Invalid:
                case CudaApiTraceData::GpuCommandType::HostCallback:
                    break;
            }
            return Base::kUnknownData;
        }

        bool ExportTraces(TTableSink& sink, const TraceExportInfo& traceExportInfo)
        {
            if (traceExportInfo.captureStartUtcNsec == Codec::InvalidTimestamp || traceExportInfo.captureEndUtcNsec == Codec::InvalidTimestamp)
            {
                NVTRACE_LOG_WRN(50, "Skipping traces row: no capture window");
                return true;
            }

            const Codec::SystemInfo& systemInfo = this->m_dataModel.systemInfo;
            const TableValue values[] = {
                std::string_view(systemInfo.hostname),
                traceExportInfo.traceDir,
                traceExportInfo.captureStartUtcNsec,
                traceExportInfo.captureEndUtcNsec,
                std::string_view(systemInfo.osVersion),
                static_cast<uint8_t>(systemInfo.osFamily),
                std::string_view(systemInfo.cpuModel),
                static_cast<uint8_t>(systemInfo.cpuArch),
                systemInfo.sysMemTotal,
                systemInfo.nvidiaRmKernelDriverVersion.major,
                systemInfo.nvidiaRmKernelDriverVersion.minor,
                systemInfo.nvidiaRmKernelDriverVersion.patch,
                systemInfo.nvidiaMrmKernelDriverVersion.major,
                systemInfo.nvidiaMrmKernelDriverVersion.minor,
                systemInfo.nvidiaMrmKernelDriverVersion.patch,
            };
            return AppendRow(sink, TableIndex_Traces, values);
        }

        bool ExportSnapshotMetadata(TTableSink& sink, std::optional<uint64_t> snapshotNumber, const TraceExportInfo& traceExportInfo)
        {
            if (traceExportInfo.captureStartUtcNsec == Codec::InvalidTimestamp || traceExportInfo.captureEndUtcNsec == Codec::InvalidTimestamp)
            {
                NVTRACE_LOG_WRN(50, "Skipping snapshot_metadata row: no capture window");
                return true;
            }

            const uint64_t observationStartUtcNsec = this->m_dataModel.m_observationStartUtcNsec;
            const uint64_t observationEndUtcNsec = this->m_dataModel.m_observationEndUtcNsec;
            if (observationStartUtcNsec == Codec::InvalidTimestamp || observationEndUtcNsec == Codec::InvalidTimestamp)
            {
                NVTRACE_LOG_WRN(50, "Skipping snapshot_metadata row: no observation window");
                return true;
            }

            const uint64_t ingestedAtUtcNsec = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

            const TableValue values[] = {
                std::string_view(this->m_dataModel.systemInfo.hostname),
                traceExportInfo.captureStartUtcNsec,
                observationStartUtcNsec,
                observationEndUtcNsec,
                snapshotNumber ? TableValue(*snapshotNumber) : TableValue(),
                ingestedAtUtcNsec,
            };
            NVTRACE_LOG_ERR_RETURN_IF(!sink.AppendRow(TableIndex_SnapshotMetadata, values, std::size(values)), false, 50, "Failed to append snapshot_metadata row");
            return true;
        }

        bool ExportProcessesAndThreads(TTableSink& sink)
        {
            const std::string_view nodeId = this->m_dataModel.systemInfo.hostname;
            const uint64_t serviceStartTsc = this->m_dataModel.systemInfo.serviceStartTimestampTsc;
            for (const auto& [processLuid, processData] : this->m_dataModel.processes)
            {
                const TableValue processValues[] = {
                    nodeId,
                    serviceStartTsc,
                    processLuid,
                    static_cast<int32_t>(processData.namespacePid),
                    static_cast<int32_t>(processData.namespaceParentPid),
                    static_cast<int32_t>(processData.globalPid),
                    processData.pidNamespaceId.inode,
                    processData.pidNamespaceId.device,
                    std::string_view(processData.path),
                    CpuTimestampToUtcValue(processData.creationTimestampTsc),
                    CpuTimestampToUtcValue(processData.destructionTimestampTsc),
                };
                if (!sink.AppendRow(TableIndex_Processes, processValues, std::size(processValues)))
                {
                    NVTRACE_LOG_ERR(50, "Failed to append processes row for processLuid=%" PRIu64, static_cast<uint64_t>(processLuid));
                    return false;
                }

                struct ThreadRow
                {
                    uint32_t nativeThreadId = 0;
                    uint64_t creationTimestampTsc = Codec::InvalidTimestamp;
                    uint64_t destructionTimestampTsc = Codec::InvalidTimestamp;
                };
                std::unordered_map<uint64_t, ThreadRow> threadRows;
                for (const auto& [_, threadEventProviderData] : processData.threadEventProviders)
                {
                    for (const auto& [uniqueThreadId, threadData] : threadEventProviderData.threads)
                    {
                        ThreadRow& threadRow = threadRows[uniqueThreadId];
                        if (threadRow.nativeThreadId != 0 && threadData.nativeThreadId != 0 && threadRow.nativeThreadId != threadData.nativeThreadId)
                        {
                            NVTRACE_LOG_ERR(
                                50,
                                "Conflicting native thread IDs for processLuid=%" PRIu64 " uniqueThreadId=%" PRIu64 ": %" PRIu32 " and %" PRIu32,
                                static_cast<uint64_t>(processLuid),
                                uniqueThreadId,
                                threadRow.nativeThreadId,
                                threadData.nativeThreadId);
                            return false;
                        }
                        if (threadRow.nativeThreadId == 0)
                        {
                            threadRow.nativeThreadId = threadData.nativeThreadId;
                        }

                        const uint64_t creationTimestampTsc = threadData.creationTimestampTsc;
                        if (IsKnownCpuTimestamp(creationTimestampTsc) && (threadRow.creationTimestampTsc == Codec::InvalidTimestamp || creationTimestampTsc < threadRow.creationTimestampTsc))
                        {
                            threadRow.creationTimestampTsc = creationTimestampTsc;
                        }
                        const uint64_t destructionTimestampTsc = threadData.destructionTimestampTsc;
                        if (IsKnownCpuTimestamp(destructionTimestampTsc) && (threadRow.destructionTimestampTsc == Codec::InvalidTimestamp || destructionTimestampTsc > threadRow.destructionTimestampTsc))
                        {
                            threadRow.destructionTimestampTsc = destructionTimestampTsc;
                        }
                    }
                }

                for (const auto& [uniqueThreadId, threadRow] : threadRows)
                {
                    const TableValue threadValues[] = {
                        nodeId,
                        serviceStartTsc,
                        processLuid,
                        uniqueThreadId,
                        threadRow.nativeThreadId,
                        CpuTimestampToUtcValue(threadRow.creationTimestampTsc),
                        CpuTimestampToUtcValue(threadRow.destructionTimestampTsc),
                    };
                    if (!sink.AppendRow(TableIndex_ThreadEventThreads, threadValues, std::size(threadValues)))
                    {
                        NVTRACE_LOG_ERR(
                            50,
                            "Failed to append thread_event_threads row for processLuid=%" PRIu64 " uniqueThreadId=%" PRIu64,
                            static_cast<uint64_t>(processLuid),
                            uniqueThreadId);
                        return false;
                    }
                }
            }
            return true;
        }

        bool ExportCudaResources(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();

                for (const CudaApiTraceData::CuDeviceData& device : cudaApiTrace.devices)
                {
                    // Ordinals the process never used leave placeholder entries without a GPU
                    if (device.gpuUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        device.gpuUuid,
                        device.cudaDeviceOrdinal,
                        device.cudeviceHandle,
                    };
                    if (!AppendRow(sink, TableIndex_CudaDevices, values))
                    {
                        return false;
                    }
                }

                for (const auto& [cucontextId, contextData] : cudaApiTrace.contexts)
                {
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        cucontextId,
                        contextData.cucontextHandle,
                        CpuTimestampToUtcValue(contextData.creationTimestampTsc),
                        CpuTimestampToUtcValue(contextData.destructionTimestampTsc),
                    };
                    if (!AppendRow(sink, TableIndex_CudaContexts, values))
                    {
                        return false;
                    }
                }

                for (const auto& [custreamId, streamData] : cudaApiTrace.streams)
                {
                    const CudaApiTraceData::CuDeviceData* pDevice = FindCuDevice(cudaApiTrace, streamData.cudaDeviceOrdinal);
                    if (!pDevice)
                    {
                        NVTRACE_LOG_INF(90, "Skipping cuda_streams row for custreamId=%" PRIu64 ": no GPU for ordinal %" PRIu32, custreamId, streamData.cudaDeviceOrdinal);
                        continue;
                    }
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        pDevice->gpuUuid,
                        custreamId,
                        context.CuContextIdValue(custreamId),
                        streamData.custreamHandle,
                        streamData.cudaDeviceOrdinal,
                        CpuTimestampToUtcValue(streamData.creationTimestampTsc),
                        CpuTimestampToUtcValue(streamData.destructionTimestampTsc),
                    };
                    if (!AppendRow(sink, TableIndex_CudaStreams, values))
                    {
                        return false;
                    }
                }

                for (const auto& [cumoduleId, moduleData] : cudaApiTrace.modules)
                {
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        cumoduleId,
                        moduleData.cumoduleHandle,
                        moduleData.contextId,
                        CpuTimestampToUtcValue(moduleData.creationTimestampTsc),
                        CpuTimestampToUtcValue(moduleData.destructionTimestampTsc),
                    };
                    if (!AppendRow(sink, TableIndex_CudaModules, values))
                    {
                        return false;
                    }
                }

                for (const auto& [cufunctionId, functionData] : cudaApiTrace.functions)
                {
                    const CuFunctionNames functionNames = this->GetCuFunctionNames(cudaApiTrace, cufunctionId);
                    const std::string_view demangledName = functionNames.pDemangledName;
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        cufunctionId,
                        functionData.cufunctionHandle,
                        functionData.cumoduleId,
                        std::string_view(functionNames.pMangledName),
                        demangledName,
                        GetUnqualifiedFunctionName(demangledName),
                        CpuTimestampToUtcValue(functionData.creationTimestampTsc),
                        CpuTimestampToUtcValue(functionData.destructionTimestampTsc),
                    };
                    if (!AppendRow(sink, TableIndex_CudaFunctions, values))
                    {
                        return false;
                    }
                }

                for (const auto& [cugraphExecId, graphExecData] : cudaApiTrace.graphExecs)
                {
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        cugraphExecId,
                        graphExecData.cugraphExecHandle,
                        CpuTimestampToUtcValue(graphExecData.creationTimestampTsc),
                        CpuTimestampToUtcValue(graphExecData.destructionTimestampTsc),
                    };
                    if (!AppendRow(sink, TableIndex_CudaGraphExecs, values))
                    {
                        return false;
                    }
                }

                for (const auto& [graphNodeCorrelationId, graphNodeData] : cudaApiTrace.graphNodes)
                {
                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        graphNodeCorrelationId,
                        graphNodeData.cugraphNodeId,
                        graphNodeData.cugraphNodeHandle,
                        graphNodeData.cugraphExecId,
                        graphNodeData.uniqueThreadId,
                        CpuTimestampToUtcValue(graphNodeData.creationTimestampTsc),
                        CpuTimestampToUtcValue(graphNodeData.destructionTimestampTsc),
                    };
                    if (!AppendRow(sink, TableIndex_CudaGraphNodes, values))
                    {
                        return false;
                    }
                }
                return true;
            });
        }

        bool ExportCudaApiCalls(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                for (const auto& [uniqueThreadId, threadData] : context.CudaApiTrace().threads)
                {
                    for (const CudaApiTraceData::RuntimeApiCall& runtimeApiCall : threadData.runtimeApiCalls)
                    {
                        uint64_t correlationId = 0;
                        if (!GetApiCallCorrelationId(runtimeApiCall, correlationId))
                        {
                            continue;
                        }
                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            static_cast<uint64_t>(uniqueThreadId),
                            correlationId,
                            CpuTimestampToUtcValue(runtimeApiCall.startTimestampTsc),
                            CpuTimestampToUtcValue(runtimeApiCall.endTimestampTsc),
                            static_cast<uint16_t>(runtimeApiCall.runtimeApiFunctionId),
                            std::string_view(this->GetRuntimeApiFunctionName(runtimeApiCall.runtimeApiFunctionId)),
                        };
                        if (!AppendRow(sink, TableIndex_CudaRuntimeApiCalls, values))
                        {
                            return false;
                        }
                    }

                    for (const CudaApiTraceData::DriverApiCall& driverApiCall : threadData.driverApiCalls)
                    {
                        uint64_t correlationId = 0;
                        if (!GetApiCallCorrelationId(driverApiCall, correlationId))
                        {
                            continue;
                        }
                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            static_cast<uint64_t>(uniqueThreadId),
                            correlationId,
                            CpuTimestampToUtcValue(driverApiCall.startTimestampTsc),
                            CpuTimestampToUtcValue(driverApiCall.endTimestampTsc),
                            static_cast<uint16_t>(driverApiCall.driverApiFunctionId),
                            std::string_view(this->GetDriverApiFunctionName(driverApiCall.driverApiFunctionId)),
                        };
                        if (!AppendRow(sink, TableIndex_CudaDriverApiCalls, values))
                        {
                            return false;
                        }
                    }
                }
                return true;
            });
        }

        // A side of the reference with no API call keys as correlation ID 0.
        bool ExportCudaStreamApiCallRefs(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();
                for (const auto& [custreamId, streamData] : cudaApiTrace.streams)
                {
                    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> exportedRefs;
                    for (const CudaApiTraceData::ApiCallRef& apiCallRef : streamData.apiCallReferences)
                    {
                        const auto threadItr = cudaApiTrace.threads.find(apiCallRef.uniqueThreadId);
                        if (threadItr == cudaApiTrace.threads.end())
                        {
                            continue;
                        }
                        const CudaApiTraceData::ThreadData& threadData = threadItr->second;

                        uint64_t driverApiCallCorrelationId = 0;
                        const bool hasDriverApiCall = apiCallRef.driverApiCallIdx < threadData.driverApiCalls.size()
                                                   && GetApiCallCorrelationId(threadData.driverApiCalls[apiCallRef.driverApiCallIdx], driverApiCallCorrelationId);
                        uint64_t runtimeApiCallCorrelationId = 0;
                        const bool hasRuntimeApiCall = apiCallRef.runtimeApiCallIdx < threadData.runtimeApiCalls.size()
                                                    && GetApiCallCorrelationId(threadData.runtimeApiCalls[apiCallRef.runtimeApiCallIdx], runtimeApiCallCorrelationId);
                        if (!hasDriverApiCall && !hasRuntimeApiCall)
                        {
                            continue;
                        }
                        if (!exportedRefs.emplace(apiCallRef.uniqueThreadId, driverApiCallCorrelationId, runtimeApiCallCorrelationId).second)
                        {
                            continue;
                        }

                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            custreamId,
                            apiCallRef.uniqueThreadId,
                            driverApiCallCorrelationId,
                            runtimeApiCallCorrelationId,
                        };
                        if (!AppendRow(sink, TableIndex_CudaStreamApiCallRefs, values))
                        {
                            return false;
                        }
                    }
                }
                return true;
            });
        }

        bool ExportCudaGraphLaunches(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();
                for (const CudaApiTraceData::GraphLaunch& graphLaunch : cudaApiTrace.graphLaunches)
                {
                    GraphLaunchKey key;
                    if (!GetGraphLaunchKey(graphLaunch, key))
                    {
                        continue;
                    }
                    const bool isHostLaunch = graphLaunch.source == CudaApiTraceData::GraphLaunchSource::Host;
                    if (!IsValidGraphLaunchIdentity(key.graphLaunchCorrelationId, graphLaunch.cugraphExecId, isHostLaunch))
                    {
                        NVTRACE_LOG_WRN(
                            90,
                            "Skipping cuda_graph_launches row with invalid graph launch identity: apiCorrelationId=%" PRIu64 " cugraphExecId=%" PRIu64,
                            key.graphLaunchCorrelationId,
                            graphLaunch.cugraphExecId);
                        continue;
                    }
                    const CudaApiTraceData::CuDeviceData* pDevice = FindCuDevice(cudaApiTrace, key.cudaDeviceOrdinal);
                    if (!pDevice)
                    {
                        NVTRACE_LOG_INF(90, "Skipping cuda_graph_launches row for hwCorrelationId=%" PRIu32 ": no GPU for ordinal %" PRIu32, graphLaunch.hwCorrelationId, key.cudaDeviceOrdinal);
                        continue;
                    }
                    const Codec::GpuPartitionData* pGpuPartition = this->m_dataModel.FindGpuPartitionByGpuUuid(pDevice->gpuUuid);

                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        pDevice->gpuUuid,
                        key.graphLaunchCorrelationId,
                        key.custreamId,
                        isHostLaunch ? context.CuContextIdValue(key.custreamId) : TableValue(),
                        static_cast<uint8_t>(graphLaunch.source),
                        graphLaunch.hwCorrelationId,
                        graphLaunch.cugraphExecId,
                        NonZeroValue(key.uniqueThreadId),
                        isHostLaunch ? GetDriverApiCorrelationIdValue(cudaApiTrace, graphLaunch.payload.host.driverApiCallRef) : TableValue(),
                        key.deviceLaunchTimestampGpuNsec,
                        key.timestampGpuNsec != Codec::InvalidTimestamp ? TableValue(key.timestampGpuNsec) : TableValue(),
                        GpuTimestampToUtcValue(pGpuPartition, key.timestampGpuNsec),
                        isHostLaunch ? TableValue() : TableValue(key.cudaDeviceOrdinal),
                    };
                    if (!AppendRow(sink, TableIndex_CudaGraphLaunches, values))
                    {
                        return false;
                    }
                }

                for (const CudaApiTraceData::CuDeviceData& device : cudaApiTrace.devices)
                {
                    if (device.gpuUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }
                    const Codec::GpuPartitionData* pGpuPartition = this->m_dataModel.FindGpuPartitionByGpuUuid(device.gpuUuid);
                    for (const auto& [hwCorrelationId, timestampsGpuNsec] : device.unmatchedHostGraphLaunchTimestampsGpuNsec)
                    {
                        for (const uint64_t timestampGpuNsec : timestampsGpuNsec)
                        {
                            const TableValue values[] = {
                                context.nodeId,
                                context.serviceStartTsc,
                                context.processLuid,
                                device.gpuUuid,
                                hwCorrelationId,
                                timestampGpuNsec,
                                GpuTimestampToUtcValue(pGpuPartition, timestampGpuNsec),
                            };
                            if (!QueueUpdate(sink, TableUpdateIndex_CudaGraphLaunchTimestamp, values))
                            {
                                return false;
                            }
                        }
                    }
                }
                return true;
            });
        }

        bool ExportCudaImmediateKernelLaunches(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                for (const CudaApiTraceData::CuDeviceData& device : context.CudaApiTrace().devices)
                {
                    if (device.gpuUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }
                    const Codec::GpuPartitionData* pGpuPartition = this->m_dataModel.FindGpuPartitionByGpuUuid(device.gpuUuid);

                    for (const CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch : device.immediateKernelLaunches)
                    {
                        const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch* pPayload = GetExportedKernelLaunchPayload(immediateKernelLaunch);
                        if (!pPayload)
                        {
                            continue;
                        }

                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            device.gpuUuid,
                            device.cudaDeviceOrdinal,
                            immediateKernelLaunch.apiCorrelationId,
                            immediateKernelLaunch.GetHwCorrelationId(),
                            immediateKernelLaunch.driverApiCallRef.uniqueThreadId,
                            GetDriverApiCorrelationIdValue(context.CudaApiTrace(), immediateKernelLaunch.driverApiCallRef),
                            pPayload->custreamId,
                            context.CuContextIdValue(pPayload->custreamId),
                            pPayload->cufunctionId,
                            pPayload->gridId,
                            pPayload->gridDim_x,
                            pPayload->gridDim_y,
                            pPayload->gridDim_z,
                            pPayload->blockDim_x,
                            pPayload->blockDim_y,
                            pPayload->blockDim_z,
                            pPayload->clusterDim_x,
                            pPayload->clusterDim_y,
                            pPayload->clusterDim_z,
                            pPayload->totalSharedMemPerBlock,
                            uint32_t{0}, // The source payload does not yet retain compiled static shared memory separately.
                            GpuTimestampToUtcValue(pGpuPartition, immediateKernelLaunch.startTimestampGpuNsec),
                            GpuTimestampToUtcValue(pGpuPartition, immediateKernelLaunch.endTimestampGpuNsec),
                        };
                        if (!AppendRow(sink, TableIndex_CudaImmediateKernelLaunches, values))
                        {
                            return false;
                        }
                    }

                    for (const auto& [hwCorrelationId, unmatchedExecutions] : device.unmatchedImmediateKernelExecutions)
                    {
                        for (const auto& unmatchedExecution : unmatchedExecutions)
                        {
                            const TableValue values[] = {
                                context.nodeId,
                                context.serviceStartTsc,
                                context.processLuid,
                                device.gpuUuid,
                                hwCorrelationId,
                                GpuTimestampToUtcValue(pGpuPartition, unmatchedExecution.startTimestampGpuNsec),
                                GpuTimestampToUtcValue(pGpuPartition, unmatchedExecution.endTimestampGpuNsec),
                            };
                            if (!QueueUpdate(sink, TableUpdateIndex_CudaImmediateKernelLaunchGpuTimes, values))
                            {
                                return false;
                            }
                        }
                    }
                }
                return true;
            });
        }

        // Rows need the kernel params of a Kernel graph node; launches of other node types are skipped.
        bool ExportCudaGraphNodeLaunches(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();
                for (const CudaApiTraceData::CuDeviceData& device : cudaApiTrace.devices)
                {
                    if (device.gpuUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }
                    const Codec::GpuPartitionData* pGpuPartition = this->m_dataModel.FindGpuPartitionByGpuUuid(device.gpuUuid);
                    if (!pGpuPartition)
                    {
                        continue;
                    }

                    for (const CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch : device.graphNodeLaunches)
                    {
                        if (graphNodeLaunch.startTimestampGpuNsec == Codec::InvalidTimestamp)
                        {
                            continue;
                        }
                        const CudaApiTraceGraphNodeParams* pGraphNodeParams = GetGraphNodeParams(cudaApiTrace, graphNodeLaunch.graphNodeCorrelationId, NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel);
                        if (!pGraphNodeParams)
                        {
                            continue;
                        }
                        const NvTrace_CUDA_ApiTrace_GraphNode_KernelParams& kernelParams = pGraphNodeParams->kernel;
                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            device.gpuUuid,
                            device.cudaDeviceOrdinal,
                            graphNodeLaunch.graphNodeCorrelationId,
                            graphNodeLaunch.startTimestampGpuNsec,
                            kernelParams.cufunctionId,
                            kernelParams.gridDimX,
                            kernelParams.gridDimY,
                            kernelParams.gridDimZ,
                            kernelParams.blockDimX,
                            kernelParams.blockDimY,
                            kernelParams.blockDimZ,
                            kernelParams.clusterDimX,
                            kernelParams.clusterDimY,
                            kernelParams.clusterDimZ,
                            kernelParams.totalSharedMemPerBlock,
                            uint32_t{0}, // The source payload does not yet retain compiled static shared memory separately.
                            GpuTimestampToUtcValue(pGpuPartition, graphNodeLaunch.startTimestampGpuNsec),
                            GpuTimestampToUtcValue(pGpuPartition, graphNodeLaunch.endTimestampGpuNsec),
                        };
                        if (!AppendRow(sink, TableIndex_CudaGraphNodeLaunches, values))
                        {
                            return false;
                        }
                    }

                    for (const CudaApiTraceData::GraphNodeLaunch& unmatchedEnd : device.unmatchedGraphNodeLaunchEnds)
                    {
                        if (!IsKernelGraphNode(cudaApiTrace, unmatchedEnd.graphNodeCorrelationId))
                        {
                            continue;
                        }
                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            device.gpuUuid,
                            unmatchedEnd.graphNodeCorrelationId,
                            unmatchedEnd.endTimestampGpuNsec,
                            GpuTimestampToUtcValue(pGpuPartition, unmatchedEnd.endTimestampGpuNsec),
                        };
                        if (!QueueUpdate(sink, TableUpdateIndex_CudaGraphNodeLaunchGpuEnd, values))
                        {
                            return false;
                        }
                    }
                }
                return true;
            });
        }

        // Mem-ops of one batch share their report refs, so the batch is one cuda_gpu_commands row with a
        // cuda_gpu_mem_ops row per operation.
        bool ExportCudaGpuCommands(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();
                for (const CudaApiTraceData::CuDeviceData& device : cudaApiTrace.devices)
                {
                    if (device.gpuUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }
                    const Codec::GpuPartitionData* pGpuPartition = this->m_dataModel.FindGpuPartitionByGpuUuid(device.gpuUuid);

                    std::set<uint64_t> exportedStartGpuReportRefs;
                    for (const CudaApiTraceData::GpuCommand& gpuCommand : device.gpuCommands)
                    {
                        GpuCommandPayload payload;
                        if (!IsReportedGpuCommand(gpuCommand) || !GetGpuCommandPayload(cudaApiTrace, gpuCommand, payload))
                        {
                            continue;
                        }

                        const NvTrace_CUDA_ApiTrace_MemOpBegin* pMemOpBegin = gpuCommand.commandType == CudaApiTraceData::GpuCommandType::MemOp
                                                                                ? this->m_dataModel.GetMemOpBeginPayload(gpuCommand.recordLocation, cudaApiTrace)
                                                                                : nullptr;
                        const TableValue graphNodeCorrelationId = gpuCommand.graphNodeCorrelationId != UINT64_MAX ? TableValue(gpuCommand.graphNodeCorrelationId) : TableValue();

                        if (exportedStartGpuReportRefs.insert(payload.startGpuReportRef).second)
                        {
                            const TableValue values[] = {
                                context.nodeId,
                                context.serviceStartTsc,
                                context.processLuid,
                                device.gpuUuid,
                                device.cudaDeviceOrdinal,
                                payload.startGpuReportRef,
                                payload.endGpuReportRef,
                                static_cast<uint8_t>(gpuCommand.commandType),
                                payload.custreamId,
                                context.CuContextIdValue(payload.custreamId),
                                GpuTimestampToUtcValue(pGpuPartition, gpuCommand.startTimestampGpuNsec),
                                GpuTimestampToUtcValue(pGpuPartition, gpuCommand.endTimestampGpuNsec),
                                graphNodeCorrelationId,
                                NonZeroValue(gpuCommand.driverApiCallRef.uniqueThreadId),
                                GetDriverApiCorrelationIdValue(cudaApiTrace, gpuCommand.driverApiCallRef),
                                GetGpuCommandName(gpuCommand.commandType, pMemOpBegin),
                            };
                            if (!AppendRow(sink, TableIndex_CudaGpuCommands, values))
                            {
                                return false;
                            }
                        }

                        if (!ExportCudaGpuCommandDetails(sink, context, device, gpuCommand, payload, pMemOpBegin, graphNodeCorrelationId))
                        {
                            return false;
                        }
                    }
                }

                // A report doesn't name its GPU or say whether it starts or ends its command, so it is offered to every
                // GPU of the process as both. Only the row on the right GPU with the right reference matches.
                for (const CudaApiTraceData::GpuTimestampReport& unmatchedReport : cudaApiTrace.unmatchedGpuTimestampReports)
                {
                    for (const CudaApiTraceData::CuDeviceData& device : cudaApiTrace.devices)
                    {
                        const Codec::GpuPartitionData* pGpuPartition = device.gpuUuid != NVTRACE_NIL_UUID ? this->m_dataModel.FindGpuPartitionByGpuUuid(device.gpuUuid) : nullptr;
                        if (!pGpuPartition)
                        {
                            continue;
                        }
                        const TableValue values[] = {
                            context.nodeId,
                            context.serviceStartTsc,
                            context.processLuid,
                            device.gpuUuid,
                            unmatchedReport.gpuTimestampRef,
                            GpuTimestampToUtcValue(pGpuPartition, unmatchedReport.ptimer),
                        };
                        for (const TableUpdateIndex updateIndex : {TableUpdateIndex_CudaGpuCommandGpuStart, TableUpdateIndex_CudaGpuCommandGpuEnd})
                        {
                            if (!QueueUpdate(sink, updateIndex, values))
                            {
                                return false;
                            }
                        }
                    }
                }
                return true;
            });
        }

        bool ExportCudaGpuCommandDetails(
            TTableSink& sink,
            const ProcessContext& context,
            const CudaApiTraceData::CuDeviceData& device,
            const CudaApiTraceData::GpuCommand& gpuCommand,
            const GpuCommandPayload& payload,
            const NvTrace_CUDA_ApiTrace_MemOpBegin* pMemOpBegin,
            const TableValue& graphNodeCorrelationId)
        {
            const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();

            // NvTrace_CUDA_ApiTrace_MemcpyBegin or NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams
            const auto fnAppendMemcpy = [&](const auto* pMemcpy) {
                if (!pMemcpy)
                {
                    return true;
                }
                const TableValue values[] = {
                    context.nodeId,
                    context.serviceStartTsc,
                    context.processLuid,
                    device.gpuUuid,
                    device.cudaDeviceOrdinal,
                    payload.startGpuReportRef,
                    payload.endGpuReportRef,
                    payload.custreamId,
                    graphNodeCorrelationId,
                    payload.enclosingCorrelationId,
                    static_cast<uint8_t>(payload.payloadCommandType),
                    static_cast<uint64_t>(pMemcpy->byteCount),
                    static_cast<uint8_t>(pMemcpy->srcAddressKind),
                    static_cast<uint8_t>(pMemcpy->dstAddressKind),
                    static_cast<uint64_t>(pMemcpy->srcDevPtr),
                    static_cast<uint64_t>(pMemcpy->dstDevPtr),
                    static_cast<uint64_t>(pMemcpy->width),
                    static_cast<uint64_t>(pMemcpy->height),
                    static_cast<uint64_t>(pMemcpy->depth),
                    (pMemcpy->memcpyFlags & NvTrace_CUDA_ApiTrace_MemcpyFlags_IsAsync) != 0,
                    (pMemcpy->memcpyFlags & NvTrace_CUDA_ApiTrace_MemcpyFlags_IsP2P) != 0,
                    GetCopyDirection(pMemcpy->srcAddressKind, pMemcpy->dstAddressKind),
                };
                return AppendRow(sink, TableIndex_CudaGpuMemcpyDetails, values);
            };

            // NvTrace_CUDA_ApiTrace_MemsetBegin or NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams
            const auto fnAppendMemset = [&](const auto* pMemset) {
                if (!pMemset)
                {
                    return true;
                }
                const TableValue values[] = {
                    context.nodeId,
                    context.serviceStartTsc,
                    context.processLuid,
                    device.gpuUuid,
                    device.cudaDeviceOrdinal,
                    payload.startGpuReportRef,
                    payload.endGpuReportRef,
                    payload.custreamId,
                    graphNodeCorrelationId,
                    payload.enclosingCorrelationId,
                    static_cast<uint8_t>(payload.payloadCommandType),
                    static_cast<uint64_t>(pMemset->devPtr),
                    static_cast<uint64_t>(pMemset->byteCount),
                    static_cast<uint32_t>(pMemset->value),
                };
                return AppendRow(sink, TableIndex_CudaGpuMemsetDetails, values);
            };

            const auto fnGraphNodeParams = [&](NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType) {
                return gpuCommand.graphNodeCorrelationId != UINT64_MAX ? GetGraphNodeParams(cudaApiTrace, gpuCommand.graphNodeCorrelationId, graphNodeType) : nullptr;
            };

            switch (gpuCommand.commandType)
            {
                case CudaApiTraceData::GpuCommandType::EagerMemcpy:
                    return fnAppendMemcpy(this->m_dataModel.GetMemcpyBeginPayload(gpuCommand.recordLocation));
                case CudaApiTraceData::GpuCommandType::EagerMemset:
                    return fnAppendMemset(this->m_dataModel.GetMemsetBeginPayload(gpuCommand.recordLocation));
                case CudaApiTraceData::GpuCommandType::GraphMemcpy:
                {
                    const CudaApiTraceGraphNodeParams* pGraphNodeParams = fnGraphNodeParams(NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy);
                    return fnAppendMemcpy(pGraphNodeParams ? &pGraphNodeParams->memcpy : nullptr);
                }
                case CudaApiTraceData::GpuCommandType::GraphMemset:
                {
                    const CudaApiTraceGraphNodeParams* pGraphNodeParams = fnGraphNodeParams(NvTrace_CUDA_ApiTrace_GraphNodeType_Memset);
                    return fnAppendMemset(pGraphNodeParams ? &pGraphNodeParams->memset : nullptr);
                }
                case CudaApiTraceData::GpuCommandType::MemOp:
                    return !pMemOpBegin || ExportCudaGpuMemOp(sink, context, device, gpuCommand, payload, *pMemOpBegin, graphNodeCorrelationId);
                case CudaApiTraceData::GpuCommandType::Invalid:
                case CudaApiTraceData::GpuCommandType::HostCallback:
                    break;
            }
            return true;
        }

        bool ExportCudaGpuMemOp(
            TTableSink& sink,
            const ProcessContext& context,
            const CudaApiTraceData::CuDeviceData& device,
            const CudaApiTraceData::GpuCommand& gpuCommand,
            const GpuCommandPayload& payload,
            const NvTrace_CUDA_ApiTrace_MemOpBegin& memOpBegin,
            const TableValue& graphNodeCorrelationId)
        {
            const auto memOpType = static_cast<NvTrace_CUDA_ApiTrace_MemOpType>(memOpBegin.memOpType);
            const NvTrace_CUDA_ApiTrace_MemOpParams* pMemOpParams = this->m_dataModel.GetMemOpParamsPayload(gpuCommand.recordLocation, context.CudaApiTrace(), memOpType);

            TableValue flags{};
            TableValue address{};
            TableValue value{};
            TableValue reductionOp{};
            TableValue dataType{};
            if (pMemOpParams)
            {
                switch (memOpType)
                {
                    case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32:
                    case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64:
                        flags = static_cast<int32_t>(pMemOpParams->waitValue.flags);
                        address = static_cast<uint64_t>(pMemOpParams->waitValue.address);
                        value = static_cast<uint64_t>(pMemOpParams->waitValue.value);
                        break;
                    case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32:
                    case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64:
                        flags = static_cast<int32_t>(pMemOpParams->writeValue.flags);
                        address = static_cast<uint64_t>(pMemOpParams->writeValue.address);
                        value = static_cast<uint64_t>(pMemOpParams->writeValue.value);
                        break;
                    case NvTrace_CUDA_ApiTrace_MemOpType_Barrier:
                        flags = static_cast<int32_t>(pMemOpParams->barrier.flags);
                        break;
                    case NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction:
                        address = static_cast<uint64_t>(pMemOpParams->atomicReduction.address);
                        value = static_cast<uint64_t>(pMemOpParams->atomicReduction.value);
                        reductionOp = static_cast<uint8_t>(pMemOpParams->atomicReduction.reductionOp);
                        dataType = static_cast<uint8_t>(pMemOpParams->atomicReduction.dataType);
                        break;
                    case NvTrace_CUDA_ApiTrace_MemOpType_FlushRemoteWrites:
                    case NvTrace_CUDA_ApiTrace_MemOpType_Invalid:
                        break;
                }
            }

            const TableValue values[] = {
                context.nodeId,
                context.serviceStartTsc,
                context.processLuid,
                device.gpuUuid,
                device.cudaDeviceOrdinal,
                payload.startGpuReportRef,
                payload.endGpuReportRef,
                payload.custreamId,
                graphNodeCorrelationId,
                payload.enclosingCorrelationId,
                static_cast<uint16_t>(memOpBegin.memOpType),
                static_cast<uint16_t>(memOpBegin.memOpIndex),
                flags,
                address,
                value,
                reductionOp,
                dataType,
            };
            return AppendRow(sink, TableIndex_CudaGpuMemOps, values);
        }

        // A graph callback recurs once per graph launch, so its launch correlation key is the enclosing graph
        // launch; an eager callback keys by its own launch with graph node 0.
        bool ExportCudaHostCallbackInstances(TTableSink& sink)
        {
            return ForEachProcess([&](const ProcessContext& context) {
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();
                std::set<std::pair<uint64_t, uint64_t>> exportedKeys;
                for (const CudaApiTraceData::HostCallbackInstance& hostCallbackInstance : cudaApiTrace.hostCallbackInstances)
                {
                    const bool isEager = hostCallbackInstance.kind == CudaApiTraceData::HostCallbackKind::Eager;
                    const uint64_t launchCorrelationId = isEager ? hostCallbackInstance.launchApiCorrelationId : hostCallbackInstance.enclosingCorrelationId;
                    const uint64_t graphNodeCorrelationId = isEager ? 0 : hostCallbackInstance.graphNodeCorrelationId;
                    if (!exportedKeys.emplace(launchCorrelationId, graphNodeCorrelationId).second)
                    {
                        continue;
                    }

                    const CudaApiTraceData::CuStreamData* pStream = this->m_dataModel.FindCuStream(cudaApiTrace, hostCallbackInstance.custreamId);
                    const CudaApiTraceData::CuDeviceData* pDevice = pStream ? FindCuDevice(cudaApiTrace, pStream->cudaDeviceOrdinal) : nullptr;
                    const TableValue graphLaunchApiCorrelationId = isEager ? TableValue() : TableValue(hostCallbackInstance.enclosingCorrelationId);
                    const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = isEager ? nullptr : this->m_dataModel.FindCuGraphNode(cudaApiTrace, graphNodeCorrelationId);
                    const TableValue graphLaunchCugraphExecId = pGraphNodeData ? NonZeroValue(pGraphNodeData->cugraphExecId) : TableValue();

                    const TableValue values[] = {
                        context.nodeId,
                        context.serviceStartTsc,
                        context.processLuid,
                        pDevice ? TableValue(pDevice->gpuUuid) : TableValue(),
                        static_cast<int8_t>(hostCallbackInstance.kind),
                        CpuTimestampToUtcValue(hostCallbackInstance.startTimestampTsc),
                        CpuTimestampToUtcValue(hostCallbackInstance.endTimestampTsc),
                        hostCallbackInstance.custreamId,
                        context.CuContextIdValue(hostCallbackInstance.custreamId),
                        launchCorrelationId,
                        graphLaunchApiCorrelationId,
                        graphLaunchCugraphExecId,
                        graphNodeCorrelationId,
                        isEager ? NonZeroValue(hostCallbackInstance.enclosingCorrelationId) : TableValue(),
                        isEager ? TableValue(hostCallbackInstance.callback) : TableValue(),
                        isEager ? TableValue(hostCallbackInstance.userData) : TableValue(),
                        NonZeroValue(hostCallbackInstance.executingUniqueThreadId),
                        NonZeroValue(hostCallbackInstance.driverApiCallRef.uniqueThreadId),
                    };
                    if (!AppendRow(sink, TableIndex_CudaHostCallbackInstances, values))
                    {
                        return false;
                    }
                }
                return true;
            });
        }

        bool ExportNvtxDomains(TTableSink& sink)
        {
            const std::string_view nodeId = this->m_dataModel.systemInfo.hostname;
            const uint64_t serviceStartTsc = this->m_dataModel.systemInfo.serviceStartTimestampTsc;
            for (const auto& [processLuid, processData] : this->m_dataModel.processes)
            {
                const Codec::NvtxTraceData& nvtxTrace = processData.nvtxTrace;
                for (const auto& [domainIndex, domainData] : nvtxTrace.domains)
                {
                    TableValue domainName{};
                    if (domainData.domainCreatedRecordLocation != Codec::InvalidRecordLocation)
                    {
                        domainName = std::string_view(this->m_dataModel.GetNvtxTraceDomainName(nvtxTrace, domainIndex));
                    }

                    const TableValue values[] = {
                        nodeId,
                        serviceStartTsc,
                        processLuid,
                        domainIndex,
                        domainName,
                    };
                    if (!sink.AppendRow(TableIndex_NvtxDomains, values, std::size(values)))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to append nvtx_domains row for processLuid=%" PRIu64 " domainIndex=%" PRIu32, static_cast<uint64_t>(processLuid), domainIndex);
                        return false;
                    }
                }
            }
            return true;
        }

        bool ExportNvtxMarks(TTableSink& sink)
        {
            std::string messageStorage;
            return ForEachProcess([&](const ProcessContext& context) {
                const NvtxTraceData& nvtxTrace = context.pProcessData->nvtxTrace;
                for (const NvtxTraceData::DomainCliqueData& domainClique : nvtxTrace.domainCliques)
                {
                    for (const auto& [uniqueThreadId, threadData] : domainClique.threads)
                    {
                        for (const NvtxTraceData::MarkData& mark : threadData.marks)
                        {
                            if (mark.timestampTsc == Codec::InvalidTimestamp)
                            {
                                continue;
                            }
                            const uint64_t timestampUtcNsec = this->m_dataModel.ConvertTscTickToUtcNsec(mark.timestampTsc);
                            if (timestampUtcNsec == Codec::InvalidTimestamp)
                            {
                                continue;
                            }

                            const TableValue values[] = {
                                context.nodeId,
                                context.serviceStartTsc,
                                context.processLuid,
                                mark.uniqueThreadId,
                                mark.correlationId,
                                timestampUtcNsec,
                                mark.domainIndex,
                                std::string_view(this->m_dataModel.GetNvtxTraceDomainName(nvtxTrace, mark.domainIndex)),
                                static_cast<uint8_t>(mark.messageType),
                                std::string_view(this->m_dataModel.GetNvtxTraceMarkMessage(nvtxTrace, mark, messageStorage)),
                            };
                            if (!AppendRow(sink, TableIndex_NvtxMarks, values))
                            {
                                return false;
                            }
                        }
                    }
                }
                return true;
            });
        }

        // Exports observation-time ranges and the record-time ranges graph nodes still reference, so every
        // association row has a range to join.
        bool ExportNvtxPushPopRanges(TTableSink& sink)
        {
            std::string messageStorage;
            return ForEachProcess([&](const ProcessContext& context) {
                const NvtxTraceData& nvtxTrace = context.pProcessData->nvtxTrace;
                std::set<uint64_t> exportedRangeRefs;
                for (const NvtxTraceData::DomainCliqueData& domainClique : nvtxTrace.domainCliques)
                {
                    for (const auto& [uniqueThreadId, threadData] : domainClique.threads)
                    {
                        const std::vector<NvtxTraceData::PushPopRangeData>& pushPopRanges = threadData.pushPopRanges;
                        for (const NvtxTraceData::PushPopRangeData& pushPopRange : pushPopRanges)
                        {
                            if (!pushPopRange.isObservationTimeRange && !pushPopRange.isRecordTimeRange)
                            {
                                continue;
                            }
                            const bool hasKnownStart = (pushPopRange.approximationMask & Codec::NvtxTraceData::FieldMask_StartTimestamp) == 0;
                            const bool hasKnownEnd = (pushPopRange.approximationMask & Codec::NvtxTraceData::FieldMask_EndTimestamp) == 0;
                            if ((hasKnownStart && pushPopRange.startTimestampTsc == Codec::InvalidTimestamp)
                                || (hasKnownEnd && pushPopRange.endTimestampTsc == Codec::InvalidTimestamp))
                            {
                                continue;
                            }
                            const TableValue startTimestamp = hasKnownStart
                                                                ? CpuTimestampToUtcValue(pushPopRange.startTimestampTsc)
                                                                : TableValue();
                            const TableValue endTimestamp = hasKnownEnd
                                                              ? CpuTimestampToUtcValue(pushPopRange.endTimestampTsc)
                                                              : TableValue();

                            const bool hasParent = pushPopRange.parentRangeRef != 0;
                            const bool isRoot = pushPopRange.sourceNestingDepth == 1;
                            if (hasParent == isRoot)
                            {
                                NVTRACE_LOG_WRN(
                                    90,
                                    "Skipping nvtx_push_pop_ranges row with inconsistent parent and nesting depth: nvtxRangeRef=%" PRIu64 " nestingDepth=%" PRIu16 " hasParent=%d",
                                    pushPopRange.rangeRef,
                                    pushPopRange.sourceNestingDepth,
                                    hasParent);
                                continue;
                            }
                            if (!exportedRangeRefs.insert(pushPopRange.rangeRef).second)
                            {
                                NVTRACE_LOG_WRN(90, "Skipping duplicate nvtx_push_pop_ranges row for nvtxRangeRef=%" PRIu64, pushPopRange.rangeRef);
                                continue;
                            }

                            const TableValue parentRangeRef = hasParent ? TableValue(pushPopRange.parentRangeRef) : TableValue();
                            const TableValue values[] = {
                                context.nodeId,
                                context.serviceStartTsc,
                                context.processLuid,
                                pushPopRange.uniqueThreadId,
                                pushPopRange.rangeRef,
                                startTimestamp,
                                endTimestamp,
                                pushPopRange.domainIndex,
                                std::string_view(this->m_dataModel.GetNvtxTraceDomainName(nvtxTrace, pushPopRange.domainIndex)),
                                pushPopRange.sourceNestingDepth,
                                static_cast<uint8_t>(pushPopRange.messageType),
                                std::string_view(this->m_dataModel.GetNvtxTracePushPopRangeMessage(nvtxTrace, pushPopRange, messageStorage)),
                                parentRangeRef,
                                pushPopRange.isRecordTimeRange,
                                pushPopRange.isObservationTimeRange,
                            };
                            if (!AppendRow(sink, TableIndex_NvtxPushPopRanges, values))
                            {
                                return false;
                            }
                        }
                    }
                }
                return true;
            });
        }

        bool ExportGpuPartitions(TTableSink& sink)
        {
            const std::string_view nodeId = this->m_dataModel.systemInfo.hostname;
            for (const auto& [gpuPartitionLuid, gpuPartitionData] : this->m_dataModel.gpuPartitions)
            {
                const Codec::GpuPartitionInfo& info = gpuPartitionData.info;
                const TableValue values[] = {
                    gpuPartitionData.gpuUuid,
                    nodeId,
                    std::string_view(info.chipName),
                    std::string_view(info.deviceName),
                };
                if (!sink.AppendRow(TableIndex_GpuPartitions, values, std::size(values)))
                {
                    NVTRACE_LOG_ERR(50, "Failed to append gpu_partitions row for gpuPartitionLuid=%" PRIu64, static_cast<uint64_t>(gpuPartitionLuid));
                    return false;
                }

                for (const NVTRACE_UUID& providerUuid : info.supportedProviders)
                {
                    const TableValue providerValues[] = {
                        gpuPartitionData.gpuUuid,
                        providerUuid,
                    };
                    if (!sink.AppendRow(TableIndex_GpuPartitionSupportedProviders, providerValues, std::size(providerValues)))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to append gpu_partition_supported_providers row for gpuPartitionLuid=%" PRIu64, static_cast<uint64_t>(gpuPartitionLuid));
                        return false;
                    }
                }

                for (const NvTrace_CudaAttribute& attribute : info.attributes)
                {
                    const TableValue attributeValues[] = {
                        gpuPartitionData.gpuUuid,
                        attribute.key,
                        attribute.value,
                    };
                    if (!sink.AppendRow(TableIndex_GpuPartitionCudaAttributes, attributeValues, std::size(attributeValues)))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to append gpu_partition_cuda_attributes row for gpuPartitionLuid=%" PRIu64 " key=%" PRIu32, static_cast<uint64_t>(gpuPartitionLuid), attribute.key);
                        return false;
                    }
                }
            }
            return true;
        }

        bool ExportGpuGrCtxResidentContexts(TTableSink& sink)
        {
            using Codec::GpuGrCtxTraceData;
            for (const auto& [gpuPartitionLuid, gpuPartition] : this->m_dataModel.gpuPartitions)
            {
                for (const GpuGrCtxTraceData::ResidentContext& residentContext : gpuPartition.gpuGrCtxTrace.residentContexts)
                {
                    const bool isLoadObserved = residentContext.loadSequenceNumber != GpuGrCtxTraceData::InvalidSequenceNumber;
                    const bool isSaveObserved = residentContext.saveSequenceNumber != GpuGrCtxTraceData::InvalidSequenceNumber;
                    const TableValue ctxSaveUtcNsec = isSaveObserved ? GpuTimestampToUtcValue(&gpuPartition, residentContext.ctxSaveTimestampGpuNsec) : TableValue();
                    if (!isLoadObserved)
                    {
                        // The builder fabricates a residency with neither event when the window has no GR context events
                        if (!isSaveObserved)
                        {
                            continue;
                        }
                        if (residentContext.isLoadFabricated)
                        {
                            NVTRACE_LOG_WRN(100, "Skipping gpu_gr_ctx_resident_contexts residency saved at %" PRIu64 " ns on GPU %s: its load event was dropped", residentContext.ctxSaveTimestampGpuNsec, ToString(gpuPartition.gpuUuid).c_str());
                            continue;
                        }

                        // The load is before this snapshot
                        const TableValue values[] = {
                            gpuPartition.gpuUuid,
                            residentContext.saveSequenceNumber - 1,
                            ctxSaveUtcNsec,
                        };
                        if (!QueueUpdate(sink, TableUpdateIndex_GpuGrCtxResidentContextSave, values))
                        {
                            return false;
                        }
                        continue;
                    }

                    const Codec::ProcessData* pProcess = residentContext.isMultipleEventsDropped ? nullptr : this->m_dataModel.FindProcess(residentContext.processLuid);
                    TableValue serviceStartTsc{};
                    TableValue processLuid{};
                    TableValue cucontextId{};
                    if (pProcess)
                    {
                        serviceStartTsc = this->m_dataModel.systemInfo.serviceStartTimestampTsc;
                        processLuid = residentContext.processLuid;
                        for (const auto& [contextId, contextData] : pProcess->cudaApiTrace.contexts)
                        {
                            if (std::find(contextData.channelRmCids.begin(), contextData.channelRmCids.end(), residentContext.rmCid) != contextData.channelRmCids.end())
                            {
                                cucontextId = contextId;
                                break;
                            }
                        }
                    }

                    const TableValue values[] = {
                        gpuPartition.gpuUuid,
                        residentContext.loadSequenceNumber,
                        serviceStartTsc,
                        processLuid,
                        GpuTimestampToUtcValue(&gpuPartition, residentContext.ctxLoadTimestampGpuNsec),
                        ctxSaveUtcNsec,
                        residentContext.vmid,
                        residentContext.pid,
                        residentContext.isMultipleEventsDropped,
                        cucontextId,
                    };
                    if (!AppendRow(sink, TableIndex_GpuGrCtxResidentContexts, values))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        bool ExportGpuPmSamples(TTableSink& sink)
        {
            for (auto& [gpuPartitionLuid, gpuPartition] : this->m_dataModel.gpuPartitions)
            {
                const Codec::GpuPmCountersData& gpuPmCounters = gpuPartition.gpuPmCounters;
                if (gpuPmCounters.sampleRecordLocations.empty() || gpuPmCounters.pmSampleRateHz == 0)
                {
                    continue;
                }

                char logContext[64];
                snprintf(logContext, sizeof(logContext), "GPU partition %" PRIu64, static_cast<uint64_t>(gpuPartitionLuid));
                Codec::GpuPmExportPrepared prepared = Codec::PrepareGpuPmMetricsForExport(this->m_dataModel, gpuPartition, gpuPartition.info.chipName, logContext);
                const Codec::GpuPmExportMetricPlan& exportPlan = prepared.plan;
                Codec::MetricsEvaluationContext& evaluationContext = prepared.evaluationContext;
                std::vector<double> offlineMetricValues(evaluationContext.GetMetricNames().size());

                const NVTRACE_UUID& gpuUuid = gpuPartition.gpuUuid;
                bool succeeded = true;
                std::set<uint64_t> exportedEndTimestamps;
                this->m_dataModel.ForEachGpuPmCountersSamples(gpuPartition, [&](const Codec::GpuPmCountersSample& sample) {
                    if (!succeeded || sample.startTimestampUtcNsec == Codec::InvalidTimestamp || sample.endTimestampUtcNsec == Codec::InvalidTimestamp)
                    {
                        return;
                    }
                    if (!exportedEndTimestamps.insert(sample.endTimestampUtcNsec).second)
                    {
                        return;
                    }

                    const TableValue sampleValues[] = {
                        gpuUuid,
                        sample.startTimestampUtcNsec,
                        sample.endTimestampUtcNsec,
                    };
                    if (!AppendRow(sink, TableIndex_GpuPmSamples, sampleValues))
                    {
                        succeeded = false;
                        return;
                    }

                    const bool offlineMetricValuesAvailable = !offlineMetricValues.empty()
                                                           && sample.pSampleRangeData
                                                           && evaluationContext.SetRangeData(sample.pSampleRangeData, sample.sampleRangeDataSize)
                                                           && evaluationContext.EvaluateToGpuValues(offlineMetricValues.data(), offlineMetricValues.size());
                    for (size_t metricIndex = 0; metricIndex < exportPlan.metricNamesForExport.size(); ++metricIndex)
                    {
                        double metricValue = 0.0;
                        if (!Codec::TryResolveGpuPmExportMetricValue(
                                exportPlan,
                                metricIndex,
                                sample.pSampleEvaluatedMetrics,
                                sample.sampleEvaluatedMetricCount,
                                offlineMetricValuesAvailable ? offlineMetricValues.data() : nullptr,
                                offlineMetricValuesAvailable ? offlineMetricValues.size() : 0,
                                metricValue))
                        {
                            continue;
                        }
                        const TableValue metricValues[] = {
                            gpuUuid,
                            sample.endTimestampUtcNsec,
                            std::string_view(exportPlan.metricNamesForExport[metricIndex]),
                            metricValue,
                        };
                        if (!AppendRow(sink, TableIndex_GpuMetrics, metricValues))
                        {
                            succeeded = false;
                            return;
                        }
                    }
                });
                if (!succeeded)
                {
                    return false;
                }
            }
            return true;
        }

        bool ExportNvtxGpuProjectionSegments(TTableSink& sink)
        {
            std::string messageStorage;
            return ForEachProcess([&](const ProcessContext& context) {
                const Codec::ProcessData& processData = *context.pProcessData;
                const NvtxTraceData& nvtxTrace = processData.nvtxTrace;
                if (nvtxTrace.providerLuid == NVTRACE_NIL_LUID || nvtxTrace.domainCliques.empty())
                {
                    return true;
                }
                const CudaApiTraceData& cudaApiTrace = context.CudaApiTrace();
                NvtxProjectionStackCache stackCache;

                // graphNodeCorrelationId is UINT64_MAX for work that does not execute as a graph node.
                const auto fnAppendOperation = [&](const CudaApiTraceData::CuDeviceData& device,
                                                   const GpuOperationKey& key,
                                                   const CudaApiTraceData::DriverApiCallRef& driverApiCallRef,
                                                   uint64_t graphNodeCorrelationId) {
                    const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = nullptr;
                    if (graphNodeCorrelationId != UINT64_MAX)
                    {
                        pGraphNodeData = this->m_dataModel.FindCuGraphNode(cudaApiTrace, graphNodeCorrelationId);
                        if (!pGraphNodeData)
                        {
                            return true;
                        }
                    }
                    if (driverApiCallRef.uniqueThreadId != 0 && !HasDriverApiCall(cudaApiTrace, driverApiCallRef))
                    {
                        return true;
                    }
                    uint64_t driverApiCorrelationId = 0;
                    GetDriverApiCorrelationId(cudaApiTrace, driverApiCallRef, driverApiCorrelationId);

                    for (uint32_t domainCliqueIndex = 0; domainCliqueIndex < nvtxTrace.domainCliques.size(); ++domainCliqueIndex)
                    {
                        const auto [stackItr, inserted] = stackCache.try_emplace({driverApiCallRef.uniqueThreadId, driverApiCallRef.driverApiCallIdx, graphNodeCorrelationId, domainCliqueIndex});
                        std::vector<Codec::NvtxAssociationStackEntry>& stack = stackItr->second;
                        if (inserted)
                        {
                            this->m_dataModel.GetNvtxAssociationStack(processData, domainCliqueIndex, driverApiCallRef, pGraphNodeData, stack);
                        }

                        for (size_t stackIndex = 0; stackIndex < stack.size(); ++stackIndex)
                        {
                            const NvtxTraceData::PushPopRangeData& pushPopRange = *stack[stackIndex].pPushPopRange;
                            const TableValue values[] = {
                                context.nodeId,
                                context.serviceStartTsc,
                                context.processLuid,
                                device.gpuUuid,
                                device.cudaDeviceOrdinal,
                                pushPopRange.domainIndex,
                                pushPopRange.uniqueThreadId,
                                pushPopRange.rangeRef,
                                stack.front().pPushPopRange->rangeRef,
                                stackIndex ? TableValue(stack[stackIndex - 1].pPushPopRange->rangeRef) : TableValue(),
                                std::string_view(this->m_dataModel.GetNvtxTraceDomainName(nvtxTrace, pushPopRange.domainIndex)),
                                std::string_view(this->m_dataModel.GetNvtxTracePushPopRangeMessage(nvtxTrace, pushPopRange, messageStorage)),
                                key.operationSource,
                                key.uniqueThreadId,
                                driverApiCorrelationId,
                                key.kernelLaunchCorrelationId,
                                key.graphLaunchApiCorrelationId,
                                key.graphLaunchCugraphExecId,
                                key.graphLaunchHwCorrelationId,
                                key.graphLaunchDeviceTimestampGpuNsec,
                                key.graphNodeCorrelationId,
                                key.startGpuReportRef,
                                stack[stackIndex].associationSources,
                                static_cast<uint16_t>(stackIndex + 1),
                            };
                            if (!AppendRow(sink, TableIndex_NvtxGpuProjectionSegments, values))
                            {
                                return false;
                            }
                        }
                    }
                    return true;
                };

                for (const CudaApiTraceData::CuDeviceData& device : cudaApiTrace.devices)
                {
                    if (device.gpuUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }

                    for (const CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch : device.immediateKernelLaunches)
                    {
                        if (!GetExportedKernelLaunchPayload(immediateKernelLaunch))
                        {
                            continue;
                        }
                        GpuOperationKey key;
                        key.operationSource = "immediate_kernel_launch";
                        key.uniqueThreadId = immediateKernelLaunch.driverApiCallRef.uniqueThreadId;
                        key.kernelLaunchCorrelationId = immediateKernelLaunch.apiCorrelationId;
                        if (!fnAppendOperation(device, key, immediateKernelLaunch.driverApiCallRef, UINT64_MAX))
                        {
                            return false;
                        }
                    }

                    std::set<uint64_t> exportedStartGpuReportRefs;
                    for (const CudaApiTraceData::GpuCommand& gpuCommand : device.gpuCommands)
                    {
                        GpuCommandPayload payload;
                        if (!IsReportedGpuCommand(gpuCommand)
                            || !GetGpuCommandPayload(cudaApiTrace, gpuCommand, payload)
                            || !exportedStartGpuReportRefs.insert(payload.startGpuReportRef).second)
                        {
                            continue;
                        }
                        GpuOperationKey key;
                        key.operationSource = "gpu_command";
                        key.uniqueThreadId = gpuCommand.driverApiCallRef.uniqueThreadId;
                        key.startGpuReportRef = payload.startGpuReportRef;
                        if (!fnAppendOperation(device, key, gpuCommand.driverApiCallRef, gpuCommand.graphNodeCorrelationId))
                        {
                            return false;
                        }
                    }
                }

                // A graph launch's nodes can execute in a later snapshot, which has neither the launch nor the NVTX
                // stack of its API call, so every kernel node of the launched executable is projected with the launch.
                std::map<uint64_t, std::vector<uint64_t>> kernelGraphNodeCorrelationIdsByExecId;
                for (const auto& [graphNodeCorrelationId, graphNodeData] : cudaApiTrace.graphNodes)
                {
                    if (IsKernelGraphNode(cudaApiTrace, graphNodeCorrelationId))
                    {
                        kernelGraphNodeCorrelationIdsByExecId[graphNodeData.cugraphExecId].push_back(graphNodeCorrelationId);
                    }
                }
                for (const CudaApiTraceData::GraphLaunch& graphLaunch : cudaApiTrace.graphLaunches)
                {
                    GraphLaunchKey graphLaunchKey;
                    const bool isHostLaunch = graphLaunch.source == CudaApiTraceData::GraphLaunchSource::Host;
                    if (!GetGraphLaunchKey(graphLaunch, graphLaunchKey) || !IsValidGraphLaunchIdentity(graphLaunchKey.graphLaunchCorrelationId, graphLaunch.cugraphExecId, isHostLaunch))
                    {
                        continue;
                    }
                    const CudaApiTraceData::CuDeviceData* pDevice = FindCuDevice(cudaApiTrace, graphLaunchKey.cudaDeviceOrdinal);
                    const auto nodesItr = kernelGraphNodeCorrelationIdsByExecId.find(graphLaunch.cugraphExecId);
                    if (!pDevice || nodesItr == kernelGraphNodeCorrelationIdsByExecId.end())
                    {
                        continue;
                    }
                    for (const uint64_t graphNodeCorrelationId : nodesItr->second)
                    {
                        GpuOperationKey key;
                        key.operationSource = "graph_node_launch";
                        key.uniqueThreadId = graphLaunchKey.uniqueThreadId;
                        key.graphLaunchApiCorrelationId = graphLaunchKey.graphLaunchCorrelationId;
                        key.graphLaunchCugraphExecId = graphLaunch.cugraphExecId;
                        key.graphLaunchHwCorrelationId = graphLaunch.hwCorrelationId;
                        key.graphLaunchDeviceTimestampGpuNsec = graphLaunchKey.deviceLaunchTimestampGpuNsec;
                        key.graphNodeCorrelationId = graphNodeCorrelationId;
                        if (!fnAppendOperation(
                                *pDevice,
                                key,
                                isHostLaunch ? graphLaunch.payload.host.driverApiCallRef : CudaApiTraceData::DriverApiCallRef{},
                                graphNodeCorrelationId))
                        {
                            return false;
                        }
                    }
                }
                return true;
            });
        }

        std::vector<ProcessContext> m_processContexts;
    };

} // namespace nv::trace
