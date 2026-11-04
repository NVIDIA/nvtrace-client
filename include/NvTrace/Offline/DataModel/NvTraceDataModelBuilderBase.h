/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cpu_clock_calibration.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_clock_calibration.h>
#include <NvTrace/DataFormats/nvtrace_gpu_grctx_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_hw_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_pm_counters.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>
#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>
#include <NvTrace/Target/Consumer/NvTraceCodecEvents_generated.h>
#include <NvTrace/Target/Consumer/NvTraceProviderTypes.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceCpuCalibrationPointTypes.h>
#include <NvTrace/Target/NvTraceCpuClockCalibrationProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConstants.h>
#include <NvTrace/Target/NvTraceCudaApiTraceRecordUtilities.h>
#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>
#include <NvTrace/Target/NvTraceGpuClockCalibrationProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceGpuGrCtxTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceGpuHwTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceNvtxTraceRecordUtilities.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <regex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nv::trace::Codec {

    template <typename TTraceReader, typename TDerived>
    class DataModelBuilderBase
    {
    public:
        struct ProviderInfo
        {
            NVTRACE_LUID providerLuid = NVTRACE_NIL_LUID;
            NVTRACE_LUID parentLuid = NVTRACE_NIL_LUID;
            ProviderType providerType = ProviderType::Invalid;
            NVTRACE_UUID providerTypeUuid = NVTRACE_NIL_UUID;
        };

        struct Options
        {
            uint64_t observationStartUtcNsec = InvalidTimestamp;
            uint64_t observationEndUtcNsec = InvalidTimestamp;
            /// Compute the observed timestamp range even when both observation window bounds are specified.
            bool forceComputeObservedTimestampRange = false;
            bool enableGpuPmCountersObservationWindowExtension = false;
            /// Keep every CPU item (API calls, host callbacks, NVTX ranges, process and thread lifetimes) with a known
            /// timestamp, whatever the observation window, with its real timestamps rather than clamped ones.
            bool preserveOutOfWindowTimestamps = false;
            size_t maxNvtxDomainCompatibilityMatrixDomainCount = 256;
            /// An NVTX message is a human-scale string, and its length arrives off the wire, so this
            /// bounds what one corrupt length can make the model hold.
            uint32_t maxNvtxTraceMessageLength = 1u << 20;
            std::function<bool(const ProviderInfo&)> fnShouldIngestProvider;
        };

    public:
        ~DataModelBuilderBase() = default;

        DataModelBuilderBase(DataModel<TTraceReader>& dataModel)
            : m_dataModel(dataModel)
            , m_systemBuilderData(m_dataModel)
        {
        }

        DataModelBuilderBase(const DataModelBuilderBase&) = delete;
        DataModelBuilderBase& operator=(const DataModelBuilderBase&) = delete;

    protected:
        void BeginBuilding(TTraceReader& traceReader, const Options& options = Options{})
        {
            m_dataModel.Clear();
            m_dataModel.m_pTraceReader = &traceReader;

            m_options = options;
        }

        bool AddRecord(uint64_t recordLocation, NVTRACE_LUID providerLuid, uint32_t recordType, uint64_t correlationId, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            // Router record
            if (providerLuid == NVTRACE_NIL_LUID)
            {
                return AddRouterRecord(recordType, correlationId, pMajorPayload, maxMajorPayloadSize);
            }
            // Provider record
            else
            {
                auto itrProviderBuilderData = m_systemBuilderData.providers.find(providerLuid);
                NVTRACE_LOG_ERR_RETURN_IF(itrProviderBuilderData == m_systemBuilderData.providers.end(), false, 50, "Unknown providerLuid=%" PRIu64, providerLuid);

                return AddProviderRecord(*itrProviderBuilderData->second, recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
            }
        }

        bool CanRecordTypeEnableOrphanReplay(uint32_t recordType) const
        {
            return recordType == FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderCreated || recordType == FB::Codec::GpuPmCountersEvents::EventTypeId_AttachState;
        }

        template <typename TOrphanRecordMap>
        void FinalizeProviderOrphanRecords(TOrphanRecordMap& providerLuidToOrphanRecords) const
        {
            for (const auto& [providerLuid, orphanRecords] : providerLuidToOrphanRecords)
            {
                NVTRACE_LOG_WRN(50, "Dropping %zu orphan records for unknown providerLuid=%" PRIu64, orphanRecords.size(), providerLuid);
            }

            providerLuidToOrphanRecords.clear();
        }

        bool GetProviderInfoFromProviderCreated(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize, ProviderInfo& providerInfo) const
        {
            if (!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::ProviderCreated>())
            {
                return false;
            }

            const FB::Codec::NvTraceServiceEvents::ProviderCreated* const pFbProviderCreated = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::ProviderCreated>(pMajorPayload);
            providerInfo.providerLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbProviderCreated->providerLuid());
            providerInfo.parentLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbProviderCreated->parentLuid());
            providerInfo.providerTypeUuid = *reinterpret_cast<const NVTRACE_UUID*>(pFbProviderCreated->providerTypeUuid());
            providerInfo.providerType = GetProviderType(providerInfo.providerTypeUuid);

            return providerInfo.providerType != ProviderType::Invalid;
        }

        void SetObservationWindowUtc(uint64_t observationStartUtcNsec, uint64_t observationEndUtcNsec)
        {
            uint64_t observationStartTsc = MinTimestampTsc;
            uint64_t observationEndTsc = MaxTimestampTsc;
            m_dataModel.m_observationStartUtcNsec = observationStartUtcNsec;
            m_dataModel.m_observationEndUtcNsec = observationEndUtcNsec;
            const bool hasExplicitObservationWindowDuration = observationStartUtcNsec != InvalidTimestamp && observationEndUtcNsec != InvalidTimestamp && observationEndUtcNsec >= observationStartUtcNsec;
            const uint64_t requestedObservationDurationUtcNsec = hasExplicitObservationWindowDuration ? observationEndUtcNsec - observationStartUtcNsec : 0;

            if (observationStartUtcNsec != InvalidTimestamp)
            {
                const uint64_t convertedObservationStartTsc = m_dataModel.ConvertUtcNsecToTscTick(observationStartUtcNsec);
                if (convertedObservationStartTsc != InvalidTimestamp)
                {
                    observationStartTsc = convertedObservationStartTsc;
                    // Convert back to UTC to ensure consistency
                    const uint64_t convertedObservationStartUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(observationStartTsc);
                    if (convertedObservationStartUtcNsec != InvalidTimestamp)
                    {
                        m_dataModel.m_observationStartUtcNsec = convertedObservationStartUtcNsec;
                    }
                }
            }

            if (observationEndUtcNsec != InvalidTimestamp)
            {
                if (hasExplicitObservationWindowDuration && m_dataModel.m_observationStartUtcNsec != InvalidTimestamp)
                {
                    observationEndUtcNsec = m_dataModel.m_observationStartUtcNsec + requestedObservationDurationUtcNsec;
                }
                const uint64_t convertedObservationEndTsc = m_dataModel.ConvertUtcNsecToTscTick(observationEndUtcNsec);
                if (convertedObservationEndTsc != InvalidTimestamp)
                {
                    observationEndTsc = convertedObservationEndTsc;
                    // Convert back to UTC to ensure consistency
                    const uint64_t convertedObservationEndUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(observationEndTsc);
                    if (convertedObservationEndUtcNsec != InvalidTimestamp)
                    {
                        m_dataModel.m_observationEndUtcNsec = convertedObservationEndUtcNsec;
                    }
                }
            }

            m_dataModel.m_observationStartTsc = observationStartTsc;
            m_dataModel.m_observationEndTsc = observationEndTsc;
        }

        template <typename TProcessBuilderData>
        void FinalizeCudaGpuCommands(TProcessBuilderData& processBuilderData)
        {
            ProcessData& processData = processBuilderData.shadowed;
            auto& cudaApiTraceBuilderData = processBuilderData.cudaApiTrace;

            // Sort GPU timestamps by gpuTimestampRef (unique identifiers).
            std::sort(
                cudaApiTraceBuilderData.gpuTimestampReports.begin(),
                cudaApiTraceBuilderData.gpuTimestampReports.end(),
                typename CudaApiTraceBuilderData::GpuTimestampReportComparator());

            // Indexed like gpuTimestampReports
            std::vector<bool> isGpuTimestampReportMatched(cudaApiTraceBuilderData.gpuTimestampReports.size(), false);
            const auto fnFindGpuTimestampReport = [&](uint64_t gpuTimestampRef) -> const typename CudaApiTraceBuilderData::GpuTimestampReport* {
                const auto reportItr = std::lower_bound(
                    cudaApiTraceBuilderData.gpuTimestampReports.begin(),
                    cudaApiTraceBuilderData.gpuTimestampReports.end(),
                    gpuTimestampRef,
                    typename CudaApiTraceBuilderData::GpuTimestampReportComparator());
                if (reportItr == cudaApiTraceBuilderData.gpuTimestampReports.end() || reportItr->gpuTimestampRef != gpuTimestampRef)
                {
                    return nullptr;
                }
                isGpuTimestampReportMatched[reportItr - cudaApiTraceBuilderData.gpuTimestampReports.begin()] = true;
                return &*reportItr;
            };

            // Sort GpuCommands into their devices, and resolve their GPU timestamps.
            for (CudaApiTraceData::GpuCommand& gpuCommand : cudaApiTraceBuilderData.gpuCommands)
            {
                // Works on any event with fields custreamId, gpuTimestampRefStart, gpuTimestampRefEnd.
                auto handleGpuCommandGpuTimestamps = [&](auto* pCommandPayload) {
                    if (const auto* pStartReport = fnFindGpuTimestampReport(pCommandPayload->gpuTimestampRefStart))
                    {
                        gpuCommand.startTimestampGpuNsec = pStartReport->ptimer;
                    }
                    if (const auto* pEndReport = fnFindGpuTimestampReport(pCommandPayload->gpuTimestampRefEnd))
                    {
                        gpuCommand.endTimestampGpuNsec = pEndReport->ptimer;
                    }

                    auto streamItr = processData.cudaApiTrace.streams.find(pCommandPayload->custreamId);
                    if (streamItr == processData.cudaApiTrace.streams.end())
                    {
                        return;
                    }
                    CudaApiTraceData::CuStreamData& cuStream = streamItr->second;
                    if (processData.cudaApiTrace.devices.size() <= cuStream.cudaDeviceOrdinal)
                    {
                        return;
                    }
                    CudaApiTraceData::CuDeviceData& cuDevice = processData.cudaApiTrace.devices[cuStream.cudaDeviceOrdinal];
                    cuDevice.gpuCommands.push_back(gpuCommand);
                };

                switch (gpuCommand.commandType)
                {
                    case CudaApiTraceData::GpuCommandType::Invalid:
                        break;
                    case CudaApiTraceData::GpuCommandType::EagerMemcpy:
                    {
                        const auto* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(gpuCommand.recordLocation).pMajorPayload;
                        const auto* pMemcpyBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemcpyBegin*>(pMajorPayload);
                        handleGpuCommandGpuTimestamps(pMemcpyBeginPayload);
                        break;
                    }
                    case CudaApiTraceData::GpuCommandType::EagerMemset:
                    {
                        const auto* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(gpuCommand.recordLocation).pMajorPayload;
                        const auto* pMemsetBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemsetBegin*>(pMajorPayload);
                        handleGpuCommandGpuTimestamps(pMemsetBeginPayload);
                        break;
                    }
                    case CudaApiTraceData::GpuCommandType::GraphMemcpy:
                    {
                        const auto* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(gpuCommand.recordLocation).pMajorPayload;
                        const auto* pGraphMemcpyBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin*>(pMajorPayload);
                        handleGpuCommandGpuTimestamps(pGraphMemcpyBeginPayload);
                        break;
                    }
                    case CudaApiTraceData::GpuCommandType::GraphMemset:
                    {
                        const auto* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(gpuCommand.recordLocation).pMajorPayload;
                        const auto* pGraphMemsetBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphMemsetBegin*>(pMajorPayload);
                        handleGpuCommandGpuTimestamps(pGraphMemsetBeginPayload);
                        break;
                    }
                    case CudaApiTraceData::GpuCommandType::MemOp:
                    {
                        const auto* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(gpuCommand.recordLocation).pMajorPayload;
                        const auto* pMemOpBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemOpBegin*>(pMajorPayload);
                        handleGpuCommandGpuTimestamps(pMemOpBeginPayload);
                        break;
                    }
                    case CudaApiTraceData::GpuCommandType::HostCallback:
                        // Host callbacks are finalized separately via FinalizeHostCallbackGpuCommands (TSC → GPU nsec).
                        break;
                }
            }

            if (m_options.preserveOutOfWindowTimestamps)
            {
                for (size_t reportIndex = 0; reportIndex < cudaApiTraceBuilderData.gpuTimestampReports.size(); ++reportIndex)
                {
                    if (!isGpuTimestampReportMatched[reportIndex])
                    {
                        processData.cudaApiTrace.unmatchedGpuTimestampReports.push_back(cudaApiTraceBuilderData.gpuTimestampReports[reportIndex]);
                    }
                }
            }
        }

        template <typename TProcessBuilderData>
        void FinalizeHostCallbackGpuCommands(TProcessBuilderData& processBuilderData)
        {
            auto& cudaApiTraceBuilderData = processBuilderData.cudaApiTrace;
            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            std::unordered_map<uint64_t, uint64_t> graphLaunchCorrelationIdToStreamId;
            auto buildGraphLaunchCorrelationIdToStreamId = [&]() {
                for (const typename CudaApiTraceBuilderData::GraphLaunch& graphLaunch : cudaApiTraceBuilderData.graphLaunches)
                {
                    if (graphLaunch.apiCorrelationId != InvalidCorrelationId)
                    {
                        graphLaunchCorrelationIdToStreamId.emplace(graphLaunch.apiCorrelationId, graphLaunch.custreamId);
                    }
                }
            };

            auto findStreamWithDevice = [&](uint64_t custreamId) -> const CudaApiTraceData::CuStreamData* {
                const auto streamItr = cudaApiTraceData.streams.find(custreamId);
                if (streamItr == cudaApiTraceData.streams.end() || cudaApiTraceData.devices.size() <= streamItr->second.cudaDeviceOrdinal)
                {
                    return nullptr;
                }
                return &streamItr->second;
            };

            for (size_t hostCallbackInstanceIndex = 0; hostCallbackInstanceIndex < cudaApiTraceData.hostCallbackInstances.size(); ++hostCallbackInstanceIndex)
            {
                const CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = cudaApiTraceData.hostCallbackInstances[hostCallbackInstanceIndex];
                if (hostCallbackInstance.startTimestampTsc == InvalidTimestamp || hostCallbackInstance.endTimestampTsc == InvalidTimestamp)
                {
                    continue;
                }

                const CudaApiTraceData::CuStreamData* pCuStream = findStreamWithDevice(hostCallbackInstance.custreamId);
                if (!pCuStream && hostCallbackInstance.kind == CudaApiTraceData::HostCallbackKind::Graph)
                {
                    if (graphLaunchCorrelationIdToStreamId.empty())
                    {
                        buildGraphLaunchCorrelationIdToStreamId();
                    }
                    const auto graphLaunchItr = graphLaunchCorrelationIdToStreamId.find(hostCallbackInstance.enclosingCorrelationId);
                    if (graphLaunchItr != graphLaunchCorrelationIdToStreamId.end())
                    {
                        pCuStream = findStreamWithDevice(graphLaunchItr->second);
                    }
                }
                if (!pCuStream)
                {
                    continue;
                }
                CudaApiTraceData::CuDeviceData& cuDevice = cudaApiTraceData.devices[pCuStream->cudaDeviceOrdinal];

                const GpuPartitionData* pGpuPartitionData = m_dataModel.FindGpuPartitionByGpuUuid(cuDevice.gpuUuid);
                if (!pGpuPartitionData || pGpuPartitionData->gpuClockCalibration.calibrationPoints.empty())
                {
                    continue;
                }

                const uint64_t startTimestampGpuNsec = m_dataModel.ConvertTscTickToGpuNsec(*pGpuPartitionData, hostCallbackInstance.startTimestampTsc);
                const uint64_t endTimestampGpuNsec = m_dataModel.ConvertTscTickToGpuNsec(*pGpuPartitionData, hostCallbackInstance.endTimestampTsc);
                if (startTimestampGpuNsec == InvalidTimestamp || endTimestampGpuNsec == InvalidTimestamp)
                {
                    continue;
                }

                CudaApiTraceData::GpuCommand gpuCommand;
                gpuCommand.commandType = CudaApiTraceData::GpuCommandType::HostCallback;
                gpuCommand.startTimestampGpuNsec = startTimestampGpuNsec;
                gpuCommand.endTimestampGpuNsec = endTimestampGpuNsec;
                if (hostCallbackInstance.kind == CudaApiTraceData::HostCallbackKind::Graph)
                {
                    gpuCommand.graphNodeCorrelationId = hostCallbackInstance.graphNodeCorrelationId;
                }
                gpuCommand.driverApiCallRef = hostCallbackInstance.driverApiCallRef;
                gpuCommand.hostCallbackInstanceIndex = hostCallbackInstanceIndex;
                cuDevice.gpuCommands.push_back(gpuCommand);
            }
        }

        void SortCudaDeviceEventsByStartTimestamp()
        {
            auto sortByStartTimestamp = [](auto& events) {
                std::stable_sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
                    return lhs.startTimestampGpuNsec < rhs.startTimestampGpuNsec;
                });
            };

            for (auto& [_, processData] : m_dataModel.processes)
            {
                for (auto& cudaDeviceData : processData.cudaApiTrace.devices)
                {
                    sortByStartTimestamp(cudaDeviceData.immediateKernelLaunches);
                    sortByStartTimestamp(cudaDeviceData.graphNodeLaunches);
                    sortByStartTimestamp(cudaDeviceData.gpuCommands);
                }
            }
        }
        void EndBuilding()
        {
            SetObservationWindowUtc(m_options.observationStartUtcNsec, m_options.observationEndUtcNsec);

            // Assign cudaApiTrace min/max API timestamps.  These are used as approximate "process lifetime" timestamps.
            for (auto& [_, processBuilderData] : m_systemBuilderData.processes)
            {
                ProcessData& processData = processBuilderData.shadowed;

                // Bound inferred thread lifetimes by the process lifetime.
                for (auto& processThreadEventProviderEntry : processData.threadEventProviders)
                {
                    for (auto& threadEntry : processThreadEventProviderEntry.second.threads)
                    {
                        ThreadEventProviderData::ThreadData& threadData = threadEntry.second;
                        threadData.creationTimestampTsc = std::max(threadData.creationTimestampTsc, processData.creationTimestampTsc);
                        threadData.destructionTimestampTsc = std::min(threadData.destructionTimestampTsc, processData.destructionTimestampTsc);
                    }
                }

                FinalizeCudaApiTraceFunctions(processBuilderData.cudaApiTrace);
                FinalizeCudaGpuCommands(processBuilderData);
                FinalizeHostCallbackGpuCommands(processBuilderData);

                // Sort immediate kernel launches and graph nodes by { hwCorrelationId, startTimestamp }
                for (auto& cudaDeviceData : processData.cudaApiTrace.devices)
                {
                    // Sort immediate kernel launches by hwCorrelationId.
                    std::stable_sort(
                        cudaDeviceData.immediateKernelLaunches.begin(),
                        cudaDeviceData.immediateKernelLaunches.end(),
                        [](const CudaApiTraceData::ImmediateKernelLaunch& lhs, const CudaApiTraceData::ImmediateKernelLaunch& rhs) {
                            return lhs.GetHwCorrelationId() < rhs.GetHwCorrelationId();
                        });

                    // Set graph nodes creation/destruction timestamps to the creation/destruction timestamps of the graph exec.
                    for (auto* pGraphNodeData : cudaDeviceData.graphNodes)
                    {
                        auto itrGraphExecData = processData.cudaApiTrace.graphExecs.find(pGraphNodeData->cugraphExecId);
                        if (itrGraphExecData != processData.cudaApiTrace.graphExecs.end())
                        {
                            CudaApiTraceData::CuGraphExecData& graphExecData = itrGraphExecData->second;
                            if (graphExecData.creationTimestampTsc != InvalidTimestamp)
                            {
                                pGraphNodeData->creationTimestampTsc = graphExecData.creationTimestampTsc;
                            }
                            if (graphExecData.destructionTimestampTsc != InvalidTimestamp)
                            {
                                pGraphNodeData->destructionTimestampTsc = graphExecData.destructionTimestampTsc;
                            }
                        }
                    }

                    // Sort graph nodes by { hwCorrelationId, startTimestamp }.
                    std::stable_sort(
                        cudaDeviceData.graphNodes.begin(),
                        cudaDeviceData.graphNodes.end(),
                        [](const CudaApiTraceData::CuGraphNodeData* pLhs, const CudaApiTraceData::CuGraphNodeData* pRhs) {
                            const uint32_t lhsHwCorrelationId = pLhs->GetHwCorrelationId();
                            const uint32_t rhsHwCorrelationId = pRhs->GetHwCorrelationId();
                            if (lhsHwCorrelationId != rhsHwCorrelationId)
                            {
                                return lhsHwCorrelationId < rhsHwCorrelationId;
                            }
                            const uint64_t lhsStartTimestamp = pLhs->creationTimestampTsc;
                            const uint64_t rhsStartTimestamp = pRhs->creationTimestampTsc;
                            if (lhsStartTimestamp != rhsStartTimestamp)
                            {
                                return lhsStartTimestamp < rhsStartTimestamp;
                            }
                            return false;
                        });
                }

                std::stable_sort(
                    processBuilderData.cudaApiTrace.graphLaunches.begin(),
                    processBuilderData.cudaApiTrace.graphLaunches.end(),
                    [](const typename CudaApiTraceBuilderData::GraphLaunch& lhs, const typename CudaApiTraceBuilderData::GraphLaunch& rhs) {
                        return lhs.GetHwCorrelationId() < rhs.GetHwCorrelationId();
                    });

                FinalizeNvtxTraceBuilding(processBuilderData.nvtxTrace, processData.nvtxTrace);
                FinalizeNvtxTraceGraphNodeStacks(processBuilderData.nvtxTrace, processBuilderData.cudaApiTrace, processData.nvtxTrace);
            }

            for (auto& [gpuPartitionLuid, gpuPartitionBuilderData] : m_systemBuilderData.gpuPartitions)
            {
                // Reconstruct GrCtx residency for each GPU partition
                ReconstructGrCtxResidency(gpuPartitionBuilderData);
                // Iterate GPU HW trace events to assign start and end timestamps to kernel executions; the indices are in the order the hardware events occurred
                ReconstructGpuKernelLaunches(gpuPartitionBuilderData);
            }

            CorrelateGraphNodeLaunchesToGraphLaunches();

            m_systemBuilderData.Clear();

            RebuildObservationWindow();

            SortCudaDeviceEventsByStartTimestamp();

            for (auto& [_, processData] : m_dataModel.processes)
            {
                AssociateCudaWithNvtxTraceEvents(processData.cudaApiTrace, processData.nvtxTrace);
            }
        }

        // Cull a vector in place by applying fnShouldKeepItem to each item and compacting items when fnShouldKeepItem returns true.
        // TVector: vector-like container whose value_type is move-assignable.
        // fnShouldKeepItem: bool(TVector::value_type& item)
        template <typename TVector, typename TFnShouldKeepItem>
        void CullVector(TVector& vector, TFnShouldKeepItem&& fnShouldKeepItem, std::vector<size_t>* pIndexMappings = nullptr)
        {
            static constexpr size_t invalidIndex = SIZE_MAX;
            if (pIndexMappings)
            {
                pIndexMappings->assign(vector.size(), invalidIndex);
            }

            size_t visibleItemIndex = 0;
            for (size_t itemIndex = 0; itemIndex < vector.size(); ++itemIndex)
            {
                auto& item = vector[itemIndex];
                if (!fnShouldKeepItem(item))
                {
                    continue;
                }

                if (pIndexMappings)
                {
                    (*pIndexMappings)[itemIndex] = visibleItemIndex;
                }
                if (visibleItemIndex != itemIndex)
                {
                    vector[visibleItemIndex] = std::move(item);
                }
                ++visibleItemIndex;
            }

            vector.resize(visibleItemIndex);
        }

        bool RemapIndex(const std::vector<size_t>& indexMappings, size_t& index)
        {
            static constexpr size_t invalidIndex = SIZE_MAX;
            if (index >= indexMappings.size())
            {
                index = invalidIndex;
                return false;
            }

            index = indexMappings[index];
            return index != invalidIndex;
        }

        void ScanGpuPmCountersObservationWindow(const GpuPartitionData& gpuPartitionData)
        {
            const GpuPmCountersData& gpuPmCountersData = gpuPartitionData.gpuPmCounters;
            if (gpuPmCountersData.pmSampleRateHz == 0)
            {
                return;
            }

            const uint64_t samplePeriodNsec = 1'000'000'000 / gpuPmCountersData.pmSampleRateHz;
            for (const uint64_t sampleRecordLocation : gpuPmCountersData.sampleRecordLocations)
            {
                const GpuPmCountersSample sample = m_dataModel.GetGpuPmCountersSample(gpuPartitionData, sampleRecordLocation);
                if (!sample.pSampleRangeData || !sample.pSampleEvaluatedMetrics)
                {
                    continue;
                }

                const uint64_t endTimestampGpuNsec = sample.counterSample.endTimestampPtimer;
                const uint64_t startTimestampGpuNsec = endTimestampGpuNsec > samplePeriodNsec ? endTimestampGpuNsec - samplePeriodNsec : 1;
                const uint64_t startTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(gpuPartitionData, startTimestampGpuNsec);
                const uint64_t endTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(gpuPartitionData, endTimestampGpuNsec);
                m_dataModel.CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, true);
            }
        }

        void ScanGpuPartitionObservationWindow(const GpuPartitionData& gpuPartitionData)
        {
            if (m_options.enableGpuPmCountersObservationWindowExtension)
            {
                ScanGpuPmCountersObservationWindow(gpuPartitionData);
            }
        }

        void ScanCudaApiObservationWindow(const ProcessData& processData)
        {
            for (const auto& [_, threadData] : processData.cudaApiTrace.threads)
            {
                for (const auto& apiCall : threadData.runtimeApiCalls)
                {
                    m_dataModel.CheckCpuTimestampRange(apiCall.startTimestampTsc, apiCall.endTimestampTsc, true);
                }
                for (const auto& apiCall : threadData.driverApiCalls)
                {
                    m_dataModel.CheckCpuTimestampRange(apiCall.startTimestampTsc, apiCall.endTimestampTsc, true);
                }
            }

            for (const auto& hostCallbackInstance : processData.cudaApiTrace.hostCallbackInstances)
            {
                m_dataModel.CheckCpuTimestampRange(hostCallbackInstance.startTimestampTsc, hostCallbackInstance.endTimestampTsc, true);
            }

            for (const auto& cudaDeviceData : processData.cudaApiTrace.devices)
            {
                const GpuPartitionData* pGpuPartitionData = m_dataModel.FindGpuPartitionByGpuUuid(cudaDeviceData.gpuUuid);
                if (!pGpuPartitionData)
                {
                    continue;
                }

                for (const auto& immediateKernelLaunch : cudaDeviceData.immediateKernelLaunches)
                {
                    const uint64_t startTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(*pGpuPartitionData, immediateKernelLaunch.startTimestampGpuNsec);
                    const uint64_t endTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(*pGpuPartitionData, immediateKernelLaunch.endTimestampGpuNsec);
                    m_dataModel.CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, true);
                }
                for (const auto& graphNodeLaunch : cudaDeviceData.graphNodeLaunches)
                {
                    if (graphNodeLaunch.graphLaunchIdx == SIZE_MAX)
                    {
                        continue;
                    }
                    const uint64_t startTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(*pGpuPartitionData, graphNodeLaunch.startTimestampGpuNsec);
                    const uint64_t endTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(*pGpuPartitionData, graphNodeLaunch.endTimestampGpuNsec);
                    m_dataModel.CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, true);
                }
                for (const auto& gpuCommand : cudaDeviceData.gpuCommands)
                {
                    const uint64_t startTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(*pGpuPartitionData, gpuCommand.startTimestampGpuNsec);
                    const uint64_t endTimestampTsc = DataModel<TTraceReader>::ConvertGpuNsecToTscTick(*pGpuPartitionData, gpuCommand.endTimestampGpuNsec);
                    m_dataModel.CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, true);
                }
            }
        }

        void ScanNvtxTraceObservationWindow(const ProcessData& processData)
        {
            for (const NvtxTraceData::DomainCliqueData& domainCliqueData : processData.nvtxTrace.domainCliques)
            {
                for (const auto& [_, threadData] : domainCliqueData.threads)
                {
                    for (const auto& mark : threadData.marks)
                    {
                        m_dataModel.CheckCpuTimestamp(mark.timestampTsc, true);
                    }
                    for (const auto& pushPopRange : threadData.pushPopRanges)
                    {
                        uint64_t startTimestampTsc = pushPopRange.startTimestampTsc;
                        uint64_t endTimestampTsc = pushPopRange.endTimestampTsc;
                        if (!m_dataModel.CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, false, startTimestampTsc, endTimestampTsc))
                        {
                            continue;
                        }
                        if (!(pushPopRange.approximationMask & NvtxTraceData::FieldMask_StartTimestamp))
                        {
                            m_dataModel.RecordObservedTimestampTsc(startTimestampTsc);
                        }
                        if (!(pushPopRange.approximationMask & NvtxTraceData::FieldMask_EndTimestamp))
                        {
                            m_dataModel.RecordObservedTimestampTsc(endTimestampTsc);
                        }
                    }
                }
            }
        }

        void ScanProcessObservationWindow(const ProcessData& processData)
        {
            ScanCudaApiObservationWindow(processData);
            ScanNvtxTraceObservationWindow(processData);
        }

        void ScanObservationWindow()
        {
            for (const auto& [_, gpuPartitionData] : m_dataModel.gpuPartitions)
            {
                ScanGpuPartitionObservationWindow(gpuPartitionData);
            }

            for (const auto& [_, processData] : m_dataModel.processes)
            {
                ScanProcessObservationWindow(processData);
            }
        }

        static bool HasKnownTimestamp(uint64_t startTimestampTsc, uint64_t endTimestampTsc)
        {
            return startTimestampTsc != InvalidTimestamp || endTimestampTsc != InvalidTimestamp;
        }

        bool KeepCpuTimestampRange(uint64_t& startTimestampTsc, uint64_t& endTimestampTsc)
        {
            if (m_options.preserveOutOfWindowTimestamps)
            {
                return HasKnownTimestamp(startTimestampTsc, endTimestampTsc);
            }
            return m_dataModel.CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, false, startTimestampTsc, endTimestampTsc);
        }

        bool KeepProcessTimestampRange(const ProcessData& processData, uint64_t& startTimestampTsc, uint64_t& endTimestampTsc)
        {
            if (m_options.preserveOutOfWindowTimestamps)
            {
                return HasKnownTimestamp(startTimestampTsc, endTimestampTsc);
            }
            return m_dataModel.CheckProcessTimestampRange(processData, startTimestampTsc, endTimestampTsc, false, startTimestampTsc, endTimestampTsc);
        }

        bool KeepThreadTimestampRange(const ThreadEventProviderData::ThreadData& threadData, uint64_t& startTimestampTsc, uint64_t& endTimestampTsc)
        {
            if (m_options.preserveOutOfWindowTimestamps)
            {
                return HasKnownTimestamp(startTimestampTsc, endTimestampTsc);
            }
            return m_dataModel.CheckThreadTimestampRange(threadData, startTimestampTsc, endTimestampTsc, false, startTimestampTsc, endTimestampTsc);
        }

        bool KeepThreadTimestamp(const ThreadEventProviderData::ThreadData& threadData, uint64_t& timestampTsc)
        {
            if (m_options.preserveOutOfWindowTimestamps)
            {
                return timestampTsc != InvalidTimestamp;
            }
            return m_dataModel.CheckThreadTimestamp(threadData, timestampTsc, false, timestampTsc);
        }

        bool KeepGpuTimestampRange(const GpuPartitionData& gpuPartitionData, uint64_t& startTimestampGpuNsec, uint64_t& endTimestampGpuNsec)
        {
            if (m_options.preserveOutOfWindowTimestamps)
            {
                return true;
            }
            return startTimestampGpuNsec != InvalidTimestamp && endTimestampGpuNsec != InvalidTimestamp && m_dataModel.CheckGpuTimestampRange(gpuPartitionData, startTimestampGpuNsec, endTimestampGpuNsec, false, startTimestampGpuNsec, endTimestampGpuNsec);
        }

        template <typename TProviderDiagnostic>
        void CullProviderDiagnosticsToObservationWindow(std::vector<TProviderDiagnostic>& providerDiagnostics)
        {
            CullVector(providerDiagnostics, [&](auto& providerDiagnostic) {
                providerDiagnostic.timestampTsc = std::max(providerDiagnostic.timestampTsc, m_dataModel.m_observationStartTsc);
                return providerDiagnostic.timestampTsc < m_dataModel.m_observationEndTsc;
            });
        }

        void CullGpuPmCountersToObservationWindow(GpuPartitionData& gpuPartitionData)
        {
            GpuPmCountersData& gpuPmCountersData = gpuPartitionData.gpuPmCounters;
            CullVector(gpuPmCountersData.providerDiagnostics, [&](auto& providerDiagnostic) {
                switch (providerDiagnostic.recordType)
                {
                    case NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_InitializationFailure:
                    {
                        auto& initializationFailure = providerDiagnostic.payload.initializationFailure;
                        initializationFailure.timestampTsc = std::max(initializationFailure.timestampTsc, m_dataModel.m_observationStartTsc);
                        return initializationFailure.timestampTsc < m_dataModel.m_observationEndTsc;
                    }
                    case NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_UpdateTimeExceeded:
                    {
                        auto& updateTimeExceeded = providerDiagnostic.payload.updateTimeExceeded;
                        // Preserve the full duration for CPU usage even when the displayed interval is clipped.
                        if (updateTimeExceeded.updateWallTimeNsec == 0)
                        {
                            const uint64_t updateStartUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(updateTimeExceeded.updateStartTimestampTsc);
                            const uint64_t updateEndUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(updateTimeExceeded.updateEndTimestampTsc);
                            if (updateStartUtcNsec != InvalidTimestamp && updateEndUtcNsec != InvalidTimestamp && updateStartUtcNsec < updateEndUtcNsec)
                            {
                                updateTimeExceeded.updateWallTimeNsec = updateEndUtcNsec - updateStartUtcNsec;
                            }
                        }
                        updateTimeExceeded.updateStartTimestampTsc = std::max(updateTimeExceeded.updateStartTimestampTsc, m_dataModel.m_observationStartTsc);
                        updateTimeExceeded.updateEndTimestampTsc = std::min(updateTimeExceeded.updateEndTimestampTsc, m_dataModel.m_observationEndTsc);
                        return updateTimeExceeded.updateStartTimestampTsc < updateTimeExceeded.updateEndTimestampTsc;
                    }
                    case NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_SampleDiscontinuity:
                    {
                        auto& sampleDiscontinuity = providerDiagnostic.payload.sampleDiscontinuity;
                        sampleDiscontinuity.startTimestampTsc = std::max(sampleDiscontinuity.startTimestampTsc, m_dataModel.m_observationStartTsc);
                        sampleDiscontinuity.endTimestampTsc = std::min(sampleDiscontinuity.endTimestampTsc, m_dataModel.m_observationEndTsc);
                        return sampleDiscontinuity.startTimestampTsc < sampleDiscontinuity.endTimestampTsc;
                    }
                    default:
                        return false;
                }
            });
            CullVector(gpuPmCountersData.lifecycleEvents, [&](auto& lifecycleEvent) {
                return m_dataModel.CheckCpuTimestampRange(lifecycleEvent.timestampTsc, lifecycleEvent.endTimestampTsc, false, lifecycleEvent.timestampTsc, lifecycleEvent.endTimestampTsc) && (lifecycleEvent.endTimestampTsc > lifecycleEvent.timestampTsc || (lifecycleEvent.lifecycleEventType == NvTrace_GPU_Provider_LifecycleEventType_Resume && lifecycleEvent.resumePhase == NvTrace_GPU_Provider_ResumePhase_Succeeded));
            });

            if (gpuPmCountersData.pmSampleRateHz == 0)
            {
                gpuPmCountersData.sampleRecordLocations.clear();
                return;
            }

            const uint64_t samplePeriodNsec = 1'000'000'000 / gpuPmCountersData.pmSampleRateHz;
            CullVector(gpuPmCountersData.sampleRecordLocations, [&](uint64_t& sampleRecordLocation) {
                const GpuPmCountersSample sample = m_dataModel.GetGpuPmCountersSample(gpuPartitionData, sampleRecordLocation);
                if (!sample.pSampleRangeData || !sample.pSampleEvaluatedMetrics)
                {
                    return false;
                }

                const uint64_t endTimestampGpuNsec = sample.counterSample.endTimestampPtimer;
                const uint64_t startTimestampGpuNsec = endTimestampGpuNsec > samplePeriodNsec ? endTimestampGpuNsec - samplePeriodNsec : 1;
                // PM values summarize the complete sample period, so clipping the interval would change their meaning.
                if (m_options.preserveOutOfWindowTimestamps)
                {
                    // Assign PM sample by its end timestamp alone.
                    // This way, if a sample crosses a snapshot boundary, it will only appear in the snapshot that has the end.
                    if (endTimestampGpuNsec == InvalidTimestamp || endTimestampGpuNsec < gpuPartitionData.observationStartGpuNsec || endTimestampGpuNsec > gpuPartitionData.observationEndGpuNsec)
                    {
                        return false;
                    }
                }
                else
                {
                    if (endTimestampGpuNsec == InvalidTimestamp || startTimestampGpuNsec < gpuPartitionData.observationStartGpuNsec || endTimestampGpuNsec > gpuPartitionData.observationEndGpuNsec)
                    {
                        return false;
                    }
                }

                const uint64_t startTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartitionData, startTimestampGpuNsec);
                const uint64_t endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartitionData, endTimestampGpuNsec);
                return startTimestampUtcNsec != InvalidTimestamp && endTimestampUtcNsec != InvalidTimestamp;
            });
        }

        void CullGpuGrCtxTraceToObservationWindow(GpuPartitionData& gpuPartitionData, bool hasObservationStartGpuNsec, bool hasObservationEndGpuNsec)
        {
            GpuGrCtxTraceData& gpuGrCtxTraceData = gpuPartitionData.gpuGrCtxTrace;
            CullProviderDiagnosticsToObservationWindow(gpuGrCtxTraceData.providerDiagnostics);
            CullVector(gpuGrCtxTraceData.residentContexts, [&](auto& residentContext) {
                if (residentContext.ctxLoadTimestampGpuNsec == GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec && !hasObservationStartGpuNsec)
                {
                    return false;
                }

                if (residentContext.ctxSaveTimestampGpuNsec == GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec && !hasObservationEndGpuNsec)
                {
                    return false;
                }

                if (m_options.preserveOutOfWindowTimestamps)
                {
                    return true;
                }

                return m_dataModel.CheckGpuTimestampRange(gpuPartitionData, residentContext.ctxLoadTimestampGpuNsec, residentContext.ctxSaveTimestampGpuNsec, false, residentContext.ctxLoadTimestampGpuNsec, residentContext.ctxSaveTimestampGpuNsec);
            });
        }

        template <typename TCalibrationPoint>
        static void CullClockCalibrationToObservationWindow(std::vector<TCalibrationPoint>& calibrationPoints, uint64_t observationStartTsc, uint64_t observationEndTsc)
        {
            const bool hasObservationStartTsc = observationStartTsc != MinTimestampTsc;
            const bool hasObservationEndTsc = observationEndTsc != MaxTimestampTsc;
            if (calibrationPoints.empty() || (!hasObservationStartTsc && !hasObservationEndTsc))
            {
                return;
            }

            size_t beginIndex = 0;
            if (hasObservationStartTsc)
            {
                const auto firstPointAtOrAfterStart = std::lower_bound(
                    calibrationPoints.begin(),
                    calibrationPoints.end(),
                    observationStartTsc,
                    [](const TCalibrationPoint& calibrationPoint, uint64_t tscTick) {
                        return calibrationPoint.tscTick < tscTick;
                    });
                beginIndex = firstPointAtOrAfterStart != calibrationPoints.end()
                               ? static_cast<size_t>(firstPointAtOrAfterStart - calibrationPoints.begin())
                               : calibrationPoints.size() - 1;
            }

            size_t endIndex = calibrationPoints.size();
            if (hasObservationEndTsc)
            {
                const auto firstPointAtOrAfterEnd = std::lower_bound(
                    calibrationPoints.begin() + beginIndex,
                    calibrationPoints.end(),
                    observationEndTsc,
                    [](const TCalibrationPoint& calibrationPoint, uint64_t tscTick) {
                        return calibrationPoint.tscTick < tscTick;
                    });
                if (firstPointAtOrAfterEnd != calibrationPoints.end())
                {
                    endIndex = static_cast<size_t>(firstPointAtOrAfterEnd - calibrationPoints.begin() + 1);
                }
            }

            size_t writeIndex = 0;
            for (size_t readIndex = beginIndex; readIndex < endIndex; ++readIndex)
            {
                calibrationPoints[writeIndex++] = calibrationPoints[readIndex];
            }

            calibrationPoints.resize(writeIndex);
        }

        void CullGpuHwTraceToObservationWindow(GpuPartitionData& gpuPartitionData)
        {
            GpuHwTraceData& gpuHwTraceData = gpuPartitionData.gpuHwTrace;
            CullProviderDiagnosticsToObservationWindow(gpuHwTraceData.providerDiagnostics);
            CullVector(gpuHwTraceData.lifecycleEvents, [&](auto& lifecycleEvent) {
                return m_dataModel.CheckCpuTimestampRange(lifecycleEvent.timestampTsc, lifecycleEvent.endTimestampTsc, false, lifecycleEvent.timestampTsc, lifecycleEvent.endTimestampTsc) && (lifecycleEvent.endTimestampTsc > lifecycleEvent.timestampTsc || (lifecycleEvent.lifecycleEventType == NvTrace_GPU_Provider_LifecycleEventType_Resume && lifecycleEvent.resumePhase == NvTrace_GPU_Provider_ResumePhase_Succeeded));
            });

            CullVector(gpuHwTraceData.contextTrace, [&](auto& contextInterval) {
                return contextInterval.loadTimestampGpuNsec != InvalidTimestamp && contextInterval.saveTimestampGpuNsec != InvalidTimestamp && m_dataModel.CheckGpuTimestampRange(gpuPartitionData, contextInterval.loadTimestampGpuNsec, contextInterval.saveTimestampGpuNsec, false, contextInterval.loadTimestampGpuNsec, contextInterval.saveTimestampGpuNsec);
            });
        }

        void CullCudaApiToObservationWindow(ProcessData& processData)
        {
            struct ApiCallIndexMappings
            {
                std::vector<size_t> runtimeApiCallIndexMappings;
                std::vector<size_t> driverApiCallIndexMappings;
            };
            std::map<uint64_t, ApiCallIndexMappings> threadToApiCallIndexMappings;

            for (auto& [uniqueThreadId, threadData] : processData.cudaApiTrace.threads)
            {
                auto cullApiCall = [&](auto& apiCall) {
                    return KeepThreadTimestampRange(*threadData.pThreadData, apiCall.startTimestampTsc, apiCall.endTimestampTsc);
                };
                ApiCallIndexMappings& apiCallIndexMappings = threadToApiCallIndexMappings[uniqueThreadId];
                CullVector(threadData.runtimeApiCalls, cullApiCall, &apiCallIndexMappings.runtimeApiCallIndexMappings);
                CullVector(threadData.driverApiCalls, cullApiCall, &apiCallIndexMappings.driverApiCallIndexMappings);
            }

            for (auto& [_, streamData] : processData.cudaApiTrace.streams)
            {
                CullVector(streamData.apiCallReferences, [&](CudaApiTraceData::ApiCallRef& apiCallRef) {
                    const auto indexMappingsItr = threadToApiCallIndexMappings.find(apiCallRef.uniqueThreadId);
                    if (indexMappingsItr == threadToApiCallIndexMappings.end())
                    {
                        return false;
                    }

                    const ApiCallIndexMappings& apiCallIndexMappings = indexMappingsItr->second;
                    const bool hasVisibleRuntimeReference = RemapIndex(apiCallIndexMappings.runtimeApiCallIndexMappings, apiCallRef.runtimeApiCallIdx);
                    const bool hasVisibleDriverReference = RemapIndex(apiCallIndexMappings.driverApiCallIndexMappings, apiCallRef.driverApiCallIdx);
                    return hasVisibleRuntimeReference || hasVisibleDriverReference;
                });
            }

            for (auto& graphLaunch : processData.cudaApiTrace.graphLaunches)
            {
                if (graphLaunch.source != CudaApiTraceData::GraphLaunchSource::Host)
                {
                    continue;
                }

                const auto indexMappingsItr = threadToApiCallIndexMappings.find(graphLaunch.payload.host.driverApiCallRef.uniqueThreadId);
                if (indexMappingsItr != threadToApiCallIndexMappings.end())
                {
                    RemapIndex(indexMappingsItr->second.driverApiCallIndexMappings, graphLaunch.payload.host.driverApiCallRef.driverApiCallIdx);
                }
            }

            std::vector<size_t> hostCallbackInstanceIndexMappings;
            CullVector(processData.cudaApiTrace.hostCallbackInstances, [&](auto& hostCallbackInstance) {
                const auto threadItr = processData.cudaApiTrace.threads.find(hostCallbackInstance.executingUniqueThreadId);
                if (threadItr == processData.cudaApiTrace.threads.end() || !threadItr->second.pThreadData
                    || !KeepThreadTimestampRange(*threadItr->second.pThreadData, hostCallbackInstance.startTimestampTsc, hostCallbackInstance.endTimestampTsc))
                {
                    return false;
                }

                const auto indexMappingsItr = threadToApiCallIndexMappings.find(hostCallbackInstance.driverApiCallRef.uniqueThreadId);
                if (indexMappingsItr != threadToApiCallIndexMappings.end())
                {
                    RemapIndex(indexMappingsItr->second.driverApiCallIndexMappings, hostCallbackInstance.driverApiCallRef.driverApiCallIdx);
                }
                return true; }, &hostCallbackInstanceIndexMappings);

            for (auto& cudaDeviceData : processData.cudaApiTrace.devices)
            {
                GpuPartitionData* pGpuPartitionData = m_dataModel.FindGpuPartitionByGpuUuid(cudaDeviceData.gpuUuid);
                if (!pGpuPartitionData)
                {
                    cudaDeviceData.immediateKernelLaunches.clear();
                    cudaDeviceData.unmatchedImmediateKernelExecutions.clear();
                    cudaDeviceData.graphNodeLaunches.clear();
                    cudaDeviceData.unmatchedGraphNodeLaunchEnds.clear();
                    cudaDeviceData.unmatchedHostGraphLaunchTimestampsGpuNsec.clear();
                    cudaDeviceData.gpuCommands.clear();
                    continue;
                }

                CullVector(cudaDeviceData.immediateKernelLaunches, [&](auto& immediateKernelLaunch) {
                    if (!KeepGpuTimestampRange(*pGpuPartitionData, immediateKernelLaunch.startTimestampGpuNsec, immediateKernelLaunch.endTimestampGpuNsec))
                    {
                        return false;
                    }
                    const auto indexMappingsItr = threadToApiCallIndexMappings.find(immediateKernelLaunch.driverApiCallRef.uniqueThreadId);
                    if (indexMappingsItr != threadToApiCallIndexMappings.end())
                    {
                        RemapIndex(indexMappingsItr->second.driverApiCallIndexMappings, immediateKernelLaunch.driverApiCallRef.driverApiCallIdx);
                    }
                    return true;
                });
                CullVector(cudaDeviceData.graphNodeLaunches, [&](auto& graphNodeLaunch) {
                    return KeepGpuTimestampRange(*pGpuPartitionData, graphNodeLaunch.startTimestampGpuNsec, graphNodeLaunch.endTimestampGpuNsec);
                });
                CullVector(cudaDeviceData.gpuCommands, [&](auto& gpuCommand) {
                    if (gpuCommand.commandType == CudaApiTraceData::GpuCommandType::HostCallback)
                    {
                        RemapIndex(hostCallbackInstanceIndexMappings, gpuCommand.hostCallbackInstanceIndex);
                    }
                    if (!KeepGpuTimestampRange(*pGpuPartitionData, gpuCommand.startTimestampGpuNsec, gpuCommand.endTimestampGpuNsec))
                    {
                        return false;
                    }
                    const auto indexMappingsItr = threadToApiCallIndexMappings.find(gpuCommand.driverApiCallRef.uniqueThreadId);
                    if (indexMappingsItr != threadToApiCallIndexMappings.end())
                    {
                        RemapIndex(indexMappingsItr->second.driverApiCallIndexMappings, gpuCommand.driverApiCallRef.driverApiCallIdx);
                    }
                    return true;
                });
            }
        }

        void CullNvtxTraceToObservationWindow(ProcessData& processData)
        {
            const auto threadEventProviderItr = processData.threadEventProviders.find(processData.nvtxTrace.threadEventProviderLuid);
            if (threadEventProviderItr == processData.threadEventProviders.end())
            {
                NVTRACE_LOG_ERR(50, "NvtxTrace provider has no mapped ThreadEvent provider");
                return;
            }

            // {uniqueThreadId, domainCliqueIndex} -> pushPopRangeIndexMappings
            using ThreadDomainCliqueKey = std::pair<uint64_t, uint32_t>;
            std::map<ThreadDomainCliqueKey, std::vector<size_t>> pushPopRangeIndexMappings;
            for (uint32_t domainCliqueIndex = 0; domainCliqueIndex < processData.nvtxTrace.domainCliques.size(); ++domainCliqueIndex)
            {
                NvtxTraceData::DomainCliqueData& domainCliqueData = processData.nvtxTrace.domainCliques[domainCliqueIndex];
                auto threadItr = domainCliqueData.threads.begin();
                while (threadItr != domainCliqueData.threads.end())
                {
                    const ThreadDomainCliqueKey key = {threadItr->first, domainCliqueIndex};
                    const auto canonicalThreadItr = threadEventProviderItr->second.threads.find(threadItr->first);
                    const ThreadEventProviderData::ThreadData* const pCanonicalThreadData =
                        canonicalThreadItr != threadEventProviderItr->second.threads.end() ? &canonicalThreadItr->second : nullptr;
                    NvtxTraceData::DomainCliqueThreadData& threadData = threadItr->second;

                    auto cullPushPopRange = [&](auto& pushPopRange) {
                        if (pCanonicalThreadData)
                        {
                            if (KeepThreadTimestampRange(*pCanonicalThreadData, pushPopRange.startTimestampTsc, pushPopRange.endTimestampTsc))
                            {
                                pushPopRange.isObservationTimeRange = true;
                                threadData.hasObservationTimeRange = true;
                                return true;
                            }
                        }
                        return pushPopRange.isRecordTimeRange;
                    };

                    std::vector<size_t>& threadPushPopRangeIndexMappings = pushPopRangeIndexMappings[key];
                    CullVector(threadData.pushPopRanges, cullPushPopRange, &threadPushPopRangeIndexMappings);
                    if (!pCanonicalThreadData)
                    {
                        if (threadData.pushPopRanges.empty())
                        {
                            threadItr = domainCliqueData.threads.erase(threadItr);
                            continue;
                        }
                        threadData.marks.clear();
                        threadData.droppedRangePushes.clear();
                        threadData.droppedRangePops.clear();
                    }
                    else
                    {
                        auto cullTimestampedEvent = [&](auto& event) {
                            return KeepThreadTimestamp(*pCanonicalThreadData, event.timestampTsc);
                        };
                        CullVector(threadData.marks, cullTimestampedEvent);
                        CullVector(threadData.droppedRangePushes, cullTimestampedEvent);
                        CullVector(threadData.droppedRangePops, cullTimestampedEvent);
                    }

                    for (auto& pushPopRange : threadData.pushPopRanges)
                    {
                        if (!RemapIndex(threadPushPopRangeIndexMappings, pushPopRange.parentRangeIndex))
                        {
                            pushPopRange.parentRangeIndex = NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex;
                        }
                    }

                    ++threadItr;
                }
            }

            // Remap graphNode NVTX references
            for (auto& [_, graphNodeData] : processData.cudaApiTrace.graphNodes)
            {
                auto rangeItr = graphNodeData.nvtxTraceDomainCliquePushPopRanges.begin();
                while (rangeItr != graphNodeData.nvtxTraceDomainCliquePushPopRanges.end())
                {
                    const auto indexMappingsItr = pushPopRangeIndexMappings.find({graphNodeData.uniqueThreadId, rangeItr->first});
                    if (indexMappingsItr == pushPopRangeIndexMappings.end() || !RemapIndex(indexMappingsItr->second, rangeItr->second))
                    {
                        rangeItr = graphNodeData.nvtxTraceDomainCliquePushPopRanges.erase(rangeItr);
                        continue;
                    }
                    ++rangeItr;
                }
            }
        }

        bool CullProcessToObservationWindow(ProcessData& processData)
        {
            if (!KeepCpuTimestampRange(processData.creationTimestampTsc, processData.destructionTimestampTsc))
            {
                return false;
            }

            for (auto& [threadEventProviderLuid, threadEventProviderData] : processData.threadEventProviders)
            {
                auto threadItr = threadEventProviderData.threads.begin();
                while (threadItr != threadEventProviderData.threads.end())
                {
                    ThreadEventProviderData::ThreadData& threadData = threadItr->second;
                    if (!KeepProcessTimestampRange(processData, threadData.creationTimestampTsc, threadData.destructionTimestampTsc))
                    {
                        const uint64_t uniqueThreadId = threadItr->first;
                        if (processData.cudaApiTrace.threadEventProviderLuid == threadEventProviderLuid)
                        {
                            processData.cudaApiTrace.threads.erase(uniqueThreadId);
                        }
                        threadItr = threadEventProviderData.threads.erase(threadItr);
                        continue;
                    }

                    ++threadItr;
                }
            }

            CullCudaApiToObservationWindow(processData);
            CullNvtxTraceToObservationWindow(processData);
            return true;
        }

        void CullToObservationWindow()
        {
            const bool hasObservationEndTsc = m_dataModel.m_observationEndTsc != MaxTimestampTsc;

            auto gpuPartitionItr = m_dataModel.gpuPartitions.begin();
            while (gpuPartitionItr != m_dataModel.gpuPartitions.end())
            {
                GpuPartitionData& gpuPartitionData = gpuPartitionItr->second;
                if (gpuPartitionData.destructionTimestampTsc == MaxTimestampTsc && !hasObservationEndTsc)
                {
                    gpuPartitionItr = m_dataModel.gpuPartitions.erase(gpuPartitionItr);
                    continue;
                }
                if (!m_dataModel.CheckCpuTimestampRange(gpuPartitionData.creationTimestampTsc, gpuPartitionData.destructionTimestampTsc, false, gpuPartitionData.creationTimestampTsc, gpuPartitionData.destructionTimestampTsc))
                {
                    gpuPartitionItr = m_dataModel.gpuPartitions.erase(gpuPartitionItr);
                    continue;
                }

                bool hasObservationStartGpuNsec = false;
                gpuPartitionData.observationStartGpuNsec = GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec;
                if (gpuPartitionData.creationTimestampTsc != InvalidTimestamp)
                {
                    const uint64_t observationStartGpuNsec = m_dataModel.ConvertTscTickToGpuNsec(gpuPartitionData, gpuPartitionData.creationTimestampTsc);
                    if (observationStartGpuNsec != InvalidTimestamp)
                    {
                        hasObservationStartGpuNsec = true;
                        gpuPartitionData.observationStartGpuNsec = observationStartGpuNsec;
                    }
                }

                bool hasObservationEndGpuNsec = false;
                gpuPartitionData.observationEndGpuNsec = GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec;
                if (gpuPartitionData.destructionTimestampTsc != MaxTimestampTsc)
                {
                    const uint64_t observationEndGpuNsec = m_dataModel.ConvertTscTickToGpuNsec(gpuPartitionData, gpuPartitionData.destructionTimestampTsc);
                    if (observationEndGpuNsec != InvalidTimestamp)
                    {
                        hasObservationEndGpuNsec = true;
                        gpuPartitionData.observationEndGpuNsec = observationEndGpuNsec;
                    }
                }

                CullGpuGrCtxTraceToObservationWindow(gpuPartitionData, hasObservationStartGpuNsec, hasObservationEndGpuNsec);
                CullGpuPmCountersToObservationWindow(gpuPartitionData);
                CullGpuHwTraceToObservationWindow(gpuPartitionData);

                CullClockCalibrationToObservationWindow(gpuPartitionData.gpuClockCalibration.calibrationPoints, gpuPartitionData.creationTimestampTsc, gpuPartitionData.destructionTimestampTsc);

                ++gpuPartitionItr;
            }

            auto processItr = m_dataModel.processes.begin();
            while (processItr != m_dataModel.processes.end())
            {
                ProcessData& processData = processItr->second;
                if (!CullProcessToObservationWindow(processData))
                {
                    processItr = m_dataModel.processes.erase(processItr);
                    continue;
                }

                ++processItr;
            }

            CullClockCalibrationToObservationWindow(m_dataModel.cpuClockCalibration.calibrationPoints, m_dataModel.m_observationStartTsc, m_dataModel.m_observationEndTsc);
        }

        void RebuildObservationWindow()
        {
            m_dataModel.m_timestampStartTsc = MaxTimestampTsc;
            m_dataModel.m_timestampEndTsc = MinTimestampTsc;

            if (m_options.forceComputeObservedTimestampRange
                || m_options.observationStartUtcNsec == InvalidTimestamp
                || m_options.observationEndUtcNsec == InvalidTimestamp)
            {
                ScanObservationWindow();
            }

            if (m_options.observationStartUtcNsec == InvalidTimestamp && m_dataModel.m_timestampStartTsc != MaxTimestampTsc)
            {
                m_dataModel.m_observationStartTsc = m_dataModel.m_timestampStartTsc;
                m_dataModel.m_observationStartUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(m_dataModel.m_timestampStartTsc);
            }
            if (m_options.observationEndUtcNsec == InvalidTimestamp && m_dataModel.m_timestampEndTsc != MinTimestampTsc)
            {
                m_dataModel.m_observationEndTsc = m_dataModel.m_timestampEndTsc;
                m_dataModel.m_observationEndUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(m_dataModel.m_timestampEndTsc);
            }

            CullToObservationWindow();
        }

    protected:
        enum class ObjectType
        {
            CudaContext,
            CudaStream,
            CudaFunction,
            CudaModule,
            CudaGraphNode,
            CudaGraphNodeCorrelation,
            CudaGraphExec,
            NvtxTraceStringTableEntry,
            NvtxTraceDomain,
            NvtxTracePushPopRange,
            NvtxTraceStackGroup,
            ThreadEventThread,
            ThreadEventProviderMapping
        };

        struct SystemBuilderData;
        struct GpuPartitionBuilderData;
        struct ProcessBuilderData;

        struct ProviderBuilderData
        {
            ProviderInfo providerInfo;

            bool invalid = false;

            ProviderBuilderData(ProviderType providerType_)
                : providerInfo{NVTRACE_NIL_LUID, NVTRACE_NIL_LUID, providerType_}
            {
            }
        };

        struct CpuClockCalibrationBuilderData : ProviderBuilderData
        {
            CpuClockCalibrationData& shadowed;

            using RecordHandler = bool (DataModelBuilderBase::*)(CpuClockCalibrationBuilderData& cpuClockCalibrationBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize);
            RecordHandler recordHandlers[NvTrace_CPU_ClockCalibration_TscRtc_RecordType_Count] = {};

            CpuClockCalibrationBuilderData(CpuClockCalibrationData& shadowed_)
                : ProviderBuilderData(ProviderType::CpuClockCalibration)
                , shadowed(shadowed_)
            {
            }
        };

        struct GpuClockCalibrationBuilderData : ProviderBuilderData
        {
            GpuClockCalibrationData& shadowed;

            using RecordHandler = bool (DataModelBuilderBase::*)(GpuClockCalibrationBuilderData& gpuClockCalibrationBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize);
            RecordHandler recordHandlers[NvTrace_GPU_ClockCalibration_TscGpu_RecordType_Count] = {};

            GpuClockCalibrationBuilderData(GpuClockCalibrationData& shadowed_)
                : ProviderBuilderData(ProviderType::GpuClockCalibration)
                , shadowed(shadowed_)
            {
            }
        };

        struct GpuPmCountersBuilderData : ProviderBuilderData
        {
            GpuPmCountersData& shadowed;
            const std::string& chipName;
            NvTrace_GPU_PmCounters_RecordVersionInfo recordVersionInfo = {};

            using RecordHandler = bool (DataModelBuilderBase::*)(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize);
            RecordHandler recordHandlers[NvTrace_GPU_PmCounters_RecordType_Count] = {};

            GpuPmCountersBuilderData(GpuPmCountersData& shadowed_, const std::string& chipName_)
                : ProviderBuilderData(ProviderType::GpuPmCounters)
                , shadowed(shadowed_)
                , chipName(chipName_)
            {
            }
        };

        struct GpuHwTraceBuilderData : ProviderBuilderData
        {
            GpuHwTraceData& shadowed;
            NvTrace_GPU_HwTrace_RecordVersionInfo recordVersionInfo = {};

            using RecordHandler = bool (DataModelBuilderBase::*)(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize);
            RecordHandler recordHandlers[NvTrace_GPU_HwTrace_RecordType_Count] = {};

            struct HwRecord
            {
                uint64_t recordLocation;
                uint32_t recordType;
            };
            // Record location + record type for each consumed HW-trace record, replayed by the reconstruction pass.
            std::vector<HwRecord> hwRecords;

            GpuHwTraceBuilderData(GpuHwTraceData& shadowed_)
                : ProviderBuilderData(ProviderType::GpuHwTrace)
                , shadowed(shadowed_)
            {
            }
        };

        struct GpuGrCtxTraceBuilderData : ProviderBuilderData
        {
            GpuGrCtxTraceData& shadowed;
            NvTrace_GPU_GrCtxTrace_RecordVersionInfo recordVersionInfo = {};

            using RecordHandler = bool (DataModelBuilderBase::*)(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize);
            RecordHandler recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_Count] = {};

            struct GrCtxRecord
            {
                uint64_t recordLocation;
                NvTrace_GPU_GrCtxTrace_RecordType recordType;
                // The record's correlation ID, which is its sequence number in the provider's buffer
                uint64_t sequenceNumber;
            };
            // Record location, record type, and sequence number for each consumed context-change record, replayed by the reconstruction pass.
            std::vector<GrCtxRecord> grCtxRecords;

            GpuGrCtxTraceBuilderData(GpuGrCtxTraceData& shadowed_)
                : ProviderBuilderData(ProviderType::GpuGrCtxTrace)
                , shadowed(shadowed_)
            {
            }
        };

        struct CudaApiTraceBuilderData : ProviderBuilderData
        {
            CudaApiTraceData& shadowed;
            ThreadEventProviderData* pThreadEventProviderData = nullptr;
            NVTRACE_LUID nvtxTraceProviderLuid = NVTRACE_NIL_LUID;
            std::vector<NvTrace_RevisionedRef> pendingNvtxTraceStackGroupRefs;
            std::map<uint64_t, NvTrace_RevisionedRef> graphNodeCorrelationIdToNvtxTraceStackGroupRef;

            struct ThreadBuilderData
            {
                CudaApiTraceData::ThreadData& shadowed;

                uint64_t currentDriverApiCorrelationId = InvalidCorrelationId;
                uint64_t currentGraphLaunchCorrelationId = InvalidCorrelationId;
                uint64_t currentMemCommandCorrelationId = InvalidCorrelationId;
                uint64_t pendingMemCommandRecordLocation = InvalidRecordLocation;

                // Version A GraphHostCallbackExecEnd records lack enclosingCorrelationId, so their
                // matching ExecBegin instance must be retained per execution thread.
                size_t currentGraphHostCallbackInstanceIndex = SIZE_MAX;
                uint64_t currentGraphHostCallbackNodeCorrelationId = InvalidCorrelationId;

                ThreadBuilderData(CudaApiTraceData::ThreadData& shadowed_)
                    : shadowed(shadowed_)
                {
                }
            };
            std::map<uint64_t, ThreadBuilderData> threads;

            // A function's FunctionCreated and FunctionNameSegment records may arrive in any order and
            // repeat, so the name is whole only once the segments tile totalNameLength.
            struct FunctionNameBuilderData
            {
                bool totalKnown = false;
                uint32_t totalNameLength = 0;
                std::map<uint32_t /*nameOffset*/, uint32_t /*nameSegmentLength*/> segments;
            };
            std::unordered_map<uint64_t /*cufunctionId*/, FunctionNameBuilderData> functionNames;

            using RecordHandler = bool (DataModelBuilderBase::*)(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize);
            struct RecordHandlerEntry
            {
                RecordHandler handler = nullptr;
                bool isThreadEvent = false;
            };
            RecordHandlerEntry recordHandlers[kCudaApiTraceRecordTypeCount] = {};

            using GpuTimestampReport = CudaApiTraceData::GpuTimestampReport;
            struct GpuTimestampReportComparator
            {
                bool operator()(const GpuTimestampReport& lhs, const GpuTimestampReport& rhs) const
                {
                    return lhs.gpuTimestampRef < rhs.gpuTimestampRef;
                }
                bool operator()(const GpuTimestampReport& lhs, uint64_t gpuTimestampRef) const
                {
                    return lhs.gpuTimestampRef < gpuTimestampRef;
                }
                bool operator()(uint64_t gpuTimestampRef, const GpuTimestampReport& rhs) const
                {
                    return gpuTimestampRef < rhs.gpuTimestampRef;
                }
            };
            std::vector<GpuTimestampReport> gpuTimestampReports;
            // GpuCommands before they are sorted into devices (lookup by stream).
            std::vector<CudaApiTraceData::GpuCommand> gpuCommands;

            struct GraphLaunch
            {
                uint64_t apiCorrelationId = 0;
                uint64_t cugraphExecId = 0;
                uint64_t custreamId = 0;
                uint64_t startTimestampTsc = InvalidTimestamp;
                uint64_t recordLocation = InvalidRecordLocation;
                uint64_t gridNullEndRecordLocation = InvalidRecordLocation;
                CudaApiTraceData::ApiCallRef apiCallRef;

                uint32_t GetHwCorrelationId() const
                {
                    return (static_cast<uint32_t>(apiCorrelationId) & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK) | NVTRACE_CUDA_HW_CORRELATION_ID_GRAPH_LAUNCH_HOST_MSBS;
                }
            };

            // Eager and graph host callback correlation key spaces are independent. A graph node executes once
            // per graph launch, so graph callback instances need both identities to preserve repeated launches.
            std::map<uint64_t, size_t> launchApiCorrelationIdToHostCallbackInstanceIndex;
            using GraphHostCallbackInstanceKey = std::pair<uint64_t /* graphNodeCorrelationId */, uint64_t /* enclosingCorrelationId */>;
            std::map<GraphHostCallbackInstanceKey, size_t> graphHostCallbackInstanceKeyToInstanceIndex;

            struct GraphLaunchDemarcator
            {
                CudaApiTraceData::GraphLaunch graphLaunch;
                uint64_t timestampGpuNsec = InvalidTimestamp;
                uint64_t timestampTsc = InvalidTimestamp;
                uint64_t cudaDeviceOrdinal = UINT64_MAX;
                uint64_t gridNullEndRecordLocation = InvalidRecordLocation;
                size_t graphLaunchIdx = SIZE_MAX;
            };
            // Host graph launches before they are matched to GridNullEnd records. Sorted by 32-bit host graph-launch HW correlation ID.
            std::vector<GraphLaunch> graphLaunches;
            std::multimap<uint32_t, uint64_t> graphExecIdsByDeviceGraphLaunchHwCorrelationId;
            std::map<uint64_t, std::vector<GraphLaunchDemarcator>> graphLaunchDemarcatorsByGraphExecId;

            CudaApiTraceBuilderData(CudaApiTraceData& shadowed_)
                : ProviderBuilderData(ProviderType::CudaApiTrace)
                , shadowed(shadowed_)
            {
            }

            ThreadBuilderData& EnsureThreadData(uint64_t uniqueThreadId)
            {
                auto itrThreadBuilderData = threads.lower_bound(uniqueThreadId);
                if (itrThreadBuilderData == threads.end() || itrThreadBuilderData->first != uniqueThreadId)
                {
                    ThreadEventProviderData::ThreadData& threadEventData = pThreadEventProviderData->EnsureThread(uniqueThreadId);
                    CudaApiTraceData::ThreadData& threadData = shadowed.threads.try_emplace(uniqueThreadId).first->second;
                    threadData.pThreadData = &threadEventData;

                    itrThreadBuilderData = threads.try_emplace(itrThreadBuilderData, uniqueThreadId, threadData);
                }

                return itrThreadBuilderData->second;
            }

            CudaApiTraceData::CuContextData& EnsureCuContextData(uint64_t cucontextId)
            {
                auto [itrContextData, inserted] = shadowed.contexts.try_emplace(cucontextId);
                if (inserted)
                {
                    itrContextData->second.cucontextId = cucontextId;
                }

                return itrContextData->second;
            }

            CudaApiTraceData::CuStreamData& EnsureCuStreamData(uint64_t custreamId)
            {
                auto [itrStreamData, inserted] = shadowed.streams.try_emplace(custreamId);
                if (inserted)
                {
                    itrStreamData->second.custreamId = custreamId;
                }

                return itrStreamData->second;
            }

            CudaApiTraceData::CuModuleData& EnsureCuModuleData(uint64_t cumoduleId)
            {
                auto [itrModuleData, inserted] = shadowed.modules.try_emplace(cumoduleId);
                if (inserted)
                {
                    itrModuleData->second.cumoduleId = cumoduleId;
                }

                return itrModuleData->second;
            }

            CudaApiTraceData::CuFunctionData& EnsureCuFunctionData(uint64_t cufunctionId)
            {
                auto [itrFunctionData, inserted] = shadowed.functions.try_emplace(cufunctionId);
                if (inserted)
                {
                    itrFunctionData->second.cufunctionId = cufunctionId;
                }

                return itrFunctionData->second;
            }

            // Above this bound a cudaDeviceOrdinal is treated as corrupt wire data rather than a real device,
            // so devices never resizes on an attacker- or bug-controlled value.
            static constexpr uint32_t kMaxCudaDeviceOrdinal = 1024;

            CudaApiTraceData::CuDeviceData* EnsureCuDeviceData(uint32_t cudaDeviceOrdinal)
            {
                if (kMaxCudaDeviceOrdinal <= cudaDeviceOrdinal)
                {
                    NVTRACE_LOG_ERR(50, "CudaApiTrace cudaDeviceOrdinal=%u exceeds the maximum of %u", cudaDeviceOrdinal, kMaxCudaDeviceOrdinal);
                    return nullptr;
                }

                if (shadowed.devices.size() <= cudaDeviceOrdinal)
                {
                    shadowed.devices.resize(cudaDeviceOrdinal + 1);
                }

                return &shadowed.devices[cudaDeviceOrdinal];
            }

            CudaApiTraceData::CuGraphExecData& EnsureCuGraphExecData(uint64_t cugraphExecId)
            {
                auto [itrGraphExecData, inserted] = shadowed.graphExecs.try_emplace(cugraphExecId);
                if (inserted)
                {
                    itrGraphExecData->second.cugraphExecId = cugraphExecId;
                }

                return itrGraphExecData->second;
            }

            CudaApiTraceData::CuGraphNodeData& EnsureCuGraphNodeData(uint64_t graphNodeCorrelationId)
            {
                auto [itrGraphNodeData, inserted] = shadowed.graphNodes.try_emplace(graphNodeCorrelationId);
                if (inserted)
                {
                    itrGraphNodeData->second.graphNodeCorrelationId = graphNodeCorrelationId;
                }

                return itrGraphNodeData->second;
            }
        };

        struct NvtxTraceBuilderData : ProviderBuilderData
        {
            NvtxTraceData& shadowed;
            ThreadEventProviderData* pThreadEventProviderData = nullptr;

            struct StringTableEntry
            {
                uint64_t stringRegisteredRecordLocation = InvalidRecordLocation;
                std::map<uint32_t /*stringOffset*/, uint64_t /*recordLocation*/> segmentRecordLocations;
            };

            struct Domain
            {
                uint64_t recordLocation = InvalidRecordLocation;
                uint64_t nameStringHandle = 0;
            };

            enum FieldMask : uint8_t
            {
                FieldMask_None = 0,
                FieldMask_StartTimestamp = 1 << 0,
                FieldMask_EndTimestamp = 1 << 1,
                FieldMask_NestingDepth = 1 << 2,
            };

            struct PushPopRange
            {
                static constexpr size_t InvalidPushPopRangeIndex = ~size_t(0);
                static constexpr uint32_t InvalidDomainIndex = UINT32_MAX;

                uint64_t rangeRef = 0;
                uint64_t recordLocation = InvalidRecordLocation;
                uint64_t startTimestampTsc = InvalidTimestamp;
                uint64_t endTimestampTsc = InvalidTimestamp;
                uint64_t uniqueThreadId = 0;
                uint32_t domainIndex = InvalidDomainIndex;
                uint16_t nestingDepth = 0;
                uint16_t messageType = 0;
                size_t ancestorIndex = InvalidPushPopRangeIndex;
                uint8_t approximationMask = FieldMask_None;
                bool isSynthetic = false;

                /// The message in the encoding messageType names, assembled as its records arrive.
                std::string message;

                size_t finalDomainCliqueIndex = SIZE_MAX;
                size_t finalPushPopRangeIndex = NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex;
            };

            struct StackRef
            {
                size_t topPushPopRangeIndex = PushPopRange::InvalidPushPopRangeIndex;
                uint16_t nestingDepth = 0;
            };

            struct StackGroup
            {
                std::map<uint32_t, size_t> domainIndexToStackRefIndex;
            };

            struct Mark
            {
                uint64_t correlationId = 0;
                uint64_t recordLocation = InvalidRecordLocation;
                uint64_t timestampTsc = InvalidTimestamp;
                uint64_t uniqueThreadId = 0;
                uint32_t domainIndex = 0;
                uint16_t messageType = 0;
                uint8_t hasEventAttributes = 0;

                /// The message in the encoding messageType names, assembled as its records arrive.
                std::string message;
            };

            struct DroppedRange
            {
                uint64_t timestampTsc = InvalidTimestamp;
                uint64_t uniqueThreadId = 0;
                uint32_t domainIndex = 0;
            };

            struct ThreadBucket
            {
                std::vector<std::vector<size_t>> pushPopRangeIndicesAtDepth;
            };
            using DomainThreadBuckets = std::map<uint32_t, std::map<uint64_t, ThreadBucket>>;
            using DomainThreadRangePreorder = std::map<uint32_t, std::map<uint64_t, std::vector<size_t>>>;

            std::map<uint64_t, StringTableEntry> stringTableEntries;
            std::map<uint32_t, Domain> domains;
            std::vector<PushPopRange> pushPopRanges;
            std::map<NvTrace_RevisionedRef, size_t> pushPopRangeRefToIndex;
            std::vector<StackGroup> stackGroups;
            std::map<NvTrace_RevisionedRef, size_t> stackGroupRefToIndex;
            std::vector<StackRef> stackRefs;
            std::vector<Mark> marks;
            std::map<NvTrace_RevisionedRef, size_t> markRefToMarkIndex;
            std::vector<DroppedRange> droppedRangePushes;
            std::vector<DroppedRange> droppedRangePops;

            DomainThreadBuckets domainThreadBuckets;
            DomainThreadRangePreorder domainThreadRangePreorder;

            using RecordHandler = bool (DataModelBuilderBase::*)(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize);
            struct RecordHandlerEntry
            {
                RecordHandler handler = nullptr;
                bool isThreadEvent = false;
            };
            RecordHandlerEntry recordHandlers[kNvtxTraceRecordTypeCount] = {};

            NvtxTraceBuilderData(NvtxTraceData& shadowed_)
                : ProviderBuilderData(ProviderType::NvtxTrace)
                , shadowed(shadowed_)
            {
            }

            // Preserve segments encountered before their owner during backward dependency loading,
            // but discard bytes outside the owner's declared total. Missing bytes remain zero.
            static void ReserveMessage(std::string& message, uint32_t totalMessageLength, uint32_t maxMessageLength)
            {
                const uint32_t reservedLength = std::min(totalMessageLength, maxMessageLength);
                NVTRACE_LOG_ERR_IF(reservedLength < totalMessageLength, 50, "NvtxTrace message states a length of %u, past the %u the data model holds; the rest is dropped", totalMessageLength, maxMessageLength);

                message.resize(reservedLength, '\0');
            }

            static void PlaceIntoMessage(std::string& message, uint32_t messageOffset, const uint8_t* pBytes, uint32_t length)
            {
                if (length == 0)
                {
                    return;
                }
                if (message.size() <= messageOffset)
                {
                    NVTRACE_LOG_ERR(50, "NvtxTrace message share at offset %u falls outside its %zu-byte message; it is dropped", messageOffset, message.size());
                    return;
                }

                const size_t placedLength = std::min<size_t>(length, message.size() - messageOffset);
                NVTRACE_LOG_ERR_IF(placedLength < length, 50, "NvtxTrace message share at offset %u overruns its %zu-byte message; %zu bytes are dropped", messageOffset, message.size(), length - placedLength);

                memcpy(message.data() + messageOffset, pBytes, placedLength);
            }

            size_t GetOrCreatePushPopRangeIndex(NvTrace_RevisionedRef pushPopRangeRef)
            {
                auto it = pushPopRangeRefToIndex.lower_bound(pushPopRangeRef);
                if (it != pushPopRangeRefToIndex.end() && it->first == pushPopRangeRef)
                {
                    return it->second;
                }
                const size_t index = pushPopRanges.size();
                pushPopRanges.emplace_back().rangeRef = pushPopRangeRef;
                pushPopRangeRefToIndex.emplace_hint(it, pushPopRangeRef, index);
                return index;
            }

            void IngestPushPopRangeState(NvTrace_RevisionedRef pushPopRangeRef, uint64_t recordLocation, const NvTrace_NvtxTrace_PushPopRangeCreated& pushPopRangeCreated, const uint8_t* pInlineMessage, uint32_t inlineMessageSegmentLength, uint32_t maxMessageLength)
            {
                const size_t index = GetOrCreatePushPopRangeIndex(pushPopRangeRef);
                {
                    PushPopRange& range = pushPopRanges[index];
                    range.recordLocation = recordLocation;
                    range.startTimestampTsc = pushPopRangeCreated.startTimestampTsc;
                    range.uniqueThreadId = pushPopRangeCreated.uniqueThreadId;
                    range.domainIndex = pushPopRangeCreated.domainIndex;
                    range.nestingDepth = pushPopRangeCreated.nestingDepth;
                    range.messageType = pushPopRangeCreated.messageType;
                    range.approximationMask &= ~(FieldMask_StartTimestamp | FieldMask_NestingDepth);

                    if (pushPopRangeCreated.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString)
                    {
                        range.message.assign(reinterpret_cast<const char*>(pInlineMessage), sizeof(uint64_t));
                    }
                    else
                    {
                        ReserveMessage(range.message, pushPopRangeCreated.totalMessageLength, maxMessageLength);
                        PlaceIntoMessage(range.message, 0, pInlineMessage, inlineMessageSegmentLength);
                    }

                    if (pushPopRangeCreated.endTimestampTsc)
                    {
                        range.endTimestampTsc = pushPopRangeCreated.endTimestampTsc;
                        range.approximationMask &= ~FieldMask_EndTimestamp;
                    }
                    else if ((range.approximationMask & FieldMask_EndTimestamp) || range.endTimestampTsc == InvalidTimestamp)
                    {
                        range.endTimestampTsc = pushPopRangeCreated.startTimestampTsc;
                        range.approximationMask |= FieldMask_EndTimestamp;
                    }
                }

                if (pushPopRangeCreated.ancestorRef != NVTRACE_INVALID_REVISIONED_REF)
                {
                    const size_t ancestorIndex = GetOrCreatePushPopRangeIndex(pushPopRangeCreated.ancestorRef);
                    pushPopRanges[index].ancestorIndex = ancestorIndex;
                    ApproximateAncestorFromDescendantPush(pushPopRanges[ancestorIndex], pushPopRanges[index]);
                }
            }

            void IngestRangePop(NvTrace_RevisionedRef pushPopRangeRef, uint64_t endTimestampTsc)
            {
                const size_t index = GetOrCreatePushPopRangeIndex(pushPopRangeRef);
                {
                    PushPopRange& range = pushPopRanges[index];
                    range.endTimestampTsc = endTimestampTsc;
                    range.approximationMask &= ~FieldMask_EndTimestamp;
                    if ((range.approximationMask & FieldMask_StartTimestamp) || range.startTimestampTsc == InvalidTimestamp)
                    {
                        range.startTimestampTsc = endTimestampTsc;
                        range.approximationMask |= FieldMask_StartTimestamp;
                    }
                }

                const size_t ancestorIndex = pushPopRanges[index].ancestorIndex;
                if (ancestorIndex != PushPopRange::InvalidPushPopRangeIndex)
                {
                    ApproximateAncestorEnd(pushPopRanges[ancestorIndex], endTimestampTsc);
                }
            }

            static void ApproximateAncestorStart(PushPopRange& ancestor, uint64_t descendantStart)
            {
                if (!((ancestor.approximationMask & FieldMask_StartTimestamp) || ancestor.startTimestampTsc == InvalidTimestamp))
                {
                    return;
                }
                ancestor.startTimestampTsc = (ancestor.startTimestampTsc == InvalidTimestamp) ? descendantStart : std::min(descendantStart, ancestor.startTimestampTsc);
                ancestor.approximationMask |= FieldMask_StartTimestamp;
            }

            static void ApproximateAncestorEnd(PushPopRange& ancestor, uint64_t descendantEnd)
            {
                if (!((ancestor.approximationMask & FieldMask_EndTimestamp) || ancestor.endTimestampTsc == InvalidTimestamp))
                {
                    return;
                }
                ancestor.endTimestampTsc = (ancestor.endTimestampTsc == InvalidTimestamp) ? descendantEnd : std::max(descendantEnd, ancestor.endTimestampTsc);
                ancestor.approximationMask |= FieldMask_EndTimestamp;
            }

            static void ApproximateAncestorFromDescendantPush(PushPopRange& ancestor, const PushPopRange& descendant)
            {
                if (!ancestor.uniqueThreadId)
                {
                    ancestor.uniqueThreadId = descendant.uniqueThreadId;
                }
                if (ancestor.domainIndex == PushPopRange::InvalidDomainIndex)
                {
                    ancestor.domainIndex = descendant.domainIndex;
                }

                ApproximateAncestorStart(ancestor, descendant.startTimestampTsc);
                if ((ancestor.approximationMask & FieldMask_NestingDepth) || !ancestor.nestingDepth)
                {
                    const uint16_t candidate = static_cast<uint16_t>(descendant.nestingDepth - 1);
                    ancestor.nestingDepth = !ancestor.nestingDepth ? candidate : std::min(candidate, ancestor.nestingDepth);
                    ancestor.approximationMask |= FieldMask_NestingDepth;
                }
                ApproximateAncestorEnd(ancestor, descendant.endTimestampTsc);
            }

            size_t GetOrCreateStackGroupIndex(NvTrace_RevisionedRef stackGroupRef)
            {
                auto it = stackGroupRefToIndex.lower_bound(stackGroupRef);
                if (it != stackGroupRefToIndex.end() && it->first == stackGroupRef)
                {
                    return it->second;
                }
                const size_t index = stackGroups.size();
                stackGroups.emplace_back();
                stackGroupRefToIndex.emplace_hint(it, stackGroupRef, index);
                return index;
            }
        };

        struct ThreadEventProviderBuilderData : ProviderBuilderData
        {
            ThreadEventProviderData& shadowed;
            std::map<NVTRACE_LUID, uint32_t> providerLuidToProviderMappingId;

            ThreadEventProviderBuilderData(ThreadEventProviderData& shadowed_)
                : ProviderBuilderData(ProviderType::ThreadEvent)
                , shadowed(shadowed_)
            {
            }
        };

        struct GpuPartitionBuilderData
        {
            GpuPartitionData& shadowed;

            GpuClockCalibrationBuilderData gpuClockCalibration;
            GpuPmCountersBuilderData gpuPmCounters;
            GpuHwTraceBuilderData gpuHwTrace;
            GpuGrCtxTraceBuilderData gpuGrCtxTrace;

            GpuPartitionBuilderData(GpuPartitionData& shadowed_)
                : shadowed(shadowed_)
                , gpuClockCalibration(shadowed.gpuClockCalibration)
                , gpuPmCounters(shadowed.gpuPmCounters, shadowed.info.chipName)
                , gpuHwTrace(shadowed.gpuHwTrace)
                , gpuGrCtxTrace(shadowed.gpuGrCtxTrace)
            {
            }
        };

        struct ProcessBuilderData
        {
            ProcessData& shadowed;

            CudaApiTraceBuilderData cudaApiTrace;
            NvtxTraceBuilderData nvtxTrace;
            std::map<NVTRACE_LUID, ThreadEventProviderBuilderData> threadEventProviders;

            ProcessBuilderData(ProcessData& shadowed_)
                : shadowed(shadowed_)
                , cudaApiTrace(shadowed.cudaApiTrace)
                , nvtxTrace(shadowed.nvtxTrace)
            {
            }
        };

        struct SystemBuilderData
        {
            CpuClockCalibrationBuilderData cpuClockCalibration;
            std::map<NVTRACE_LUID, GpuPartitionBuilderData> gpuPartitions;
            std::map<NVTRACE_LUID, ProcessBuilderData> processes;
            std::map<NVTRACE_LUID, ProviderBuilderData*> providers;
            // Backing entries for the providers this build does not decode; providers points into it.
            std::map<NVTRACE_LUID, ProviderBuilderData> droppedProviders;

            SystemBuilderData(SystemData& shadowed)
                : cpuClockCalibration(shadowed.cpuClockCalibration)
            {
            }

            void Clear()
            {
                providers.clear();
                droppedProviders.clear();
                processes.clear();
                gpuPartitions.clear();
            }
        };

        bool AddProviderRecord(ProviderBuilderData& providerBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            if (providerBuilderData.invalid)
            {
                return false;
            }
            switch (providerBuilderData.providerInfo.providerType)
            {
                case ProviderType::Invalid:
                case ProviderType::Unknown:
                case ProviderType::Count:
                    return false;
                case ProviderType::CpuClockCalibration:
                    return AddCpuClockCalibrationRecord(static_cast<CpuClockCalibrationBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::GpuClockCalibration:
                    return AddGpuClockCalibrationRecord(static_cast<GpuClockCalibrationBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::GpuPmCounters:
                    return AddGpuPmCountersRecord(static_cast<GpuPmCountersBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::GpuHwTrace:
                    return AddGpuHwTraceRecord(static_cast<GpuHwTraceBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::GpuGrCtxTrace:
                    return AddGpuGrCtxTraceRecord(static_cast<GpuGrCtxTraceBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::CudaApiTrace:
                    return AddCudaApiTraceRecord(static_cast<CudaApiTraceBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::NvtxTrace:
                    return AddNvtxTraceRecord(static_cast<NvtxTraceBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
                case ProviderType::ThreadEvent:
                    return AddThreadEventRecord(static_cast<ThreadEventProviderBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
            }

            return false;
        }

        bool CanReplayProviderOrphans(const ProviderBuilderData& providerBuilderData) const
        {
            switch (providerBuilderData.providerInfo.providerType)
            {
                case ProviderType::Invalid:
                case ProviderType::Unknown:
                case ProviderType::Count:
                    return false;
                case ProviderType::CpuClockCalibration:
                case ProviderType::GpuClockCalibration:
                case ProviderType::GpuPmCounters:
                case ProviderType::GpuHwTrace:
                case ProviderType::GpuGrCtxTrace:
                case ProviderType::CudaApiTrace:
                case ProviderType::NvtxTrace:
                case ProviderType::ThreadEvent:
                    return true;
            }

            return false;
        }

        template <typename TOrphanRecordMap, typename TReplayOrphanRecord>
        void ReplayProviderOrphanRecords(const ProviderBuilderData& providerBuilderData, TOrphanRecordMap& providerLuidToOrphanRecords, TReplayOrphanRecord fnReplayOrphanRecord)
        {
            if (!CanReplayProviderOrphans(providerBuilderData))
            {
                return;
            }

            const auto itrOrphanRecords = providerLuidToOrphanRecords.find(providerBuilderData.providerInfo.providerLuid);
            if (itrOrphanRecords == providerLuidToOrphanRecords.end())
            {
                return;
            }

            for (const auto& orphanRecord : itrOrphanRecords->second)
            {
                fnReplayOrphanRecord(orphanRecord);
            }

            providerLuidToOrphanRecords.erase(itrOrphanRecords);
        }

        void ApplyThreadEventProviderMappingToProvider(ProviderBuilderData& providerBuilderData, ThreadEventProviderData& threadEventProviderData)
        {
            const NVTRACE_LUID threadEventProviderLuid = threadEventProviderData.providerLuid;
            switch (providerBuilderData.providerInfo.providerType)
            {
                case ProviderType::CudaApiTrace:
                {
                    CudaApiTraceBuilderData& cudaApiTraceBuilderData = static_cast<CudaApiTraceBuilderData&>(providerBuilderData);
                    cudaApiTraceBuilderData.shadowed.threadEventProviderLuid = threadEventProviderLuid;
                    cudaApiTraceBuilderData.pThreadEventProviderData = &threadEventProviderData;
                    break;
                }

                case ProviderType::NvtxTrace:
                {
                    NvtxTraceBuilderData& nvtxTraceBuilderData = static_cast<NvtxTraceBuilderData&>(providerBuilderData);
                    nvtxTraceBuilderData.shadowed.threadEventProviderLuid = threadEventProviderLuid;
                    nvtxTraceBuilderData.pThreadEventProviderData = &threadEventProviderData;
                    break;
                }

                default:
                    break;
            }
        }

        void ApplyThreadEventProviderMappingToMappedProvider(ThreadEventProviderBuilderData& threadEventProviderBuilderData, NVTRACE_LUID mappedProviderLuid)
        {
            const NVTRACE_LUID threadEventProviderLuid = threadEventProviderBuilderData.shadowed.providerLuid;
            if (threadEventProviderLuid == NVTRACE_NIL_LUID)
            {
                return;
            }

            const auto itrProvider = m_systemBuilderData.providers.find(mappedProviderLuid);
            if (itrProvider == m_systemBuilderData.providers.end())
            {
                return;
            }

            ApplyThreadEventProviderMappingToProvider(*itrProvider->second, threadEventProviderBuilderData.shadowed);
        }

        void ApplyThreadEventProviderMappingForMappedProvider(ProcessBuilderData& processBuilderData, NVTRACE_LUID mappedProviderLuid)
        {
            for (auto& [_, threadEventProviderBuilderData] : processBuilderData.threadEventProviders)
            {
                const auto itrProviderMappingId = threadEventProviderBuilderData.providerLuidToProviderMappingId.find(mappedProviderLuid);
                if (itrProviderMappingId == threadEventProviderBuilderData.providerLuidToProviderMappingId.end())
                {
                    continue;
                }

                const auto itrProviderMapping = threadEventProviderBuilderData.shadowed.providerMappings.find(itrProviderMappingId->second);
                if (itrProviderMapping == threadEventProviderBuilderData.shadowed.providerMappings.end())
                {
                    continue;
                }

                ApplyThreadEventProviderMappingToMappedProvider(threadEventProviderBuilderData, itrProviderMapping->second);
                return;
            }
        }

        void ApplyThreadEventProviderMapping(ThreadEventProviderBuilderData& threadEventProviderBuilderData, uint32_t providerMappingId, NVTRACE_LUID mappedProviderLuid)
        {
            ThreadEventProviderData& threadEventProviderData = threadEventProviderBuilderData.shadowed;
            const auto itrProviderMapping = threadEventProviderData.providerMappings.lower_bound(providerMappingId);
            if (itrProviderMapping != threadEventProviderData.providerMappings.end() && itrProviderMapping->first == providerMappingId)
            {
                if (itrProviderMapping->second == mappedProviderLuid)
                {
                    return;
                }
                threadEventProviderBuilderData.providerLuidToProviderMappingId.erase(itrProviderMapping->second);
                itrProviderMapping->second = mappedProviderLuid;
            }
            else
            {
                threadEventProviderData.providerMappings.emplace_hint(itrProviderMapping, providerMappingId, mappedProviderLuid);
            }

            threadEventProviderBuilderData.providerLuidToProviderMappingId[mappedProviderLuid] = providerMappingId;

            ApplyThreadEventProviderMappingToMappedProvider(threadEventProviderBuilderData, mappedProviderLuid);
            static_cast<TDerived*>(this)->OnObjectDefined(threadEventProviderData.providerLuid, ObjectType::ThreadEventProviderMapping, providerMappingId);
        }

        bool AddSystemCreatedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::SystemInfo>(), false, 50, "Invalid SystemInfo record");
            const FB::SystemInfo* const pFbSystemInfo = flatbuffers::GetRoot<FB::SystemInfo>(pMajorPayload);

            m_dataModel.systemInfo.hostname = pFbSystemInfo->hostname()->str();
            m_dataModel.systemInfo.serviceStartTimestampTsc = pFbSystemInfo->serviceStartTimestampTsc();
            m_dataModel.systemInfo.osVersion = pFbSystemInfo->osVersion()->str();
            m_dataModel.systemInfo.osFamily = pFbSystemInfo->osFamily();
            m_dataModel.systemInfo.cpuModel = pFbSystemInfo->cpuModel()->str();
            m_dataModel.systemInfo.cpuArch = pFbSystemInfo->cpuArch();
            m_dataModel.systemInfo.sysMemTotal = pFbSystemInfo->sysMemTotal();
            m_dataModel.systemInfo.nvidiaRmKernelDriverVersion = ConvertFBSemVer(pFbSystemInfo->nvidiaRmKernelDriverVersion());
            m_dataModel.systemInfo.nvidiaMrmKernelDriverVersion = ConvertFBSemVer(pFbSystemInfo->nvidiaMrmKernelDriverVersion());

            return true;
        }

        bool AddGpuPartitionCreatedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::GpuPartitionCreated>(), false, 50, "Invalid GpuPartitionCreated record");
            const FB::Codec::NvTraceServiceEvents::GpuPartitionCreated* const pFbGpuPartitionCreated = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::GpuPartitionCreated>(pMajorPayload);

            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pFbGpuPartitionCreated->gpuPartitionInfo()->data(), pFbGpuPartitionCreated->gpuPartitionInfo()->size()).VerifyBuffer<FB::GpuPartitionInfo>(), false, 50, "Invalid GpuPartitionInfo record");
            const FB::GpuPartitionInfo* const pFbGpuPartitionInfo = flatbuffers::GetRoot<FB::GpuPartitionInfo>(pFbGpuPartitionCreated->gpuPartitionInfo()->data());

            const NVTRACE_LUID gpuPartitionLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbGpuPartitionCreated->gpuPartitionLuid());
            auto [itrGpuPartitionData, inserted] = m_dataModel.gpuPartitions.try_emplace(gpuPartitionLuid);
            if (!inserted)
            {
                // Already added
                return true;
            }

            GpuPartitionData& gpuPartitionData = itrGpuPartitionData->second;
            gpuPartitionData.gpuPartitionLuid = gpuPartitionLuid;
            gpuPartitionData.creationTimestampTsc = pFbGpuPartitionCreated->creationTimestampTsc();
            gpuPartitionData.destructionTimestampTsc = MaxTimestampTsc;
            gpuPartitionData.gpuUuid = *reinterpret_cast<const NVTRACE_UUID*>(pFbGpuPartitionCreated->gpuUuid());
            gpuPartitionData.info.chipName = pFbGpuPartitionInfo->chipName()->string_view();
            gpuPartitionData.info.deviceName = pFbGpuPartitionInfo->deviceName()->string_view();
            gpuPartitionData.info.driverVersion = ConvertFBSemVer(pFbGpuPartitionInfo->driverVersion());
            gpuPartitionData.info.driverType = pFbGpuPartitionInfo->driverType();
            gpuPartitionData.info.supportedProviders.reserve(pFbGpuPartitionInfo->supportedProviders()->size());
            for (const auto& supportedProvider : *pFbGpuPartitionInfo->supportedProviders())
            {
                gpuPartitionData.info.supportedProviders.push_back(*reinterpret_cast<const NVTRACE_UUID*>(supportedProvider));
            }
            gpuPartitionData.info.attributes.reserve(pFbGpuPartitionInfo->cudaAttributes()->size());
            for (const auto& cudaAttribute : *pFbGpuPartitionInfo->cudaAttributes())
            {
                if (cudaAttribute->value_type() == FB::CudaAttributeValue_s32)
                {
                    gpuPartitionData.info.attributes.push_back({cudaAttribute->key(), cudaAttribute->value_as_s32()->value()});
                }
                else
                {
                    NVTRACE_LOG_ERR(50, "Unsupported CUDA attribute value type: %u", cudaAttribute->value_type());
                }
            }
            gpuPartitionData.info.deviceMemorySize = pFbGpuPartitionInfo->deviceMemorySize();
            gpuPartitionData.info.isProfiled = pFbGpuPartitionInfo->isProfiled();

            m_systemBuilderData.gpuPartitions.try_emplace(gpuPartitionLuid, gpuPartitionData);
            return true;
        }

        bool AddGpuPartitionDestroyedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::GpuPartitionDestroyed>(), false, 50, "Invalid GpuPartitionDestroyed record");
            const FB::Codec::NvTraceServiceEvents::GpuPartitionDestroyed* const pFbGpuPartitionDestroyed = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::GpuPartitionDestroyed>(pMajorPayload);

            auto itrGpuPartitionData = m_dataModel.gpuPartitions.find(pFbGpuPartitionDestroyed->uniqueGpuPartitionLuid()->value());
            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionData == m_dataModel.gpuPartitions.end(), false, 50, "Unknown uniqueGpuPartitionLuid=%" PRIu64, pFbGpuPartitionDestroyed->uniqueGpuPartitionLuid()->value());
            itrGpuPartitionData->second.destructionTimestampTsc = pFbGpuPartitionDestroyed->destructionTimestampTsc();

            return true;
        }

        bool AddProcessCreatedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::ProcessCreated>(), false, 50, "Invalid ProcessCreated record");
            const FB::Codec::NvTraceServiceEvents::ProcessCreated* const pFbProcessCreated = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::ProcessCreated>(pMajorPayload);

            const NVTRACE_LUID processLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbProcessCreated->uniqueProcessId());
            auto [itrProcessData, inserted] = m_dataModel.processes.try_emplace(processLuid);
            if (!inserted)
            {
                // Already added
                return true;
            }

            ProcessData& processData = itrProcessData->second;
            processData.processLuid = processLuid;
            processData.namespacePid = pFbProcessCreated->namespacePid();
            processData.namespaceParentPid = pFbProcessCreated->namespaceParentPid();
            processData.globalPid = pFbProcessCreated->globalPid();
            // Optional struct field; null (older trace that omitted it) => {0,0}.
            const auto* pFbPidNamespaceId = pFbProcessCreated->pidNamespaceId();
            processData.pidNamespaceId = pFbPidNamespaceId ? NvTrace_LinuxNamespaceId{pFbPidNamespaceId->inode(), pFbPidNamespaceId->device()} : NvTrace_LinuxNamespaceId{};
            processData.path = pFbProcessCreated->path()->str();
            processData.creationTimestampTsc = pFbProcessCreated->creationTimestampTsc();
            processData.destructionTimestampTsc = MaxTimestampTsc;

            m_systemBuilderData.processes.try_emplace(processLuid, processData);
            return true;
        }

        bool AddProcessDestroyedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::ProcessDestroyed>(), false, 50, "Invalid ProcessDestroyed record");
            const FB::Codec::NvTraceServiceEvents::ProcessDestroyed* const pFbProcessDestroyed = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::ProcessDestroyed>(pMajorPayload);

            auto itrProcessData = m_dataModel.processes.find(pFbProcessDestroyed->uniqueProcessId()->value());
            NVTRACE_LOG_ERR_RETURN_IF(itrProcessData == m_dataModel.processes.end(), false, 50, "Unknown uniqueProcessId=%" PRIu64, pFbProcessDestroyed->uniqueProcessId()->value());
            itrProcessData->second.destructionTimestampTsc = pFbProcessDestroyed->destructionTimestampTsc();
            return true;
        }

        // Registered invalid so its records resolve and drop silently, instead of each reporting an unknown provider.
        bool AddDroppedProvider(const ProviderInfo& providerInfo)
        {
            auto [itrDroppedProvider, droppedProviderInserted] = m_systemBuilderData.droppedProviders.try_emplace(providerInfo.providerLuid, ProviderType::Unknown);
            NVTRACE_LOG_ERR_RETURN_IF(!droppedProviderInserted, false, 50, "Duplicate provider LUID %" PRIu64, providerInfo.providerLuid);

            NVTRACE_LOG_ERR(50, "Provider (luid=%" PRIu64 ") has providerTypeUuid=%s, which this build does not decode; its records are dropped.", providerInfo.providerLuid, ToString(providerInfo.providerTypeUuid).c_str());
            itrDroppedProvider->second.providerInfo = providerInfo;
            itrDroppedProvider->second.invalid = true;
            m_systemBuilderData.providers.try_emplace(providerInfo.providerLuid, &itrDroppedProvider->second);
            return false;
        }

        bool AddProviderCreatedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            ProviderInfo providerInfo;
            NVTRACE_LOG_ERR_RETURN_IF(!GetProviderInfoFromProviderCreated(pMajorPayload, maxMajorPayloadSize, providerInfo), false, 50, "Invalid or unsupported ProviderCreated record");
            return AddProviderCreatedRecord(pMajorPayload, maxMajorPayloadSize, providerInfo);
        }

        bool AddProviderCreatedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize, const ProviderInfo& providerInfo)
        {
            ProviderBuilderData* pProviderBuilderData = nullptr;
            return AddProviderCreatedRecord(pMajorPayload, maxMajorPayloadSize, providerInfo, pProviderBuilderData);
        }

        bool AddProviderCreatedRecord(const uint8_t* pMajorPayload, size_t maxMajorPayloadSize, const ProviderInfo& providerInfo, ProviderBuilderData*& pProviderBuilderData)
        {
            pProviderBuilderData = nullptr;
            const FB::Codec::NvTraceServiceEvents::ProviderCreated* const pFbProviderCreated = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::ProviderCreated>(pMajorPayload);

            const NVTRACE_LUID providerLuid = providerInfo.providerLuid;
            const auto itrExistingProviderBuilderData = m_systemBuilderData.providers.find(providerLuid);
            if (itrExistingProviderBuilderData != m_systemBuilderData.providers.end())
            {
                // Already added
                pProviderBuilderData = itrExistingProviderBuilderData->second;
                return true;
            }

            const NVTRACE_LUID parentLuid = providerInfo.parentLuid;
            auto itrProviderBuilderData = m_systemBuilderData.providers.end();
            switch (providerInfo.providerType)
            {
                case ProviderType::CpuClockCalibration:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(m_systemBuilderData.cpuClockCalibration.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one CPU_ClockCalibration provider");
                    NVTRACE_LOG_ERR_RETURN_IF(parentLuid != NVTRACE_NIL_LUID, false, 50, "Unknown parentLuid=%" PRIu64 " for a CPU_ClockCalibration provider", parentLuid);

                    m_systemBuilderData.cpuClockCalibration.shadowed.providerLuid = providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &m_systemBuilderData.cpuClockCalibration).first;

                    // Register only the RecordVersionInfo handler; the other handlers will be registered along with the RecordVersionInfo record.
                    m_systemBuilderData.cpuClockCalibration.recordHandlers[NvTrace_CPU_ClockCalibration_TscRtc_RecordType_RecordVersionInfo] = &DataModelBuilderBase::AddCpuClockCalibrationRecordVersionInfoRecord;
                    break;
                }
                case ProviderType::GpuClockCalibration:
                {
                    auto itrGpuPartitionBuilderData = m_systemBuilderData.gpuPartitions.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionBuilderData == m_systemBuilderData.gpuPartitions.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a GPU_ClockCalibration provider", parentLuid);

                    GpuPartitionBuilderData& gpuPartitionBuilderData = itrGpuPartitionBuilderData->second;
                    NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionBuilderData.gpuClockCalibration.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one GPU_ClockCalibration provider in the same GpuPartition");

                    gpuPartitionBuilderData.gpuClockCalibration.shadowed.providerLuid = providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &gpuPartitionBuilderData.gpuClockCalibration).first;

                    // Register only the RecordVersionInfo handler; the other handlers will be registered along with the RecordVersionInfo record.
                    gpuPartitionBuilderData.gpuClockCalibration.recordHandlers[NvTrace_GPU_ClockCalibration_TscGpu_RecordType_RecordVersionInfo] = &DataModelBuilderBase::AddGpuClockCalibrationRecordVersionInfoRecord;
                    break;
                }
                case ProviderType::GpuPmCounters:
                {
                    auto itrGpuPartitionBuilderData = m_systemBuilderData.gpuPartitions.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionBuilderData == m_systemBuilderData.gpuPartitions.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a GPU_PmCounters provider", parentLuid);

                    GpuPartitionBuilderData& gpuPartitionBuilderData = itrGpuPartitionBuilderData->second;
                    NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionBuilderData.gpuPmCounters.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one GPU_PmCounters provider in the same GpuPartition");

                    gpuPartitionBuilderData.gpuPmCounters.shadowed.providerLuid = providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &gpuPartitionBuilderData.gpuPmCounters).first;

                    // Register only the RecordVersionInfo handler; the other handlers will be registered along with the RecordVersionIndo record
                    gpuPartitionBuilderData.gpuPmCounters.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_RecordVersionInfo] = &DataModelBuilderBase::AddGpuPmCountersRecordVersionInfoRecord;
                    break;
                }
                case ProviderType::GpuHwTrace:
                {
                    auto itrGpuPartitionBuilderData = m_systemBuilderData.gpuPartitions.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionBuilderData == m_systemBuilderData.gpuPartitions.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a GPU_HwTrace provider", parentLuid);

                    GpuPartitionBuilderData& gpuPartitionBuilderData = itrGpuPartitionBuilderData->second;
                    NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionBuilderData.gpuHwTrace.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one GPU_HwTrace provider in the same GpuPartition");

                    gpuPartitionBuilderData.gpuHwTrace.shadowed.providerLuid = providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &gpuPartitionBuilderData.gpuHwTrace).first;

                    // Register only the RecordVersionInfo handler; the other handlers will be registered along with the RecordVersionInfo record.
                    gpuPartitionBuilderData.gpuHwTrace.recordHandlers[NvTrace_GPU_HwTrace_RecordType_RecordVersionInfo] = &DataModelBuilderBase::AddGpuHwTraceRecordVersionInfoRecord;
                    break;
                }
                case ProviderType::GpuGrCtxTrace:
                {
                    auto itrGpuPartitionBuilderData = m_systemBuilderData.gpuPartitions.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionBuilderData == m_systemBuilderData.gpuPartitions.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a GPU_GrCtxTrace provider", parentLuid);

                    GpuPartitionBuilderData& gpuPartitionBuilderData = itrGpuPartitionBuilderData->second;
                    NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionBuilderData.gpuGrCtxTrace.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one GPU_GrCtxTrace provider in the same GpuPartition");

                    gpuPartitionBuilderData.gpuGrCtxTrace.shadowed.providerLuid = providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &gpuPartitionBuilderData.gpuGrCtxTrace).first;

                    // Register only the RecordVersionInfo handler; the other handlers will be registered along with the RecordVersionInfo record.
                    gpuPartitionBuilderData.gpuGrCtxTrace.recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_RecordVersionInfo] = &DataModelBuilderBase::AddGpuGrCtxTraceRecordVersionInfoRecord;
                    break;
                }
                case ProviderType::CudaApiTrace:
                {
                    auto itrProcessBuilderData = m_systemBuilderData.processes.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrProcessBuilderData == m_systemBuilderData.processes.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a CUDA_ApiTrace provider", parentLuid);

                    ProcessBuilderData& processBuilderData = itrProcessBuilderData->second;
                    NVTRACE_LOG_ERR_RETURN_IF(processBuilderData.cudaApiTrace.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one CUDA_ApiTrace provider in the same Process");

                    processBuilderData.cudaApiTrace.shadowed.providerLuid = providerLuid;
                    processBuilderData.cudaApiTrace.nvtxTraceProviderLuid = processBuilderData.nvtxTrace.shadowed.providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &processBuilderData.cudaApiTrace).first;

                    const flatbuffers::Vector<uint8_t>* pEventVersionInfoBlob = pFbProviderCreated->eventVersionInfo();
                    if (!pEventVersionInfoBlob)
                    {
                        NVTRACE_LOG_ERR(50, "CudaApiTrace RecordVersionInfo is missing");
                        processBuilderData.cudaApiTrace.invalid = true;
                        return false;
                    }
                    if (pFbProviderCreated->eventVersionInfoMinorVersion() < NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A)
                    {
                        NVTRACE_LOG_ERR(50, "CudaApiTrace RecordVersionInfo has incompatible minor version %u (expected >= %u)", pFbProviderCreated->eventVersionInfoMinorVersion(), NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A);
                        processBuilderData.cudaApiTrace.invalid = true;
                        return false;
                    }

                    // The minor-version scalar above is separate FlatBuffer state; the untrusted blob itself must be
                    // large enough (and aligned) to carry the version-A fields before it is trusted.
                    if (pEventVersionInfoBlob->size() < NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A || (reinterpret_cast<uintptr_t>(pEventVersionInfoBlob->data()) % alignof(NvTrace_CUDA_ApiTrace_RecordVersionInfo)) != 0)
                    {
                        NVTRACE_LOG_ERR(50, "CudaApiTrace RecordVersionInfo blob is too small or misaligned (size=%u expectedMinSize=%u)", pEventVersionInfoBlob->size(), NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A);
                        processBuilderData.cudaApiTrace.invalid = true;
                        return false;
                    }

                    const size_t recordVersionInfoCopySize = std::min(sizeof(NvTrace_CUDA_ApiTrace_RecordVersionInfo), static_cast<size_t>(pEventVersionInfoBlob->size()));
                    memcpy(&processBuilderData.cudaApiTrace.shadowed.recordVersionInfo, pEventVersionInfoBlob->data(), recordVersionInfoCopySize);
                    PopulateCudaApiTraceRecordHandlers(processBuilderData.cudaApiTrace, processBuilderData.cudaApiTrace.shadowed.recordVersionInfo);

                    const flatbuffers::Vector<uint8_t>* pConfigBlob = pFbProviderCreated->configuration();
                    if (pConfigBlob && pConfigBlob->size() > 0 && !flatbuffers::Verifier(pConfigBlob->data(), pConfigBlob->size()).VerifyBuffer<FB::Provider::CudaApiTrace::ProviderConfiguration>())
                    {
                        NVTRACE_LOG_ERR(50, "CudaApiTrace ProviderConfiguration verification failed");
                    }
                    ApplyThreadEventProviderMappingForMappedProvider(processBuilderData, providerLuid);
                    break;
                }
                case ProviderType::NvtxTrace:
                {
                    auto itrProcessBuilderData = m_systemBuilderData.processes.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrProcessBuilderData == m_systemBuilderData.processes.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a NVTX Trace provider", parentLuid);

                    ProcessBuilderData& processBuilderData = itrProcessBuilderData->second;
                    NVTRACE_LOG_ERR_RETURN_IF(processBuilderData.nvtxTrace.shadowed.providerLuid != NVTRACE_NIL_LUID, false, 50, "More than one NVTX Trace provider in the same Process");

                    processBuilderData.nvtxTrace.shadowed.providerLuid = providerLuid;
                    processBuilderData.cudaApiTrace.nvtxTraceProviderLuid = providerLuid;
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &processBuilderData.nvtxTrace).first;

                    // Reference StackGroups that arrived from CUDA before the NvtxTrace provider
                    for (const NvTrace_RevisionedRef stackGroupRef : processBuilderData.cudaApiTrace.pendingNvtxTraceStackGroupRefs)
                    {
                        static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTraceStackGroup, stackGroupRef);
                    }
                    processBuilderData.cudaApiTrace.pendingNvtxTraceStackGroupRefs.clear();

                    const flatbuffers::Vector<uint8_t>* pEventVersionInfoBlob = pFbProviderCreated->eventVersionInfo();
                    if (!pEventVersionInfoBlob)
                    {
                        NVTRACE_LOG_ERR(50, "NvtxTrace RecordVersionInfo is missing");
                        processBuilderData.nvtxTrace.invalid = true;
                        return false;
                    }
                    if (pFbProviderCreated->eventVersionInfoMinorVersion() < NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A)
                    {
                        NVTRACE_LOG_ERR(50, "NvtxTrace RecordVersionInfo has incompatible minor version %u (expected >= %u)", pFbProviderCreated->eventVersionInfoMinorVersion(), NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A);
                        processBuilderData.nvtxTrace.invalid = true;
                        return false;
                    }

                    // The minor-version scalar above is separate FlatBuffer state; the untrusted blob itself must be
                    // large enough (and aligned) to carry the version-A fields before it is trusted.
                    if (pEventVersionInfoBlob->size() < NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A || (reinterpret_cast<uintptr_t>(pEventVersionInfoBlob->data()) % alignof(NvTrace_NvtxTrace_RecordVersionInfo)) != 0)
                    {
                        NVTRACE_LOG_ERR(50, "NvtxTrace RecordVersionInfo blob is too small or misaligned (size=%u expectedMinSize=%u)", pEventVersionInfoBlob->size(), NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A);
                        processBuilderData.nvtxTrace.invalid = true;
                        return false;
                    }

                    const size_t recordVersionInfoCopySize = std::min(sizeof(NvTrace_NvtxTrace_RecordVersionInfo), static_cast<size_t>(pEventVersionInfoBlob->size()));
                    memcpy(&processBuilderData.nvtxTrace.shadowed.recordVersionInfo, pEventVersionInfoBlob->data(), recordVersionInfoCopySize);
                    processBuilderData.nvtxTrace.shadowed.recordVersionInfoMinorVersion = pFbProviderCreated->eventVersionInfoMinorVersion();
                    PopulateNvtxTraceRecordHandlers(processBuilderData.nvtxTrace, processBuilderData.nvtxTrace.shadowed.recordVersionInfo);

                    ApplyThreadEventProviderMappingForMappedProvider(processBuilderData, providerLuid);
                    break;
                }
                // Streams recorded before the V2 format carry the V1 id.
                case ProviderType::ThreadEvent:
                {
                    auto itrProcessBuilderData = m_systemBuilderData.processes.find(parentLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrProcessBuilderData == m_systemBuilderData.processes.end(), false, 50, "Unknown parentLuid=%" PRIu64 " for a ThreadEvent provider", parentLuid);

                    ProcessBuilderData& processBuilderData = itrProcessBuilderData->second;
                    auto [itrThreadEventProviderData, threadEventProviderDataInserted] = processBuilderData.shadowed.threadEventProviders.try_emplace(providerLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(!threadEventProviderDataInserted, false, 50, "Duplicate ThreadEvent provider LUID %" PRIu64 " in the same Process", providerLuid);
                    itrThreadEventProviderData->second.providerLuid = providerLuid;

                    auto [itrThreadEventProviderBuilderData, threadEventProviderBuilderDataInserted] = processBuilderData.threadEventProviders.try_emplace(providerLuid, itrThreadEventProviderData->second);
                    NVTRACE_LOG_ERR_RETURN_IF(!threadEventProviderBuilderDataInserted, false, 50, "Duplicate ThreadEvent provider builder LUID %" PRIu64 " in the same Process", providerLuid);
                    itrProviderBuilderData = m_systemBuilderData.providers.try_emplace(providerLuid, &itrThreadEventProviderBuilderData->second).first;
                    break;
                }
                case ProviderType::Unknown:
                    return AddDroppedProvider(providerInfo);

                case ProviderType::Invalid:
                case ProviderType::Count:
                    return false;
            }

            NVTRACE_LOG_ERR_RETURN_IF(itrProviderBuilderData == m_systemBuilderData.providers.end(), false, 50, "ProviderBuilderData was not registered for providerLuid=%" PRIu64, providerLuid);
            itrProviderBuilderData->second->providerInfo = providerInfo;
            pProviderBuilderData = itrProviderBuilderData->second;
            return true;
        }

        bool AddRouterRecord(uint32_t recordType, uint64_t /* correlationId */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            switch (static_cast<FB::Codec::NvTraceServiceEvents::EventTypeId>(recordType))
            {
                case FB::Codec::NvTraceServiceEvents::EventTypeId_Invalid:
                    return false;

                case FB::Codec::NvTraceServiceEvents::EventTypeId_SystemCreated:
                    return AddSystemCreatedRecord(pMajorPayload, maxMajorPayloadSize);

                case FB::Codec::NvTraceServiceEvents::EventTypeId_GpuPartitionCreated:
                    return AddGpuPartitionCreatedRecord(pMajorPayload, maxMajorPayloadSize);

                case FB::Codec::NvTraceServiceEvents::EventTypeId_GpuPartitionDestroyed:
                    return AddGpuPartitionDestroyedRecord(pMajorPayload, maxMajorPayloadSize);

                case FB::Codec::NvTraceServiceEvents::EventTypeId_ProcessCreated:
                    return AddProcessCreatedRecord(pMajorPayload, maxMajorPayloadSize);

                case FB::Codec::NvTraceServiceEvents::EventTypeId_ProcessDestroyed:
                    return AddProcessDestroyedRecord(pMajorPayload, maxMajorPayloadSize);

                case FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderCreated:
                    return AddProviderCreatedRecord(pMajorPayload, maxMajorPayloadSize);

                case FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::ProviderDestroyed>(), false, 50, "Invalid ProviderDestroyed record");
                    return true;

                case FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderAttached:
                    NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::ProviderAttached>(), false, 50, "Invalid ProviderAttached record");
                    return true;
            }

            NVTRACE_LOG_ERR(50, "Unknown NvTraceService record type %d", recordType);
            return false;
        }

        bool AddCpuClockCalibrationRecordVersionInfoRecord(CpuClockCalibrationBuilderData& cpuClockCalibrationBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_CPU_ClockCalibration_TscRtc_RecordVersionInfo_MINOR_VERSION_A, false, 50, "Invalid CpuClockCalibration RecordVersionInfo record");

            const NvTrace_CPU_ClockCalibration_TscRtc_RecordVersionInfo recordVersionInfo = *reinterpret_cast<const NvTrace_CPU_ClockCalibration_TscRtc_RecordVersionInfo*>(pPayload);

            // Register handlers within NvTrace_CPU_ClockCalibration_TscRtc_RecordVersionInfo_MINOR_VERSION_A.
            cpuClockCalibrationBuilderData.recordHandlers[NvTrace_CPU_ClockCalibration_TscRtc_RecordType_ProviderConfiguration] = &DataModelBuilderBase::AddCpuClockCalibrationProviderConfigurationRecord;

            if (NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint_MINOR_VERSION_A <= recordVersionInfo.calibrationPointMinorVersion)
            {
                cpuClockCalibrationBuilderData.recordHandlers[NvTrace_CPU_ClockCalibration_TscRtc_RecordType_CalibrationPoint] = &DataModelBuilderBase::AddCpuClockCalibrationPointRecordA;
            }

            return true;
        }

        bool AddCpuClockCalibrationProviderConfigurationRecord(CpuClockCalibrationBuilderData& /* cpuClockCalibrationBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pPayload, payloadSize).VerifyBuffer<FB::Provider::CpuClockCalibration::ProviderConfiguration>(), false, 50, "Invalid CpuClockCalibration ProviderConfiguration record");
            return true;
        }

        bool AddCpuClockCalibrationPointRecordA(CpuClockCalibrationBuilderData& cpuClockCalibrationBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint_MINOR_VERSION_A, false, 50, "Invalid CpuClockCalibration CalibrationPoint record");
            cpuClockCalibrationBuilderData.shadowed.calibrationPoints.push_back(*reinterpret_cast<const TscRtcCalibrationPoint*>(pPayload));
            return true;
        }

        bool AddCpuClockCalibrationRecord(CpuClockCalibrationBuilderData& cpuClockCalibrationBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(NvTrace_CPU_ClockCalibration_TscRtc_RecordType_Count <= recordType, false, 50, "Unknown CpuClockCalibration record type %u", recordType);

            const typename CpuClockCalibrationBuilderData::RecordHandler& recordHandler = cpuClockCalibrationBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandler, false, 50, "No handler for CpuClockCalibration record type %u", recordType);

            return (this->*recordHandler)(cpuClockCalibrationBuilderData, recordLocation, correlationId, pMajorPayload, maxMajorPayloadSize);
        }

        bool AddGpuClockCalibrationRecordVersionInfoRecord(GpuClockCalibrationBuilderData& gpuClockCalibrationBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo_MINOR_VERSION_A, false, 50, "Invalid GpuClockCalibration RecordVersionInfo record");

            const NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo recordVersionInfo = *reinterpret_cast<const NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo*>(pPayload);

            // Register handlers within NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo_MINOR_VERSION_A.
            gpuClockCalibrationBuilderData.recordHandlers[NvTrace_GPU_ClockCalibration_TscGpu_RecordType_ProviderConfiguration] = &DataModelBuilderBase::AddGpuClockCalibrationProviderConfigurationRecord;

            if (NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint_MINOR_VERSION_A <= recordVersionInfo.calibrationPointMinorVersion)
            {
                gpuClockCalibrationBuilderData.recordHandlers[NvTrace_GPU_ClockCalibration_TscGpu_RecordType_CalibrationPoint] = &DataModelBuilderBase::AddGpuClockCalibrationPointRecordA;
            }

            if (NvTrace_GPU_DeviceInfo_MINOR_VERSION_A <= recordVersionInfo.gpuDeviceInfoMinorVersion)
            {
                gpuClockCalibrationBuilderData.recordHandlers[NvTrace_GPU_ClockCalibration_TscGpu_RecordType_GpuDeviceInfo] = &DataModelBuilderBase::AddGpuClockCalibrationGpuDeviceInfoRecordA;
            }

            return true;
        }

        bool AddGpuClockCalibrationProviderConfigurationRecord(GpuClockCalibrationBuilderData& gpuClockCalibrationBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pPayload, payloadSize).VerifyBuffer<FB::Provider::GpuClockCalibration::ProviderConfiguration>(), false, 50, "Invalid GpuClockCalibration ProviderConfiguration record");

            const FB::Provider::GpuClockCalibration::ProviderConfiguration* const pFbConfig = flatbuffers::GetRoot<FB::Provider::GpuClockCalibration::ProviderConfiguration>(pPayload);
            NVTRACE_LOG_ERR_RETURN_IF(!pFbConfig, false, 50, "Invalid GpuClockCalibration ProviderConfiguration record");

            GpuClockCalibrationData& gpuClockCalibrationData = gpuClockCalibrationBuilderData.shadowed;
            gpuClockCalibrationData.circularBufferCellCount = pFbConfig->circularBufferCellCount();
            gpuClockCalibrationData.updateIntervalNsec = pFbConfig->updateIntervalNsec();
            return true;
        }

        bool AddGpuClockCalibrationPointRecordA(GpuClockCalibrationBuilderData& gpuClockCalibrationBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint_MINOR_VERSION_A, false, 50, "Invalid GpuClockCalibration CalibrationPoint record");
            gpuClockCalibrationBuilderData.shadowed.calibrationPoints.push_back(*reinterpret_cast<const TscGpuCalibrationPoint*>(pPayload));
            return true;
        }

        bool AddGpuClockCalibrationGpuDeviceInfoRecordA(GpuClockCalibrationBuilderData& /* gpuClockCalibrationBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_DeviceInfo_MINOR_VERSION_A, false, 50, "Invalid GpuClockCalibration GpuDeviceInfo record");
            return true;
        }

        bool AddGpuClockCalibrationRecord(GpuClockCalibrationBuilderData& gpuClockCalibrationBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(NvTrace_GPU_ClockCalibration_TscGpu_RecordType_Count <= recordType, false, 50, "Unknown GpuClockCalibration record type %u", recordType);

            const typename GpuClockCalibrationBuilderData::RecordHandler& recordHandler = gpuClockCalibrationBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandler, false, 50, "No handler for GpuClockCalibration record type %u", recordType);

            return (this->*recordHandler)(gpuClockCalibrationBuilderData, recordLocation, correlationId, pMajorPayload, maxMajorPayloadSize);
        }

        bool AddGpuPmCountersRecordVersionInfoRecord(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_PmCounters_RecordVersionInfo_MINOR_VERSION_A, false, 50, "Invalid GpuPmCounters RecordVersionInfo record");

            gpuPmCountersBuilderData.recordVersionInfo = {};
            memcpy(&gpuPmCountersBuilderData.recordVersionInfo, pPayload, std::min(payloadSize, sizeof(NvTrace_GPU_PmCounters_RecordVersionInfo)));
            const NvTrace_GPU_PmCounters_RecordVersionInfo& recordVersionInfo = gpuPmCountersBuilderData.recordVersionInfo;

            // Version A record handlers
            gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_ProviderConfiguration] = &DataModelBuilderBase::AddGpuPmCountersProviderConfigurationRecord;

            if (NvTrace_GPU_DeviceInfo_MINOR_VERSION_A <= recordVersionInfo.gpuDeviceInfoMinorVersion)
            {
                gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_GpuDeviceInfo] = &DataModelBuilderBase::AddGpuPmCountersGpuDeviceInfoRecordA;
            }

            if (NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticInitializationFailureMinorVersion)
            {
                gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_InitializationFailure] =
                    &DataModelBuilderBase::AddGpuPmCountersProviderDiagnosticInitializationFailureRecordA;
            }

            if (NvTrace_GPU_PmCounters_CounterDataInfo_MINOR_VERSION_A <= recordVersionInfo.counterDataInfoMinorVersion && NvTrace_GPU_PmCounters_CounterDataInfo_MINOR_VERSION_A <= recordVersionInfo.counterDataInfoSize)
            {
                gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_CounterDataInfo] = &DataModelBuilderBase::AddGpuPmCountersCounterDataInfoRecordA;
            }

            if (NvTrace_GPU_PmCounters_CounterSample_MINOR_VERSION_A <= recordVersionInfo.counterSampleMinorVersion && NvTrace_GPU_PmCounters_CounterSample_MINOR_VERSION_A <= recordVersionInfo.counterSampleSize)
            {
                gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_CounterSample] = &DataModelBuilderBase::AddGpuPmCountersCounterSampleRecordA;
                gpuPmCountersBuilderData.shadowed.counterSampleFixedFieldSize = recordVersionInfo.counterSampleSize;
            }

            if (NvTrace_GPU_Provider_LifecycleEvent_Pause_MINOR_VERSION_A <= recordVersionInfo.lifecyclePauseMinorVersion)
            {
                gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_LifecyclePause] = &DataModelBuilderBase::AddGpuPmCountersLifecyclePauseRecordA;
            }

            if (NvTrace_GPU_Provider_LifecycleEvent_Resume_MINOR_VERSION_A <= recordVersionInfo.lifecycleResumeMinorVersion)
            {
                gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_LifecycleResume] = &DataModelBuilderBase::AddGpuPmCountersLifecycleResumeRecordA;
            }

            // Version B record handlers
            if (NvTrace_GPU_PmCounters_RecordVersionInfo_MINOR_VERSION_B <= payloadSize)
            {
                if (NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticUpdateTimeExceededMinorVersion && NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticUpdateTimeExceededSize)
                {
                    gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_UpdateTimeExceeded] = &DataModelBuilderBase::AddGpuPmCountersProviderDiagnosticUpdateTimeExceededRecordA;
                }
                if (NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticSampleDiscontinuityMinorVersion && NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticSampleDiscontinuitySize)
                {
                    gpuPmCountersBuilderData.recordHandlers[NvTrace_GPU_PmCounters_RecordType::NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_SampleDiscontinuity] = &DataModelBuilderBase::AddGpuPmCountersProviderDiagnosticSampleDiscontinuityRecordA;
                }
            }

            return true;
        }

        bool AddGpuPmCountersGpuDeviceInfoRecordA(GpuPmCountersBuilderData& /* gpuPmCountersBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_DeviceInfo_MINOR_VERSION_A, false, 50, "Invalid GpuPmCounters GpuDeviceInfo record");
            return true;
        }

        bool AddGpuPmCountersProviderConfigurationRecord(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            GpuPmCountersData& gpuPmCountersData = gpuPmCountersBuilderData.shadowed;

            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pPayload, payloadSize).VerifyBuffer<FB::Provider::GpuPmCounters::ProviderConfiguration>(), false, 50, "Invalid GpuPmCounters ProviderConfiguration record");

            const FB::Provider::GpuPmCounters::ProviderConfiguration* const pFbGpuPmConfig = flatbuffers::GetRoot<FB::Provider::GpuPmCounters::ProviderConfiguration>(pPayload);
            NVTRACE_LOG_ERR_RETURN_IF(!pFbGpuPmConfig, false, 50, "Invalid GpuPmCounters ProviderConfiguration record");

            gpuPmCountersData.pmSampleRateHz = pFbGpuPmConfig->pmSampleRateHz();
            gpuPmCountersData.updateIntervalNsec = pFbGpuPmConfig->updateIntervalNsec();
            gpuPmCountersData.readbackHoldoffNsec = pFbGpuPmConfig->readbackHoldoffNsec();
            gpuPmCountersData.keepLatest = pFbGpuPmConfig->keepLatest();

            gpuPmCountersData.metricsToConfigure.clear();
            gpuPmCountersData.metricsToEvaluate.clear();
            gpuPmCountersData.timeline = {};
            gpuPmCountersData.customMetricsScript.clear();
            gpuPmCountersData.metricConfigDocument.clear();
            if (const auto* pFbArchitectures = pFbGpuPmConfig->architectures(); pFbArchitectures)
            {
                // Rows nest to whatever depth the set declared, so this recurses. Reading a fixed depth
                // would silently drop levels a document is free to add.
                // Matches the depth the writer bounds itself to. The trace is a file from elsewhere, so a
                // cyclic or absurdly deep buffer must stop the walk rather than the process.
                constexpr size_t kMaxTimelineRowDepth = 32;
                const auto fnReadRow = [&](const auto& self, const FB::Provider::GpuPmCounters::TimelineRow* pFbRow, Codec::GpuPmCountersData::TimelineRow& outRow, size_t depth) -> void {
                    if (!pFbRow)
                    {
                        return;
                    }
                    outRow.metricGroupName = pFbRow->metricGroupName() ? pFbRow->metricGroupName()->str() : std::string();
                    outRow.stacked = pFbRow->type() == FB::Provider::GpuPmCounters::TimelineRowType_Stacked;
                    outRow.heightScale = pFbRow->heightScale();
                    if (depth + 1 >= kMaxTimelineRowDepth)
                    {
                        return;
                    }
                    for (::flatbuffers::uoffset_t childIdx = 0; pFbRow->children() && childIdx < pFbRow->children()->size(); ++childIdx)
                    {
                        self(self, pFbRow->children()->Get(childIdx), outRow.children.emplace_back(), depth + 1);
                    }
                };

                auto fnReadTimeline = [&fnReadRow](const FB::Provider::GpuPmCounters::TimelineConfig* pFbTimeline, Codec::GpuPmCountersData::TimelineConfig& outTimeline) {
                    if (!pFbTimeline)
                    {
                        return;
                    }
                    outTimeline.version = pFbTimeline->version();
                    for (::flatbuffers::uoffset_t viewIdx = 0; pFbTimeline->views() && viewIdx < pFbTimeline->views()->size(); ++viewIdx)
                    {
                        const auto* pFbView = pFbTimeline->views()->Get(viewIdx);
                        auto& outView = outTimeline.views.emplace_back();
                        outView.name = pFbView->name() ? pFbView->name()->str() : std::string();
                        for (::flatbuffers::uoffset_t rowIdx = 0; pFbView->rows() && rowIdx < pFbView->rows()->size(); ++rowIdx)
                        {
                            fnReadRow(fnReadRow, pFbView->rows()->Get(rowIdx), outView.rows.emplace_back(), 0);
                        }
                    }
                    for (::flatbuffers::uoffset_t groupIdx = 0; pFbTimeline->metricGroups() && groupIdx < pFbTimeline->metricGroups()->size(); ++groupIdx)
                    {
                        const auto* pFbGroup = pFbTimeline->metricGroups()->Get(groupIdx);
                        auto& outGroup = outTimeline.metricGroups.emplace_back();
                        outGroup.name = pFbGroup->name() ? pFbGroup->name()->str() : std::string();
                        outGroup.description = pFbGroup->description() ? pFbGroup->description()->str() : std::string();
                        outGroup.sumAcrossMetrics = pFbGroup->aggregationOp() == FB::Provider::GpuPmCounters::TimelineAggregationOp_Sum;
                        for (::flatbuffers::uoffset_t nameIdx = 0; pFbGroup->metricNames() && nameIdx < pFbGroup->metricNames()->size(); ++nameIdx)
                        {
                            outGroup.metricNames.push_back(pFbGroup->metricNames()->Get(nameIdx)->str());
                        }
                        // Absent in traces written before the group carried a selection, which read as "all".
                        for (::flatbuffers::uoffset_t subIdx = 0; pFbGroup->subMetricNames() && subIdx < pFbGroup->subMetricNames()->size(); ++subIdx)
                        {
                            outGroup.subMetricNames.push_back(pFbGroup->subMetricNames()->Get(subIdx)->str());
                        }
                    }
                    for (::flatbuffers::uoffset_t metricIdx = 0; pFbTimeline->metrics() && metricIdx < pFbTimeline->metrics()->size(); ++metricIdx)
                    {
                        const auto* pFbMetric = pFbTimeline->metrics()->Get(metricIdx);
                        auto& outMetric = outTimeline.metrics.emplace_back();
                        outMetric.name = pFbMetric->name() ? pFbMetric->name()->str() : std::string();
                        outMetric.baseId = pFbMetric->baseId() ? pFbMetric->baseId()->str() : std::string();
                        outMetric.description = pFbMetric->description() ? pFbMetric->description()->str() : std::string();
                        outMetric.colorString = pFbMetric->color() ? pFbMetric->color()->str() : std::string();
                        for (::flatbuffers::uoffset_t subIdx = 0; pFbMetric->subMetrics() && subIdx < pFbMetric->subMetrics()->size(); ++subIdx)
                        {
                            const auto* pFbSub = pFbMetric->subMetrics()->Get(subIdx);
                            auto& outSub = outMetric.subMetrics.emplace_back();
                            outSub.name = pFbSub->name() ? pFbSub->name()->str() : std::string();
                            outSub.suffix = pFbSub->suffix() ? pFbSub->suffix()->str() : std::string();
                            outSub.fullId = pFbSub->fullId() ? pFbSub->fullId()->str() : std::string();
                            outSub.multiplier = pFbSub->multiplier();
                            outSub.maxValue = pFbSub->maxValue();
                        }
                    }
                };

                auto fnCopyFbVectorOfStrings = [](const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* pFbVector, std::vector<std::string>& out_vector) {
                    if (pFbVector)
                    {
                        out_vector.reserve(pFbVector->size());
                        for (const auto* pFbString : *pFbVector)
                        {
                            if (pFbString)
                            {
                                out_vector.emplace_back(pFbString->str());
                            }
                        }
                    }
                };

                // The provider records the entry it selected and ships only that one, so the answer is already in
                // the trace; re-running chipRegex here would be a second copy free to disagree with it. Older traces
                // predate that rewrite and carry several entries, so those still match on the chip.
                const FB::Provider::GpuPmCounters::Architecture* pFbResolvedArchitecture = nullptr;
                if (pFbArchitectures->size() == 1)
                {
                    pFbResolvedArchitecture = pFbArchitectures->Get(0);
                }
                else
                {
                    // A trace is read long after it was captured, possibly by a build that never wrote
                    // it, so the pattern is treated as data: absent, or malformed enough to throw out of
                    // std::regex, must skip the entry rather than take down the export.
                    for (const FB::Provider::GpuPmCounters::Architecture* pFbArchitecture : *pFbArchitectures)
                    {
                        if (!pFbArchitecture || !pFbArchitecture->chipRegex())
                        {
                            continue;
                        }
                        try
                        {
                            if (std::regex_match(gpuPmCountersBuilderData.chipName, std::regex(pFbArchitecture->chipRegex()->c_str(), std::regex_constants::icase)))
                            {
                                pFbResolvedArchitecture = pFbArchitecture;
                                break;
                            }
                        }
                        catch (const std::regex_error&)
                        {
                            NVTRACE_LOG_WRN(50, "GPU PM counters: ignoring the malformed chip pattern \"%s\" recorded in this trace", pFbArchitecture->chipRegex()->c_str());
                        }
                    }
                }

                if (pFbResolvedArchitecture)
                {
                    if (const auto* pFbMetricsToConfigure = pFbResolvedArchitecture->metricsToConfigure(); pFbMetricsToConfigure)
                    {
                        gpuPmCountersData.metricsToConfigure.reserve(pFbMetricsToConfigure->size());
                        for (const auto* pFbMetric : *pFbMetricsToConfigure)
                        {
                            if (pFbMetric && pFbMetric->metricName())
                            {
                                gpuPmCountersData.metricsToConfigure.push_back({pFbMetric->metricName()->str(), pFbMetric->keepInstances(), pFbMetric->displayName() ? pFbMetric->displayName()->str() : std::string(), pFbMetric->category() ? pFbMetric->category()->str() : std::string()});
                            }
                        }
                    }
                    fnReadTimeline(pFbResolvedArchitecture->timeline(), gpuPmCountersData.timeline);
                    if (const auto* pFbCustomMetricsScript = pFbResolvedArchitecture->customMetricsScript(); pFbCustomMetricsScript)
                    {
                        gpuPmCountersData.customMetricsScript = pFbCustomMetricsScript->str();
                    }
                    fnCopyFbVectorOfStrings(pFbResolvedArchitecture->metricsToEvaluate(), gpuPmCountersData.metricsToEvaluate);

                    // The document that produced this entry, taken from the selection it points at. The
                    // provider ships only the surviving selection, so index 0 is that one; older traces
                    // carry no document at all and leave this empty.
                    if (const auto* pFbSelections = pFbGpuPmConfig->metricConfigSelections(); pFbSelections)
                    {
                        const int32_t selectionIndex = pFbResolvedArchitecture->metricConfigIndex();
                        const flatbuffers::uoffset_t selectionIdx = pFbSelections->size() == 1 ? 0u : (selectionIndex >= 0 ? (flatbuffers::uoffset_t)selectionIndex : pFbSelections->size());
                        if (selectionIdx < pFbSelections->size())
                        {
                            if (const auto* pFbSelection = pFbSelections->Get(selectionIdx); pFbSelection && pFbSelection->document())
                            {
                                gpuPmCountersData.metricConfigDocument = pFbSelection->document()->str();
                            }
                        }
                    }
                }
            }

            return true;
        }

        bool AddGpuPmCountersProviderDiagnosticInitializationFailureRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A, false, 50, "Invalid GpuPmCounters ProviderDiagnostic_InitializationFailure record");

            GpuPmCountersData& gpuPmCountersData = gpuPmCountersBuilderData.shadowed;

            const NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure& initFailurePayload = *reinterpret_cast<const NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure*>(pPayload);

            GpuPmCountersData::ProviderDiagnostic& providerDiagnosticEmplaced = gpuPmCountersData.providerDiagnostics.emplace_back();
            providerDiagnosticEmplaced.recordType = NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_InitializationFailure;
            providerDiagnosticEmplaced.payload.initializationFailure.timestampTsc = initFailurePayload.failureTimestampTsc;
            providerDiagnosticEmplaced.payload.initializationFailure.failureReason = initFailurePayload.failureReason;
            return true;
        }

        bool AddGpuPmCountersCounterDataInfoRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_PmCounters_CounterDataInfo_MINOR_VERSION_A, false, 50, "Invalid GpuPmCounters CounterDataInfo record");

            GpuPmCountersData& gpuPmCountersData = gpuPmCountersBuilderData.shadowed;

            const NvTrace_GPU_PmCounters_CounterDataInfo& counterDataInfoPayload = *reinterpret_cast<const NvTrace_GPU_PmCounters_CounterDataInfo*>(pPayload);

            gpuPmCountersData.counterDataPreambleSize = counterDataInfoPayload.counterDataPreambleSize;
            gpuPmCountersData.sampleRangeDataSize = counterDataInfoPayload.sampleRangeDataSize;
            gpuPmCountersData.sampleEvaluatedMetricCount = counterDataInfoPayload.sampleEvaluatedMetricCount;

            // The dynamic preamble begins at the aligned end of the fixed-field segment published by the provider.
            const size_t preambleOffsetInPayload = AlignUpP2(static_cast<size_t>(gpuPmCountersBuilderData.recordVersionInfo.counterDataInfoSize), alignof(uint64_t));
            const size_t availablePreambleSize = payloadSize >= preambleOffsetInPayload ? payloadSize - preambleOffsetInPayload : 0;
            if (availablePreambleSize < gpuPmCountersData.counterDataPreambleSize)
            {
                NVTRACE_LOG_ERR(50, "Invalid GpuPmCounters CounterDataInfo preamble size: %zu < %zu", availablePreambleSize, gpuPmCountersData.counterDataPreambleSize);
                return false;
            }
            gpuPmCountersData.counterDataPreamble.assign(pPayload + preambleOffsetInPayload, pPayload + preambleOffsetInPayload + gpuPmCountersData.counterDataPreambleSize);

            return true;
        }

        bool AddGpuPmCountersCounterSampleRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < gpuPmCountersBuilderData.recordVersionInfo.counterSampleSize, false, 50, "Invalid GpuPmCounters CounterSample record");

            GpuPmCountersData& gpuPmCountersData = gpuPmCountersBuilderData.shadowed;
            gpuPmCountersData.sampleRecordLocations.push_back(recordLocation);

            return true;
        }

        bool AddGpuPmCountersProviderDiagnosticUpdateTimeExceededRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded_MINOR_VERSION_A, false, 50, "Invalid GpuPmCounters ProviderDiagnostic_UpdateTimeExceeded record");

            GpuPmCountersData& gpuPmCountersData = gpuPmCountersBuilderData.shadowed;

            const NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded& updateTimeExceededPayload = *reinterpret_cast<const NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded*>(pPayload);

            GpuPmCountersData::ProviderDiagnostic& providerDiagnosticEmplaced = gpuPmCountersData.providerDiagnostics.emplace_back();
            providerDiagnosticEmplaced.recordType = NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_UpdateTimeExceeded;
            providerDiagnosticEmplaced.payload.updateTimeExceeded.updateStartTimestampTsc = updateTimeExceededPayload.updateStartTimestampTsc;
            providerDiagnosticEmplaced.payload.updateTimeExceeded.updateEndTimestampTsc = updateTimeExceededPayload.updateEndTimestampTsc;
            providerDiagnosticEmplaced.payload.updateTimeExceeded.updateThreadCpuTimeNsec = updateTimeExceededPayload.updateThreadCpuTimeNsec;
            providerDiagnosticEmplaced.payload.updateTimeExceeded.updateWallTimeNsec = 0;
            return true;
        }

        bool AddGpuPmCountersProviderDiagnosticSampleDiscontinuityRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity_MINOR_VERSION_A, false, 50, "Invalid GpuPmCounters ProviderDiagnostic_SampleDiscontinuity record");

            GpuPmCountersData& gpuPmCountersData = gpuPmCountersBuilderData.shadowed;

            const NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity& sampleDiscontinuityPayload = *reinterpret_cast<const NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity*>(pPayload);

            GpuPmCountersData::ProviderDiagnostic& providerDiagnosticEmplaced = gpuPmCountersData.providerDiagnostics.emplace_back();
            providerDiagnosticEmplaced.recordType = NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_SampleDiscontinuity;
            providerDiagnosticEmplaced.payload.sampleDiscontinuity.startTimestampTsc = sampleDiscontinuityPayload.startTimestampTsc;
            providerDiagnosticEmplaced.payload.sampleDiscontinuity.endTimestampTsc = sampleDiscontinuityPayload.endTimestampTsc;
            providerDiagnosticEmplaced.payload.sampleDiscontinuity.reason = sampleDiscontinuityPayload.reason;
            return true;
        }

        enum class GpuProviderLifecycleParserState
        {
            Empty,
            OpenPause,
            OpenResumeStarted,
            OpenResumeFailed,
        };

        template <typename TGpuProviderData>
        GpuProviderLifecycleParserState GetGpuProviderLifecycleParserState(const TGpuProviderData& gpuProviderData)
        {
            if (gpuProviderData.lifecycleEvents.empty() || gpuProviderData.lifecycleEvents.back().endTimestampTsc != MaxTimestampTsc)
            {
                return GpuProviderLifecycleParserState::Empty;
            }

            // Get Lifecycle parser state based on previous Lifecycle event
            const GpuProviderLifecycleEvent& previousLifecycleEvent = gpuProviderData.lifecycleEvents.back();
            switch (previousLifecycleEvent.lifecycleEventType)
            {
                case NvTrace_GPU_Provider_LifecycleEventType_Pause:
                    return GpuProviderLifecycleParserState::OpenPause;

                case NvTrace_GPU_Provider_LifecycleEventType_Resume:
                    switch (previousLifecycleEvent.resumePhase)
                    {
                        case NvTrace_GPU_Provider_ResumePhase_Started:
                            return GpuProviderLifecycleParserState::OpenResumeStarted;
                        case NvTrace_GPU_Provider_ResumePhase_Failed:
                            return GpuProviderLifecycleParserState::OpenResumeFailed;
                        case NvTrace_GPU_Provider_ResumePhase_Succeeded:
                        case NvTrace_GPU_Provider_ResumePhase_Invalid:
                        case NvTrace_GPU_Provider_ResumePhase_COUNT:
                        case NvTrace_GPU_Provider_ResumePhase_FORCEINT:
                            return GpuProviderLifecycleParserState::Empty;
                    }
                    return GpuProviderLifecycleParserState::Empty;

                case NvTrace_GPU_Provider_LifecycleEventType_Invalid:
                case NvTrace_GPU_Provider_LifecycleEventType_COUNT:
                case NvTrace_GPU_Provider_LifecycleEventType_FORCEINT:
                    return GpuProviderLifecycleParserState::Empty;
            }

            return GpuProviderLifecycleParserState::Empty;
        }

        template <typename TGpuProviderData>
        bool AddGpuProviderLifecyclePauseRecordA(TGpuProviderData& gpuProviderData, const char* pProviderName, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_Provider_LifecycleEvent_Pause_MINOR_VERSION_A, false, 50, "Invalid %s LifecyclePause record", pProviderName);

            const NvTrace_GPU_Provider_LifecycleEvent_Pause& pausePayload = *reinterpret_cast<const NvTrace_GPU_Provider_LifecycleEvent_Pause*>(pPayload);
            if (pausePayload.pauseTimestampTsc == InvalidTimestamp)
            {
                NVTRACE_LOG_WRN(50, "%s LifecyclePause has invalid pauseTimestampTsc; dropping.", pProviderName);
                return false;
            }

            const GpuProviderLifecycleParserState parserState = GetGpuProviderLifecycleParserState(gpuProviderData);
            if (parserState != GpuProviderLifecycleParserState::Empty)
            {
                gpuProviderData.lifecycleEvents.back().endTimestampTsc = pausePayload.pauseTimestampTsc;
            }

            GpuProviderLifecycleEvent& lifecycleEvent = gpuProviderData.lifecycleEvents.emplace_back();
            lifecycleEvent.lifecycleEventType = NvTrace_GPU_Provider_LifecycleEventType_Pause;
            lifecycleEvent.timestampTsc = pausePayload.pauseTimestampTsc;
            lifecycleEvent.endTimestampTsc = MaxTimestampTsc;
            lifecycleEvent.pauseReason = pausePayload.pauseReason;

            return true;
        }

        template <typename TGpuProviderData>
        bool AddGpuProviderLifecycleResumeRecordA(TGpuProviderData& gpuProviderData, const char* pProviderName, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_Provider_LifecycleEvent_Resume_MINOR_VERSION_A, false, 50, "Invalid %s LifecycleResume record", pProviderName);

            const NvTrace_GPU_Provider_LifecycleEvent_Resume& resumePayload = *reinterpret_cast<const NvTrace_GPU_Provider_LifecycleEvent_Resume*>(pPayload);
            if (resumePayload.resumeTimestampTsc == InvalidTimestamp)
            {
                NVTRACE_LOG_WRN(50, "%s LifecycleResume has invalid resumeTimestampTsc; dropping.", pProviderName);
                return false;
            }

            const GpuProviderLifecycleParserState parserState = GetGpuProviderLifecycleParserState(gpuProviderData);
            switch (resumePayload.resumePhase)
            {
                case NvTrace_GPU_Provider_ResumePhase_Started:
                {
                    if (parserState != GpuProviderLifecycleParserState::Empty)
                    {
                        gpuProviderData.lifecycleEvents.back().endTimestampTsc = resumePayload.resumeTimestampTsc;
                    }

                    GpuProviderLifecycleEvent& lifecycleEvent = gpuProviderData.lifecycleEvents.emplace_back();
                    lifecycleEvent.lifecycleEventType = NvTrace_GPU_Provider_LifecycleEventType_Resume;
                    lifecycleEvent.timestampTsc = resumePayload.resumeTimestampTsc;
                    lifecycleEvent.endTimestampTsc = MaxTimestampTsc;
                    lifecycleEvent.pauseReason = resumePayload.pauseReason;
                    lifecycleEvent.retryCount = resumePayload.retryCount;
                    lifecycleEvent.resumePhase = resumePayload.resumePhase;
                    lifecycleEvent.resumeFailureReason = resumePayload.resumeFailureReason;
                    return true;
                }

                case NvTrace_GPU_Provider_ResumePhase_Failed:
                {
                    if (parserState == GpuProviderLifecycleParserState::OpenResumeStarted || parserState == GpuProviderLifecycleParserState::OpenResumeFailed)
                    {
                        GpuProviderLifecycleEvent& previousLifecycleEvent = gpuProviderData.lifecycleEvents.back();
                        previousLifecycleEvent.endTimestampTsc = MaxTimestampTsc;
                        previousLifecycleEvent.pauseReason = resumePayload.pauseReason;
                        previousLifecycleEvent.retryCount = resumePayload.retryCount;
                        previousLifecycleEvent.resumePhase = resumePayload.resumePhase;
                        previousLifecycleEvent.resumeFailureReason = resumePayload.resumeFailureReason;
                        return true;
                    }

                    if (parserState == GpuProviderLifecycleParserState::OpenPause)
                    {
                        gpuProviderData.lifecycleEvents.back().endTimestampTsc = resumePayload.resumeTimestampTsc;
                    }

                    GpuProviderLifecycleEvent& lifecycleEvent = gpuProviderData.lifecycleEvents.emplace_back();
                    lifecycleEvent.lifecycleEventType = NvTrace_GPU_Provider_LifecycleEventType_Resume;
                    lifecycleEvent.timestampTsc = resumePayload.resumeTimestampTsc;
                    lifecycleEvent.endTimestampTsc = MaxTimestampTsc;
                    lifecycleEvent.pauseReason = resumePayload.pauseReason;
                    lifecycleEvent.retryCount = resumePayload.retryCount;
                    lifecycleEvent.resumePhase = resumePayload.resumePhase;
                    lifecycleEvent.resumeFailureReason = resumePayload.resumeFailureReason;
                    return true;
                }

                case NvTrace_GPU_Provider_ResumePhase_Succeeded:
                {
                    if (parserState == GpuProviderLifecycleParserState::OpenResumeStarted || parserState == GpuProviderLifecycleParserState::OpenResumeFailed)
                    {
                        GpuProviderLifecycleEvent& previousLifecycleEvent = gpuProviderData.lifecycleEvents.back();
                        previousLifecycleEvent.endTimestampTsc = resumePayload.resumeTimestampTsc;
                        previousLifecycleEvent.pauseReason = resumePayload.pauseReason;
                        previousLifecycleEvent.retryCount = resumePayload.retryCount;
                        previousLifecycleEvent.resumePhase = resumePayload.resumePhase;
                        previousLifecycleEvent.resumeFailureReason = resumePayload.resumeFailureReason;
                        return true;
                    }

                    if (parserState == GpuProviderLifecycleParserState::OpenPause)
                    {
                        gpuProviderData.lifecycleEvents.back().endTimestampTsc = resumePayload.resumeTimestampTsc;
                    }

                    // Like a failed resume with a missing start, retain an event in
                    // the common lifecycle stream. A completed attempt has no known
                    // duration here, so use its completion timestamp for both ends.
                    GpuProviderLifecycleEvent& lifecycleEvent = gpuProviderData.lifecycleEvents.emplace_back();
                    lifecycleEvent.lifecycleEventType = NvTrace_GPU_Provider_LifecycleEventType_Resume;
                    lifecycleEvent.timestampTsc = resumePayload.resumeTimestampTsc;
                    lifecycleEvent.endTimestampTsc = resumePayload.resumeTimestampTsc;
                    lifecycleEvent.pauseReason = resumePayload.pauseReason;
                    lifecycleEvent.retryCount = resumePayload.retryCount;
                    lifecycleEvent.resumePhase = resumePayload.resumePhase;
                    lifecycleEvent.resumeFailureReason = resumePayload.resumeFailureReason;
                    return true;
                }

                case NvTrace_GPU_Provider_ResumePhase_Invalid:
                case NvTrace_GPU_Provider_ResumePhase_COUNT:
                case NvTrace_GPU_Provider_ResumePhase_FORCEINT:
                    NVTRACE_LOG_WRN(50, "%s LifecycleEvent has invalid resumePhase=%d at timestampTsc=%" PRIu64 "; dropping.", pProviderName, static_cast<int>(resumePayload.resumePhase), resumePayload.resumeTimestampTsc);
                    return false;
            }

            return false;
        }

        bool AddGpuPmCountersLifecyclePauseRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            return AddGpuProviderLifecyclePauseRecordA(gpuPmCountersBuilderData.shadowed, "GpuPmCounters", pPayload, payloadSize);
        }

        bool AddGpuPmCountersLifecycleResumeRecordA(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            return AddGpuProviderLifecycleResumeRecordA(gpuPmCountersBuilderData.shadowed, "GpuPmCounters", pPayload, payloadSize);
        }

        bool AddGpuPmCountersRecord(GpuPmCountersBuilderData& gpuPmCountersBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(NvTrace_GPU_PmCounters_RecordType_Count <= recordType, false, 50, "Unknown GpuPmCounters record type %u", recordType);

            const typename GpuPmCountersBuilderData::RecordHandler& recordHandler = gpuPmCountersBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandler, false, 50, "No handler for GpuPmCounters record type %u", recordType);

            return (this->*recordHandler)(gpuPmCountersBuilderData, recordLocation, correlationId, pMajorPayload, maxMajorPayloadSize);
        }

        bool AddGpuHwTraceRecordVersionInfoRecord(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_RecordVersionInfo_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace RecordVersionInfo record");

            gpuHwTraceBuilderData.recordVersionInfo = *reinterpret_cast<const NvTrace_GPU_HwTrace_RecordVersionInfo*>(pPayload);
            const NvTrace_GPU_HwTrace_RecordVersionInfo& recordVersionInfo = gpuHwTraceBuilderData.recordVersionInfo;

            // Register handlers within NvTrace_GPU_HwTrace_RecordVersionInfo_MINOR_VERSION_A.
            gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ProviderConfiguration] = &DataModelBuilderBase::AddGpuHwTraceProviderConfigurationRecord;

            if (NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticInitializationFailureMinorVersion)
            {
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ProviderDiagnostic_InitializationFailure] = &DataModelBuilderBase::AddGpuHwTraceProviderDiagnosticInitializationFailureRecordA;
            }

            if (NvTrace_GPU_HwTrace_ComputeGrid_MINOR_VERSION_A <= recordVersionInfo.computeGridMinorVersion)
            {
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ComputeGridStart] = &DataModelBuilderBase::AddGpuHwTraceComputeGridStartRecordA;
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ComputeGridEnd] = &DataModelBuilderBase::AddGpuHwTraceComputeGridEndRecordA;
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ComputeGridNullEnd] = &DataModelBuilderBase::AddGpuHwTraceComputeGridNullEndRecordA;
            }

            if (NvTrace_GPU_HwTrace_ContextChange_MINOR_VERSION_A <= recordVersionInfo.contextChangeMinorVersion)
            {
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ContextChangeLoad] = &DataModelBuilderBase::AddGpuHwTraceContextChangeLoadRecordA;
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_ContextChangeSave] = &DataModelBuilderBase::AddGpuHwTraceContextChangeSaveRecordA;
            }

            if (NvTrace_GPU_Provider_LifecycleEvent_Pause_MINOR_VERSION_A <= recordVersionInfo.lifecyclePauseMinorVersion)
            {
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_LifecyclePause] = &DataModelBuilderBase::AddGpuHwTraceLifecyclePauseRecordA;
            }

            if (NvTrace_GPU_Provider_LifecycleEvent_Resume_MINOR_VERSION_A <= recordVersionInfo.lifecycleResumeMinorVersion)
            {
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_LifecycleResume] = &DataModelBuilderBase::AddGpuHwTraceLifecycleResumeRecordA;
            }

            if (NvTrace_GPU_DeviceInfo_MINOR_VERSION_A <= recordVersionInfo.gpuDeviceInfoMinorVersion)
            {
                gpuHwTraceBuilderData.recordHandlers[NvTrace_GPU_HwTrace_RecordType_GpuDeviceInfo] = &DataModelBuilderBase::AddGpuHwTraceGpuDeviceInfoRecordA;
            }

            return true;
        }

        bool AddGpuHwTraceProviderConfigurationRecord(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pPayload, payloadSize).VerifyBuffer<FB::Provider::GpuHwTrace::ProviderConfiguration>(), false, 50, "Invalid GpuHwTrace ProviderConfiguration record");

            const auto* const pFbConfig = flatbuffers::GetRoot<FB::Provider::GpuHwTrace::ProviderConfiguration>(pPayload);
            NVTRACE_LOG_ERR_RETURN_IF(!pFbConfig, false, 50, "Invalid GpuHwTrace ProviderConfiguration record");

            GpuHwTraceData& gpuHwTraceData = gpuHwTraceBuilderData.shadowed;
            gpuHwTraceData.circularBufferCellCount = pFbConfig->circularBufferCellCount();
            gpuHwTraceData.updateIntervalNsec = pFbConfig->updateIntervalNsec();
            gpuHwTraceData.readbackHoldoffNsec = pFbConfig->readbackHoldoffNsec();
            gpuHwTraceData.keepLatest = pFbConfig->keepLatest();
            gpuHwTraceData.showInternalEvents = pFbConfig->showInternalEvents();
            gpuHwTraceData.recordBufferSize = pFbConfig->recordBufferSize();
            gpuHwTraceData.allowLossyProvider = pFbConfig->allowLossyProvider();
            return true;
        }

        bool AddGpuHwTraceComputeGridStartRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_ComputeGrid_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace ComputeGridStart record");
            gpuHwTraceBuilderData.hwRecords.push_back({recordLocation, NvTrace_GPU_HwTrace_RecordType_ComputeGridStart});
            return true;
        }

        bool AddGpuHwTraceComputeGridEndRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_ComputeGrid_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace ComputeGridEnd record");
            gpuHwTraceBuilderData.hwRecords.push_back({recordLocation, NvTrace_GPU_HwTrace_RecordType_ComputeGridEnd});
            return true;
        }

        bool AddGpuHwTraceComputeGridNullEndRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_ComputeGrid_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace ComputeGridNullEnd record");
            gpuHwTraceBuilderData.hwRecords.push_back({recordLocation, NvTrace_GPU_HwTrace_RecordType_ComputeGridNullEnd});
            return true;
        }

        bool AddGpuHwTraceContextChangeLoadRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_ContextChange_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace ContextChangeLoad record");
            gpuHwTraceBuilderData.hwRecords.push_back({recordLocation, NvTrace_GPU_HwTrace_RecordType_ContextChangeLoad});

            GpuHwTraceData& gpuHwTraceData = gpuHwTraceBuilderData.shadowed;
            const NvTrace_GPU_HwTrace_ContextChange& contextEvent = *reinterpret_cast<const NvTrace_GPU_HwTrace_ContextChange*>(pPayload);
            const uint32_t contextId = contextEvent.contextId;
            if (!contextId)
            {
                NVTRACE_LOG_ERR(50, "Invalid GPU HW trace context change load event with contextId=0.");
                return false;
            }

            if (!gpuHwTraceData.contextTrace.empty())
            {
                typename GpuHwTraceData::ContextInterval& lastContextInterval = gpuHwTraceData.contextTrace.back();
                if (lastContextInterval.contextId != contextId)
                {
                    if (lastContextInterval.saveTimestampGpuNsec == InvalidTimestamp)
                    {
                        NVTRACE_LOG_ERR(50, "Missing GPU HW trace context change save event for contextId=%u.", lastContextInterval.contextId);
                        return false;
                    }
                }
                else
                {
                    // For now we will treat this as a warning and return true since we're currently seeing duplicate
                    //   context load events in NvTrace HW trace.
                    NVTRACE_LOG_WRN(50, "Duplicate GPU HW trace context change load event for contextId=%u.", contextId);
                    return true;
                }
            }
            // create new interval with start time = contextEvent.ptimer
            gpuHwTraceData.contextTrace.push_back([&]() {
                typename GpuHwTraceData::ContextInterval contextInterval;
                contextInterval.contextId = contextId;
                contextInterval.loadTimestampGpuNsec = contextEvent.ptimer;
                // default contextInterval.saveTimestampGpuNsec == 0
                return contextInterval;
            }());
            return true;
        }

        bool AddGpuHwTraceContextChangeSaveRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_ContextChange_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace ContextChangeSave record");
            gpuHwTraceBuilderData.hwRecords.push_back({recordLocation, NvTrace_GPU_HwTrace_RecordType_ContextChangeSave});

            GpuHwTraceData& gpuHwTraceData = gpuHwTraceBuilderData.shadowed;
            const NvTrace_GPU_HwTrace_ContextChange& contextEvent = *reinterpret_cast<const NvTrace_GPU_HwTrace_ContextChange*>(pPayload);
            const uint32_t contextId = contextEvent.contextId;

            if (!contextId)
            {
                NVTRACE_LOG_ERR(50, "Invalid GPU HW trace context change save event with contextId=0.");
                return false;
            }

            if (!gpuHwTraceData.contextTrace.empty())
            {
                typename GpuHwTraceData::ContextInterval& lastContextInterval = gpuHwTraceData.contextTrace.back();
                if (lastContextInterval.contextId != contextId)
                {
                    NVTRACE_LOG_ERR(50, "Missing GPU HW trace context change load event for contextId=%u.", contextId);
                    return false;
                }
                else
                {
                    if (lastContextInterval.saveTimestampGpuNsec != InvalidTimestamp)
                    {
                        NVTRACE_LOG_ERR(50, "Duplicate GPU HW trace context change save event for contextId=%u.", contextId);
                        return false;
                    }
                    lastContextInterval.saveTimestampGpuNsec = contextEvent.ptimer;
                    return true;
                }
            }

            // context save with no previous context events; process as a context interval with indeterminate start
            gpuHwTraceData.contextTrace.push_back([&]() {
                typename GpuHwTraceData::ContextInterval contextInterval;
                contextInterval.contextId = contextId;
                contextInterval.saveTimestampGpuNsec = contextEvent.ptimer;
                return contextInterval;
            }());

            return true;
        }

        bool AddGpuHwTraceLifecyclePauseRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            return AddGpuProviderLifecyclePauseRecordA(gpuHwTraceBuilderData.shadowed, "GpuHwTrace", pPayload, payloadSize);
        }

        bool AddGpuHwTraceLifecycleResumeRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            return AddGpuProviderLifecycleResumeRecordA(gpuHwTraceBuilderData.shadowed, "GpuHwTrace", pPayload, payloadSize);
        }

        bool AddGpuHwTraceProviderDiagnosticInitializationFailureRecordA(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace ProviderDiagnostic InitializationFailure record");

            GpuHwTraceData& gpuHwTraceData = gpuHwTraceBuilderData.shadowed;
            const NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure& providerDiagnosticPayload = *reinterpret_cast<const NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure*>(pPayload);

            gpuHwTraceData.providerDiagnostics.push_back([&]() {
                GpuHwTraceData::ProviderDiagnostic providerDiagnostic;
                providerDiagnostic.timestampTsc = providerDiagnosticPayload.failureTimestampTsc;
                providerDiagnostic.recordType = NvTrace_GPU_HwTrace_RecordType_ProviderDiagnostic_InitializationFailure;
                providerDiagnostic.initializationFailureReason = providerDiagnosticPayload.failureReason;
                return providerDiagnostic;
            }());
            return true;
        }

        bool AddGpuHwTraceGpuDeviceInfoRecordA(GpuHwTraceBuilderData& /* gpuHwTraceBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_DeviceInfo_MINOR_VERSION_A, false, 50, "Invalid GpuHwTrace GpuDeviceInfo record");
            return true;
        }

        bool AddGpuHwTraceRecord(GpuHwTraceBuilderData& gpuHwTraceBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(NvTrace_GPU_HwTrace_RecordType_Count <= recordType, false, 50, "Unknown GpuHwTrace record type %u", recordType);

            const typename GpuHwTraceBuilderData::RecordHandler& recordHandler = gpuHwTraceBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandler, false, 50, "No handler for GpuHwTrace record type %u", recordType);

            return (this->*recordHandler)(gpuHwTraceBuilderData, recordLocation, correlationId, pMajorPayload, maxMajorPayloadSize);
        }

        bool AddGpuGrCtxTraceRecordVersionInfoRecord(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_GrCtxTrace_RecordVersionInfo_MINOR_VERSION_A, false, 50, "Invalid GpuGrCtxTrace RecordVersionInfo record");

            gpuGrCtxTraceBuilderData.recordVersionInfo = *reinterpret_cast<const NvTrace_GPU_GrCtxTrace_RecordVersionInfo*>(pPayload);
            const NvTrace_GPU_GrCtxTrace_RecordVersionInfo& recordVersionInfo = gpuGrCtxTraceBuilderData.recordVersionInfo;

            // Register handlers within NvTrace_GPU_GrCtxTrace_RecordVersionInfo_MINOR_VERSION_A.
            gpuGrCtxTraceBuilderData.recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_ProviderConfiguration] = &DataModelBuilderBase::AddGpuGrCtxTraceProviderConfigurationRecord;

            if (NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A <= recordVersionInfo.providerDiagnosticInitializationFailureMinorVersion)
            {
                gpuGrCtxTraceBuilderData.recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_ProviderDiagnostic_InitializationFailure] = &DataModelBuilderBase::AddGpuGrCtxTraceProviderDiagnosticInitializationFailureRecordA;
            }

            if (NvTrace_GPU_GrCtxTrace_ContextChange_MINOR_VERSION_A <= recordVersionInfo.contextChangeMinorVersion)
            {
                gpuGrCtxTraceBuilderData.recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_ContextLoadEnd] = &DataModelBuilderBase::AddGpuGrCtxTraceContextLoadEndRecordA;
                gpuGrCtxTraceBuilderData.recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_ContextSaveEnd] = &DataModelBuilderBase::AddGpuGrCtxTraceContextSaveEndRecordA;
            }

            if (NvTrace_GPU_DeviceInfo_MINOR_VERSION_A <= recordVersionInfo.gpuDeviceInfoMinorVersion)
            {
                gpuGrCtxTraceBuilderData.recordHandlers[NvTrace_GPU_GrCtxTrace_RecordType_GpuDeviceInfo] = &DataModelBuilderBase::AddGpuGrCtxTraceGpuDeviceInfoRecordA;
            }

            return true;
        }

        bool AddGpuGrCtxTraceProviderConfigurationRecord(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pPayload, payloadSize).VerifyBuffer<FB::Provider::GpuGrCtxTrace::ProviderConfiguration>(), false, 50, "Invalid GpuGrCtxTrace ProviderConfiguration record");

            const FB::Provider::GpuGrCtxTrace::ProviderConfiguration* const pFbConfig = flatbuffers::GetRoot<FB::Provider::GpuGrCtxTrace::ProviderConfiguration>(pPayload);
            NVTRACE_LOG_ERR_RETURN_IF(!pFbConfig, false, 50, "Invalid GpuGrCtxTrace ProviderConfiguration record");

            GpuGrCtxTraceData& gpuGrCtxTraceData = gpuGrCtxTraceBuilderData.shadowed;
            gpuGrCtxTraceData.circularBufferCellCount = pFbConfig->circularBufferCellCount();
            gpuGrCtxTraceData.updateIntervalNsec = pFbConfig->updateIntervalNsec();
            return true;
        }

        bool AddGpuGrCtxTraceProviderDiagnosticInitializationFailureRecordA(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* pPayload, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A, false, 50, "Invalid GpuGrCtxTrace ProviderDiagnostic_InitializationFailure record");

            const NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure& initializationFailure = *reinterpret_cast<const NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure*>(pPayload);

            GpuGrCtxTraceData::ProviderDiagnostic& providerDiagnostic = gpuGrCtxTraceBuilderData.shadowed.providerDiagnostics.emplace_back();
            providerDiagnostic.timestampTsc = initializationFailure.failureTimestampTsc;
            providerDiagnostic.recordType = NvTrace_GPU_GrCtxTrace_RecordType_ProviderDiagnostic_InitializationFailure;
            providerDiagnostic.initializationFailureReason = initializationFailure.failureReason;
            return true;
        }

        bool AddGpuGrCtxTraceContextLoadEndRecordA(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_GrCtxTrace_ContextChange_MINOR_VERSION_A, false, 50, "Invalid GpuGrCtxTrace ContextLoadEnd record");
            gpuGrCtxTraceBuilderData.grCtxRecords.push_back({recordLocation, NvTrace_GPU_GrCtxTrace_RecordType_ContextLoadEnd, correlationId});
            return true;
        }

        bool AddGpuGrCtxTraceContextSaveEndRecordA(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_GrCtxTrace_ContextChange_MINOR_VERSION_A, false, 50, "Invalid GpuGrCtxTrace ContextSaveEnd record");
            gpuGrCtxTraceBuilderData.grCtxRecords.push_back({recordLocation, NvTrace_GPU_GrCtxTrace_RecordType_ContextSaveEnd, correlationId});
            return true;
        }

        bool AddGpuGrCtxTraceGpuDeviceInfoRecordA(GpuGrCtxTraceBuilderData& /* gpuGrCtxTraceBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const uint8_t* /* pPayload */, size_t payloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(payloadSize < NvTrace_GPU_DeviceInfo_MINOR_VERSION_A, false, 50, "Invalid GpuGrCtxTrace GpuDeviceInfo record");
            return true;
        }

        bool AddGpuGrCtxTraceRecord(GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(NvTrace_GPU_GrCtxTrace_RecordType_Count <= recordType, false, 50, "Unknown GpuGrCtxTrace record type %u", recordType);

            const typename GpuGrCtxTraceBuilderData::RecordHandler& recordHandler = gpuGrCtxTraceBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandler, false, 50, "No handler for GpuGrCtxTrace record type %u", recordType);

            return (this->*recordHandler)(gpuGrCtxTraceBuilderData, recordLocation, correlationId, pMajorPayload, maxMajorPayloadSize);
        }

        static CudaApiTraceData::ApiCallRef LinkCudaApiTraceToEnclosingApiCalls(CudaApiTraceData& cudaApiTraceData, CudaApiTraceData::ThreadData& threadData, CudaApiTraceData::CuStreamData& streamData, uint64_t recordLocation)
        {
            CudaApiTraceData::ApiCallRef apiCallRef;
            apiCallRef.uniqueThreadId = threadData.pThreadData->uniqueThreadId;

            if (!threadData.runtimeApiCalls.empty() && !threadData.runtimeApiCalls.back().endRecordLocation)
            {
                threadData.runtimeApiCalls.back().innerRecordLocations.push_back(recordLocation);
                apiCallRef.runtimeApiCallIdx = threadData.runtimeApiCalls.size() - 1;
            }

            if (!threadData.driverApiCalls.empty() && !threadData.driverApiCalls.back().endRecordLocation)
            {
                threadData.driverApiCalls.back().innerRecordLocations.push_back(recordLocation);
                apiCallRef.driverApiCallIdx = threadData.driverApiCalls.size() - 1;

                if (streamData.apiCallReferences.empty() || streamData.apiCallReferences.back().driverApiCallIdx != apiCallRef.driverApiCallIdx)
                {
                    streamData.apiCallReferences.push_back(apiCallRef);
                }
            }

            return apiCallRef;
        }

        static uint64_t GetApiCallStartTimestampTsc(const CudaApiTraceData& cudaApiTraceData, const CudaApiTraceData::ApiCallRef& apiCallRef)
        {
            auto itrThreadData = cudaApiTraceData.threads.lower_bound(apiCallRef.uniqueThreadId);
            if (itrThreadData == cudaApiTraceData.threads.end() || cudaApiTraceData.threads.key_comp()(apiCallRef.uniqueThreadId, itrThreadData->first))
            {
                return InvalidTimestamp;
            }

            const CudaApiTraceData::ThreadData& threadData = itrThreadData->second;
            if (apiCallRef.driverApiCallIdx != SIZE_MAX && apiCallRef.driverApiCallIdx < threadData.driverApiCalls.size())
            {
                return threadData.driverApiCalls[apiCallRef.driverApiCallIdx].startTimestampTsc;
            }

            if (apiCallRef.runtimeApiCallIdx != SIZE_MAX && apiCallRef.runtimeApiCallIdx < threadData.runtimeApiCalls.size())
            {
                return threadData.runtimeApiCalls[apiCallRef.runtimeApiCallIdx].startTimestampTsc;
            }

            return InvalidTimestamp;
        }

        template <typename TKey>
        static CudaApiTraceData::HostCallbackInstance& FindOrEmplaceHostCallbackInstance(CudaApiTraceBuilderData& cudaApiTraceBuilderData, std::map<TKey, size_t>& keyToHostCallbackInstanceIndex, const TKey& key)
        {
            auto [itr, inserted] = keyToHostCallbackInstanceIndex.try_emplace(key, cudaApiTraceBuilderData.shadowed.hostCallbackInstances.size());
            if (inserted)
            {
                cudaApiTraceBuilderData.shadowed.hostCallbackInstances.emplace_back();
            }
            // Return a reference to the found or newly inserted  HostCallbackInstance.
            return cudaApiTraceBuilderData.shadowed.hostCallbackInstances[itr->second];
        }

        bool AddCudaApiTraceDriverApiEnterRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_CudaDriverApiEnter), false, 50, "Invalid CudaApiTrace DriverApiEnter record");

            const auto* pDriverApiEnterPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_CudaDriverApiEnter*>(pMajorPayload);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;

            CudaApiTraceData::DriverApiCall driverApiCall;
            driverApiCall.driverApiFunctionId = static_cast<FB::Provider::CudaApiTrace::CudaDriverApiFunctionId>(pDriverApiEnterPayload->driverApiFunctionId);
            driverApiCall.startTimestampTsc = pDriverApiEnterPayload->timestampTsc;
            driverApiCall.startRecordLocation = recordLocation;
            threadData.driverApiCalls.push_back(driverApiCall);

            threadBuilderData.currentDriverApiCorrelationId = correlationId;

            return true;
        }

        bool AddCudaApiTraceDriverApiExitRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_CudaDriverApiExit), false, 50, "Invalid CudaApiTrace DriverApiExit record");

            const auto* pDriverApiExitPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_CudaDriverApiExit*>(pMajorPayload);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;

            if (!threadData.driverApiCalls.empty() && !threadData.driverApiCalls.back().endRecordLocation)
            {
                threadData.driverApiCalls.back().endTimestampTsc = pDriverApiExitPayload->timestampTsc;
                threadData.driverApiCalls.back().endRecordLocation = recordLocation;
            }
            else
            {
                CudaApiTraceData::DriverApiCall driverApiCall;
                driverApiCall.driverApiFunctionId = static_cast<FB::Provider::CudaApiTrace::CudaDriverApiFunctionId>(pDriverApiExitPayload->driverApiFunctionId);
                driverApiCall.endTimestampTsc = pDriverApiExitPayload->timestampTsc;
                driverApiCall.endRecordLocation = recordLocation;
                threadData.driverApiCalls.push_back(driverApiCall);
            }

            threadBuilderData.currentDriverApiCorrelationId = InvalidCorrelationId;
            threadBuilderData.currentGraphLaunchCorrelationId = InvalidCorrelationId;
            threadBuilderData.currentMemCommandCorrelationId = InvalidCorrelationId;

            return true;
        }

        bool AddCudaApiTraceRuntimeApiEnterRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter), false, 50, "Invalid CudaApiTrace RuntimeApiEnter record");

            const auto* pRuntimeApiEnterPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter*>(pMajorPayload);

            CudaApiTraceData::ThreadData& threadData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId).shadowed;

            CudaApiTraceData::RuntimeApiCall runtimeApiCall;
            runtimeApiCall.runtimeApiFunctionId = static_cast<FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId>(pRuntimeApiEnterPayload->runtimeApiFunctionId);
            runtimeApiCall.startTimestampTsc = pRuntimeApiEnterPayload->timestampTsc;
            runtimeApiCall.startRecordLocation = recordLocation;
            threadData.runtimeApiCalls.push_back(runtimeApiCall);

            return true;
        }

        bool AddCudaApiTraceRuntimeApiExitRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit), false, 50, "Invalid CudaApiTrace RuntimeApiExit record");

            const auto* pRuntimeApiExitPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit*>(pMajorPayload);

            CudaApiTraceData::ThreadData& threadData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId).shadowed;

            if (!threadData.runtimeApiCalls.empty() && !threadData.runtimeApiCalls.back().endRecordLocation)
            {
                threadData.runtimeApiCalls.back().endTimestampTsc = pRuntimeApiExitPayload->timestampTsc;
                threadData.runtimeApiCalls.back().endRecordLocation = recordLocation;
            }
            else
            {
                typename CudaApiTraceData::RuntimeApiCall runtimeApiCall;
                runtimeApiCall.runtimeApiFunctionId = static_cast<FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId>(pRuntimeApiExitPayload->runtimeApiFunctionId);
                runtimeApiCall.endTimestampTsc = pRuntimeApiExitPayload->timestampTsc;
                runtimeApiCall.endRecordLocation = recordLocation;
                threadData.runtimeApiCalls.push_back(runtimeApiCall);
            }

            return true;
        }

        bool AddCudaApiTraceImmediateKernelLaunchRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch), false, 50, "Invalid CudaApiTrace ImmediateKernelLaunch record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pImmediateKernelLaunchPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pImmediateKernelLaunchPayload->custreamId);
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaFunction, pImmediateKernelLaunchPayload->cufunctionId);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pImmediateKernelLaunchPayload->custreamId);

            const uint32_t cudaDeviceOrdinal = pImmediateKernelLaunchPayload->cudaDeviceOrdinal;
            streamData.cudaDeviceOrdinal = cudaDeviceOrdinal;
            CudaApiTraceData::CuDeviceData* const pDeviceData = cudaApiTraceBuilderData.EnsureCuDeviceData(cudaDeviceOrdinal);
            if (!pDeviceData)
            {
                return false;
            }

            CudaApiTraceData::CuDeviceData& deviceData = *pDeviceData;
            CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch = deviceData.immediateKernelLaunches.emplace_back();
            immediateKernelLaunch.driverApiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);
            immediateKernelLaunch.apiCorrelationId = correlationId;

            if (pImmediateKernelLaunchPayload->enclosingCorrelationId == threadBuilderData.currentDriverApiCorrelationId)
            {
                immediateKernelLaunch.recordLocation = recordLocation;
            }
            else if (pImmediateKernelLaunchPayload->enclosingCorrelationId == threadBuilderData.currentMemCommandCorrelationId)
            {
                immediateKernelLaunch.recordLocation = threadBuilderData.pendingMemCommandRecordLocation;
            }
            else
            {
                NVTRACE_LOG_INF(90, "Immediate Kernel Launch failed to match its enclosingCorrelationId=%" PRIu64 " with a parent driver API call. CurrentDriverApiCorrelationId=%" PRIu64 ", CurrentMemCommandCorrelationId=%" PRIu64 "", pImmediateKernelLaunchPayload->enclosingCorrelationId, threadBuilderData.currentDriverApiCorrelationId, threadBuilderData.currentMemCommandCorrelationId);
                immediateKernelLaunch.recordLocation = recordLocation;
            }

            return true;
        }

        bool AddCudaApiTraceGraphLaunchRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_GraphLaunch), false, 50, "Invalid CudaApiTrace GraphLaunch record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pGraphLaunchPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphLaunch*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pGraphLaunchPayload->custreamId);
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaGraphExec, pGraphLaunchPayload->cugraphExecId);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pGraphLaunchPayload->custreamId);
            cudaApiTraceBuilderData.EnsureCuGraphExecData(pGraphLaunchPayload->cugraphExecId);

            CudaApiTraceData::ApiCallRef apiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            typename CudaApiTraceBuilderData::GraphLaunch& graphLaunch = cudaApiTraceBuilderData.graphLaunches.emplace_back();
            graphLaunch.apiCorrelationId = correlationId;
            graphLaunch.cugraphExecId = pGraphLaunchPayload->cugraphExecId;
            graphLaunch.custreamId = pGraphLaunchPayload->custreamId;
            graphLaunch.recordLocation = recordLocation;
            graphLaunch.apiCallRef = apiCallRef;
            graphLaunch.startTimestampTsc = GetApiCallStartTimestampTsc(cudaApiTraceData, apiCallRef);

            threadBuilderData.currentGraphLaunchCorrelationId = correlationId;

            return true;
        }

        bool AddCudaApiTraceMemcpyBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_MemcpyBegin_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace MemcpyBegin record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pMemcpyBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemcpyBegin*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pMemcpyBeginPayload->custreamId);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pMemcpyBeginPayload->custreamId);

            CudaApiTraceData::ApiCallRef apiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            threadBuilderData.currentMemCommandCorrelationId = correlationId;
            threadBuilderData.pendingMemCommandRecordLocation = recordLocation;

            if (pMemcpyBeginPayload->gpuTimestampRefStart)
            {
                auto& gpuCommand = cudaApiTraceBuilderData.gpuCommands.emplace_back();
                gpuCommand.commandType = CudaApiTraceData::GpuCommandType::EagerMemcpy;
                gpuCommand.recordLocation = recordLocation;
                gpuCommand.driverApiCallRef = apiCallRef;
            }

            return true;
        }

        bool AddCudaApiTraceMemsetBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_MemsetBegin), false, 50, "Invalid CudaApiTrace MemsetBegin record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pMemsetBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemsetBegin*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pMemsetBeginPayload->custreamId);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pMemsetBeginPayload->custreamId);

            CudaApiTraceData::ApiCallRef apiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            threadBuilderData.currentMemCommandCorrelationId = correlationId;
            threadBuilderData.pendingMemCommandRecordLocation = recordLocation;

            if (pMemsetBeginPayload->gpuTimestampRefStart)
            {
                auto& gpuCommand = cudaApiTraceBuilderData.gpuCommands.emplace_back();
                gpuCommand.commandType = CudaApiTraceData::GpuCommandType::EagerMemset;
                gpuCommand.recordLocation = recordLocation;
                gpuCommand.driverApiCallRef = apiCallRef;
            }

            return true;
        }

        bool AddCudaApiTraceGraphMemcpyBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphMemcpyBegin_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphMemcpyBegin record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pGraphMemcpyBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pGraphMemcpyBeginPayload->custreamId);
            if (pGraphMemcpyBeginPayload->graphNodeCorrelationId)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaGraphNodeCorrelation, pGraphMemcpyBeginPayload->graphNodeCorrelationId);
            }

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pGraphMemcpyBeginPayload->custreamId);

            CudaApiTraceData::ApiCallRef apiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            threadBuilderData.currentMemCommandCorrelationId = correlationId;
            threadBuilderData.pendingMemCommandRecordLocation = recordLocation;

            if (pGraphMemcpyBeginPayload->gpuTimestampRefStart)
            {
                auto& gpuCommand = cudaApiTraceBuilderData.gpuCommands.emplace_back();
                gpuCommand.commandType = CudaApiTraceData::GpuCommandType::GraphMemcpy;
                gpuCommand.recordLocation = recordLocation;
                gpuCommand.graphNodeCorrelationId = pGraphMemcpyBeginPayload->graphNodeCorrelationId;
                gpuCommand.driverApiCallRef = apiCallRef;
            }

            return true;
        }

        bool AddCudaApiTraceGraphMemsetBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < sizeof(NvTrace_CUDA_ApiTrace_GraphMemsetBegin), false, 50, "Invalid CudaApiTrace GraphMemsetBegin record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pGraphMemsetBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphMemsetBegin*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pGraphMemsetBeginPayload->custreamId);
            if (pGraphMemsetBeginPayload->graphNodeCorrelationId)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaGraphNodeCorrelation, pGraphMemsetBeginPayload->graphNodeCorrelationId);
            }

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pGraphMemsetBeginPayload->custreamId);

            CudaApiTraceData::ApiCallRef apiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            threadBuilderData.currentMemCommandCorrelationId = correlationId;
            threadBuilderData.pendingMemCommandRecordLocation = recordLocation;

            if (pGraphMemsetBeginPayload->gpuTimestampRefStart)
            {
                auto& gpuCommand = cudaApiTraceBuilderData.gpuCommands.emplace_back();
                gpuCommand.commandType = CudaApiTraceData::GpuCommandType::GraphMemset;
                gpuCommand.recordLocation = recordLocation;
                gpuCommand.graphNodeCorrelationId = pGraphMemsetBeginPayload->graphNodeCorrelationId;
                gpuCommand.driverApiCallRef = apiCallRef;
            }

            return true;
        }

        bool AddCudaApiTraceMemOpBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_MemOpBegin_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace MemOpBegin record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pMemOpBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemOpBegin*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pMemOpBeginPayload->custreamId);
            if (pMemOpBeginPayload->graphNodeCorrelationId)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaGraphNodeCorrelation, pMemOpBeginPayload->graphNodeCorrelationId);
            }

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pMemOpBeginPayload->custreamId);

            CudaApiTraceData::ApiCallRef apiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            if (pMemOpBeginPayload->gpuTimestampRefStart)
            {
                auto& gpuCommand = cudaApiTraceBuilderData.gpuCommands.emplace_back();
                gpuCommand.commandType = CudaApiTraceData::GpuCommandType::MemOp;
                gpuCommand.recordLocation = recordLocation;
                if (pMemOpBeginPayload->graphNodeCorrelationId != 0)
                {
                    gpuCommand.graphNodeCorrelationId = pMemOpBeginPayload->graphNodeCorrelationId;
                }
                gpuCommand.driverApiCallRef = apiCallRef;
            }

            return true;
        }

        bool AddCudaApiTraceHostCallbackLaunchRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_HostCallbackLaunch_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace HostCallbackLaunch record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pHostCallbackLaunchPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_HostCallbackLaunch*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pHostCallbackLaunchPayload->custreamId);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pHostCallbackLaunchPayload->custreamId);

            const uint64_t key = correlationId;
            CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = FindOrEmplaceHostCallbackInstance(cudaApiTraceBuilderData, cudaApiTraceBuilderData.launchApiCorrelationIdToHostCallbackInstanceIndex, key);
            hostCallbackInstance.kind = CudaApiTraceData::HostCallbackKind::Eager;
            hostCallbackInstance.custreamId = pHostCallbackLaunchPayload->custreamId;
            hostCallbackInstance.callback = pHostCallbackLaunchPayload->callback;
            hostCallbackInstance.userData = pHostCallbackLaunchPayload->userData;
            hostCallbackInstance.launchApiCorrelationId = key;

            hostCallbackInstance.driverApiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);

            return true;
        }

        bool AddCudaApiTraceHostCallbackExecBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /*recordLocation*/, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_HostCallbackExecBegin_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace HostCallbackExecBegin record");

            const auto* pHostCallbackExecBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_HostCallbackExecBegin*>(pMajorPayload);

            // Callback runs on a different thread with no enclosing CUDA API call, so no call to LinkCudaApiTraceToEnclosingApiCalls.
            cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);

            const uint64_t key = pHostCallbackExecBeginPayload->launchApiCorrelationId;
            CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = FindOrEmplaceHostCallbackInstance(cudaApiTraceBuilderData, cudaApiTraceBuilderData.launchApiCorrelationIdToHostCallbackInstanceIndex, key);
            hostCallbackInstance.kind = CudaApiTraceData::HostCallbackKind::Eager;
            hostCallbackInstance.custreamId = pHostCallbackExecBeginPayload->custreamId;
            hostCallbackInstance.launchApiCorrelationId = key;
            hostCallbackInstance.startTimestampTsc = pHostCallbackExecBeginPayload->timestampTsc;
            hostCallbackInstance.enclosingCorrelationId = pHostCallbackExecBeginPayload->enclosingCorrelationId;
            hostCallbackInstance.executingUniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;

            return true;
        }

        bool AddCudaApiTraceHostCallbackExecEndRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /*recordLocation*/, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_HostCallbackExecEnd_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace HostCallbackExecEnd record");

            const auto* pHostCallbackExecEndPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_HostCallbackExecEnd*>(pMajorPayload);

            // Callback runs on a different thread with no enclosing CUDA API call, so no call to LinkCudaApiTraceToEnclosingApiCalls.
            cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);

            const uint64_t key = pHostCallbackExecEndPayload->launchApiCorrelationId;
            CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = FindOrEmplaceHostCallbackInstance(cudaApiTraceBuilderData, cudaApiTraceBuilderData.launchApiCorrelationIdToHostCallbackInstanceIndex, key);
            hostCallbackInstance.kind = CudaApiTraceData::HostCallbackKind::Eager;
            hostCallbackInstance.custreamId = pHostCallbackExecEndPayload->custreamId;
            hostCallbackInstance.launchApiCorrelationId = key;
            hostCallbackInstance.endTimestampTsc = pHostCallbackExecEndPayload->timestampTsc;
            hostCallbackInstance.executingUniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;

            return true;
        }

        bool AddCudaApiTraceGraphHostCallbackLaunchRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphHostCallbackLaunch record");

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* pGraphHostCallbackLaunchPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch*>(pMajorPayload);

            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, pGraphHostCallbackLaunchPayload->custreamId);

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);
            CudaApiTraceData::ThreadData& threadData = threadBuilderData.shadowed;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(pGraphHostCallbackLaunchPayload->custreamId);

            const uint64_t graphNodeCorrelationId = pGraphHostCallbackLaunchPayload->graphNodeCorrelationId;
            uint64_t enclosingCorrelationId = threadBuilderData.currentGraphLaunchCorrelationId;
            if (cudaApiTraceData.recordVersionInfo.graphHostCallbackLaunchMinorVersion >= NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_B)
            {
                NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_B, false, 50, "Invalid CudaApiTrace GraphHostCallbackLaunch version B record");
                enclosingCorrelationId = pGraphHostCallbackLaunchPayload->enclosingCorrelationId;
            }

            if (enclosingCorrelationId != InvalidCorrelationId)
            {
                const typename CudaApiTraceBuilderData::GraphHostCallbackInstanceKey key = {graphNodeCorrelationId, enclosingCorrelationId};
                CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = FindOrEmplaceHostCallbackInstance(cudaApiTraceBuilderData, cudaApiTraceBuilderData.graphHostCallbackInstanceKeyToInstanceIndex, key);
                hostCallbackInstance.kind = CudaApiTraceData::HostCallbackKind::Graph;
                hostCallbackInstance.custreamId = pGraphHostCallbackLaunchPayload->custreamId;
                hostCallbackInstance.graphNodeCorrelationId = graphNodeCorrelationId;
                hostCallbackInstance.enclosingCorrelationId = enclosingCorrelationId;
                hostCallbackInstance.driverApiCallRef = LinkCudaApiTraceToEnclosingApiCalls(cudaApiTraceData, threadData, streamData, recordLocation);
            }

            return true;
        }

        bool AddCudaApiTraceGraphHostCallbackExecBeginRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /*recordLocation*/, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphHostCallbackExecBegin record");

            const auto* pGraphHostCallbackExecBeginPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin*>(pMajorPayload);

            // Callback runs on a different thread with no enclosing CUDA API call, so no call to LinkCudaApiTraceToEnclosingApiCalls.
            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);

            const uint64_t graphNodeCorrelationId = pGraphHostCallbackExecBeginPayload->graphNodeCorrelationId;
            const typename CudaApiTraceBuilderData::GraphHostCallbackInstanceKey key = {graphNodeCorrelationId, pGraphHostCallbackExecBeginPayload->enclosingCorrelationId};
            CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = FindOrEmplaceHostCallbackInstance(cudaApiTraceBuilderData, cudaApiTraceBuilderData.graphHostCallbackInstanceKeyToInstanceIndex, key);
            hostCallbackInstance.kind = CudaApiTraceData::HostCallbackKind::Graph;
            hostCallbackInstance.custreamId = pGraphHostCallbackExecBeginPayload->custreamId;
            hostCallbackInstance.graphNodeCorrelationId = graphNodeCorrelationId;
            hostCallbackInstance.startTimestampTsc = pGraphHostCallbackExecBeginPayload->timestampTsc;
            hostCallbackInstance.enclosingCorrelationId = pGraphHostCallbackExecBeginPayload->enclosingCorrelationId;
            hostCallbackInstance.executingUniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;

            const bool execEndHasEnclosingCorrelationId = cudaApiTraceBuilderData.shadowed.recordVersionInfo.graphHostCallbackExecEndMinorVersion >= NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_B;
            if (!execEndHasEnclosingCorrelationId)
            {
                threadBuilderData.currentGraphHostCallbackInstanceIndex = cudaApiTraceBuilderData.graphHostCallbackInstanceKeyToInstanceIndex.at(key);
                threadBuilderData.currentGraphHostCallbackNodeCorrelationId = graphNodeCorrelationId;
            }

            return true;
        }

        bool AddCudaApiTraceGraphHostCallbackExecEndRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /*recordLocation*/, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphHostCallbackExecEnd record");

            const auto* pGraphHostCallbackExecEndPayload = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd*>(pMajorPayload);

            // Callback runs on a different thread with no enclosing CUDA API call, so no call to LinkCudaApiTraceToEnclosingApiCalls.
            const uint64_t graphNodeCorrelationId = pGraphHostCallbackExecEndPayload->graphNodeCorrelationId;
            if (cudaApiTraceBuilderData.shadowed.recordVersionInfo.graphHostCallbackExecEndMinorVersion >= NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_B)
            {
                const typename CudaApiTraceBuilderData::GraphHostCallbackInstanceKey key = {graphNodeCorrelationId, pGraphHostCallbackExecEndPayload->enclosingCorrelationId};
                CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = FindOrEmplaceHostCallbackInstance(cudaApiTraceBuilderData, cudaApiTraceBuilderData.graphHostCallbackInstanceKeyToInstanceIndex, key);
                hostCallbackInstance.kind = CudaApiTraceData::HostCallbackKind::Graph;
                hostCallbackInstance.custreamId = pGraphHostCallbackExecEndPayload->custreamId;
                hostCallbackInstance.graphNodeCorrelationId = graphNodeCorrelationId;
                hostCallbackInstance.enclosingCorrelationId = pGraphHostCallbackExecEndPayload->enclosingCorrelationId;
                hostCallbackInstance.endTimestampTsc = pGraphHostCallbackExecEndPayload->timestampTsc;
                hostCallbackInstance.executingUniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;
                return true;
            }

            typename CudaApiTraceBuilderData::ThreadBuilderData& threadBuilderData = cudaApiTraceBuilderData.EnsureThreadData(pThreadEventInlinePayload->uniqueThreadId);

            auto exitGuard = ScopeExitGuard([&]() {
                threadBuilderData.currentGraphHostCallbackInstanceIndex = SIZE_MAX;
                threadBuilderData.currentGraphHostCallbackNodeCorrelationId = InvalidCorrelationId;
            });

            const size_t graphHostCallbackInstanceIndex = threadBuilderData.currentGraphHostCallbackInstanceIndex;

            if (graphHostCallbackInstanceIndex != SIZE_MAX
                && graphHostCallbackInstanceIndex < cudaApiTraceBuilderData.shadowed.hostCallbackInstances.size()
                && threadBuilderData.currentGraphHostCallbackNodeCorrelationId == graphNodeCorrelationId)
            {
                CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = cudaApiTraceBuilderData.shadowed.hostCallbackInstances[graphHostCallbackInstanceIndex];
                hostCallbackInstance.endTimestampTsc = pGraphHostCallbackExecEndPayload->timestampTsc;
            }
            return true;
        }

        bool AddCudaApiTraceConfigurationRecord(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            const bool verified = flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Provider::CudaApiTrace::ProviderConfiguration>();
            NVTRACE_LOG_ERR_RETURN_IF(!verified, false, 50, "Invalid CudaApiTrace Configuration record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            if (cudaApiTraceData.configurationRecordLocation == InvalidRecordLocation)
            {
                cudaApiTraceData.configurationRecordLocation = recordLocation;
            }

            return true;
        }

        bool AddCudaApiTraceDeviceCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace DeviceCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pDeviceCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_DeviceCreated*>(pMajorPayload);

            const uint32_t cudaDeviceOrdinal = pDeviceCreated->cudaDeviceOrdinal;
            typename CudaApiTraceData::CuDeviceData* const pCudaDeviceData = cudaApiTraceBuilderData.EnsureCuDeviceData(cudaDeviceOrdinal);
            if (!pCudaDeviceData)
            {
                return false;
            }

            typename CudaApiTraceData::CuDeviceData& cudaDeviceData = *pCudaDeviceData;
            cudaDeviceData.cudaDeviceOrdinal = cudaDeviceOrdinal;
            cudaDeviceData.cudeviceHandle = pDeviceCreated->cudeviceHandle;
            cudaDeviceData.gpuUuid = pDeviceCreated->gpuUuid;

            return true;
        }

        bool AddCudaApiTraceContextCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace ContextCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pContextCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ContextCreated*>(pMajorPayload);

            const uint64_t cucontextId = pContextCreated->contextId;
            CudaApiTraceData::CuContextData& contextData = cudaApiTraceBuilderData.EnsureCuContextData(cucontextId);
            contextData.creationTimestampTsc = pContextCreated->creationTimestampTsc;
            contextData.destructionTimestampTsc = std::max(contextData.destructionTimestampTsc, pContextCreated->destructionTimestampTsc);
            contextData.cucontextHandle = pContextCreated->cucontextHandle;

            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaContext, cucontextId);

            return true;
        }

        bool AddCudaApiTraceContextDestroyedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace ContextDestroyed record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pContextDestroyed = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ContextDestroyed*>(pMajorPayload);

            const uint64_t cucontextId = pContextDestroyed->contextId;
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaContext, cucontextId);

            CudaApiTraceData::CuContextData& contextData = cudaApiTraceBuilderData.EnsureCuContextData(cucontextId);
            contextData.destructionTimestampTsc = std::max(contextData.destructionTimestampTsc, pContextDestroyed->timestampTsc);

            return true;
        }

        bool AddCudaApiTraceStreamCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace StreamCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pStreamCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_StreamCreated*>(pMajorPayload);

            const uint64_t custreamId = pStreamCreated->streamId;
            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(custreamId);
            streamData.creationTimestampTsc = pStreamCreated->creationTimestampTsc;
            streamData.destructionTimestampTsc = std::max(streamData.destructionTimestampTsc, pStreamCreated->destructionTimestampTsc);
            streamData.custreamHandle = pStreamCreated->custreamHandle;
            streamData.cudaDeviceOrdinal = pStreamCreated->cudaDeviceOrdinal;

            const uint64_t cucontextId = pStreamCreated->contextId;
            if (cucontextId != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaContext, cucontextId);
                CudaApiTraceData::CuContextData& contextData = cudaApiTraceBuilderData.EnsureCuContextData(cucontextId);
                if (std::find(contextData.custreamIds.begin(), contextData.custreamIds.end(), custreamId) == contextData.custreamIds.end())
                {
                    contextData.custreamIds.push_back(custreamId);
                }
            }

            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaStream, custreamId);

            return true;
        }

        bool AddCudaApiTraceStreamDestroyedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace StreamDestroyed record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pStreamDestroyed = reinterpret_cast<const NvTrace_CUDA_ApiTrace_StreamDestroyed*>(pMajorPayload);

            const uint64_t custreamId = pStreamDestroyed->streamId;
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaStream, custreamId);

            CudaApiTraceData::CuStreamData& streamData = cudaApiTraceBuilderData.EnsureCuStreamData(custreamId);
            streamData.destructionTimestampTsc = std::max(streamData.destructionTimestampTsc, pStreamDestroyed->timestampTsc);

            return true;
        }

        bool AddCudaApiTraceModuleLoadedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace ModuleLoaded record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pModuleLoaded = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ModuleLoaded*>(pMajorPayload);

            const uint64_t cumoduleId = pModuleLoaded->moduleId;
            CudaApiTraceData::CuModuleData& moduleData = cudaApiTraceBuilderData.EnsureCuModuleData(cumoduleId);
            moduleData.creationTimestampTsc = pModuleLoaded->creationTimestampTsc;
            moduleData.destructionTimestampTsc = std::max(moduleData.destructionTimestampTsc, pModuleLoaded->destructionTimestampTsc);
            moduleData.cumoduleHandle = pModuleLoaded->cumoduleHandle;
            moduleData.contextId = pModuleLoaded->contextId;
            if (moduleData.contextId != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaContext, moduleData.contextId);
            }

            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaModule, cumoduleId);

            return true;
        }

        bool AddCudaApiTraceModuleUnloadedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace ModuleUnloaded record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pModuleUnloaded = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ModuleUnloaded*>(pMajorPayload);

            const uint64_t cumoduleId = pModuleUnloaded->moduleId;
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaModule, cumoduleId);

            CudaApiTraceData::CuModuleData& moduleData = cudaApiTraceBuilderData.EnsureCuModuleData(cumoduleId);
            moduleData.destructionTimestampTsc = std::max(moduleData.destructionTimestampTsc, pModuleUnloaded->timestampTsc);

            return true;
        }

        bool AddCudaApiTraceFunctionCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace FunctionCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pFunctionCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_FunctionCreated*>(pMajorPayload);

            const uint64_t cumoduleId = pFunctionCreated->moduleId;
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaModule, cumoduleId);

            const uint64_t cufunctionId = pFunctionCreated->cufunctionId;
            CudaApiTraceData::CuFunctionData& functionData = cudaApiTraceBuilderData.EnsureCuFunctionData(cufunctionId);
            functionData.cufunctionHandle = pFunctionCreated->cufunctionHandle;
            functionData.cumoduleId = cumoduleId;

            const size_t nameBytesOffset = cudaApiTraceData.recordVersionInfo.functionCreatedSize;
            const uint32_t totalNameLength = pFunctionCreated->totalNameLength;
            const uint32_t inlineNameSegmentLength = pFunctionCreated->inlineNameSegmentLength;
            const bool inlineNameFits = inlineNameSegmentLength <= totalNameLength && nameBytesOffset <= maxMajorPayloadSize && inlineNameSegmentLength <= maxMajorPayloadSize - nameBytesOffset;
            NVTRACE_LOG_ERR_RETURN_IF(!inlineNameFits, false, 50, "Invalid CudaApiTrace FunctionCreated inline name (size=%zu nameBytesOffset=%zu totalNameLength=%u inlineNameSegmentLength=%u)", maxMajorPayloadSize, nameBytesOffset, totalNameLength, inlineNameSegmentLength);

            const auto itrFunctionName = cudaApiTraceBuilderData.functionNames.try_emplace(cufunctionId).first;
            typename CudaApiTraceBuilderData::FunctionNameBuilderData& functionNameBuilderData = itrFunctionName->second;
            const bool totalAgrees = !functionNameBuilderData.totalKnown || functionNameBuilderData.totalNameLength == totalNameLength;
            NVTRACE_LOG_ERR_RETURN_IF(!totalAgrees, false, 50, "CudaApiTrace FunctionCreated records disagree on the name length (cufunctionId=%" PRIu64 " totalNameLength=%u expectedTotalNameSize=%u)", cufunctionId, totalNameLength, functionNameBuilderData.totalNameLength);

            functionNameBuilderData.totalKnown = true;
            functionNameBuilderData.totalNameLength = totalNameLength;

            // A segment that preceded this record may have grown the name past the total stated here, which
            // no later record can reconcile; the total is recorded first so the finalizer drops the name.
            const size_t assembledNameByteCount = functionData.mangledName.size();
            NVTRACE_LOG_ERR_RETURN_IF(totalNameLength < assembledNameByteCount, false, 50, "CudaApiTrace FunctionNameSegment segments overrun the name (cufunctionId=%" PRIu64 " totalNameLength=%u assembledNameByteCount=%zu)", cufunctionId, totalNameLength, assembledNameByteCount);

            functionData.mangledName.resize(totalNameLength);
            if (inlineNameSegmentLength != 0)
            {
                functionNameBuilderData.segments.try_emplace(0, inlineNameSegmentLength);
                memcpy(&functionData.mangledName[0], pMajorPayload + nameBytesOffset, inlineNameSegmentLength);
            }

            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaFunction, cufunctionId);

            return true;
        }

        bool AddCudaApiTraceFunctionNameRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace FunctionNameSegment record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pFunctionNameSegment = reinterpret_cast<const NvTrace_CUDA_ApiTrace_FunctionNameSegment*>(pMajorPayload);
            const size_t nameBytesOffset = cudaApiTraceBuilderData.shadowed.recordVersionInfo.functionNameSegmentSize;
            const uint64_t cufunctionId = pFunctionNameSegment->cufunctionId;
            const uint32_t nameOffset = pFunctionNameSegment->nameOffset;
            const uint32_t nameSegmentLength = pFunctionNameSegment->nameSegmentLength;
            const bool segmentFitsRecord = nameBytesOffset <= maxMajorPayloadSize && nameSegmentLength <= maxMajorPayloadSize - nameBytesOffset;
            NVTRACE_LOG_ERR_RETURN_IF(!segmentFitsRecord, false, 50, "Invalid CudaApiTrace FunctionNameSegment segment (size=%zu nameBytesOffset=%zu nameOffset=%u nameSegmentLength=%u)", maxMajorPayloadSize, nameBytesOffset, nameOffset, nameSegmentLength);

            // An empty segment carries nothing, and recording it would shadow the segment that does cover
            // its offset.
            if (nameSegmentLength == 0)
            {
                return true;
            }

            const auto itrFunctionName = cudaApiTraceBuilderData.functionNames.try_emplace(cufunctionId).first;
            typename CudaApiTraceBuilderData::FunctionNameBuilderData& functionNameBuilderData = itrFunctionName->second;
            const uint32_t totalNameLength = functionNameBuilderData.totalNameLength;
            const bool segmentFitsName = !functionNameBuilderData.totalKnown || (nameOffset <= totalNameLength && nameSegmentLength <= totalNameLength - nameOffset);
            NVTRACE_LOG_ERR_RETURN_IF(!segmentFitsName, false, 50, "CudaApiTrace FunctionNameSegment segment falls outside the name (cufunctionId=%" PRIu64 " nameOffset=%u nameSegmentLength=%u totalNameLength=%u)", cufunctionId, nameOffset, nameSegmentLength, totalNameLength);

            CudaApiTraceData::CuFunctionData& functionData = cudaApiTraceBuilderData.EnsureCuFunctionData(cufunctionId);

            // The FunctionCreated record that states the total may still be ahead, so the name grows to hold
            // whatever a segment claims and the finalizer settles the coverage against the total.
            const size_t segmentEndOffset = static_cast<size_t>(nameOffset) + nameSegmentLength;
            if (functionData.mangledName.size() < segmentEndOffset)
            {
                functionData.mangledName.resize(segmentEndOffset);
            }

            functionNameBuilderData.segments.try_emplace(nameOffset, nameSegmentLength);
            memcpy(&functionData.mangledName[nameOffset], pMajorPayload + nameBytesOffset, nameSegmentLength);

            // The function is deliberately not defined here: reference chasing walks backwards, so a
            // segment can precede its FunctionCreated, and defining the function on a segment would stop
            // the chase from pulling in the rest of the name.
            return true;
        }

        bool AddCudaApiTraceGraphExecCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphExecCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pGraphExecCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphExecCreated*>(pMajorPayload);

            const uint64_t cugraphExecId = pGraphExecCreated->cugraphExecId;
            CudaApiTraceData::CuGraphExecData& graphExecData = cudaApiTraceBuilderData.EnsureCuGraphExecData(cugraphExecId);
            graphExecData.creationTimestampTsc = pGraphExecCreated->creationTimestampTsc;
            graphExecData.destructionTimestampTsc = std::max(graphExecData.destructionTimestampTsc, pGraphExecCreated->destructionTimestampTsc);
            graphExecData.cugraphExecHandle = pGraphExecCreated->cugraphExecHandle;

            const uint32_t deviceGraphLaunchHwCorrelationId = (static_cast<uint32_t>(correlationId) & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK) | NVTRACE_CUDA_HW_CORRELATION_ID_GRAPH_LAUNCH_DEVICE_MSBS;
            cudaApiTraceBuilderData.graphExecIdsByDeviceGraphLaunchHwCorrelationId.emplace(deviceGraphLaunchHwCorrelationId, cugraphExecId);

            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaGraphExec, cugraphExecId);

            return true;
        }

        bool AddCudaApiTraceGraphExecDestroyedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphExecDestroyed record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pGraphExecDestroyed = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphExecDestroyed*>(pMajorPayload);

            const uint64_t cugraphExecId = pGraphExecDestroyed->cugraphExecId;
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaGraphExec, cugraphExecId);

            CudaApiTraceData::CuGraphExecData& graphExecData = cudaApiTraceBuilderData.EnsureCuGraphExecData(cugraphExecId);
            graphExecData.destructionTimestampTsc = std::max(graphExecData.destructionTimestampTsc, pGraphExecDestroyed->timestampTsc);

            return true;
        }

        bool AddCudaApiTraceGraphNodeCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GraphNodeCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pGraphNodeCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphNodeCreated*>(pMajorPayload);

            typename CudaApiTraceData::CuGraphNodeData* const pGraphNodeData = &cudaApiTraceBuilderData.EnsureCuGraphNodeData(correlationId);
            const bool wasDefined = pGraphNodeData->recordLocation != InvalidRecordLocation;
            pGraphNodeData->creationTimestampTsc = pGraphNodeCreated->creationTimestampTsc;
            pGraphNodeData->destructionTimestampTsc = std::max(pGraphNodeData->destructionTimestampTsc, pGraphNodeCreated->destructionTimestampTsc);
            pGraphNodeData->cugraphNodeId = pGraphNodeCreated->cugraphNodeId;
            pGraphNodeData->cugraphNodeHandle = pGraphNodeCreated->cugraphNodeHandle;
            pGraphNodeData->cugraphExecId = pGraphNodeCreated->cugraphExecId;
            pGraphNodeData->recordLocation = recordLocation;
            if (pGraphNodeData->cugraphExecId != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaGraphExec, pGraphNodeData->cugraphExecId);
            }

            const NvTrace_RevisionedRef nvtxTraceStackGroupRef = pGraphNodeCreated->nvtxTraceStackGroupRef;
            if (nvtxTraceStackGroupRef != NVTRACE_INVALID_REVISIONED_REF)
            {
                cudaApiTraceBuilderData.graphNodeCorrelationIdToNvtxTraceStackGroupRef[pGraphNodeData->graphNodeCorrelationId] = nvtxTraceStackGroupRef;
                if (cudaApiTraceBuilderData.nvtxTraceProviderLuid != NVTRACE_NIL_LUID)
                {
                    static_cast<TDerived*>(this)->OnObjectReferenced(
                        cudaApiTraceBuilderData.nvtxTraceProviderLuid,
                        ObjectType::NvtxTraceStackGroup,
                        nvtxTraceStackGroupRef);
                }
                else
                {
                    cudaApiTraceBuilderData.pendingNvtxTraceStackGroupRefs.push_back(nvtxTraceStackGroupRef);
                }
            }

            const uint32_t cudaDeviceOrdinal = pGraphNodeCreated->cudaDeviceOrdinal;
            typename CudaApiTraceData::CuDeviceData* const pCudaDeviceData = cudaApiTraceBuilderData.EnsureCuDeviceData(cudaDeviceOrdinal);
            if (!pCudaDeviceData)
            {
                return false;
            }

            typename CudaApiTraceData::CuDeviceData& cudaDeviceData = *pCudaDeviceData;
            if (!wasDefined)
            {
                cudaDeviceData.graphNodes.emplace_back(pGraphNodeData);
            }

            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaGraphNode, pGraphNodeData->cugraphNodeId);
            static_cast<TDerived*>(this)->OnObjectDefined(cudaApiTraceData.providerLuid, ObjectType::CudaGraphNodeCorrelation, pGraphNodeData->graphNodeCorrelationId);

            return true;
        }

        bool AddCudaApiTraceChannelCreatedRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace ChannelCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;
            const auto* const pChannelCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ChannelCreated*>(pMajorPayload);

            const uint64_t cucontextId = pChannelCreated->cucontextId;
            static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceData.providerLuid, ObjectType::CudaContext, cucontextId);

            CudaApiTraceData::CuContextData& contextData = cudaApiTraceBuilderData.EnsureCuContextData(cucontextId);
            const uint32_t rmCid = pChannelCreated->rmCid;
            if (std::find(contextData.channelRmCids.begin(), contextData.channelRmCids.end(), rmCid) == contextData.channelRmCids.end())
            {
                contextData.channelRmCids.push_back(rmCid);
            }

            return true;
        }

        bool AddCudaApiTraceGpuTimestampReportRecordA(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A, false, 50, "Invalid CudaApiTrace GpuTimestampReport record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pGpuTimestampReport = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GpuTimestampReport*>(pMajorPayload);

            // A zero ptimer is a pad: the provider zeroes the timestamps of slots it
            // retires unwritten, and the reader may deliver a stale-ref pad verbatim, so
            // joining one would stamp a command with timestamp 0.
            if (pGpuTimestampReport->ptimer == 0)
            {
                return true;
            }

            auto& gpuTimestampReport = cudaApiTraceBuilderData.gpuTimestampReports.emplace_back();
            gpuTimestampReport.gpuTimestampRef = pGpuTimestampReport->reportRef;
            gpuTimestampReport.ptimer = pGpuTimestampReport->ptimer;

            return true;
        }

        bool IgnoreCudaApiTraceRecord(CudaApiTraceBuilderData& /* cudaApiTraceBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* /* pMajorPayload */, size_t /* maxMajorPayloadSize */)
        {
            return true;
        }

        void PopulateCudaApiTraceRecordHandlers(CudaApiTraceBuilderData& cudaApiTraceBuilderData, const NvTrace_CUDA_ApiTrace_RecordVersionInfo& recordVersionInfo)
        {
            // minimumCompatibleVersion is compared against both the record's own minor version and its
            // wire payload size: registering a handler needs the provider new enough to know the record
            // (minor version) and the record it sent large enough to carry the fields the handler reads
            // (payload size), and every record's forward-compatible floor for both is its _MINOR_VERSION_A macro.
            const auto fnRegisterHandler = [&](NvTrace_CUDA_ApiTrace_RecordType recordType, bool isThreadEvent, uint32_t minorVersion, uint32_t payloadSize, uint32_t minimumCompatibleVersion, typename CudaApiTraceBuilderData::RecordHandler handler) {
                if (minimumCompatibleVersion <= minorVersion && minimumCompatibleVersion <= payloadSize)
                {
                    cudaApiTraceBuilderData.recordHandlers[recordType] = {handler, isThreadEvent};
                }
            };

            // RecordVersionInfo is consumed out-of-band from the ProviderCreated record before this table
            // is built, so a live record of this type carries no further state to add.
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_RecordVersionInfo, false, 0, 0, 0, &DataModelBuilderBase::IgnoreCudaApiTraceRecord);
            // The configuration is a FlatBuffer whose own schema carries its versioning, so
            // NvTrace_CUDA_ApiTrace_RecordVersionInfo describes no size or minor version for it.
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_Configuration, false, 0, 0, 0, &DataModelBuilderBase::AddCudaApiTraceConfigurationRecord);
            // ProviderStatistics carries no state the data model tracks.
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ProviderStatistics, false, 0, 0, 0, &DataModelBuilderBase::IgnoreCudaApiTraceRecord);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_DeviceCreated, false, recordVersionInfo.deviceCreatedMinorVersion, recordVersionInfo.deviceCreatedSize, NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceDeviceCreatedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ContextCreated, false, recordVersionInfo.contextCreatedMinorVersion, recordVersionInfo.contextCreatedSize, NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceContextCreatedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed, false, recordVersionInfo.contextDestroyedMinorVersion, recordVersionInfo.contextDestroyedSize, NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceContextDestroyedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_StreamCreated, false, recordVersionInfo.streamCreatedMinorVersion, recordVersionInfo.streamCreatedSize, NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceStreamCreatedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed, false, recordVersionInfo.streamDestroyedMinorVersion, recordVersionInfo.streamDestroyedSize, NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceStreamDestroyedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded, false, recordVersionInfo.moduleLoadedMinorVersion, recordVersionInfo.moduleLoadedSize, NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceModuleLoadedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded, false, recordVersionInfo.moduleUnloadedMinorVersion, recordVersionInfo.moduleUnloadedSize, NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceModuleUnloadedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_FunctionCreated, false, recordVersionInfo.functionCreatedMinorVersion, recordVersionInfo.functionCreatedSize, NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceFunctionCreatedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_FunctionNameSegment, false, recordVersionInfo.functionNameSegmentMinorVersion, recordVersionInfo.functionNameSegmentSize, NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceFunctionNameRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated, false, recordVersionInfo.graphExecCreatedMinorVersion, recordVersionInfo.graphExecCreatedSize, NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphExecCreatedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed, false, recordVersionInfo.graphExecDestroyedMinorVersion, recordVersionInfo.graphExecDestroyedSize, NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphExecDestroyedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated, false, recordVersionInfo.graphNodeCreatedMinorVersion, recordVersionInfo.graphNodeCreatedSize, NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphNodeCreatedRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated, false, recordVersionInfo.channelCreatedMinorVersion, recordVersionInfo.channelCreatedSize, NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceChannelCreatedRecordA);

            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GpuTimestampReport, false, recordVersionInfo.gpuTimestampReportMinorVersion, recordVersionInfo.gpuTimestampReportSize, NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGpuTimestampReportRecordA);

            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_CudaDriverApiEnter, true, recordVersionInfo.cudaDriverApiEnterMinorVersion, recordVersionInfo.cudaDriverApiEnterSize, NvTrace_CUDA_ApiTrace_CudaDriverApiEnter_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceDriverApiEnterRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_CudaDriverApiExit, true, recordVersionInfo.cudaDriverApiExitMinorVersion, recordVersionInfo.cudaDriverApiExitSize, NvTrace_CUDA_ApiTrace_CudaDriverApiExit_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceDriverApiExitRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_CudaRuntimeApiEnter, true, recordVersionInfo.cudaRuntimeApiEnterMinorVersion, recordVersionInfo.cudaRuntimeApiEnterSize, NvTrace_CUDA_ApiTrace_CudaRuntimeApiEnter_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceRuntimeApiEnterRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_CudaRuntimeApiExit, true, recordVersionInfo.cudaRuntimeApiExitMinorVersion, recordVersionInfo.cudaRuntimeApiExitSize, NvTrace_CUDA_ApiTrace_CudaRuntimeApiExit_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceRuntimeApiExitRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch, true, recordVersionInfo.immediateKernelLaunchMinorVersion, recordVersionInfo.immediateKernelLaunchSize, NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceImmediateKernelLaunchRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphLaunch, true, recordVersionInfo.graphLaunchMinorVersion, recordVersionInfo.graphLaunchSize, NvTrace_CUDA_ApiTrace_GraphLaunch_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphLaunchRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin, true, recordVersionInfo.memcpyBeginMinorVersion, recordVersionInfo.memcpyBeginSize, NvTrace_CUDA_ApiTrace_MemcpyBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceMemcpyBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin, true, recordVersionInfo.memsetBeginMinorVersion, recordVersionInfo.memsetBeginSize, NvTrace_CUDA_ApiTrace_MemsetBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceMemsetBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyBegin, true, recordVersionInfo.graphMemcpyBeginMinorVersion, recordVersionInfo.graphMemcpyBeginSize, NvTrace_CUDA_ApiTrace_GraphMemcpyBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphMemcpyBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetBegin, true, recordVersionInfo.graphMemsetBeginMinorVersion, recordVersionInfo.graphMemsetBeginSize, NvTrace_CUDA_ApiTrace_GraphMemsetBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphMemsetBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_MemOpBegin, true, recordVersionInfo.memOpBeginMinorVersion, recordVersionInfo.memOpBeginSize, NvTrace_CUDA_ApiTrace_MemOpBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceMemOpBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_HostCallbackLaunch, true, recordVersionInfo.hostCallbackLaunchMinorVersion, recordVersionInfo.hostCallbackLaunchSize, NvTrace_CUDA_ApiTrace_HostCallbackLaunch_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceHostCallbackLaunchRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_HostCallbackExecBegin, true, recordVersionInfo.hostCallbackExecBeginMinorVersion, recordVersionInfo.hostCallbackExecBeginSize, NvTrace_CUDA_ApiTrace_HostCallbackExecBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceHostCallbackExecBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_HostCallbackExecEnd, true, recordVersionInfo.hostCallbackExecEndMinorVersion, recordVersionInfo.hostCallbackExecEndSize, NvTrace_CUDA_ApiTrace_HostCallbackExecEnd_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceHostCallbackExecEndRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackLaunch, true, recordVersionInfo.graphHostCallbackLaunchMinorVersion, recordVersionInfo.graphHostCallbackLaunchSize, NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphHostCallbackLaunchRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackExecBegin, true, recordVersionInfo.graphHostCallbackExecBeginMinorVersion, recordVersionInfo.graphHostCallbackExecBeginSize, NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphHostCallbackExecBeginRecordA);
            fnRegisterHandler(NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackExecEnd, true, recordVersionInfo.graphHostCallbackExecEndMinorVersion, recordVersionInfo.graphHostCallbackExecEndSize, NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd_MINOR_VERSION_A, &DataModelBuilderBase::AddCudaApiTraceGraphHostCallbackExecEndRecordA);
        }

        /// A name travels as its FunctionCreated record plus independent FunctionNameSegment records, so
        /// an observation window cut short would otherwise leave a partial name that reads as a plausible
        /// but wrong symbol.
        void FinalizeCudaApiTraceFunctions(CudaApiTraceBuilderData& cudaApiTraceBuilderData)
        {
            CudaApiTraceData& cudaApiTraceData = cudaApiTraceBuilderData.shadowed;

            for (auto& [cufunctionId, functionData] : cudaApiTraceData.functions)
            {
                const auto itrFunctionName = cudaApiTraceBuilderData.functionNames.find(cufunctionId);
                const bool isNameComplete = [&] {
                    if (itrFunctionName == cudaApiTraceBuilderData.functionNames.end() || !itrFunctionName->second.totalKnown)
                    {
                        return false;
                    }
                    const typename CudaApiTraceBuilderData::FunctionNameBuilderData& functionNameBuilderData = itrFunctionName->second;

                    // The segments are keyed by offset, so the name is whole exactly when each one
                    // starts where the previous ended and the last ends at the total.
                    uint64_t nextSegmentOffset = 0;
                    for (const auto& [segmentOffset, segmentLength] : functionNameBuilderData.segments)
                    {
                        if (segmentOffset != nextSegmentOffset)
                        {
                            return false;
                        }
                        nextSegmentOffset += segmentLength;
                    }

                    return nextSegmentOffset == functionNameBuilderData.totalNameLength;
                }();

                if (!isNameComplete)
                {
                    functionData.mangledName.clear();
                }

                // A function lives exactly as long as its module, so its record carries no timestamps of
                // its own. A module whose record never arrived is not synthesized here.
                const auto itrModuleData = cudaApiTraceData.modules.find(functionData.cumoduleId);
                if (itrModuleData != cudaApiTraceData.modules.end())
                {
                    CudaApiTraceData::CuModuleData& moduleData = itrModuleData->second;
                    if (!functionData.mangledName.empty())
                    {
                        moduleData.cufunctionIds.try_emplace(functionData.mangledName, cufunctionId);
                    }
                    ++moduleData.functionCount;
                    functionData.creationTimestampTsc = moduleData.creationTimestampTsc;
                    functionData.destructionTimestampTsc = moduleData.destructionTimestampTsc;
                }
            }
        }

        /// Reports a CUDA record's object identifiers to the derived builders' reference chasing: the id a
        /// created record defines, and the owner a dependent record hangs off -- a graph node's exec, a name
        /// record's function -- so a referenced owner pulls its dependents along.
        template <typename TCheckObjectIdentifier>
        static bool TryCheckCudaApiTraceObjectIdentifiers(uint32_t recordType, NvTrace_CorrelationId correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize, TCheckObjectIdentifier&& checkObjectIdentifier)
        {
            const auto fnCheckSize = [&](size_t floorSize, const char* pRecordName) {
                NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < floorSize, false, 50, "Invalid CudaApiTrace %s record (payload=%p size=%zu)", pRecordName, static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);
                return true;
            };

            switch (recordType)
            {
                case NvTrace_CUDA_ApiTrace_RecordType_ContextCreated:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A, "ContextCreated");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::CudaContext, reinterpret_cast<const NvTrace_CUDA_ApiTrace_ContextCreated*>(pMajorPayload)->contextId);
                    return true;
                }
                case NvTrace_CUDA_ApiTrace_RecordType_StreamCreated:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A, "StreamCreated");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::CudaStream, reinterpret_cast<const NvTrace_CUDA_ApiTrace_StreamCreated*>(pMajorPayload)->streamId);
                    return true;
                }
                case NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A, "ModuleLoaded");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::CudaModule, reinterpret_cast<const NvTrace_CUDA_ApiTrace_ModuleLoaded*>(pMajorPayload)->moduleId);
                    return true;
                }
                case NvTrace_CUDA_ApiTrace_RecordType_FunctionCreated:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A, "FunctionCreated");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::CudaFunction, reinterpret_cast<const NvTrace_CUDA_ApiTrace_FunctionCreated*>(pMajorPayload)->cufunctionId);
                    return true;
                }
                case NvTrace_CUDA_ApiTrace_RecordType_FunctionNameSegment:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A, "FunctionNameSegment");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::CudaFunction, reinterpret_cast<const NvTrace_CUDA_ApiTrace_FunctionNameSegment*>(pMajorPayload)->cufunctionId);
                    return true;
                }
                case NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A, "GraphExecCreated");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::CudaGraphExec, reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphExecCreated*>(pMajorPayload)->cugraphExecId);
                    return true;
                }
                case NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated:
                {
                    const bool sized = fnCheckSize(NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A, "GraphNodeCreated");
                    if (!sized)
                    {
                        return false;
                    }
                    const auto* const pGraphNodeCreated = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphNodeCreated*>(pMajorPayload);
                    checkObjectIdentifier(ObjectType::CudaGraphNodeCorrelation, correlationId);
                    checkObjectIdentifier(ObjectType::CudaGraphNode, pGraphNodeCreated->cugraphNodeId);
                    checkObjectIdentifier(ObjectType::CudaGraphExec, pGraphNodeCreated->cugraphExecId);
                    return true;
                }
                default:
                    return true;
            }
        }

        bool AddCudaApiTraceRecord(CudaApiTraceBuilderData& cudaApiTraceBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(kCudaApiTraceRecordTypeCount <= recordType, false, 50, "Unknown CudaApiTrace record type %u", recordType);

            const typename CudaApiTraceBuilderData::RecordHandlerEntry& recordHandlerEntry = cudaApiTraceBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandlerEntry.handler, false, 50, "No handler for CudaApiTrace record type %u", recordType);

            const Codec::CudaApiTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload = nullptr;
            if (recordHandlerEntry.isThreadEvent)
            {
                pThreadEventInlinePayload = reinterpret_cast<const Codec::CudaApiTraceEvents::ThreadEventInlinePayload*>(pInlinePayload);
                NVTRACE_LOG_ERR_RETURN_IF(!pThreadEventInlinePayload, false, 50, "CudaApiTrace record type %u is missing its inline payload", recordType);
                NVTRACE_LOG_ERR_RETURN_IF(!cudaApiTraceBuilderData.pThreadEventProviderData, false, 50, "CudaApiTrace record type %u is missing its ThreadEvent provider mapping", recordType);

                static_cast<TDerived*>(this)->OnObjectReferenced(cudaApiTraceBuilderData.shadowed.threadEventProviderLuid, ObjectType::ThreadEventThread, pThreadEventInlinePayload->uniqueThreadId);
            }

            return (this->*recordHandlerEntry.handler)(cudaApiTraceBuilderData, recordLocation, correlationId, pThreadEventInlinePayload, pMajorPayload, maxMajorPayloadSize);
        }

        void IngestNvtxTraceStackRefs(NvtxTraceBuilderData& nvtxTraceBuilderData, NvTrace_RevisionedRef stackGroupRef, const uint8_t* pStackRefs, size_t stackRefCount)
        {
            const size_t stackGroupIndex = nvtxTraceBuilderData.GetOrCreateStackGroupIndex(stackGroupRef);
            const uint32_t stackRefSize = nvtxTraceBuilderData.shadowed.recordVersionInfo.stackRefSize;
            std::map<uint32_t, size_t>& domainIndexToStackRefIndex = nvtxTraceBuilderData.stackGroups[stackGroupIndex].domainIndexToStackRefIndex;

            for (size_t stackRefIndex = 0; stackRefIndex < stackRefCount; stackRefIndex++)
            {
                const auto& stackRef = *reinterpret_cast<const NvTrace_NvtxTrace_StackRef*>(pStackRefs + (stackRefIndex * stackRefSize));

                size_t topPushPopRangeIndex = NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex;
                if (stackRef.topRef != NVTRACE_INVALID_REVISIONED_REF)
                {
                    topPushPopRangeIndex = nvtxTraceBuilderData.GetOrCreatePushPopRangeIndex(stackRef.topRef);
                    static_cast<TDerived*>(this)->OnObjectReferenced(
                        nvtxTraceBuilderData.shadowed.providerLuid,
                        ObjectType::NvtxTracePushPopRange,
                        stackRef.topRef);
                }

                auto itrStackRef = domainIndexToStackRefIndex.lower_bound(stackRef.domainIndex);
                size_t stackRefStorageIndex;
                if (itrStackRef != domainIndexToStackRefIndex.end() && itrStackRef->first == stackRef.domainIndex)
                {
                    stackRefStorageIndex = itrStackRef->second;
                }
                else
                {
                    stackRefStorageIndex = nvtxTraceBuilderData.stackRefs.size();
                    nvtxTraceBuilderData.stackRefs.emplace_back();
                    domainIndexToStackRefIndex.emplace_hint(itrStackRef, stackRef.domainIndex, stackRefStorageIndex);
                }
                nvtxTraceBuilderData.stackRefs[stackRefStorageIndex] = {topPushPopRangeIndex, stackRef.nestingDepth};
            }
        }

        bool AddNvtxTraceStringRegisteredRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace StringRegistered record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pStringRegistered = reinterpret_cast<const NvTrace_NvtxTrace_StringRegistered*>(pMajorPayload);
            const uint64_t stringHandle = pStringRegistered->stringHandle;

            typename NvtxTraceBuilderData::StringTableEntry& stringTableEntry = nvtxTraceBuilderData.stringTableEntries[stringHandle];
            stringTableEntry.stringRegisteredRecordLocation = recordLocation;

            static_cast<TDerived*>(this)->OnObjectDefined(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceStringTableEntry, stringHandle);

            return true;
        }

        bool AddNvtxTraceStringSegmentRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace StringSegment record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pStringSegment = reinterpret_cast<const NvTrace_NvtxTrace_StringSegment*>(pMajorPayload);
            const size_t segmentBytesOffset = nvtxTraceBuilderData.shadowed.recordVersionInfo.stringSegmentSize;
            const uint64_t stringHandle = pStringSegment->stringHandle;
            const uint32_t stringOffset = pStringSegment->stringOffset;
            const uint32_t stringSegmentLength = pStringSegment->stringSegmentLength;
            const bool segmentFitsRecord = stringOffset <= UINT32_MAX - stringSegmentLength && segmentBytesOffset <= maxMajorPayloadSize && stringSegmentLength <= maxMajorPayloadSize - segmentBytesOffset;
            NVTRACE_LOG_ERR_RETURN_IF(!segmentFitsRecord, false, 50, "Invalid NvtxTrace StringSegment (size=%zu segmentBytesOffset=%zu stringOffset=%u stringSegmentLength=%u)", maxMajorPayloadSize, segmentBytesOffset, stringOffset, stringSegmentLength);

            nvtxTraceBuilderData.stringTableEntries[stringHandle].segmentRecordLocations[stringOffset] = recordLocation;

            // The string is deliberately not defined here: reference chasing walks backwards, so a chain
            // record can precede the record that created its owner, and defining the owner on a chain
            // record would stop the chase from pulling that record in.
            static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceStringTableEntry, stringHandle);

            return true;
        }

        bool AddNvtxTraceDomainCreatedRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace DomainCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pDomainCreated = reinterpret_cast<const NvTrace_NvtxTrace_DomainCreated*>(pMajorPayload);
            const uint32_t domainIndex = pDomainCreated->domainIndex;
            const uint64_t nameStringHandle = pDomainCreated->nameStringHandle;
            nvtxTraceBuilderData.domains[domainIndex] = {recordLocation, nameStringHandle};

            const NVTRACE_LUID providerLuid = nvtxTraceBuilderData.shadowed.providerLuid;
            static_cast<TDerived*>(this)->OnObjectDefined(providerLuid, ObjectType::NvtxTraceDomain, domainIndex);
            if (nameStringHandle != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTraceStringTableEntry, nameStringHandle);
            }

            return true;
        }

        bool AddNvtxTracePushPopRangeCreatedRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace PushPopRangeCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);
            NVTRACE_LOG_ERR_RETURN_IF(!nvtxTraceBuilderData.pThreadEventProviderData, false, 50, "NvtxTrace PushPopRangeCreated record is missing its ThreadEvent provider mapping");

            const auto& pushPopRangeCreated = *reinterpret_cast<const NvTrace_NvtxTrace_PushPopRangeCreated*>(pMajorPayload);
            const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo = nvtxTraceBuilderData.shadowed.recordVersionInfo;
            const size_t rangeCreatedSize = recordVersionInfo.pushPopRangeCreatedSize;
            const size_t eventAttributesSize = pushPopRangeCreated.hasEventAttributes != 0 ? recordVersionInfo.eventAttributesSize : 0;
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < rangeCreatedSize || maxMajorPayloadSize - rangeCreatedSize < eventAttributesSize, false, 50, "Invalid NvtxTrace PushPopRangeCreated attributes (size=%zu rangeCreatedSize=%zu eventAttributesSize=%zu)", maxMajorPayloadSize, rangeCreatedSize, eventAttributesSize);

            const size_t messageBytesOffset = rangeCreatedSize + eventAttributesSize;
            const bool carriesStringHandle = pushPopRangeCreated.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString;
            NVTRACE_LOG_ERR_RETURN_IF(carriesStringHandle && maxMajorPayloadSize - messageBytesOffset < sizeof(uint64_t), false, 50, "Invalid NvtxTrace PushPopRangeCreated string handle (size=%zu messageBytesOffset=%zu)", maxMajorPayloadSize, messageBytesOffset);

            const uint32_t inlineMessageSegmentLength = static_cast<uint32_t>(std::min<size_t>(pushPopRangeCreated.inlineMessageSegmentLength, maxMajorPayloadSize - messageBytesOffset));

            nvtxTraceBuilderData.pThreadEventProviderData->EnsureThread(pushPopRangeCreated.uniqueThreadId);
            nvtxTraceBuilderData.IngestPushPopRangeState(correlationId, recordLocation, pushPopRangeCreated, pMajorPayload + messageBytesOffset, inlineMessageSegmentLength, m_options.maxNvtxTraceMessageLength);

            const NVTRACE_LUID providerLuid = nvtxTraceBuilderData.shadowed.providerLuid;
            static_cast<TDerived*>(this)->OnObjectDefined(providerLuid, ObjectType::NvtxTracePushPopRange, correlationId);
            if (pushPopRangeCreated.domainIndex != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTraceDomain, pushPopRangeCreated.domainIndex);
            }
            if (pushPopRangeCreated.ancestorRef != NVTRACE_INVALID_REVISIONED_REF)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTracePushPopRange, pushPopRangeCreated.ancestorRef);
            }

            if (carriesStringHandle)
            {
                uint64_t stringHandle;
                memcpy(&stringHandle, pMajorPayload + messageBytesOffset, sizeof(uint64_t));
                if (stringHandle != 0)
                {
                    static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTraceStringTableEntry, stringHandle);
                }
            }

            return true;
        }

        bool AddNvtxTracePushPopRangeMessageSegmentRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace PushPopRangeMessageSegment record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pMessageSegment = reinterpret_cast<const NvTrace_NvtxTrace_PushPopRangeMessageSegment*>(pMajorPayload);
            const size_t segmentBytesOffset = nvtxTraceBuilderData.shadowed.recordVersionInfo.pushPopRangeMessageSegmentSize;
            const bool segmentFitsRecord = segmentBytesOffset <= maxMajorPayloadSize && pMessageSegment->messageSegmentLength <= maxMajorPayloadSize - segmentBytesOffset;
            NVTRACE_LOG_ERR_RETURN_IF(!segmentFitsRecord, false, 50, "Invalid NvtxTrace PushPopRangeMessageSegment segment (size=%zu segmentBytesOffset=%zu messageOffset=%u messageSegmentLength=%u)", maxMajorPayloadSize, segmentBytesOffset, pMessageSegment->messageOffset, pMessageSegment->messageSegmentLength);

            typename NvtxTraceBuilderData::PushPopRange& range = nvtxTraceBuilderData.pushPopRanges[nvtxTraceBuilderData.GetOrCreatePushPopRangeIndex(pMessageSegment->pushPopRangeRef)];
            NVTRACE_LOG_ERR_RETURN_IF(range.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString, false, 50, "A registered-string push-pop range cannot carry message segments");

            if (range.recordLocation == InvalidRecordLocation)
            {
                // Created may be in an older dependency block; retain bytes without re-entering the reader.
                const size_t pendingLength = static_cast<size_t>(std::min<uint64_t>(uint64_t(pMessageSegment->messageOffset) + pMessageSegment->messageSegmentLength, m_options.maxNvtxTraceMessageLength));
                if (range.message.size() < pendingLength && pMessageSegment->messageSegmentLength != 0)
                {
                    range.message.resize(pendingLength, '\0');
                }
            }
            NvtxTraceBuilderData::PlaceIntoMessage(range.message, pMessageSegment->messageOffset, pMajorPayload + segmentBytesOffset, pMessageSegment->messageSegmentLength);

            static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTracePushPopRange, pMessageSegment->pushPopRangeRef);

            return true;
        }

        // A mark's segments ride the thread event buffer rather than a state table entry's chain, so
        // one can outlive the mark it names; an orphan is dropped rather than creating a mark.
        bool AddNvtxTraceMarkMessageSegmentRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_MarkMessageSegment_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace MarkMessageSegment record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto* const pMarkMessageSegment = reinterpret_cast<const NvTrace_NvtxTrace_MarkMessageSegment*>(pMajorPayload);
            const size_t segmentBytesOffset = nvtxTraceBuilderData.shadowed.recordVersionInfo.markMessageSegmentSize;
            const bool segmentFitsRecord = segmentBytesOffset <= maxMajorPayloadSize && pMarkMessageSegment->messageSegmentLength <= maxMajorPayloadSize - segmentBytesOffset;
            NVTRACE_LOG_ERR_RETURN_IF(!segmentFitsRecord, false, 50, "Invalid NvtxTrace MarkMessageSegment segment (size=%zu segmentBytesOffset=%zu messageOffset=%u messageSegmentLength=%u)", maxMajorPayloadSize, segmentBytesOffset, pMarkMessageSegment->messageOffset, pMarkMessageSegment->messageSegmentLength);

            const auto itrMark = nvtxTraceBuilderData.markRefToMarkIndex.find(pMarkMessageSegment->markRef);
            if (itrMark == nvtxTraceBuilderData.markRefToMarkIndex.end())
            {
                return true;
            }

            typename NvtxTraceBuilderData::Mark& mark = nvtxTraceBuilderData.marks[itrMark->second];
            NVTRACE_LOG_ERR_RETURN_IF(mark.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString, false, 50, "A registered-string Mark cannot carry message segments");

            NvtxTraceBuilderData::PlaceIntoMessage(mark.message, pMarkMessageSegment->messageOffset, pMajorPayload + segmentBytesOffset, pMarkMessageSegment->messageSegmentLength);

            return true;
        }

        bool AddNvtxTraceStackGroupCreatedRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t correlationId, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_StackGroupCreated_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace StackGroupCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo = nvtxTraceBuilderData.shadowed.recordVersionInfo;
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackRefSize < NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A, false, 50, "NvtxTrace StackRef is too small (size=%u)", recordVersionInfo.stackRefSize);

            const size_t inlineStackRefsOffset = recordVersionInfo.stackGroupCreatedSize;
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < inlineStackRefsOffset, false, 50, "Invalid NvtxTrace StackGroupCreated inline stack refs (size=%zu inlineStackRefsOffset=%zu)", maxMajorPayloadSize, inlineStackRefsOffset);

            static_cast<TDerived*>(this)->OnObjectDefined(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceStackGroup, correlationId);

            const auto* const pStackGroupCreated = reinterpret_cast<const NvTrace_NvtxTrace_StackGroupCreated*>(pMajorPayload);
            const size_t inlineStackRefCapacity = (maxMajorPayloadSize - inlineStackRefsOffset) / recordVersionInfo.stackRefSize;
            const size_t inlineStackRefCount = std::min(static_cast<size_t>(pStackGroupCreated->inlineStackRefCount), inlineStackRefCapacity);
            IngestNvtxTraceStackRefs(nvtxTraceBuilderData, correlationId, pMajorPayload + inlineStackRefsOffset, inlineStackRefCount);

            return true;
        }

        bool AddNvtxTraceStackGroupStackRefsRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace StackGroupStackRefs record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo = nvtxTraceBuilderData.shadowed.recordVersionInfo;
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackRefSize < NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A, false, 50, "NvtxTrace StackRef is too small (size=%u)", recordVersionInfo.stackRefSize);

            const size_t stackRefsOffset = recordVersionInfo.stackGroupStackRefsSize;
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < stackRefsOffset, false, 50, "Invalid NvtxTrace StackGroupStackRefs tail (size=%zu stackRefsOffset=%zu)", maxMajorPayloadSize, stackRefsOffset);

            const auto* const pStackRefsRecord = reinterpret_cast<const NvTrace_NvtxTrace_StackGroupStackRefs*>(pMajorPayload);
            const size_t stackRefCapacity = (maxMajorPayloadSize - stackRefsOffset) / recordVersionInfo.stackRefSize;
            const size_t stackRefCount = std::min(static_cast<size_t>(pStackRefsRecord->stackRefCount), stackRefCapacity);
            IngestNvtxTraceStackRefs(nvtxTraceBuilderData, pStackRefsRecord->stackGroupRef, pMajorPayload + stackRefsOffset, stackRefCount);

            static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceStackGroup, pStackRefsRecord->stackGroupRef);

            return true;
        }

        bool AddNvtxTraceStackGroupDestroyedRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace StackGroupDestroyed record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            // The data model keeps no end time for a stack group; the record only keeps its group alive
            // through reference chasing.
            const auto* const pStackGroupDestroyed = reinterpret_cast<const NvTrace_NvtxTrace_StackGroupDestroyed*>(pMajorPayload);
            static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceStackGroup, pStackGroupDestroyed->stackGroupRef);

            return true;
        }

        bool AddNvtxTraceMarkRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint64_t correlationId, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_Mark_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace Mark record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto& markEvent = *reinterpret_cast<const NvTrace_NvtxTrace_Mark*>(pMajorPayload);
            const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo = nvtxTraceBuilderData.shadowed.recordVersionInfo;
            const size_t eventAttributesOffset = AlignUp(static_cast<size_t>(recordVersionInfo.markSize), alignof(NvTrace_NvtxTrace_EventAttributes));
            const size_t messageBytesOffset = markEvent.hasEventAttributes ? eventAttributesOffset + recordVersionInfo.eventAttributesSize : static_cast<size_t>(recordVersionInfo.markSize);
            const bool carriesStringHandle = markEvent.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString;
            NVTRACE_LOG_ERR_RETURN_IF(carriesStringHandle && (maxMajorPayloadSize < messageBytesOffset || maxMajorPayloadSize - messageBytesOffset < sizeof(uint64_t)), false, 50, "Invalid NvtxTrace Mark string handle (size=%zu messageBytesOffset=%zu)", maxMajorPayloadSize, messageBytesOffset);

            typename NvtxTraceBuilderData::Mark& mark = nvtxTraceBuilderData.marks.emplace_back();
            mark.correlationId = correlationId;
            mark.recordLocation = recordLocation;
            mark.timestampTsc = markEvent.timestampTsc;
            mark.uniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;
            mark.domainIndex = markEvent.domainIndex;
            mark.messageType = markEvent.messageType;
            mark.hasEventAttributes = markEvent.hasEventAttributes;

            nvtxTraceBuilderData.markRefToMarkIndex[correlationId] = nvtxTraceBuilderData.marks.size() - 1;

            if (carriesStringHandle)
            {
                mark.message.assign(reinterpret_cast<const char*>(pMajorPayload + messageBytesOffset), sizeof(uint64_t));
            }
            else if (messageBytesOffset <= maxMajorPayloadSize)
            {
                const uint32_t inlineLength = static_cast<uint32_t>(std::min<size_t>(markEvent.inlineMessageSegmentLength, maxMajorPayloadSize - messageBytesOffset));
                NvtxTraceBuilderData::ReserveMessage(mark.message, markEvent.totalMessageLength, m_options.maxNvtxTraceMessageLength);
                NvtxTraceBuilderData::PlaceIntoMessage(mark.message, 0, pMajorPayload + messageBytesOffset, inlineLength);
            }

            const NVTRACE_LUID providerLuid = nvtxTraceBuilderData.shadowed.providerLuid;
            if (markEvent.domainIndex != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTraceDomain, markEvent.domainIndex);
            }
            if (carriesStringHandle)
            {
                uint64_t stringHandle;
                memcpy(&stringHandle, pMajorPayload + messageBytesOffset, sizeof(uint64_t));
                if (stringHandle != 0)
                {
                    static_cast<TDerived*>(this)->OnObjectReferenced(providerLuid, ObjectType::NvtxTraceStringTableEntry, stringHandle);
                }
            }

            return true;
        }

        bool AddNvtxTraceRangePopRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_RangePop_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace RangePop record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto& rangePopEvent = *reinterpret_cast<const NvTrace_NvtxTrace_RangePop*>(pMajorPayload);
            nvtxTraceBuilderData.IngestRangePop(rangePopEvent.pushPopRangeRef, rangePopEvent.endTimestampTsc);
            static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTracePushPopRange, rangePopEvent.pushPopRangeRef);

            return true;
        }

        bool AddNvtxTraceDroppedRangePushRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_DroppedRangePush_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace DroppedRangePush record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto& droppedRangePushEvent = *reinterpret_cast<const NvTrace_NvtxTrace_DroppedRangePush*>(pMajorPayload);

            typename NvtxTraceBuilderData::DroppedRange& droppedRangePush = nvtxTraceBuilderData.droppedRangePushes.emplace_back();
            droppedRangePush.timestampTsc = droppedRangePushEvent.timestampTsc;
            droppedRangePush.uniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;
            droppedRangePush.domainIndex = droppedRangePushEvent.domainIndex;

            if (droppedRangePushEvent.domainIndex != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceDomain, droppedRangePushEvent.domainIndex);
            }

            return true;
        }

        bool AddNvtxTraceDroppedRangePopRecordA(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_NvtxTrace_DroppedRangePop_MINOR_VERSION_A, false, 50, "Invalid NvtxTrace DroppedRangePop record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

            const auto& droppedRangePopEvent = *reinterpret_cast<const NvTrace_NvtxTrace_DroppedRangePop*>(pMajorPayload);

            typename NvtxTraceBuilderData::DroppedRange& droppedRangePop = nvtxTraceBuilderData.droppedRangePops.emplace_back();
            droppedRangePop.timestampTsc = droppedRangePopEvent.timestampTsc;
            droppedRangePop.uniqueThreadId = pThreadEventInlinePayload->uniqueThreadId;
            droppedRangePop.domainIndex = droppedRangePopEvent.domainIndex;

            if (droppedRangePopEvent.domainIndex != 0)
            {
                static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.providerLuid, ObjectType::NvtxTraceDomain, droppedRangePopEvent.domainIndex);
            }

            return true;
        }

        bool IgnoreNvtxTraceRecord(NvtxTraceBuilderData& /* nvtxTraceBuilderData */, uint64_t /* recordLocation */, uint64_t /* correlationId */, const Codec::NvtxTraceEvents::ThreadEventInlinePayload* /* pThreadEventInlinePayload */, const uint8_t* /* pMajorPayload */, size_t /* maxMajorPayloadSize */)
        {
            return true;
        }

        void PopulateNvtxTraceRecordHandlers(NvtxTraceBuilderData& nvtxTraceBuilderData, const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo)
        {
            // minimumCompatibleVersion is compared against both the record's own minor version and its
            // wire payload size: registering a handler needs the provider new enough to know the record
            // (minor version) and the record it sent large enough to carry the fields the handler reads
            // (payload size).
            const auto fnRegisterHandler = [&](NvTrace_NvtxTrace_RecordType recordType, bool isThreadEvent, uint32_t minorVersion, uint32_t payloadSize, uint32_t minimumCompatibleVersion, typename NvtxTraceBuilderData::RecordHandler handler) {
                if (minimumCompatibleVersion <= minorVersion && minimumCompatibleVersion <= payloadSize)
                {
                    nvtxTraceBuilderData.recordHandlers[recordType] = {handler, isThreadEvent};
                }
            };

            // RecordVersionInfo is consumed out-of-band from the ProviderCreated record before this table
            // is built, so a live record of this type carries no further state to add.
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_RecordVersionInfo, false, 0, 0, 0, &DataModelBuilderBase::IgnoreNvtxTraceRecord);
            // The configuration is a FlatBuffer no part of the NVTX data model reads.
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_Configuration, false, 0, 0, 0, &DataModelBuilderBase::IgnoreNvtxTraceRecord);
            // ProviderStatistics carries no state the data model tracks.
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_ProviderStatistics, false, 0, 0, 0, &DataModelBuilderBase::IgnoreNvtxTraceRecord);

            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_StringRegistered, false, recordVersionInfo.stringRegisteredMinorVersion, recordVersionInfo.stringRegisteredSize, NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceStringRegisteredRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_StringSegment, false, recordVersionInfo.stringSegmentMinorVersion, recordVersionInfo.stringSegmentSize, NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceStringSegmentRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_DomainCreated, false, recordVersionInfo.domainCreatedMinorVersion, recordVersionInfo.domainCreatedSize, NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceDomainCreatedRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_PushPopRangeCreated, false, recordVersionInfo.pushPopRangeCreatedMinorVersion, recordVersionInfo.pushPopRangeCreatedSize, NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTracePushPopRangeCreatedRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_PushPopRangeMessageSegment, false, recordVersionInfo.pushPopRangeMessageSegmentMinorVersion, recordVersionInfo.pushPopRangeMessageSegmentSize, NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTracePushPopRangeMessageSegmentRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_StackGroupCreated, false, recordVersionInfo.stackGroupCreatedMinorVersion, recordVersionInfo.stackGroupCreatedSize, NvTrace_NvtxTrace_StackGroupCreated_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceStackGroupCreatedRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_StackGroupStackRefs, false, recordVersionInfo.stackGroupStackRefsMinorVersion, recordVersionInfo.stackGroupStackRefsSize, NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceStackGroupStackRefsRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_StackGroupDestroyed, false, recordVersionInfo.stackGroupDestroyedMinorVersion, recordVersionInfo.stackGroupDestroyedSize, NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceStackGroupDestroyedRecordA);

            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_Mark, true, recordVersionInfo.markMinorVersion, recordVersionInfo.markSize, NvTrace_NvtxTrace_Mark_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceMarkRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_MarkMessageSegment, true, recordVersionInfo.markMessageSegmentMinorVersion, recordVersionInfo.markMessageSegmentSize, NvTrace_NvtxTrace_MarkMessageSegment_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceMarkMessageSegmentRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_RangePop, true, recordVersionInfo.rangePopMinorVersion, recordVersionInfo.rangePopSize, NvTrace_NvtxTrace_RangePop_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceRangePopRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_DroppedRangePush, true, recordVersionInfo.droppedRangePushMinorVersion, recordVersionInfo.droppedRangePushSize, NvTrace_NvtxTrace_DroppedRangePush_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceDroppedRangePushRecordA);
            fnRegisterHandler(NvTrace_NvtxTrace_RecordType_DroppedRangePop, true, recordVersionInfo.droppedRangePopMinorVersion, recordVersionInfo.droppedRangePopSize, NvTrace_NvtxTrace_DroppedRangePop_MINOR_VERSION_A, &DataModelBuilderBase::AddNvtxTraceDroppedRangePopRecordA);
        }

        /// The id a created record defines, and the owner a dependent record hangs off -- a segment's string,
        /// a chain record's range or stack group -- so a referenced owner pulls its dependents along.
        template <typename TCheckObjectIdentifier>
        static bool TryCheckNvtxTraceObjectIdentifiers(uint32_t recordType, uint64_t correlationId, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize, TCheckObjectIdentifier&& checkObjectIdentifier)
        {
            const auto fnCheckSize = [&](size_t floorSize, const char* pRecordName) {
                NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < floorSize, false, 50, "Invalid NvtxTrace %s record (payload=%p size=%zu)", pRecordName, static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);
                return true;
            };

            switch (recordType)
            {
                case NvTrace_NvtxTrace_RecordType_StringRegistered:
                {
                    const bool sized = fnCheckSize(NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A, "StringRegistered");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::NvtxTraceStringTableEntry, reinterpret_cast<const NvTrace_NvtxTrace_StringRegistered*>(pMajorPayload)->stringHandle);
                    return true;
                }
                case NvTrace_NvtxTrace_RecordType_StringSegment:
                {
                    const bool sized = fnCheckSize(NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A, "StringSegment");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::NvtxTraceStringTableEntry, reinterpret_cast<const NvTrace_NvtxTrace_StringSegment*>(pMajorPayload)->stringHandle);
                    return true;
                }
                case NvTrace_NvtxTrace_RecordType_DomainCreated:
                {
                    const bool sized = fnCheckSize(NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A, "DomainCreated");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::NvtxTraceDomain, reinterpret_cast<const NvTrace_NvtxTrace_DomainCreated*>(pMajorPayload)->domainIndex);
                    return true;
                }
                case NvTrace_NvtxTrace_RecordType_StackGroupDestroyed:
                {
                    const bool sized = fnCheckSize(NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A, "StackGroupDestroyed");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::NvtxTraceStackGroup, reinterpret_cast<const NvTrace_NvtxTrace_StackGroupDestroyed*>(pMajorPayload)->stackGroupRef);
                    return true;
                }

                // A push-pop range and a stack group have no id of their own, so an entry record is
                // identified by the entry ref the wire carries as its correlationId; a chain record
                // carries its own address there instead, and names its owner in its payload.
                case NvTrace_NvtxTrace_RecordType_PushPopRangeCreated:
                    checkObjectIdentifier(ObjectType::NvtxTracePushPopRange, correlationId);
                    return true;

                case NvTrace_NvtxTrace_RecordType_PushPopRangeMessageSegment:
                {
                    const bool sized = fnCheckSize(NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A, "MessageSegment");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::NvtxTracePushPopRange, reinterpret_cast<const NvTrace_NvtxTrace_PushPopRangeMessageSegment*>(pMajorPayload)->pushPopRangeRef);
                    return true;
                }
                case NvTrace_NvtxTrace_RecordType_StackGroupCreated:
                    checkObjectIdentifier(ObjectType::NvtxTraceStackGroup, correlationId);
                    return true;

                case NvTrace_NvtxTrace_RecordType_StackGroupStackRefs:
                {
                    const bool sized = fnCheckSize(NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A, "StackGroupStackRefs");
                    if (!sized)
                    {
                        return false;
                    }
                    checkObjectIdentifier(ObjectType::NvtxTraceStackGroup, reinterpret_cast<const NvTrace_NvtxTrace_StackGroupStackRefs*>(pMajorPayload)->stackGroupRef);
                    return true;
                }

                default:
                    return false;
            }
        }

        bool AddNvtxTraceRecord(NvtxTraceBuilderData& nvtxTraceBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t correlationId, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(kNvtxTraceRecordTypeCount <= recordType, false, 50, "Unknown NvtxTrace record type %u", recordType);

            const typename NvtxTraceBuilderData::RecordHandlerEntry& recordHandlerEntry = nvtxTraceBuilderData.recordHandlers[recordType];
            NVTRACE_LOG_ERR_RETURN_IF(!recordHandlerEntry.handler, false, 50, "No handler for NvtxTrace record type %u", recordType);

            const Codec::NvtxTraceEvents::ThreadEventInlinePayload* pThreadEventInlinePayload = nullptr;
            if (recordHandlerEntry.isThreadEvent)
            {
                pThreadEventInlinePayload = reinterpret_cast<const Codec::NvtxTraceEvents::ThreadEventInlinePayload*>(pInlinePayload);
                NVTRACE_LOG_ERR_RETURN_IF(!pThreadEventInlinePayload, false, 50, "NvtxTrace record type %u is missing its inline payload", recordType);
                NVTRACE_LOG_ERR_RETURN_IF(!nvtxTraceBuilderData.pThreadEventProviderData, false, 50, "NvtxTrace record type %u is missing its ThreadEvent provider mapping", recordType);

                nvtxTraceBuilderData.pThreadEventProviderData->EnsureThread(pThreadEventInlinePayload->uniqueThreadId);
                static_cast<TDerived*>(this)->OnObjectReferenced(nvtxTraceBuilderData.shadowed.threadEventProviderLuid, ObjectType::ThreadEventThread, pThreadEventInlinePayload->uniqueThreadId);
            }

            return (this->*recordHandlerEntry.handler)(nvtxTraceBuilderData, recordLocation, correlationId, pThreadEventInlinePayload, pMajorPayload, maxMajorPayloadSize);
        }

        void BucketNvtxTracePushPopRanges(NvtxTraceBuilderData& nvtxTraceBuilderData)
        {
            for (size_t index = 0; index < nvtxTraceBuilderData.pushPopRanges.size(); index++)
            {
                const auto& range = nvtxTraceBuilderData.pushPopRanges[index];
                const bool rangeMissingBucketKey = range.domainIndex == NvtxTraceBuilderData::PushPopRange::InvalidDomainIndex || !range.uniqueThreadId || !range.nestingDepth;
                NVTRACE_LOG_ERR_CONTINUE_IF(rangeMissingBucketKey, 50, "NvtxTrace finalization: push-pop range %zu cannot be bucketed (uniqueThreadId=%" PRIu64 " domainIndex=%u nestingDepth=%u); skipping", index, range.uniqueThreadId, range.domainIndex, range.nestingDepth);

                auto& bucket = nvtxTraceBuilderData.domainThreadBuckets[range.domainIndex][range.uniqueThreadId];
                if (range.nestingDepth >= bucket.pushPopRangeIndicesAtDepth.size())
                {
                    bucket.pushPopRangeIndicesAtDepth.resize(range.nestingDepth + 1);
                }
                bucket.pushPopRangeIndicesAtDepth[range.nestingDepth].push_back(index);
            }
        }

        void BoundNvtxTracePushPopRangeTimestamps(NvtxTraceBuilderData& nvtxTraceBuilderData)
        {
            for (auto& [domainIndex, threadBuckets] : nvtxTraceBuilderData.domainThreadBuckets)
            {
                for (auto& [uniqueThreadId, bucket] : threadBuckets)
                {
                    for (size_t depth = bucket.pushPopRangeIndicesAtDepth.size(); depth-- > 2;)
                    {
                        for (const size_t index : bucket.pushPopRangeIndicesAtDepth[depth])
                        {
                            const auto& range = nvtxTraceBuilderData.pushPopRanges[index];
                            const size_t ancestorIndex = range.ancestorIndex;
                            if (ancestorIndex == NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex)
                            {
                                continue;
                            }
                            const uint64_t descendantStart = range.startTimestampTsc;
                            const uint64_t descendantEnd = range.endTimestampTsc;
                            auto& ancestor = nvtxTraceBuilderData.pushPopRanges[ancestorIndex];
                            NVTRACE_LOG_ERR_CONTINUE_IF(ancestor.nestingDepth >= depth, 50, "NvtxTrace finalization: ancestor %zu (nestingDepth=%u) not shallower than descendant %zu (nestingDepth=%zu); skipping bound propagation", ancestorIndex, ancestor.nestingDepth, index, depth);
                            NvtxTraceBuilderData::ApproximateAncestorStart(ancestor, descendantStart);
                            NvtxTraceBuilderData::ApproximateAncestorEnd(ancestor, descendantEnd);
                        }
                    }
                }
            }
        }

        void SortNvtxTraceBucketsForParentSearch(NvtxTraceBuilderData& nvtxTraceBuilderData)
        {
            for (auto& [domainIndex, threadBuckets] : nvtxTraceBuilderData.domainThreadBuckets)
            {
                for (auto& [uniqueThreadId, bucket] : threadBuckets)
                {
                    for (std::vector<size_t>& depthIndices : bucket.pushPopRangeIndicesAtDepth)
                    {
                        std::sort(depthIndices.begin(), depthIndices.end(), [&](size_t leftIndex, size_t rightIndex) {
                            const auto& leftRange = nvtxTraceBuilderData.pushPopRanges[leftIndex];
                            const auto& rightRange = nvtxTraceBuilderData.pushPopRanges[rightIndex];
                            if (leftRange.startTimestampTsc != rightRange.startTimestampTsc)
                            {
                                return leftRange.startTimestampTsc < rightRange.startTimestampTsc;
                            }
                            if (leftRange.endTimestampTsc != rightRange.endTimestampTsc)
                            {
                                return leftRange.endTimestampTsc < rightRange.endTimestampTsc;
                            }
                            return leftIndex < rightIndex;
                        });
                    }
                }
            }
        }

        void RepairNvtxTracePushPopRangeStacks(NvtxTraceBuilderData& nvtxTraceBuilderData)
        {
            for (auto& [domainIndex, threadBuckets] : nvtxTraceBuilderData.domainThreadBuckets)
            {
                for (auto& [uniqueThreadId, bucket] : threadBuckets)
                {
                    std::vector<std::vector<size_t>>& pushPopRangeIndicesAtDepth = bucket.pushPopRangeIndicesAtDepth;

                    auto findContainingParent = [&](size_t childIndex, uint16_t targetParentDepth) -> size_t {
                        if (targetParentDepth >= pushPopRangeIndicesAtDepth.size())
                        {
                            return NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex;
                        }
                        const uint64_t childStart = nvtxTraceBuilderData.pushPopRanges[childIndex].startTimestampTsc;
                        const uint64_t childEnd = nvtxTraceBuilderData.pushPopRanges[childIndex].endTimestampTsc;
                        const std::vector<size_t>& candidateIndices = pushPopRangeIndicesAtDepth[targetParentDepth];
                        const auto candidateIt = std::upper_bound(candidateIndices.begin(), candidateIndices.end(), childStart, [&](uint64_t start, size_t candidateIndex) {
                            return start < nvtxTraceBuilderData.pushPopRanges[candidateIndex].startTimestampTsc;
                        });
                        if (candidateIt == candidateIndices.begin())
                        {
                            return NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex;
                        }
                        const size_t candidateIndex = *(candidateIt - 1);
                        const auto& candidate = nvtxTraceBuilderData.pushPopRanges[candidateIndex];
                        if (candidate.startTimestampTsc <= childStart && childEnd <= candidate.endTimestampTsc)
                        {
                            return candidateIndex;
                        }
                        return NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex;
                    };

                    auto synthesizeParent = [&](size_t childIndex, uint16_t targetParentDepth, size_t inheritedAncestorIndex) -> size_t {
                        const uint64_t childStart = nvtxTraceBuilderData.pushPopRanges[childIndex].startTimestampTsc;
                        const uint64_t childEnd = nvtxTraceBuilderData.pushPopRanges[childIndex].endTimestampTsc;
                        const uint64_t childThreadId = nvtxTraceBuilderData.pushPopRanges[childIndex].uniqueThreadId;
                        const uint32_t childDomainIndex = nvtxTraceBuilderData.pushPopRanges[childIndex].domainIndex;

                        // A synthesized parent is named by the child it was synthesized for, so re-ingesting the same
                        // ranges in a later snapshot yields the same ref.
                        uint64_t synthesizedRangeRef = nvtxTraceBuilderData.pushPopRanges[childIndex].rangeRef;
                        synthesizedRangeRef = (synthesizedRangeRef ^ (synthesizedRangeRef >> 30)) * 0xBF58476D1CE4E5B9ull;
                        synthesizedRangeRef = (synthesizedRangeRef ^ (synthesizedRangeRef >> 27)) * 0x94D049BB133111EBull;
                        synthesizedRangeRef = (synthesizedRangeRef ^ (synthesizedRangeRef >> 31)) | NvtxTraceData::SyntheticPushPopRangeRefBit;

                        const size_t synthesizedRangeIndex = nvtxTraceBuilderData.pushPopRanges.size();
                        nvtxTraceBuilderData.pushPopRanges.emplace_back();
                        auto& synthesizedRange = nvtxTraceBuilderData.pushPopRanges.back();
                        synthesizedRange.rangeRef = synthesizedRangeRef;
                        synthesizedRange.isSynthetic = true;
                        synthesizedRange.startTimestampTsc = childStart;
                        synthesizedRange.endTimestampTsc = childEnd;
                        synthesizedRange.uniqueThreadId = childThreadId;
                        synthesizedRange.domainIndex = childDomainIndex;
                        synthesizedRange.nestingDepth = targetParentDepth;
                        synthesizedRange.ancestorIndex = inheritedAncestorIndex;
                        synthesizedRange.approximationMask = NvtxTraceBuilderData::FieldMask_StartTimestamp | NvtxTraceBuilderData::FieldMask_EndTimestamp | NvtxTraceBuilderData::FieldMask_NestingDepth;

                        return synthesizedRangeIndex;
                    };

                    auto repairPushPopRangeStack = [&](size_t startIndex) {
                        size_t currentIndex = startIndex;
                        while (true)
                        {
                            const uint16_t currentDepth = nvtxTraceBuilderData.pushPopRanges[currentIndex].nestingDepth;
                            if (currentDepth <= 1)
                            {
                                break;
                            }
                            const uint16_t targetParentDepth = static_cast<uint16_t>(currentDepth - 1);

                            const uint64_t currentStart = nvtxTraceBuilderData.pushPopRanges[currentIndex].startTimestampTsc;
                            const uint64_t currentEnd = nvtxTraceBuilderData.pushPopRanges[currentIndex].endTimestampTsc;
                            if (currentStart == InvalidTimestamp || currentEnd == InvalidTimestamp)
                            {
                                NVTRACE_LOG_ERR(50, "NvtxTrace finalization: push-pop range %zu at depth %u has unset bounds entering stack repair; skipping", currentIndex, static_cast<uint32_t>(currentDepth));
                                break;
                            }

                            size_t ancestorIndex = nvtxTraceBuilderData.pushPopRanges[currentIndex].ancestorIndex;
                            if (ancestorIndex == NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex)
                            {
                                ancestorIndex = findContainingParent(currentIndex, targetParentDepth);
                                if (ancestorIndex != NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex)
                                {
                                    nvtxTraceBuilderData.pushPopRanges[currentIndex].ancestorIndex = ancestorIndex;
                                    break;
                                }
                                ancestorIndex = synthesizeParent(currentIndex, targetParentDepth, NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex);
                                nvtxTraceBuilderData.pushPopRanges[currentIndex].ancestorIndex = ancestorIndex;
                                currentIndex = ancestorIndex;
                            }
                            else if (nvtxTraceBuilderData.pushPopRanges[ancestorIndex].nestingDepth < targetParentDepth)
                            {
                                ancestorIndex = synthesizeParent(currentIndex, targetParentDepth, ancestorIndex);
                                nvtxTraceBuilderData.pushPopRanges[currentIndex].ancestorIndex = ancestorIndex;
                                currentIndex = ancestorIndex;
                            }
                            else
                            {
                                break;
                            }
                        }
                    };

                    for (size_t depth = 2; depth < pushPopRangeIndicesAtDepth.size(); depth++)
                    {
                        for (const size_t realIndex : pushPopRangeIndicesAtDepth[depth])
                        {
                            repairPushPopRangeStack(realIndex);
                        }
                    }
                }
            }
        }

        void BuildNvtxTracePreorder(NvtxTraceBuilderData& nvtxTraceBuilderData)
        {
            auto orderByStart = [&](size_t leftIndex, size_t rightIndex) {
                const auto& leftRange = nvtxTraceBuilderData.pushPopRanges[leftIndex];
                const auto& rightRange = nvtxTraceBuilderData.pushPopRanges[rightIndex];
                if (leftRange.startTimestampTsc != rightRange.startTimestampTsc)
                {
                    return leftRange.startTimestampTsc < rightRange.startTimestampTsc;
                }
                return leftIndex < rightIndex;
            };

            typename NvtxTraceBuilderData::DomainThreadRangePreorder domainThreadRangeIndices;
            for (size_t index = 0; index < nvtxTraceBuilderData.pushPopRanges.size(); index++)
            {
                const auto& range = nvtxTraceBuilderData.pushPopRanges[index];
                if (range.domainIndex == NvtxTraceBuilderData::PushPopRange::InvalidDomainIndex || !range.uniqueThreadId || !range.nestingDepth)
                {
                    continue;
                }
                domainThreadRangeIndices[range.domainIndex][range.uniqueThreadId].push_back(index);
            }

            for (auto& [domainIndex, threadRanges] : domainThreadRangeIndices)
            {
                for (auto& [uniqueThreadId, rangeIndices] : threadRanges)
                {
                    std::map<size_t, std::vector<size_t>> childIndicesByParentIndex;
                    std::vector<size_t> rootIndices;
                    for (const size_t index : rangeIndices)
                    {
                        const auto& range = nvtxTraceBuilderData.pushPopRanges[index];
                        const size_t ancestorIndex = range.ancestorIndex;
                        if (ancestorIndex == NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex || range.nestingDepth == 1)
                        {
                            rootIndices.push_back(index);
                            continue;
                        }
                        const uint16_t ancestorDepth = nvtxTraceBuilderData.pushPopRanges[ancestorIndex].nestingDepth;
                        const bool stackBroken = ancestorDepth != range.nestingDepth - 1;
                        NVTRACE_LOG_ERR_IF(stackBroken, 50, "NvtxTrace finalization: push-pop range %zu at depth %u has an ancestor at depth %u, expected %u, after stack repair", index, static_cast<uint32_t>(range.nestingDepth), static_cast<uint32_t>(ancestorDepth), static_cast<uint32_t>(range.nestingDepth - 1));
                        childIndicesByParentIndex[ancestorIndex].push_back(index);
                    }

                    for (auto& [parentIndex, childIndices] : childIndicesByParentIndex)
                    {
                        std::sort(childIndices.begin(), childIndices.end(), orderByStart);
                    }
                    std::sort(rootIndices.begin(), rootIndices.end(), orderByStart);

                    std::vector<size_t>& preorderIndices = nvtxTraceBuilderData.domainThreadRangePreorder[domainIndex][uniqueThreadId];
                    std::vector<size_t> traversalStack;
                    for (auto rootReverseIt = rootIndices.rbegin(); rootReverseIt != rootIndices.rend(); ++rootReverseIt)
                    {
                        traversalStack.push_back(*rootReverseIt);
                    }
                    while (!traversalStack.empty())
                    {
                        const size_t index = traversalStack.back();
                        traversalStack.pop_back();
                        preorderIndices.push_back(index);

                        auto childIt = childIndicesByParentIndex.find(index);
                        if (childIt != childIndicesByParentIndex.end())
                        {
                            for (auto childReverseIt = childIt->second.rbegin(); childReverseIt != childIt->second.rend(); ++childReverseIt)
                            {
                                traversalStack.push_back(*childReverseIt);
                            }
                        }
                    }
                }
            }
        }

        void ExtendNvtxTracePreorderOpenRangeEnds(NvtxTraceBuilderData& nvtxTraceBuilderData)
        {
            for (const auto& domainThreadRanges : nvtxTraceBuilderData.domainThreadRangePreorder)
            {
                for (const auto& threadRangeIndices : domainThreadRanges.second)
                {
                    const std::vector<size_t>& rangeIndices = threadRangeIndices.second;
                    if (rangeIndices.empty())
                    {
                        continue;
                    }

                    // Build a stack of the last ranges at all depths.
                    std::vector<size_t> lastRangePath;
                    size_t lastRangeIndex = rangeIndices.back();
                    while (true)
                    {
                        lastRangePath.push_back(lastRangeIndex);
                        const auto& range = nvtxTraceBuilderData.pushPopRanges[lastRangeIndex];
                        if (range.ancestorIndex == NvtxTraceBuilderData::PushPopRange::InvalidPushPopRangeIndex || range.nestingDepth == 1)
                        {
                            break;
                        }
                        lastRangeIndex = range.ancestorIndex;
                    }
                    // For each range in the stack (starting at the root), if the end is approximated, extend to the maximum timestamp.
                    for (auto rangeItr = lastRangePath.rbegin(); rangeItr != lastRangePath.rend(); ++rangeItr)
                    {
                        auto& range = nvtxTraceBuilderData.pushPopRanges[*rangeItr];
                        if (!(range.approximationMask & NvtxTraceBuilderData::FieldMask_EndTimestamp))
                        {
                            break;
                        }
                        range.endTimestampTsc = MaxTimestampTsc;
                    }
                }
            }
        }

        void FinalizeNvtxTraceBuilding(NvtxTraceBuilderData& nvtxTraceBuilderData, NvtxTraceData& nvtxTraceData)
        {
            for (const auto& [stringHandle, stringTableEntry] : nvtxTraceBuilderData.stringTableEntries)
            {
                NvtxTraceData::StringTableEntryData& stringTableEntryData = nvtxTraceData.stringTableEntries[stringHandle];
                stringTableEntryData.stringRegisteredRecordLocation = stringTableEntry.stringRegisteredRecordLocation;
                stringTableEntryData.segmentRecordLocations = stringTableEntry.segmentRecordLocations;
            }
            for (const auto& [domainIndex, domain] : nvtxTraceBuilderData.domains)
            {
                nvtxTraceData.domains[domainIndex].domainCreatedRecordLocation = domain.recordLocation;
            }

            BucketNvtxTracePushPopRanges(nvtxTraceBuilderData);
            BoundNvtxTracePushPopRangeTimestamps(nvtxTraceBuilderData);
            SortNvtxTraceBucketsForParentSearch(nvtxTraceBuilderData);
            RepairNvtxTracePushPopRangeStacks(nvtxTraceBuilderData);
            BuildNvtxTracePreorder(nvtxTraceBuilderData);
            ExtendNvtxTracePreorderOpenRangeEnds(nvtxTraceBuilderData);
            SquashNvtxTraceDomains(nvtxTraceBuilderData, nvtxTraceData);
            BuildNvtxTraceDomainCliqueEvents(nvtxTraceBuilderData, nvtxTraceData);
        }

        struct NvtxTraceDomainCompatibilityMatrix
        {
            std::vector<uint32_t> domainIndices;
            std::vector<bool> upperTriangle;
            std::vector<size_t> domainConnectivity;

            void Reset(std::vector<uint32_t> newDomainIndices)
            {
                domainIndices = std::move(newDomainIndices);
                std::sort(domainIndices.begin(), domainIndices.end());
                domainIndices.erase(std::unique(domainIndices.begin(), domainIndices.end()), domainIndices.end());

                upperTriangle.assign(GetUpperTriangleSize(domainIndices.size()), true);
                domainConnectivity.assign(domainIndices.size(), domainIndices.empty() ? 0 : domainIndices.size() - 1);
            }

            bool AreCompatible(size_t lhsPosition, size_t rhsPosition) const
            {
                size_t upperTriangleIndex = 0;
                return TryGetUpperTriangleIndex(lhsPosition, rhsPosition, upperTriangleIndex) && upperTriangle[upperTriangleIndex];
            }

            size_t GetConnectivity(size_t domainPosition) const
            {
                return domainPosition < domainConnectivity.size() ? domainConnectivity[domainPosition] : 0;
            }

            void SetIncompatible(size_t lhsPosition, size_t rhsPosition)
            {
                size_t upperTriangleIndex = 0;
                if (!TryGetUpperTriangleIndex(lhsPosition, rhsPosition, upperTriangleIndex) || !upperTriangle[upperTriangleIndex])
                {
                    return;
                }

                upperTriangle[upperTriangleIndex] = false;
                if (domainConnectivity[lhsPosition] > 0)
                {
                    domainConnectivity[lhsPosition]--;
                }
                if (domainConnectivity[rhsPosition] > 0)
                {
                    domainConnectivity[rhsPosition]--;
                }
            }

        private:
            static size_t GetUpperTriangleSize(size_t domainCount)
            {
                return domainCount > 1 ? (domainCount * (domainCount - 1)) / 2 : 0;
            }

            bool TryGetUpperTriangleIndex(size_t lhsPosition, size_t rhsPosition, size_t& upperTriangleIndex) const
            {
                if (lhsPosition == rhsPosition)
                {
                    return false;
                }
                if (lhsPosition >= domainIndices.size() || rhsPosition >= domainIndices.size())
                {
                    return false;
                }
                if (lhsPosition > rhsPosition)
                {
                    std::swap(lhsPosition, rhsPosition);
                }

                const size_t domainCount = domainIndices.size();
                upperTriangleIndex = (lhsPosition * domainCount) - ((lhsPosition * (lhsPosition + 1)) / 2) + (rhsPosition - lhsPosition - 1);
                return upperTriangleIndex < upperTriangle.size();
            }
        };

        void BuildNvtxTraceDomainCliques(NvtxTraceData& nvtxTraceData, const std::map<uint32_t, size_t>& pushPopRangeCountByDomain, const std::vector<uint32_t>& unselectedDomainIndices, const NvtxTraceDomainCompatibilityMatrix& compatibilityMatrix)
        {
            const size_t domainCount = compatibilityMatrix.domainIndices.size();

            std::vector<size_t> remainingDomainPositions;
            size_t totalConnectivity = 0;
            remainingDomainPositions.reserve(domainCount);
            for (size_t domainPosition = 0; domainPosition < domainCount; domainPosition++)
            {
                remainingDomainPositions.push_back(domainPosition);
                totalConnectivity += compatibilityMatrix.GetConnectivity(domainPosition);
            }

            std::vector<size_t> adjacencyOffsets(domainCount + 1);
            std::vector<size_t> adjacentDomainPositions(totalConnectivity);
            adjacencyOffsets[0] = 0;
            for (size_t domainPosition = 0; domainPosition < domainCount; domainPosition++)
            {
                size_t curOffset = adjacencyOffsets[domainPosition];
                for (size_t adjacentDomainPosition = 0; adjacentDomainPosition < domainCount; adjacentDomainPosition++)
                {
                    if (compatibilityMatrix.AreCompatible(domainPosition, adjacentDomainPosition))
                    {
                        adjacentDomainPositions[curOffset] = adjacentDomainPosition;
                        curOffset++;
                    }
                }
                adjacencyOffsets[domainPosition + 1] = curOffset;
            }

            std::vector<int64_t> connectivityScores(domainCount);
            for (size_t domainPosition = 0; domainPosition < domainCount; domainPosition++)
            {
                connectivityScores[domainPosition] = static_cast<int64_t>(compatibilityMatrix.domainConnectivity[domainPosition]);
            }
            std::vector<int64_t> localConnectivityScores;

            auto rangeCountForDomain = [&](uint32_t domainIndex) -> size_t {
                auto it = pushPopRangeCountByDomain.find(domainIndex);
                return it != pushPopRangeCountByDomain.end() ? it->second : 0;
            };

            auto candidateDomainComparator = [&](size_t lhsDomainPosition, size_t rhsDomainPosition) {
                const int64_t lhsConnectivity = localConnectivityScores[lhsDomainPosition];
                const int64_t rhsConnectivity = localConnectivityScores[rhsDomainPosition];
                if (lhsConnectivity != rhsConnectivity)
                {
                    return lhsConnectivity < rhsConnectivity;
                }
                const uint32_t lhsDomainIndex = compatibilityMatrix.domainIndices[lhsDomainPosition];
                const uint32_t rhsDomainIndex = compatibilityMatrix.domainIndices[rhsDomainPosition];
                const size_t lhsPushPopRangeCount = rangeCountForDomain(lhsDomainIndex);
                const size_t rhsPushPopRangeCount = rangeCountForDomain(rhsDomainIndex);
                if (lhsPushPopRangeCount != rhsPushPopRangeCount)
                {
                    return lhsPushPopRangeCount < rhsPushPopRangeCount;
                }
                return lhsDomainIndex > rhsDomainIndex;
            };

            nvtxTraceData.domainCliques.clear();
            std::vector<size_t> domainsInConsideration;
            while (!remainingDomainPositions.empty())
            {
                std::vector<uint32_t> cliqueDomainPositions;
                domainsInConsideration = std::move(remainingDomainPositions);
                remainingDomainPositions.clear();
                localConnectivityScores.assign(connectivityScores.begin(), connectivityScores.end());

                while (!domainsInConsideration.empty())
                {
                    auto selectedDomainItr = std::max_element(domainsInConsideration.begin(), domainsInConsideration.end(), candidateDomainComparator);
                    size_t selectedDomainPosition = *selectedDomainItr;

                    if (cliqueDomainPositions.empty() && connectivityScores[selectedDomainPosition] == 0)
                    {
                        for (size_t domainPosition : domainsInConsideration)
                        {
                            NvtxTraceData::DomainCliqueData singletonClique;
                            singletonClique.domainIndices.push_back(compatibilityMatrix.domainIndices[domainPosition]);
                            nvtxTraceData.domainCliques.push_back(std::move(singletonClique));
                        }
                        domainsInConsideration.clear();
                        break;
                    }

                    cliqueDomainPositions.push_back(static_cast<uint32_t>(selectedDomainPosition));
                    *selectedDomainItr = domainsInConsideration.back();
                    domainsInConsideration.pop_back();

                    for (size_t adjacencyIndex = adjacencyOffsets[selectedDomainPosition]; adjacencyIndex < adjacencyOffsets[selectedDomainPosition + 1]; adjacencyIndex++)
                    {
                        connectivityScores[adjacentDomainPositions[adjacencyIndex]]--;
                    }

                    size_t writeIndex = 0;
                    for (size_t readIndex = 0; readIndex < domainsInConsideration.size(); readIndex++)
                    {
                        const size_t candidateDomainPosition = domainsInConsideration[readIndex];
                        if (compatibilityMatrix.AreCompatible(selectedDomainPosition, candidateDomainPosition))
                        {
                            domainsInConsideration[writeIndex++] = candidateDomainPosition;
                            localConnectivityScores[candidateDomainPosition]--;
                        }
                        else
                        {
                            remainingDomainPositions.push_back(candidateDomainPosition);
                            for (size_t adjacencyIndex = adjacencyOffsets[candidateDomainPosition]; adjacencyIndex < adjacencyOffsets[candidateDomainPosition + 1]; adjacencyIndex++)
                            {
                                localConnectivityScores[adjacentDomainPositions[adjacencyIndex]]--;
                            }
                        }
                    }
                    domainsInConsideration.resize(writeIndex);
                }

                if (!cliqueDomainPositions.empty())
                {
                    NvtxTraceData::DomainCliqueData clique;
                    for (uint32_t domainPosition : cliqueDomainPositions)
                    {
                        clique.domainIndices.push_back(compatibilityMatrix.domainIndices[domainPosition]);
                    }
                    std::sort(clique.domainIndices.begin(), clique.domainIndices.end());
                    nvtxTraceData.domainCliques.push_back(std::move(clique));
                }
            }

            for (const uint32_t domainIndex : unselectedDomainIndices)
            {
                NvtxTraceData::DomainCliqueData singletonClique;
                singletonClique.domainIndices.push_back(domainIndex);
                nvtxTraceData.domainCliques.push_back(std::move(singletonClique));
            }

            if (nvtxTraceData.domainCliques.empty())
            {
                NvtxTraceData::DomainCliqueData defaultClique;
                defaultClique.domainIndices.push_back(0);
                nvtxTraceData.domainCliques.push_back(std::move(defaultClique));
            }
        }

        void SquashNvtxTraceDomains(NvtxTraceBuilderData& nvtxTraceBuilderData, NvtxTraceData& nvtxTraceData)
        {
            std::map<uint32_t, size_t> pushPopRangeCountByDomain;
            for (const auto& [domainIndex, threadRanges] : nvtxTraceBuilderData.domainThreadRangePreorder)
            {
                for (const auto& [uniqueThreadId, rangeIndices] : threadRanges)
                {
                    pushPopRangeCountByDomain[domainIndex] += rangeIndices.size();
                }
            }

            std::set<uint32_t> presentDomainIndices;
            presentDomainIndices.insert(0);
            for (const auto& [domainIndex, threadRanges] : nvtxTraceBuilderData.domainThreadRangePreorder)
            {
                presentDomainIndices.insert(domainIndex);
            }
            for (const auto& mark : nvtxTraceBuilderData.marks)
            {
                presentDomainIndices.insert(mark.domainIndex);
            }
            for (const auto& droppedRangePush : nvtxTraceBuilderData.droppedRangePushes)
            {
                presentDomainIndices.insert(droppedRangePush.domainIndex);
            }
            for (const auto& droppedRangePop : nvtxTraceBuilderData.droppedRangePops)
            {
                presentDomainIndices.insert(droppedRangePop.domainIndex);
            }

            std::vector<uint32_t> selectedDomainIndices(presentDomainIndices.begin(), presentDomainIndices.end());
            std::vector<uint32_t> unselectedDomainIndices;
            if (selectedDomainIndices.size() > m_options.maxNvtxDomainCompatibilityMatrixDomainCount)
            {
                std::sort(selectedDomainIndices.begin(), selectedDomainIndices.end(), [&](uint32_t lhsDomainIndex, uint32_t rhsDomainIndex) {
                    const auto itrLhsCount = pushPopRangeCountByDomain.find(lhsDomainIndex);
                    const auto itrRhsCount = pushPopRangeCountByDomain.find(rhsDomainIndex);
                    const size_t lhsCount = itrLhsCount != pushPopRangeCountByDomain.end() ? itrLhsCount->second : 0;
                    const size_t rhsCount = itrRhsCount != pushPopRangeCountByDomain.end() ? itrRhsCount->second : 0;
                    if (lhsCount != rhsCount)
                    {
                        return lhsCount > rhsCount;
                    }
                    return lhsDomainIndex < rhsDomainIndex;
                });
                for (size_t position = m_options.maxNvtxDomainCompatibilityMatrixDomainCount; position < selectedDomainIndices.size(); position++)
                {
                    unselectedDomainIndices.push_back(selectedDomainIndices[position]);
                }
                selectedDomainIndices.resize(m_options.maxNvtxDomainCompatibilityMatrixDomainCount);
            }

            NvtxTraceDomainCompatibilityMatrix compatibilityMatrix;
            compatibilityMatrix.Reset(selectedDomainIndices);

            std::map<uint32_t, uint32_t> domainPositionByIndex;
            for (uint32_t position = 0; position < compatibilityMatrix.domainIndices.size(); position++)
            {
                domainPositionByIndex[compatibilityMatrix.domainIndices[position]] = position;
            }

            std::map<uint64_t, std::vector<std::pair<size_t, uint32_t>>> threadRangeEntries;
            for (const auto& [domainIndex, threadRanges] : nvtxTraceBuilderData.domainThreadRangePreorder)
            {
                auto positionIt = domainPositionByIndex.find(domainIndex);
                if (positionIt == domainPositionByIndex.end())
                {
                    continue;
                }
                const uint32_t domainPosition = positionIt->second;
                for (const auto& [uniqueThreadId, rangeIndices] : threadRanges)
                {
                    for (const size_t rangeIndex : rangeIndices)
                    {
                        threadRangeEntries[uniqueThreadId].push_back({rangeIndex, domainPosition});
                    }
                }
            }

            struct StackEntry
            {
                size_t rangeIndex;
                uint32_t domainPosition;
            };

            struct PopQueueEntry
            {
                size_t rangeIndex;
                uint64_t endTimestampTsc;
                uint32_t domainPosition;
            };

            auto popQueueComparator = [](const PopQueueEntry& lhs, const PopQueueEntry& rhs) { return lhs.endTimestampTsc > rhs.endTimestampTsc; };

            for (auto& [uniqueThreadId, rangeEntries] : threadRangeEntries)
            {
                std::sort(rangeEntries.begin(), rangeEntries.end(), [&](const std::pair<size_t, uint32_t>& lhs, const std::pair<size_t, uint32_t>& rhs) {
                    const auto& lhsRange = nvtxTraceBuilderData.pushPopRanges[lhs.first];
                    const auto& rhsRange = nvtxTraceBuilderData.pushPopRanges[rhs.first];
                    if (lhsRange.startTimestampTsc != rhsRange.startTimestampTsc)
                    {
                        return lhsRange.startTimestampTsc < rhsRange.startTimestampTsc;
                    }
                    if (lhsRange.endTimestampTsc != rhsRange.endTimestampTsc)
                    {
                        return lhsRange.endTimestampTsc > rhsRange.endTimestampTsc;
                    }
                    if (lhsRange.domainIndex != rhsRange.domainIndex)
                    {
                        return lhsRange.domainIndex < rhsRange.domainIndex;
                    }
                    return lhs.first < rhs.first;
                });

                std::vector<StackEntry> simulatedStack;
                std::priority_queue<PopQueueEntry, std::vector<PopQueueEntry>, decltype(popQueueComparator)> popQueue(popQueueComparator);

                auto processPop = [&]() {
                    const PopQueueEntry popQueueEntry = popQueue.top();
                    popQueue.pop();
                    auto simulatedStackItr = std::find_if(simulatedStack.begin(), simulatedStack.end(), [&](const StackEntry& stackEntry) { return stackEntry.rangeIndex == popQueueEntry.rangeIndex; });
                    if (simulatedStackItr == simulatedStack.end())
                    {
                        return;
                    }
                    for (auto conflictingStackItr = simulatedStackItr + 1; conflictingStackItr != simulatedStack.end(); ++conflictingStackItr)
                    {
                        if (popQueueEntry.endTimestampTsc == nvtxTraceBuilderData.pushPopRanges[conflictingStackItr->rangeIndex].endTimestampTsc)
                        {
                            continue;
                        }
                        compatibilityMatrix.SetIncompatible(popQueueEntry.domainPosition, conflictingStackItr->domainPosition);
                    }
                    simulatedStack.erase(simulatedStackItr);
                };

                for (const auto& [rangeIndex, domainPosition] : rangeEntries)
                {
                    const auto& range = nvtxTraceBuilderData.pushPopRanges[rangeIndex];
                    while (!popQueue.empty() && popQueue.top().endTimestampTsc <= range.startTimestampTsc)
                    {
                        processPop();
                    }
                    simulatedStack.push_back({rangeIndex, domainPosition});
                    popQueue.push({rangeIndex, range.endTimestampTsc, domainPosition});
                }
                while (!popQueue.empty())
                {
                    processPop();
                }
            }

            BuildNvtxTraceDomainCliques(nvtxTraceData, pushPopRangeCountByDomain, unselectedDomainIndices, compatibilityMatrix);
        }

        void BuildNvtxTraceDomainCliqueEvents(NvtxTraceBuilderData& nvtxTraceBuilderData, NvtxTraceData& nvtxTraceData)
        {
            std::map<uint32_t, size_t> domainCliqueIndexByDomain;
            for (size_t domainCliqueIndex = 0; domainCliqueIndex < nvtxTraceData.domainCliques.size(); domainCliqueIndex++)
            {
                for (const uint32_t domainIndex : nvtxTraceData.domainCliques[domainCliqueIndex].domainIndices)
                {
                    domainCliqueIndexByDomain[domainIndex] = domainCliqueIndex;
                }
            }

            for (size_t domainCliqueIndex = 0; domainCliqueIndex < nvtxTraceData.domainCliques.size(); domainCliqueIndex++)
            {
                const std::vector<uint32_t> domainClique = nvtxTraceData.domainCliques[domainCliqueIndex].domainIndices;

                std::map<uint64_t, std::vector<std::pair<size_t, uint32_t>>> threadRangeEntries;
                for (const uint32_t domainIndex : domainClique)
                {
                    auto domainRangesItr = nvtxTraceBuilderData.domainThreadRangePreorder.find(domainIndex);
                    if (domainRangesItr == nvtxTraceBuilderData.domainThreadRangePreorder.end())
                    {
                        continue;
                    }
                    for (const auto& [uniqueThreadId, rangeIndices] : domainRangesItr->second)
                    {
                        for (const size_t rangeIndex : rangeIndices)
                        {
                            threadRangeEntries[uniqueThreadId].push_back({rangeIndex, domainIndex});
                        }
                    }
                }

                for (auto& [uniqueThreadId, rangeEntries] : threadRangeEntries)
                {
                    std::sort(rangeEntries.begin(), rangeEntries.end(), [&](const std::pair<size_t, uint32_t>& lhs, const std::pair<size_t, uint32_t>& rhs) {
                        const auto& lhsRange = nvtxTraceBuilderData.pushPopRanges[lhs.first];
                        const auto& rhsRange = nvtxTraceBuilderData.pushPopRanges[rhs.first];
                        if (lhsRange.startTimestampTsc != rhsRange.startTimestampTsc)
                        {
                            return lhsRange.startTimestampTsc < rhsRange.startTimestampTsc;
                        }
                        if (lhsRange.endTimestampTsc != rhsRange.endTimestampTsc)
                        {
                            return lhsRange.endTimestampTsc > rhsRange.endTimestampTsc;
                        }
                        if (lhsRange.nestingDepth != rhsRange.nestingDepth)
                        {
                            return lhsRange.nestingDepth < rhsRange.nestingDepth;
                        }
                        if (lhsRange.domainIndex != rhsRange.domainIndex)
                        {
                            return lhsRange.domainIndex < rhsRange.domainIndex;
                        }
                        return lhs.first < rhs.first;
                    });

                    NvtxTraceData::DomainCliqueThreadData& domainCliqueThreadData = nvtxTraceData.domainCliques[domainCliqueIndex].threads[uniqueThreadId];
                    domainCliqueThreadData.pushPopRanges.reserve(rangeEntries.size());
                    std::vector<size_t> simulatedStack;
                    for (const auto& [rangeIndex, domainIndex] : rangeEntries)
                    {
                        (void)domainIndex;
                        auto& sourceRange = nvtxTraceBuilderData.pushPopRanges[rangeIndex];
                        while (!simulatedStack.empty())
                        {
                            const auto& stackTop = domainCliqueThreadData.pushPopRanges[simulatedStack.back()];
                            if (sourceRange.endTimestampTsc <= stackTop.endTimestampTsc)
                            {
                                break;
                            }
                            simulatedStack.pop_back();
                        }

                        const size_t cliqueParentRangeIndex = simulatedStack.empty() ? NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex : simulatedStack.back();
                        const uint16_t cliqueDepth = static_cast<uint16_t>(simulatedStack.size() + 1);
                        const size_t cliquePushPopRangeIndex = domainCliqueThreadData.pushPopRanges.size();
                        const uint64_t sourceParentRangeRef = sourceRange.ancestorIndex < nvtxTraceBuilderData.pushPopRanges.size()
                                                                ? nvtxTraceBuilderData.pushPopRanges[sourceRange.ancestorIndex].rangeRef
                                                                : 0;

                        NvtxTraceData::PushPopRangeData cliquePushPopRange;
                        cliquePushPopRange.rangeRef = sourceRange.rangeRef;
                        cliquePushPopRange.startTimestampTsc = sourceRange.startTimestampTsc;
                        cliquePushPopRange.endTimestampTsc = sourceRange.endTimestampTsc;
                        cliquePushPopRange.uniqueThreadId = sourceRange.uniqueThreadId;
                        cliquePushPopRange.domainIndex = sourceRange.domainIndex;
                        cliquePushPopRange.nestingDepth = cliqueDepth;
                        cliquePushPopRange.sourceNestingDepth = sourceRange.nestingDepth;
                        cliquePushPopRange.messageType = sourceRange.messageType;
                        cliquePushPopRange.parentRangeRef = sourceParentRangeRef;
                        cliquePushPopRange.parentRangeIndex = cliqueParentRangeIndex;
                        cliquePushPopRange.approximationMask = sourceRange.approximationMask & (NvtxTraceData::FieldMask_StartTimestamp | NvtxTraceData::FieldMask_EndTimestamp);
                        cliquePushPopRange.isSynthetic = sourceRange.isSynthetic;
                        cliquePushPopRange.stateRecordLocation = sourceRange.recordLocation;
                        cliquePushPopRange.message = sourceRange.message;
                        domainCliqueThreadData.pushPopRanges.push_back(std::move(cliquePushPopRange));
                        sourceRange.finalDomainCliqueIndex = domainCliqueIndex;
                        sourceRange.finalPushPopRangeIndex = cliquePushPopRangeIndex;

                        simulatedStack.push_back(cliquePushPopRangeIndex);
                    }
                }
            }

            for (const auto& mark : nvtxTraceBuilderData.marks)
            {
                auto domainCliqueIndexItr = domainCliqueIndexByDomain.find(mark.domainIndex);
                if (domainCliqueIndexItr == domainCliqueIndexByDomain.end())
                {
                    continue;
                }
                NvtxTraceData::DomainCliqueThreadData& domainCliqueThreadData = nvtxTraceData.domainCliques[domainCliqueIndexItr->second].threads[mark.uniqueThreadId];
                NvtxTraceData::MarkData markData;
                markData.correlationId = mark.correlationId;
                markData.timestampTsc = mark.timestampTsc;
                markData.uniqueThreadId = mark.uniqueThreadId;
                markData.domainIndex = mark.domainIndex;
                markData.messageType = mark.messageType;
                markData.recordLocation = mark.recordLocation;
                markData.message = mark.message;
                domainCliqueThreadData.marks.push_back(std::move(markData));
            }

            for (const auto& droppedRangePush : nvtxTraceBuilderData.droppedRangePushes)
            {
                auto domainCliqueIndexItr = domainCliqueIndexByDomain.find(droppedRangePush.domainIndex);
                if (domainCliqueIndexItr == domainCliqueIndexByDomain.end())
                {
                    continue;
                }
                NvtxTraceData::DomainCliqueThreadData& domainCliqueThreadData = nvtxTraceData.domainCliques[domainCliqueIndexItr->second].threads[droppedRangePush.uniqueThreadId];
                domainCliqueThreadData.droppedRangePushes.push_back({droppedRangePush.timestampTsc, droppedRangePush.uniqueThreadId, droppedRangePush.domainIndex});
            }

            for (const auto& droppedRangePop : nvtxTraceBuilderData.droppedRangePops)
            {
                auto domainCliqueIndexItr = domainCliqueIndexByDomain.find(droppedRangePop.domainIndex);
                if (domainCliqueIndexItr == domainCliqueIndexByDomain.end())
                {
                    continue;
                }
                NvtxTraceData::DomainCliqueThreadData& domainCliqueThreadData = nvtxTraceData.domainCliques[domainCliqueIndexItr->second].threads[droppedRangePop.uniqueThreadId];
                domainCliqueThreadData.droppedRangePops.push_back({droppedRangePop.timestampTsc, droppedRangePop.uniqueThreadId, droppedRangePop.domainIndex});
            }

            for (auto& domainClique : nvtxTraceData.domainCliques)
            {
                for (auto& [uniqueThreadId, domainCliqueThreadData] : domainClique.threads)
                {
                    std::stable_sort(domainCliqueThreadData.marks.begin(), domainCliqueThreadData.marks.end(), [](const NvtxTraceData::MarkData& lhs, const NvtxTraceData::MarkData& rhs) { return lhs.timestampTsc < rhs.timestampTsc; });
                    std::stable_sort(domainCliqueThreadData.droppedRangePushes.begin(), domainCliqueThreadData.droppedRangePushes.end(), [](const NvtxTraceData::DroppedRangePushData& lhs, const NvtxTraceData::DroppedRangePushData& rhs) { return lhs.timestampTsc < rhs.timestampTsc; });
                    std::stable_sort(domainCliqueThreadData.droppedRangePops.begin(), domainCliqueThreadData.droppedRangePops.end(), [](const NvtxTraceData::DroppedRangePopData& lhs, const NvtxTraceData::DroppedRangePopData& rhs) { return lhs.timestampTsc < rhs.timestampTsc; });
                }
            }
        }

        void FinalizeNvtxTraceGraphNodeStacks(
            const NvtxTraceBuilderData& nvtxTraceBuilderData,
            CudaApiTraceBuilderData& cudaApiTraceBuilderData,
            NvtxTraceData& nvtxTraceData)
        {
            for (const auto& [graphNodeCorrelationId, stackGroupRef] : cudaApiTraceBuilderData.graphNodeCorrelationIdToNvtxTraceStackGroupRef)
            {
                const auto graphNodeItr = cudaApiTraceBuilderData.shadowed.graphNodes.find(graphNodeCorrelationId);
                const auto stackGroupRefItr = nvtxTraceBuilderData.stackGroupRefToIndex.find(stackGroupRef);
                if (graphNodeItr == cudaApiTraceBuilderData.shadowed.graphNodes.end()
                    || stackGroupRefItr == nvtxTraceBuilderData.stackGroupRefToIndex.end()
                    || stackGroupRefItr->second >= nvtxTraceBuilderData.stackGroups.size())
                {
                    continue;
                }

                CudaApiTraceData::CuGraphNodeData& graphNodeData = graphNodeItr->second;
                const auto& stackGroup = nvtxTraceBuilderData.stackGroups[stackGroupRefItr->second];
                for (const auto& [_, stackRefIndex] : stackGroup.domainIndexToStackRefIndex)
                {
                    if (stackRefIndex >= nvtxTraceBuilderData.stackRefs.size())
                    {
                        continue;
                    }

                    const auto& stackRef = nvtxTraceBuilderData.stackRefs[stackRefIndex];
                    if (stackRef.topPushPopRangeIndex >= nvtxTraceBuilderData.pushPopRanges.size())
                    {
                        continue;
                    }

                    const auto& topPushPopRange = nvtxTraceBuilderData.pushPopRanges[stackRef.topPushPopRangeIndex];
                    if (topPushPopRange.finalDomainCliqueIndex >= nvtxTraceData.domainCliques.size()
                        || topPushPopRange.finalPushPopRangeIndex == NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex)
                    {
                        continue;
                    }

                    auto& cliqueThreads = nvtxTraceData.domainCliques[topPushPopRange.finalDomainCliqueIndex].threads;
                    auto cliqueThreadItr = cliqueThreads.find(topPushPopRange.uniqueThreadId);
                    if (cliqueThreadItr == cliqueThreads.end()
                        || topPushPopRange.finalPushPopRangeIndex >= cliqueThreadItr->second.pushPopRanges.size())
                    {
                        continue;
                    }

                    if (graphNodeData.uniqueThreadId == 0)
                    {
                        graphNodeData.uniqueThreadId = topPushPopRange.uniqueThreadId;
                    }
                    else if (graphNodeData.uniqueThreadId != topPushPopRange.uniqueThreadId)
                    {
                        continue;
                    }

                    // Mark all ranges in the stack as record-time
                    std::vector<NvtxTraceData::PushPopRangeData>& pushPopRanges = cliqueThreadItr->second.pushPopRanges;
                    for (size_t pushPopRangeIndex = topPushPopRange.finalPushPopRangeIndex;
                         pushPopRangeIndex != NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex && pushPopRangeIndex < pushPopRanges.size() && !pushPopRanges[pushPopRangeIndex].isRecordTimeRange;
                         pushPopRangeIndex = pushPopRanges[pushPopRangeIndex].parentRangeIndex)
                    {
                        pushPopRanges[pushPopRangeIndex].isRecordTimeRange = true;
                    }

                    auto [nodeStackItr, inserted] = graphNodeData.nvtxTraceDomainCliquePushPopRanges.try_emplace(
                        static_cast<uint32_t>(topPushPopRange.finalDomainCliqueIndex),
                        topPushPopRange.finalPushPopRangeIndex);
                    // Select the pushPopRange with the highest nesting depth
                    if (!inserted)
                    {
                        const auto& candidateRange = cliqueThreadItr->second.pushPopRanges[topPushPopRange.finalPushPopRangeIndex];
                        const auto& existingRange = cliqueThreadItr->second.pushPopRanges[nodeStackItr->second];
                        if (candidateRange.nestingDepth > existingRange.nestingDepth)
                        {
                            nodeStackItr->second = topPushPopRange.finalPushPopRangeIndex;
                        }
                    }
                }
            }
        }

        bool AddThreadEventRecord(ThreadEventProviderBuilderData& threadEventProviderBuilderData, uint64_t recordLocation, uint32_t recordType, uint64_t /* correlationId */, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize)
        {
            ThreadEventProviderData& threadEventProviderData = threadEventProviderBuilderData.shadowed;

            switch (static_cast<FB::Codec::ThreadLifetimeEvents::EventTypeId>(recordType))
            {
                case FB::Codec::ThreadLifetimeEvents::EventTypeId_Invalid:
                    return false;

                case FB::Codec::ThreadLifetimeEvents::EventTypeId_ThreadCreated:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_ThreadEvent_ThreadCreated_MINOR_VERSION_A, false, 50, "Invalid ThreadEvent ThreadCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);

                    const auto& threadCreated = *reinterpret_cast<const NvTrace_ThreadEvent_ThreadCreated*>(pMajorPayload);
                    ThreadEventProviderData::ThreadData& threadData = threadEventProviderData.EnsureThread(threadCreated.uniqueThreadId);
                    threadData.creationTimestampTsc = threadCreated.creationTimestampTsc;
                    threadData.destructionTimestampTsc = std::min(threadData.destructionTimestampTsc, threadCreated.destructionTimestampTsc != InvalidTimestamp ? threadCreated.destructionTimestampTsc : MaxTimestampTsc);
                    threadData.nativeThreadId = threadCreated.nativeThreadId;

                    static_cast<TDerived*>(this)->OnObjectDefined(threadEventProviderData.providerLuid, ObjectType::ThreadEventThread, threadCreated.uniqueThreadId);
                    return true;
                }

                case FB::Codec::ThreadLifetimeEvents::EventTypeId_ThreadDestroyed:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_ThreadEvent_ThreadDestroyed_MINOR_VERSION_A, false, 50, "Invalid ThreadEvent ThreadDestroyed record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);
                    NVTRACE_LOG_ERR_RETURN_IF(!pInlinePayload, false, 50, "ThreadEvent ThreadDestroyed record is missing its inline payload");

                    const uint64_t uniqueThreadId = reinterpret_cast<const Codec::ThreadLifetimeEvents::ThreadEventInlinePayload*>(pInlinePayload)->uniqueThreadId;
                    static_cast<TDerived*>(this)->OnObjectReferenced(threadEventProviderData.providerLuid, ObjectType::ThreadEventThread, uniqueThreadId);

                    const auto& threadDestroyed = *reinterpret_cast<const NvTrace_ThreadEvent_ThreadDestroyed*>(pMajorPayload);
                    ThreadEventProviderData::ThreadData& threadData = threadEventProviderData.EnsureThread(uniqueThreadId);
                    threadData.destructionTimestampTsc = std::min(threadData.destructionTimestampTsc, threadDestroyed.destructionTimestampTsc);
                    return true;
                }

                case FB::Codec::ThreadLifetimeEvents::EventTypeId_ProviderMapped:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::ThreadLifetimeEvents::ProviderMapped>(), false, 50, "Invalid ThreadEvent ProviderMapped record");
                    const FB::Codec::ThreadLifetimeEvents::ProviderMapped* const pFbProviderMapped = flatbuffers::GetRoot<FB::Codec::ThreadLifetimeEvents::ProviderMapped>(pMajorPayload);

                    const uint32_t providerMappingId = pFbProviderMapped->providerMappingId();
                    const NVTRACE_LUID mappedProviderLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbProviderMapped->mappedProviderLuid());
                    ApplyThreadEventProviderMapping(threadEventProviderBuilderData, providerMappingId, mappedProviderLuid);
                    return true;
                }
            }

            NVTRACE_LOG_ERR(50, "Unknown ThreadEvent event type %u", recordType);
            return false;
        }

        void AssociateCudaWithNvtxTraceEvents(CudaApiTraceData& cudaApiTrace, const NvtxTraceData& nvtxTraceData)
        {
            struct NvtxTraceProjectionDomainCliqueState
            {
                const NvtxTraceData::DomainCliqueThreadData& domainCliqueEvents;
                size_t headIndex = 0;
                size_t tailIndex = NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex;
            };

            for (auto& [uniqueThreadId, cudaThreadData] : cudaApiTrace.threads)
            {
                std::map<uint32_t, NvtxTraceProjectionDomainCliqueState> cliqueStates;
                for (uint32_t domainCliqueIndex = 0; domainCliqueIndex < nvtxTraceData.domainCliques.size(); ++domainCliqueIndex)
                {
                    const auto& cliqueThreads = nvtxTraceData.domainCliques[domainCliqueIndex].threads;
                    const auto nvtxThreadItr = cliqueThreads.find(uniqueThreadId);
                    if (nvtxThreadItr == cliqueThreads.end())
                    {
                        continue;
                    }
                    if (!nvtxThreadItr->second.pushPopRanges.empty())
                    {
                        cliqueStates.emplace(domainCliqueIndex, NvtxTraceProjectionDomainCliqueState{nvtxThreadItr->second});
                    }
                }

                for (auto& driverApiCall : cudaThreadData.driverApiCalls)
                {
                    // A thread cannot push or pop NVTX ranges while it is inside an API call, so the ranges enclosing a
                    // call whose start is unknown are the ones open at its end.
                    const uint64_t apiCallTimestampTsc = driverApiCall.startTimestampTsc != InvalidTimestamp ? driverApiCall.startTimestampTsc : driverApiCall.endTimestampTsc;

                    // Skip API calls that have no records (impossible to have associated GPU events)
                    if (driverApiCall.innerRecordLocations.empty() || apiCallTimestampTsc == InvalidTimestamp)
                    {
                        continue;
                    }

                    for (auto& [domainCliqueIndex, cliqueState] : cliqueStates)
                    {
                        const auto& pushPopRanges = cliqueState.domainCliqueEvents.pushPopRanges;
                        size_t& headIndex = cliqueState.headIndex;
                        size_t& tailIndex = cliqueState.tailIndex;

                        // Check if current head is before the API call
                        if (headIndex < pushPopRanges.size() && pushPopRanges[headIndex].startTimestampTsc <= apiCallTimestampTsc)
                        {
                            // Move head to the first element that is after the API call
                            while (headIndex < pushPopRanges.size() && pushPopRanges[headIndex].startTimestampTsc <= apiCallTimestampTsc)
                            {
                                if (pushPopRanges[headIndex].isObservationTimeRange)
                                {
                                    tailIndex = headIndex;
                                }
                                ++headIndex;
                            }
                        }

                        // Find the parent NVTX clique range that contains the API call
                        while (tailIndex != NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex)
                        {
                            const auto& pushPopRange = pushPopRanges[tailIndex];
                            if (pushPopRange.endTimestampTsc >= apiCallTimestampTsc)
                            {
                                // Found a parent NVTX clique range that contains the API call, link it.
                                driverApiCall.nvtxTraceDomainCliquePushPopRanges[domainCliqueIndex] = tailIndex;
                                break;
                            }
                            tailIndex = pushPopRange.parentRangeIndex;
                        }
                    }
                }
            }
        }

        // GrCtx reconstruction is a non-nested parentheses problem (a load should be followed by a save), where the timestamp are monotonically increasing.
        void ReconstructGrCtxResidency(GpuPartitionBuilderData& gpuPartitionBuilderData)
        {
            GpuPartitionData& gpuPartitionData = gpuPartitionBuilderData.shadowed;

            GpuGrCtxTraceBuilderData& gpuGrCtxTraceBuilderData = gpuPartitionBuilderData.gpuGrCtxTrace;
            GpuGrCtxTraceData& gpuGrCtxTraceData = gpuGrCtxTraceBuilderData.shadowed;

            struct ProcessLifetimeGpuNsec
            {
                uint64_t creation = GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec;
                uint64_t destruction = GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec;
            };
            auto fnGetProcessLifetime = [&](const ProcessData& processData) {
                ProcessLifetimeGpuNsec lifetime;
                if (processData.creationTimestampTsc != MinTimestampTsc)
                {
                    lifetime.creation = m_dataModel.ConvertTscTickToGpuNsec(gpuPartitionData, processData.creationTimestampTsc);
                }
                if (processData.destructionTimestampTsc != MaxTimestampTsc)
                {
                    lifetime.destruction = m_dataModel.ConvertTscTickToGpuNsec(gpuPartitionData, processData.destructionTimestampTsc);
                }
                return lifetime;
            };
            auto fnIsProcessLive = [](const ProcessLifetimeGpuNsec& lifetime, uint64_t timestampGpuNsec) {
                if (timestampGpuNsec == InvalidTimestamp)
                {
                    return true;
                }
                if (lifetime.creation != InvalidTimestamp && timestampGpuNsec < lifetime.creation)
                {
                    return false;
                }
                return lifetime.destruction == InvalidTimestamp || timestampGpuNsec < lifetime.destruction;
            };

            // Owner = matching-PID process whose ProcessCreated/ProcessDestroyed lifetime contains the
            // context load. Prefer the most recently created matching process to disambiguate PID reuse.
            // Prefer a globalPid (host) match; globalPid==0 falls back to namespacePid.
            // TODO: linear scan; a {matchPid, start}-sorted lookup would be O(log n).
            auto fnPickOwnerByPid = [&](uint64_t ctxGlobalPid, uint64_t ctxLoadTimestampGpuNsec, bool useGlobalPid) -> NVTRACE_LUID {
                NVTRACE_LUID best = NVTRACE_NIL_LUID;
                uint64_t bestCreation = InvalidTimestamp;
                for (auto& [processLuid, processData] : m_dataModel.processes)
                {
                    const bool match = useGlobalPid
                                         ? (processData.globalPid != 0 && processData.globalPid == static_cast<int>(ctxGlobalPid))
                                         : (processData.globalPid == 0 && processData.namespacePid == static_cast<int>(ctxGlobalPid));
                    if (!match)
                    {
                        continue;
                    }
                    const ProcessLifetimeGpuNsec lifetime = fnGetProcessLifetime(processData);
                    if (!fnIsProcessLive(lifetime, ctxLoadTimestampGpuNsec))
                    {
                        continue;
                    }
                    if (best == NVTRACE_NIL_LUID || (lifetime.creation != InvalidTimestamp && (bestCreation == InvalidTimestamp || bestCreation <= lifetime.creation)))
                    {
                        best = processLuid;
                        bestCreation = lifetime.creation;
                    }
                }
                return best;
            };

            // A FECS GrCtx event identifies its context by one of the context's compute-channel RM CIDs.
            // The CUDA provider records every channel CID per context (ChannelCreated events), so build a
            // sorted CID -> owning-process table once for this GPU partition and binary-search it per
            // event, instead of scanning every process's contexts for every event.
            // RM allocates CIDs from a single global monotonic 32-bit counter
            // (resman kernel_channel.c: portAtomicIncrementU32(&pSys->currentChannelUniqueId)), so a CID
            // is never reused within a trace for practical purposes -- aliasing requires the counter to
            // wrap (2^32 channel allocations, on the order of a year of continuous churn). If a duplicate
            // does appear, the entry is marked ambiguous and correlation falls back to PID rather than
            // guess between owners.
            struct CidOwner
            {
                uint32_t rmCid;
                bool ambiguous;
                NVTRACE_LUID processLuid;
                ProcessLifetimeGpuNsec processLifetime;
            };
            // Build the sorted CID -> owner table: one entry per (context channel CID, owning process),
            // sorted by CID, with any CID claimed by two processes marked ambiguous in a single pass.
            const std::vector<CidOwner> cidOwners = [&]() {
                std::vector<CidOwner> owners;
                for (auto& [processLuid, processData] : m_dataModel.processes)
                {
                    const auto* pCuDeviceData = m_dataModel.FindCuDeviceByGpuUuid(processData, gpuPartitionData.gpuUuid);
                    if (pCuDeviceData == nullptr)
                    {
                        continue;
                    }
                    const ProcessLifetimeGpuNsec processLifetime = fnGetProcessLifetime(processData);
                    for (const auto& [cucontextId, contextData] : processData.cudaApiTrace.contexts)
                    {
                        for (const uint32_t channelRmCid : contextData.channelRmCids)
                        {
                            owners.push_back({channelRmCid, false, processLuid, processLifetime});
                        }
                    }
                }
                std::sort(owners.begin(), owners.end(), [](const CidOwner& lhs, const CidOwner& rhs) {
                    return lhs.rmCid < rhs.rmCid;
                });
                // If a CID is claimed by more than one process, mark every entry for that CID ambiguous.
                // Checking only adjacent pairs misses the case where one process lists the same CID twice:
                // the first entry stays unmarked, lower_bound() returns it, and the contested CID resolves
                // to that process instead of falling back to PID. Flag the whole equal-CID run.
                size_t ambiguousCidCount = 0;
                for (size_t runStart = 0; runStart < owners.size();)
                {
                    size_t runEnd = runStart + 1;
                    bool multipleProcesses = false;
                    while (runEnd < owners.size() && owners[runEnd].rmCid == owners[runStart].rmCid)
                    {
                        multipleProcesses = multipleProcesses || (owners[runEnd].processLuid != owners[runStart].processLuid);
                        ++runEnd;
                    }
                    if (multipleProcesses)
                    {
                        for (size_t ownerIndex = runStart; ownerIndex < runEnd; ++ownerIndex)
                        {
                            owners[ownerIndex].ambiguous = true;
                        }
                        ++ambiguousCidCount;
                    }
                    runStart = runEnd;
                }
                // Should not happen short of a CID counter wrap (see above). If this fires in the field,
                // it is the signal to add lifetime-window disambiguation ({rmCid, timespan}) to this table.
                NVTRACE_LOG_WRN_IF(0 < ambiguousCidCount, 50, "GPU GR-context correlation: %zu channel CID(s) claimed by multiple processes (CID counter wrap?); their events fall back to PID matching.", ambiguousCidCount);
                return owners;
            }();

            auto fnPickOwnerByRmCid = [&](uint32_t ctxRmCid, uint64_t ctxLoadTimestampGpuNsec) -> NVTRACE_LUID {
                if (ctxRmCid == 0)
                {
                    return NVTRACE_NIL_LUID;
                }
                const auto itrOwner = std::lower_bound(cidOwners.begin(), cidOwners.end(), ctxRmCid, [](const CidOwner& owner, uint32_t cid) {
                    return owner.rmCid < cid;
                });
                if (itrOwner == cidOwners.end() || itrOwner->rmCid != ctxRmCid || itrOwner->ambiguous)
                {
                    return NVTRACE_NIL_LUID;
                }
                if (!fnIsProcessLive(itrOwner->processLifetime, ctxLoadTimestampGpuNsec))
                {
                    return NVTRACE_NIL_LUID;
                }
                return itrOwner->processLuid;
            };

            auto fnFindCtxCorrespondingProcessLuid = [&](uint32_t ctxRmCid, uint64_t ctxGlobalPid, uint64_t ctxLoadTimestampGpuNsec) -> NVTRACE_LUID {
                // Prefer the namespace-independent rmCid match; fall back to host globalPid (SO_PEERCRED),
                // then namespacePid (legacy traces / host / --pid=host).
                NVTRACE_LUID result = fnPickOwnerByRmCid(ctxRmCid, ctxLoadTimestampGpuNsec);
                if (result == NVTRACE_NIL_LUID)
                {
                    result = fnPickOwnerByPid(ctxGlobalPid, ctxLoadTimestampGpuNsec, /*useGlobalPid=*/true);
                }
                if (result == NVTRACE_NIL_LUID)
                {
                    result = fnPickOwnerByPid(ctxGlobalPid, ctxLoadTimestampGpuNsec, /*useGlobalPid=*/false);
                }
                return result;
            };

            if (gpuGrCtxTraceBuilderData.grCtxRecords.empty())
            {
                // corner case: no workload
                // corner case: only one context but both load and save are out of the observation window. That context must be the GPU partition's last useful context.
                for (auto itrProcessData = m_dataModel.processes.rbegin(); itrProcessData != m_dataModel.processes.rend(); ++itrProcessData)
                {
                    ProcessData& processData = itrProcessData->second;

                    const CudaApiTraceData::CuDeviceData* pCuDevice = m_dataModel.FindCuDeviceByGpuUuid(processData, gpuPartitionData.gpuUuid);
                    if (!pCuDevice)
                    {
                        continue;
                    }

                    // fabricate a resident context; grCtx.pid is the host PID (else namespacePid).
                    const int resolvedPid = processData.globalPid != 0 ? processData.globalPid : processData.namespacePid;
                    GpuGrCtxTraceData::ResidentContext grCtx;
                    grCtx.ctxLoadTimestampGpuNsec = GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec;
                    grCtx.ctxSaveTimestampGpuNsec = GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec;
                    grCtx.pid = resolvedPid;
                    grCtx.processLuid = itrProcessData->first;
                    grCtx.isMultipleEventsDropped = true; // so the process will be shown as "??" in the trace to indicate the context is not fully captured.
                    gpuGrCtxTraceData.residentContexts.push_back(grCtx);
                    NVTRACE_LOG_INF(50, "GPU %s: Activities are detected but the GR context event is out of the observation window. Fabricated a resident context with pid = %d", ToString(gpuPartitionData.gpuUuid).c_str(), resolvedPid);
                    break;
                }

                return; // return value might be: (1) no resident or (2) a single resident by guessing the last process in the GPU partition

                // Note: An extreme scenario but might not worth handling:
                // Two contexts in the observation window but
                // 1st context: load is out of the observation window, while save is in the observation window but dropped.
                // 2nd context: load is in the observation window but dropped, while save is out of the observation window.
            }

            auto fnGetContextChange = [&](const typename GpuGrCtxTraceBuilderData::GrCtxRecord& record) -> const NvTrace_GPU_GrCtxTrace_ContextChange& {
                const uint8_t* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(record.recordLocation).pMajorPayload;
                return *reinterpret_cast<const NvTrace_GPU_GrCtxTrace_ContextChange*>(pMajorPayload);
            };

            auto fnIsContextLoadEnd = [](const typename GpuGrCtxTraceBuilderData::GrCtxRecord& record) {
                return record.recordType == NvTrace_GPU_GrCtxTrace_RecordType_ContextLoadEnd;
            };

            auto fnIsContextSaveEnd = [](const typename GpuGrCtxTraceBuilderData::GrCtxRecord& record) {
                return record.recordType == NvTrace_GPU_GrCtxTrace_RecordType_ContextSaveEnd;
            };

            const typename GpuGrCtxTraceBuilderData::GrCtxRecord* pFirstRecord = &gpuGrCtxTraceBuilderData.grCtxRecords.front();
            const auto& firstContextChange = fnGetContextChange(*pFirstRecord);

            if (fnIsContextSaveEnd(*pFirstRecord))
            {
                // load event is out of the observation window. Put 0 as the load timestamp to indicate the context exists even before the observation window.
                auto processLuid = fnFindCtxCorrespondingProcessLuid(firstContextChange.rmCid, firstContextChange.pid, InvalidTimestamp);
                gpuGrCtxTraceData.residentContexts.emplace_back(GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec, firstContextChange.timestampGpuNsec, firstContextChange, processLuid, false, GpuGrCtxTraceData::InvalidSequenceNumber, pFirstRecord->sequenceNumber);
                NVTRACE_LOG_INF(50, "GPU %s: The first GR context load event precedes the observation window. Fabricated a load event with timestamp = 0", ToString(gpuPartitionData.gpuUuid).c_str());
            }

            constexpr uint64_t fabricatedTimestampOffsetGpuNsec = 100;
            const typename GpuGrCtxTraceBuilderData::GrCtxRecord* pPrevRecord = pFirstRecord;
            const auto* pPrevContextChange = &firstContextChange;
            for (size_t eventIndicesIdx = 1; eventIndicesIdx < gpuGrCtxTraceBuilderData.grCtxRecords.size(); eventIndicesIdx++)
            {
                const typename GpuGrCtxTraceBuilderData::GrCtxRecord* pCurRecord = &gpuGrCtxTraceBuilderData.grCtxRecords[eventIndicesIdx];
                const auto& curContextChange = fnGetContextChange(*pCurRecord);

                if (fnIsContextLoadEnd(*pCurRecord))
                {
                    if (fnIsContextLoadEnd(*pPrevRecord))
                    {
                        bool isMultipleEventsDropped = false;
                        uint64_t fabricatedSaveTimestampGpuNsec = curContextChange.timestampGpuNsec - fabricatedTimestampOffsetGpuNsec;
                        if (curContextChange.rmSequenceNumber - pPrevContextChange->rmSequenceNumber > 2) // more than 1 save event is dropped between two load events
                        {
                            isMultipleEventsDropped = true;
                            NVTRACE_LOG_WRN(50, "GPU %s: Multiple dropped GR context events are detected between two load events", ToString(gpuPartitionData.gpuUuid).c_str());
                        }
                        else
                        {
                            NVTRACE_LOG_INF(50, "GPU %s is missing the a GR context save event for process %" PRIu64 ". Fabricated a save event with timestamp = %" PRIu64 "", ToString(gpuPartitionData.gpuUuid).c_str(), pPrevContextChange->pid, fabricatedSaveTimestampGpuNsec);
                        }

                        auto processLuid = fnFindCtxCorrespondingProcessLuid(pPrevContextChange->rmCid, pPrevContextChange->pid, pPrevContextChange->timestampGpuNsec);
                        gpuGrCtxTraceData.residentContexts.emplace_back(pPrevContextChange->timestampGpuNsec, fabricatedSaveTimestampGpuNsec, *pPrevContextChange, processLuid, isMultipleEventsDropped, pPrevRecord->sequenceNumber, GpuGrCtxTraceData::InvalidSequenceNumber).isSaveFabricated = true;
                    }
                }
                else if (fnIsContextSaveEnd(*pCurRecord))
                {
                    if (fnIsContextLoadEnd(*pPrevRecord))
                    {
                        bool isMultipleEventsDropped = false;
                        if (curContextChange.rmSequenceNumber - pPrevContextChange->rmSequenceNumber > 1)
                        {
                            isMultipleEventsDropped = true;
                            NVTRACE_LOG_WRN(50, "GPU %s: Multiple dropped GR context events are detected between a load event and a save event", ToString(gpuPartitionData.gpuUuid).c_str());
                        }

                        auto processLuid = fnFindCtxCorrespondingProcessLuid(pPrevContextChange->rmCid, pPrevContextChange->pid, pPrevContextChange->timestampGpuNsec);
                        gpuGrCtxTraceData.residentContexts.emplace_back(pPrevContextChange->timestampGpuNsec, curContextChange.timestampGpuNsec, *pPrevContextChange, processLuid, isMultipleEventsDropped, pPrevRecord->sequenceNumber, pCurRecord->sequenceNumber);
                    }
                    else if (fnIsContextSaveEnd(*pPrevRecord))
                    {
                        bool isMultipleEventsDropped = false;
                        uint64_t fabricatedLoadTimestampGpuNsec = pPrevContextChange->timestampGpuNsec + fabricatedTimestampOffsetGpuNsec;
                        if (curContextChange.rmSequenceNumber - pPrevContextChange->rmSequenceNumber > 2) // more than 1 load event is dropped between two save events
                        {
                            isMultipleEventsDropped = true;
                            NVTRACE_LOG_WRN(50, "GPU %s: Multiple dropped GR context events are detected between two save events", ToString(gpuPartitionData.gpuUuid).c_str());
                        }
                        else
                        {
                            NVTRACE_LOG_INF(50, "GPU %s is missing the a GR context load event for process %" PRIu64 ". Fabricated a load event with timestamp = %" PRIu64 "", ToString(gpuPartitionData.gpuUuid).c_str(), curContextChange.pid, fabricatedLoadTimestampGpuNsec);
                        }

                        auto processLuid = fnFindCtxCorrespondingProcessLuid(curContextChange.rmCid, curContextChange.pid, fabricatedLoadTimestampGpuNsec);
                        gpuGrCtxTraceData.residentContexts.emplace_back(fabricatedLoadTimestampGpuNsec, curContextChange.timestampGpuNsec, *pPrevContextChange, processLuid, isMultipleEventsDropped, GpuGrCtxTraceData::InvalidSequenceNumber, pCurRecord->sequenceNumber).isLoadFabricated = true;
                    }
                }
                else
                {
                    NVTRACE_LOG_WRN(50, "GPU %s: Unrecognized GR context record type %u", ToString(gpuPartitionData.gpuUuid).c_str(), pCurRecord->recordType);
                    continue;
                }

                pPrevRecord = pCurRecord;
                pPrevContextChange = &curContextChange;
            }

            if (fnIsContextLoadEnd(*pPrevRecord))
            {
                // save event is out of the observation window. Put maximal timestamp as the save timestamp to indicate the context exists even after the observation window.
                auto processLuid = fnFindCtxCorrespondingProcessLuid(pPrevContextChange->rmCid, pPrevContextChange->pid, pPrevContextChange->timestampGpuNsec);
                gpuGrCtxTraceData.residentContexts.emplace_back(pPrevContextChange->timestampGpuNsec, GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec, *pPrevContextChange, processLuid, false, pPrevRecord->sequenceNumber, GpuGrCtxTraceData::InvalidSequenceNumber);
                NVTRACE_LOG_INF(50, "GPU %s: The last GR context save event is past the observation window. Fabricated a save event with timestamp = UINT64_MAX", ToString(gpuPartitionData.gpuUuid).c_str());
            }

            return;
        }

        void CorrelateGraphNodeLaunchesToGraphLaunches()
        {
            auto graphLaunchDemarcatorTimestampLess = [](const typename CudaApiTraceBuilderData::GraphLaunchDemarcator& lhs, const typename CudaApiTraceBuilderData::GraphLaunchDemarcator& rhs) {
                return lhs.timestampTsc < rhs.timestampTsc;
            };

            auto graphLaunchTimestampLessThanDemarcator = [](uint64_t timestampTsc, const typename CudaApiTraceBuilderData::GraphLaunchDemarcator& rhs) {
                return timestampTsc < rhs.timestampTsc;
            };

            for (auto& processMapPair : m_systemBuilderData.processes)
            {
                ProcessBuilderData& processBuilderData = processMapPair.second;
                ProcessData& processData = processBuilderData.shadowed;
                CudaApiTraceData& cudaApiTraceData = processData.cudaApiTrace;
                cudaApiTraceData.graphLaunches.clear();

                for (auto& [cugraphExecId, graphLaunchDemarcators] : processBuilderData.cudaApiTrace.graphLaunchDemarcatorsByGraphExecId)
                {
                    std::stable_sort(graphLaunchDemarcators.begin(), graphLaunchDemarcators.end(), graphLaunchDemarcatorTimestampLess);
                    for (auto& graphLaunchDemarcator : graphLaunchDemarcators)
                    {
                        graphLaunchDemarcator.graphLaunchIdx = cudaApiTraceData.graphLaunches.size();
                        graphLaunchDemarcator.graphLaunch.cugraphExecId = cugraphExecId;
                        cudaApiTraceData.graphLaunches.push_back(graphLaunchDemarcator.graphLaunch);
                    }
                }

                if (m_options.preserveOutOfWindowTimestamps)
                {
                    // A host launch whose GridNullEnd is not in this trace still gets a row.
                    for (const typename CudaApiTraceBuilderData::GraphLaunch& pendingGraphLaunch : processBuilderData.cudaApiTrace.graphLaunches)
                    {
                        if (pendingGraphLaunch.gridNullEndRecordLocation != InvalidRecordLocation)
                        {
                            continue;
                        }
                        const auto itrStream = cudaApiTraceData.streams.find(pendingGraphLaunch.custreamId);
                        CudaApiTraceData::GraphLaunch& graphLaunch = cudaApiTraceData.graphLaunches.emplace_back();
                        graphLaunch.source = CudaApiTraceData::GraphLaunchSource::Host;
                        graphLaunch.hwCorrelationId = pendingGraphLaunch.GetHwCorrelationId();
                        graphLaunch.cugraphExecId = pendingGraphLaunch.cugraphExecId;
                        graphLaunch.payload.host = {
                            pendingGraphLaunch.apiCorrelationId,
                            pendingGraphLaunch.apiCallRef,
                            InvalidTimestamp,
                            itrStream != cudaApiTraceData.streams.end() ? itrStream->second.cudaDeviceOrdinal : UINT32_MAX,
                            pendingGraphLaunch.custreamId};
                    }
                }

                for (CudaApiTraceData::CuDeviceData& cuDeviceData : cudaApiTraceData.devices)
                {
                    if (cuDeviceData.graphNodeLaunches.empty() && cuDeviceData.unmatchedGraphNodeLaunchEnds.empty())
                    {
                        continue;
                    }

                    GpuPartitionData* pGpuPartitionData = m_dataModel.FindGpuPartitionByGpuUuid(cuDeviceData.gpuUuid);
                    if (!pGpuPartitionData)
                    {
                        NVTRACE_LOG_WRN(50, "No GPU partition found for graph-node launch correlation, GPU UUID=%s", ToString(cuDeviceData.gpuUuid).c_str());
                        continue;
                    }

                    for (CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch : cuDeviceData.graphNodeLaunches)
                    {
                        if (graphNodeLaunch.startTimestampGpuNsec == InvalidTimestamp)
                        {
                            continue;
                        }

                        const uint64_t graphNodeStartTimestampTsc = m_dataModel.ConvertGpuNsecToTscTick(*pGpuPartitionData, graphNodeLaunch.startTimestampGpuNsec);
                        if (graphNodeStartTimestampTsc == InvalidTimestamp)
                        {
                            continue;
                        }

                        const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = m_dataModel.FindCuGraphNode(cuDeviceData, graphNodeLaunch.hwCorrelationId, graphNodeStartTimestampTsc);
                        if (!pGraphNodeData)
                        {
                            continue;
                        }
                        if (m_options.preserveOutOfWindowTimestamps)
                        {
                            graphNodeLaunch.graphNodeCorrelationId = pGraphNodeData->graphNodeCorrelationId;
                        }

                        auto itrGraphExecData = cudaApiTraceData.graphExecs.lower_bound(pGraphNodeData->cugraphExecId);
                        if (itrGraphExecData == cudaApiTraceData.graphExecs.end() || cudaApiTraceData.graphExecs.key_comp()(pGraphNodeData->cugraphExecId, itrGraphExecData->first))
                        {
                            continue;
                        }

                        auto itrGraphLaunchDemarcators = processBuilderData.cudaApiTrace.graphLaunchDemarcatorsByGraphExecId.lower_bound(pGraphNodeData->cugraphExecId);
                        if (itrGraphLaunchDemarcators == processBuilderData.cudaApiTrace.graphLaunchDemarcatorsByGraphExecId.end() || processBuilderData.cudaApiTrace.graphLaunchDemarcatorsByGraphExecId.key_comp()(pGraphNodeData->cugraphExecId, itrGraphLaunchDemarcators->first))
                        {
                            if (!m_options.preserveOutOfWindowTimestamps)
                            {
                                NVTRACE_LOG_WRN(100, "Dropping graph node launch hwCorrelationId=0x%08X: no graph launch of cugraphExecId=%" PRIu64 " in this snapshot (its GridNullEnd may be in an earlier snapshot)", graphNodeLaunch.hwCorrelationId, pGraphNodeData->cugraphExecId);
                            }
                            continue;
                        }

                        std::vector<typename CudaApiTraceBuilderData::GraphLaunchDemarcator>& graphLaunchDemarcators = itrGraphLaunchDemarcators->second;
                        auto itrGraphLaunchDemarcator = std::upper_bound(graphLaunchDemarcators.begin(), graphLaunchDemarcators.end(), graphNodeStartTimestampTsc, graphLaunchTimestampLessThanDemarcator);
                        if (itrGraphLaunchDemarcator == graphLaunchDemarcators.begin())
                        {
                            if (!m_options.preserveOutOfWindowTimestamps)
                            {
                                NVTRACE_LOG_WRN(100, "Dropping graph node launch hwCorrelationId=0x%08X: it starts before every graph launch of cugraphExecId=%" PRIu64 " in this snapshot (its GridNullEnd may be in an earlier snapshot)", graphNodeLaunch.hwCorrelationId, pGraphNodeData->cugraphExecId);
                            }
                            continue;
                        }
                        --itrGraphLaunchDemarcator;
                        if (itrGraphLaunchDemarcator->timestampTsc == InvalidTimestamp)
                        {
                            continue;
                        }

                        graphNodeLaunch.graphLaunchIdx = itrGraphLaunchDemarcator->graphLaunchIdx;
                        graphNodeLaunch.graphNodeCorrelationId = pGraphNodeData->graphNodeCorrelationId;
                    }

                    CullVector(cuDeviceData.unmatchedGraphNodeLaunchEnds, [&](CudaApiTraceData::GraphNodeLaunch& unmatchedEnd) {
                        const uint64_t endTimestampTsc = m_dataModel.ConvertGpuNsecToTscTick(*pGpuPartitionData, unmatchedEnd.endTimestampGpuNsec);
                        const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = endTimestampTsc == InvalidTimestamp ? nullptr : m_dataModel.FindCuGraphNode(cuDeviceData, unmatchedEnd.hwCorrelationId, endTimestampTsc);
                        if (!pGraphNodeData)
                        {
                            NVTRACE_LOG_WRN(100, "Dropping end of graph node launch hwCorrelationId=0x%08X: no graph node for it", unmatchedEnd.hwCorrelationId);
                            return false;
                        }
                        unmatchedEnd.graphNodeCorrelationId = pGraphNodeData->graphNodeCorrelationId;
                        return true;
                    });
                }
            }
        }

        void ReconstructGpuKernelLaunches(GpuPartitionBuilderData& gpuPartitionBuilderData)
        {
            GpuPartitionData& gpuPartitionData = gpuPartitionBuilderData.shadowed;
            GpuGrCtxTraceData& gpuGrCtxTraceData = gpuPartitionData.gpuGrCtxTrace;

            GpuHwTraceBuilderData& gpuHwTraceBuilderData = gpuPartitionBuilderData.gpuHwTrace;

            if (gpuGrCtxTraceData.residentContexts.empty())
            {
                NVTRACE_LOG_WRN(20, "No GR contexts found in GPU partition %s while %zu HW trace events are found. Skip HW trace and CUDA API correlation", ToString(gpuPartitionData.gpuUuid).c_str(), gpuHwTraceBuilderData.hwRecords.size());
                return;
            }

            // state machine to track the current context
            struct ContextState
            {
                const GpuGrCtxTraceData::ResidentContext* pResidentContext = nullptr;
                ProcessData* pProcessData = nullptr;
                CudaApiTraceData::CuDeviceData* pCuDeviceData = nullptr;

                void Update(DataModel<TTraceReader>& dataModel, const NVTRACE_UUID& gpuUuid, const GpuGrCtxTraceData::ResidentContext& residentContext)
                {
                    pResidentContext = &residentContext;
                    pProcessData = dataModel.FindProcess(residentContext.processLuid);
                    pCuDeviceData = pProcessData ? dataModel.FindCuDeviceByGpuUuid(*pProcessData, gpuUuid) : nullptr;
                }

                bool IsValid() const
                {
                    return pCuDeviceData;
                }

                bool InRange(uint64_t timestampGpuNsec) const
                {
                    return timestampGpuNsec >= pResidentContext->ctxLoadTimestampGpuNsec && timestampGpuNsec <= pResidentContext->ctxSaveTimestampGpuNsec;
                }
            };
            ContextState ctxState;

            size_t residentIdx = 0; // a special index to track the current resident context in the gpuPartition's GrCtx resident contexts vector
            const auto& firstGrCtx = gpuGrCtxTraceData.residentContexts[residentIdx];
            ctxState.Update(m_dataModel, gpuPartitionData.gpuUuid, firstGrCtx);

            // Update "ctxState" based on the HW trace event timestamp.
            auto updateCtxState = [&](uint64_t timestampGpuNsec) -> bool {
                if (residentIdx == gpuGrCtxTraceData.residentContexts.size())
                {
                    NVTRACE_LOG_INF(50, "GPU %s: HW trace event is past the last resident context. Fail to update the context state.", ToString(gpuPartitionData.gpuUuid).c_str());
                    return false; // fast early out
                }

                if (ctxState.pResidentContext && ctxState.InRange(timestampGpuNsec))
                {
                    return ctxState.IsValid(); // fast early-out: the current context is still valid
                }
                else if (ctxState.pResidentContext && timestampGpuNsec <= ctxState.pResidentContext->ctxSaveTimestampGpuNsec) // check with save timestamp because load timestamp can be zero (if it's the first context and fabricated)
                {
                    // corner case: HW trace event precedes the current context -> keep the current context but return false to indicate "no context found".
                    NVTRACE_LOG_INF(50, "GPU %s: HW trace event is in between two resident contexts. Fail to update the context state.", ToString(gpuPartitionData.gpuUuid).c_str());
                    return false; // fast early-out: HW trace event is in between two resident contexts
                }

                ++residentIdx;
                for (; residentIdx < gpuGrCtxTraceData.residentContexts.size(); ++residentIdx)
                {
                    const auto& grCtx = gpuGrCtxTraceData.residentContexts[residentIdx];

                    if (timestampGpuNsec >= grCtx.ctxLoadTimestampGpuNsec && timestampGpuNsec <= grCtx.ctxSaveTimestampGpuNsec)
                    {
                        // normal case: HW trace event is within a resident context -> update the context.
                        ctxState.Update(m_dataModel, gpuPartitionData.gpuUuid, grCtx);
                        return ctxState.IsValid();
                    }
                    else if (timestampGpuNsec < grCtx.ctxSaveTimestampGpuNsec)
                    {
                        // corner case: HW trace event is in-between two resident contexts -> update the context but return false to indicate "no context found".
                        ctxState.Update(m_dataModel, gpuPartitionData.gpuUuid, grCtx);
                        NVTRACE_LOG_INF(50, "GPU %s: HW trace event is in between two resident contexts. Fail to update the context state.", ToString(gpuPartitionData.gpuUuid).c_str());
                        return false;
                    }
                }

                if (residentIdx != gpuGrCtxTraceData.residentContexts.size())
                {
                    residentIdx = gpuGrCtxTraceData.residentContexts.size();
                    NVTRACE_LOG_INF(50, "GPU %s: HW trace event is past the last resident context. Fail to update the context state.", ToString(gpuPartitionData.gpuUuid).c_str());
                }

                return false;
            };

            // Auxiliary data structure to build GraphNodeLaunch events.  Assumes no nesting of launches, which is guaranteed by the programming model.
            std::map<uint32_t, size_t> mapHwCorrelationIdToGraphNodeLaunchIndex;

            for (const auto& hwRecord : gpuHwTraceBuilderData.hwRecords)
            {
                const uint64_t recordLocation = hwRecord.recordLocation;
                const uint32_t recordType = hwRecord.recordType;
                const uint8_t* pMajorPayload = m_dataModel.m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload;

                // hwRecords entries were range-gated by AddGpuHwTraceRecord, so an out-of-range type here can only be
                // corruption; skip it observably rather than falling past the switch silently.
                NVTRACE_LOG_ERR_CONTINUE_IF(NvTrace_GPU_HwTrace_RecordType_Count <= recordType, 50, "Skipping unknown GpuHwTrace record type %u during reconstruction", recordType);

                // These records were size-validated to >= their payload's minorVersion in the first pass
                // (AddGpuHwTraceRecord), so only fields within that range are read below. No default: adding a record
                // type is a compile error here until it is explicitly correlated or ignored.
                switch (static_cast<NvTrace_GPU_HwTrace_RecordType>(recordType))
                {
                    case NvTrace_GPU_HwTrace_RecordType_ProviderDiagnostic_InitializationFailure:
                        // ignored: provider diagnostics are consumed into GpuHwTraceData::providerDiagnostics in the first pass
                        break;

                    case NvTrace_GPU_HwTrace_RecordType_ContextChangeLoad:
                    case NvTrace_GPU_HwTrace_RecordType_ContextChangeSave:
                        // ignored: context change load and save events are handled by the context state machine
                        break;

                    case NvTrace_GPU_HwTrace_RecordType_ComputeGridNullEnd:
                    {
                        const NvTrace_GPU_HwTrace_ComputeGrid& computeGrid = *reinterpret_cast<const NvTrace_GPU_HwTrace_ComputeGrid*>(pMajorPayload);
                        const uint32_t gridNullEndCorrelationId = computeGrid.correlationId;
                        const uint64_t gridNullEndTimestampGpuNsec = computeGrid.ptimer;
                        const uint32_t gridNullEndCorrelationMsbs = gridNullEndCorrelationId & NVTRACE_CUDA_HW_CORRELATION_ID_MSB_MASK;

                        const bool updateSucceeded = updateCtxState(gridNullEndTimestampGpuNsec);
                        if (!updateSucceeded)
                        {
                            NVTRACE_LOG_INF(50, "Invalid context state for GridNullEnd correlationId=0x%08X event. Skipping.", gridNullEndCorrelationId);
                            break;
                        }

                        const uint64_t gridNullEndTimestampTsc = m_dataModel.ConvertGpuNsecToTscTick(gpuPartitionData, gridNullEndTimestampGpuNsec);
                        if (gridNullEndTimestampTsc == InvalidTimestamp)
                        {
                            NVTRACE_LOG_INF(50, "Invalid converted timestamp for GridNullEnd correlationId=0x%08X event. Leaving as an unclassified GridNullEnd.", gridNullEndCorrelationId);
                            break;
                        }

                        CudaApiTraceData& cudaApiTraceData = ctxState.pProcessData->cudaApiTrace;
                        auto itrProcessBuilderData = m_systemBuilderData.processes.lower_bound(ctxState.pResidentContext->processLuid);
                        if (itrProcessBuilderData == m_systemBuilderData.processes.end() || m_systemBuilderData.processes.key_comp()(ctxState.pResidentContext->processLuid, itrProcessBuilderData->first))
                        {
                            NVTRACE_LOG_INF(50, "No process builder state for GridNullEnd correlationId=0x%08X event. Skipping.", gridNullEndCorrelationId);
                            break;
                        }
                        CudaApiTraceBuilderData& cudaApiTraceBuilderData = itrProcessBuilderData->second.cudaApiTrace;
                        auto isTimestampAfterResourceCreation = [](const auto& resourceData, uint64_t timestampTsc) {
                            if (resourceData.creationTimestampTsc != InvalidTimestamp && timestampTsc < resourceData.creationTimestampTsc)
                            {
                                return false;
                            }
                            return true;
                        };

                        auto correlateHostGraphLaunchGridNullEnd = [&]() {
                            auto graphLaunchIter = std::lower_bound(
                                cudaApiTraceBuilderData.graphLaunches.begin(),
                                cudaApiTraceBuilderData.graphLaunches.end(),
                                gridNullEndCorrelationId,
                                [](const typename CudaApiTraceBuilderData::GraphLaunch& graphLaunch, uint32_t hwCorrelationId) {
                                    return graphLaunch.GetHwCorrelationId() < hwCorrelationId;
                                });

                            size_t unmatchedGraphLaunchCount = 0;
                            size_t selectedGraphLaunchIndex = SIZE_MAX;
                            for (; graphLaunchIter != cudaApiTraceBuilderData.graphLaunches.end() && graphLaunchIter->GetHwCorrelationId() == gridNullEndCorrelationId; ++graphLaunchIter)
                            {
                                if (graphLaunchIter->gridNullEndRecordLocation == InvalidRecordLocation)
                                {
                                    ++unmatchedGraphLaunchCount;
                                }

                                const auto itrGraphExecData = cudaApiTraceData.graphExecs.lower_bound(graphLaunchIter->cugraphExecId);
                                const bool graphExecExists = itrGraphExecData != cudaApiTraceData.graphExecs.end() && !cudaApiTraceData.graphExecs.key_comp()(graphLaunchIter->cugraphExecId, itrGraphExecData->first);
                                if (graphLaunchIter->gridNullEndRecordLocation == InvalidRecordLocation
                                    && graphLaunchIter->startTimestampTsc != InvalidTimestamp
                                    && graphLaunchIter->startTimestampTsc <= gridNullEndTimestampTsc
                                    && graphExecExists)
                                {
                                    const size_t graphLaunchIndex = static_cast<size_t>(std::distance(cudaApiTraceBuilderData.graphLaunches.begin(), graphLaunchIter));
                                    selectedGraphLaunchIndex = graphLaunchIndex;
                                    break;
                                }
                            }

                            if (selectedGraphLaunchIndex == SIZE_MAX)
                            {
                                if (m_options.preserveOutOfWindowTimestamps && unmatchedGraphLaunchCount == 0)
                                {
                                    ctxState.pCuDeviceData->unmatchedHostGraphLaunchTimestampsGpuNsec[gridNullEndCorrelationId].push_back(gridNullEndTimestampGpuNsec);
                                    return;
                                }
                                NVTRACE_LOG_INF(
                                    50,
                                    "Host-tagged GridNullEnd did not match graph-launch metadata: correlationId=0x%08X exactCandidateCount=%zu. Leaving as an unclassified GridNullEnd.",
                                    gridNullEndCorrelationId,
                                    unmatchedGraphLaunchCount);
                                return;
                            }

                            typename CudaApiTraceBuilderData::GraphLaunch& selectedGraphLaunch = cudaApiTraceBuilderData.graphLaunches[selectedGraphLaunchIndex];

                            auto [itrGraphExecData, inserted] = cudaApiTraceData.graphExecs.try_emplace(selectedGraphLaunch.cugraphExecId);
                            if (inserted)
                            {
                                itrGraphExecData->second.cugraphExecId = selectedGraphLaunch.cugraphExecId;
                            }

                            typename CudaApiTraceBuilderData::GraphLaunchDemarcator& graphLaunchDemarcator =
                                cudaApiTraceBuilderData.graphLaunchDemarcatorsByGraphExecId[selectedGraphLaunch.cugraphExecId].emplace_back();
                            graphLaunchDemarcator.timestampGpuNsec = gridNullEndTimestampGpuNsec;
                            graphLaunchDemarcator.timestampTsc = gridNullEndTimestampTsc;
                            graphLaunchDemarcator.cudaDeviceOrdinal = ctxState.pCuDeviceData->cudaDeviceOrdinal;
                            graphLaunchDemarcator.gridNullEndRecordLocation = recordLocation;
                            graphLaunchDemarcator.graphLaunch.source = CudaApiTraceData::GraphLaunchSource::Host;
                            graphLaunchDemarcator.graphLaunch.hwCorrelationId = selectedGraphLaunch.GetHwCorrelationId();
                            graphLaunchDemarcator.graphLaunch.cugraphExecId = selectedGraphLaunch.cugraphExecId;
                            graphLaunchDemarcator.graphLaunch.payload.host = {
                                selectedGraphLaunch.apiCorrelationId,
                                selectedGraphLaunch.apiCallRef,
                                gridNullEndTimestampGpuNsec,
                                static_cast<uint32_t>(ctxState.pCuDeviceData->cudaDeviceOrdinal),
                                selectedGraphLaunch.custreamId};

                            selectedGraphLaunch.gridNullEndRecordLocation = recordLocation;
                        };

                        auto correlateDeviceGraphLaunchGridNullEnd = [&]() {
                            size_t exactGraphExecCount = 0;
                            std::vector<uint64_t> plausibleGraphExecIds;
                            const auto graphExecIdsRange = cudaApiTraceBuilderData.graphExecIdsByDeviceGraphLaunchHwCorrelationId.equal_range(gridNullEndCorrelationId);
                            for (auto itrGraphExecId = graphExecIdsRange.first; itrGraphExecId != graphExecIdsRange.second; ++itrGraphExecId)
                            {
                                const uint64_t cugraphExecId = itrGraphExecId->second;
                                const auto itrGraphExecData = cudaApiTraceData.graphExecs.find(cugraphExecId);
                                if (itrGraphExecData == cudaApiTraceData.graphExecs.end())
                                {
                                    continue;
                                }

                                ++exactGraphExecCount;
                                if (isTimestampAfterResourceCreation(itrGraphExecData->second, gridNullEndTimestampTsc))
                                {
                                    plausibleGraphExecIds.push_back(cugraphExecId);
                                }
                            }

                            if (plausibleGraphExecIds.empty())
                            {
                                NVTRACE_LOG_INF(
                                    50,
                                    "Device-tagged GridNullEnd did not match graph exec metadata: correlationId=0x%08X exactCandidateCount=%zu. Leaving as an unclassified GridNullEnd.",
                                    gridNullEndCorrelationId,
                                    exactGraphExecCount);
                                return;
                            }

                            if (plausibleGraphExecIds.size() != 1)
                            {
                                NVTRACE_LOG_WRN(
                                    50,
                                    "Ambiguous device graph-launch GridNullEnd correlation: correlationId=0x%08X candidateCount=%zu. Leaving unmatched.",
                                    gridNullEndCorrelationId,
                                    plausibleGraphExecIds.size());

                                return;
                            }

                            typename CudaApiTraceBuilderData::GraphLaunchDemarcator& graphLaunchDemarcator =
                                cudaApiTraceBuilderData.graphLaunchDemarcatorsByGraphExecId[plausibleGraphExecIds.front()].emplace_back();
                            graphLaunchDemarcator.timestampGpuNsec = gridNullEndTimestampGpuNsec;
                            graphLaunchDemarcator.timestampTsc = gridNullEndTimestampTsc;
                            graphLaunchDemarcator.cudaDeviceOrdinal = ctxState.pCuDeviceData->cudaDeviceOrdinal;
                            graphLaunchDemarcator.gridNullEndRecordLocation = recordLocation;
                            graphLaunchDemarcator.graphLaunch.source = CudaApiTraceData::GraphLaunchSource::Device;
                            graphLaunchDemarcator.graphLaunch.hwCorrelationId = gridNullEndCorrelationId;
                            graphLaunchDemarcator.graphLaunch.cugraphExecId = plausibleGraphExecIds.front();
                            graphLaunchDemarcator.graphLaunch.payload.device.timestampGpuNsec = gridNullEndTimestampGpuNsec;
                            graphLaunchDemarcator.graphLaunch.payload.device.cudaDeviceOrdinal = static_cast<uint32_t>(ctxState.pCuDeviceData->cudaDeviceOrdinal);
                        };

                        if (gridNullEndCorrelationMsbs == NVTRACE_CUDA_HW_CORRELATION_ID_GRAPH_LAUNCH_HOST_MSBS)
                        {
                            correlateHostGraphLaunchGridNullEnd();
                        }
                        else if (gridNullEndCorrelationMsbs == NVTRACE_CUDA_HW_CORRELATION_ID_GRAPH_LAUNCH_DEVICE_MSBS)
                        {
                            correlateDeviceGraphLaunchGridNullEnd();
                        }
                        else
                        {
                            NVTRACE_LOG_INF(50, "Ignoring non-graph-launch GridNullEnd correlationId=0x%08X event.", gridNullEndCorrelationId);
                        }

                        break;
                    }

                    case NvTrace_GPU_HwTrace_RecordType_ComputeGridStart:
                    {
                        const NvTrace_GPU_HwTrace_ComputeGrid& computeGrid = *reinterpret_cast<const NvTrace_GPU_HwTrace_ComputeGrid*>(pMajorPayload);
                        const uint32_t computeCorrelationId = computeGrid.correlationId;
                        if (!computeCorrelationId)
                        {
                            continue;
                        }
                        const bool updateSucceeded = updateCtxState(computeGrid.ptimer);
                        if (!updateSucceeded)
                        {
                            NVTRACE_LOG_INF(50, "Invalid context state for computeCorrelationId=0x%08X compute grid start event. Skipping.", computeCorrelationId);
                            continue;
                        }

                        auto* pCuDeviceData = ctxState.pCuDeviceData;
                        if ((computeCorrelationId & NVTRACE_CUDA_HW_CORRELATION_ID_MSB_MASK) == NVTRACE_CUDA_HW_CORRELATION_ID_IMMEDIATE_MSBS) // immediate kernel launch
                        {
                            // Immediate kernel start event; find the first immediate kernel launch with a matching correlation ID
                            auto launchIter = std::lower_bound(
                                pCuDeviceData->immediateKernelLaunches.begin(),
                                pCuDeviceData->immediateKernelLaunches.end(),
                                computeCorrelationId,
                                [](const typename CudaApiTraceData::ImmediateKernelLaunch& launch, uint32_t computeCorrelationId) {
                                    return launch.GetHwCorrelationId() < computeCorrelationId;
                                });
                            // Advance to the first launch with an unset start timestamp
                            bool found = false;
                            for (; launchIter != pCuDeviceData->immediateKernelLaunches.end() && launchIter->GetHwCorrelationId() == computeCorrelationId; ++launchIter)
                            {
                                if (launchIter->startTimestampGpuNsec == InvalidTimestamp)
                                {
                                    launchIter->startTimestampGpuNsec = computeGrid.ptimer;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                if (!m_options.preserveOutOfWindowTimestamps)
                                {
                                    NVTRACE_LOG_INF(50, "Failed to find API immediate kernel launch for computeCorrelationId=0x%08X start event.", computeCorrelationId);
                                    continue;
                                }
                                pCuDeviceData->unmatchedImmediateKernelExecutions[computeCorrelationId].push_back({computeGrid.ptimer, InvalidTimestamp});
                                continue;
                            }
                        }
                        else // graph launch
                        {
                            const auto itrGraphNodeLaunch = mapHwCorrelationIdToGraphNodeLaunchIndex.lower_bound(computeCorrelationId);
                            if (itrGraphNodeLaunch != mapHwCorrelationIdToGraphNodeLaunchIndex.end() && itrGraphNodeLaunch->first == computeCorrelationId)
                            {
                                NVTRACE_LOG_INF(50, "Duplicate GPU HW trace compute grid start event for graph node correlationId=0x%08X.  Overwriting.", computeCorrelationId);
                                continue;
                            }

                            const size_t graphNodeLaunchIndex = pCuDeviceData->graphNodeLaunches.size();
                            typename CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch = pCuDeviceData->graphNodeLaunches.emplace_back();
                            graphNodeLaunch.hwCorrelationId = computeCorrelationId;
                            graphNodeLaunch.startTimestampGpuNsec = computeGrid.ptimer;
                            mapHwCorrelationIdToGraphNodeLaunchIndex.emplace_hint(itrGraphNodeLaunch, computeCorrelationId, graphNodeLaunchIndex);
                        }
                        break;
                    }

                    case NvTrace_GPU_HwTrace_RecordType_ComputeGridEnd:
                    {
                        const NvTrace_GPU_HwTrace_ComputeGrid& computeGrid = *reinterpret_cast<const NvTrace_GPU_HwTrace_ComputeGrid*>(pMajorPayload);
                        const uint32_t computeCorrelationId = computeGrid.correlationId;
                        if (!computeCorrelationId)
                        {
                            continue;
                        }
                        const bool updateSucceeded = updateCtxState(computeGrid.ptimer);
                        if (!updateSucceeded)
                        {
                            NVTRACE_LOG_INF(50, "Invalid context state for computeCorrelationId=0x%08X compute grid end event. Skipping.", computeCorrelationId);
                            continue;
                        }

                        auto* pCuDeviceData = ctxState.pCuDeviceData;
                        if ((computeCorrelationId & NVTRACE_CUDA_HW_CORRELATION_ID_MSB_MASK) == NVTRACE_CUDA_HW_CORRELATION_ID_IMMEDIATE_MSBS)
                        {
                            // Immediate kernel end event; find the first immediate kernel launch with a matching correlation ID
                            auto launchIter = std::lower_bound(
                                pCuDeviceData->immediateKernelLaunches.begin(),
                                pCuDeviceData->immediateKernelLaunches.end(),
                                computeCorrelationId,
                                [](const typename CudaApiTraceData::ImmediateKernelLaunch& launch, uint32_t computeCorrelationId) {
                                    return launch.GetHwCorrelationId() < computeCorrelationId;
                                });
                            // Advance to the first launch with an unset end timestamp
                            bool found = false;
                            for (; launchIter != pCuDeviceData->immediateKernelLaunches.end() && launchIter->GetHwCorrelationId() == computeCorrelationId; ++launchIter)
                            {
                                if (launchIter->endTimestampGpuNsec == InvalidTimestamp)
                                {
                                    launchIter->endTimestampGpuNsec = computeGrid.ptimer;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                if (!m_options.preserveOutOfWindowTimestamps)
                                {
                                    NVTRACE_LOG_INF(50, "Failed to find API immediate kernel launch for computeCorrelationId=0x%08X end event.", computeCorrelationId);
                                    continue;
                                }
                                std::vector<GpuHwTraceData::KernelExecution>& unmatchedExecutions = pCuDeviceData->unmatchedImmediateKernelExecutions[computeCorrelationId];
                                const auto itrUnmatchedExecution = std::find_if(unmatchedExecutions.begin(), unmatchedExecutions.end(), [](const auto& unmatchedExecution) {
                                    return unmatchedExecution.endTimestampGpuNsec == InvalidTimestamp;
                                });
                                if (itrUnmatchedExecution != unmatchedExecutions.end())
                                {
                                    itrUnmatchedExecution->endTimestampGpuNsec = computeGrid.ptimer;
                                }
                                else
                                {
                                    unmatchedExecutions.push_back({InvalidTimestamp, computeGrid.ptimer});
                                }
                                continue;
                            }
                        }
                        else
                        {
                            auto itr = mapHwCorrelationIdToGraphNodeLaunchIndex.find(computeCorrelationId);
                            if (itr == mapHwCorrelationIdToGraphNodeLaunchIndex.end())
                            {
                                if (!m_options.preserveOutOfWindowTimestamps)
                                {
                                    NVTRACE_LOG_WRN(50, "Failed to find graph node launch for correlationId=0x%08X end event.", computeCorrelationId);
                                    continue;
                                }
                                CudaApiTraceData::GraphNodeLaunch& unmatchedEnd = pCuDeviceData->unmatchedGraphNodeLaunchEnds.emplace_back();
                                unmatchedEnd.hwCorrelationId = computeCorrelationId;
                                unmatchedEnd.endTimestampGpuNsec = computeGrid.ptimer;
                                continue;
                            }

                            const size_t graphNodeLaunchIndex = itr->second;
                            mapHwCorrelationIdToGraphNodeLaunchIndex.erase(itr);
                            typename CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch = pCuDeviceData->graphNodeLaunches[graphNodeLaunchIndex];
                            graphNodeLaunch.endTimestampGpuNsec = computeGrid.ptimer;
                        }
                        break;
                    }

                    case NvTrace_GPU_HwTrace_RecordType_Invalid:
                    case NvTrace_GPU_HwTrace_RecordType_RecordVersionInfo:
                    case NvTrace_GPU_HwTrace_RecordType_ProviderConfiguration:
                    case NvTrace_GPU_HwTrace_RecordType_LifecyclePause:
                    case NvTrace_GPU_HwTrace_RecordType_LifecycleResume:
                    case NvTrace_GPU_HwTrace_RecordType_GpuDeviceInfo:
                    case NvTrace_GPU_HwTrace_RecordType_Count:
                    case NvTrace_GPU_HwTrace_RecordType_FORCE_INT:
                        // Not correlated here (AddGpuHwTraceRecord only admits the handled HW-event types above).
                        NVTRACE_LOG_WRN(50, "Correlating HW trace events: unhandled record type %u for event index %" PRIu64 ".", recordType, recordLocation);
                        break;
                }
            }
        }

    protected:
        void OnObjectDefined(NVTRACE_LUID providerLuid, ObjectType objectType, uint64_t objectId) {}
        void OnObjectReferenced(NVTRACE_LUID providerLuid, ObjectType objectType, uint64_t objectId) {}

    protected:
        DataModel<TTraceReader>& m_dataModel;
        SystemBuilderData m_systemBuilderData;

        Options m_options = {};
    };


    template <typename TTraceReader>
    class DataModelBuilder;

} // namespace nv::trace::Codec
