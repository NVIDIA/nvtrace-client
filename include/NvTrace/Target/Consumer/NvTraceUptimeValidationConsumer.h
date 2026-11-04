/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceConsumerBase.h>

namespace nv::trace {

    struct UptimeValidationConsumerObjectModel : ConsumerObjectModelBase
    {
        struct ProviderState : ConsumerObjectModelBase::ProviderState
        {
            uint64_t sampleRateIntervalTsc = 0;
            uint64_t lastSampleWriteTsc = 0;
            uint64_t lastSampleReadTsc = 0;
            bool providerHalted = false;

            void Reset() noexcept
            {
                providerHalted = false;
                lastSampleReadTsc = 0;
                lastSampleWriteTsc = 0;
                sampleRateIntervalTsc = 0;
                ConsumerObjectModelBase::ProviderState::Reset();
            }
        };
        using CpuClockCalibrationProviderState = TypedProviderState<CpuClockCalibrationObserver, ProviderState>;
        using GpuClockCalibrationProviderState = TypedProviderState<GpuClockCalibrationObserver, ProviderState>;
        using GpuPmCountersProviderState = TypedProviderState<GpuPmCounterObserver, ProviderState>;
        using GpuHwTraceProviderState = TypedProviderState<GpuHwTraceObserver, ProviderState>;
        using GpuGrCtxTraceProviderState = TypedProviderState<GpuGrCtxTraceObserver, ProviderState>;
        using CudaApiTraceProviderState = TypedCudaApiTraceProviderState<ProviderState>;
        using NvtxTraceProviderState = TypedNvtxTraceProviderState<ProviderState>;
        using ThreadEventProviderState = TypedThreadEventProviderState<ProviderState>;
    };


    class UptimeValidationConsumer : protected ConsumerBase<UptimeValidationConsumer, UptimeValidationConsumerObjectModel>
    {
        using Super = ConsumerBase<UptimeValidationConsumer, UptimeValidationConsumerObjectModel>;
        friend Super;

    public:
        struct Options : Super::Options
        {
            uint64_t gpuHwTraceUpdateIntervalNsec = 0;
            uint64_t gpuGrCtxTraceUpdateIntervalNsec = 0;
            uint64_t cudaApiTraceUpdateIntervalNsec = 0;
        };

    public:
        using Super::ConvertTscTickToUtcNsec;

        ~UptimeValidationConsumer() = default;
        UptimeValidationConsumer() = default;

        void Reset() noexcept
        {
            m_counters = {};
            m_cudaApiTraceUpdateIntervalNsec = 0;
            m_gpuHwTraceUpdateIntervalNsec = 0;
            m_gpuGrCtxTraceUpdateIntervalNsec = 0;
            m_waitMultiplexer.Reset();
            Super::Reset();
        }

        bool Initialize(const Options& options)
        {
            if (!Super::Initialize(options))
            {
                return false;
            }

            auto errorGuard = ScopeExitGuard([this]() { Reset(); });

            bool succeeded = BuildWaitMultiplexer();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "BuildWaitMultiplexer failed");

            m_gpuHwTraceUpdateIntervalNsec = options.gpuHwTraceUpdateIntervalNsec;
            m_gpuGrCtxTraceUpdateIntervalNsec = options.gpuGrCtxTraceUpdateIntervalNsec;
            m_cudaApiTraceUpdateIntervalNsec = options.cudaApiTraceUpdateIntervalNsec;

            if (!ScheduleImmediateConsumeIteration())
            {
                return false;
            }

            errorGuard.Dismiss();
            return true;
        }

        bool Update()
        {
            EventIdentifier signaledIdentifier;
            bool succeeded = m_waitMultiplexer.Wait(signaledIdentifier);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "WaitMultiplexer::Wait failed");

            switch (signaledIdentifier)
            {
                case EventIdentifier::RouterNotification:
                {
                    succeeded = m_pRouterObserverClient->ReceiveNotification([this](NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) {
                        return HandleRouterNotification(notificationType, pNotification);
                    });
                    if (!succeeded)
                    {
                        NVTRACE_LOG_WRN(50, "Router observer connection lost");
                        if (!EnterRouterlessMode())
                        {
                            NVTRACE_LOG_ERR(50, "Failed to continue without router notifications");
                            return false;
                        }
                        NVTRACE_LOG_WRN(50, "Continuing in routerless mode with known provider buffers");
                    }

                    break;
                }

                case EventIdentifier::ConsumeData:
                {
                    const bool nextIterationScheduled = ScheduleNextConsumeIteration();
                    ConsumeData();
                    if (!nextIterationScheduled)
                    {
                        return false;
                    }
                    break;
                }
            }

            return true;
        }

        uint32_t GetNonMeasuringProviderCount() const
        {
            uint32_t count = 0;

            if (m_systemState.cpuClockCalibrationProvider.IsInitialized() && !m_systemState.cpuClockCalibrationProvider.observer.IsProviderMeasuring())
            {
                ++count;
            }

            for (const auto& pair : m_systemState.gpuPartitions)
            {
                const GpuPartitionState& gpuPartitionState = pair.second;
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized() && !gpuPartitionState.gpuClockCalibrationProvider.observer.IsProviderMeasuring())
                {
                    ++count;
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized() && !gpuPartitionState.gpuPmCountersProvider.observer.IsProviderMeasuring())
                {
                    ++count;
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized() && !gpuPartitionState.gpuHwTraceProvider.observer.IsProviderMeasuring())
                {
                    ++count;
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized() && !gpuPartitionState.gpuGrCtxTraceProvider.observer.IsProviderMeasuring())
                {
                    ++count;
                }
            }

            for (const auto& pair : m_systemState.processes)
            {
                const ProcessState& processState = pair.second;
                if (processState.cudaApiTraceProvider.IsInitialized() && !processState.cudaApiTraceProvider.observer.IsProviderMeasuring())
                {
                    ++count;
                }
            }

            return count;
        }

        struct Counters
        {
            uint64_t outOfOrderSampleCount = 0;
            uint64_t tooLongIntervalCount = 0;
            uint64_t providerHaltedCount = 0;
        };
        const Counters& GetCounters() const
        {
            return m_counters;
        }
        void ResetCounters()
        {
            m_counters = {};
        }

    protected:
        void OnProviderCreated_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            m_halfSecTsc = static_cast<uint64_t>(500 * 1000 * 1000 * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.sampleRateIntervalTsc = static_cast<uint64_t>(static_cast<double>(providerState.observer.GetUpdateIntervalNsec()) * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.lastSampleReadTsc = ReadCpuTscTimestamp();
        }
        void OnProviderCreated_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.sampleRateIntervalTsc = static_cast<uint64_t>(static_cast<double>(providerState.observer.GetUpdateIntervalNsec()) * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.lastSampleReadTsc = ReadCpuTscTimestamp();
        }
        void OnProviderCreated_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.sampleRateIntervalTsc = static_cast<uint64_t>(static_cast<double>(NanosecondsPerSecond / providerState.observer.GetPmSampleRateHz()) * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.lastSampleReadTsc = ReadCpuTscTimestamp();
        }
        void OnProviderCreated_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.sampleRateIntervalTsc = static_cast<uint64_t>(static_cast<double>(m_gpuHwTraceUpdateIntervalNsec) * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.lastSampleReadTsc = ReadCpuTscTimestamp();
        }
        void OnProviderCreated_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.sampleRateIntervalTsc = static_cast<uint64_t>(static_cast<double>(m_gpuGrCtxTraceUpdateIntervalNsec) * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.lastSampleReadTsc = ReadCpuTscTimestamp();
        }
        void OnProviderCreated_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            providerState.sampleRateIntervalTsc = static_cast<uint64_t>(static_cast<double>(m_cudaApiTraceUpdateIntervalNsec) * m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor);
            providerState.lastSampleReadTsc = ReadCpuTscTimestamp();
        }
        void OnProviderDestroyed_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            // Ensure any pending data is consumed before destruction
            ConsumeData_CpuClockCalibration(providerState, ReadEndpoints{ReadCpuTscTimestamp(), 0});
        }
        void OnProviderDestroyed_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            // Ensure any pending data is consumed before destruction
            const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
            ConsumeData_GpuClockCalibration(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
        }
        void OnProviderDestroyed_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            // Ensure any pending data is consumed before destruction
            const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
            ConsumeData_GpuPmCounters(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
        }
        void OnProviderDestroyed_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            // Ensure any pending data is consumed before destruction
            const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
            ConsumeData_GpuHwTrace(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
        }
        void OnProviderDestroyed_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            // Ensure any pending data is consumed before destruction
            const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
            ConsumeData_GpuGrCtxTrace(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
        }
        template <typename TProviderState>
        void ValidateSampleTiming(TProviderState& providerState, uint64_t currentTsc, const char* providerName)
        {
            if (providerState.lastSampleWriteTsc != 0)
            {
                if (currentTsc < providerState.lastSampleWriteTsc)
                {
                    NVTRACE_LOG_ERR(50, "%s: out-of-order sample went backwards by %llu ns", providerName, static_cast<uint64_t>((providerState.lastSampleWriteTsc - currentTsc) * m_systemState.initialTscRtcCalibrationPoint.tscToUtcFactor));
                    ++m_counters.outOfOrderSampleCount;
                }
                else if (currentTsc - providerState.lastSampleWriteTsc > 2 * providerState.sampleRateIntervalTsc)
                {
                    NVTRACE_LOG_ERR(50, "%s: too long an interval between samples with duration of %llu ns", providerName, static_cast<uint64_t>((currentTsc - providerState.lastSampleWriteTsc) * m_systemState.initialTscRtcCalibrationPoint.tscToUtcFactor));
                    ++m_counters.tooLongIntervalCount;
                }
            }
        }

        bool ShouldReportProviderHalted(const ProviderState& providerState, uint64_t sampleReadTsc) const
        {
            const uint64_t twiceSampleRateIntervalTsc = 2 * providerState.sampleRateIntervalTsc;
            const uint64_t haltThresholdTsc = m_halfSecTsc < twiceSampleRateIntervalTsc ? twiceSampleRateIntervalTsc : m_halfSecTsc;
            return !providerState.providerHalted && haltThresholdTsc < sampleReadTsc - providerState.lastSampleReadTsc;
        }

        void ConsumeData_CpuClockCalibration(CpuClockCalibrationProviderState& providerState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.sampleRateIntervalTsc == 0)
            {
                return;
            }

            uint64_t sampleReadTsc = ReadCpuTscTimestamp();

            uint64_t calibrationPointTscTick = 0;
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    calibrationPointTscTick = 0;
                    if (recordType != NvTrace_CPU_ClockCalibration_TscRtc_RecordType_CalibrationPoint || payloadSize < NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint_MINOR_VERSION_A)
                    {
                        return;
                    }

                    calibrationPointTscTick = reinterpret_cast<const NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint*>(pPayload)->tscTick;
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed && calibrationPointTscTick != 0)
                    {
                        ValidateSampleTiming(providerState, calibrationPointTscTick, "CpuClockCalibration");
                        providerState.lastSampleWriteTsc = calibrationPointTscTick;
                        providerState.lastSampleReadTsc = sampleReadTsc;
                        providerState.providerHalted = false;
                    }

                    calibrationPointTscTick = 0;
                });

            if (ShouldReportProviderHalted(providerState, sampleReadTsc))
            {
                NVTRACE_LOG_ERR(50, "CpuClockCalibration: no sample observed for a while, provider appears to be halted");
                providerState.providerHalted = true;
                ++m_counters.providerHaltedCount;
            }
        }

        void ConsumeData_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.sampleRateIntervalTsc == 0)
            {
                return;
            }

            uint64_t sampleReadTsc = ReadCpuTscTimestamp();

            uint64_t calibrationPointTscTick = 0;
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    calibrationPointTscTick = 0;
                    if (recordType != NvTrace_GPU_ClockCalibration_TscGpu_RecordType_CalibrationPoint || payloadSize < NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint_MINOR_VERSION_A)
                    {
                        return;
                    }

                    calibrationPointTscTick = reinterpret_cast<const NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint*>(pPayload)->tscTick;
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed && calibrationPointTscTick != 0)
                    {
                        ValidateSampleTiming(providerState, calibrationPointTscTick, "GpuClockCalibration");
                        providerState.lastSampleWriteTsc = calibrationPointTscTick;
                        providerState.lastSampleReadTsc = sampleReadTsc;
                        providerState.providerHalted = false;
                    }

                    calibrationPointTscTick = 0;
                });

            if (ShouldReportProviderHalted(providerState, sampleReadTsc))
            {
                NVTRACE_LOG_ERR(50, "GpuClockCalibration: no sample observed for a while, provider appears to be halted");
                providerState.providerHalted = true;
                ++m_counters.providerHaltedCount;
            }
        }

        void ConsumeData_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.sampleRateIntervalTsc == 0)
            {
                return;
            }

            uint64_t sampleReadTsc = ReadCpuTscTimestamp();

            uint64_t currentSampleWriteTsc = 0;
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                // fnReadSample
                [&currentSampleWriteTsc](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    currentSampleWriteTsc = 0;
                    if (recordType != NvTrace_GPU_PmCounters_RecordType_CounterSample)
                    {
                        return;
                    }
                    if (payloadSize < NvTrace_GPU_PmCounters_CounterSample_MINOR_VERSION_A)
                    {
                        return;
                    }

                    currentSampleWriteTsc = reinterpret_cast<const NvTrace_GPU_PmCounters_CounterSample*>(pPayload)->endTimestampTsc;
                },
                // fnCommitSample
                [this, &providerState, &currentSampleWriteTsc, sampleReadTsc](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed && currentSampleWriteTsc != 0)
                    {
                        ValidateSampleTiming(providerState, currentSampleWriteTsc, "GpuPmCounters");
                        providerState.lastSampleWriteTsc = currentSampleWriteTsc;
                        providerState.lastSampleReadTsc = sampleReadTsc;
                        providerState.providerHalted = false;
                    }

                    currentSampleWriteTsc = 0;
                });

            if (ShouldReportProviderHalted(providerState, sampleReadTsc))
            {
                NVTRACE_LOG_ERR(50, "GpuPmCounters: no sample observed for a while, provider appears to be halted");
                providerState.providerHalted = true;
                ++m_counters.providerHaltedCount;
            }
        }

        void ConsumeData_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.sampleRateIntervalTsc == 0)
            {
                return;
            }

            uint64_t sampleReadTsc = ReadCpuTscTimestamp();

            // The latest calibration point converts each event's ptimer back to TSC for the validation below.
            TscGpuCalibrationPoint calibrationPoint;
            const bool hasCalibration = parentState.gpuClockCalibrationProvider.IsInitialized() && parentState.gpuClockCalibrationProvider.observer.GetLatestCalibrationPoint(calibrationPoint);

            uint64_t currentEventWriteTsc = 0;
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                // fnReadSample
                [&currentEventWriteTsc, &calibrationPoint, hasCalibration](uint64_t /* sequenceNumber */, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    currentEventWriteTsc = 0;
                    if (recordType != NvTrace_GPU_HwTrace_RecordType_ComputeGridStart
                        && recordType != NvTrace_GPU_HwTrace_RecordType_ComputeGridEnd
                        && recordType != NvTrace_GPU_HwTrace_RecordType_ContextChangeLoad
                        && recordType != NvTrace_GPU_HwTrace_RecordType_ContextChangeSave
                        && recordType != NvTrace_GPU_HwTrace_RecordType_ComputeGridNullEnd)
                    {
                        return;
                    }

                    // Every HW payload leads with its GPU-ptimer timestamp by construction; a shorter payload is a
                    // protocol violation.
                    if (payloadSize < sizeof(uint64_t))
                    {
                        NVTRACE_LOG_ERR(50, "GpuHwTrace event payload is too small to contain its leading ptimer (payloadSize=%zu)", payloadSize);
                        return;
                    }

                    // Convert the leading GPU-ptimer timestamp back to TSC for validation.
                    uint64_t eventPtimer;
                    memcpy(&eventPtimer, pPayload, sizeof(eventPtimer));
                    // With no calibration we cannot convert the ptimer to TSC; feed UINT64_MAX rather than a raw ptimer masquerading as a TSC.
                    currentEventWriteTsc = hasCalibration ? calibrationPoint.ToTsc(eventPtimer) : UINT64_MAX;
                },
                // fnCommitSample
                [this, &providerState, &currentEventWriteTsc, sampleReadTsc](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed && currentEventWriteTsc != 0)
                    {
                        ValidateSampleTiming(providerState, currentEventWriteTsc, "GpuHwTrace");
                        providerState.lastSampleWriteTsc = currentEventWriteTsc;
                        providerState.lastSampleReadTsc = sampleReadTsc;
                        providerState.providerHalted = false;
                    }

                    currentEventWriteTsc = 0;
                });

            if (ShouldReportProviderHalted(providerState, sampleReadTsc))
            {
                NVTRACE_LOG_ERR(50, "GpuHwTrace: no event observed for a while, provider appears to be halted");
                providerState.providerHalted = true;
                ++m_counters.providerHaltedCount;
            }
        }

        void ConsumeData_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.sampleRateIntervalTsc == 0)
            {
                return;
            }

            uint64_t sampleReadTsc = ReadCpuTscTimestamp();

            // The latest calibration point converts each event's ptimer back to TSC for the validation below.
            TscGpuCalibrationPoint calibrationPoint = {};
            const bool hasCalibration = parentState.gpuClockCalibrationProvider.IsInitialized() && parentState.gpuClockCalibrationProvider.observer.GetLatestCalibrationPoint(calibrationPoint);

            uint64_t currentEventWriteTsc = 0;
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                // fnReadEvent
                [&currentEventWriteTsc, &calibrationPoint, hasCalibration](uint64_t /* sequenceNumber */, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordType != NvTrace_GPU_GrCtxTrace_RecordType_ContextLoadEnd && recordType != NvTrace_GPU_GrCtxTrace_RecordType_ContextSaveEnd)
                    {
                        return;
                    }
                    if (payloadSize < NvTrace_GPU_GrCtxTrace_ContextChange_MINOR_VERSION_A)
                    {
                        return;
                    }

                    uint64_t timestampGpuNsec = reinterpret_cast<const NvTrace_GPU_GrCtxTrace_ContextChange*>(pPayload)->timestampGpuNsec;
                    currentEventWriteTsc = hasCalibration ? calibrationPoint.ToTsc(timestampGpuNsec) : UINT64_MAX;
                },
                // fnCommitEvent
                [this, &providerState, &currentEventWriteTsc, sampleReadTsc](bool consistencyCheckPassed) {
                    if (!consistencyCheckPassed || currentEventWriteTsc == 0)
                    {
                        return;
                    }

                    ValidateSampleTiming(providerState, currentEventWriteTsc, "GpuGrCtxTrace");
                    providerState.lastSampleWriteTsc = currentEventWriteTsc;
                    providerState.lastSampleReadTsc = sampleReadTsc;
                    providerState.providerHalted = false;
                    currentEventWriteTsc = 0;
                });

            if (ShouldReportProviderHalted(providerState, sampleReadTsc))
            {
                NVTRACE_LOG_ERR(50, "GpuGrCtxTrace: no event observed for a while, provider appears to be halted");
                providerState.providerHalted = true;
                ++m_counters.providerHaltedCount;
            }
        }

    protected:
        bool BuildWaitMultiplexer()
        {
            constexpr uint32_t eventCount = 2;
            bool succeeded = m_waitMultiplexer.Initialize(eventCount);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "WaitMultiplexer::Initialize failed");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::RouterNotification, m_pRouterObserverClient->GetWaitForPendingNotificationHandle());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "WaitMultiplexer::AddObject");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::ConsumeData, m_consumeTimer.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "WaitMultiplexer::AddObject");
            return true;
        }

        bool EnterRouterlessMode()
        {
            bool succeeded = m_waitMultiplexer.RemoveLastSignaledObject();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "WaitMultiplexer::RemoveLastSignaledObject failed");
            m_pRouterObserverClient.Reset();
            return true;
        }

    private:
        enum class EventIdentifier
        {
            RouterNotification,
            ConsumeData
        };
        WaitMultiplexer m_waitMultiplexer;

        uint64_t m_gpuHwTraceUpdateIntervalNsec = 0;
        uint64_t m_gpuGrCtxTraceUpdateIntervalNsec = 0;
        uint64_t m_cudaApiTraceUpdateIntervalNsec = 0;
        uint64_t m_halfSecTsc = 0;

        Counters m_counters = {};
    };

} // namespace nv::trace
