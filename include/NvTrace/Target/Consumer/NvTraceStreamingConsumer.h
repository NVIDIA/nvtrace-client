/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceConsumerBase.h>
#include <NvTrace/Target/Consumer/NvTraceCudaApiTraceSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceNvtxTraceSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceRecordWriterUtilities.h>
#include <NvTrace/Target/Consumer/NvTraceStreamDatabaseWriter.h>
#include <NvTrace/Target/Consumer/NvTraceThreadEventSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceTsfBufferEventEncoders.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <system_error>
#include <thread>
#include <utility>

namespace nv::trace {

    struct StreamingConsumerObjectModel : ConsumerObjectModelBase
    {
        struct ProviderState : ConsumerObjectModelBase::ProviderState
        {
            uint64_t providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            std::vector<uint64_t> onAttachRecordLocations;
            uint32_t providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            bool isReadyToConsume = true;

            void Reset() noexcept
            {
                threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                onAttachRecordLocations.clear();
                providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
                providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
                isReadyToConsume = true;
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

        template <typename TGpuClockCalibrationProviderState, typename TGpuPmCountersProviderState, typename TGpuHwTraceProviderState, typename TGpuGrCtxTraceProviderState>
        struct GpuPartitionState : ConsumerObjectModelBase::GpuPartitionState<TGpuClockCalibrationProviderState, TGpuPmCountersProviderState, TGpuHwTraceProviderState, TGpuGrCtxTraceProviderState>
        {
            uint64_t gpuPartitionCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t observationStartGpuNsec = 0;
            bool isObservationStartResolved = true;
        };

        template <typename TCudaApiTraceProviderState, typename TNvtxTraceProviderState, typename TThreadEventProviderState>
        struct ProcessState : ConsumerObjectModelBase::ProcessState<TCudaApiTraceProviderState, TNvtxTraceProviderState, TThreadEventProviderState>
        {
            uint64_t processCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        };

        template <typename TCpuClockCalibrationProviderState, typename TProviderState, typename TGpuPartitionState, typename TProcessState>
        struct SystemState : ConsumerObjectModelBase::SystemState<TCpuClockCalibrationProviderState, TProviderState, TGpuPartitionState, TProcessState>
        {
            uint64_t systemCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
        };
    };

    template <typename TStorageWriter>
    struct StreamingConsumerTestAccessor;

    template <typename TStorageWriter>
    class StreamingConsumer : private ConsumerBase<StreamingConsumer<TStorageWriter>, StreamingConsumerObjectModel>
    {
        using Super = ConsumerBase<StreamingConsumer, StreamingConsumerObjectModel>;
        friend Super;
        friend struct StreamingConsumerTestAccessor<TStorageWriter>;

        using typename Super::CpuClockCalibrationProviderState;
        using typename Super::CudaApiTraceProviderState;
        using typename Super::GpuClockCalibrationProviderState;
        using typename Super::GpuGrCtxTraceProviderState;
        using typename Super::GpuHwTraceProviderState;
        using typename Super::GpuPartitionState;
        using typename Super::GpuPmCountersProviderState;
        using typename Super::NvtxTraceProviderState;
        using typename Super::ProcessState;
        using typename Super::ProviderState;
        using typename Super::SystemState;
        using typename Super::ThreadEventProviderState;

    public:
        struct Options : Super::Options
        {
            size_t dataBlockSize = 10 * 1024 * 1024; // 10 MiB
            typename TraceStreamDatabaseWriter<TStorageWriter>::OnStorageWriteCompleted fnOnStorageWriteCompleted = nullptr;
            void* pUserDataOnStorageWriteCompleted = nullptr;
        };

    public:
        using Super::ConvertTscTickToUtcNsec;

        ~StreamingConsumer()
        {
            TerminateConsumeThread();
        }

        StreamingConsumer()
            : m_recordWriter(*this)
        {
        }

        StreamingConsumer(const StreamingConsumer&) = delete;
        StreamingConsumer& operator=(const StreamingConsumer&) = delete;

        void Reset() noexcept
        {
            TerminateConsumeThread();
            m_isConsuming = false;
            m_consumeThreadExited = false;
            m_waitMultiplexer.Reset();
            m_consumeThreadCommandCompletedEvent.Reset();
            m_consumeThreadCommandPendingEvent.Reset();
            m_consumeThreadCommand = ConsumeThreadCommand::Stop;
            m_recordWriter.Reset();
            m_recordTimestampUpperBoundUtcNsec = 0;
            m_recordTimestampLowerBoundUtcNsec = 0;
            m_observationStartTsc = 0;
            m_observationStartUtcNsec = 0;
            m_finalDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_recordTimestampLimitUtcNsec.store(UINT64_MAX, std::memory_order_relaxed);
            m_dataBlockWriter.Reset();
            m_databaseWriter.Reset();
            Super::Reset();
        }

        template <typename TFN_InitializeStorageWriter>
        bool Initialize(const Options& options, TFN_InitializeStorageWriter&& fnInitializeStorageWriter)
        {
            if (m_consumeThread.joinable())
            {
                return false;
            }

            NvTrace_RouterObserverClient_RouterConfig routerConfig;
            if (!Super::Initialize(options, &routerConfig))
            {
                return false;
            }

            auto resetGuard = ScopeExitGuard([this]() { Reset(); });

            typename TraceStreamDatabaseWriter<TStorageWriter>::Options databaseWriterOptions;
            databaseWriterOptions.dataBlockSize = options.dataBlockSize;
            databaseWriterOptions.maxGpuPartitionCountPerDataBlock = routerConfig.maxGpuPartitionCount * 5 / 4;
            databaseWriterOptions.maxProcessCountPerDataBlock = routerConfig.maxProcessCount * 5 / 4;
            databaseWriterOptions.maxProviderCountPerDataBlock = routerConfig.maxProviderCount * 5 / 4;
            databaseWriterOptions.maxThreadEventProviderMappingCountPerDataBlock = routerConfig.maxProviderCount * 5 / 4;
            databaseWriterOptions.chunkCountPerDataBlock = 1;
            databaseWriterOptions.writeSlotCount = 4;
            databaseWriterOptions.fnOnStorageWriteCompleted = options.fnOnStorageWriteCompleted;
            databaseWriterOptions.pUserDataOnStorageWriteCompleted = options.pUserDataOnStorageWriteCompleted;
            if (!m_databaseWriter.Initialize(databaseWriterOptions, std::forward<TFN_InitializeStorageWriter>(fnInitializeStorageWriter)))
            {
                return false;
            }

            bool succeeded = m_consumeThreadCommandPendingEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: AutoResetEvent::Initialize failed for consumeThreadCommandPendingEvent");
            succeeded = m_consumeThreadCommandCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: AutoResetEvent::Initialize failed for consumeThreadCommandCompletedEvent");

            succeeded = BuildWaitMultiplexer();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: BuildWaitMultiplexer failed");

            try
            {
                m_consumeThread = std::thread(&StreamingConsumer::ConsumeLoop, this);
            }
            catch (const std::system_error& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to create consume thread (std::system_error %d: %s).", e.code().value(), e.what());
                return false;
            }

            resetGuard.Dismiss();
            return true;
        }


        bool Start(uint64_t observationStartUtcNsec = 0)
        {
            if (!m_consumeThread.joinable() || m_consumeThreadCommand != ConsumeThreadCommand::Stop || m_consumeThreadExited)
            {
                return false;
            }

            m_recordTimestampLimitUtcNsec.store(UINT64_MAX, std::memory_order_release);
            m_observationStartUtcNsec = observationStartUtcNsec;
            m_consumeThreadCommand = ConsumeThreadCommand::Start;
            std::atomic_thread_fence(std::memory_order_release);

            bool succeeded = m_consumeThreadCommandPendingEvent.Signal();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer::Start: AutoResetEvent::Signal failed (consumeThreadCommandPendingEvent)");
            succeeded = m_consumeThreadCommandCompletedEvent.Wait();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer::Start: AutoResetEvent::Wait failed (consumeThreadCommandCompletedEvent)");

            std::atomic_thread_fence(std::memory_order_acquire);
            if (m_consumeThreadExited)
            {
                return false;
            }

            return true;
        }

        // Stop at the current timestamp after allowing delayed provider data to arrive.
        bool Stop(uint64_t* pOutFinalDataBlockSequenceNumber = nullptr)
        {
            return StopAtTsc(ReadCpuTscTimestamp(), pOutFinalDataBlockSequenceNumber);
        }

        bool StopAtTsc(uint64_t targetTsc, uint64_t* pOutFinalDataBlockSequenceNumber = nullptr)
        {
            return StopAtUtc(ConvertTscTickToUtcNsec(targetTsc), pOutFinalDataBlockSequenceNumber);
        }

        bool StopAtUtc(uint64_t targetUtcNsec, uint64_t* pOutFinalDataBlockSequenceNumber = nullptr)
        {
            if (pOutFinalDataBlockSequenceNumber)
            {
                *pOutFinalDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            }
            if (m_consumeThreadCommand != ConsumeThreadCommand::Start || m_consumeThreadExited)
            {
                return false;
            }

            // Continue consuming through the grace period without extending the requested recording timespan.
            m_recordTimestampLimitUtcNsec.store(targetUtcNsec, std::memory_order_release);
            this->ConsumeUpToUtc(targetUtcNsec);
            return StopImpl(pOutFinalDataBlockSequenceNumber);
        }

    protected:
        bool StopImpl(uint64_t* pOutFinalDataBlockSequenceNumber = nullptr)
        {
            if (pOutFinalDataBlockSequenceNumber)
            {
                *pOutFinalDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            }
            if (m_consumeThreadCommand != ConsumeThreadCommand::Start || m_consumeThreadExited)
            {
                return false;
            }

            m_consumeThreadCommand = ConsumeThreadCommand::Stop;
            std::atomic_thread_fence(std::memory_order_release);

            bool succeeded = m_consumeThreadCommandPendingEvent.Signal();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer::StopImpl: AutoResetEvent::Signal failed (consumeThreadCommandPendingEvent)");
            succeeded = m_consumeThreadCommandCompletedEvent.Wait();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer::StopImpl: AutoResetEvent::Wait failed (consumeThreadCommandCompletedEvent)");

            std::atomic_thread_fence(std::memory_order_acquire);
            if (m_consumeThreadExited)
            {
                return false;
            }

            m_isConsuming = false;
            if (pOutFinalDataBlockSequenceNumber)
            {
                *pOutFinalDataBlockSequenceNumber = m_finalDataBlockSequenceNumber;
            }
            return true;
        }

    private:
        using Super::ConsumeData;
        using Super::HandleRouterNotification;
        using Super::ScheduleImmediateConsumeIteration;
        using Super::ScheduleNextConsumeIteration;

        void OnSystemCreated(SystemState& systemState)
        {
            if (m_isConsuming)
            {
                AttachSystem(systemState);
            }
        }
        void OnGpuPartitionCreated(GpuPartitionState& gpuPartitionState)
        {
            if (m_isConsuming)
            {
                gpuPartitionState.observationStartGpuNsec = 0;
                gpuPartitionState.isObservationStartResolved = m_observationStartTsc == 0;
                AttachGpuPartition(gpuPartitionState);
            }
        }
        void OnProcessCreated(ProcessState& processState)
        {
            if (m_isConsuming)
            {
                AttachProcess(processState);
            }
        }
        void OnProviderCreated_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            if (m_isConsuming)
            {
                ConfigureCpuObservationStart(providerState);
                AttachProvider_CpuClockCalibration(providerState);
            }
        }
        void OnProviderCreated_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                ConfigureGpuClockObservationStart(providerState);
                AttachProvider_GpuClockCalibration(providerState, parentState);
            }
        }
        void OnProviderCreated_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareGpuPmCounters(providerState, parentState);
            }
        }
        void OnProviderCreated_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareGpuHwTrace(providerState, parentState);
            }
        }
        void OnProviderCreated_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareGpuGrCtxTrace(providerState, parentState);
            }
        }
        void OnProviderCreated_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                AttachProvider_CudaApiTrace(providerState, parentState);
            }
        }
        void OnProviderCreated_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                AttachProvider_NvtxTrace(providerState, parentState);
            }
        }

        void OnProviderCreated_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareThreadEvent(providerState, parentState, ReadCpuTscTimestamp());
            }
        }

        void OnGpuPartitionDestroyed(GpuPartitionState& gpuPartitionState, uint64_t destructionTimestampTsc)
        {
            if (m_isConsuming)
            {
                AttachGpuPartitionDestroyed(gpuPartitionState, destructionTimestampTsc);
            }
        }
        void OnProcessDestroyed(ProcessState& processState, uint64_t destructionTimestampTsc)
        {
            if (m_isConsuming)
            {
                AttachProcessDestroyed(processState, destructionTimestampTsc);
            }
        }
        void OnProviderDestroyed(ProviderState& providerState, ProviderDestroyReason reason)
        {
            auto itrProcessState = m_systemState.processes.find(providerState.processLuid);
            if (m_isConsuming && itrProcessState != m_systemState.processes.end())
            {
                ProcessState& parentState = itrProcessState->second;
                if (reason == ProviderDestroyReason::Explicit)
                {
                    if (ThreadEventProviderState* const pThreadEventProviderState = parentState.FindThreadEventProvider(providerState.threadEventProviderLuid))
                    {
                        ConsumeData_ThreadEvent(*pThreadEventProviderState, parentState, ReadCpuTscTimestamp());
                    }
                }
            }
        }
        void OnProviderDestroyed_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                ConsumeData_CpuClockCalibration(providerState, ReadEndpoints{ReadCpuTscTimestamp(), 0});
            }
        }
        void OnProviderDestroyed_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuClockCalibration(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuPmCounters(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuHwTrace(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuGrCtxTrace(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_CudaApiTrace(providerState, parentState, readEndpointTsc);
            }
        }
        void OnProviderDestroyed_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            (void)providerState;
            (void)parentState;
        }
        void OnProviderDestroyed_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                // Only reached when the owning process is destroyed. No producer can still be writing, so drain every
                // cell, otherwise the final cells before process exit are lost.
                ConsumeData_ThreadEvent(providerState, parentState, UINT64_MAX);
            }
        }

        void ConfigureExistingObservationStarts()
        {
            if (m_systemState.cpuClockCalibrationProvider.IsInitialized())
            {
                ConfigureCpuObservationStart(m_systemState.cpuClockCalibrationProvider);
            }
            for (auto& [_, gpuPartitionState] : m_systemState.gpuPartitions)
            {
                gpuPartitionState.observationStartGpuNsec = 0;
                gpuPartitionState.isObservationStartResolved = m_observationStartTsc == 0;
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                {
                    ConfigureGpuClockObservationStart(gpuPartitionState.gpuClockCalibrationProvider);
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                {
                    gpuPartitionState.gpuPmCountersProvider.isReadyToConsume = false;
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                {
                    gpuPartitionState.gpuHwTraceProvider.isReadyToConsume = false;
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                {
                    gpuPartitionState.gpuGrCtxTraceProvider.isReadyToConsume = false;
                }
            }
            for (auto& [_, processState] : m_systemState.processes)
            {
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    threadEventProviderEntry.second.isReadyToConsume = false;
                }
            }
        }

        void ConfigureCpuObservationStart(CpuClockCalibrationProviderState& providerState)
        {
            providerState.observer.SetObservationStart(ReadEndpoints{m_observationStartTsc, 0});
        }

        void ConfigureGpuClockObservationStart(GpuClockCalibrationProviderState& providerState)
        {
            providerState.observer.SetObservationStart(ReadEndpoints{m_observationStartTsc, 0});
        }

        bool TryResolveGpuObservationStart(GpuPartitionState& parentState)
        {
            if (parentState.isObservationStartResolved)
            {
                return true;
            }

            TscGpuCalibrationPoint calibrationPoint;
            if (!parentState.gpuClockCalibrationProvider.IsInitialized() || !parentState.gpuClockCalibrationProvider.observer.GetLatestCalibrationPoint(calibrationPoint))
            {
                return false;
            }
            parentState.observationStartGpuNsec = calibrationPoint.ToGpu(m_observationStartTsc);
            parentState.isObservationStartResolved = true;
            return true;
        }

        template <typename TProviderState>
        bool TryPrepareGpuProvider(
            TProviderState& providerState,
            GpuPartitionState& parentState,
            void (StreamingConsumer::*fnAttachProvider)(TProviderState&, GpuPartitionState&))
        {
            if (providerState.isReadyToConsume)
            {
                return true;
            }
            if (!TryResolveGpuObservationStart(parentState))
            {
                return false;
            }

            providerState.observer.SetObservationStart(ReadEndpoints{m_observationStartTsc, parentState.observationStartGpuNsec});
            (this->*fnAttachProvider)(providerState, parentState);
            providerState.isReadyToConsume = providerState.providerCreatedRecordLocation != TSF_INVALID_RECORD_LOCATION;
            return providerState.isReadyToConsume;
        }

        bool TryPrepareGpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            return TryPrepareGpuProvider(providerState, parentState, &StreamingConsumer::AttachProvider_GpuPmCounters);
        }

        bool TryPrepareGpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            return TryPrepareGpuProvider(providerState, parentState, &StreamingConsumer::AttachProvider_GpuHwTrace);
        }

        bool TryPrepareGpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            return TryPrepareGpuProvider(providerState, parentState, &StreamingConsumer::AttachProvider_GpuGrCtxTrace);
        }

        void ConsumeData_CpuClockCalibration(CpuClockCalibrationProviderState& providerState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.providerCreatedRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                return; // Not an error.
            }

            m_recordWriter.SetSystemProviderRef(providerState);

            // Retry only while an on-attach record remains pending. If another record becomes Ready, publish a new
            // ProviderAttached snapshot so subsequent provider references use the expanded attachment set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::CpuClockCalibration>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                    m_recordWriter.SetSystemProviderRef(providerState);
                }
            }

            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (!Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CpuClockCalibration>(
                            m_recordWriter,
                            recordType,
                            sequenceNumber,
                            pPayload,
                            payloadSize))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write CPU clock calibration record.");
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (providerState.providerCreatedRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                return; // Not an error.
            }
            m_recordWriter.SetGpuProviderRef(providerState, parentState);

            // Retry only while an on-attach record remains pending. If another record becomes Ready, publish a new
            // ProviderAttached snapshot so subsequent provider references use the expanded attachment set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuClockCalibration>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                    m_recordWriter.SetGpuProviderRef(providerState, parentState);
                }
            }

            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (!Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuClockCalibration>(
                            m_recordWriter,
                            recordType,
                            sequenceNumber,
                            pPayload,
                            payloadSize))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU clock calibration record.");
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!TryPrepareGpuPmCounters(providerState, parentState))
            {
                return;
            }
            // Retry only while an on-attach DataBlob remains pending. If another record becomes Ready, publish a new
            // ProviderAttached snapshot so subsequent provider references use the expanded attachment set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                m_recordWriter.SetGpuProviderRef(providerState, parentState);
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuPmCounters>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                }
            }

            m_recordWriter.SetGpuProviderRef(providerState, parentState);

            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    const bool succeeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuPmCounters>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!succeeded)
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU PM counters recurring record (providerLuid=%" PRIu64 " sequenceNumber=%" PRIu64 " recordType=%" PRIu32 ").", providerState.luid, sequenceNumber, recordType);
                        return;
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!TryPrepareGpuHwTrace(providerState, parentState))
            {
                return;
            }
            // Retry only while an on-attach DataBlob remains pending. Publish a new ProviderAttached snapshot when the
            // attachment set grows so subsequent provider references resolve the complete set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                m_recordWriter.SetGpuProviderRef(providerState, parentState);
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuHwTrace>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                }
            }

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    const bool succeeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuHwTrace>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!succeeded)
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU HW trace recurring record (providerLuid=%" PRIu64 " sequenceNumber=%" PRIu64 " recordType=%" PRIu32 ").", providerState.luid, sequenceNumber, recordType);
                        return;
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!TryPrepareGpuGrCtxTrace(providerState, parentState))
            {
                return;
            }
            // Retry only while an on-attach DataBlob remains pending. Publish a new ProviderAttached snapshot when the
            // attachment set grows so subsequent provider references resolve the complete set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                m_recordWriter.SetGpuProviderRef(providerState, parentState);
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuGrCtxTrace>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                }
            }

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    const bool succeeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuGrCtxTrace>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!succeeded)
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU GR context trace recurring record (providerLuid=%" PRIu64 " sequenceNumber=%" PRIu64 " recordType=%" PRIu32 ").", providerState.luid, sequenceNumber, recordType);
                        return;
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc)
        {
            (void)readEndpointTsc;
            if (providerState.providerCreatedRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                return; // Not an error.
            }

            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            providerState.observer.ConsumeGpuTimestampEvents(
                [&](uint64_t gpuTimestampRef, uint64_t ptimer) {
                    const NvTrace_CUDA_ApiTrace_GpuTimestampReport report = {gpuTimestampRef, ptimer};
                    const bool reportWritten = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CudaApiTrace>(m_recordWriter, NvTrace_CUDA_ApiTrace_RecordType_GpuTimestampReport, NVTRACE_NO_CORRELATION_ID, reinterpret_cast<const uint8_t*>(&report), sizeof(report));
                    if (reportWritten)
                    {
                        m_recordWriter.Commit();
                    }
                });
        }

        void ConsumeData_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc)
        {
            if (!TryPrepareThreadEvent(providerState, parentState, readEndpointTsc))
            {
                return;
            }

            Super::ConsumeThreadEventSourceProviderRegistrations(parentState, providerState, readEndpointTsc);

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            // Ensure sufficient inline payload capacity; this assert is specific to TraceStreamDatabaseWriter and should be removed when dependence on the inline payload is removed.
            static_assert(sizeof(Codec::NvtxTraceEvents::ThreadEventInlinePayload) <= TSF_RECORD_INLINE_PAYLOAD_SIZE, "NVTX Trace event inline payload exceeds TSF inline payload capacity");

            providerState.observer.ConsumeThreadEvents(
                readEndpointTsc,
                [&](uint32_t sourceProviderIndex, const ThreadEventRecord& record) -> bool {
                    return Super::ConsumeThreadEvent(parentState, providerState, sourceProviderIndex, record, readEndpointTsc);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        bool TryPrepareThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc)
        {
            if (providerState.isReadyToConsume)
            {
                return true;
            }
            if (providerState.providerCreatedRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                AttachProvider_ThreadEvent(providerState, parentState);
                if (providerState.providerCreatedRecordLocation == TSF_INVALID_RECORD_LOCATION)
                {
                    return false;
                }
            }
            if (m_observationStartTsc == 0)
            {
                providerState.isReadyToConsume = true;
                return true;
            }

            const uint64_t warmupEndpointTsc = std::min(readEndpointTsc, m_observationStartTsc - 1);

            // Consume complete FIFO cells preceding the observation window so object state remains
            // reconstructible, but leave the first straddling cell for the normal consume path. Nothing
            // is held here: the skip below would discard a held record and everything behind it.
            {
                const auto dropUnboundGuard = Super::ScopeGuardDropUnboundThreadEvents();
                providerState.observer.ConsumeThreadEvents(
                    warmupEndpointTsc,
                    [&](uint32_t sourceProviderIndex, const ThreadEventRecord& record) -> bool {
                        if (!Super::IsObjectLifetimeThreadEvent(providerState, sourceProviderIndex, record))
                        {
                            return true;
                        }
                        return Super::ConsumeThreadEvent(parentState, providerState, sourceProviderIndex, record, warmupEndpointTsc);
                    },
                    [&](bool consistencyCheckPassed) {
                        if (consistencyCheckPassed)
                        {
                            m_recordWriter.Commit();
                        }
                        else
                        {
                            m_recordWriter.Rollback();
                        }
                    });
            }

            if (readEndpointTsc < m_observationStartTsc || !providerState.observer.SkipEventBufferEntriesBefore(m_observationStartTsc))
            {
                return false;
            }
            providerState.isReadyToConsume = true;
            return true;
        }

        void RegisterThreadEventConsumeCallbacks()
        {
            Super::RegisterThreadEventConsumeCallback(NvTrace_ThreadEvent_ProviderTypeUuid, &StreamingConsumer::ConsumeThreadEvent_ThreadEventCallback);
            Super::RegisterThreadEventConsumeCallback(NvTrace_CUDA_ApiTrace_ProviderTypeUuid, &StreamingConsumer::ConsumeThreadEvent_CudaApiTraceCallback);
            Super::RegisterThreadEventConsumeCallback(NvTrace_NvtxTrace_ProviderTypeUuid, &StreamingConsumer::ConsumeThreadEvent_NvtxTraceCallback);
        }

        static bool ConsumeThreadEvent_ThreadEventCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            StreamingConsumer& consumer = *reinterpret_cast<StreamingConsumer*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            ThreadEventProviderState& providerState = *static_cast<ThreadEventProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            switch (record.recordType)
            {
                case NvTrace_ReservedRecordType_RecordReference:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < sizeof(NvTrace_RecordReference), true, 50, "Invalid record reference ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

                    NvTrace_RecordReference reference;
                    memcpy(&reference, record.pPayload, sizeof(reference));

                    bool eventResolved = false;
                    bool eventSucceeded = false;
                    providerState.observer.ResolveThreadStateReference(
                        reference,
                        [&](const NvTrace_ThreadEvent_ThreadCreated& threadCreated) {
                            eventResolved = true;
                            eventSucceeded = Codec::ThreadEventEncoders::WriteThreadCreatedRecord(consumer.m_recordWriter, threadCreated);
                        },
                        [&](bool consistencyCheckPassed) { CommitOrRollback(consumer.m_recordWriter, consistencyCheckPassed && eventSucceeded); });
                    return !eventResolved || eventSucceeded;
                }
                case NvTrace_ThreadEvent_RecordType_ThreadCreated:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < sizeof(NvTrace_ThreadEvent_ThreadCreated), true, 50, "Invalid thread created ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

                    NvTrace_ThreadEvent_ThreadCreated threadCreated;
                    memcpy(&threadCreated, record.pPayload, sizeof(threadCreated));
                    return Codec::ThreadEventEncoders::WriteThreadCreatedRecord(consumer.m_recordWriter, threadCreated);
                }
                case NvTrace_ThreadEvent_RecordType_ThreadDestroyed:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < sizeof(NvTrace_ThreadEvent_ThreadDestroyed), true, 50, "Invalid thread destroyed ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

                    NvTrace_ThreadEvent_ThreadDestroyed threadDestroyed;
                    memcpy(&threadDestroyed, record.pPayload, sizeof(threadDestroyed));
                    return Codec::ThreadEventEncoders::WriteThreadDestroyedRecord(consumer.m_recordWriter, threadDestroyed, record.uniqueThreadId);
                }
                default:
                    return true;
            }
        }

        static bool ConsumeThreadEvent_CudaApiTraceCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            StreamingConsumer& consumer = *reinterpret_cast<StreamingConsumer*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            CudaApiTraceProviderState& providerState = *static_cast<CudaApiTraceProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            CudaApiTraceObserver::ThreadEventRecordStorage recordStorage;
            const ThreadEventRecord* const pRecord = providerState.observer.TranslateThreadEventRecord(record, recordStorage);
            if (!pRecord)
            {
                return true;
            }

            if (pRecord->recordType == NvTrace_ReservedRecordType_RecordReference)
            {
                NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < sizeof(NvTrace_RecordReference), true, 50, "Invalid record reference ThreadEvent record (recordType=%u, payloadSize=%zu).", pRecord->recordType, pRecord->payloadSize);

                NvTrace_RecordReference reference;
                memcpy(&reference, pRecord->pPayload, sizeof(reference));

                return CudaApiTraceSerialization::SerializeReferencedStateEvent(providerState.observer, consumer.m_recordWriter, reference);
            }

            const auto recordType = static_cast<NvTrace_CUDA_ApiTrace_RecordType>(pRecord->recordType);
            // The records written below ride the delivering thread event's own commit; committing
            // here would publish them before that record's consistency verdict.
            switch (recordType)
            {
                case NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA context-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA stream-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A, true, 50, "Invalid CUDA module-unloaded ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA graph-exec-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                default:
                    return Codec::CudaApiTraceEventEncoders::WriteApiTraceEventRecord(consumer.m_recordWriter, *pRecord);
            }

            const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CudaApiTrace>(consumer.m_recordWriter, recordType, pRecord->correlationId, pRecord->pPayload, pRecord->payloadSize);
            NVTRACE_LOG_ERR_IF(!written, 50, "Failed WriteRecord() for CUDA object-destroyed record (recordType=%u).", static_cast<uint32_t>(recordType));
            return written;
        }

        static bool ConsumeThreadEvent_NvtxTraceCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            StreamingConsumer& consumer = *reinterpret_cast<StreamingConsumer*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            NvtxTraceProviderState& providerState = *static_cast<NvtxTraceProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            NvtxTraceObserver::ThreadEventRecordStorage recordStorage;
            const ThreadEventRecord* const pRecord = providerState.observer.TranslateThreadEventRecord(record, recordStorage);
            if (!pRecord)
            {
                return true;
            }

            if (pRecord->recordType == NvTrace_ReservedRecordType_RecordReference)
            {
                NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < sizeof(NvTrace_RecordReference), true, 50, "Invalid record reference ThreadEvent record (recordType=%u, payloadSize=%zu).", pRecord->recordType, pRecord->payloadSize);

                NvTrace_RecordReference reference;
                memcpy(&reference, pRecord->pPayload, sizeof(reference));

                return NvtxTraceSerialization::SerializeReferencedStateEvent(providerState.observer, consumer.m_recordWriter, reference);
            }

            const auto recordType = static_cast<NvTrace_NvtxTrace_RecordType>(pRecord->recordType);
            // The records written below ride the delivering thread event's own commit; committing
            // here would publish them before that record's consistency verdict.
            switch (recordType)
            {
                case NvTrace_NvtxTrace_RecordType_StackGroupDestroyed:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A, true, 50, "Invalid NVTX stack-group-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);

                    const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::NvtxTrace>(consumer.m_recordWriter, recordType, pRecord->correlationId, pRecord->pPayload, pRecord->payloadSize);
                    NVTRACE_LOG_ERR_IF(!written, 50, "Failed WriteRecord() for NVTX stack-group-destroyed record.");
                    return written;
                }
                case NvTrace_NvtxTrace_RecordType_Mark:
                case NvTrace_NvtxTrace_RecordType_MarkMessageSegment:
                case NvTrace_NvtxTrace_RecordType_RangePop:
                case NvTrace_NvtxTrace_RecordType_DroppedRangePush:
                case NvTrace_NvtxTrace_RecordType_DroppedRangePop:
                    return Codec::NvtxTraceEventEncoders::WriteThreadEventRecord(consumer.m_recordWriter, *pRecord);
                default:
                    NVTRACE_LOG_ERR(50, "Unknown NVTX Trace ThreadEvent record type %u.", pRecord->recordType);
                    return true;
            }
        }

    private:
        void AttachSystem(SystemState& systemState)
        {
            m_systemState.systemCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteSystemCreatedRecord(m_recordWriter, systemState.systemInfo))
            {
                return;
            }
            m_recordWriter.Commit();
            m_systemState.systemCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();
            m_dataBlockWriter.SetSystemInfoRecordLocation(m_systemState.systemCreatedRecordLocation);
            m_databaseWriter.SetSystemInfoRecordLocation(m_systemState.systemCreatedRecordLocation);
        }

        void AttachGpuPartition(GpuPartitionState& gpuPartitionState)
        {
            gpuPartitionState.gpuPartitionCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteGpuPartitionCreatedRecord(m_recordWriter, gpuPartitionState.luid, gpuPartitionState.gpuUuid, gpuPartitionState.creationTimestampTsc, gpuPartitionState.gpuPartitionInfo))
            {
                return;
            }
            m_recordWriter.Commit();
            gpuPartitionState.gpuPartitionCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();
        }
        void AttachGpuPartitionDestroyed(GpuPartitionState& gpuPartitionState, uint64_t destructionTimestampTsc)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteGpuPartitionDestroyedRecord(m_recordWriter, gpuPartitionState.luid, destructionTimestampTsc))
            {
                return;
            }
            m_recordWriter.Commit();
        }


        void AttachProcess(ProcessState& processState)
        {
            processState.processCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProcessCreatedRecord(m_recordWriter, processState.luid, processState.namespacePid, processState.globalPid, processState.pidNamespaceId, processState.path.c_str(), processState.creationTimestampTsc))
            {
                return;
            }
            m_recordWriter.Commit();
            processState.processCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();
        }

        void AttachProcessDestroyed(ProcessState& processState, uint64_t destructionTimestampTsc)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProcessDestroyedRecord(m_recordWriter, processState.luid, destructionTimestampTsc))
            {
                return;
            }
            m_recordWriter.Commit();
        }

        template <Codec::ProviderTypeId TProviderTypeId, typename TProviderState>
        bool TryConsumeOnAttachRecords(TProviderState& providerState)
        {
            const size_t previousRecordCount = providerState.onAttachRecordLocations.size();

            providerState.observer.ConsumeOnAttachDataBlobRecords([&](uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                if (!Codec::EventEncoders::WriteRecord<TProviderTypeId>(m_recordWriter, recordType, NVTRACE_NO_CORRELATION_ID, pPayload, payloadSize))
                {
                    NVTRACE_LOG_ERR(50, "Failed to write %s on-attach record (providerLuid=%" PRIu64 " recordType=%" PRIu32 ").", Codec::ToCString(TProviderTypeId), providerState.luid, recordType);
                    return;
                }

                m_recordWriter.Commit();
                providerState.onAttachRecordLocations.push_back(m_recordWriter.GetLastCommittedRecordLocation());
            });

            return previousRecordCount != providerState.onAttachRecordLocations.size();
        }

        template <typename TProviderState>
        void EmitProviderAttachedRecord(TProviderState& providerState)
        {
            if (!Codec::NvTraceServiceEncoders::WriteProviderAttachedRecord(m_recordWriter, providerState.luid, providerState.onAttachRecordLocations))
            {
                return;
            }

            m_recordWriter.Commit();
            providerState.providerAttachedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            // The consumer generated a new consumer-owned ProviderAttached record with new on-attach records being ready.
            // Invalidate the cached ProviderRef so the next provider data record creates a TsfProviderRef containing the
            // existing ProviderCreated location and the new ProviderAttached location.
            providerState.dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        }

        void AttachProvider_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, NVTRACE_NIL_LUID, NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid, "CPU Clock Calibration"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetSystemProviderRef(providerState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::CpuClockCalibration>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, "GPU Clock Calibration"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuClockCalibration>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_PmCounters_ProviderTypeUuid, "GPU PM Counters"))
            {
                return;
            }
            m_recordWriter.Commit();

            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuPmCounters>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_HwTrace_ProviderTypeUuid, "GPU HW Trace"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuHwTrace>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_GrCtxTrace_ProviderTypeUuid, "GPU GR Context Trace"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuGrCtxTrace>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();

            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_CUDA_ApiTrace_ProviderTypeUuid, "CUDA API Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            // The on-attach records lead every other state record: the DataModel needs the devices to
            // place the resources that follow under them, and a partial load reaches them through the
            // ProviderAttached record alone.
            CudaApiTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_recordWriter, [&] {
                providerState.onAttachRecordLocations.push_back(m_recordWriter.GetLastCommittedRecordLocation());
            });

            // CUDA keeps writing provider-owned records after ProviderAttached, which the null
            // provider ref carries, so the process provider ref is re-set behind it.
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            CudaApiTraceSerialization::SerializeStateRecords(providerState.observer, m_recordWriter);
        }

        void AttachProvider_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();

            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_NvtxTrace_ProviderTypeUuid, "NVTX Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            NvtxTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_recordWriter, [&] {
                providerState.onAttachRecordLocations.push_back(m_recordWriter.GetLastCommittedRecordLocation());
            });

            // NVTX keeps writing provider-owned records after ProviderAttached, which the null
            // provider ref carries, so the process provider ref is re-set behind it.
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            NvtxTraceSerialization::SerializeStateRecords(providerState.observer, m_recordWriter);
        }

        // A provider not attached yet writes its mapping ref along with the provider ref its attach sets.
        void OnThreadEventSourceProviderBound(ThreadEventProviderState&, ProviderState& providerState)
        {
            providerState.threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            if (!m_isConsuming || providerState.providerCreatedRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                return;
            }

            const auto itrProcessState = m_systemState.processes.find(providerState.processLuid);
            if (itrProcessState != m_systemState.processes.end())
            {
                m_recordWriter.SetProcessProviderRef(providerState, itrProcessState->second);
            }
        }

        void AttachProvider_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_ThreadEvent_ProviderTypeUuid, "Thread Event"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();

            Super::ConsumeThreadEventSourceProviderRegistrations(parentState, providerState, ReadCpuTscTimestamp());

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            ThreadEventSerialization::SerializeThreadStateEvents(providerState.observer, m_recordWriter);
        }

        void AttachExistingObjects()
        {
            // Attach System created event
            if (m_systemState.pSystemInfoMemory)
            {
                AttachSystem(m_systemState);
            }

            // Attach System providers
            {
                if (m_systemState.cpuClockCalibrationProvider.IsInitialized())
                {
                    AttachProvider_CpuClockCalibration(m_systemState.cpuClockCalibrationProvider);
                }
            }

            // Attach GpuPartitions and their providers
            for (auto& [_, gpuPartitionState] : m_systemState.gpuPartitions)
            {
                AttachGpuPartition(gpuPartitionState);
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                {
                    AttachProvider_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                {
                    TryPrepareGpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                {
                    TryPrepareGpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                {
                    TryPrepareGpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                }
            }

            // Attach Processes and their providers
            for (auto& [_, processState] : m_systemState.processes)
            {
                AttachProcess(processState);
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    AttachProvider_ThreadEvent(threadEventProviderEntry.second, processState);
                }
                if (processState.cudaApiTraceProvider.IsInitialized())
                {
                    AttachProvider_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                }
                if (processState.nvtxTraceProvider.IsInitialized())
                {
                    AttachProvider_NvtxTrace(processState.nvtxTraceProvider, processState);
                }
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    TryPrepareThreadEvent(threadEventProviderEntry.second, processState, ReadCpuTscTimestamp());
                }
            }
        }

    private:
        using Super::ConvertTscTickToUtcNsecForConsume;
        using Super::UpdateConsumeCpuClockCalibration;

        bool BuildWaitMultiplexer()
        {
            constexpr uint32_t eventCount = 3;
            bool succeeded = m_waitMultiplexer.Initialize(eventCount);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: WaitMultiplexer::Initialize failed (maxObjectCount=%" PRIu32 ")", eventCount);
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::Command, m_consumeThreadCommandPendingEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: WaitMultiplexer::AddObject failed (identifier=Command)");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::RouterNotification, m_pRouterObserverClient->GetWaitForPendingNotificationHandle());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: WaitMultiplexer::AddObject failed (identifier=RouterNotification)");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::ConsumeData, m_consumeTimer.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: WaitMultiplexer::AddObject failed (identifier=ConsumeData)");
            return true;
        }

        bool EnterRouterlessMode()
        {
            bool succeeded = m_waitMultiplexer.RemoveLastSignaledObject();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StreamingConsumer: WaitMultiplexer::RemoveLastSignaledObject failed");
            m_pRouterObserverClient.Reset();
            return true;
        }

        void ConsumeData(uint64_t readEndpointTsc = ReadCpuTscTimestamp())
        {
            UpdateConsumeCpuClockCalibration();
            const uint64_t readEndpointUtcNsec = ConvertTscTickToUtcNsecForConsume(readEndpointTsc);
            const uint64_t recordTimestampEndpointUtcNsec = std::max(m_recordTimestampLowerBoundUtcNsec, std::min(readEndpointUtcNsec, m_recordTimestampLimitUtcNsec.load(std::memory_order_acquire)));

            m_recordTimestampUpperBoundUtcNsec = recordTimestampEndpointUtcNsec;
            if (m_dataBlockWriter.IsInitialized())
            {
                m_dataBlockWriter.SetRecordTimestampUpperBoundUtcNsec(m_recordTimestampUpperBoundUtcNsec);
            }

            Super::ConsumeData(readEndpointTsc);

            m_recordTimestampLowerBoundUtcNsec = recordTimestampEndpointUtcNsec;
        }

        void ConsumeLoop()
        {
            auto exitGuard = ScopeExitGuard([this]() {
                m_consumeThreadExited = true;
                std::atomic_thread_fence(std::memory_order_release);

                bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                NVTRACE_LOG_ERR_IF(!succeeded, 50, "ConsumeLoop exit: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");
            });

            while (true)
            {
                EventIdentifier signaledIdentifier;
                if (!m_waitMultiplexer.Wait(signaledIdentifier))
                {
                    NVTRACE_LOG_ERR(50, "ConsumeLoop: WaitMultiplexer::Wait returned false (see preceding WaitMultiplexer log for errno or Win32 error).");
                    return;
                }

                switch (signaledIdentifier)
                {
                    case EventIdentifier::Command:
                    {
                        std::atomic_thread_fence(std::memory_order_acquire);
                        switch (m_consumeThreadCommand)
                        {
                            case ConsumeThreadCommand::Start:
                            {
                                const uint64_t startTimestampUtcNsec = m_observationStartUtcNsec != 0
                                                                         ? m_observationStartUtcNsec
                                                                         : ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
                                m_recordTimestampLowerBoundUtcNsec = startTimestampUtcNsec;
                                m_recordTimestampUpperBoundUtcNsec = startTimestampUtcNsec;
                                m_observationStartTsc = m_observationStartUtcNsec != 0
                                                          ? this->GetLatestCpuClockCalibrationPoint().ToTsc(m_observationStartUtcNsec)
                                                          : 0;
                                ConfigureExistingObservationStarts();
                                m_isConsuming = true;

                                // Release the thread that called Start before Attaching
                                bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "ConsumeLoop Start: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");

                                AttachExistingObjects();

                                if (!ScheduleImmediateConsumeIteration())
                                {
                                    return;
                                }

                                break;
                            }

                            case ConsumeThreadCommand::Stop:
                            {
                                // Ensure at least one iteration of data consumption is performed before stopping
                                {
                                    const auto dropUnboundGuard = Super::ScopeGuardDropUnboundThreadEvents();
                                    ConsumeData();
                                }
                                m_isConsuming = false;

                                if (m_dataBlockWriter.IsInitialized())
                                {
                                    m_finalDataBlockSequenceNumber = m_dataBlockWriter.DataBlockSequenceNumber();
                                    m_databaseWriter.FinalizeDataBlock(m_dataBlockWriter);
                                }
                                // A subsequent Start must acquire a new chunk rather than reuse
                                // the chunk belonging to the finalized data block.
                                m_recordWriter.Reset();

                                bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "ConsumeLoop Stop: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");

                                break;
                            }

                            case ConsumeThreadCommand::Terminate:
                                return;
                        }

                        break;
                    }

                    case EventIdentifier::RouterNotification:
                    {
                        bool succeeded = m_pRouterObserverClient->ReceiveNotificationIfPending([this](NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) {
                            return HandleRouterNotification(notificationType, pNotification);
                        });
                        if (!succeeded)
                        {
                            NVTRACE_LOG_WRN(50, "ConsumeLoop: Router observer connection lost");
                            if (!EnterRouterlessMode())
                            {
                                NVTRACE_LOG_ERR(50, "ConsumeLoop: Failed to continue without router notifications.");
                                return;
                            }
                            NVTRACE_LOG_WRN(50, "ConsumeLoop: Continuing in routerless mode with known provider buffers.");
                        }

                        break;
                    }

                    case EventIdentifier::ConsumeData:
                    {
                        if (m_isConsuming)
                        {
                            const bool nextIterationScheduled = ScheduleNextConsumeIteration();
                            ConsumeData();
                            if (!nextIterationScheduled)
                            {
                                return;
                            }
                        }

                        break;
                    }
                }
            }
        }

        void TerminateConsumeThread() noexcept
        {
            if (m_consumeThread.joinable())
            {
                m_consumeThreadCommand = ConsumeThreadCommand::Terminate;
                std::atomic_thread_fence(std::memory_order_release);

                bool succeeded = m_consumeThreadCommandPendingEvent.Signal();
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "TerminateConsumeThread: AutoResetEvent::Signal failed (consumeThreadCommandPendingEvent)");

                m_consumeThread.join();
            }
        }

    private:
        class TsfRecordWriter
        {
        public:
            TsfRecordWriter(StreamingConsumer& consumer)
                : m_consumer(consumer)
            {
            }

            /// void fnWritePayload(TInlinePayload* pInlinePayload, TMajorPayload* pPayload);
            template <typename TFN_WritePayload>
            bool WriteRecord(uint32_t recordType, uint64_t correlationId, size_t majorPayloadSize, TFN_WritePayload&& fnWritePayload)
            {
                if (!m_chunkWriter.HasSpaceForRecord(majorPayloadSize))
                {
                    if (m_consumer.m_databaseWriter.MaxMajorPayloadSize() < majorPayloadSize)
                    {
                        return false;
                    }

                    if (!AcquireNewChunk())
                    {
                        return false;
                    }
                }

                m_chunkWriter.WriteRecord(m_providerRefIndex, recordType, correlationId, majorPayloadSize, fnWritePayload);
                return true;
            }

            /// void fnWritePayload(TInlinePayload* pInlinePayload, TMajorPayload* pPayload);
            template <typename TFN_WritePayload>
            bool WriteRecord(uint32_t recordType, TFN_WritePayload&& fnWritePayload)
            {
                using MajorPayloadType = typename GetWritePayloadTypes<TFN_WritePayload>::MajorPayloadType;
                return WriteRecord(recordType, sizeof(MajorPayloadType), fnWritePayload);
            }

            void Commit()
            {
                m_chunkWriter.Commit();
            }

            void Rollback()
            {
                m_chunkWriter.Rollback();
            }

            void Reset() noexcept
            {
                m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                m_pProcessState = nullptr;
                m_pGpuPartitionState = nullptr;
                m_pProviderState = nullptr;
                m_chunkWriter.Reset();
            }

            uint64_t GetLastCommittedRecordLocation() const
            {
                return m_consumer.m_databaseWriter.EncodeRecordLocation(m_consumer.m_dataBlockWriter.DataBlockSequenceNumber(), m_chunkWriter.ChunkIndex(), m_chunkWriter.GetLastCommittedRecordIndex());
            }

            void SetNullProviderRef()
            {
                m_pProviderState = nullptr;
                m_pGpuPartitionState = nullptr;
                m_pProcessState = nullptr;
                m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            }
            void SetSystemProviderRef(ProviderState& providerState)
            {
                m_pProviderState = &providerState;
                m_pGpuPartitionState = nullptr;
                m_pProcessState = nullptr;
                WriteProviderRef();
            }
            void SetGpuProviderRef(ProviderState& providerState, GpuPartitionState& gpuPartitionState)
            {
                m_pProviderState = &providerState;
                m_pGpuPartitionState = &gpuPartitionState;
                m_pProcessState = nullptr;
                WriteProviderRef();
            }
            void SetProcessProviderRef(ProviderState& providerState, ProcessState& processState)
            {
                m_pProviderState = &providerState;
                m_pGpuPartitionState = nullptr;
                m_pProcessState = &processState;
                WriteProviderRef();
            }

        private:
            bool AcquireNewDataBlock()
            {
                if (m_consumer.m_dataBlockWriter.IsInitialized())
                {
                    m_consumer.m_databaseWriter.FinalizeDataBlock(m_consumer.m_dataBlockWriter);
                }

                if (!m_consumer.m_databaseWriter.AcquireDataBlock(m_consumer.m_dataBlockWriter, m_consumer.m_recordTimestampLowerBoundUtcNsec, m_consumer.m_recordTimestampUpperBoundUtcNsec))
                {
                    return false;
                }

                WriteProviderRef();
                return true;
            }

            bool AcquireNewChunk()
            {
                m_chunkWriter.Reset();

                if (!m_consumer.m_dataBlockWriter.IsInitialized() || !m_consumer.m_dataBlockWriter.HasSpaceForChunk())
                {
                    if (!AcquireNewDataBlock())
                    {
                        return false;
                    }
                }

                m_consumer.m_dataBlockWriter.AcquireChunk(m_chunkWriter);
                return true;
            }

            void WriteProviderRef()
            {
                if (!m_consumer.m_dataBlockWriter.IsInitialized() || !m_pProviderState)
                {
                    return;
                }

                const uint64_t dataBlockSequenceNumber = m_consumer.m_dataBlockWriter.DataBlockSequenceNumber();

                // Determine which parent refs need to be written
                const bool requiresNewProviderRef = m_pProviderState->dataBlockSequenceNumber != dataBlockSequenceNumber;
                const bool requiresNewGpuPartitionRef = m_pGpuPartitionState && m_pGpuPartitionState->dataBlockSequenceNumber != dataBlockSequenceNumber;
                const bool requiresNewProcessRef = m_pProcessState && m_pProcessState->dataBlockSequenceNumber != dataBlockSequenceNumber;

                // ThreadEventProvider
                ThreadEventProviderState* const pThreadEventProviderState = m_pProcessState ? m_pProcessState->FindThreadEventProvider(m_pProviderState->threadEventProviderLuid) : nullptr;
                const bool hasThreadEventProviderMappingRef = pThreadEventProviderState && m_pProviderState->threadEventSourceProviderIndex != ProviderState::kInvalidThreadEventSourceProviderIndex;
                const bool requiresNewThreadEventProviderMappingRef = hasThreadEventProviderMappingRef && m_pProviderState->threadEventProviderMappingRefDataBlockSequenceNumber != dataBlockSequenceNumber;
                const bool requiresNewThreadEventProviderRef = requiresNewThreadEventProviderMappingRef && pThreadEventProviderState->dataBlockSequenceNumber != dataBlockSequenceNumber;

                if (!requiresNewProviderRef && !requiresNewThreadEventProviderMappingRef)
                {
                    m_providerRefIndex = m_pProviderState->providerRefIndex;
                    return;
                }
                const uint32_t requiredProviderRefCount = (requiresNewProviderRef ? 1 : 0) + (requiresNewThreadEventProviderRef ? 1 : 0);

                // Check if there's space for all required refs
                // clang-format off
                if (!m_consumer.m_dataBlockWriter.HasSpaceForProviderRef(requiredProviderRefCount) ||
                    (requiresNewGpuPartitionRef && !m_consumer.m_dataBlockWriter.HasSpaceForGpuPartitionRef()) ||
                    (requiresNewProcessRef && !m_consumer.m_dataBlockWriter.HasSpaceForProcessRef()) ||
                    (requiresNewThreadEventProviderMappingRef && !m_consumer.m_dataBlockWriter.HasSpaceForThreadEventProviderMappingRef()))
                // clang-format on
                {
                    m_chunkWriter.Reset();
                    m_consumer.m_databaseWriter.FinalizeDataBlock(m_consumer.m_dataBlockWriter);
                    m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                    return;
                }

                // Write parent refs if needed
                if (requiresNewGpuPartitionRef)
                {
                    m_consumer.m_dataBlockWriter.WriteGpuPartitionRef(m_pGpuPartitionState->luid, m_pGpuPartitionState->gpuPartitionCreatedRecordLocation);
                    m_pGpuPartitionState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                if (requiresNewProcessRef)
                {
                    m_consumer.m_dataBlockWriter.WriteProcessRef(m_pProcessState->luid, m_pProcessState->processCreatedRecordLocation);
                    m_pProcessState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                if (requiresNewThreadEventProviderRef)
                {
                    pThreadEventProviderState->providerRefIndex = m_consumer.m_dataBlockWriter.WriteProviderRef(pThreadEventProviderState->luid, pThreadEventProviderState->providerCreatedRecordLocation, pThreadEventProviderState->providerAttachedRecordLocation);
                    pThreadEventProviderState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                if (requiresNewThreadEventProviderMappingRef)
                {
                    m_consumer.m_dataBlockWriter.WriteThreadEventProviderMappingRef(pThreadEventProviderState->luid, m_pProviderState->luid, m_pProviderState->threadEventSourceProviderIndex);
                    m_pProviderState->threadEventProviderMappingRefDataBlockSequenceNumber = dataBlockSequenceNumber;
                }

                if (requiresNewProviderRef)
                {
                    m_pProviderState->providerRefIndex = m_consumer.m_dataBlockWriter.WriteProviderRef(m_pProviderState->luid, m_pProviderState->providerCreatedRecordLocation, m_pProviderState->providerAttachedRecordLocation);
                    m_pProviderState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                m_providerRefIndex = m_pProviderState->providerRefIndex;
            }

        private:
            StreamingConsumer& m_consumer;
            TraceStreamChunkWriter m_chunkWriter;

            ProviderState* m_pProviderState = nullptr;
            GpuPartitionState* m_pGpuPartitionState = nullptr;
            ProcessState* m_pProcessState = nullptr;

            uint32_t m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
        };

    private:
        using Super::m_consumeTimer;
        using Super::m_pRouterObserverClient;
        using Super::m_systemState;

        TraceStreamDatabaseWriter<TStorageWriter> m_databaseWriter;
        TraceStreamDataBlockWriter m_dataBlockWriter;

        uint64_t m_recordTimestampLowerBoundUtcNsec = 0;
        uint64_t m_observationStartTsc = 0;
        uint64_t m_observationStartUtcNsec = 0;
        uint64_t m_finalDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        uint64_t m_recordTimestampUpperBoundUtcNsec = 0;
        std::atomic<uint64_t> m_recordTimestampLimitUtcNsec = UINT64_MAX;

        TsfRecordWriter m_recordWriter;

        enum class ConsumeThreadCommand
        {
            Start,
            Stop,
            Terminate
        };
        ConsumeThreadCommand m_consumeThreadCommand = ConsumeThreadCommand::Stop;
        AutoResetEvent m_consumeThreadCommandPendingEvent;
        AutoResetEvent m_consumeThreadCommandCompletedEvent;

        enum class EventIdentifier
        {
            Command,
            RouterNotification,
            ConsumeData
        };
        WaitMultiplexer m_waitMultiplexer;

        std::thread m_consumeThread;
        bool m_consumeThreadExited = false;

        bool m_isConsuming = false;
    };

}; // namespace nv::trace
