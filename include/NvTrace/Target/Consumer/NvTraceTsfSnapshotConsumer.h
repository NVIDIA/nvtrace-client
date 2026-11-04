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
#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>
#include <NvTrace/Target/Consumer/NvTraceThreadEventSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceTsfBufferEventEncoders.h>
#include <NvTrace/Target/Consumer/NvTraceTsfSnapshotHistoryBuffer.h>
#include <NvTrace/Target/Consumer/NvTraceTsfSnapshotRecordWriter.h>
#include <NvTrace/Target/Consumer/NvTraceTsfSnapshotTracker.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    struct TsfSnapshotConsumerObjectModel : ConsumerObjectModelBase
    {
        enum class ZombieObjectType
        {
            Invalid,
            Provider,
            Process,
            GpuPartition
        };

        struct ZombieObjectId
        {
            NVTRACE_LUID luid = NVTRACE_NIL_LUID;
            ZombieObjectType type = ZombieObjectType::Invalid;

            bool IsValid() const
            {
                return type != ZombieObjectType::Invalid && luid != NVTRACE_NIL_LUID;
            }
        };

        struct ZombieObjectQueueEntry
        {
            ZombieObjectId zombieObject = {};
            void* pPendingDestroy = nullptr;
            uint64_t lastDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            size_t retainedBytes = 0;
        };

        using ZombieObjectQueue = std::list<ZombieObjectQueueEntry>;

        template <typename TNotification, NvTrace_RouterObserverClient_NotificationType TNotificationType>
        struct PendingDestroyState
        {
            static constexpr NvTrace_RouterObserverClient_NotificationType notificationType = TNotificationType;

            ZombieObjectId zombieObject = {};
            TNotification notification = {};
            ZombieObjectQueue::iterator zombieObjectQueueItr = {};

            void Reset() noexcept
            {
                zombieObject = {};
                notification = {};
                zombieObjectQueueItr = {};
            }

            explicit operator bool() const
            {
                return zombieObject.IsValid();
            }
        };

        using PendingProviderDestroy = PendingDestroyState<NvTrace_RouterObserverClient_ProviderDestroyedNotification, NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed>;
        using PendingProcessDestroy = PendingDestroyState<NvTrace_RouterObserverClient_ProcessDestroyedNotification, NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed>;
        using PendingGpuPartitionDestroy = PendingDestroyState<NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification, NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed>;

        struct ProviderState : ConsumerObjectModelBase::ProviderState
        {
            PendingProviderDestroy pendingDestroy;
            uint64_t providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            std::vector<uint64_t> onAttachRecordLocations;
            uint32_t providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            bool isReadyToConsume = true;

            void Reset() noexcept
            {
                pendingDestroy.Reset();
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
        struct GpuPmCountersProviderState : TypedProviderState<GpuPmCounterObserver, ProviderState>
        {
        };
        using GpuHwTraceProviderState = TypedProviderState<GpuHwTraceObserver, ProviderState>;
        using GpuGrCtxTraceProviderState = TypedProviderState<GpuGrCtxTraceObserver, ProviderState>;
        using CudaApiTraceProviderState = TypedCudaApiTraceProviderState<ProviderState>;
        using NvtxTraceProviderState = TypedNvtxTraceProviderState<ProviderState>;
        using ThreadEventProviderState = TypedThreadEventProviderState<ProviderState>;

        template <typename TGpuClockCalibrationProviderState, typename TGpuPmCountersProviderState, typename TGpuHwTraceProviderState, typename TGpuGrCtxTraceProviderState>
        struct GpuPartitionState : ConsumerObjectModelBase::GpuPartitionState<TGpuClockCalibrationProviderState, TGpuPmCountersProviderState, TGpuHwTraceProviderState, TGpuGrCtxTraceProviderState>
        {
            PendingGpuPartitionDestroy pendingDestroy;
            uint64_t gpuPartitionCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t observationStartGpuNsec = 0;
            bool isObservationStartResolved = true;
        };

        template <typename TCudaApiTraceProviderState, typename TNvtxTraceProviderState, typename TThreadEventProviderState>
        struct ProcessState : ConsumerObjectModelBase::ProcessState<TCudaApiTraceProviderState, TNvtxTraceProviderState, TThreadEventProviderState>
        {
            PendingProcessDestroy pendingDestroy;
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
    class TsfSnapshotConsumer : protected ConsumerBase<TsfSnapshotConsumer<TStorageWriter>, TsfSnapshotConsumerObjectModel>
    {
        using Super = ConsumerBase<TsfSnapshotConsumer, TsfSnapshotConsumerObjectModel>;
        friend Super;

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

        using ZombieObjectType = TsfSnapshotConsumerObjectModel::ZombieObjectType;
        using ZombieObjectQueueEntry = TsfSnapshotConsumerObjectModel::ZombieObjectQueueEntry;
        using ZombieObjectQueue = TsfSnapshotConsumerObjectModel::ZombieObjectQueue;
        using PendingProviderDestroy = TsfSnapshotConsumerObjectModel::PendingProviderDestroy;
        using PendingProcessDestroy = TsfSnapshotConsumerObjectModel::PendingProcessDestroy;
        using PendingGpuPartitionDestroy = TsfSnapshotConsumerObjectModel::PendingGpuPartitionDestroy;

    private:
        using DatabaseWriter = TraceStreamDatabaseWriter<TStorageWriter>;
        using TsfSnapshotHistoryBuffer = nv::trace::TsfSnapshotHistoryBuffer;
        using TsfSnapshotCommand = nv::trace::TsfSnapshotCommand<TStorageWriter>;
        using TsfSnapshotTracker = nv::trace::TsfSnapshotTracker<TStorageWriter>;
        using TsfSnapshotRecordWriter = nv::trace::TsfSnapshotRecordWriter<
            ProviderState,
            GpuPartitionState,
            ProcessState,
            ThreadEventProviderState>;
        using TsfSnapshotAttachRecordWriter = nv::trace::TsfSnapshotRecordWriter<
            ProviderState,
            GpuPartitionState,
            ProcessState,
            ThreadEventProviderState,
            true>;

        using FnOnSnapshotTaken = void (*)(void* pUserData, uint32_t snapshotNumber);
        using FnOnSnapshotCompleted = void (*)(void* pUserData, uint32_t snapshotNumber, bool succeeded);
        using FnOnStorageWriteCompleted = void (*)(void* pUserData, uint64_t dataBlockSequenceNumber, bool succeeded);

    public:
        struct Options : Super::Options
        {
            size_t dataBlockSize = 10 * 1024 * 1024;
            // Total circular-FIFO slots, including the currently unpublished
            // writable block when one exists.
            size_t numDataBlocks = 8;
            size_t maxZombieObjectBytes = 2ULL << 30;
            FnOnSnapshotTaken fnOnSnapshotTaken = nullptr;
            void* pUserDataOnSnapshotTaken = nullptr;
            FnOnSnapshotCompleted fnOnSnapshotCompleted = nullptr;
            void* pUserDataOnSnapshotCompleted = nullptr;
            uint64_t observationStartUtcNsec = 0;
        };

        struct SnapshotParam
        {
            // Zero indicates the maximum history is wanted
            uint64_t historyDurationNsec = 0;
            uint64_t futureDurationNsec = 0;
        };

        using Super::ConvertTscTickToUtcNsec;

        TsfSnapshotConsumer() = default;

        ~TsfSnapshotConsumer()
        {
            Reset();
        }

        TsfSnapshotConsumer(const TsfSnapshotConsumer&) = delete;
        TsfSnapshotConsumer& operator=(const TsfSnapshotConsumer&) = delete;

        void Reset() noexcept
        {
            TerminateConsumeThread();
            TerminateSnapshotThread();
            m_isConsuming = false;
            m_consumeThreadExited = false;
            m_consumeThreadWaitMultiplexer.Reset();
            m_consumeThreadCommandCompletedEvent.Reset();
            m_consumeThreadCommandPendingEvent.Reset();
            m_consumeThreadCommand = ConsumeThreadCommand::Terminate;
            m_recordWriter.Reset();
            m_attachRecordWriter.Reset();
            m_snapshotTracker.Reset();
            m_databaseWriter.Reset();
            m_snapshotThreadExited = false;
            m_snapshotThreadWaitMultiplexer.Reset();
            m_snapshotThreadSnapshotCompletedEvent.Reset();
            m_snapshotThreadStorageWriteCompletedEvent.Reset();
            m_snapshotThreadCheckpointCompletedEvent.Reset();
            m_snapshotThreadCommandCompletedEvent.Reset();
            m_snapshotThreadCommandPendingEvent.Reset();
            m_snapshotThreadCommand = SnapshotThreadCommand::Terminate;
            m_snapshotStorageWriter.reset();
            m_pendingCheckpoints = decltype(m_pendingCheckpoints)();
            m_snapshotNumbersForCurrentCheckpoint.clear();
            m_checkpointCompletedEvents.clear();
            m_storageWriteCompletedEvents.clear();
            m_snapshotCompletedEvents.clear();
            m_historyBuffer.Reset();
            m_zombieObjectQueue.clear();
            m_zombieObjectBytes = 0;
            m_maxZombieObjectBytes = 0;
            m_lastAgedOutDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_observationStartTsc = 0;
            m_isReplayingZombieDestroy = false;
            Super::Reset();
        }

        template <typename TFN_InitializeStorageWriter>
        bool Initialize(const Options& options, TFN_InitializeStorageWriter&& fnInitializeStorageWriter)
        {
            if ((options.numDataBlocks) < 2 || options.numDataBlocks > UINT8_MAX)
            {
                return false;
            }

            NvTrace_RouterObserverClient_RouterConfig routerConfig;
            if (!Super::Initialize(options, &routerConfig))
            {
                return false;
            }

            auto resetGuard = ScopeExitGuard([this]() { Reset(); });
            if (!InitializeStorage(options, routerConfig, std::forward<TFN_InitializeStorageWriter>(fnInitializeStorageWriter)))
            {
                return false;
            }

            m_observationStartTsc = options.observationStartUtcNsec != 0
                                      ? this->GetLatestCpuClockCalibrationPoint().ToTsc(options.observationStartUtcNsec)
                                      : 0;
            ConfigureExistingObservationStarts();

            // Like SnapshotConsumer, capture begins with initialization.
            m_isConsuming = true;

            bool succeeded = m_consumeThreadCommandPendingEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for consumeThreadCommandPendingEvent");
            succeeded = m_consumeThreadCommandCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for consumeThreadCommandCompletedEvent");
            succeeded = m_snapshotThreadCommandPendingEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for snapshotThreadCommandPendingEvent");
            succeeded = m_snapshotThreadCommandCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for snapshotThreadCommandCompletedEvent");
            succeeded = m_snapshotThreadStorageWriteCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for snapshotThreadStorageWriteCompletedEvent");
            succeeded = m_snapshotThreadCheckpointCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for snapshotThreadCheckpointCompletedEvent");
            succeeded = m_snapshotThreadSnapshotCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: AutoResetEvent::Initialize failed for snapshotThreadSnapshotCompletedEvent");

            constexpr uint32_t eventCount = 3;
            succeeded = m_consumeThreadWaitMultiplexer.Initialize(eventCount);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::Initialize failed");
            succeeded = m_consumeThreadWaitMultiplexer.AddObject(ConsumeThreadEventIdentifier::Command, m_consumeThreadCommandPendingEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (Command)");
            succeeded = m_consumeThreadWaitMultiplexer.AddObject(ConsumeThreadEventIdentifier::RouterNotification, m_pRouterObserverClient->GetWaitForPendingNotificationHandle());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (RouterNotification)");
            succeeded = m_consumeThreadWaitMultiplexer.AddObject(ConsumeThreadEventIdentifier::ConsumeData, m_consumeTimer.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (ConsumeData)");

            const uint32_t snapshotThreadEventCount = 4;
            succeeded = m_snapshotThreadWaitMultiplexer.Initialize(snapshotThreadEventCount);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::Initialize failed");
            succeeded = m_snapshotThreadWaitMultiplexer.AddObject(SnapshotThreadEventIdentifier::Command, m_snapshotThreadCommandPendingEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (Command)");
            succeeded = m_snapshotThreadWaitMultiplexer.AddObject(SnapshotThreadEventIdentifier::CheckpointCompleted, m_snapshotThreadCheckpointCompletedEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (CheckpointCompleted)");
            succeeded = m_snapshotThreadWaitMultiplexer.AddObject(SnapshotThreadEventIdentifier::StorageWriteCompleted, m_snapshotThreadStorageWriteCompletedEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (StorageWriteCompleted)");
            succeeded = m_snapshotThreadWaitMultiplexer.AddObject(SnapshotThreadEventIdentifier::SnapshotCompleted, m_snapshotThreadSnapshotCompletedEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: WaitMultiplexer::AddObject failed (SnapshotCompleted)");

            succeeded = ScheduleImmediateConsumeIteration();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TsfSnapshotConsumer: ScheduleImmediateConsumeIteration failed");

            try
            {
                m_consumeThread = std::thread(&TsfSnapshotConsumer::ConsumeLoop, this);
            }
            catch (const std::system_error& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to create consume thread (std::system_error %d: %s).", e.code().value(), e.what());
                return false;
            }

            try
            {
                m_snapshotThread = std::thread(&TsfSnapshotConsumer::SnapshotLoop, this);
            }
            catch (const std::system_error& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to create snapshot thread (std::system_error %d: %s).", e.code().value(), e.what());
                return false;
            }

            resetGuard.Dismiss();
            return true;
        }

        template <typename TFN_InitializeSnapshotStorageWriter>
        bool Snapshot(TFN_InitializeSnapshotStorageWriter&& fnInitializeSnapshotStorageWriter, const SnapshotParam& snapshotParam)
        {
            if (m_consumeThreadExited || m_snapshotThreadExited)
            {
                return false;
            }

            m_snapshotTriggerTimestampUtcNsec = ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
            m_snapshotParam = snapshotParam;
            auto snapshotStorageWriter = std::make_unique<TStorageWriter>();
            if (!fnInitializeSnapshotStorageWriter(*snapshotStorageWriter))
            {
                return false;
            }
            m_snapshotStorageWriter = std::move(snapshotStorageWriter);

            // Execute snapshot command
            m_snapshotThreadCommandSucceeded = false;
            m_snapshotThreadCommand = SnapshotThreadCommand::Snapshot;
            std::atomic_thread_fence(std::memory_order_release);
            bool succeeded = m_snapshotThreadCommandPendingEvent.Signal();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "Snapshot: AutoResetEvent::Signal failed (snapshotThreadCommandPendingEvent)");
            succeeded = m_snapshotThreadCommandCompletedEvent.Wait();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "Snapshot: AutoResetEvent::Wait failed (snapshotThreadCommandCompletedEvent)");
            std::atomic_thread_fence(std::memory_order_acquire);
            NVTRACE_LOG_ERR_IF(m_snapshotThreadExited, 50, "Snapshot: Snapshot thread exited.");
            return succeeded && m_snapshotThreadCommandSucceeded;
        }

    protected:
        template <typename TFN_InitializeStorageWriter>
        bool InitializeStorage(
            const Options& options,
            const NvTrace_RouterObserverClient_RouterConfig& routerConfig,
            TFN_InitializeStorageWriter&& fnInitializeStorageWriter)
        {
            const uint8_t numDataBlocks = static_cast<uint8_t>(options.numDataBlocks);
            m_maxZombieObjectBytes = options.maxZombieObjectBytes;
            m_fnOnSnapshotTaken = options.fnOnSnapshotTaken;
            m_pUserDataOnSnapshotTaken = options.pUserDataOnSnapshotTaken;
            m_fnOnSnapshotCompleted = options.fnOnSnapshotCompleted;
            m_pUserDataOnSnapshotCompleted = options.pUserDataOnSnapshotCompleted;

            typename DatabaseWriter::Options databaseWriterOptions;
            databaseWriterOptions.dataBlockSize = options.dataBlockSize;
            databaseWriterOptions.maxGpuPartitionCountPerDataBlock = routerConfig.maxGpuPartitionCount * 5 / 4;
            databaseWriterOptions.maxProcessCountPerDataBlock = routerConfig.maxProcessCount * 5 / 4;
            databaseWriterOptions.maxProviderCountPerDataBlock = routerConfig.maxProviderCount * 5 / 4;
            databaseWriterOptions.maxThreadEventProviderMappingCountPerDataBlock = routerConfig.maxProviderCount * 5 / 4;
            databaseWriterOptions.chunkCountPerDataBlock = 1;
            databaseWriterOptions.writeSlotCount = numDataBlocks;
            databaseWriterOptions.reservedSlotCount = 2;
            databaseWriterOptions.fnOnStorageWriteCompleted = [](void* pUserData, uint8_t, uint64_t dataBlockSequenceNumber, bool isStorageWriteCompletedSuccessfully) {
                TsfSnapshotConsumer& consumer = *static_cast<TsfSnapshotConsumer*>(pUserData);
                consumer.OnDatabaseStorageWriteCompletedCallback(
                    dataBlockSequenceNumber,
                    isStorageWriteCompletedSuccessfully);
            };
            databaseWriterOptions.pUserDataOnStorageWriteCompleted = this;
            if (!m_databaseWriter.Initialize(databaseWriterOptions, std::forward<TFN_InitializeStorageWriter>(fnInitializeStorageWriter)))
            {
                return false;
            }
            m_recordWriter.Initialize(
                m_databaseWriter.MaxMajorPayloadSize(),
                &TsfSnapshotConsumer::EncodeRecordLocation,
                &TsfSnapshotConsumer::StageCurrentDataBlock,
                &TsfSnapshotConsumer::ObtainNewDataBlock,
                this);
            m_attachRecordWriter.Initialize(
                m_databaseWriter.MaxMajorPayloadSize(),
                &TsfSnapshotConsumer::EncodeRecordLocation,
                &TsfSnapshotConsumer::StageCurrentAttachDataBlock,
                &TsfSnapshotConsumer::ObtainNewAttachDataBlock,
                this);
            if (!m_historyBuffer.Initialize(
                    numDataBlocks,
                    [](void* pUserData, TraceStreamDataBlockWriter& dataBlockWriter) {
                        static_cast<TsfSnapshotConsumer*>(pUserData)->OnHistoryBufferDataBlockAgedInCallback(dataBlockWriter);
                    },
                    this,
                    [](void* pUserData, const TsfSnapshotHistoryBuffer::DataBlockInfo& dataBlockInfo) {
                        static_cast<TsfSnapshotConsumer*>(pUserData)->OnHistoryBufferDataBlockAgedOutCallback(dataBlockInfo);
                    },
                    this))
            {
                return false;
            }

            if (!m_snapshotTracker.Initialize(
                    [](void* pUserData, uint32_t snapshotNumber, bool isSnapshotCompletedSuccessfully) {
                        static_cast<TsfSnapshotConsumer*>(pUserData)->OnSnapshotTrackerSnapshotCompletedCallback(snapshotNumber, isSnapshotCompletedSuccessfully);
                    },
                    this))
            {
                return false;
            }

            return true;
        }

    private:
        using Super::ScheduleImmediateConsumeIteration;
        using Super::ScheduleNextConsumeIteration;

    protected:
        bool HandleRouterNotification(NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification)
        {
            switch (notificationType)
            {
                case NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed:
                    return DelayGpuPartitionDestroyedNotification(*reinterpret_cast<const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification*>(pNotification));
                case NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed:
                    return DelayProcessDestroyedNotification(*reinterpret_cast<const NvTrace_RouterObserverClient_ProcessDestroyedNotification*>(pNotification));
                case NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed:
                {
                    const auto& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProviderDestroyedNotification*>(pNotification);
                    // ConsumerBase intentionally treats ThreadEvent providers as
                    // process-lifetime objects, so there is nothing to defer.
                    if (Super::ResolveProviderTypeUuidAlias(notification.providerTypeUuid) == NvTrace_ThreadEvent_ProviderTypeUuid)
                    {
                        return Super::HandleRouterNotification(notificationType, pNotification);
                    }
                    return DelayProviderDestroyedNotification(notification);
                }
                default:
                    return Super::HandleRouterNotification(notificationType, pNotification);
            }
        }

        bool DelayProviderDestroyedNotification(const NvTrace_RouterObserverClient_ProviderDestroyedNotification& notification)
        {
            ProviderState* const pProviderState = m_systemState.GetProviderState(notification.providerLuid);
            NVTRACE_LOG_ERR_RETURN_IF(!pProviderState, false, 50, "Received ProviderDestroyed notification for an unregistered provider (providerTypeUuid=%s providerLuid=%" PRIu64 " parentLuid=%" PRIu64 ")", ToString(notification.providerTypeUuid).c_str(), notification.providerLuid, notification.parentLuid);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderState->pendingDestroy, false, 50, "Received duplicate ProviderDestroyed notification for a pending zombie provider (providerTypeUuid=%s providerLuid=%" PRIu64 " parentLuid=%" PRIu64 ")", ToString(notification.providerTypeUuid).c_str(), notification.providerLuid, notification.parentLuid);

            auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.parentLuid);
            if (itrGpuPartitionState != m_systemState.gpuPartitions.end() && itrGpuPartitionState->second.pendingDestroy)
            {
                NVTRACE_LOG_WRN(50, "Ignoring ProviderDestroyed notification for a provider whose GPU partition is already pending destruction (providerLuid=%" PRIu64 ").", notification.providerLuid);
                return true;
            }

            auto itrProcessState = m_systemState.processes.find(notification.parentLuid);
            if (itrProcessState != m_systemState.processes.end() && itrProcessState->second.pendingDestroy)
            {
                NVTRACE_LOG_WRN(50, "Ignoring ProviderDestroyed notification for a provider whose process is already pending destruction (providerLuid=%" PRIu64 ").", notification.providerLuid);
                return true;
            }

            // Consume pending ThreadEvent records while the bound provider is still live. Replaying this
            // notification after the retained history ages out would be too late to preserve them.
            OnProviderDestroyed(*pProviderState, ProviderDestroyReason::Explicit);

            PendingProviderDestroy& pendingDestroy = pProviderState->pendingDestroy;
            pendingDestroy.notification = notification;
            pendingDestroy.zombieObject = {notification.providerLuid, ZombieObjectType::Provider};
            return QueueZombieObject(pendingDestroy, GetProviderRetainedBytes(*pProviderState));
        }

        bool DelayProcessDestroyedNotification(const NvTrace_RouterObserverClient_ProcessDestroyedNotification& notification)
        {
            auto itrProcessState = m_systemState.processes.find(notification.processLuid);
            NVTRACE_LOG_ERR_RETURN_IF(itrProcessState == m_systemState.processes.end(), false, 50, "Received ProcessDestroyed notification for a non-existing process with LUID %" PRIu64, notification.processLuid);
            NVTRACE_LOG_ERR_RETURN_IF(itrProcessState->second.pendingDestroy, false, 50, "Received duplicate ProcessDestroyed notification for pending zombie process LUID %" PRIu64, notification.processLuid);

            const size_t retainedBytes = GetProcessRetainedBytes(itrProcessState->second);
            if (retainedBytes > m_maxZombieObjectBytes)
            {
                ReplayPendingChildProviderDestroyedNotifications(itrProcessState->second);
            }

            PendingProcessDestroy& pendingDestroy = itrProcessState->second.pendingDestroy;
            pendingDestroy.notification = notification;
            pendingDestroy.zombieObject = {notification.processLuid, ZombieObjectType::Process};
            if (m_isConsuming)
            {
                m_recordWriter.SetNullProviderRef();
                if (Codec::NvTraceServiceEncoders::WriteProcessDestroyedRecord(m_recordWriter, notification.processLuid, notification.destructionTimestampTsc))
                {
                    m_recordWriter.Commit();
                }
            }
            return QueueZombieObject(pendingDestroy, retainedBytes);
        }

        bool DelayGpuPartitionDestroyedNotification(const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification& notification)
        {
            auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.gpuPartitionLuid);
            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState == m_systemState.gpuPartitions.end(), false, 50, "Received GpuPartitionDestroyed notification for a non-existing GPU partition with LUID %" PRIu64, notification.gpuPartitionLuid);
            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState->second.pendingDestroy, false, 50, "Received duplicate GpuPartitionDestroyed notification for pending zombie GPU partition LUID %" PRIu64, notification.gpuPartitionLuid);

            GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;
            const size_t retainedBytes = GetGpuPartitionRetainedBytes(gpuPartitionState);
            if (retainedBytes > m_maxZombieObjectBytes)
            {
                ReplayPendingChildProviderDestroyedNotifications(gpuPartitionState);
            }

            PendingGpuPartitionDestroy& pendingDestroy = gpuPartitionState.pendingDestroy;
            pendingDestroy.notification = notification;
            pendingDestroy.zombieObject = {notification.gpuPartitionLuid, ZombieObjectType::GpuPartition};
            if (m_isConsuming)
            {
                m_recordWriter.SetNullProviderRef();
                if (Codec::NvTraceServiceEncoders::WriteGpuPartitionDestroyedRecord(m_recordWriter, notification.gpuPartitionLuid, notification.destructionTimestampTsc))
                {
                    m_recordWriter.Commit();
                }
            }
            return QueueZombieObject(pendingDestroy, retainedBytes);
        }

        template <typename TProviderState>
        void ReplayPendingChildProviderDestroyedNotification(TProviderState& providerState)
        {
            if (providerState.pendingDestroy)
            {
                ReplayDestroyNotification(providerState.pendingDestroy);
            }
        }

        void ReplayPendingChildProviderDestroyedNotifications(ProcessState& processState)
        {
            ReplayPendingChildProviderDestroyedNotification(processState.cudaApiTraceProvider);
            ReplayPendingChildProviderDestroyedNotification(processState.nvtxTraceProvider);
            for (auto& [_, threadEventProviderState] : processState.threadEventProviders)
            {
                ReplayPendingChildProviderDestroyedNotification(threadEventProviderState);
            }
        }

        void ReplayPendingChildProviderDestroyedNotifications(GpuPartitionState& gpuPartitionState)
        {
            ReplayPendingChildProviderDestroyedNotification(gpuPartitionState.gpuClockCalibrationProvider);
            ReplayPendingChildProviderDestroyedNotification(gpuPartitionState.gpuPmCountersProvider);
            ReplayPendingChildProviderDestroyedNotification(gpuPartitionState.gpuHwTraceProvider);
            ReplayPendingChildProviderDestroyedNotification(gpuPartitionState.gpuGrCtxTraceProvider);
        }

        template <typename TPendingDestroy>
        bool ForwardPendingDestroy(TPendingDestroy& pendingDestroy)
        {
            const auto notification = pendingDestroy.notification;
            pendingDestroy.Reset();
            m_isReplayingZombieDestroy = true;
            auto replayGuard = ScopeExitGuard([this]() { m_isReplayingZombieDestroy = false; });
            return Super::HandleRouterNotification(TPendingDestroy::notificationType, &notification);
        }

        template <typename TPendingDestroy>
        bool QueueZombieObject(TPendingDestroy& pendingDestroy, size_t retainedBytes)
        {
            ZombieObjectQueueEntry entry = {};
            entry.zombieObject = pendingDestroy.zombieObject;
            entry.pPendingDestroy = &pendingDestroy;
            entry.retainedBytes = retainedBytes;

            if (retainedBytes > m_maxZombieObjectBytes)
            {
                NVTRACE_LOG_WRN(50, "Forwarding a zombie object immediately because it retains %zu bytes, exceeding maxZombieObjectBytes=%zu; retained history may be incomplete.", retainedBytes, m_maxZombieObjectBytes);
                return ForwardPendingDestroy(pendingDestroy);
            }

            const size_t requiredCapacity = m_maxZombieObjectBytes - retainedBytes;
            while (m_zombieObjectBytes > requiredCapacity && !m_zombieObjectQueue.empty())
            {
                NVTRACE_LOG_WRN(50, "Evicting a zombie object before its history expires to satisfy maxZombieObjectBytes=%zu; retained history may be incomplete.", m_maxZombieObjectBytes);
                if (!ReplayZombieObject(m_zombieObjectQueue.begin()))
                {
                    // The new destroy has not entered the queue, so its
                    // pending state has no valid queue iterator. Forwarding
                    // it clears that state before reporting the replay
                    // failure to the caller.
                    ForwardPendingDestroy(pendingDestroy);
                    return false;
                }
            }

            pendingDestroy.zombieObjectQueueItr = m_zombieObjectQueue.insert(m_zombieObjectQueue.end(), entry);
            m_zombieObjectBytes += retainedBytes;
            return true;
        }

        template <typename TPendingDestroy>
        bool ReplayDestroyNotification(TPendingDestroy& pendingDestroy)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!pendingDestroy, false, 50, "TsfSnapshotConsumer tried to replay a destroy notification that is not pending.");
            const auto notification = pendingDestroy.notification;
            m_zombieObjectBytes = m_zombieObjectBytes > pendingDestroy.zombieObjectQueueItr->retainedBytes ? m_zombieObjectBytes - pendingDestroy.zombieObjectQueueItr->retainedBytes : 0;
            m_zombieObjectQueue.erase(pendingDestroy.zombieObjectQueueItr);
            pendingDestroy.Reset();

            m_isReplayingZombieDestroy = true;
            auto replayGuard = ScopeExitGuard([this]() { m_isReplayingZombieDestroy = false; });
            const bool succeeded = Super::HandleRouterNotification(TPendingDestroy::notificationType, &notification);
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "TsfSnapshotConsumer failed to replay a queued destroy notification.");
            return succeeded;
        }

        void SealQueuedZombieObjects()
        {
            const uint64_t lastDataBlockSequenceNumber = m_recordWriter.LastWrittenDataBlockSequenceNumber();
            for (auto itrZombieObject = m_zombieObjectQueue.rbegin(); itrZombieObject != m_zombieObjectQueue.rend(); ++itrZombieObject)
            {
                if (itrZombieObject->lastDataBlockSequenceNumber != TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
                {
                    break;
                }
                itrZombieObject->lastDataBlockSequenceNumber = lastDataBlockSequenceNumber;
            }
        }

    private:
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

        void OnGpuPartitionCreated(GpuPartitionState& gpuPartitionState)
        {
            gpuPartitionState.observationStartGpuNsec = 0;
            gpuPartitionState.isObservationStartResolved = m_observationStartTsc == 0;
        }

        void OnProviderCreated_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            ConfigureCpuObservationStart(providerState);
        }

        void OnProviderCreated_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState&)
        {
            ConfigureGpuClockObservationStart(providerState);
        }

        void OnProviderCreated_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.isReadyToConsume = false;
            TryPrepareGpuObservationStart(providerState, parentState);
        }

        void OnProviderCreated_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.isReadyToConsume = false;
            TryPrepareGpuObservationStart(providerState, parentState);
        }

        void OnProviderCreated_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.isReadyToConsume = false;
            TryPrepareGpuObservationStart(providerState, parentState);
        }

        void OnProviderCreated_ThreadEvent(ThreadEventProviderState& providerState, ProcessState&)
        {
            providerState.isReadyToConsume = false;
            TryPrepareThreadEventObservationStart(providerState, ReadCpuTscTimestamp());
        }

        size_t GetProviderRetainedBytes(const ProviderState& providerState) const
        {
            return providerState.providerBufferSize;
        }

        template <typename TProviderState>
        size_t GetProviderRetainedBytesIfNotPending(const TProviderState& providerState) const
        {
            return providerState.IsInitialized() && !providerState.pendingDestroy ? GetProviderRetainedBytes(providerState) : 0;
        }

        size_t GetProcessRetainedBytes(const ProcessState& processState) const
        {
            size_t retainedBytes = GetProviderRetainedBytesIfNotPending(processState.cudaApiTraceProvider);
            retainedBytes += GetProviderRetainedBytesIfNotPending(processState.nvtxTraceProvider);
            for (const auto& [_, threadEventProviderState] : processState.threadEventProviders)
            {
                retainedBytes += GetProviderRetainedBytesIfNotPending(threadEventProviderState);
            }
            return retainedBytes;
        }

        size_t GetGpuPartitionRetainedBytes(const GpuPartitionState& gpuPartitionState) const
        {
            size_t retainedBytes = gpuPartitionState.gpuPartitionInfo.size;
            retainedBytes += GetProviderRetainedBytesIfNotPending(gpuPartitionState.gpuClockCalibrationProvider);
            retainedBytes += GetProviderRetainedBytesIfNotPending(gpuPartitionState.gpuPmCountersProvider);
            retainedBytes += GetProviderRetainedBytesIfNotPending(gpuPartitionState.gpuHwTraceProvider);
            retainedBytes += GetProviderRetainedBytesIfNotPending(gpuPartitionState.gpuGrCtxTraceProvider);
            return retainedBytes;
        }

        template <typename TPendingDestroy>
        bool ReplayZombieObject(ZombieObjectQueue::iterator itrZombieObject)
        {
            auto* const pPendingDestroy = static_cast<TPendingDestroy*>(itrZombieObject->pPendingDestroy);
            if (!pPendingDestroy)
            {
                m_zombieObjectQueue.erase(itrZombieObject);
                return true;
            }
            return ReplayDestroyNotification(*pPendingDestroy);
        }

        bool ReplayZombieObject(ZombieObjectQueue::iterator itrZombieObject)
        {
            switch (itrZombieObject->zombieObject.type)
            {
                case ZombieObjectType::Provider:
                    return ReplayZombieObject<PendingProviderDestroy>(itrZombieObject);
                case ZombieObjectType::Process:
                    return ReplayZombieObject<PendingProcessDestroy>(itrZombieObject);
                case ZombieObjectType::GpuPartition:
                    return ReplayZombieObject<PendingGpuPartitionDestroy>(itrZombieObject);
                case ZombieObjectType::Invalid:
                    break;
            }

            NVTRACE_LOG_ERR(50, "TsfSnapshotConsumer zombie object queue entry has an invalid object type.");
            m_zombieObjectQueue.erase(itrZombieObject);
            return true;
        }

        void OnProviderDestroyed(ProviderState& providerState, ProviderDestroyReason reason)
        {
            if (m_isReplayingZombieDestroy || !m_isConsuming || reason != ProviderDestroyReason::Explicit)
            {
                return;
            }

            const auto itrProcessState = m_systemState.processes.find(providerState.processLuid);
            if (itrProcessState == m_systemState.processes.end())
            {
                return;
            }

            ProcessState& parentState = itrProcessState->second;
            if (providerState.threadEventProviderLuid == NVTRACE_NIL_LUID
                || providerState.threadEventSourceProviderIndex == ProviderState::kInvalidThreadEventSourceProviderIndex)
            {
                return;
            }

            ThreadEventProviderState* const pThreadEventProviderState = parentState.FindThreadEventProvider(providerState.threadEventProviderLuid);
            if (pThreadEventProviderState)
            {
                ConsumeData_ThreadEvent(*pThreadEventProviderState, parentState, ReadCpuTscTimestamp());
            }
        }

        void OnProviderDestroyed_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming && !m_isReplayingZombieDestroy)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ThreadEventProviderState* const pThreadEventProviderState = parentState.FindThreadEventProvider(providerState.threadEventProviderLuid);
                if (pThreadEventProviderState)
                {
                    ConsumeData_ThreadEvent(*pThreadEventProviderState, parentState, readEndpointTsc);
                }
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
                ConsumeData_ThreadEvent(providerState, parentState, ReadCpuTscTimestamp());
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
        bool TryPrepareGpuObservationStart(TProviderState& providerState, GpuPartitionState& parentState)
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
            providerState.isReadyToConsume = true;
            return true;
        }

        bool TryPrepareThreadEventObservationStart(ThreadEventProviderState& providerState, uint64_t readEndpointTsc)
        {
            if (providerState.isReadyToConsume)
            {
                return true;
            }
            if (m_observationStartTsc == 0)
            {
                providerState.isReadyToConsume = true;
                return true;
            }
            if (readEndpointTsc < m_observationStartTsc || !providerState.observer.SkipEventBufferEntriesBefore(m_observationStartTsc))
            {
                return false;
            }

            providerState.isReadyToConsume = true;
            return true;
        }

        void ConsumeData_CpuClockCalibration(CpuClockCalibrationProviderState& providerState, const ReadEndpoints& readEndpoints)
        {
            m_recordWriter.SetSystemProviderRef(providerState);

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
            m_recordWriter.SetGpuProviderRef(providerState, parentState);

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
            if (!TryPrepareGpuObservationStart(providerState, parentState))
            {
                return;
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
            if (!TryPrepareGpuObservationStart(providerState, parentState))
            {
                return;
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
            if (!TryPrepareGpuObservationStart(providerState, parentState))
            {
                return;
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
            if (!TryPrepareThreadEventObservationStart(providerState, readEndpointTsc))
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

    protected:
        void RegisterThreadEventConsumeCallbacks()
        {
            Super::RegisterThreadEventConsumeCallback(NvTrace_ThreadEvent_ProviderTypeUuid, &TsfSnapshotConsumer::ConsumeThreadEvent_ThreadEventCallback);
            Super::RegisterThreadEventConsumeCallback(NvTrace_CUDA_ApiTrace_ProviderTypeUuid, &TsfSnapshotConsumer::ConsumeThreadEvent_CudaApiTraceCallback);
            Super::RegisterThreadEventConsumeCallback(NvTrace_NvtxTrace_ProviderTypeUuid, &TsfSnapshotConsumer::ConsumeThreadEvent_NvtxTraceCallback);
        }

        static bool ConsumeThreadEvent_ThreadEventCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            TsfSnapshotConsumer& consumer = *reinterpret_cast<TsfSnapshotConsumer*>(pConsumer);
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

        // The destroyed record rides its own thread-event commit, bytes intact. An object destroyed
        // inside the window keeps only this record in the snapshot: its created record is neither in
        // the history, where it would have aged out, nor in the finalize attach block, where the entry
        // is already retired.
        static bool ConsumeCudaApiTraceObjectDestroyed(TsfSnapshotConsumer& consumer, NvTrace_CUDA_ApiTrace_RecordType recordType, const ThreadEventRecord& record)
        {
            switch (recordType)
            {
                case NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA context-destroyed ThreadEvent record (payloadSize=%zu).", record.payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA stream-destroyed ThreadEvent record (payloadSize=%zu).", record.payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded:
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A, true, 50, "Invalid CUDA module-unloaded ThreadEvent record (payloadSize=%zu).", record.payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA graph-exec-destroyed ThreadEvent record (payloadSize=%zu).", record.payloadSize);
                    break;
                default:
                    NVTRACE_LOG_ERR(50, "Unknown CUDA object-destroyed ThreadEvent event type %u.", static_cast<uint32_t>(recordType));
                    return true;
            }

            const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CudaApiTrace>(consumer.m_recordWriter, recordType, record.correlationId, record.pPayload, record.payloadSize);
            NVTRACE_LOG_ERR_IF(!written, 50, "Failed WriteRecord() for CUDA object-destroyed record (recordType=%u).", static_cast<uint32_t>(recordType));
            return written;
        }

        static bool ConsumeThreadEvent_CudaApiTraceCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            TsfSnapshotConsumer& consumer = *reinterpret_cast<TsfSnapshotConsumer*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            CudaApiTraceProviderState& providerState = *static_cast<CudaApiTraceProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            CudaApiTraceObserver::ThreadEventRecordStorage recordStorage;
            const ThreadEventRecord* const pRecord = providerState.observer.TranslateThreadEventRecord(record, recordStorage);
            if (!pRecord)
            {
                return true;
            }

            // Created-object announcements are emitted at finalize instead, while the object is still alive.
            if (pRecord->recordType == NvTrace_ReservedRecordType_RecordReference)
            {
                return true;
            }

            const auto recordType = static_cast<NvTrace_CUDA_ApiTrace_RecordType>(pRecord->recordType);
            switch (recordType)
            {
                case NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed:
                case NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed:
                case NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded:
                case NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed:
                    return ConsumeCudaApiTraceObjectDestroyed(consumer, recordType, *pRecord);
                default:
                    return Codec::CudaApiTraceEventEncoders::WriteApiTraceEventRecord(consumer.m_recordWriter, *pRecord);
            }
        }

        static bool ConsumeThreadEvent_NvtxTraceCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            TsfSnapshotConsumer& consumer = *reinterpret_cast<TsfSnapshotConsumer*>(pConsumer);
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

                // Strings and domains never retire, so the finalize attach block carries them whole. A
                // range or a stack group can be created and released inside the window, its slot reused
                // before finalize, so its group is written into the history the moment it is announced.
                return NvtxTraceSerialization::SerializeReferencedStateEvent(providerState.observer, consumer.m_recordWriter, reference, [](uint32_t leadingRecordType) {
                    return leadingRecordType == NvTrace_NvtxTrace_RecordType_PushPopRangeCreated || leadingRecordType == NvTrace_NvtxTrace_RecordType_StackGroupCreated;
                });
            }

            const auto recordType = static_cast<NvTrace_NvtxTrace_RecordType>(pRecord->recordType);
            switch (recordType)
            {
                case NvTrace_NvtxTrace_RecordType_StackGroupDestroyed:
                {
                    // The destroyed record rides its own thread-event commit, bytes intact.
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

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteSystemCreatedRecord(m_attachRecordWriter, systemState.systemInfo))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            m_systemState.systemCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();
            m_attachRecordWriter.SetSystemInfoRecordLocation(m_systemState.systemCreatedRecordLocation);
        }

        void AttachGpuPartition(GpuPartitionState& gpuPartitionState)
        {
            gpuPartitionState.gpuPartitionCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteGpuPartitionCreatedRecord(m_attachRecordWriter, gpuPartitionState.luid, gpuPartitionState.gpuUuid, gpuPartitionState.creationTimestampTsc, gpuPartitionState.gpuPartitionInfo))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            gpuPartitionState.gpuPartitionCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();
        }
        void AttachProcess(ProcessState& processState)
        {
            processState.processCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProcessCreatedRecord(m_attachRecordWriter, processState.luid, processState.namespacePid, processState.globalPid, processState.pidNamespaceId, processState.path.c_str(), processState.creationTimestampTsc))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            processState.processCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();
        }

        template <Codec::ProviderTypeId TProviderTypeId, typename TProviderState>
        bool TryConsumeOnAttachRecords(TProviderState& providerState)
        {
            const size_t previousRecordCount = providerState.onAttachRecordLocations.size();

            // TsfSnapshotConsumer reuses its observer across snapshots, so every checkpoint must read all ready
            // on-attach records regardless of whether an earlier snapshot already consumed them.
            providerState.observer.ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(
                [&](uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (!Codec::EventEncoders::WriteRecord<TProviderTypeId>(m_attachRecordWriter, recordType, NVTRACE_NO_CORRELATION_ID, pPayload, payloadSize))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write %s on-attach record (providerLuid=%" PRIu64 " recordType=%" PRIu32 ").", Codec::ToCString(TProviderTypeId), providerState.luid, recordType);
                        return;
                    }

                    m_attachRecordWriter.Commit();
                    providerState.onAttachRecordLocations.push_back(m_attachRecordWriter.LastCommittedRecordLocation());
                });

            return previousRecordCount != providerState.onAttachRecordLocations.size();
        }

        template <typename TProviderState>
        void EmitProviderAttachedRecord(TProviderState& providerState)
        {
            if (!Codec::NvTraceServiceEncoders::WriteProviderAttachedRecord(m_attachRecordWriter, providerState.luid, providerState.onAttachRecordLocations))
            {
                return;
            }

            m_attachRecordWriter.Commit();
            providerState.providerAttachedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();
        }

        void AttachProvider_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, NVTRACE_NIL_LUID, NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid, "CPU Clock Calibration"))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetSystemProviderRef(providerState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::CpuClockCalibration>(providerState);
            m_attachRecordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, "GPU Clock Calibration"))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuClockCalibration>(providerState);
            m_attachRecordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_GPU_PmCounters_ProviderTypeUuid, "GPU PM Counters"))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuPmCounters>(providerState);
            m_attachRecordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_GPU_HwTrace_ProviderTypeUuid, "GPU HW Trace"))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuHwTrace>(providerState);
            m_attachRecordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_GPU_GrCtxTrace_ProviderTypeUuid, "GPU GR Context Trace"))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuGrCtxTrace>(providerState);
            m_attachRecordWriter.SetNullProviderRef();
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
            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_CUDA_ApiTrace_ProviderTypeUuid, "CUDA API Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetProcessProviderRef(providerState, parentState);

            CudaApiTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_attachRecordWriter, [&] {
                providerState.onAttachRecordLocations.push_back(m_attachRecordWriter.LastCommittedRecordLocation());
            });

            // CUDA keeps writing provider-owned records after ProviderAttached, which the null
            // provider ref carries, so the process provider ref is re-set behind it.
            m_attachRecordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
            m_attachRecordWriter.SetProcessProviderRef(providerState, parentState);

            CudaApiTraceSerialization::SerializeStateRecords(providerState.observer, m_attachRecordWriter);
        }

        void AttachProvider_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();

            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_NvtxTrace_ProviderTypeUuid, "NVTX Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            m_attachRecordWriter.SetProcessProviderRef(providerState, parentState);

            NvtxTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_attachRecordWriter, [&] {
                providerState.onAttachRecordLocations.push_back(m_attachRecordWriter.LastCommittedRecordLocation());
            });

            // NVTX keeps writing provider-owned records after ProviderAttached, which the null
            // provider ref carries, so the process provider ref is re-set behind it.
            m_attachRecordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
            m_attachRecordWriter.SetProcessProviderRef(providerState, parentState);

            NvtxTraceSerialization::SerializeStateRecords(providerState.observer, m_attachRecordWriter);
        }

        // TSF records mappings in each DataBlock's reference table, rather than serializing a
        // ProviderMapped record, so a new binding only has to re-emit its reference.
        void OnThreadEventSourceProviderBound(ThreadEventProviderState&, ProviderState& providerState)
        {
            providerState.threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        }

        void AttachProvider_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_attachRecordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_attachRecordWriter, providerState.luid, parentState.luid, NvTrace_ThreadEvent_ProviderTypeUuid, "Thread Event"))
            {
                return;
            }
            m_attachRecordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_attachRecordWriter.LastCommittedRecordLocation();

            Super::ConsumeThreadEventSourceProviderRegistrations(parentState, providerState, ReadCpuTscTimestamp());

            m_attachRecordWriter.SetProcessProviderRef(providerState, parentState);

            ThreadEventSerialization::SerializeThreadStateEvents(providerState.observer, m_attachRecordWriter);
        }

        void AttachExistingObjects(bool includeZombies)
        {
            m_attachRecordWriter.SetLastConsumeTimestampUtcNsec(ConvertTscTickToUtcNsec(ReadCpuTscTimestamp()));

            // Attach System created event
            if (m_systemState.pSystemInfoMemory)
            {
                AttachSystem(m_systemState);
            }

            // Attach System providers
            {
                if (m_systemState.cpuClockCalibrationProvider.IsInitialized() && (includeZombies || !m_systemState.cpuClockCalibrationProvider.pendingDestroy))
                {
                    AttachProvider_CpuClockCalibration(m_systemState.cpuClockCalibrationProvider);
                }
            }

            // Attach GpuPartitions and their providers
            for (auto& [_, gpuPartitionState] : m_systemState.gpuPartitions)
            {
                if (!includeZombies && gpuPartitionState.pendingDestroy)
                {
                    continue;
                }
                AttachGpuPartition(gpuPartitionState);
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized() && (includeZombies || !gpuPartitionState.gpuClockCalibrationProvider.pendingDestroy))
                {
                    AttachProvider_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized() && (includeZombies || !gpuPartitionState.gpuPmCountersProvider.pendingDestroy))
                {
                    AttachProvider_GpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized() && (includeZombies || !gpuPartitionState.gpuHwTraceProvider.pendingDestroy))
                {
                    AttachProvider_GpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized() && (includeZombies || !gpuPartitionState.gpuGrCtxTraceProvider.pendingDestroy))
                {
                    AttachProvider_GpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                }
            }

            // Attach Processes and their providers
            for (auto& [_, processState] : m_systemState.processes)
            {
                if (!includeZombies && processState.pendingDestroy)
                {
                    continue;
                }
                AttachProcess(processState);
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    if (threadEventProviderEntry.second.IsInitialized() && (includeZombies || !threadEventProviderEntry.second.pendingDestroy))
                    {
                        AttachProvider_ThreadEvent(threadEventProviderEntry.second, processState);
                    }
                }
                if (processState.cudaApiTraceProvider.IsInitialized() && (includeZombies || !processState.cudaApiTraceProvider.pendingDestroy))
                {
                    AttachProvider_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                }
                if (processState.nvtxTraceProvider.IsInitialized() && (includeZombies || !processState.nvtxTraceProvider.pendingDestroy))
                {
                    AttachProvider_NvtxTrace(processState.nvtxTraceProvider, processState);
                }
            }
        }

        void OnHistoryBufferDataBlockAgedInCallback(TraceStreamDataBlockWriter& dataBlockWriter)
        {
            m_snapshotTracker.AcknowledgePublishedDataBlock(dataBlockWriter);
        }

        void OnHistoryBufferDataBlockAgedOutCallback(const TsfSnapshotHistoryBuffer::DataBlockInfo& dataBlockInfo)
        {
            m_lastAgedOutDataBlockSequenceNumber = dataBlockInfo.sequenceNumber;
        }

        void OnDatabaseStorageWriteCompletedCallback(uint64_t dataBlockSequenceNumber, bool isStorageWriteCompletedSuccessfully)
        {
            SignalStorageWriteCompletedEvent(dataBlockSequenceNumber, isStorageWriteCompletedSuccessfully);
        }

        void OnSnapshotTrackerSnapshotCompletedCallback(uint32_t snapshotNumber, bool isSnapshotCompletedSuccessfully)
        {
            SignalSnapshotCompletedEvent(snapshotNumber, isSnapshotCompletedSuccessfully);
        }

        bool ScheduleCheckpoint(uint32_t snapshotNumber, uint64_t timestampUtcNsec)
        {
            std::lock_guard<std::mutex> lock(m_consumeThreadEventsMutex);
            m_pendingCheckpoints.push(std::make_pair(timestampUtcNsec, snapshotNumber));
            return true;
        }

    protected:
        void ReplayExpiredZombieObjects()
        {
            // Replay can itself consume records and publish blocks. Leave any newly aged-out
            // sequences for the next cycle rather than extending this cycle's cleanup boundary.
            const uint64_t agedOutDataBlockSequenceNumber = m_lastAgedOutDataBlockSequenceNumber;
            m_recordWriter.SetNullProviderRef();
            m_attachRecordWriter.SetNullProviderRef();
            auto clearProviderRefsGuard = ScopeExitGuard([this]() {
                m_recordWriter.SetNullProviderRef();
                m_attachRecordWriter.SetNullProviderRef();
            });
            while (!m_zombieObjectQueue.empty())
            {
                const uint64_t lastDataBlockSequenceNumber = m_zombieObjectQueue.front().lastDataBlockSequenceNumber;
                if (lastDataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER || lastDataBlockSequenceNumber > agedOutDataBlockSequenceNumber)
                {
                    return;
                }
                if (!ReplayZombieObject(m_zombieObjectQueue.begin()))
                {
                    return;
                }
            }
        }

        using Super::ConvertTscTickToUtcNsecForConsume;
        using Super::UpdateConsumeCpuClockCalibration;

        void HandleCheckpointRequests()
        {
            const uint64_t currentTimestampTsc = ReadCpuTscTimestamp();
            const uint64_t currentTimestampUtcNsec = ConvertTscTickToUtcNsecForConsume(currentTimestampTsc);
            m_snapshotNumbersForCurrentCheckpoint.clear();
            {
                std::lock_guard<std::mutex> lock(m_consumeThreadEventsMutex);
                while (!m_pendingCheckpoints.empty() && m_pendingCheckpoints.top().first <= currentTimestampUtcNsec)
                {
                    m_snapshotNumbersForCurrentCheckpoint.push_back(m_pendingCheckpoints.top().second);
                    m_pendingCheckpoints.pop();
                }
            }
            if (m_snapshotNumbersForCurrentCheckpoint.empty())
            {
                return;
            }

            // Signal even on failure, otherwise the snapshot commands would never be finalized
            auto signalGuard = ScopeExitGuard([&]() {
                for (const uint32_t snapshotNumber : m_snapshotNumbersForCurrentCheckpoint)
                {
                    SignalCheckpointCompletedEvent(snapshotNumber);
                }
                m_snapshotNumbersForCurrentCheckpoint.clear();
            });

            AttachExistingObjects(true);

            if (!m_attachRecordWriter.Checkpoint())
            {
                NVTRACE_LOG_ERR(50, "HandleCheckpointRequests: Failed to stage attach data block.");
                return;
            }

            if (!m_recordWriter.Checkpoint())
            {
                NVTRACE_LOG_ERR(50, "HandleCheckpointRequests: Failed to stage data block.");
                return;
            }
        }

        void ConsumeData(uint64_t readEndpointTsc = ReadCpuTscTimestamp())
        {
            UpdateConsumeCpuClockCalibration();
            const uint64_t readEndpointUtcNsec = ConvertTscTickToUtcNsecForConsume(readEndpointTsc);
            m_recordWriter.SetLastConsumeTimestampUtcNsec(readEndpointUtcNsec);

            // [TODO] Avoid IsInitialized() check in every WriteRecord() path
            // We can try one-time acquire here and stop this consuming round if the record writer's data block remains not initialized, meaning no free data block at this moment

            Super::ConsumeData(readEndpointTsc);

            SealQueuedZombieObjects();
        }

        bool HandleSnapshotCommand()
        {
            uint64_t historyDurationNsec = m_snapshotParam.historyDurationNsec;
            const uint64_t futureDurationNsec = m_snapshotParam.futureDurationNsec;
            uint32_t snapshotNumber = TsfSnapshotTracker::INVALID_SNAPSHOT_NUMBER;
            auto cancelSnapshotGuard = ScopeExitGuard([&]() {
                if (snapshotNumber != TsfSnapshotTracker::INVALID_SNAPSHOT_NUMBER)
                {
                    m_snapshotTracker.CancelSnapshot(snapshotNumber);
                }
            });
            m_historyBuffer.ForEachDataBlockLatestFirst(
                [&](size_t historySize, const TsfSnapshotHistoryBuffer::DataBlockInfo& earliestDataBlockInfo) {
                    uint64_t initialDataBlockSequenceNumber = TSF_DATA_BLOCK_SEQUENCE_NUMBER_LOW;
                    if (historySize > 0)
                    {
                        initialDataBlockSequenceNumber = earliestDataBlockInfo.sequenceNumber;
                        if (historyDurationNsec == 0)
                        {
                            historyDurationNsec = m_snapshotTriggerTimestampUtcNsec - earliestDataBlockInfo.lowerBoundTimestampUtcNsec;
                        }
                    }
                    snapshotNumber = m_snapshotTracker.CreateSnapshot(
                        m_snapshotTriggerTimestampUtcNsec,
                        historyDurationNsec,
                        futureDurationNsec,
                        std::move(m_snapshotStorageWriter),
                        initialDataBlockSequenceNumber);
                },
                [&](const TsfSnapshotHistoryBuffer::DataBlockInfo& dataBlockInfo, TraceStreamDataBlockWriter& dataBlockWriter) {
                    if (snapshotNumber == TsfSnapshotTracker::INVALID_SNAPSHOT_NUMBER)
                    {
                        return false;
                    }
                    if (dataBlockWriter.IsInitialized())
                    {
                        m_snapshotTracker.AcknowledgePublishedDataBlock(
                            snapshotNumber,
                            dataBlockWriter);
                    }
                    else
                    {
                        m_snapshotTracker.AcknowledgePublishedDataBlock(
                            snapshotNumber,
                            dataBlockInfo.sequenceNumber,
                            dataBlockInfo.lowerBoundTimestampUtcNsec,
                            dataBlockInfo.upperBoundTimestampUtcNsec,
                            m_databaseWriter.HasDataBlockInStorage(dataBlockInfo.sequenceNumber));
                    }
                    return true;
                });
            if (snapshotNumber == TsfSnapshotTracker::INVALID_SNAPSHOT_NUMBER)
            {
                NVTRACE_LOG_ERR(50, "HandleSnapshotCommand: Failed to create snapshot command.");
                return false;
            }

            // Schedule the checkpoint that finalizes this snapshot.
            if (!ScheduleCheckpoint(snapshotNumber, m_snapshotTriggerTimestampUtcNsec + futureDurationNsec))
            {
                NVTRACE_LOG_ERR(50, "HandleSnapshotCommand: Failed to schedule checkpoint (snapshotNumber=%" PRIu32 ").", snapshotNumber);
                return false;
            }

            cancelSnapshotGuard.Dismiss();
            if (m_fnOnSnapshotTaken)
            {
                m_fnOnSnapshotTaken(m_pUserDataOnSnapshotTaken, snapshotNumber);
            }
            return true;
        }

        void SignalCheckpointCompletedEvent(uint32_t snapshotNumber)
        {
            {
                std::lock_guard<std::mutex> lock(m_snapshotThreadEventsMutex);
                m_checkpointCompletedEvents.emplace_back(snapshotNumber);
            }
            const bool succeeded = m_snapshotThreadCheckpointCompletedEvent.Signal();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "SignalCheckpointCompletedEvent: AutoResetEvent::Signal failed (m_snapshotThreadCheckpointCompletedEvent)");
        }

        void HandleCheckpointCompletedEvents()
        {
            std::vector<uint32_t> checkpointCompletedEvents;
            {
                std::lock_guard<std::mutex> lock(m_snapshotThreadEventsMutex);
                while (!m_checkpointCompletedEvents.empty())
                {
                    checkpointCompletedEvents.emplace_back(m_checkpointCompletedEvents.back());
                    m_checkpointCompletedEvents.pop_back();
                }
            }
            for (const uint32_t snapshotNumber : checkpointCompletedEvents)
            {
                if (!m_snapshotTracker.FinalizeSnapshot(snapshotNumber, [this](TraceStreamDataBlockWriter& dataBlockWriter) {
                        return m_databaseWriter.FinalizeDataBlock(dataBlockWriter);
                    }))
                {
                    NVTRACE_LOG_ERR(50, "HandleCheckpointCompletedEvents: Failed to finalize snapshot (snapshotNumber=%" PRIu32 ").", snapshotNumber);
                }
            }
        }

        void SignalStorageWriteCompletedEvent(uint64_t dataBlockSequenceNumber, bool isStorageWriteCompletedSuccessfully)
        {
            {
                std::lock_guard<std::mutex> lock(m_snapshotThreadEventsMutex);
                m_storageWriteCompletedEvents.emplace_back(dataBlockSequenceNumber, isStorageWriteCompletedSuccessfully);
            }
            const bool succeeded = m_snapshotThreadStorageWriteCompletedEvent.Signal();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "SignalStorageWriteCompletedEvent: AutoResetEvent::Signal failed (m_snapshotThreadStorageWriteCompletedEvent)");
        }

        void HandleStorageWriteCompletedEvents()
        {
            std::vector<std::pair<uint64_t, bool>> storageWriteCompletedEvents;
            {
                std::lock_guard<std::mutex> lock(m_snapshotThreadEventsMutex);
                while (!m_storageWriteCompletedEvents.empty())
                {
                    storageWriteCompletedEvents.emplace_back(std::move(m_storageWriteCompletedEvents.back()));
                    m_storageWriteCompletedEvents.pop_back();
                }
            }
            for (const auto& storageWriteCompletedEvent : storageWriteCompletedEvents)
            {
                const uint64_t dataBlockSequenceNumber = storageWriteCompletedEvent.first;
                const bool isStorageWriteCompletedSuccessfully = storageWriteCompletedEvent.second;
                const bool isAttachDataBlock = m_databaseWriter.IsReservedDataBlock(dataBlockSequenceNumber);
                m_snapshotTracker.AcknowledgeCompletedDataBlock(dataBlockSequenceNumber, isStorageWriteCompletedSuccessfully, isAttachDataBlock);
            }
        }

        void SignalSnapshotCompletedEvent(uint32_t snapshotNumber, bool isSnapshotCompletedSuccessfully)
        {
            {
                std::lock_guard<std::mutex> lock(m_snapshotThreadEventsMutex);
                m_snapshotCompletedEvents.emplace_back(snapshotNumber, isSnapshotCompletedSuccessfully);
            }
            const bool succeeded = m_snapshotThreadSnapshotCompletedEvent.Signal();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "SignalSnapshotCompletedEvent: AutoResetEvent::Signal failed (m_snapshotThreadSnapshotCompletedEvent)");
        }

        void HandleSnapshotCompletedEvents()
        {
            std::vector<std::pair<uint32_t, bool>> snapshotCompletedEvents;
            {
                std::lock_guard<std::mutex> lock(m_snapshotThreadEventsMutex);
                while (!m_snapshotCompletedEvents.empty())
                {
                    snapshotCompletedEvents.emplace_back(std::move(m_snapshotCompletedEvents.back()));
                    m_snapshotCompletedEvents.pop_back();
                }
            }

            for (const auto& snapshotCompletedEvent : snapshotCompletedEvents)
            {
                const uint32_t snapshotNumber = snapshotCompletedEvent.first;
                TsfSnapshotCommand snapshotCommand;
                if (!m_snapshotTracker.RemoveCompletedSnapshot(snapshotNumber, snapshotCommand))
                {
                    NVTRACE_LOG_ERR(50, "HandleSnapshotCompletedEvents: Completed snapshot command is not found (snapshotNumber=%" PRIu32 ").", snapshotNumber);
                    continue;
                }

                const TStorageWriter& primaryStorageWriter = m_databaseWriter.StorageWriter();
                TStorageWriter& snapshotStorageWriter = snapshotCommand.StorageWriter();

                auto hardlinkDataBlockRange = [&](uint64_t beginDataBlockSequenceNumber, uint64_t endDataBlockSequenceNumber) {
                    if (beginDataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER || endDataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
                    {
                        return true;
                    }
                    // Hardlink in descending sequence number and stop at the first failed data block,
                    // so that a snapshot never spans a gap in the data block sequence numbers.
                    for (uint64_t dataBlockSequenceNumberToHardlink = endDataBlockSequenceNumber;;)
                    {
                        if (!primaryStorageWriter.HardlinkDataBlock(snapshotStorageWriter, dataBlockSequenceNumberToHardlink))
                        {
                            NVTRACE_LOG_ERR(50, "HandleSnapshotCompletedEvents: Failed to hardlink data block (snapshotNumber=%" PRIu32 ", dataBlockSequenceNumber=%" PRIu64 ").", snapshotNumber, dataBlockSequenceNumberToHardlink);
                            return false;
                        }
                        if (dataBlockSequenceNumberToHardlink == beginDataBlockSequenceNumber)
                        {
                            break;
                        }
                        dataBlockSequenceNumberToHardlink--;
                    }
                    return true;
                };

                bool succeeded = snapshotCompletedEvent.second;
                if (!hardlinkDataBlockRange(snapshotCommand.BeginDataBlockSequenceNumber(), snapshotCommand.EndDataBlockSequenceNumber()))
                {
                    succeeded = false;
                }
                if (!hardlinkDataBlockRange(snapshotCommand.BeginAttachDataBlockSequenceNumber(), snapshotCommand.EndAttachDataBlockSequenceNumber()))
                {
                    succeeded = false;
                }
                if (m_fnOnSnapshotCompleted)
                {
                    m_fnOnSnapshotCompleted(m_pUserDataOnSnapshotCompleted, snapshotNumber, succeeded);
                }
            }
        }

    private:
        void ConsumeLoop() noexcept
        {
            auto exitGuard = ScopeExitGuard([this]() {
                m_consumeThreadExited.store(true, std::memory_order_release);

                bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                NVTRACE_LOG_ERR_IF(!succeeded, 50, "ConsumeLoop exit: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");
            });

            try
            {
                ConsumeLoopImpl();
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotConsumer consume thread terminated after an exception: %s.", e.what());
            }
            catch (...)
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotConsumer consume thread terminated after an unknown exception.");
            }
        }

        void ConsumeLoopImpl()
        {
            while (true)
            {
                ConsumeThreadEventIdentifier signaledIdentifier;
                if (!m_consumeThreadWaitMultiplexer.Wait(signaledIdentifier))
                {
                    NVTRACE_LOG_ERR(50, "ConsumeLoop: WaitMultiplexer::Wait returned false (see preceding WaitMultiplexer log for errno or Win32 error).");
                    return;
                }

                switch (signaledIdentifier)
                {
                    case ConsumeThreadEventIdentifier::Command:
                    {
                        std::atomic_thread_fence(std::memory_order_acquire);
                        switch (m_consumeThreadCommand)
                        {
                            case ConsumeThreadCommand::Terminate:
                                return;
                        }

                        break;
                    }

                    case ConsumeThreadEventIdentifier::RouterNotification:
                    {
                        bool succeeded = m_pRouterObserverClient->ReceiveNotificationIfPending([this](NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) {
                            return HandleRouterNotification(notificationType, pNotification);
                        });
                        NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "RouterObserverClient::ReceiveNotificationIfPending failed (router observer handle may be invalid).");

                        break;
                    }

                    case ConsumeThreadEventIdentifier::ConsumeData:
                    {
                        if (m_isConsuming)
                        {
                            if (!ScheduleNextConsumeIteration())
                            {
                                return;
                            }

                            ConsumeData();

                            ReplayExpiredZombieObjects();
                        }

                        HandleCheckpointRequests();

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

        void SnapshotLoop()
        {
            auto exitGuard = ScopeExitGuard([this]() {
                m_snapshotThreadExited.store(true, std::memory_order_release);

                bool succeeded = m_snapshotThreadCommandCompletedEvent.Signal();
                NVTRACE_LOG_ERR_IF(!succeeded, 50, "SnapshotLoop exit: AutoResetEvent::Signal failed (snapshotThreadCommandCompletedEvent)");
            });

            try
            {
                SnapshotLoopImpl();
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotConsumer snapshot thread terminated after an exception: %s.", e.what());
            }
            catch (...)
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotConsumer snapshot thread terminated after an unknown exception.");
            }
        }

        void SnapshotLoopImpl()
        {
            while (true)
            {
                SnapshotThreadEventIdentifier signaledIdentifier;
                if (!m_snapshotThreadWaitMultiplexer.Wait(signaledIdentifier))
                {
                    NVTRACE_LOG_ERR(50, "SnapshotLoop: WaitMultiplexer::Wait returned false (see preceding WaitMultiplexer log for errno or Win32 error).");
                    return;
                }

                switch (signaledIdentifier)
                {
                    case SnapshotThreadEventIdentifier::Command:
                    {
                        std::atomic_thread_fence(std::memory_order_acquire);
                        switch (m_snapshotThreadCommand)
                        {
                            case SnapshotThreadCommand::Terminate:
                                return;

                            case SnapshotThreadCommand::Snapshot:
                            {
                                auto exitGuard = ScopeExitGuard([this]() {
                                    bool succeeded = m_snapshotThreadCommandCompletedEvent.Signal();
                                    NVTRACE_LOG_ERR_IF(!succeeded, 50, "SnapshotLoop Snapshot: AutoResetEvent::Signal failed (snapshotThreadCommandCompletedEvent)");
                                });

                                m_snapshotThreadCommandSucceeded = HandleSnapshotCommand();
                                if (!m_snapshotThreadCommandSucceeded)
                                {
                                    NVTRACE_LOG_ERR(50, "SnapshotLoop: HandleSnapshotCommand failed.");
                                    break;
                                }

                                break;
                            }
                        }

                        break;
                    }

                    case SnapshotThreadEventIdentifier::CheckpointCompleted:
                    {
                        HandleCheckpointCompletedEvents();
                        break;
                    }

                    case SnapshotThreadEventIdentifier::StorageWriteCompleted:
                    {
                        HandleStorageWriteCompletedEvents();
                        break;
                    }

                    case SnapshotThreadEventIdentifier::SnapshotCompleted:
                    {
                        HandleSnapshotCompletedEvents();
                        break;
                    }
                }
            }
        }

        void TerminateSnapshotThread() noexcept
        {
            if (m_snapshotThread.joinable())
            {
                m_snapshotThreadCommand = SnapshotThreadCommand::Terminate;
                std::atomic_thread_fence(std::memory_order_release);

                bool succeeded = m_snapshotThreadCommandPendingEvent.Signal();
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "TerminateSnapshotThread: AutoResetEvent::Signal failed (snapshotThreadCommandPendingEvent)");

                m_snapshotThread.join();
            }
        }

    private:
        static uint64_t EncodeRecordLocation(void* pUserData, uint64_t dataBlockSequenceNumber, uint32_t chunkIndex, uint32_t recordIndex)
        {
            auto& consumer = *static_cast<TsfSnapshotConsumer*>(pUserData);
            return consumer.m_databaseWriter.EncodeRecordLocation(dataBlockSequenceNumber, chunkIndex, recordIndex);
        }

        static bool StageCurrentDataBlock(void* pUserData, TraceStreamDataBlockWriter&& dataBlockWriter)
        {
            auto& consumer = *static_cast<TsfSnapshotConsumer*>(pUserData);
            if (!dataBlockWriter.IsInitialized())
            {
                return true;
            }
            if (!consumer.m_historyBuffer.PushPublishedDataBlock(std::move(dataBlockWriter)))
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotHistoryBuffer::PushPublishedDataBlock failed (This should never happen).");
                return false;
            }
            return true;
        }

        // [TODO] There is a known issue that:
        // When there is no available data block, the data will be lost silently.
        // However, snapshots are not aware of data loss and may falsely complete with a successful result.
        // To fix this, one option is that we reserve a non-transferable data block for continuous data block consumption.
        // Always obtain a new data block first before staging the current data block.
        // If fail to obtain a new data block, still publish to history while retaining the current data block.
        static bool ObtainNewDataBlock(void* pUserData, TraceStreamDataBlockWriter& dataBlockWriter)
        {
            auto& consumer = *static_cast<TsfSnapshotConsumer*>(pUserData);
            if (consumer.m_databaseWriter.AcquireDataBlock(dataBlockWriter))
            {
                return true;
            }
            if (!consumer.m_historyBuffer.RelinquishDataBlock(dataBlockWriter))
            {
                if (!consumer.m_snapshotTracker.RelinquishDataBlock(dataBlockWriter))
                {
                    return false;
                }
            }
            if (!consumer.m_databaseWriter.RecycleDataBlock(dataBlockWriter))
            {
                return false;
            }
            return consumer.m_databaseWriter.AcquireDataBlock(dataBlockWriter);
        }

        static bool StageCurrentAttachDataBlock(void* pUserData, TraceStreamDataBlockWriter&& dataBlockWriter)
        {
            auto& consumer = *static_cast<TsfSnapshotConsumer*>(pUserData);
            if (!dataBlockWriter.IsInitialized())
            {
                return true;
            }
            const uint64_t dataBlockSequenceNumber = dataBlockWriter.DataBlockSequenceNumber();
            // Register the attach data block with every snapshot of the ongoing checkpoint before the
            // data block is handed to the storage writer, otherwise its completion event can arrive
            // before the snapshot is finalized and would be dropped by the snapshot tracker
            for (const uint32_t snapshotNumber : consumer.m_snapshotNumbersForCurrentCheckpoint)
            {
                if (!consumer.m_snapshotTracker.AddAttachDataBlock(snapshotNumber, dataBlockSequenceNumber))
                {
                    NVTRACE_LOG_ERR(50, "StageCurrentAttachDataBlock: Failed to add attach data block (snapshotNumber=%" PRIu32 ", dataBlockSequenceNumber=%" PRIu64 ").", snapshotNumber, dataBlockSequenceNumber);
                }
            }
            return consumer.m_databaseWriter.FinalizeDataBlock(dataBlockWriter);
        }

        static bool ObtainNewAttachDataBlock(void* pUserData, TraceStreamDataBlockWriter& dataBlockWriter)
        {
            auto& consumer = *static_cast<TsfSnapshotConsumer*>(pUserData);
            return consumer.m_databaseWriter.AcquireDataBlock(dataBlockWriter, 0, 0, true);
        }

    private:
        using Super::m_consumeTimer;
        using Super::m_pRouterObserverClient;

    protected:
        using Super::m_systemState;

        ZombieObjectQueue m_zombieObjectQueue;
        size_t m_zombieObjectBytes = 0;
        size_t m_maxZombieObjectBytes = 0;
        uint64_t m_lastAgedOutDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        uint64_t m_observationStartTsc = 0;
        // [TODO] Implement Start() and Stop() using m_isConsuming
        bool m_isConsuming = false;

        // To create a snapshot
        uint64_t m_snapshotTriggerTimestampUtcNsec = 0;
        SnapshotParam m_snapshotParam;
        std::unique_ptr<TStorageWriter> m_snapshotStorageWriter;

        FnOnSnapshotTaken m_fnOnSnapshotTaken = nullptr;
        void* m_pUserDataOnSnapshotTaken = nullptr;
        FnOnSnapshotCompleted m_fnOnSnapshotCompleted = nullptr;
        void* m_pUserDataOnSnapshotCompleted = nullptr;

        DatabaseWriter m_databaseWriter;
        TsfSnapshotHistoryBuffer m_historyBuffer;
        TsfSnapshotTracker m_snapshotTracker;

        TsfSnapshotRecordWriter m_recordWriter;
        TsfSnapshotAttachRecordWriter m_attachRecordWriter;

    private:
        // [TODO] Refactor to a class that manages the zombie object queue
        bool m_isReplayingZombieDestroy = false;

        enum class ConsumeThreadCommand
        {
            Terminate
        };

        enum class ConsumeThreadEventIdentifier
        {
            Command,
            RouterNotification,
            ConsumeData
        };

        std::mutex m_consumeThreadEventsMutex;
        // <TimestampUtcNsec, SnapshotNumber>, ordered by the first value (smallest on top)
        std::priority_queue<std::pair<uint64_t, uint32_t>, std::vector<std::pair<uint64_t, uint32_t>>, std::greater<std::pair<uint64_t, uint32_t>>> m_pendingCheckpoints;

        // Snapshots whose checkpoint is being served by the ongoing attach pass
        // Consume thread only
        std::vector<uint32_t> m_snapshotNumbersForCurrentCheckpoint;

        ConsumeThreadCommand m_consumeThreadCommand = ConsumeThreadCommand::Terminate;
        bool m_consumeThreadCommandSucceeded = false;
        AutoResetEvent m_consumeThreadCommandPendingEvent;
        AutoResetEvent m_consumeThreadCommandCompletedEvent;
        WaitMultiplexer m_consumeThreadWaitMultiplexer;
        std::thread m_consumeThread;
        std::atomic_bool m_consumeThreadExited = false;

        enum class SnapshotThreadCommand
        {
            Terminate,
            Snapshot
        };

        enum class SnapshotThreadEventIdentifier
        {
            Command,
            CheckpointCompleted,
            StorageWriteCompleted,
            SnapshotCompleted
        };

        std::mutex m_snapshotThreadEventsMutex;
        // <SnapshotNumber>, the snapshot whose checkpoint has completed
        std::vector<uint32_t> m_checkpointCompletedEvents;
        // <DataBlockSequenceNumber, isStorageWriteCompletedSuccessfully>
        std::vector<std::pair<uint64_t, bool>> m_storageWriteCompletedEvents;
        // <SnapshotNumber, isSnapshotCompletedSuccessfully>
        std::vector<std::pair<uint32_t, bool>> m_snapshotCompletedEvents;

        SnapshotThreadCommand m_snapshotThreadCommand = SnapshotThreadCommand::Terminate;
        bool m_snapshotThreadCommandSucceeded = false;
        AutoResetEvent m_snapshotThreadCommandPendingEvent;
        AutoResetEvent m_snapshotThreadCommandCompletedEvent;
        AutoResetEvent m_snapshotThreadCheckpointCompletedEvent;
        AutoResetEvent m_snapshotThreadStorageWriteCompletedEvent;
        AutoResetEvent m_snapshotThreadSnapshotCompletedEvent;
        WaitMultiplexer m_snapshotThreadWaitMultiplexer;
        std::thread m_snapshotThread;
        std::atomic_bool m_snapshotThreadExited = false;
    };

} // namespace nv::trace
