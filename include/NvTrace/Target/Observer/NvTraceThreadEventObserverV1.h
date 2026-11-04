/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/DataFormats/nvtrace_thread_event_v1.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceCircularBufferReaderV1.h>
#include <NvTrace/Target/Observer/NvTraceObserverUtilities.h>
#include <NvTrace/Target/Observer/NvTraceStateTableReader.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    /// \brief Thread Event Observer over the V1 provider buffer format -- the old-driver shim.
    ///
    /// Reads buffers published by drivers that predate the generic provider-buffer format and serves
    /// them through the same interface ThreadEventObserver exposes. Attribution comes from the V1
    /// per-chunk uniqueThreadId instead of ThreadChanged records, and the transport's own lifecycle
    /// records are translated on delivery -- their V1 record type values collide with the V2
    /// vocabulary -- so consumers see V2-shaped records only: a created event is resolved against the
    /// thread state table here and delivered as a full V2 ThreadCreated record, a destroyed event as
    /// the V2 destruction record.
    ///
    /// Source provider records are shared vocabulary and pass through, correlation ids kept on the
    /// shipped V1 chunk-geometry derivation.
    ///
    /// A V1 provider id is 1-based and only the static ones are served: static id N is delivered as
    /// source provider index N - 1, registered once the driver publishes its V1 mapping slot. A V1 static
    /// id fixes its provider type, so the registration names that type rather than the slot's. V1 dynamic
    /// ids are never registered, and their records are dropped.
    ///
    /// Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class ThreadEventObserverV1
    {
    public:
        ~ThreadEventObserverV1() = default;
        ThreadEventObserverV1() = default;

        ThreadEventObserverV1(const ThreadEventObserverV1&) = delete;
        ThreadEventObserverV1(ThreadEventObserverV1&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        ThreadEventObserverV1& operator=(const ThreadEventObserverV1&) = delete;
        ThreadEventObserverV1& operator=(ThreadEventObserverV1&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_sections = std::exchange(rhs.m_sections, {});
                m_eventBufferReaders = std::move(rhs.m_eventBufferReaders);
                m_deliveredStaticMappingsMask = std::exchange(rhs.m_deliveredStaticMappingsMask, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_deliveredStaticMappingsMask = 0;
            m_eventBufferReaders.clear();
            m_sections = {};
        }

        bool Initialize(const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "ThreadEventObserverV1::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            // The buffer is shared memory under the profiled process's control: every offset,
            // stride and count must be proven inside the mapping before it shapes a pointer or an
            // allocation. The divide form cannot overflow; every divisor is proven nonzero before
            // its check runs.
            const auto fnRangeFits = [&](uint64_t itemsOffset, size_t itemSize, uint64_t itemCount) {
                return itemsOffset != 0 && itemsOffset <= providerBufferSize && itemCount <= (providerBufferSize - itemsOffset) / itemSize;
            };

            NVTRACE_LOG_ERR_RETURN_IF(pMappedProviderBufferMemory == nullptr || providerBufferSize < sizeof(NvTrace_ProviderBufferHeader_V1), false, 20, "The mapped V1 provider buffer cannot hold its header (providerBufferSize %zu).", providerBufferSize);

            const NvTrace_ProviderBufferHeader_V1* const pProviderBufferHeader = reinterpret_cast<const NvTrace_ProviderBufferHeader_V1*>(pMappedProviderBufferMemory);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->providerTypeUuid != NvTrace_ThreadEvent_ProviderTypeUuid_V1, false, 20, "Invalid provider type UUID for ThreadEvent V1 observer (got %s, expected %s).", ToString(pProviderBufferHeader->providerTypeUuid).c_str(), ToString(NvTrace_ThreadEvent_ProviderTypeUuid_V1).c_str());
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->minorVersion < NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A, false, 20, "Provider buffer header minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", pProviderBufferHeader->minorVersion, NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->providerBufferLayoutMinorVersion < NvTrace_ThreadEvent_ProviderBufferLayout_V1_MINOR_VERSION_A, false, 20, "Provider buffer layout minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", pProviderBufferHeader->providerBufferLayoutMinorVersion, NvTrace_ThreadEvent_ProviderBufferLayout_V1_MINOR_VERSION_A);

            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(pProviderBufferHeader->providerBufferLayoutOffset, sizeof(NvTrace_ThreadEvent_ProviderBufferLayout_V1), 1), false, 20, "The provider buffer layout lies outside the mapping (offset %" PRIu64 ").", pProviderBufferHeader->providerBufferLayoutOffset);

            m_sections.pProviderBufferHeader = pProviderBufferHeader;
            AssignSectionPointer(pMappedProviderBufferMemory, pProviderBufferHeader->providerBufferLayoutOffset, m_sections.pProviderBufferLayout);

            const NvTrace_ThreadEvent_ProviderBufferLayout_V1& providerBufferLayout = *m_sections.pProviderBufferLayout;
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventVersionInfoMinorVersion < NvTrace_ThreadEvent_EventVersionInfo_V1_MINOR_VERSION_A, false, 20, "Event version info minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", providerBufferLayout.eventVersionInfoMinorVersion, NvTrace_ThreadEvent_EventVersionInfo_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.eventVersionInfoOffset, sizeof(NvTrace_ThreadEvent_EventVersionInfo_V1), 1), false, 20, "The event version info lies outside the mapping (offset %" PRIu64 ").", providerBufferLayout.eventVersionInfoOffset);
            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.eventVersionInfoOffset, m_sections.pEventVersionInfo);

            const NvTrace_ThreadEvent_EventVersionInfo_V1& eventVersionInfo = *m_sections.pEventVersionInfo;
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.threadStateMinorVersion < NvTrace_ThreadEvent_ThreadState_V1_MINOR_VERSION_A, false, 20, "Thread state minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", eventVersionInfo.threadStateMinorVersion, NvTrace_ThreadEvent_ThreadState_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.providerIdMappingMinorVersion < NvTrace_ThreadEvent_ProviderIdMapping_V1_MINOR_VERSION_A, false, 20, "Provider ID mapping minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", eventVersionInfo.providerIdMappingMinorVersion, NvTrace_ThreadEvent_ProviderIdMapping_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.threadLifecycleEventMinorVersion < NvTrace_ThreadEvent_ThreadLifecycle_Event_V1_MINOR_VERSION_A, false, 20, "Thread lifecycle event minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", eventVersionInfo.threadLifecycleEventMinorVersion, NvTrace_ThreadEvent_ThreadLifecycle_Event_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.eventBufferRecordMinorVersion < NvTrace_ThreadEvent_EventBufferRecord_V1_MINOR_VERSION_A, false, 20, "Event buffer record minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", eventVersionInfo.eventBufferRecordMinorVersion, NvTrace_ThreadEvent_EventBufferRecord_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.eventBufferRecordSize < eventVersionInfo.eventBufferRecordMinorVersion, false, 20, "Event buffer record size %u is below its minor version %u.", eventVersionInfo.eventBufferRecordSize, eventVersionInfo.eventBufferRecordMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.eventBufferChunkHeaderSize < NvTrace_ThreadEvent_EventBufferChunkHeader_V1_MINOR_VERSION_A, false, 20, "Event buffer chunk header size %u is incompatible with the ThreadEvent V1 observer (expected %u).", eventVersionInfo.eventBufferChunkHeaderSize, NvTrace_ThreadEvent_EventBufferChunkHeader_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventBufferStateMinorVersion < NvTrace_AdaptiveBufferState_V1_MINOR_VERSION_A, false, 20, "Event buffer state minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", providerBufferLayout.eventBufferStateMinorVersion, NvTrace_AdaptiveBufferState_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventBufferCellMinorVersion < NvTrace_AdaptiveBufferCell_V1_MINOR_VERSION_A, false, 20, "Event buffer cell minor version %u is incompatible with the ThreadEvent V1 observer (expected %u).", providerBufferLayout.eventBufferCellMinorVersion, NvTrace_AdaptiveBufferCell_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.chunkSize < static_cast<uint64_t>(eventVersionInfo.eventBufferChunkHeaderSize) + eventVersionInfo.eventBufferRecordSize, false, 20, "Chunk size %u is too small for header size %u and record size %u.", providerBufferLayout.chunkSize, eventVersionInfo.eventBufferChunkHeaderSize, eventVersionInfo.eventBufferRecordSize);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.providerIdMappingsCount < providerBufferLayout.providerIdMappingsStaticCount, false, 20, "Provider ID mappings count %u is below the static count %u.", providerBufferLayout.providerIdMappingsCount, providerBufferLayout.providerIdMappingsStaticCount);

            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.providerIdMappingSize < NvTrace_ThreadEvent_ProviderIdMapping_V1_MINOR_VERSION_A, false, 20, "Provider ID mapping size %u is below its minor version floor %u.", eventVersionInfo.providerIdMappingSize, NvTrace_ThreadEvent_ProviderIdMapping_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.threadStateSize < NvTrace_ThreadEvent_ThreadState_V1_MINOR_VERSION_A, false, 20, "Thread state size %u is below its minor version floor %u.", eventVersionInfo.threadStateSize, NvTrace_ThreadEvent_ThreadState_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventBufferStateSize < NvTrace_AdaptiveBufferState_V1_MINOR_VERSION_A, false, 20, "Event buffer state size %u is below its minor version floor %u.", providerBufferLayout.eventBufferStateSize, NvTrace_AdaptiveBufferState_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventBufferCellSize < NvTrace_AdaptiveBufferCell_V1_MINOR_VERSION_A, false, 20, "Event buffer cell size %u is below its minor version floor %u.", providerBufferLayout.eventBufferCellSize, NvTrace_AdaptiveBufferCell_V1_MINOR_VERSION_A);

            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.providerIdMappingsOffset, eventVersionInfo.providerIdMappingSize, providerBufferLayout.providerIdMappingsCount), false, 20, "The provider-id mappings lie outside the mapping (offset %" PRIu64 " count %u).", providerBufferLayout.providerIdMappingsOffset, providerBufferLayout.providerIdMappingsCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.threadStatesOffset, eventVersionInfo.threadStateSize, providerBufferLayout.threadStatesCount), false, 20, "The thread states lie outside the mapping (offset %" PRIu64 " count %u).", providerBufferLayout.threadStatesOffset, providerBufferLayout.threadStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.eventBufferStatesOffset, providerBufferLayout.eventBufferStateSize, providerBufferLayout.eventBufferStatesCount), false, 20, "The event buffer states lie outside the mapping (offset %" PRIu64 " count %u).", providerBufferLayout.eventBufferStatesOffset, providerBufferLayout.eventBufferStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.eventBufferCellsOffset, providerBufferLayout.eventBufferCellSize, static_cast<uint64_t>(providerBufferLayout.eventBufferStatesCount) * providerBufferLayout.eventBufferCellsCount), false, 20, "The event buffer cells lie outside the mapping (offset %" PRIu64 " count %u x %u).", providerBufferLayout.eventBufferCellsOffset, providerBufferLayout.eventBufferStatesCount, providerBufferLayout.eventBufferCellsCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.chunksOffset, providerBufferLayout.chunkSize, providerBufferLayout.chunkCount), false, 20, "The chunks lie outside the mapping (offset %" PRIu64 " count %u).", providerBufferLayout.chunksOffset, providerBufferLayout.chunkCount);

            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.providerIdMappingsOffset, m_sections.pProviderIdMappings);
            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.threadStatesOffset, m_sections.pThreadStates);
            const NvTrace_AdaptiveBufferState_V1* pEventBufferStates = nullptr;
            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.eventBufferStatesOffset, pEventBufferStates);
            const NvTrace_AdaptiveBufferCell_V1* pEventBufferCells = nullptr;
            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.eventBufferCellsOffset, pEventBufferCells);
            const uint8_t* pChunks = nullptr;
            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.chunksOffset, pChunks);

            std::vector<EventBufferReader> eventBufferReaders(providerBufferLayout.eventBufferStatesCount);
            for (uint32_t eventBufferIndex = 0; eventBufferIndex < providerBufferLayout.eventBufferStatesCount; ++eventBufferIndex)
            {
                const auto* const pEventBufferState = reinterpret_cast<const NvTrace_AdaptiveBufferState_V1*>(reinterpret_cast<const uint8_t*>(pEventBufferStates) + static_cast<size_t>(eventBufferIndex) * providerBufferLayout.eventBufferStateSize);
                const auto* const pEventBufferCell = reinterpret_cast<const NvTrace_AdaptiveBufferCell_V1*>(reinterpret_cast<const uint8_t*>(pEventBufferCells) + static_cast<size_t>(eventBufferIndex) * providerBufferLayout.eventBufferCellsCount * providerBufferLayout.eventBufferCellSize);
                eventBufferReaders[eventBufferIndex].Initialize(
                    pEventBufferState,
                    pEventBufferCell,
                    providerBufferLayout.eventBufferCellsCount,
                    providerBufferLayout.eventBufferCellSize,
                    eventVersionInfo.eventBufferChunkHeaderSize,
                    eventVersionInfo.eventBufferRecordSize,
                    pChunks,
                    providerBufferLayout.chunkCount,
                    providerBufferLayout.chunkSize);
            }

            m_eventBufferReaders = std::move(eventBufferReaders);

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully over a V1 provider buffer.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_sections.pProviderBufferHeader != nullptr;
        }

        // Delivers the registration of each V1 static id once the driver has published its mapping, exactly
        // once across calls, as fnOnRegistered(uint32_t sourceProviderIndex, const NVTRACE_UUID& providerTypeUuid).
        // A V1 static id is its mapping table index, and may sit unpublished between published ones.
        template <typename TFN_OnRegistered>
        void ConsumeSourceProviderRegistrations(TFN_OnRegistered&& fnOnRegistered)
        {
            if (!m_sections.pProviderIdMappings)
            {
                return;
            }

            const uint32_t staticMappingCount = std::min<uint32_t>(m_sections.pProviderBufferLayout->providerIdMappingsStaticCount, NvTrace_ThreadEvent_StaticProviderId_V1_COUNT);
            for (uint32_t v1ProviderId = NvTrace_ThreadEvent_StaticProviderId_V1_ThreadEvent; v1ProviderId < staticMappingCount; ++v1ProviderId)
            {
                static_assert(NvTrace_ThreadEvent_StaticProviderId_V1_COUNT <= 32, "m_deliveredStaticMappingsMask holds one bit per V1 static id.");
                const uint32_t v1ProviderIdBit = uint32_t(1) << v1ProviderId;
                if ((m_deliveredStaticMappingsMask & v1ProviderIdBit) != 0 || !IsProviderIdMappingPublished(v1ProviderId))
                {
                    continue;
                }

                m_deliveredStaticMappingsMask |= v1ProviderIdBit;
                fnOnRegistered(v1ProviderId - 1, StaticProviderTypeUuid(v1ProviderId));
            }
        }

        size_t GetMaxRecordPayloadSize() const
        {
            if (!m_sections.pProviderBufferLayout)
            {
                return 0;
            }
            const size_t chunkSize = m_sections.pProviderBufferLayout->chunkSize;
            const size_t recordOverhead = sizeof(NvTrace_ThreadEvent_EventBufferChunkHeader_V1) + sizeof(NvTrace_ThreadEvent_EventBufferRecord_V1);
            if (chunkSize <= recordOverhead)
            {
                return 0;
            }
            return AlignDnP2(chunkSize - recordOverhead, alignof(uint64_t));
        }

        // The last V1 static id, delivered as id - 1
        static uint32_t MaxStaticSourceProviderIndex()
        {
            return (NvTrace_ThreadEvent_StaticProviderId_V1_COUNT - 1) - 1;
        }

        // Only the V1 static ids, delivered as id - 1: V1 dynamic ids are never registered.
        static uint32_t MaxSourceProviderCount()
        {
            return NvTrace_ThreadEvent_StaticProviderId_V1_COUNT - 1;
        }

        uint32_t ThreadStateCount() const
        {
            return m_sections.pThreadStates ? m_sections.pProviderBufferLayout->threadStatesCount : 0;
        }

        // The V1 revision protocol: 0 = never written, odd = a writer mid-update, and a moved revision
        // between the fenced reads = a racing write. All three deliver nothing, and so commit nothing.
        // fnOnRecord(uint32_t entryIndex, uint64_t entryRevision, const NvTrace_ThreadEvent_ThreadCreated&)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadThreadState(uint32_t threadStateIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pThreadStates || m_sections.pProviderBufferLayout->threadStatesCount <= threadStateIndex)
            {
                return;
            }

            const auto& threadState = *reinterpret_cast<const NvTrace_ThreadEvent_ThreadState_V1*>(reinterpret_cast<const uint8_t*>(m_sections.pThreadStates) + static_cast<size_t>(m_sections.pEventVersionInfo->threadStateSize) * threadStateIndex);

            const uint64_t revisionBefore = LoadAcquire(&threadState.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return;
            }

            RevisionCursor cursor{&threadState.revision, revisionBefore};

            NvTrace_ThreadEvent_ThreadCreated record = {};
            record.creationTimestampTsc = threadState.creationTimestampTsc;
            record.destructionTimestampTsc = threadState.destructionTimestampTsc;
            record.uniqueThreadId = threadState.uniqueThreadId;
            record.nativeThreadId = threadState.nativeThreadId;
            fnOnRecord(threadStateIndex, revisionBefore, record);

            fnCommitRecord(cursor.CheckConsistency());
        }

        //   bool fnReadEvent(uint32_t sourceProviderIndex, const ThreadEventRecord& record)
        //     Return true to accept the event and advance that thread's cursor.
        //     Return false to retain the event and stop reading that thread until the next ConsumeThreadEvents().
        //
        //   void fnCommitEvent(bool consistencyCheckPassed)
        //     Called once per accepted event. If false, roll back state staged by fnReadEvent.
        template <typename TFN_ReadEvent, typename TFN_CommitEvent>
        void ConsumeThreadEvents(uint64_t readEndpointTsc, TFN_ReadEvent&& fnReadEvent, TFN_CommitEvent&& fnCommitEvent)
        {
            for (EventBufferReader& eventBufferReader : m_eventBufferReaders)
            {
                eventBufferReader.ConsumeRecords(
                    readEndpointTsc,
                    [&](uint32_t recordIdentifier, uint64_t uniqueThreadId, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize) -> bool {
                        const uint32_t v1ProviderId = NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX(recordIdentifier);
                        const uint32_t recordType = NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_TYPE(recordIdentifier);

                        // No registration names INVALID or a dynamic id, so nothing could ever read the record.
                        // INVALID (0) wraps to UINT32_MAX, so one comparison rejects both.
                        const uint32_t sourceProviderIndex = v1ProviderId - 1;
                        if (MaxSourceProviderCount() <= sourceProviderIndex)
                        {
                            return true;
                        }

                        ThreadEventRecord record;
                        record.uniqueThreadId = uniqueThreadId;
                        record.correlationId = correlationId;

                        if (v1ProviderId == NvTrace_ThreadEvent_StaticProviderId_V1_ThreadEvent)
                        {
                            // The transport's own V1 lifecycle record types collide with the V2 record
                            // type values, so they are translated rather than passed through. The
                            // translated payload is tentative like the bytes it came from; the commit
                            // verdict that follows covers whatever the consumer staged from it.
                            if (payloadSize < NvTrace_ThreadEvent_ThreadLifecycle_Event_V1_MINOR_VERSION_A)
                            {
                                return true;
                            }

                            NvTrace_ThreadEvent_ThreadLifecycle_Event_V1 lifecycleEvent;
                            memcpy(&lifecycleEvent, pPayload, sizeof(lifecycleEvent));

                            NvTrace_ThreadEvent_ThreadCreated threadCreated = {};
                            NvTrace_ThreadEvent_ThreadDestroyed threadDestroyed;
                            switch (recordType)
                            {
                                case NvTrace_ThreadEvent_EventType_V1_ThreadCreated:
                                {
                                    if (std::numeric_limits<uint32_t>::max() < lifecycleEvent.threadStateIndex)
                                    {
                                        return true;
                                    }
                                    // V1 carries (threadStateIndex, revision) inline rather than a state-table
                                    // reference, so the shim resolves the thread state here and consumers only
                                    // ever see plain ThreadCreated records from a V1 buffer.
                                    bool threadStateResolved = false;
                                    ReadThreadState(
                                        static_cast<uint32_t>(lifecycleEvent.threadStateIndex),
                                        [&](uint32_t /* entryIndex */, uint64_t entryRevision, const NvTrace_ThreadEvent_ThreadCreated& threadState) {
                                            threadStateResolved = entryRevision == lifecycleEvent.threadStateRevision;
                                            threadCreated = threadState;
                                        },
                                        [&](bool consistencyCheckPassed) { threadStateResolved = threadStateResolved && consistencyCheckPassed; });
                                    if (!threadStateResolved)
                                    {
                                        return true; // The thread state was reassigned since the event; nothing to attribute.
                                    }
                                    record.recordType = NvTrace_ThreadEvent_RecordType_ThreadCreated;
                                    record.pPayload = reinterpret_cast<const uint8_t*>(&threadCreated);
                                    record.payloadSize = sizeof(threadCreated);
                                    break;
                                }
                                case NvTrace_ThreadEvent_EventType_V1_ThreadDestroyed:
                                {
                                    threadDestroyed.destructionTimestampTsc = lifecycleEvent.timestampTsc;
                                    record.recordType = NvTrace_ThreadEvent_RecordType_ThreadDestroyed;
                                    record.pPayload = reinterpret_cast<const uint8_t*>(&threadDestroyed);
                                    record.payloadSize = sizeof(threadDestroyed);
                                    break;
                                }
                                default:
                                    return true; // An unrecognised transport record; dropped, per the format's skip rule.
                            }
                            // note: this invocation can point at a ThreadEvent-specific stack variable in this scope
                            return fnReadEvent(sourceProviderIndex, record);
                        }
                        else // if (v1ProviderId == NvTrace_ThreadEvent_StaticProviderId_V1_CUDA || v1ProviderId == NvTrace_ThreadEvent_StaticProviderId_V1_NVTX)
                        {
                            record.recordType = recordType;
                            record.pPayload = pPayload;
                            record.payloadSize = payloadSize;
                            return fnReadEvent(sourceProviderIndex, record);
                        }
                    },
                    fnCommitEvent);
            }
        }

        bool SkipEventBufferEntriesBefore(uint64_t timestampTsc)
        {
            bool succeeded = true;
            for (EventBufferReader& eventBufferReader : m_eventBufferReaders)
            {
                if (!eventBufferReader.SkipCellsBeforeTimestamp(timestampTsc))
                {
                    succeeded = false;
                }
            }
            return succeeded;
        }

    private:
        /// One V1 per-thread event buffer: a cell ring naming chunk chains, records ascending from the
        /// chunk header and their payloads descending from the chunk end. Attribution is the chunk
        /// header's uniqueThreadId -- the V1 provider opens a fresh chunk on every slot rebind -- and a
        /// record's correlation id is its position in the chunk over the shipped V1 derivation.
        class EventBufferReader
        {
        public:
            ~EventBufferReader() = default;
            EventBufferReader() = default;

            EventBufferReader(const EventBufferReader&) = delete;
            EventBufferReader(EventBufferReader&& rhs) noexcept
            {
                *this = std::move(rhs);
            }

            EventBufferReader& operator=(const EventBufferReader&) = delete;
            EventBufferReader& operator=(EventBufferReader&& rhs) noexcept
            {
                if (this != &rhs)
                {
                    m_cellReader = std::move(rhs.m_cellReader);
                    m_pChunks = std::exchange(rhs.m_pChunks, nullptr);
                    m_chunkCount = std::exchange(rhs.m_chunkCount, 0);
                    m_chunkSize = std::exchange(rhs.m_chunkSize, 0);
                    m_chunkHeaderSize = std::exchange(rhs.m_chunkHeaderSize, 0);
                    m_recordSize = std::exchange(rhs.m_recordSize, 0);
                    m_maxRecordCountInChunk = std::exchange(rhs.m_maxRecordCountInChunk, 0);
                    m_chunkIndexShiftCount = std::exchange(rhs.m_chunkIndexShiftCount, 0);
                    m_chunkRevisionShiftCount = std::exchange(rhs.m_chunkRevisionShiftCount, 0);
                    m_currentCellNextChunkRef = std::exchange(rhs.m_currentCellNextChunkRef, NvTrace_ChunkRef_V1{NVTRACE_INVALID_CHUNK_REF_VALUE_V1});
                    m_currentChunkRef = std::exchange(rhs.m_currentChunkRef, NvTrace_ChunkRef_V1{NVTRACE_INVALID_CHUNK_REF_VALUE_V1});
                    m_pCurrentChunkHeader = std::exchange(rhs.m_pCurrentChunkHeader, nullptr);
                    m_currentChunkRecordCount = std::exchange(rhs.m_currentChunkRecordCount, 0);
                    m_pNextRecordReadPos = std::exchange(rhs.m_pNextRecordReadPos, nullptr);
                    m_pEndRecordReadPos = std::exchange(rhs.m_pEndRecordReadPos, nullptr);
                    m_pNextPayloadReadPos = std::exchange(rhs.m_pNextPayloadReadPos, nullptr);
                    m_uniqueThreadId = std::exchange(rhs.m_uniqueThreadId, 0);
                    m_nextCorrelationId = std::exchange(rhs.m_nextCorrelationId, 0);
                }
                return *this;
            }

            void Initialize(const NvTrace_AdaptiveBufferState_V1* pState, const NvTrace_AdaptiveBufferCell_V1* pCells, uint32_t cellCount, size_t cellSize, size_t chunkHeaderSize, size_t recordSize, const uint8_t* pChunks, uint32_t chunkCount, size_t chunkSize)
            {
                m_cellReader.Initialize(&pState->circularBufferState, CellBufferAccessor{pCells, cellSize}, cellCount);
                m_pChunks = pChunks;
                m_chunkCount = chunkCount;
                m_chunkSize = chunkSize;
                m_chunkHeaderSize = chunkHeaderSize;
                m_recordSize = recordSize;

                m_maxRecordCountInChunk = static_cast<uint32_t>((chunkSize - chunkHeaderSize) / recordSize);
                m_chunkIndexShiftCount = CeilLog2(m_maxRecordCountInChunk);
                m_chunkRevisionShiftCount = m_chunkIndexShiftCount + CeilLog2(chunkCount);
            }

            bool SkipCellsBeforeTimestamp(uint64_t timestampTsc)
            {
                return m_cellReader.SkipRecordsBeforeTimestamp(timestampTsc);
            }

            //   void fnReadRecord(uint32_t recordIdentifier, uint64_t uniqueThreadId, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize)
            //     Accepts every delivered record.
            //
            //   bool fnReadRecord(uint32_t recordIdentifier, uint64_t uniqueThreadId, uint64_t correlationId, const uint8_t* pPayload, size_t payloadSize)
            //     Return true to accept the record and advance the cursor.
            //     Return false to retain the record and stop until the next ConsumeRecords().
            //
            //   void fnCommitRecord(bool consistencyCheckPassed)
            //     Called once per accepted record. If false, roll back state staged by fnReadRecord.
            template <typename TFN_ReadRecord, typename TFN_CommitRecord>
            void ConsumeRecords(uint64_t readEndpointTsc, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
            {
                const uint64_t cellReadEndSequenceNumber = m_cellReader.AcquireCurrentEndSequenceNumber();
                while (true)
                {
                    if (IsCurrentChunkPartiallyWritten())
                    {
                        RefreshCurrentChunkRecordCount();
                        if (m_currentChunkRecordCount != 0)
                        {
                            m_pEndRecordReadPos = GetCurrentChunkPtr() + m_chunkHeaderSize + CurrentChunkRecordCount() * m_recordSize;
                        }
                        else
                        {
                            m_pNextRecordReadPos = nullptr;
                            m_pEndRecordReadPos = nullptr;
                        }
                    }

                    while (m_pNextRecordReadPos < m_pEndRecordReadPos)
                    {
                        const auto* const pRecord = reinterpret_cast<const NvTrace_ThreadEvent_EventBufferRecord_V1*>(m_pNextRecordReadPos);
                        const size_t payloadSize = pRecord->payloadSize;
                        const size_t alignedPayloadSize = AlignUpP2(payloadSize, alignof(uint64_t));

                        // Signed on purpose: a corrupt chunk can put the record cursor past the payload
                        // cursor, and an unsigned difference would wrap and admit any payload size.
                        if (m_pNextPayloadReadPos - m_pEndRecordReadPos < static_cast<ptrdiff_t>(alignedPayloadSize))
                        {
                            NVTRACE_LOG_WRN(50, "V1 chunk corruption detected; this implies a chunk overrun.");
                            m_pNextRecordReadPos = m_pEndRecordReadPos;
                            break;
                        }

                        bool recordAccepted = true;
                        if constexpr (std::is_same_v<std::invoke_result_t<TFN_ReadRecord, uint32_t, uint64_t, uint64_t, const uint8_t*, size_t>, void>)
                        {
                            fnReadRecord(pRecord->recordIdentifier, m_uniqueThreadId, m_nextCorrelationId, m_pNextPayloadReadPos - alignedPayloadSize, payloadSize);
                        }
                        else
                        {
                            recordAccepted = static_cast<bool>(fnReadRecord(pRecord->recordIdentifier, m_uniqueThreadId, m_nextCorrelationId, m_pNextPayloadReadPos - alignedPayloadSize, payloadSize));
                        }

                        // Ensure fnReadRecord()'s loads complete before the chunk revision is re-read below.
                        ReadFence();
                        const bool consistencyCheckPassed = IsCurrentChunkConsistent();
                        if (!consistencyCheckPassed)
                        {
                            if (recordAccepted)
                            {
                                fnCommitRecord(false);
                            }
                            NVTRACE_LOG_WRN(50, "V1 chunk overrun detected.");
                            m_pNextRecordReadPos = m_pEndRecordReadPos;
                            break;
                        }
                        if (!recordAccepted)
                        {
                            return;
                        }

                        fnCommitRecord(true);
                        m_pNextRecordReadPos += m_recordSize;
                        m_pNextPayloadReadPos -= alignedPayloadSize;
                        m_nextCorrelationId += 1;
                    }

                    if (IsCurrentChunkPartiallyWritten() && m_pEndRecordReadPos)
                    {
                        break; // Stay on this chunk until it has been finalized.
                    }

                    if (!ConsumeNextChunk(cellReadEndSequenceNumber, readEndpointTsc))
                    {
                        break;
                    }

                    const uint8_t* const pCurrentChunk = GetCurrentChunkPtr();
                    m_pNextRecordReadPos = pCurrentChunk + m_chunkHeaderSize;
                    m_pEndRecordReadPos = pCurrentChunk + m_chunkHeaderSize + CurrentChunkRecordCount() * m_recordSize;
                    m_pNextPayloadReadPos = pCurrentChunk + m_chunkSize;

                    m_uniqueThreadId = reinterpret_cast<const NvTrace_ThreadEvent_EventBufferChunkHeader_V1*>(pCurrentChunk)->uniqueThreadId;
                    m_nextCorrelationId = (static_cast<uint64_t>(m_currentChunkRef.chunkIndex) << m_chunkIndexShiftCount) | (static_cast<uint64_t>(m_currentChunkRef.chunkRevision) << m_chunkRevisionShiftCount);
                }
            }

        private:
            struct CellBufferAccessor
            {
                const NvTrace_AdaptiveBufferCell_V1* pCells = nullptr;
                size_t cellSize = 0;

                uint64_t ReadRecordTimestamp(uint32_t readCellIndex) const
                {
                    return CellAt(readCellIndex).readableAfterTsc;
                }

                void ReadRecord(uint32_t readCellIndex, uint64_t /* readSequenceNumber */, NvTrace_AdaptiveBufferCell_V1& out_cell) const
                {
                    out_cell = CellAt(readCellIndex);
                }

                const NvTrace_AdaptiveBufferCell_V1& CellAt(uint32_t cellIndex) const
                {
                    return *reinterpret_cast<const NvTrace_AdaptiveBufferCell_V1*>(reinterpret_cast<const uint8_t*>(pCells) + cellIndex * cellSize);
                }
            };

            const uint8_t* GetCurrentChunkPtr() const
            {
                return reinterpret_cast<const uint8_t*>(m_pCurrentChunkHeader);
            }

            size_t CurrentChunkRecordCount() const
            {
                // While a chunk is still held by a thread, recordCount is the negated true count.
                return static_cast<size_t>(0 < m_currentChunkRecordCount ? m_currentChunkRecordCount : -m_currentChunkRecordCount);
            }

            bool IsCurrentChunkPartiallyWritten() const
            {
                return m_currentChunkRecordCount <= 0;
            }

            bool IsCurrentChunkConsistent() const
            {
                return m_pCurrentChunkHeader->chunkHeader.chunkRevision == m_currentChunkRef.chunkRevision;
            }

            // A count past the chunk's capacity is corruption the revision check cannot catch;
            // clamping bounds the record walk, whose per-record payload guard then abandons the chunk.
            int32_t ClampChunkRecordCount(int32_t chunkRecordCount) const
            {
                const int32_t maxRecordCountInChunk = static_cast<int32_t>(m_maxRecordCountInChunk);
                return std::clamp(chunkRecordCount, -maxRecordCountInChunk, maxRecordCountInChunk);
            }

            void RefreshCurrentChunkRecordCount()
            {
                if (!m_pCurrentChunkHeader)
                {
                    return;
                }

                const int32_t chunkRecordCount = LoadAcquire(&m_pCurrentChunkHeader->recordCount);
                m_currentChunkRecordCount = IsCurrentChunkConsistent() ? ClampChunkRecordCount(chunkRecordCount) : 0;
            }

            bool ConsumeNextChunk(uint64_t cellReadEndSequenceNumber, uint64_t readEndpointTsc)
            {
                while (true)
                {
                    if (m_currentCellNextChunkRef.chunkIndex == NVTRACE_INVALID_CHUNK_INDEX_V1)
                    {
                        NvTrace_AdaptiveBufferCell_V1 cell;
                        if (!m_cellReader.ConsumeOneRecord(cellReadEndSequenceNumber, readEndpointTsc, cell))
                        {
                            return false;
                        }

                        m_currentCellNextChunkRef = cell.firstChunkRef;
                        continue;
                    }

                    if (m_chunkCount <= m_currentCellNextChunkRef.chunkIndex)
                    {
                        NVTRACE_LOG_WRN(50, "V1 chunk index %u is out of bounds (chunkCount=%u).", m_currentCellNextChunkRef.chunkIndex, m_chunkCount);
                        m_currentCellNextChunkRef.value = NVTRACE_INVALID_CHUNK_REF_VALUE_V1;
                        continue;
                    }

                    const auto* const pNextChunkHeader = reinterpret_cast<const NvTrace_AdaptiveBufferChunkHeader_V1*>(m_pChunks + m_currentCellNextChunkRef.chunkIndex * m_chunkSize);
                    const int32_t nextChunkRecordCount = pNextChunkHeader->recordCount;
                    const NvTrace_ChunkRef_V1 nextChunkNextChunkRef = {pNextChunkHeader->nextChunkRef.value};
                    ReadFence();

                    if (pNextChunkHeader->chunkHeader.chunkRevision != m_currentCellNextChunkRef.chunkRevision)
                    {
                        NVTRACE_LOG_WRN(50, "V1 chunk overrun detected while chasing a chunk chain.");
                        m_currentCellNextChunkRef.value = NVTRACE_INVALID_CHUNK_REF_VALUE_V1;
                        continue;
                    }

                    m_currentChunkRef = m_currentCellNextChunkRef;
                    m_pCurrentChunkHeader = pNextChunkHeader;
                    m_currentChunkRecordCount = ClampChunkRecordCount(nextChunkRecordCount);
                    m_currentCellNextChunkRef.value = nextChunkNextChunkRef.value;
                    return true;
                }
            }

        private:
            CircularBufferReaderV1<CellBufferAccessor> m_cellReader;
            const uint8_t* m_pChunks = nullptr;
            uint32_t m_chunkCount = 0;
            size_t m_chunkSize = 0;
            size_t m_chunkHeaderSize = 0;
            size_t m_recordSize = 0;
            uint32_t m_maxRecordCountInChunk = 0;
            uint32_t m_chunkIndexShiftCount = 0;
            uint32_t m_chunkRevisionShiftCount = 0;

            NvTrace_ChunkRef_V1 m_currentCellNextChunkRef = {NVTRACE_INVALID_CHUNK_REF_VALUE_V1};
            NvTrace_ChunkRef_V1 m_currentChunkRef = {NVTRACE_INVALID_CHUNK_REF_VALUE_V1};
            const NvTrace_AdaptiveBufferChunkHeader_V1* m_pCurrentChunkHeader = nullptr;
            int32_t m_currentChunkRecordCount = 0;

            const uint8_t* m_pNextRecordReadPos = nullptr;
            const uint8_t* m_pEndRecordReadPos = nullptr;
            const uint8_t* m_pNextPayloadReadPos = nullptr;
            uint64_t m_uniqueThreadId = 0;
            uint64_t m_nextCorrelationId = 0;
        };

        static NVTRACE_UUID StaticProviderTypeUuid(uint32_t v1ProviderId)
        {
            switch (v1ProviderId)
            {
                case NvTrace_ThreadEvent_StaticProviderId_V1_ThreadEvent:
                    return NvTrace_ThreadEvent_ProviderTypeUuid;
                case NvTrace_ThreadEvent_StaticProviderId_V1_CUDA:
                    return NvTrace_CUDA_ApiTrace_ProviderTypeUuid;
                case NvTrace_ThreadEvent_StaticProviderId_V1_NVTX:
                    return NvTrace_NvtxTrace_ProviderTypeUuid;
                default:
                    return NVTRACE_NIL_UUID;
            }
        }

        const NvTrace_ThreadEvent_ProviderIdMapping_V1& MappingAt(uint32_t mappingIndex) const
        {
            return *reinterpret_cast<const NvTrace_ThreadEvent_ProviderIdMapping_V1*>(reinterpret_cast<const uint8_t*>(m_sections.pProviderIdMappings) + static_cast<size_t>(m_sections.pEventVersionInfo->providerIdMappingSize) * mappingIndex);
        }

        // A published slot names both a provider type and the provider it maps to; anything less is a
        // half-published entry the revision check cannot catch.
        bool IsProviderIdMappingPublished(uint32_t mappingIndex) const
        {
            if (!m_sections.pProviderIdMappings || m_sections.pProviderBufferLayout->providerIdMappingsCount <= mappingIndex)
            {
                return false;
            }

            const NvTrace_ThreadEvent_ProviderIdMapping_V1& mapping = MappingAt(mappingIndex);

            const uint64_t revisionBefore = LoadAcquire(&mapping.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return false;
            }

            const NVTRACE_UUID providerTypeUuid = mapping.providerTypeUuid;
            const NVTRACE_LUID mappedProviderLuid = mapping.mappedProviderLuid;

            ReadFence();
            if (mapping.revision != revisionBefore)
            {
                return false;
            }

            return providerTypeUuid != NVTRACE_NIL_UUID && mappedProviderLuid != NVTRACE_NIL_LUID;
        }

    private:
        struct ProviderBufferSections
        {
            const NvTrace_ProviderBufferHeader_V1* pProviderBufferHeader = nullptr;
            const NvTrace_ThreadEvent_ProviderBufferLayout_V1* pProviderBufferLayout = nullptr;
            const NvTrace_ThreadEvent_EventVersionInfo_V1* pEventVersionInfo = nullptr;
            const NvTrace_ThreadEvent_ProviderIdMapping_V1* pProviderIdMappings = nullptr;
            const NvTrace_ThreadEvent_ThreadState_V1* pThreadStates = nullptr;
        };

        ProviderBufferSections m_sections;
        std::vector<EventBufferReader> m_eventBufferReaders;

        // The exactly-once delivery cursor of ConsumeSourceProviderRegistrations, one bit per V1 static id
        uint32_t m_deliveredStaticMappingsMask = 0;
    };

} // namespace nv::trace
