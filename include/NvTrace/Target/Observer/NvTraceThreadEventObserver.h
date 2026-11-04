/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/DataFormats/nvtrace_thread_event_v1.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceGenericObserver.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventObserverV1.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    /// \brief Thread Event Observer
    ///
    /// Typed wrapper over GenericObserver. Each thread slot's event buffer opens every cell with a
    /// ThreadChanged record, which this observer consumes to attribute the records that follow it;
    /// the thread-states table is drained at attach and resolved by ref thereafter, both delivering
    /// records tentatively, each closed by its own commit, so a consumer never keeps a record
    /// a racing writer disturbed. Every event record names its source provider by index, and the
    /// source provider registrations say which provider type each index belongs to: each
    /// registration is published once into a never-wrapping circular buffer and delivered exactly
    /// once.
    ///
    /// Initialize dispatches on the Router-provided provider type UUID: the V1 id names an
    /// old-driver buffer, served through the shim (ThreadEventObserverV1), which attributes from
    /// the V1 per-chunk uniqueThreadId, translates the transport's own lifecycle records and maps
    /// V1 provider ids and provider type UUIDs to current ones, so consumers see one format either
    /// way.
    ///
    /// Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class ThreadEventObserver
    {
    public:
        ~ThreadEventObserver() = default;
        ThreadEventObserver() = default;

        ThreadEventObserver(const ThreadEventObserver&) = delete;
        ThreadEventObserver(ThreadEventObserver&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        ThreadEventObserver& operator=(const ThreadEventObserver&) = delete;
        ThreadEventObserver& operator=(ThreadEventObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_genericObserver = std::move(rhs.m_genericObserver);
                m_pSourceProviderRegistrations = std::exchange(rhs.m_pSourceProviderRegistrations, nullptr);
                m_stagedSourceProviderRegistrations = std::move(rhs.m_stagedSourceProviderRegistrations);
                m_maxStaticSourceProviderIndex = std::exchange(rhs.m_maxStaticSourceProviderIndex, 0u);
                m_maxSourceProviderCount = std::exchange(rhs.m_maxSourceProviderCount, 0u);
                m_cellWriteDurationNsec = std::exchange(rhs.m_cellWriteDurationNsec, 0u);
                m_threadStatesSectionIndex = std::exchange(rhs.m_threadStatesSectionIndex, NVTRACE_INVALID_SECTION_INDEX);
                m_pThreadStates = std::exchange(rhs.m_pThreadStates, &s_emptyStateTable);
                m_eventBuffers = std::move(rhs.m_eventBuffers);
                m_v1Observer = std::move(rhs.m_v1Observer);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_v1Observer.Reset();
            m_eventBuffers.clear();
            m_pThreadStates = &s_emptyStateTable;
            m_threadStatesSectionIndex = NVTRACE_INVALID_SECTION_INDEX;
            m_cellWriteDurationNsec = 0;
            m_maxSourceProviderCount = 0;
            m_maxStaticSourceProviderIndex = 0;
            m_stagedSourceProviderRegistrations.clear();
            m_pSourceProviderRegistrations = nullptr;
            m_genericObserver.Reset();
        }

        // providerTypeUuid arrives out-of-band from the Router, which took it from the provider's
        // advertisement: the V1 id means an old driver serving a V1 buffer. The generic path
        // re-reads the id from the buffer and requires it to match.
        bool Initialize(const NVTRACE_UUID& providerTypeUuid, const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            if (providerTypeUuid == NvTrace_ThreadEvent_ProviderTypeUuid_V1)
            {
                return m_v1Observer.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            }

            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "ThreadEventObserver::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            const bool genericObserverInitialized = m_genericObserver.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            if (!genericObserverInitialized)
            {
                return false;
            }

            const NVTRACE_UUID bufferProviderTypeUuid = m_genericObserver.ProviderTypeUuid();
            NVTRACE_LOG_ERR_RETURN_IF(bufferProviderTypeUuid != providerTypeUuid, false, 20, "Provider type UUID mismatch for ThreadEvent observer (buffer %s, router %s).", ToString(bufferProviderTypeUuid).c_str(), ToString(providerTypeUuid).c_str());

            // Nothing here serves it, but a buffer missing it is not a complete ThreadEvent buffer.
            const DataBlobReader* const pConfigurationReader = FindDataBlobReader(NvTrace_ThreadEvent_SectionTag_Configuration);
            NVTRACE_LOG_ERR_RETURN_IF(!pConfigurationReader, false, 20, "ThreadEvent configuration section not found or invalid.");

            DataBlobReader* const pRecordVersionInfoReader = FindDataBlobReader(NvTrace_ThreadEvent_SectionTag_RecordVersionInfo);
            NVTRACE_LOG_ERR_RETURN_IF(!pRecordVersionInfoReader, false, 20, "ThreadEvent record version info section not found or invalid.");

            NvTrace_ThreadEvent_RecordVersionInfo recordVersionInfo = {};
            size_t recordVersionInfoPayloadSize = 0;
            pRecordVersionInfoReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                recordVersionInfoPayloadSize = payloadSize;
                if (sizeof(recordVersionInfo) <= payloadSize)
                {
                    recordVersionInfo = *reinterpret_cast<const NvTrace_ThreadEvent_RecordVersionInfo*>(pPayload);
                }
            });
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfoPayloadSize < NvTrace_ThreadEvent_RecordVersionInfo_MINOR_VERSION_A, false, 20, "Record version info is incompatible for ThreadEvent (publishedSize=%zu expectedMinorVersion=%u).", recordVersionInfoPayloadSize, NvTrace_ThreadEvent_RecordVersionInfo_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.recordVersionInfoMinorVersion < NvTrace_ThreadEvent_RecordVersionInfo_MINOR_VERSION_A, false, 20, "Record version info describes itself below the consumer's floor (recordVersionInfoMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.recordVersionInfoMinorVersion, NvTrace_ThreadEvent_RecordVersionInfo_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.sourceProviderRegisteredMinorVersion < NvTrace_ThreadEvent_SourceProviderRegistered_MINOR_VERSION_A, false, 20, "Source provider registered record layout is incompatible for ThreadEvent (sourceProviderRegisteredMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.sourceProviderRegisteredMinorVersion, NvTrace_ThreadEvent_SourceProviderRegistered_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.threadChangedMinorVersion < NvTrace_ThreadEvent_ThreadChanged_MINOR_VERSION_A, false, 20, "Thread changed record layout is incompatible for ThreadEvent (threadChangedMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.threadChangedMinorVersion, NvTrace_ThreadEvent_ThreadChanged_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.threadCreatedMinorVersion < NvTrace_ThreadEvent_ThreadCreated_MINOR_VERSION_A, false, 20, "Thread state record layout is incompatible for ThreadEvent (threadCreatedMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.threadCreatedMinorVersion, NvTrace_ThreadEvent_ThreadCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.threadDestroyedMinorVersion < NvTrace_ThreadEvent_ThreadDestroyed_MINOR_VERSION_A, false, 20, "Thread destroyed record layout is incompatible for ThreadEvent (threadDestroyedMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.threadDestroyedMinorVersion, NvTrace_ThreadEvent_ThreadDestroyed_MINOR_VERSION_A);

            DataBlobReader* const pProviderPropertiesReader = FindDataBlobReader(NvTrace_ThreadEvent_SectionTag_ProviderProperties);
            NVTRACE_LOG_ERR_RETURN_IF(!pProviderPropertiesReader, false, 20, "ThreadEvent provider properties section not found or invalid.");

            NvTrace_ThreadEvent_ProviderProperties providerProperties = {};
            size_t providerPropertiesPayloadSize = 0;
            pProviderPropertiesReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                providerPropertiesPayloadSize = payloadSize;
                if (sizeof(providerProperties) <= payloadSize)
                {
                    providerProperties = *reinterpret_cast<const NvTrace_ThreadEvent_ProviderProperties*>(pPayload);
                }
            });
            NVTRACE_LOG_ERR_RETURN_IF(providerPropertiesPayloadSize < NvTrace_ThreadEvent_ProviderProperties_MINOR_VERSION_A, false, 20, "Provider properties are incompatible for ThreadEvent (publishedSize=%zu expectedMinorVersion=%u).", providerPropertiesPayloadSize, NvTrace_ThreadEvent_ProviderProperties_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(providerProperties.maxSourceProviderCount < NvTrace_ThreadEvent_StaticSourceProviderIndex_COUNT, false, 20, "ThreadEvent source provider count %u cannot hold the static source provider indices (staticSourceProviderIndexCount=%u).", providerProperties.maxSourceProviderCount, static_cast<uint32_t>(NvTrace_ThreadEvent_StaticSourceProviderIndex_COUNT));
            const uint32_t sourceProviderIndexFieldCapacity = NVTRACE_FIELD_MASK32_UNSHIFTED(NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX_FIELD) + 1;
            NVTRACE_LOG_ERR_RETURN_IF(sourceProviderIndexFieldCapacity < providerProperties.maxSourceProviderCount, false, 20, "ThreadEvent source provider count %u exceeds what the record identifier's source provider index field can name (%u).", providerProperties.maxSourceProviderCount, sourceProviderIndexFieldCapacity);

            const GenericObserver::Section* const pSourceProviderRegistrationsSection = m_genericObserver.FindSection(NvTrace_ThreadEvent_SectionTag_SourceProviderRegistrations);
            NVTRACE_LOG_ERR_RETURN_IF(!pSourceProviderRegistrationsSection, false, 20, "ThreadEvent source provider registrations section not found.");

            CircularBufferReader* const pSourceProviderRegistrations = pSourceProviderRegistrationsSection->AsCircularBuffer();
            NVTRACE_LOG_ERR_RETURN_IF(!pSourceProviderRegistrations, false, 20, "ThreadEvent source provider registrations section is reader-less.");

            const GenericObserver::Section* const pThreadStatesSection = m_genericObserver.FindSection(NvTrace_ThreadEvent_SectionTag_ThreadStates);
            NVTRACE_LOG_ERR_RETURN_IF(!pThreadStatesSection, false, 20, "ThreadEvent state table section not found (sectionTag=%u).", static_cast<uint32_t>(NvTrace_ThreadEvent_SectionTag_ThreadStates));

            const StateTableReader* const pThreadStates = pThreadStatesSection->AsStateTable();
            NVTRACE_LOG_ERR_RETURN_IF(!pThreadStates, false, 20, "ThreadEvent state table section is reader-less (sectionTag=%u).", static_cast<uint32_t>(NvTrace_ThreadEvent_SectionTag_ThreadStates));

            std::vector<EventBuffer> eventBuffers = CollectEventBuffers();

            // Commit observer state only now, after all validation has succeeded.
            m_pSourceProviderRegistrations = pSourceProviderRegistrations;
            m_maxStaticSourceProviderIndex = providerProperties.maxStaticSourceProviderIndex;
            m_maxSourceProviderCount = providerProperties.maxSourceProviderCount;
            m_cellWriteDurationNsec = providerProperties.cellWriteDurationNsec;
            m_threadStatesSectionIndex = pThreadStatesSection->SectionIndex();
            m_pThreadStates = pThreadStates;
            m_eventBuffers = std::move(eventBuffers);

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_genericObserver.IsInitialized() || m_v1Observer.IsInitialized();
        }

        // Drains newly published source provider registrations, delivering each exactly once across
        // calls as fnOnRegistered(uint32_t sourceProviderIndex, const NVTRACE_UUID& providerTypeUuid), in
        // registration order. The provider type UUID is always a current one, never a V1 alias. Records are
        // staged per batch and forwarded only after the batch's consistency check passes, since acting on a
        // registration is side-effectful.
        template <typename TFN_OnRegistered>
        void ConsumeSourceProviderRegistrations(uint64_t readEndpointTsc, TFN_OnRegistered&& fnOnRegistered)
        {
            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ConsumeSourceProviderRegistrations(std::forward<TFN_OnRegistered>(fnOnRegistered));
                return;
            }

            if (!m_pSourceProviderRegistrations)
            {
                return;
            }

            m_stagedSourceProviderRegistrations.clear();
            m_pSourceProviderRegistrations->ConsumeRecords(
                readEndpointTsc,
                [&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                    if (payloadSize < NvTrace_ThreadEvent_SourceProviderRegistered_MINOR_VERSION_A)
                    {
                        return;
                    }

                    // Copy the published prefix out of the concurrently-writable mapped buffer
                    // before interpreting; fields past it stay zero.
                    NvTrace_ThreadEvent_SourceProviderRegistered registration = {};
                    memcpy(&registration, pPayload, std::min(sizeof(registration), payloadSize));
                    m_stagedSourceProviderRegistrations.push_back(registration);
                },
                [&](bool consistencyCheckPassed) {
                    if (!consistencyCheckPassed)
                    {
                        NVTRACE_LOG_WRN(50, "A source provider registration batch failed its consistency check; the producer overran a buffer sized never to wrap.");
                        m_stagedSourceProviderRegistrations.clear();
                        return;
                    }

                    for (const NvTrace_ThreadEvent_SourceProviderRegistered& registration : m_stagedSourceProviderRegistrations)
                    {
                        fnOnRegistered(registration.sourceProviderIndex, registration.providerTypeUuid);
                    }
                    m_stagedSourceProviderRegistrations.clear();
                });
        }

        // The last index the producer reserves for a static provider type; indices above it are dynamic.
        uint32_t MaxStaticSourceProviderIndex() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.MaxStaticSourceProviderIndex() : m_maxStaticSourceProviderIndex;
        }

        // Every source provider index, static or dynamic, is below it; sizes per-index consumer state.
        uint32_t MaxSourceProviderCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.MaxSourceProviderCount() : m_maxSourceProviderCount;
        }

        // Every event buffer has the same cell count, so the first answers for all. 0 when unknown,
        // as for a V1 buffer, which publishes no cell write duration.
        uint64_t GetEventBufferDurationNsec() const
        {
            if (m_eventBuffers.empty())
            {
                return 0;
            }

            // The duration is the producer's to publish: saturate rather than wrap to a short one.
            const uint64_t cellCount = m_eventBuffers.front().pReader->CellCount();
            if (cellCount != 0 && UINT64_MAX / cellCount < m_cellWriteDurationNsec)
            {
                return UINT64_MAX;
            }

            return cellCount * m_cellWriteDurationNsec;
        }

        uint32_t ThreadStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.ThreadStateCount() : m_pThreadStates->EntryCount();
        }

        // Drains the thread-state table, delivering every published entry tentatively: fnOnRecord
        // stages a record and the fnCommitRecord that follows says whether to keep it.
        // fnOnRecord(uint32_t entryIndex, uint64_t entryRevision, const NvTrace_ThreadEvent_ThreadCreated& record)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachThreadStates(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t threadStateIndex = 0; threadStateIndex < m_v1Observer.ThreadStateCount(); ++threadStateIndex)
                {
                    m_v1Observer.ReadThreadState(threadStateIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }

            const uint32_t expectedRecordType = NVTRACE_COMMON_THREAD_EVENT_RECORD_IDENTIFIER(NvTrace_ThreadEvent_RecordType_ThreadCreated);

            m_pThreadStates->ConsumeAttachRecords<false>(
                [&](uint64_t entryRef, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordType != expectedRecordType)
                    {
                        return;
                    }

                    // The shared prefix only. A producer built against an older record publishes a
                    // shorter one, and the fields past it stay zero; which of them a consumer may
                    // trust is what the record minor version, gated at Initialize, says.
                    NvTrace_ThreadEvent_ThreadCreated record = {};
                    Memcpy64(&record, pPayload, std::min(sizeof(record), AlignDnP2(payloadSize, sizeof(uint64_t))));

                    const auto [entryIndex, entryRevision] = m_pThreadStates->DecodeEntryRef(entryRef);
                    fnOnRecord(entryIndex, entryRevision, record);
                },
                fnCommitRecord);
        }

        // Every event buffer shares the one chunk pool, so the first reader answers for all.
        size_t GetMaxRecordPayloadSize() const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.GetMaxRecordPayloadSize();
            }

            if (m_eventBuffers.empty())
            {
                return 0;
            }
            return m_eventBuffers.front().pReader->MaxRecordPayloadSize();
        }

        // A resolve compares in ref space inside the reader, so a stale reference -- the entry
        // modified or reassigned since the event -- resolves to nothing. A V1 buffer never carries
        // reference records: the shim resolves thread states inline and synthesizes plain
        // ThreadCreated records, so a reference reaching it is malformed input.
        // fnOnThreadCreated(const NvTrace_ThreadEvent_ThreadCreated& record)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnThreadCreated, typename TFN_CommitRecord>
        void ResolveThreadStateReference(const NvTrace_RecordReference& reference, TFN_OnThreadCreated&& fnOnThreadCreated, TFN_CommitRecord&& fnCommitRecord)
        {
            const bool isV1Shimmed = m_v1Observer.IsInitialized();
            NVTRACE_LOG_ERR_RETURN_IF(isV1Shimmed, , 20, "A record reference reached a V1-shimmed observer (recordRef=%" PRIu64 ").", reference.recordRef);
            NVTRACE_LOG_ERR_RETURN_IF(reference.sectionIndex != m_threadStatesSectionIndex, , 20, "A record reference names a section this observer does not resolve (sectionIndex=%u, threadStatesSectionIndex=%u).", reference.sectionIndex, m_threadStatesSectionIndex);

            ReadStateTableRecordByRef<NvTrace_ThreadEvent_ThreadCreated>(
                *m_pThreadStates,
                reference.recordRef,
                NVTRACE_COMMON_THREAD_EVENT_RECORD_IDENTIFIER(NvTrace_ThreadEvent_RecordType_ThreadCreated),
                std::forward<TFN_OnThreadCreated>(fnOnThreadCreated),
                std::forward<TFN_CommitRecord>(fnCommitRecord));
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
            bool recordAccepted = false;
            const auto fnCommitAcceptedEvent = [&](bool consistencyCheckPassed) {
                if (std::exchange(recordAccepted, false))
                {
                    fnCommitEvent(consistencyCheckPassed);
                }
            };
            const auto fnForwardEvent = [&](uint32_t sourceProviderIndex, const ThreadEventRecord& record) {
                recordAccepted = fnReadEvent(sourceProviderIndex, record);
                return recordAccepted;
            };

            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ConsumeThreadEvents(
                    readEndpointTsc,
                    fnForwardEvent,
                    fnCommitAcceptedEvent);
                return;
            }

            for (EventBuffer& eventBuffer : m_eventBuffers)
            {
                eventBuffer.pReader->ConsumeRecords(
                    readEndpointTsc,
                    [&](uint32_t recordIdentifier, const uint8_t* pPayload, size_t payloadSize) -> bool {
                        const uint32_t sourceProviderIndex = NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX(recordIdentifier);
                        const uint32_t recordType = NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_TYPE(recordIdentifier);

                        // Transport metadata, not a trace event: it re-establishes attribution for the
                        // records after it and is never forwarded.
                        if (sourceProviderIndex == NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent && recordType == NvTrace_ThreadEvent_RecordType_ThreadChanged)
                        {
                            NvTrace_ThreadEvent_ThreadChanged threadChanged = {};
                            if (NvTrace_ThreadEvent_ThreadChanged_MINOR_VERSION_A <= payloadSize)
                            {
                                memcpy(&threadChanged, pPayload, std::min(sizeof(threadChanged), payloadSize));
                            }
                            eventBuffer.uniqueThreadId = threadChanged.uniqueThreadId;
                            return true;
                        }

                        ThreadEventRecord record;
                        record.recordType = recordType;
                        record.uniqueThreadId = eventBuffer.uniqueThreadId;
                        record.correlationId = eventBuffer.pReader->CurrentRecordCorrelationId();
                        record.pPayload = pPayload;
                        record.payloadSize = payloadSize;
                        return fnForwardEvent(sourceProviderIndex, record);
                    },
                    fnCommitAcceptedEvent);
            }
        }

        bool SkipEventBufferEntriesBefore(uint64_t timestampTsc)
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.SkipEventBufferEntriesBefore(timestampTsc);
            }

            bool succeeded = true;
            for (EventBuffer& eventBuffer : m_eventBuffers)
            {
                if (!eventBuffer.pReader->SkipCellsBeforeTimestamp(timestampTsc))
                {
                    succeeded = false;
                }
            }
            return succeeded;
        }

    private:
        // One thread slot's event buffer, plus the thread its most recent ThreadChanged named. The id
        // is per slot and survives across polls, since a cell is not re-read from its start on resume.
        struct EventBuffer
        {
            AdaptiveBufferReader* pReader = nullptr;
            uint64_t uniqueThreadId = 0;
        };

        DataBlobReader* FindDataBlobReader(uint32_t sectionTag)
        {
            const GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            return pSection ? pSection->AsDataBlob() : nullptr;
        }

        // Names the entry by an NvTrace_RecordReference's recordRef, so a stale ref -- the entry
        // modified or reassigned since -- delivers nothing, rather than the slot's current occupant.
        // fnOnRecord(const TRecord& record)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TRecord, typename TFN_OnRecord, typename TFN_CommitRecord>
        static void ReadStateTableRecordByRef(const StateTableReader& stateTableReader, uint64_t recordRef, uint32_t expectedRecordType, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            static_assert((sizeof(TRecord) % sizeof(uint64_t)) == 0, "A state-table record must be a whole number of 64-bit words, since Memcpy64 copies whole words only");

            // The commit follows the read even when the record's type made it undeliverable.
            bool recordDelivered = false;
            stateTableReader.ReadEntry<false>(
                recordRef,
                [&](uint64_t /* entryRef */, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordType != expectedRecordType)
                    {
                        return;
                    }

                    // The shared prefix only. A producer built against an older record publishes a
                    // shorter one, and the fields past it stay zero; which of them a consumer may
                    // trust is what the record minor version, gated at Initialize, says.
                    TRecord record = {};
                    Memcpy64(&record, pPayload, std::min(sizeof(record), AlignDnP2(payloadSize, sizeof(uint64_t))));

                    recordDelivered = true;
                    fnOnRecord(record);
                },
                [&](bool consistencyCheckPassed) {
                    if (recordDelivered)
                    {
                        fnCommitRecord(consistencyCheckPassed);
                    }
                });
        }

        std::vector<EventBuffer> CollectEventBuffers()
        {
            std::vector<EventBuffer> eventBuffers;
            for (const GenericObserver::Section* const pSection : m_genericObserver.RecordSections())
            {
                if (pSection->SectionTag() != NvTrace_ThreadEvent_SectionTag_EventBuffer)
                {
                    continue;
                }

                if (AdaptiveBufferReader* const pReader = pSection->AsAdaptiveBuffer())
                {
                    eventBuffers.push_back({pReader, 0});
                }
            }
            return eventBuffers;
        }

    private:
        // What the table pointers name until Initialize succeeds, and again after Reset: an
        // uninitialized reader reports no entries and reads none, so every accessor answers as it
        // would for an empty table instead of dereferencing null.
        inline static const StateTableReader s_emptyStateTable;

        GenericObserver m_genericObserver;

        // Null until Initialize succeeds; ConsumeSourceProviderRegistrations answers as it would for an
        // empty stream.
        CircularBufferReader* m_pSourceProviderRegistrations = nullptr;

        // Reused per ConsumeSourceProviderRegistrations batch; holds tentative records until the batch's
        // consistency check passes.
        std::vector<NvTrace_ThreadEvent_SourceProviderRegistered> m_stagedSourceProviderRegistrations;

        uint32_t m_maxStaticSourceProviderIndex = 0;
        uint32_t m_maxSourceProviderCount = 0;
        uint64_t m_cellWriteDurationNsec = 0;

        uint32_t m_threadStatesSectionIndex = NVTRACE_INVALID_SECTION_INDEX;
        const StateTableReader* m_pThreadStates = &s_emptyStateTable;
        std::vector<EventBuffer> m_eventBuffers;

        // The old-driver shim; at most one of it and m_genericObserver is initialized.
        ThreadEventObserverV1 m_v1Observer;
    };

} // namespace nv::trace
