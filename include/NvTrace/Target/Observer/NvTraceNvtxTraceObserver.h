/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace_v1.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceNvtxTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceGenericObserver.h>
#include <NvTrace/Target/Observer/NvTraceNvtxTraceObserverV1.h>
#include <NvTrace/Target/Observer/NvTraceStateArenaReader.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nv::trace {

    /// \brief NVTX Trace Observer
    ///
    /// Typed wrapper over GenericObserver. Each state section is drained at attach and resolved by ref
    /// thereafter -- against the record references the event stream carries -- both delivering records
    /// tentatively, each closed by its own commit, so a consumer never keeps a record a racing writer
    /// disturbed.
    ///
    /// Initialize sniffs the buffer's format: a generic (V2) buffer opens with its major version, a V1
    /// buffer with the provider type UUID. A V1 buffer is served through NvtxTraceObserverV1, which
    /// converts each V1 state struct to its V2 record on delivery, so consumers see one format either way.
    ///
    /// A reference may announce the same entry more than once (a pre-attach announce drained after
    /// the attach pass already emitted it, or a stale announce resolving to a slot's new occupant);
    /// every delivery is internally consistent, so downstream keying on the entry's ref must be
    /// idempotent.
    ///
    /// Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class NvtxTraceObserver
    {
    public:
        using FbConfigurationView = FlatBufferView<FB::Provider::NvtxTrace::ProviderConfiguration>;
        using ThreadEventRecordStorage = NvtxTraceObserverV1::ThreadEventRecordStorage;

        ~NvtxTraceObserver() = default;
        NvtxTraceObserver() = default;

        NvtxTraceObserver(const NvtxTraceObserver&) = delete;
        NvtxTraceObserver(NvtxTraceObserver&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        NvtxTraceObserver& operator=(const NvtxTraceObserver&) = delete;
        NvtxTraceObserver& operator=(NvtxTraceObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_genericObserver = std::move(rhs.m_genericObserver);
                m_recordVersionInfo = std::exchange(rhs.m_recordVersionInfo, {});
                m_configurationView = std::exchange(rhs.m_configurationView, {});
                m_pProviderStatistics = std::exchange(rhs.m_pProviderStatistics, nullptr);
                m_pStrings = std::exchange(rhs.m_pStrings, nullptr);
                m_pDomains = std::exchange(rhs.m_pDomains, nullptr);
                m_pPushPopRanges = std::exchange(rhs.m_pPushPopRanges, nullptr);
                m_pStackGroups = std::exchange(rhs.m_pStackGroups, nullptr);
                m_v1Observer = std::move(rhs.m_v1Observer);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_v1Observer.Reset();
            m_pStackGroups = nullptr;
            m_pPushPopRanges = nullptr;
            m_pDomains = nullptr;
            m_pStrings = nullptr;
            m_pProviderStatistics = nullptr;
            m_configurationView = {};
            m_recordVersionInfo = {};
            m_genericObserver.Reset();
        }

        // providerTypeUuid arrives out-of-band from the Router, which took it from the provider's
        // advertisement: the V1 id means an old driver serving a V1 buffer.
        bool Initialize(const NVTRACE_UUID& providerTypeUuid, const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            if (providerTypeUuid == NvTrace_NvtxTrace_ProviderTypeUuid_V1)
            {
                return m_v1Observer.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            }

            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "NvtxTraceObserver::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            const bool genericObserverInitialized = m_genericObserver.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            if (!genericObserverInitialized)
            {
                return false;
            }

            const NVTRACE_UUID bufferProviderTypeUuid = m_genericObserver.ProviderTypeUuid();
            NVTRACE_LOG_ERR_RETURN_IF(bufferProviderTypeUuid != providerTypeUuid, false, 20, "Provider type UUID mismatch for NVTX Trace observer (buffer %s, router %s).", ToString(bufferProviderTypeUuid).c_str(), ToString(providerTypeUuid).c_str());

            DataBlobReader* const pRecordVersionInfoReader = FindDataBlobReader(NvTrace_NvtxTrace_SectionTag_RecordVersionInfo);
            NVTRACE_LOG_ERR_RETURN_IF(!pRecordVersionInfoReader, false, 20, "NVTX Trace record version info section not found or invalid.");

            static_assert((sizeof(NvTrace_NvtxTrace_RecordVersionInfo) % sizeof(uint64_t)) == 0, "Memcpy64 copies whole words only");
            alignas(uint64_t) NvTrace_NvtxTrace_RecordVersionInfo recordVersionInfo = {};
            size_t recordVersionInfoPayloadSize = 0;
            pRecordVersionInfoReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                recordVersionInfoPayloadSize = payloadSize;
                Memcpy64(&recordVersionInfo, pPayload, std::min(sizeof(recordVersionInfo), payloadSize));
            });
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfoPayloadSize < NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A, false, 20, "Record version info is incompatible for NVTX Trace (publishedSize=%zu expectedMinorVersion=%u).", recordVersionInfoPayloadSize, NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.recordVersionInfoMinorVersion < NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A, false, 20, "Record version info describes itself below the consumer's floor (recordVersionInfoMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.recordVersionInfoMinorVersion, NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A);

            // What flows downstream (GetRecordVersionInfoByteSpan) is this build's extent of the struct,
            // so the self-description must not claim a newer producer's larger shape over those bytes.
            recordVersionInfo.recordVersionInfoSize = std::min(recordVersionInfo.recordVersionInfoSize, static_cast<uint32_t>(sizeof(NvTrace_NvtxTrace_RecordVersionInfo)));
            recordVersionInfo.recordVersionInfoMinorVersion = std::min(recordVersionInfo.recordVersionInfoMinorVersion, static_cast<uint32_t>(NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A));

            // Gate only the state section records, whose layout the attach drain rests on; the thread
            // event payloads pass through to the consumers, which size-check each record.
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stringRegisteredMinorVersion < NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A, false, 20, "String registered record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.stringRegisteredMinorVersion, NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stringSegmentMinorVersion < NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A, false, 20, "String segment record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.stringSegmentMinorVersion, NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.domainCreatedMinorVersion < NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A, false, 20, "Domain created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.domainCreatedMinorVersion, NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.pushPopRangeCreatedMinorVersion < NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A, false, 20, "Push-pop range created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.pushPopRangeCreatedMinorVersion, NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.pushPopRangeMessageSegmentMinorVersion < NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A, false, 20, "Push-pop range message segment record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.pushPopRangeMessageSegmentMinorVersion, NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackGroupCreatedMinorVersion < NvTrace_NvtxTrace_StackGroupCreated_MINOR_VERSION_A, false, 20, "Stack group created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.stackGroupCreatedMinorVersion, NvTrace_NvtxTrace_StackGroupCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackGroupStackRefsMinorVersion < NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A, false, 20, "Stack group stack refs record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.stackGroupStackRefsMinorVersion, NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.eventAttributesMinorVersion < NvTrace_NvtxTrace_EventAttributes_MINOR_VERSION_A, false, 20, "Event attributes record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.eventAttributesMinorVersion, NvTrace_NvtxTrace_EventAttributes_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackRefMinorVersion < NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A, false, 20, "Stack ref record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.stackRefMinorVersion, NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A);

            // Each of these records leads a tail -- inline characters, message bytes, stack refs --
            // that a reader takes at the published size and lifts 64 bits at a time, so the leading
            // segment must both cover its fields and span whole 8-byte words.
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stringRegisteredSize < recordVersionInfo.stringRegisteredMinorVersion || (recordVersionInfo.stringRegisteredSize % alignof(uint64_t)) != 0, false, 20, "String registered record size cannot lead its inline characters (size=%u minorVersion=%u).", recordVersionInfo.stringRegisteredSize, recordVersionInfo.stringRegisteredMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stringSegmentSize < recordVersionInfo.stringSegmentMinorVersion || (recordVersionInfo.stringSegmentSize % alignof(uint64_t)) != 0, false, 20, "String segment record size cannot lead its characters (size=%u minorVersion=%u).", recordVersionInfo.stringSegmentSize, recordVersionInfo.stringSegmentMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.pushPopRangeCreatedSize < recordVersionInfo.pushPopRangeCreatedMinorVersion || (recordVersionInfo.pushPopRangeCreatedSize % alignof(uint64_t)) != 0, false, 20, "Push-pop range created record size cannot lead its inline message bytes (size=%u minorVersion=%u).", recordVersionInfo.pushPopRangeCreatedSize, recordVersionInfo.pushPopRangeCreatedMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.pushPopRangeMessageSegmentSize < recordVersionInfo.pushPopRangeMessageSegmentMinorVersion || (recordVersionInfo.pushPopRangeMessageSegmentSize % alignof(uint64_t)) != 0, false, 20, "Push-pop range message segment record size cannot lead its message bytes (size=%u minorVersion=%u).", recordVersionInfo.pushPopRangeMessageSegmentSize, recordVersionInfo.pushPopRangeMessageSegmentMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackGroupCreatedSize < recordVersionInfo.stackGroupCreatedMinorVersion || (recordVersionInfo.stackGroupCreatedSize % alignof(uint64_t)) != 0, false, 20, "Stack group created record size cannot lead its inline stack refs (size=%u minorVersion=%u).", recordVersionInfo.stackGroupCreatedSize, recordVersionInfo.stackGroupCreatedMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.stackGroupStackRefsSize < recordVersionInfo.stackGroupStackRefsMinorVersion || (recordVersionInfo.stackGroupStackRefsSize % alignof(uint64_t)) != 0, false, 20, "Stack group stack refs record size cannot lead its stack refs (size=%u minorVersion=%u).", recordVersionInfo.stackGroupStackRefsSize, recordVersionInfo.stackGroupStackRefsMinorVersion);

            DataBlobReader* const pConfigurationReader = FindDataBlobReader(NvTrace_NvtxTrace_SectionTag_Configuration);
            NVTRACE_LOG_ERR_RETURN_IF(!pConfigurationReader, false, 20, "NVTX Trace configuration section not found or invalid.");

            FbConfigurationView configurationView = {};
            pConfigurationReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                configurationView = {pPayload, payloadSize};
            });

            // The statistics blob is best-effort telemetry; a buffer without one still attaches.
            const NvTrace_NvtxTrace_ProviderStatistics* pProviderStatistics = nullptr;
            if (DataBlobReader* const pProviderStatisticsReader = FindDataBlobReader(NvTrace_NvtxTrace_SectionTag_ProviderStatistics))
            {
                pProviderStatisticsReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordVersionInfo.providerStatisticsMinorVersion < NvTrace_NvtxTrace_ProviderStatistics_MINOR_VERSION_A || payloadSize < NvTrace_NvtxTrace_ProviderStatistics_MINOR_VERSION_A)
                    {
                        return;
                    }
                    pProviderStatistics = reinterpret_cast<const NvTrace_NvtxTrace_ProviderStatistics*>(pPayload);
                });
            }

            const StateArenaReader* const pStrings = FindStateArenaReader(NvTrace_NvtxTrace_SectionTag_Strings);
            const StateTableReader* const pDomains = FindStateTableReader(NvTrace_NvtxTrace_SectionTag_Domains);
            const StateTableReader* const pPushPopRanges = FindStateTableReader(NvTrace_NvtxTrace_SectionTag_PushPopRanges);
            const StateTableReader* const pStackGroups = FindStateTableReader(NvTrace_NvtxTrace_SectionTag_StackGroups);
            if (!pStrings || !pDomains || !pPushPopRanges || !pStackGroups)
            {
                return false;
            }

            // Commit observer state only now, after all validation has succeeded.
            m_recordVersionInfo = recordVersionInfo;
            m_configurationView = configurationView;
            m_pProviderStatistics = pProviderStatistics;
            m_pStrings = pStrings;
            m_pDomains = pDomains;
            m_pPushPopRanges = pPushPopRanges;
            m_pStackGroups = pStackGroups;

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_genericObserver.IsInitialized() || m_v1Observer.IsInitialized();
        }

        uint32_t ThreadEventSourceProviderIndex() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.ThreadEventSourceProviderIndex() : m_genericObserver.ThreadEventSourceProviderIndex();
        }

        bool IsProviderMeasuring() const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.IsProviderMeasuring();
            }
            return m_genericObserver.ProviderStatus() == NvTrace_ProviderStatus_measuring;
        }

        const NvTrace_NvtxTrace_RecordVersionInfo& GetRecordVersionInfo() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.RecordVersionInfo() : m_recordVersionInfo;
        }

        NvTrace_ByteSpan GetRecordVersionInfoByteSpan() const
        {
            return {reinterpret_cast<const uint8_t*>(&GetRecordVersionInfo()), sizeof(NvTrace_NvtxTrace_RecordVersionInfo)};
        }

        FbConfigurationView GetProviderConfigurationFlatBufferView() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetProviderConfigurationFlatBufferView() : m_configurationView;
        }

        const FB::Provider::NvtxTrace::ProviderConfiguration* GetProviderConfiguration() const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.GetProviderConfiguration();
            }
            return m_configurationView.pData ? m_configurationView.GetRoot() : nullptr;
        }

        // Null for a V1 provider, which publishes no statistics blob.
        const NvTrace_NvtxTrace_ProviderStatistics* GetProviderStatistics() const
        {
            return m_pProviderStatistics;
        }

        // A V1 buffer's strings are an untyped byte pool with no records to count, so the shim reports none.
        uint32_t GetStringStateCount() const
        {
            return m_pStrings ? m_pStrings->GetRecordCount() : 0;
        }

        uint32_t GetDomainStateCount() const
        {
            return m_pDomains ? m_pDomains->EntryCount() : m_v1Observer.GetDomainStateCount();
        }

        uint32_t GetPushPopRangeStateCount() const
        {
            return m_pPushPopRanges ? m_pPushPopRanges->EntryCount() : m_v1Observer.GetPushPopRangeStateCount();
        }

        uint32_t GetStackGroupStateCount() const
        {
            return m_pStackGroups ? m_pStackGroups->EntryCount() : m_v1Observer.GetStackGroupStateCount();
        }

        // Drains every state section's records verbatim, in section order, each followed by
        // fnCommitRecord(bool). Every record names what it belongs to in its own payload, so the drain
        // carries no grouping and needs no knowledge of which record type continues which.
        // fnOnRecordBytes(NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount)
        //   -- the record verbatim, valid only during the call, under its own address.
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ConsumeAttachStateRecords(TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                // A V1 buffer has no section descriptors to order, so the shim reproduces the order a
                // generic buffer's tags give: strings, domains, push-pop ranges, stack groups.
                const auto fnDrainV1Table = [](uint32_t entryCount, auto&& fnReadEntry) {
                    for (uint32_t entryIndex = 0; entryIndex < entryCount; ++entryIndex)
                    {
                        fnReadEntry(entryIndex);
                    }
                };

                m_v1Observer.ReadAllStringRecordBytes(fnOnRecordBytes, fnCommitRecord);
                fnDrainV1Table(m_v1Observer.GetDomainStateCount(), [&](uint32_t entryIndex) { m_v1Observer.ReadDomainRecordBytes(entryIndex, fnOnRecordBytes, fnCommitRecord); });
                fnDrainV1Table(m_v1Observer.GetPushPopRangeStateCount(), [&](uint32_t entryIndex) { m_v1Observer.ReadPushPopRangeGroupRecordBytes(entryIndex, fnOnRecordBytes, fnCommitRecord); });
                fnDrainV1Table(m_v1Observer.GetStackGroupStateCount(), [&](uint32_t entryIndex) { m_v1Observer.ReadStackGroupGroupRecordBytes(entryIndex, fnOnRecordBytes, fnCommitRecord); });
                return;
            }

            m_genericObserver.ConsumeAttachStateRecords(fnOnRecordBytes, fnCommitRecord);
        }

        // The bytes stay tentative until fnCommitRecord(true). A nested reference among the records
        // delivered is dropped: resolution is one hop.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ResolveRecordReference(const NvTrace_RecordReference& reference, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ResolveRecordReference(reference, std::forward<TFN_ReadRecord>(fnReadRecord), std::forward<TFN_CommitRecord>(fnCommitRecord));
                return;
            }

            if (m_genericObserver.SectionCount() <= reference.sectionIndex)
            {
                return;
            }

            // A reference names a section, and what it resolves to is the section's own business: a
            // state table entry leads a chunk chain, while an arena record stands alone.
            const GenericObserver::Section& section = m_genericObserver.GetSection(reference.sectionIndex);

            const auto fnOnResolvedRecord = [&](NvTrace_CorrelationId recordCorrelationId, uint32_t resolvedRecordType, const uint8_t* pResolvedPayload, size_t resolvedPayloadSize) {
                if (resolvedRecordType == NvTrace_ReservedRecordType_RecordReference)
                {
                    return;
                }
                fnReadRecord(recordCorrelationId, resolvedRecordType, pResolvedPayload, resolvedPayloadSize);
            };

            if (const StateArenaReader* const pStateArena = section.AsStateArena())
            {
                pStateArena->ReadRecord(reference.recordRef, fnOnResolvedRecord, fnCommitRecord);
                return;
            }

            if (const StateTableReader* const pStateTable = section.AsStateTable())
            {
                pStateTable->ReadEntry(reference.recordRef, fnOnResolvedRecord, fnCommitRecord);
            }
        }

        // Judges the record type as read, before TranslateThreadEventRecord.
        bool IsObjectLifetimeThreadEventRecord(uint32_t recordType) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.IsObjectLifetimeThreadEventRecord(recordType);
            }

            return recordType == NvTrace_ReservedRecordType_RecordReference || recordType == NvTrace_NvtxTrace_RecordType_StackGroupDestroyed;
        }

        // The result may point into storage, which must outlive it; a translated Mark's payload lives
        // until the next Mark translation.
        const ThreadEventRecord* TranslateThreadEventRecord(const ThreadEventRecord& record, ThreadEventRecordStorage& storage) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.TranslateThreadEventRecord(record, storage);
            }

            return &record;
        }

    private:
        DataBlobReader* FindDataBlobReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            return pSection ? pSection->AsDataBlob() : nullptr;
        }

        const StateTableReader* FindStateTableReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, nullptr, 20, "NVTX Trace state table section not found (sectionTag=%u).", sectionTag);

            const StateTableReader* const pStateTableReader = pSection->AsStateTable();
            NVTRACE_LOG_ERR_RETURN_IF(!pStateTableReader, nullptr, 20, "NVTX Trace state table section is reader-less (sectionTag=%u).", sectionTag);

            return pStateTableReader;
        }

        const StateArenaReader* FindStateArenaReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, nullptr, 20, "NVTX Trace state arena section not found (sectionTag=%u).", sectionTag);

            const StateArenaReader* const pStateArenaReader = pSection->AsStateArena();
            NVTRACE_LOG_ERR_RETURN_IF(!pStateArenaReader, nullptr, 20, "NVTX Trace state arena section is reader-less (sectionTag=%u).", sectionTag);

            return pStateArenaReader;
        }

    private:
        GenericObserver m_genericObserver;
        NvTrace_NvtxTrace_RecordVersionInfo m_recordVersionInfo = {};
        FbConfigurationView m_configurationView = {};
        const NvTrace_NvtxTrace_ProviderStatistics* m_pProviderStatistics = nullptr;

        const StateArenaReader* m_pStrings = nullptr;
        const StateTableReader* m_pDomains = nullptr;
        const StateTableReader* m_pPushPopRanges = nullptr;
        const StateTableReader* m_pStackGroups = nullptr;

        // At most one of this and m_genericObserver is initialized.
        NvtxTraceObserverV1 m_v1Observer;
    };

} // namespace nv::trace
