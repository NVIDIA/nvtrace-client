/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace_v1.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer_v1.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/DataFormats/nvtrace_thread_event_v1.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceNvtxTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceNvtxTraceRecordUtilities.h>
#include <NvTrace/Target/NvTraceRevisionedRefCodec.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceNvtxTraceConsumerTypes.h>
#include <NvTrace/Target/Observer/NvTraceObserverUtilities.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace nv::trace {

    inline NvTrace_NvtxTrace_RecordType TranslateNvtxTraceV1ObjectLifetimeCreatedRecordType(uint32_t v1RecordType)
    {
        switch (v1RecordType)
        {
            case NvTrace_NvtxTrace_EventType_ObjectLifetime_StringTableEntryCreated_V1: return NvTrace_NvtxTrace_RecordType_StringRegistered;
            case NvTrace_NvtxTrace_EventType_ObjectLifetime_DomainCreated_V1: return NvTrace_NvtxTrace_RecordType_DomainCreated;
            case NvTrace_NvtxTrace_EventType_ObjectLifetime_PushPopRangeCreated_V1: return NvTrace_NvtxTrace_RecordType_PushPopRangeCreated;
            case NvTrace_NvtxTrace_EventType_ObjectLifetime_StackGroupCreated_V1: return NvTrace_NvtxTrace_RecordType_StackGroupCreated;
            default: return NvTrace_NvtxTrace_RecordType_INVALID;
        }
    }

    // Mark also needs payload translation; the other thread-event layouts are shared.
    inline NvTrace_NvtxTrace_RecordType TranslateNvtxTraceV1ThreadEventRecordType(uint32_t v1RecordType)
    {
        switch (v1RecordType)
        {
            case NvTrace_NvtxTrace_EventType_Thread_Mark_V1: return NvTrace_NvtxTrace_RecordType_Mark;
            case NvTrace_NvtxTrace_EventType_Thread_RangePop_V1: return NvTrace_NvtxTrace_RecordType_RangePop;
            case NvTrace_NvtxTrace_EventType_Thread_DroppedRangePush_V1: return NvTrace_NvtxTrace_RecordType_DroppedRangePush;
            case NvTrace_NvtxTrace_EventType_Thread_DroppedRangePop_V1: return NvTrace_NvtxTrace_RecordType_DroppedRangePop;
            default: return NvTrace_NvtxTrace_RecordType_INVALID;
        }
    }

    /// \brief NVTX Trace Observer over the V1 provider buffer format -- the old-driver shim.
    ///
    /// Converts every V1 state struct to its V2 record on delivery, behind the record reads
    /// NvtxTraceObserver exposes. The V1 ref currency is kept: a push-pop range's and a stack
    /// group's ref packs (revision, entry index) over the V1 table's entry count, a string's ref is
    /// its byte offset into the V1 string table and a domain's is its entry index -- so a reference
    /// the event stream carries resolves against the values the old driver minted.
    ///
    /// Synthesized records lay out as sizeof(the fixed segment) followed immediately by the variable
    /// segment, exactly as a generic producer of this build writes them; a V1 "none" ref (all ones)
    /// becomes NVTRACE_INVALID_REVISIONED_REF. The V1 tables carry no creation timestamp for strings,
    /// domains and stack groups, so those records are delivered with 0.
    ///
    /// Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class NvtxTraceObserverV1
    {
    public:
        using FbConfigurationView = FlatBufferView<FB::Provider::NvtxTrace::ProviderConfiguration>;

        struct ThreadEventRecordStorage
        {
            NvtxTraceTranslatedRecordV1 translated;
            ThreadEventRecord record;
        };

        ~NvtxTraceObserverV1() = default;
        NvtxTraceObserverV1() = default;

        NvtxTraceObserverV1(const NvtxTraceObserverV1&) = delete;
        NvtxTraceObserverV1(NvtxTraceObserverV1&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        NvtxTraceObserverV1& operator=(const NvtxTraceObserverV1&) = delete;
        NvtxTraceObserverV1& operator=(NvtxTraceObserverV1&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_sections = std::exchange(rhs.m_sections, {});
                m_recordVersionInfo = std::exchange(rhs.m_recordVersionInfo, {});
                m_maxRecordCountPerStateChunk = std::exchange(rhs.m_maxRecordCountPerStateChunk, 0);
                m_stateChunkRefCodec = std::exchange(rhs.m_stateChunkRefCodec, {});
                m_pushPopRangeStateStride = std::exchange(rhs.m_pushPopRangeStateStride, 0);
                m_pushPopRangeRefCodec = std::exchange(rhs.m_pushPopRangeRefCodec, {});
                m_stackGroupStateStride = std::exchange(rhs.m_stackGroupStateStride, 0);
                m_stackGroupRefCodec = std::exchange(rhs.m_stackGroupRefCodec, {});
                m_recordBytesBuffer = std::exchange(rhs.m_recordBytesBuffer, {});
                m_eventAttributesBytesBuffer = std::exchange(rhs.m_eventAttributesBytesBuffer, {});
                m_markPayloadBuffer = std::exchange(rhs.m_markPayloadBuffer, {});
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_markPayloadBuffer.clear();
            m_eventAttributesBytesBuffer.clear();
            m_recordBytesBuffer.clear();
            m_stackGroupRefCodec.Reset();
            m_stackGroupStateStride = 0;
            m_pushPopRangeRefCodec.Reset();
            m_pushPopRangeStateStride = 0;
            m_stateChunkRefCodec.Reset();
            m_maxRecordCountPerStateChunk = 0;
            m_recordVersionInfo = {};
            m_sections = {};
        }

        bool Initialize(const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "NvtxTraceObserverV1::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            // The buffer is shared memory under the profiled process's control: every offset, stride
            // and count must be proven inside the mapping before it shapes a pointer.
            const auto fnRangeFits = [&](uint64_t itemsOffset, uint64_t itemSize, uint64_t itemCount) {
                return itemsOffset == 0 || (itemSize != 0 && itemsOffset <= providerBufferSize && itemCount <= (providerBufferSize - itemsOffset) / itemSize);
            };

            // Entry N of a section sits at itemsOffset + itemSize * N, so both must carry the entry's
            // alignment for the 64-bit loads the readers perform on every entry to be aligned.
            const auto fnRangeAligned = [](uint64_t itemsOffset, uint64_t itemSize, uint64_t itemAlignment) {
                return itemsOffset == 0 || ((itemsOffset % itemAlignment) == 0 && (itemSize % itemAlignment) == 0);
            };

            // A section whose entries do not span everything the observer reads out of them, or that
            // holds no entries, is served as absent, exactly like one the layout omits.
            const auto fnReadableSectionOffset = [](uint64_t sectionOffset, uint32_t entryCount, uint32_t entrySize, uint32_t entryMinorVersion, uint32_t readMinorVersion, const char* pSectionName) -> uint64_t {
                if (sectionOffset == 0 || entryCount == 0)
                {
                    NVTRACE_LOG_INF(20, "V1 provider buffer %s section is not present; %s data will not be available.", pSectionName, pSectionName);
                    return 0;
                }

                if (entryMinorVersion < readMinorVersion || entrySize < readMinorVersion)
                {
                    NVTRACE_LOG_INF(20, "V1 provider buffer %s entries (size %u minor version %u) are below the %u bytes the observer reads; %s data will not be available.", pSectionName, entrySize, entryMinorVersion, readMinorVersion, pSectionName);
                    return 0;
                }

                return sectionOffset;
            };

            NVTRACE_LOG_ERR_RETURN_IF(pMappedProviderBufferMemory == nullptr || providerBufferSize < sizeof(NvTrace_ProviderBufferHeader_V1), false, 20, "The mapped V1 provider buffer cannot hold its header (providerBufferSize %zu).", providerBufferSize);

            // Header/layout/sections are read in place; require the base to meet the strictest alignment.
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pMappedProviderBufferMemory) % alignof(NvTrace_ProviderBufferHeader_V1)) != 0, false, 20, "Provider buffer base must be %zu-byte aligned (pMappedProviderBufferMemory=%p).", alignof(NvTrace_ProviderBufferHeader_V1), static_cast<const void*>(pMappedProviderBufferMemory));

            const NvTrace_ProviderBufferHeader_V1* const pProviderBufferHeader = reinterpret_cast<const NvTrace_ProviderBufferHeader_V1*>(pMappedProviderBufferMemory);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->providerTypeUuid != NvTrace_NvtxTrace_ProviderTypeUuid_V1, false, 20, "Invalid provider type UUID for NVTX Trace V1 observer (got %s, expected %s).", ToString(pProviderBufferHeader->providerTypeUuid).c_str(), ToString(NvTrace_NvtxTrace_ProviderTypeUuid_V1).c_str());
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->minorVersion < NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A, false, 20, "Provider buffer header minor version %u is incompatible with the NVTX Trace V1 observer (expected %u).", pProviderBufferHeader->minorVersion, NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->providerBufferLayoutMinorVersion < NvTrace_NvtxTrace_ProviderBufferLayout_V1_MINOR_VERSION_A, false, 20, "Provider buffer layout minor version %u is incompatible with the NVTX Trace V1 observer (expected %u).", pProviderBufferHeader->providerBufferLayoutMinorVersion, NvTrace_NvtxTrace_ProviderBufferLayout_V1_MINOR_VERSION_A);

            const uint64_t providerBufferLayoutOffset = pProviderBufferHeader->providerBufferLayoutOffset;
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayoutOffset == 0, false, 20, "The V1 provider buffer header names no provider buffer layout.");
            const bool providerBufferLayoutAligned = fnRangeAligned(providerBufferLayoutOffset, sizeof(NvTrace_NvtxTrace_ProviderBufferLayout_V1), alignof(NvTrace_NvtxTrace_ProviderBufferLayout_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!providerBufferLayoutAligned, false, 20, "The provider buffer layout is misaligned (offset %" PRIu64 " alignment %zu).", providerBufferLayoutOffset, alignof(NvTrace_NvtxTrace_ProviderBufferLayout_V1));
            const bool providerBufferLayoutFits = fnRangeFits(providerBufferLayoutOffset, sizeof(NvTrace_NvtxTrace_ProviderBufferLayout_V1), 1);
            NVTRACE_LOG_ERR_RETURN_IF(!providerBufferLayoutFits, false, 20, "The provider buffer layout lies outside the mapping (offset %" PRIu64 ").", providerBufferLayoutOffset);

            const NvTrace_NvtxTrace_ProviderBufferLayout_V1& providerBufferLayout = *reinterpret_cast<const NvTrace_NvtxTrace_ProviderBufferLayout_V1*>(pMappedProviderBufferMemory + providerBufferLayoutOffset);
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventVersionInfoMinorVersion < NvTrace_NvtxTrace_EventVersionInfo_V1_MINOR_VERSION_A, false, 20, "Event version info minor version %u is incompatible with the NVTX Trace V1 observer (expected %u).", providerBufferLayout.eventVersionInfoMinorVersion, NvTrace_NvtxTrace_EventVersionInfo_V1_MINOR_VERSION_A);

            const uint64_t eventVersionInfoOffset = providerBufferLayout.eventVersionInfoOffset;
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfoOffset == 0, false, 20, "The V1 provider buffer layout names no event version info.");
            const bool eventVersionInfoAligned = fnRangeAligned(eventVersionInfoOffset, sizeof(NvTrace_NvtxTrace_EventVersionInfo_V1), alignof(NvTrace_NvtxTrace_EventVersionInfo_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!eventVersionInfoAligned, false, 20, "The event version info is misaligned (offset %" PRIu64 " alignment %zu).", eventVersionInfoOffset, alignof(NvTrace_NvtxTrace_EventVersionInfo_V1));
            const bool eventVersionInfoFits = fnRangeFits(eventVersionInfoOffset, sizeof(NvTrace_NvtxTrace_EventVersionInfo_V1), 1);
            NVTRACE_LOG_ERR_RETURN_IF(!eventVersionInfoFits, false, 20, "The event version info lies outside the mapping (offset %" PRIu64 ").", eventVersionInfoOffset);

            const NvTrace_NvtxTrace_EventVersionInfo_V1& eventVersionInfo = *reinterpret_cast<const NvTrace_NvtxTrace_EventVersionInfo_V1*>(pMappedProviderBufferMemory + eventVersionInfoOffset);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.stateChunkHeaderMinorVersion < NvTrace_AdaptiveChunkHeader_MINOR_VERSION_A, false, 20, "State chunk header minor version %u is incompatible with the NVTX Trace V1 observer (expected %u).", eventVersionInfo.stateChunkHeaderMinorVersion, NvTrace_AdaptiveChunkHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.stateChunkRecordMinorVersion < NvTrace_RecordDesc_MINOR_VERSION_A, false, 20, "State chunk record minor version %u is incompatible with the NVTX Trace V1 observer (expected %u).", eventVersionInfo.stateChunkRecordMinorVersion, NvTrace_RecordDesc_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.stateChunkRecordSize == 0 || providerBufferLayout.stateChunkSize < eventVersionInfo.stateChunkHeaderSize, false, 20, "Invalid V1 state chunk layout (chunk size=%u, header size=%u, record size=%u).", providerBufferLayout.stateChunkSize, eventVersionInfo.stateChunkHeaderSize, eventVersionInfo.stateChunkRecordSize);
            NVTRACE_LOG_ERR_RETURN_IF((eventVersionInfo.stateChunkHeaderSize % alignof(NvTrace_RecordDesc)) != 0 || (eventVersionInfo.stateChunkRecordSize % alignof(NvTrace_RecordDesc)) != 0, false, 20, "The V1 state chunk records are misaligned (header size=%u record size=%u alignment=%zu).", eventVersionInfo.stateChunkHeaderSize, eventVersionInfo.stateChunkRecordSize, alignof(NvTrace_RecordDesc));

            const uint64_t stringTableOffset = providerBufferLayout.stringTableOffset;
            NVTRACE_LOG_INF_IF(stringTableOffset == 0, 20, "V1 provider buffer string table section is not present; registered string data will not be available.");

            const uint64_t stateChunksOffset = providerBufferLayout.stateChunksCount == 0 ? 0 : providerBufferLayout.stateChunksOffset;
            NVTRACE_LOG_INF_IF(stateChunksOffset == 0, 20, "V1 provider buffer state chunk pool is not present; the chain records of push-pop ranges and stack groups will not be available.");

            const uint64_t domainStatesOffset = fnReadableSectionOffset(providerBufferLayout.domainStatesOffset, providerBufferLayout.domainStatesCount, eventVersionInfo.domainStateSize, eventVersionInfo.domainStateMinorVersion, NvTrace_NvtxTrace_DomainState_V1_MINOR_VERSION_A, "domain state");
            const uint64_t pushPopRangeStatesOffset = fnReadableSectionOffset(providerBufferLayout.pushPopRangeStatesOffset, providerBufferLayout.pushPopRangeStatesCount, eventVersionInfo.pushPopRangeStateSize, eventVersionInfo.pushPopRangeStateMinorVersion, NvTrace_NvtxTrace_PushPopRangeState_V1_MINOR_VERSION_A, "push-pop range state");

            const uint64_t stackGroupStatesSectionOffset = fnReadableSectionOffset(providerBufferLayout.stackGroupStatesOffset, providerBufferLayout.stackGroupStatesCount, eventVersionInfo.stackGroupStateSize, eventVersionInfo.stackGroupStateMinorVersion, NvTrace_NvtxTrace_StackGroupState_V1_MINOR_VERSION_A, "stack group state");
            // The inline stack refs are read out of every stack group entry, so a stack ref shape this
            // build cannot read disables the whole table.
            const bool stackRefsReadable = NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A <= eventVersionInfo.stackRefMinorVersion && NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A <= eventVersionInfo.stackRefSize;
            NVTRACE_LOG_INF_IF(!stackRefsReadable, 20, "V1 provider buffer stack refs (size %u minor version %u) are below the %u bytes the observer reads; stack group data will not be available.", eventVersionInfo.stackRefSize, eventVersionInfo.stackRefMinorVersion, NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A);
            const uint64_t stackGroupStatesOffset = stackRefsReadable ? stackGroupStatesSectionOffset : 0;

            // One entry's inline trailer must be proven inside the mapping before it shapes a stride.
            const bool inlineStackRefsFit = fnRangeFits(stackGroupStatesOffset, eventVersionInfo.stackRefSize, providerBufferLayout.stackGroupInlineStackRefCount);
            NVTRACE_LOG_ERR_RETURN_IF(!inlineStackRefsFit, false, 20, "The inline stack refs of a V1 stack group entry lie outside the mapping (offset %" PRIu64 " size %u count %u).", stackGroupStatesOffset, eventVersionInfo.stackRefSize, providerBufferLayout.stackGroupInlineStackRefCount);

            const uint64_t pushPopRangeStateStride = static_cast<uint64_t>(eventVersionInfo.pushPopRangeStateSize) + providerBufferLayout.pushPopRangeInlineMessageSize;
            const uint64_t stackGroupStateStride = AlignUpP2(static_cast<uint64_t>(eventVersionInfo.stackGroupStateSize) + static_cast<uint64_t>(providerBufferLayout.stackGroupInlineStackRefCount) * eventVersionInfo.stackRefSize, NVTRACE_CPU_CACHELINE_SIZE);

            NVTRACE_LOG_ERR_RETURN_IF((providerBufferLayout.configurationOffset % alignof(uint64_t)) != 0, false, 20, "The provider configuration flatbuffer is misaligned (offset %" PRIu64 " alignment %zu).", providerBufferLayout.configurationOffset, alignof(uint64_t));
            const bool stateChunksAligned = fnRangeAligned(stateChunksOffset, providerBufferLayout.stateChunkSize, alignof(NvTrace_AdaptiveChunkHeader));
            NVTRACE_LOG_ERR_RETURN_IF(!stateChunksAligned, false, 20, "The state chunks are misaligned (offset %" PRIu64 " size %u alignment %zu).", stateChunksOffset, providerBufferLayout.stateChunkSize, alignof(NvTrace_AdaptiveChunkHeader));
            const bool domainStatesAligned = fnRangeAligned(domainStatesOffset, eventVersionInfo.domainStateSize, alignof(NvTrace_NvtxTrace_DomainState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!domainStatesAligned, false, 20, "The domain states are misaligned (offset %" PRIu64 " size %u alignment %zu).", domainStatesOffset, eventVersionInfo.domainStateSize, alignof(NvTrace_NvtxTrace_DomainState_V1));
            const bool pushPopRangeStatesAligned = fnRangeAligned(pushPopRangeStatesOffset, pushPopRangeStateStride, alignof(NvTrace_NvtxTrace_PushPopRangeState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!pushPopRangeStatesAligned, false, 20, "The push-pop range states are misaligned (offset %" PRIu64 " stride %" PRIu64 " alignment %zu).", pushPopRangeStatesOffset, pushPopRangeStateStride, alignof(NvTrace_NvtxTrace_PushPopRangeState_V1));
            const bool stackGroupStatesAligned = fnRangeAligned(stackGroupStatesOffset, stackGroupStateStride, alignof(NvTrace_NvtxTrace_StackGroupState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!stackGroupStatesAligned, false, 20, "The stack group states are misaligned (offset %" PRIu64 " stride %" PRIu64 " alignment %zu).", stackGroupStatesOffset, stackGroupStateStride, alignof(NvTrace_NvtxTrace_StackGroupState_V1));

            const bool configurationFits = fnRangeFits(providerBufferLayout.configurationOffset, 1, providerBufferLayout.configurationSize);
            NVTRACE_LOG_ERR_RETURN_IF(!configurationFits, false, 20, "The provider configuration lies outside the mapping (offset %" PRIu64 " size %u).", providerBufferLayout.configurationOffset, providerBufferLayout.configurationSize);
            // The flatbuffer root offset is read out of the blob's first word before anything else.
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.configurationOffset != 0 && providerBufferLayout.configurationSize < sizeof(flatbuffers::uoffset_t), false, 20, "The provider configuration is too small to hold a flatbuffer root (size %u, expected %zu at least).", providerBufferLayout.configurationSize, sizeof(flatbuffers::uoffset_t));
            const bool stringTableFits = fnRangeFits(stringTableOffset, 1, providerBufferLayout.stringTableSize);
            NVTRACE_LOG_ERR_RETURN_IF(!stringTableFits, false, 20, "The string table lies outside the mapping (offset %" PRIu64 " size %" PRIu64 ").", stringTableOffset, providerBufferLayout.stringTableSize);
            const bool stateChunksFit = fnRangeFits(stateChunksOffset, providerBufferLayout.stateChunkSize, providerBufferLayout.stateChunksCount);
            NVTRACE_LOG_ERR_RETURN_IF(!stateChunksFit, false, 20, "The state chunks lie outside the mapping (offset %" PRIu64 " size %u count %u).", stateChunksOffset, providerBufferLayout.stateChunkSize, providerBufferLayout.stateChunksCount);
            const bool domainStatesFit = fnRangeFits(domainStatesOffset, eventVersionInfo.domainStateSize, providerBufferLayout.domainStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!domainStatesFit, false, 20, "The domain states lie outside the mapping (offset %" PRIu64 " size %u count %u).", domainStatesOffset, eventVersionInfo.domainStateSize, providerBufferLayout.domainStatesCount);
            const bool pushPopRangeStatesFit = fnRangeFits(pushPopRangeStatesOffset, pushPopRangeStateStride, providerBufferLayout.pushPopRangeStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!pushPopRangeStatesFit, false, 20, "The push-pop range states lie outside the mapping (offset %" PRIu64 " stride %" PRIu64 " count %u).", pushPopRangeStatesOffset, pushPopRangeStateStride, providerBufferLayout.pushPopRangeStatesCount);
            const bool stackGroupStatesFit = fnRangeFits(stackGroupStatesOffset, stackGroupStateStride, providerBufferLayout.stackGroupStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!stackGroupStatesFit, false, 20, "The stack group states lie outside the mapping (offset %" PRIu64 " stride %" PRIu64 " count %u).", stackGroupStatesOffset, stackGroupStateStride, providerBufferLayout.stackGroupStatesCount);

            // Commit observer state only now, after all validation has succeeded.
            m_sections.pProviderBufferHeader = pProviderBufferHeader;
            m_sections.pProviderBufferLayout = &providerBufferLayout;
            m_sections.pEventVersionInfo = &eventVersionInfo;
            AssignFlatbufferSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.configurationOffset, m_sections.fbProviderConfiguration);
            AssignSectionPointer(pMappedProviderBufferMemory, stringTableOffset, m_sections.pStringTable);
            AssignSectionPointer(pMappedProviderBufferMemory, domainStatesOffset, m_sections.pDomainStates);
            AssignSectionPointer(pMappedProviderBufferMemory, stateChunksOffset, m_sections.pStateChunks);
            AssignSectionPointer(pMappedProviderBufferMemory, pushPopRangeStatesOffset, m_sections.pPushPopRangeStates);
            AssignSectionPointer(pMappedProviderBufferMemory, stackGroupStatesOffset, m_sections.pStackGroupStates);

            m_recordVersionInfo = MakeRecordVersionInfoFromV1(eventVersionInfo);
            m_maxRecordCountPerStateChunk = (providerBufferLayout.stateChunkSize - eventVersionInfo.stateChunkHeaderSize) / eventVersionInfo.stateChunkRecordSize;

            // A codec's field widths come from a non-zero item count; an absent table mints and
            // resolves no ref, so its codec stays in the zero-width state Reset() leaves it in.
            if (stateChunksOffset != 0)
            {
                m_stateChunkRefCodec.Initialize(providerBufferLayout.stateChunksCount, m_maxRecordCountPerStateChunk);
            }
            m_pushPopRangeStateStride = static_cast<size_t>(pushPopRangeStateStride);
            if (pushPopRangeStatesOffset != 0)
            {
                m_pushPopRangeRefCodec.Initialize(providerBufferLayout.pushPopRangeStatesCount);
            }
            m_stackGroupStateStride = static_cast<size_t>(stackGroupStateStride);
            if (stackGroupStatesOffset != 0)
            {
                m_stackGroupRefCodec.Initialize(providerBufferLayout.stackGroupStatesCount);
            }

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully over a V1 provider buffer.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_sections.pProviderBufferHeader != nullptr;
        }

        bool IsProviderMeasuring() const
        {
            return IsInitialized() && m_sections.pProviderBufferHeader->providerStatus == NvTrace_ProviderStatus_measuring;
        }

        // The ThreadEvent V1 shim delivers V1 static id N as source provider index N - 1.
        static uint32_t ThreadEventSourceProviderIndex()
        {
            return NvTrace_ThreadEvent_StaticProviderId_V1_NVTX - 1;
        }

        static bool IsObjectLifetimeThreadEventRecord(uint32_t recordType)
        {
            return NvTrace_NvtxTrace_EventType_ObjectLifetime_LO_V1 < recordType && recordType < NvTrace_NvtxTrace_EventType_ObjectLifetime_HI_V1;
        }

        // A translated Mark's payload lives until the next Mark translation or Reset.
        const ThreadEventRecord* TranslateThreadEventRecord(const ThreadEventRecord& record, ThreadEventRecordStorage& storage) const
        {
            if (IsObjectLifetimeThreadEventRecord(record.recordType))
            {
                const bool translationSucceeded = TranslateObjectLifetimeEvent(record, storage.translated);
                if (!translationSucceeded)
                {
                    return nullptr;
                }

                storage.record = record;
                storage.record.recordType = storage.translated.recordType;
                storage.record.pPayload = reinterpret_cast<const uint8_t*>(&storage.translated.payload);
                storage.record.payloadSize = storage.translated.payloadSize;
                return &storage.record;
            }

            if (record.recordType == NvTrace_NvtxTrace_EventType_Thread_Mark_V1)
            {
                const bool translationSucceeded = TranslateMarkEvent(record, storage.record);
                return translationSucceeded ? &storage.record : nullptr;
            }

            if (NvTrace_NvtxTrace_EventType_Thread_LO_V1 < record.recordType && record.recordType < NvTrace_NvtxTrace_EventType_Thread_HI_V1)
            {
                storage.record = record;
                storage.record.recordType = TranslateNvtxTraceV1ThreadEventRecordType(record.recordType);
                return &storage.record;
            }

            NVTRACE_LOG_ERR(50, "Unknown V1 NVTX Trace ThreadEvent record type %u.", record.recordType);
            return nullptr;
        }

        const NvTrace_NvtxTrace_RecordVersionInfo& RecordVersionInfo() const
        {
            return m_recordVersionInfo;
        }

        FbConfigurationView GetProviderConfigurationFlatBufferView() const
        {
            if (!m_sections.fbProviderConfiguration)
            {
                return {};
            }

            // Whole words, as a generic data blob is sized; the V1 layout aligns the next section, so
            // the pad is the zeroed gap behind the blob.
            return {reinterpret_cast<const uint8_t*>(m_sections.pProviderBufferHeader) + m_sections.pProviderBufferLayout->configurationOffset, NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8(m_sections.pProviderBufferLayout->configurationSize) * sizeof(uint64_t)};
        }

        const FB::Provider::NvtxTrace::ProviderConfiguration* GetProviderConfiguration() const
        {
            return m_sections.fbProviderConfiguration;
        }

        uint32_t GetDomainStateCount() const
        {
            return m_sections.pDomainStates ? m_sections.pProviderBufferLayout->domainStatesCount : 0;
        }

        uint32_t GetPushPopRangeStateCount() const
        {
            return m_sections.pPushPopRangeStates ? m_sections.pProviderBufferLayout->pushPopRangeStatesCount : 0;
        }

        uint32_t GetStackGroupStateCount() const
        {
            return m_sections.pStackGroupStates ? m_sections.pProviderBufferLayout->stackGroupStatesCount : 0;
        }

        // The V1 string table is a byte arena of NUL-terminated strings from offset 1 -- offset 0 is the
        // null sentinel every "no string" field names, and the first unwritten entry ends the walk. The
        // delivered bytes live in an observer-owned staging buffer, valid only during the call.
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadAllStringRecordBytes(TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pStringTable)
            {
                return;
            }

            const uint64_t stringTableSize = m_sections.pProviderBufferLayout->stringTableSize;
            for (uint64_t stringTableEntryOffset = 1; stringTableEntryOffset < stringTableSize;)
            {
                const size_t byteCount = DeliverStringRecordBytes(stringTableEntryOffset, fnOnRecordBytes, fnCommitRecord);
                if (byteCount == 0)
                {
                    return;
                }

                stringTableEntryOffset += byteCount + 1;
            }
        }

        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadStringRecordBytes(uint64_t stringTableEntryOffset, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            DeliverStringRecordBytes(stringTableEntryOffset, fnOnRecordBytes, fnCommitRecord);
        }

        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadDomainRecordBytes(uint32_t entryIndex, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pDomainStates || m_sections.pProviderBufferLayout->domainStatesCount <= entryIndex)
            {
                return;
            }

            const auto& domainState = *reinterpret_cast<const NvTrace_NvtxTrace_DomainState_V1*>(m_sections.pDomainStates + static_cast<size_t>(m_sections.pEventVersionInfo->domainStateSize) * entryIndex);

            const uint64_t revisionBefore = LoadAcquire(&domainState.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return;
            }

            const RevisionCursor cursor{&domainState.revision, revisionBefore};

            NvTrace_NvtxTrace_DomainCreated record = {};
            record.nameStringHandle = domainState.nameStringTableEntryOffset;
            record.domainIndex = entryIndex;

            fnOnRecordBytes(entryIndex, NvTrace_NvtxTrace_RecordType_DomainCreated, reinterpret_cast<const uint8_t*>(&record), sizeof(record));

            fnCommitRecord(cursor.CheckConsistency());
        }

        // Every record is delivered with the entry's ref and followed by its own verdict; the chain is
        // abandoned at the first failed one.
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadPushPopRangeGroupRecordBytes(uint32_t entryIndex, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pPushPopRangeStates || m_sections.pProviderBufferLayout->pushPopRangeStatesCount <= entryIndex)
            {
                return;
            }

            const uint8_t* const pEntryBytes = m_sections.pPushPopRangeStates + m_pushPopRangeStateStride * entryIndex;
            const auto& pushPopRangeState = *reinterpret_cast<const NvTrace_NvtxTrace_PushPopRangeState_V1*>(pEntryBytes);

            const uint64_t revisionBefore = LoadAcquire(&pushPopRangeState.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return;
            }

            const RevisionCursor cursor{&pushPopRangeState.revision, revisionBefore};
            const NvTrace_RevisionedRef entryRef = EncodePushPopRangeRef(revisionBefore, entryIndex);
            const NvTrace_RevisionedRef headChunkRef = pushPopRangeState.headChunkRef;

            // A V1 buffer carries a range's attributes as a chain record, while the generic format
            // carries them in the entry record's own payload -- so the chain is scanned for them before
            // the entry record is staged. The bytes are copied out under the chunk's own verdict, which
            // the entry's cursor cannot stand in for: it answers for the entry's revision, not for a
            // chunk a racing writer recycled out from under the scan.
            m_eventAttributesBytesBuffer.clear();
            bool eventAttributesConsistent = true;
            WalkStateChunkChain(
                headChunkRef,
                [&](NvTrace_RevisionedRef, uint32_t chunkRecordType, const uint8_t* pPayload, size_t payloadByteCount) {
                    if (chunkRecordType != NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_EventAttributes_V1)
                    {
                        return false;
                    }
                    m_eventAttributesBytesBuffer.assign(pPayload, pPayload + payloadByteCount);
                    return true;
                },
                [&](bool recordCommitted) { eventAttributesConsistent = recordCommitted; });

            if (!eventAttributesConsistent)
            {
                m_eventAttributesBytesBuffer.clear();
            }

            const bool carriesStringHandle = pushPopRangeState.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString;
            const uint32_t inlineMessageCapacity = m_sections.pProviderBufferLayout->pushPopRangeInlineMessageSize;
            NVTRACE_LOG_ERR_RETURN_IF(carriesStringHandle && (pushPopRangeState.messageSize < sizeof(uint64_t) || inlineMessageCapacity < sizeof(uint64_t)), , 50, "Invalid V1 push-pop range string handle (length=%u capacity=%u).", pushPopRangeState.messageSize, inlineMessageCapacity);

            NvTrace_NvtxTrace_PushPopRangeCreated record;
            record.startTimestampTsc = pushPopRangeState.startTimestampTsc;
            record.endTimestampTsc = LoadRelaxed(&pushPopRangeState.endTimestampTsc);
            record.ancestorRef = TranslateV1Ref(pushPopRangeState.ancestorRef);
            record.uniqueThreadId = pushPopRangeState.uniqueThreadId;
            record.domainIndex = pushPopRangeState.domainIndex;
            record.nestingDepth = pushPopRangeState.nestingDepth;
            record.messageType = pushPopRangeState.messageType;
            record.totalMessageLength = carriesStringHandle ? 0 : pushPopRangeState.messageSize;
            record.inlineMessageSegmentLength = std::min(record.totalMessageLength, inlineMessageCapacity);
            record.hasEventAttributes = static_cast<uint8_t>(m_eventAttributesBytesBuffer.empty() ? 0 : 1);

            StagePushPopRangeRecordBytes(record, m_eventAttributesBytesBuffer.data(), m_eventAttributesBytesBuffer.size(), pEntryBytes + m_sections.pEventVersionInfo->pushPopRangeStateSize);
            fnOnRecordBytes(entryRef, NvTrace_NvtxTrace_RecordType_PushPopRangeCreated, m_recordBytesBuffer.data(), m_recordBytesBuffer.size());

            const bool entryDelivered = cursor.CheckConsistency();
            fnCommitRecord(entryDelivered);

            // An entry a racing writer disturbed carries nothing behind it: the chain the slot's next
            // occupant hangs off it is not this range's.
            if (!entryDelivered || carriesStringHandle)
            {
                return;
            }

            uint32_t messageByteOffset = record.inlineMessageSegmentLength;
            WalkStateChunkChain(
                headChunkRef,
                [&](NvTrace_RevisionedRef recordCorrelationId, uint32_t chunkRecordType, const uint8_t* pPayload, size_t payloadByteCount) {
                    if (chunkRecordType == NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_MessageSegment_V1)
                    {
                        // A V1 segment's payload is padded to 8 bytes, so the last one runs past the
                        // message; the V2 record counts message bytes alone.
                        const uint32_t remainingMessageByteCount = messageByteOffset < record.totalMessageLength ? record.totalMessageLength - messageByteOffset : 0;
                        if (remainingMessageByteCount == 0)
                        {
                            return true;
                        }

                        NvTrace_NvtxTrace_PushPopRangeMessageSegment messageSegment;
                        messageSegment.pushPopRangeRef = entryRef;
                        messageSegment.messageOffset = messageByteOffset;
                        messageSegment.messageSegmentLength = std::min(static_cast<uint32_t>(payloadByteCount), remainingMessageByteCount);
                        messageByteOffset += messageSegment.messageSegmentLength;

                        StageRecordBytes(&messageSegment, sizeof(messageSegment), pPayload, messageSegment.messageSegmentLength);
                        fnOnRecordBytes(recordCorrelationId, NvTrace_NvtxTrace_RecordType_PushPopRangeMessageSegment, m_recordBytesBuffer.data(), m_recordBytesBuffer.size());
                        return true;
                    }

                    return false;
                },
                fnCommitRecord);
        }

        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadStackGroupGroupRecordBytes(uint32_t entryIndex, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pStackGroupStates || m_sections.pProviderBufferLayout->stackGroupStatesCount <= entryIndex)
            {
                return;
            }

            const uint8_t* const pEntryBytes = m_sections.pStackGroupStates + m_stackGroupStateStride * entryIndex;
            const auto& stackGroupState = *reinterpret_cast<const NvTrace_NvtxTrace_StackGroupState_V1*>(pEntryBytes);

            const uint64_t revisionBefore = LoadAcquire(&stackGroupState.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return;
            }

            const RevisionCursor cursor{&stackGroupState.revision, revisionBefore};
            const NvTrace_RevisionedRef entryRef = EncodeStackGroupRef(revisionBefore, entryIndex);
            const NvTrace_RevisionedRef headChunkRef = stackGroupState.headChunkRef;

            NvTrace_NvtxTrace_StackGroupCreated record = {};
            record.totalStackRefCount = stackGroupState.activeDomainCount;
            record.inlineStackRefCount = std::min(stackGroupState.activeDomainCount, m_sections.pProviderBufferLayout->stackGroupInlineStackRefCount);

            StageStackGroupRecordBytes(record, pEntryBytes + m_sections.pEventVersionInfo->stackGroupStateSize);
            fnOnRecordBytes(entryRef, NvTrace_NvtxTrace_RecordType_StackGroupCreated, m_recordBytesBuffer.data(), m_recordBytesBuffer.size());

            const bool entryDelivered = cursor.CheckConsistency();
            fnCommitRecord(entryDelivered);

            if (!entryDelivered)
            {
                return;
            }

            WalkStateChunkChain(
                headChunkRef,
                [&](NvTrace_RevisionedRef recordCorrelationId, uint32_t chunkRecordType, const uint8_t* pPayload, size_t payloadByteCount) {
                    if (chunkRecordType != NvTrace_NvtxTrace_StackGroupStateChunkRecordType_StackRef_V1)
                    {
                        return false;
                    }

                    StageStackGroupStackRefsBytes(entryRef, pPayload, payloadByteCount);
                    fnOnRecordBytes(recordCorrelationId, NvTrace_NvtxTrace_RecordType_StackGroupStackRefs, m_recordBytesBuffer.data(), m_recordBytesBuffer.size());
                    return true;
                },
                fnCommitRecord);
        }

        // Eager expansion of the created announce the old driver sends: delivers the named entry's whole group.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ResolveRecordReference(const NvTrace_RecordReference& reference, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            switch (static_cast<NvTrace_NvtxTrace_RecordType>(reference.sectionIndex))
            {
                case NvTrace_NvtxTrace_RecordType_StringRegistered:
                    ReadStringRecordBytes(reference.recordRef, fnReadRecord, fnCommitRecord);
                    return;
                case NvTrace_NvtxTrace_RecordType_DomainCreated:
                    if (reference.recordRef <= std::numeric_limits<uint32_t>::max())
                    {
                        ReadDomainRecordBytes(static_cast<uint32_t>(reference.recordRef), fnReadRecord, fnCommitRecord);
                    }
                    return;
                case NvTrace_NvtxTrace_RecordType_PushPopRangeCreated:
                {
                    const uint32_t entryIndex = DecodeEntryIndex(m_pushPopRangeRefCodec, reference.recordRef);
                    ResolveEntryGroupReference([&](auto&& fnOnRecordBytes, auto&& fnCommit) { ReadPushPopRangeGroupRecordBytes(entryIndex, fnOnRecordBytes, fnCommit); }, reference.recordRef, fnReadRecord, fnCommitRecord);
                    return;
                }
                case NvTrace_NvtxTrace_RecordType_StackGroupCreated:
                {
                    const uint32_t entryIndex = DecodeEntryIndex(m_stackGroupRefCodec, reference.recordRef);
                    ResolveEntryGroupReference([&](auto&& fnOnRecordBytes, auto&& fnCommit) { ReadStackGroupGroupRecordBytes(entryIndex, fnOnRecordBytes, fnCommit); }, reference.recordRef, fnReadRecord, fnCommitRecord);
                    return;
                }
                default:
                    NVTRACE_LOG_ERR(50, "Unknown V1 record reference section index %u.", reference.sectionIndex);
                    return;
            }
        }

        NvTrace_RevisionedRef EncodePushPopRangeRef(uint64_t revision, uint32_t entryIndex) const
        {
            return m_pushPopRangeRefCodec.EncodeItemRef(entryIndex, revision);
        }

        NvTrace_RevisionedRef EncodeStackGroupRef(uint64_t revision, uint32_t entryIndex) const
        {
            return m_stackGroupRefCodec.EncodeItemRef(entryIndex, revision);
        }

    private:
        static NvTrace_RevisionedRef TranslateV1Ref(uint64_t v1Ref)
        {
            return v1Ref == NvTrace_NvtxTrace_INVALID_REF_V1 ? NVTRACE_INVALID_REVISIONED_REF : v1Ref;
        }

        // Precondition: pStagedStackRef is zeroed, so a shorter V1 stack ref leaves the bytes past it 0.
        void StageStackRefBytesAt(uint8_t* pStagedStackRef, const uint8_t* pStackRef, size_t stackRefByteCount) const
        {
            memcpy(pStagedStackRef, pStackRef, std::min(stackRefByteCount, sizeof(NvTrace_NvtxTrace_StackRef)));

            auto* const pStagedStackRefTyped = reinterpret_cast<NvTrace_NvtxTrace_StackRef*>(pStagedStackRef);
            pStagedStackRefTyped->topRef = TranslateV1Ref(pStagedStackRefTyped->topRef);
        }

        // A V1 chain carries one stack ref per chunk record, so the V2 record staged here always holds one.
        void StageStackGroupStackRefsBytes(NvTrace_RevisionedRef stackGroupRef, const uint8_t* pStackRef, size_t stackRefByteCount)
        {
            m_recordBytesBuffer.assign(sizeof(NvTrace_NvtxTrace_StackGroupStackRefs) + sizeof(NvTrace_NvtxTrace_StackRef), 0);

            auto* const pStackRefsRecord = reinterpret_cast<NvTrace_NvtxTrace_StackGroupStackRefs*>(m_recordBytesBuffer.data());
            pStackRefsRecord->stackGroupRef = stackGroupRef;
            pStackRefsRecord->stackRefCount = 1;

            StageStackRefBytesAt(reinterpret_cast<uint8_t*>(pStackRefsRecord + 1), pStackRef, stackRefByteCount);
        }

        void StageRecordBytes(const void* pRecord, size_t recordByteCount, const uint8_t* pTailBytes, size_t tailByteCount)
        {
            m_recordBytesBuffer.assign(NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8(recordByteCount + tailByteCount) * sizeof(uint64_t), 0);
            memcpy(m_recordBytesBuffer.data(), pRecord, recordByteCount);
            memcpy(m_recordBytesBuffer.data() + recordByteCount, pTailBytes, tailByteCount);
        }

        // [PushPopRangeCreated][EventAttributes only if hasEventAttributes][inline message bytes], the
        // attributes at the eventAttributesSize this shim publishes -- the V1 buffer's own -- so a
        // shorter V1 struct leaves the bytes past it 0 rather than shifting the message.
        void StagePushPopRangeRecordBytes(const NvTrace_NvtxTrace_PushPopRangeCreated& record, const uint8_t* pEventAttributes, size_t eventAttributesByteCount, const uint8_t* pInlineMessage)
        {
            const size_t attributesExtent = record.hasEventAttributes != 0 ? m_sections.pEventVersionInfo->eventAttributesSize : 0;
            const size_t inlinePayloadSize = record.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString ? sizeof(uint64_t) : record.inlineMessageSegmentLength;
            m_recordBytesBuffer.assign(NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8(sizeof(record) + attributesExtent + inlinePayloadSize) * sizeof(uint64_t), 0);

            memcpy(m_recordBytesBuffer.data(), &record, NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A);
            if (attributesExtent != 0)
            {
                memcpy(m_recordBytesBuffer.data() + sizeof(record), pEventAttributes, std::min(eventAttributesByteCount, attributesExtent));
            }
            memcpy(m_recordBytesBuffer.data() + sizeof(record) + attributesExtent, pInlineMessage, inlinePayloadSize);
        }

        void StageStackGroupRecordBytes(const NvTrace_NvtxTrace_StackGroupCreated& record, const uint8_t* pInlineStackRefs)
        {
            const size_t stackRefSize = m_sections.pEventVersionInfo->stackRefSize;

            m_recordBytesBuffer.assign(sizeof(record) + static_cast<size_t>(record.inlineStackRefCount) * sizeof(NvTrace_NvtxTrace_StackRef), 0);
            memcpy(m_recordBytesBuffer.data(), &record, sizeof(record));
            for (uint32_t inlineStackRefIndex = 0; inlineStackRefIndex < record.inlineStackRefCount; ++inlineStackRefIndex)
            {
                StageStackRefBytesAt(m_recordBytesBuffer.data() + sizeof(record) + static_cast<size_t>(inlineStackRefIndex) * sizeof(NvTrace_NvtxTrace_StackRef), pInlineStackRefs + static_cast<size_t>(inlineStackRefIndex) * stackRefSize, stackRefSize);
            }
        }

        // 0 when the offset names no written entry: an unwritten entry's first byte is a NUL.
        size_t StageStringRecordBytes(uint64_t stringTableEntryOffset)
        {
            if (!m_sections.pStringTable || stringTableEntryOffset == 0 || m_sections.pProviderBufferLayout->stringTableSize <= stringTableEntryOffset)
            {
                return 0;
            }

            const char* const pStringBytes = m_sections.pStringTable + stringTableEntryOffset;
            const size_t remainingByteCount = static_cast<size_t>(m_sections.pProviderBufferLayout->stringTableSize - stringTableEntryOffset);
            const void* const pTerminator = memchr(pStringBytes, '\0', remainingByteCount);
            const size_t byteCount = pTerminator == nullptr ? remainingByteCount : static_cast<size_t>(static_cast<const char*>(pTerminator) - pStringBytes);
            if (byteCount == 0)
            {
                return 0;
            }

            NvTrace_NvtxTrace_StringRegistered record = {};
            record.stringHandle = stringTableEntryOffset;
            record.totalStringLength = static_cast<uint32_t>(byteCount);
            record.inlineStringSegmentLength = static_cast<uint32_t>(byteCount);

            StageRecordBytes(&record, sizeof(record), reinterpret_cast<const uint8_t*>(pStringBytes), byteCount);
            return byteCount;
        }

        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        size_t DeliverStringRecordBytes(uint64_t stringTableEntryOffset, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            const size_t byteCount = StageStringRecordBytes(stringTableEntryOffset);
            if (byteCount == 0)
            {
                return 0;
            }

            fnOnRecordBytes(stringTableEntryOffset, NvTrace_NvtxTrace_RecordType_StringRegistered, m_recordBytesBuffer.data(), m_recordBytesBuffer.size());

            // The V1 string table is append-only: a published entry is never rewritten, so a read of
            // one is always consistent.
            fnCommitRecord(true);
            return byteCount;
        }

        // The walk stops at the first failed verdict, and at the first chunk a racing writer recycled or
        // malformed. fnOnChunkRecord(recordCorrelationId, recordType, pPayload, payloadByteCount) takes
        // the record's own address, as the V2 chain reader gives it, and returns false for a record type
        // this build does not carry over -- delivered to nobody, so it takes no verdict.
        template <typename TFN_OnChunkRecord, typename TFN_CommitRecord>
        void WalkStateChunkChain(NvTrace_RevisionedRef headChunkRef, TFN_OnChunkRecord&& fnOnChunkRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            if (!m_sections.pStateChunks)
            {
                return;
            }

            const uint32_t stateChunkHeaderSize = m_sections.pEventVersionInfo->stateChunkHeaderSize;
            const uint32_t stateChunkRecordSize = m_sections.pEventVersionInfo->stateChunkRecordSize;

            NvTrace_RevisionedRef chunkRef = headChunkRef;
            while (chunkRef != NVTRACE_INVALID_REVISIONED_REF)
            {
                const RevisionedRefCodec::DecodedItemRef decodedChunkRef = m_stateChunkRefCodec.DecodeItemRef(chunkRef);
                if (m_sections.pProviderBufferLayout->stateChunksCount <= decodedChunkRef.itemIndex)
                {
                    return;
                }

                const uint8_t* const pChunk = m_sections.pStateChunks + decodedChunkRef.itemIndex * m_sections.pProviderBufferLayout->stateChunkSize;
                const auto& chunkHeader = *reinterpret_cast<const NvTrace_AdaptiveChunkHeader*>(pChunk);

                const NvTrace_RevisionedRef nextChunkRef = LoadAcquire(&chunkHeader.nextChunkRef);
                const uint32_t recordCount = LoadAcquire(&chunkHeader.recordCount);
                if (m_maxRecordCountPerStateChunk < recordCount)
                {
                    return;
                }

                ReadFence();
                if (m_stateChunkRefCodec.TruncateItemRevision(LoadRelaxed(&chunkHeader.revision)) != decodedChunkRef.revision)
                {
                    return;
                }

                // V1 payload offsets are chunk-start-relative (see NvTrace_NvtxTrace_ProviderBufferLayout_V1);
                // a payload ends where the previous record's begins, the first at the chunk's end.
                const size_t minPayloadOffset = stateChunkHeaderSize + static_cast<size_t>(recordCount) * stateChunkRecordSize;
                size_t previousPayloadOffset = m_sections.pProviderBufferLayout->stateChunkSize;
                for (uint32_t recordIndex = 0; recordIndex < recordCount; ++recordIndex)
                {
                    const auto& chunkRecord = *reinterpret_cast<const NvTrace_RecordDesc*>(pChunk + stateChunkHeaderSize + static_cast<size_t>(recordIndex) * stateChunkRecordSize);
                    const size_t payloadOffset = static_cast<size_t>(chunkRecord.payloadEndOffsetDiv8) * 8;
                    if (payloadOffset < minPayloadOffset || previousPayloadOffset < payloadOffset)
                    {
                        return;
                    }

                    const bool recordDelivered = fnOnChunkRecord(m_stateChunkRefCodec.EncodeRecordRef(chunkRef, recordIndex), chunkRecord.recordType, pChunk + payloadOffset, previousPayloadOffset - payloadOffset);
                    if (recordDelivered)
                    {
                        ReadFence();
                        const bool consistencyCheckPassed = m_stateChunkRefCodec.TruncateItemRevision(LoadRelaxed(&chunkHeader.revision)) == decodedChunkRef.revision;
                        fnCommitRecord(consistencyCheckPassed);
                        if (!consistencyCheckPassed)
                        {
                            return;
                        }
                    }

                    previousPayloadOffset = payloadOffset;
                }

                chunkRef = nextChunkRef;
            }
        }

        // The index is masked to the codec's field width, so the table's range check answers for the value.
        static uint32_t DecodeEntryIndex(const RevisionedRefCodec& refCodec, NvTrace_RevisionedRef entryRef)
        {
            return static_cast<uint32_t>(refCodec.DecodeItemRef(entryRef).itemIndex);
        }

        // A resolve validates in ref space -- an entry's ref carries the revision it was read at -- so a
        // stale announce, the entry modified or reassigned since the event, resolves to nothing. The
        // group is read by entry index, so the entry record -- the first delivered, and the only one
        // addressed by the entry's ref -- is what that check is made against; the chain records behind
        // it, each addressed in its own chunk, ride the same decision.
        template <typename TFN_ReadEntryGroup, typename TFN_ReadRecord, typename TFN_CommitRecord>
        static void ResolveEntryGroupReference(TFN_ReadEntryGroup&& fnReadEntryGroup, NvTrace_RevisionedRef entryRef, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            bool entryRecordDelivered = false;
            bool entryMatched = false;
            fnReadEntryGroup(
                [&](NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount) {
                    if (!entryRecordDelivered)
                    {
                        entryRecordDelivered = true;
                        entryMatched = recordCorrelationId == entryRef;
                    }

                    if (entryMatched)
                    {
                        fnReadRecord(recordCorrelationId, recordType, pRecordBytes, recordByteCount);
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (entryMatched)
                    {
                        fnCommitRecord(consistencyCheckPassed);
                    }
                });
        }

        bool TranslateMarkEvent(const ThreadEventRecord& record, ThreadEventRecord& out_record) const
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_sections.pEventVersionInfo, false, 50, "Cannot translate a V1 Mark without its version info.");

            const auto& eventVersionInfo = *m_sections.pEventVersionInfo;
            const size_t markSize = eventVersionInfo.markSize;
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.markMinorVersion < NvTrace_NvtxTrace_Mark_V1_MINOR_VERSION_A || markSize < NvTrace_NvtxTrace_Mark_V1_MINOR_VERSION_A || record.payloadSize < markSize, false, 50, "Invalid V1 Mark header (minor=%u size=%zu payload=%zu).", eventVersionInfo.markMinorVersion, markSize, record.payloadSize);

            NvTrace_NvtxTrace_Mark_V1 legacyMark = {};
            memcpy(&legacyMark, record.pPayload, NvTrace_NvtxTrace_Mark_V1_MINOR_VERSION_A);
            const size_t attributesOffset = AlignUpP2(markSize, alignof(NvTrace_NvtxTrace_EventAttributes));
            const size_t attributesSize = legacyMark.hasEventAttributes != 0 ? eventVersionInfo.eventAttributesSize : 0;
            const size_t messageOffset = legacyMark.hasEventAttributes != 0 ? attributesOffset + attributesSize : markSize;
            NVTRACE_LOG_ERR_RETURN_IF(record.payloadSize < messageOffset || record.payloadSize - messageOffset < legacyMark.messageSize, false, 50, "Invalid V1 Mark message (offset=%zu length=%u payload=%zu).", messageOffset, legacyMark.messageSize, record.payloadSize);

            const bool carriesStringHandle = legacyMark.messageType == NvTrace_NvtxTrace_MessageType_RegisteredString;
            NVTRACE_LOG_ERR_RETURN_IF(carriesStringHandle && legacyMark.messageSize < sizeof(uint64_t), false, 50, "Invalid V1 Mark string handle (length=%u).", legacyMark.messageSize);
            const size_t inlinePayloadSize = carriesStringHandle ? sizeof(uint64_t) : legacyMark.messageSize;
            const size_t payloadSize = sizeof(NvTrace_NvtxTrace_Mark) + attributesSize + inlinePayloadSize;
            m_markPayloadBuffer.assign(AlignUpP2(payloadSize, sizeof(uint64_t)) / sizeof(uint64_t), 0);
            auto* const pPayload = reinterpret_cast<uint8_t*>(m_markPayloadBuffer.data());

            auto& mark = *reinterpret_cast<NvTrace_NvtxTrace_Mark*>(pPayload);
            mark.timestampTsc = legacyMark.timestampTsc;
            mark.domainIndex = legacyMark.domainIndex;
            mark.totalMessageLength = carriesStringHandle ? 0 : legacyMark.messageSize;
            mark.inlineMessageSegmentLength = mark.totalMessageLength;
            mark.messageType = legacyMark.messageType;
            mark.hasEventAttributes = legacyMark.hasEventAttributes;

            if (attributesSize != 0)
            {
                memcpy(pPayload + sizeof(mark), record.pPayload + attributesOffset, attributesSize);
            }

            memcpy(pPayload + sizeof(mark) + attributesSize, record.pPayload + messageOffset, inlinePayloadSize);

            out_record = record;
            out_record.recordType = NvTrace_NvtxTrace_RecordType_Mark;
            out_record.pPayload = pPayload;
            out_record.payloadSize = payloadSize;
            return true;
        }

        bool TranslateObjectLifetimeEvent(const ThreadEventRecord& record, NvtxTraceTranslatedRecordV1& out_translated) const
        {
            const NvTrace_NvtxTrace_RecordType createdRecordType = TranslateNvtxTraceV1ObjectLifetimeCreatedRecordType(record.recordType);
            if (createdRecordType != NvTrace_NvtxTrace_RecordType_INVALID)
            {
                return TranslateObjectCreatedEvent(record, createdRecordType, out_translated);
            }

            return TranslateObjectDestroyedEvent(record, out_translated);
        }

        // A created announce carries nothing but its event header: the object's state is read from the
        // table on resolve.
        bool TranslateObjectCreatedEvent(const ThreadEventRecord& record, NvTrace_NvtxTrace_RecordType createdRecordType, NvtxTraceTranslatedRecordV1& out_translated) const
        {
            const bool refPacksRevision = createdRecordType == NvTrace_NvtxTrace_RecordType_PushPopRangeCreated || createdRecordType == NvTrace_NvtxTrace_RecordType_StackGroupCreated;
            const size_t requiredPayloadSize = refPacksRevision ? NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1, tableEntryRevision) : NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1, tableEntryIndex);

            NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1 header = {};
            const bool headerRead = TryReadObjectLifetimeEventHeader(record, requiredPayloadSize, header);
            if (!headerRead)
            {
                return false;
            }

            NvTrace_RevisionedRef entryRecordRef;
            const bool entryRecordRefMade = TryMakeCreatedEntryRecordRef(createdRecordType, header, entryRecordRef);
            if (!entryRecordRefMade)
            {
                return false;
            }

            MakeCreatedReference(createdRecordType, entryRecordRef, out_translated.payload.reference);

            out_translated.recordType = NvTrace_ReservedRecordType_RecordReference;
            out_translated.payloadSize = sizeof(NvTrace_RecordReference);
            return true;
        }

        bool TranslateObjectDestroyedEvent(const ThreadEventRecord& record, NvtxTraceTranslatedRecordV1& out_translated) const
        {
            NVTRACE_LOG_ERR_RETURN_IF(record.recordType != NvTrace_NvtxTrace_EventType_ObjectLifetime_StackGroupDestroyed_V1, false, 50, "Unknown V1 NVTX object-lifetime event type %u.", record.recordType);

            // The destroyed record carries the ref itself, so the announce that mints it must carry the
            // revision the ref packs.
            NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1 header = {};
            const bool headerRead = TryReadObjectLifetimeEventHeader(record, NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1, tableEntryRevision), header);
            if (!headerRead)
            {
                return false;
            }

            NVTRACE_LOG_ERR_RETURN_IF(std::numeric_limits<uint32_t>::max() < header.tableEntryIndex, false, 50, "V1 NVTX stack-group-destroyed ThreadEvent record names an out-of-range table entry (tableEntryIndex=%" PRIu64 ").", header.tableEntryIndex);

            out_translated.recordType = NvTrace_NvtxTrace_RecordType_StackGroupDestroyed;
            out_translated.payloadSize = sizeof(NvTrace_NvtxTrace_StackGroupDestroyed);
            out_translated.payload.stackGroupDestroyed = {header.timestampTsc, EncodeStackGroupRef(header.tableEntryRevision, static_cast<uint32_t>(header.tableEntryIndex))};
            return true;
        }

        // A V1 object-lifetime record is its header and nothing else.
        static bool TryReadObjectLifetimeEventHeader(const ThreadEventRecord& record, size_t requiredPayloadSize, NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1& out_header)
        {
            NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < requiredPayloadSize, false, 50, "Invalid V1 NVTX object-lifetime ThreadEvent record (recordType=%u, payloadSize=%zu, expected %zu at least).", record.recordType, record.payloadSize, requiredPayloadSize);

            memcpy(&out_header, record.pPayload, std::min(sizeof(out_header), record.payloadSize));
            return true;
        }

        // Neither the string table nor the domain table recycles an entry, so their refs carry no revision.
        bool TryMakeCreatedEntryRecordRef(NvTrace_NvtxTrace_RecordType createdRecordType, const NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1& header, NvTrace_RevisionedRef& out_entryRecordRef) const
        {
            if (createdRecordType == NvTrace_NvtxTrace_RecordType_StringRegistered || createdRecordType == NvTrace_NvtxTrace_RecordType_DomainCreated)
            {
                out_entryRecordRef = header.tableEntryIndex;
                return true;
            }

            NVTRACE_LOG_ERR_RETURN_IF(std::numeric_limits<uint32_t>::max() < header.tableEntryIndex, false, 50, "V1 NVTX object-lifetime ThreadEvent record names a table entry index wider than the shim's ref currency carries (recordType=%u, tableEntryIndex=%" PRIu64 ").", static_cast<uint32_t>(createdRecordType), header.tableEntryIndex);

            const uint32_t tableEntryIndex = static_cast<uint32_t>(header.tableEntryIndex);
            if (createdRecordType == NvTrace_NvtxTrace_RecordType_PushPopRangeCreated)
            {
                out_entryRecordRef = EncodePushPopRangeRef(header.tableEntryRevision, tableEntryIndex);
            }
            else // if (createdRecordType == NvTrace_NvtxTrace_RecordType_StackGroupCreated)
            {
                out_entryRecordRef = EncodeStackGroupRef(header.tableEntryRevision, tableEntryIndex);
            }
            return true;
        }

        // A V1 buffer has no section indices, so the shim's private sectionIndex vocabulary is the
        // created record type itself.
        static void MakeCreatedReference(NvTrace_NvtxTrace_RecordType createdRecordType, NvTrace_RevisionedRef entryRecordRef, NvTrace_RecordReference& out_reference)
        {
            out_reference = {};
            out_reference.sectionIndex = createdRecordType;
            out_reference.recordRef = entryRecordRef;
        }

        // This build's values for the records the shim converts or synthesizes; the producer's V1 values
        // for the payloads that pass through verbatim, with the stack ref clamped to what the shim
        // stages inline.
        static NvTrace_NvtxTrace_RecordVersionInfo MakeRecordVersionInfoFromV1(const NvTrace_NvtxTrace_EventVersionInfo_V1& eventVersionInfo)
        {
            NvTrace_NvtxTrace_RecordVersionInfo recordVersionInfo;
            FillNvtxTraceRecordVersionInfo(recordVersionInfo);

            recordVersionInfo.eventAttributesSize = eventVersionInfo.eventAttributesSize;
            recordVersionInfo.eventAttributesMinorVersion = eventVersionInfo.eventAttributesMinorVersion;
            recordVersionInfo.stackRefSize = std::min<uint32_t>(eventVersionInfo.stackRefSize, sizeof(NvTrace_NvtxTrace_StackRef));
            recordVersionInfo.stackRefMinorVersion = std::min<uint32_t>(eventVersionInfo.stackRefMinorVersion, NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A);

            recordVersionInfo.rangePopSize = eventVersionInfo.rangePopSize;
            recordVersionInfo.rangePopMinorVersion = eventVersionInfo.rangePopMinorVersion;
            recordVersionInfo.droppedRangePushSize = eventVersionInfo.droppedRangePushSize;
            recordVersionInfo.droppedRangePushMinorVersion = eventVersionInfo.droppedRangePushMinorVersion;
            recordVersionInfo.droppedRangePopSize = eventVersionInfo.droppedRangePopSize;
            recordVersionInfo.droppedRangePopMinorVersion = eventVersionInfo.droppedRangePopMinorVersion;

            return recordVersionInfo;
        }

    private:
        struct ProviderBufferSections
        {
            const NvTrace_ProviderBufferHeader_V1* pProviderBufferHeader = nullptr;
            const NvTrace_NvtxTrace_ProviderBufferLayout_V1* pProviderBufferLayout = nullptr;
            const NvTrace_NvtxTrace_EventVersionInfo_V1* pEventVersionInfo = nullptr;
            const FB::Provider::NvtxTrace::ProviderConfiguration* fbProviderConfiguration = nullptr;
            const char* pStringTable = nullptr;
            const uint8_t* pDomainStates = nullptr;
            const uint8_t* pStateChunks = nullptr;
            const uint8_t* pPushPopRangeStates = nullptr;
            const uint8_t* pStackGroupStates = nullptr;
        };

        ProviderBufferSections m_sections;
        NvTrace_NvtxTrace_RecordVersionInfo m_recordVersionInfo = {};

        uint32_t m_maxRecordCountPerStateChunk = 0;
        RevisionedRefCodec m_stateChunkRefCodec;

        size_t m_pushPopRangeStateStride = 0;
        RevisionedRefCodec m_pushPopRangeRefCodec;

        size_t m_stackGroupStateStride = 0;
        RevisionedRefCodec m_stackGroupRefCodec;

        std::vector<uint8_t> m_recordBytesBuffer;

        std::vector<uint8_t> m_eventAttributesBytesBuffer;

        mutable std::vector<uint64_t> m_markPayloadBuffer;
    };

} // namespace nv::trace
