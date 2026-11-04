/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceRevisionedRefCodec.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nv::trace {

    // Reader of a State Arena section: a monotonic bump arena of variable-size records, published in
    // one forward-growing desc array whose payloads stack backward from the arena's high edge, so
    // consecutive records' payloadEndOffsetDiv8 values bracket each payload. A payload may not reach
    // into the published desc array, which bounds it more tightly than the arena does and is the same
    // bound the producer allocated against.
    //
    // An arena is a snapshot, drained once in full at attach and never rescanned; anything registered
    // afterwards reaches a consumer as its own record reference. A record whose recordType is still 0 was
    // reserved but not yet committed, and the producer announces it by reference moments later.
    //
    // Every record stands alone, both to the drain and to a ref read. A producer spreading one logical
    // object over several records announces each of them, and makes each carry what ties it back, so
    // the arena never has to model the relation.
    //
    // The section's revision gates the whole arena the way an entry's revision gates one state table
    // entry: a reset moves it, so every ref taken before the reset resolves to nothing. Revision 0 reads
    // as never initialized. fnCommitRecord(bool) follows every record; false means the arena was reset
    // under the read, so the tentative record must be discarded and the rest of the walk abandoned.
    //
    // Payloads are delivered where they live, never staged, and sized in whole words. A producer may
    // store an unset->set 64-bit field into a published payload in place, moving no revision, so a
    // consumer reads such a payload through fields of its own type or copies it out with Memcpy64.
    //
    // Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class StateArenaReader
    {
    public:
        ~StateArenaReader() = default;
        StateArenaReader() = default;

        StateArenaReader(const StateArenaReader&) = delete;
        StateArenaReader(StateArenaReader&& rhs) noexcept
            : m_pState(std::exchange(rhs.m_pState, nullptr))
            , m_pBeginRecordDescPos(std::exchange(rhs.m_pBeginRecordDescPos, nullptr))
            , m_pBeginPayloadPos(std::exchange(rhs.m_pBeginPayloadPos, nullptr))
            , m_recordDescSize(std::exchange(rhs.m_recordDescSize, 0))
            , m_maxPayloadSize(std::exchange(rhs.m_maxPayloadSize, 0))
            , m_maxRecordCount(std::exchange(rhs.m_maxRecordCount, 0))
            , m_refCodec(std::move(rhs.m_refCodec))
        {
        }

        StateArenaReader& operator=(const StateArenaReader&) = delete;
        StateArenaReader& operator=(StateArenaReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pState = std::exchange(rhs.m_pState, nullptr);
                m_pBeginRecordDescPos = std::exchange(rhs.m_pBeginRecordDescPos, nullptr);
                m_pBeginPayloadPos = std::exchange(rhs.m_pBeginPayloadPos, nullptr);
                m_recordDescSize = std::exchange(rhs.m_recordDescSize, 0);
                m_maxPayloadSize = std::exchange(rhs.m_maxPayloadSize, 0);
                m_maxRecordCount = std::exchange(rhs.m_maxRecordCount, 0);
                m_refCodec = std::move(rhs.m_refCodec);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_refCodec.Reset();
            m_maxRecordCount = 0;
            m_maxPayloadSize = 0;
            m_recordDescSize = 0;
            m_pBeginPayloadPos = nullptr;
            m_pBeginRecordDescPos = nullptr;
            m_pState = nullptr;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t stateArenaSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            const auto [pSection, sectionSize, sectionHeaderSize, sectionHeaderMinorVersion] = fnResolveSection(stateArenaSectionIndex, NvTrace_ProviderBufferSectionType_StateArena);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, false, 50, "StateArenaReader: failed to resolve state-arena section (stateArenaSectionIndex %u)", stateArenaSectionIndex);

            SectionLayoutValidator sectionValidator(sectionSize);
            bool succeeded = sectionValidator.AddAndValidateSection(0, sectionHeaderSize, sectionHeaderMinorVersion, NvTrace_StateArenaSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateArenaReader: invalid section header layout (size %u, minorVersion %u)", sectionHeaderSize, sectionHeaderMinorVersion);

            const NvTrace_StateArenaSectionHeader* const pSectionHeader = reinterpret_cast<const NvTrace_StateArenaSectionHeader*>(pSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pSectionHeader) % alignof(NvTrace_StateArenaSectionHeader)) != 0, false, 50, "StateArenaReader: section header must be %zu-byte aligned (pSectionHeader %p)", alignof(NvTrace_StateArenaSectionHeader), pSectionHeader);

            succeeded = sectionValidator.AddAndValidateSection(pSectionHeader->stateArenaStateOffsetInSection, pSectionHeader->stateArenaStateSize, pSectionHeader->stateArenaStateMinorVersion, NvTrace_StateArenaState_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateArenaReader: invalid arena state layout (offset %" PRIu64 ", size %u, minorVersion %u)", pSectionHeader->stateArenaStateOffsetInSection, pSectionHeader->stateArenaStateSize, pSectionHeader->stateArenaStateMinorVersion);

            const NvTrace_StateArenaState* const pState = reinterpret_cast<const NvTrace_StateArenaState*>(pSection + pSectionHeader->stateArenaStateOffsetInSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pState) % alignof(NvTrace_StateArenaState)) != 0, false, 50, "StateArenaReader: arena state must be %zu-byte aligned (pState %p)", alignof(NvTrace_StateArenaState), pState);

            succeeded = sectionValidator.AddAndValidateSection(pSectionHeader->recordArenaOffsetInSection, pSectionHeader->recordArenaSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateArenaReader: invalid record arena layout (offset %" PRIu64 ", size %" PRIu64 ")", pSectionHeader->recordArenaOffsetInSection, pSectionHeader->recordArenaSize);

            const uint8_t* const pRecordArena = pSection + pSectionHeader->recordArenaOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pRecordArena) % alignof(uint64_t)) != 0 || (pSectionHeader->recordArenaSize % alignof(uint64_t)) != 0, false, 50, "StateArenaReader: the record arena must be %zu-byte aligned in both base and size (pRecordArena %p, recordArenaSize %" PRIu64 ")", alignof(uint64_t), pRecordArena, pSectionHeader->recordArenaSize);

            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->recordDescMinorVersion < NvTrace_RecordDesc_MINOR_VERSION_A || pSectionHeader->recordDescSize < sizeof(NvTrace_RecordDesc) || (pSectionHeader->recordDescSize % alignof(NvTrace_RecordDesc)) != 0, false, 50, "StateArenaReader: invalid record desc struct (recordDescSize %u, recordDescMinorVersion %u, required %zu)", pSectionHeader->recordDescSize, pSectionHeader->recordDescMinorVersion, sizeof(NvTrace_RecordDesc));

            NVTRACE_LOG_ERR_RETURN_IF(UINT32_MAX < pSectionHeader->recordArenaSize / sizeof(uint64_t), false, 50, "StateArenaReader: recordArenaSize (%" PRIu64 ") exceeds the record descriptor's payload offset capacity", pSectionHeader->recordArenaSize);

            // The producer publishes the index space rather than leaving it to be re-derived here, but it
            // can name no more records than the arena could hold were it all descs and no payloads.
            const uint64_t maxDescCount = pSectionHeader->recordArenaSize / pSectionHeader->recordDescSize;
            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->maxRecordCount == 0 || maxDescCount < pSectionHeader->maxRecordCount, false, 50, "StateArenaReader: maxRecordCount (%u) must be non-zero and name no more records than the arena could hold were it all descs (recordArenaSize %" PRIu64 ", recordDescSize %u, maxDescCount %" PRIu64 ")", pSectionHeader->maxRecordCount, pSectionHeader->recordArenaSize, pSectionHeader->recordDescSize, maxDescCount);

            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->maxPayloadSizeDiv8 == 0 || (pSectionHeader->recordArenaSize - pSectionHeader->recordDescSize) / sizeof(uint64_t) < pSectionHeader->maxPayloadSizeDiv8, false, 50, "StateArenaReader: maxPayloadSizeDiv8 (%u) must be non-zero and leave room for its record's desc in the arena (recordArenaSize %" PRIu64 ", recordDescSize %u)", pSectionHeader->maxPayloadSizeDiv8, pSectionHeader->recordArenaSize, pSectionHeader->recordDescSize);

            m_pState = pState;
            m_pBeginRecordDescPos = pRecordArena;
            m_pBeginPayloadPos = pRecordArena + pSectionHeader->recordArenaSize;
            m_recordDescSize = pSectionHeader->recordDescSize;
            m_maxPayloadSize = pSectionHeader->maxPayloadSizeDiv8 * sizeof(uint64_t);
            m_maxRecordCount = pSectionHeader->maxRecordCount;
            m_refCodec.Initialize(pSectionHeader->maxRecordCount);

            return true;
        }

        uint32_t GetRecordCount() const
        {
            if (!m_pState)
            {
                return 0;
            }

            const uint32_t recordCount = LoadAcquire(&m_pState->recordCount);
            return m_maxRecordCount < recordCount ? m_maxRecordCount : recordCount;
        }

        std::pair<uint32_t, uint64_t> DecodeRecordRef(NvTrace_RevisionedRef recordRef) const
        {
            const auto [recordIndex, revision] = m_refCodec.DecodeItemRef(recordRef);
            return {static_cast<uint32_t>(recordIndex), revision};
        }

        // Drains the arena once, in full: every committed record, in record order, each followed by
        // fnCommitRecord(bool). Records are delivered one by one; a provider that spreads one logical
        // object over several of them announces each by reference and reassembles them from their own
        // payloads, so the arena neither carries nor recovers any grouping.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeAttachRecords(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            if (!m_pState)
            {
                return;
            }

            const uint64_t revision = LoadAcquire(&m_pState->revision);
            if (revision == 0 || (revision & NVTRACE_STATE_ARENA_REVISION_RESETTING_BIT) != 0)
            {
                return;
            }

            const uint64_t expectedRevision = m_refCodec.TruncateItemRevision(revision);

            const uint32_t recordCount = LoadAcquire(&m_pState->recordCount);
            if (m_maxRecordCount < recordCount)
            {
                return;
            }

            const size_t maxPayloadEndOffset = m_pBeginPayloadPos - (m_pBeginRecordDescPos + recordCount * m_recordDescSize);

            size_t prevPayloadEndOffset = 0;
            for (uint32_t recordIndex = 0; recordIndex < recordCount; ++recordIndex)
            {
                const NvTrace_RecordDesc* const pRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(m_pBeginRecordDescPos + recordIndex * m_recordDescSize);
                const uint32_t recordType = LoadAcquire(&pRecordDesc->recordType);
                const size_t payloadEndOffset = pRecordDesc->payloadEndOffsetDiv8 * sizeof(uint64_t);
                const size_t payloadSize = payloadEndOffset - prevPayloadEndOffset;

                // One subtraction check covers range and monotonicity via unsigned wrap (see NvTraceAdaptiveBufferReader.h)
                if (m_maxPayloadSize < payloadSize || maxPayloadEndOffset - prevPayloadEndOffset < payloadSize)
                {
                    break;
                }

                prevPayloadEndOffset = payloadEndOffset;

                // The watermark covers reservations, so a record still at recordType 0 has its extent but
                // not its payload; the acquire above pairs with CommitRecord's release. The drain strides
                // over it under its own extent, and the producer announces it by reference once it commits.
                if (recordType == 0)
                {
                    continue;
                }

                fnReadRecord(m_refCodec.EncodeItemRef(recordIndex, expectedRevision), recordType, m_pBeginPayloadPos - payloadEndOffset, payloadSize);

                ReadFence();
                const bool consistencyCheckPassed = m_refCodec.TruncateItemRevision(LoadRelaxed(&m_pState->revision)) == expectedRevision;
                fnCommitRecord(consistencyCheckPassed);
                if (!consistencyCheckPassed)
                {
                    break;
                }
            }
        }

        // A point read of the one record a ref names. A stale ref, an out-of-range index or a malformed
        // payload extent delivers nothing. The record needs no commit gate: a producer announces a
        // reference only once the record has committed, and the thread-event record that carried the
        // reference here is what orders the payload writes ahead of this read.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ReadRecord(NvTrace_RevisionedRef recordRef, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            if (!m_pState)
            {
                return;
            }

            const auto [recordIndex, expectedRevision] = DecodeRecordRef(recordRef);

            if (m_refCodec.TruncateItemRevision(LoadAcquire(&m_pState->revision)) != expectedRevision)
            {
                return;
            }

            const uint32_t recordCount = LoadAcquire(&m_pState->recordCount);
            if (m_maxRecordCount < recordCount || recordCount <= recordIndex)
            {
                return;
            }

            const size_t maxPayloadEndOffset = m_pBeginPayloadPos - (m_pBeginRecordDescPos + recordCount * m_recordDescSize);

            const uint8_t* const pRecordDescPos = m_pBeginRecordDescPos + recordIndex * m_recordDescSize;
            const NvTrace_RecordDesc* const pRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(pRecordDescPos);
            const uint32_t recordType = pRecordDesc->recordType;
            const size_t payloadEndOffset = pRecordDesc->payloadEndOffsetDiv8 * sizeof(uint64_t);
            const size_t prevPayloadEndOffset = recordIndex == 0 ? 0 : reinterpret_cast<const NvTrace_RecordDesc*>(pRecordDescPos - m_recordDescSize)->payloadEndOffsetDiv8 * sizeof(uint64_t);
            const size_t payloadSize = payloadEndOffset - prevPayloadEndOffset;

            // Both offsets are untrusted bytes read fresh, so unlike the sequential consumer -- whose
            // previous offset was validated on the previous iteration -- a point read must bound
            // prevPayloadEndOffset itself before subtracting from it; that subtraction then covers a payload
            // reaching into the record descs area and a non-monotone pair alike (which wraps payloadSize huge).
            if (m_maxPayloadSize < payloadSize || maxPayloadEndOffset < prevPayloadEndOffset || maxPayloadEndOffset - prevPayloadEndOffset < payloadSize)
            {
                return;
            }

            fnReadRecord(recordRef, recordType, m_pBeginPayloadPos - payloadEndOffset, payloadSize);

            ReadFence();
            fnCommitRecord(m_refCodec.TruncateItemRevision(LoadRelaxed(&m_pState->revision)) == expectedRevision);
        }

    private:
        const NvTrace_StateArenaState* m_pState = nullptr;

        const uint8_t* m_pBeginRecordDescPos = nullptr;
        const uint8_t* m_pBeginPayloadPos = nullptr;

        size_t m_recordDescSize = 0;

        size_t m_maxPayloadSize = 0;
        uint32_t m_maxRecordCount = 0;

        RevisionedRefCodec m_refCodec;
    };

} // namespace nv::trace
