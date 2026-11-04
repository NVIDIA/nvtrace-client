/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceRevisionedRefCodec.h>
#include <NvTrace/Target/Observer/NvTraceAdaptiveChunkPoolReader.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nv::trace {

    // Reader of a State Table section: fixed-stride, revision-protected entries, each one record plus
    // an optional chunk chain of additional records.
    //
    // A table is a snapshot, drained once in full at attach and never rescanned. One entry yields one
    // group -- its main record, then its chain records in order -- and fnCommitRecord(bool) follows
    // every record: false means a writer raced the read, so the tentative record must be discarded,
    // and the rest of the group is abandoned.
    //
    // Every record is delivered under its own address: the entry's ref for an entry record, a
    // (chunk, recordIndex) ref for a chain record. A chain record's address is never its entry's --
    // what a chain record belongs to is a field in its payload (see NvTrace_RecordDesc).
    //
    // Payloads are delivered where they live, never staged, and sized in whole words as their record
    // desc sizes them. A producer may store an unset->set 64-bit field into an entry in place, moving no
    // revision, so a consumer reads a payload through fields of its own type or copies it out with
    // Memcpy64 -- a plain memcpy can tear across such a store, and no consistency check can catch that,
    // since the revision never moves.
    //
    // Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class StateTableReader
    {
    public:
        ~StateTableReader() = default;
        StateTableReader() = default;

        StateTableReader(const StateTableReader&) = delete;
        StateTableReader(StateTableReader&& rhs) noexcept
            : m_pEntries(std::exchange(rhs.m_pEntries, nullptr))
            , m_entryCount(std::exchange(rhs.m_entryCount, 0))
            , m_entrySize(std::exchange(rhs.m_entrySize, 0))
            , m_refCodec(std::move(rhs.m_refCodec))
            , m_hasFixedRecordDesc(std::exchange(rhs.m_hasFixedRecordDesc, false))
            , m_fixedRecordDesc(std::exchange(rhs.m_fixedRecordDesc, {}))
            , m_variableRecordDescOffsetInEntry(std::exchange(rhs.m_variableRecordDescOffsetInEntry, NVTRACE_INVALID_OFFSET32))
            , m_payloadOffsetInEntry(std::exchange(rhs.m_payloadOffsetInEntry, 0))
            , m_maxPayloadSizeDiv8(std::exchange(rhs.m_maxPayloadSizeDiv8, 0))
            , m_nextChunkRefOffsetInEntry(std::exchange(rhs.m_nextChunkRefOffsetInEntry, NVTRACE_INVALID_OFFSET32))
            , m_chunkPoolReader(std::move(rhs.m_chunkPoolReader))
        {
        }

        StateTableReader& operator=(const StateTableReader&) = delete;
        StateTableReader& operator=(StateTableReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pEntries = std::exchange(rhs.m_pEntries, nullptr);
                m_entryCount = std::exchange(rhs.m_entryCount, 0);
                m_entrySize = std::exchange(rhs.m_entrySize, 0);
                m_refCodec = std::move(rhs.m_refCodec);
                m_hasFixedRecordDesc = std::exchange(rhs.m_hasFixedRecordDesc, false);
                m_fixedRecordDesc = std::exchange(rhs.m_fixedRecordDesc, {});
                m_variableRecordDescOffsetInEntry = std::exchange(rhs.m_variableRecordDescOffsetInEntry, NVTRACE_INVALID_OFFSET32);
                m_payloadOffsetInEntry = std::exchange(rhs.m_payloadOffsetInEntry, 0);
                m_maxPayloadSizeDiv8 = std::exchange(rhs.m_maxPayloadSizeDiv8, 0);
                m_nextChunkRefOffsetInEntry = std::exchange(rhs.m_nextChunkRefOffsetInEntry, NVTRACE_INVALID_OFFSET32);
                m_chunkPoolReader = std::move(rhs.m_chunkPoolReader);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_chunkPoolReader.Reset();
            m_nextChunkRefOffsetInEntry = NVTRACE_INVALID_OFFSET32;
            m_maxPayloadSizeDiv8 = 0;
            m_payloadOffsetInEntry = 0;
            m_variableRecordDescOffsetInEntry = NVTRACE_INVALID_OFFSET32;
            m_fixedRecordDesc = {};
            m_hasFixedRecordDesc = false;
            m_refCodec.Reset();
            m_entrySize = 0;
            m_entryCount = 0;
            m_pEntries = nullptr;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t stateTableSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            const auto [pStateTableSection, stateTableSectionSize, stateTableSectionHeaderSize, stateTableSectionHeaderMinorVersion] = fnResolveSection(stateTableSectionIndex, NvTrace_ProviderBufferSectionType_StateTable);
            NVTRACE_LOG_ERR_RETURN_IF(!pStateTableSection, false, 50, "StateTableReader: failed to resolve state-table section (stateTableSectionIndex %u)", stateTableSectionIndex);

            SectionLayoutValidator stateTableSectionValidator(stateTableSectionSize);
            bool succeeded = stateTableSectionValidator.AddAndValidateSection(0, stateTableSectionHeaderSize, stateTableSectionHeaderMinorVersion, NvTrace_StateTableSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateTableReader: invalid section header layout (size %u, minorVersion %u)", stateTableSectionHeaderSize, stateTableSectionHeaderMinorVersion);

            const NvTrace_StateTableSectionHeader* const pStateTableHeader = reinterpret_cast<const NvTrace_StateTableSectionHeader*>(pStateTableSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pStateTableHeader) % alignof(NvTrace_StateTableSectionHeader)) != 0, false, 50, "StateTableReader: section header must be %zu-byte aligned (pStateTableHeader %p)", alignof(NvTrace_StateTableSectionHeader), pStateTableHeader);

            succeeded = stateTableSectionValidator.AddAndValidateSection(pStateTableHeader->entriesOffsetInSection, pStateTableHeader->entryCount, pStateTableHeader->entrySize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateTableReader: invalid entries array layout (offset %" PRIu64 ", entryCount %u, entrySize %u)", pStateTableHeader->entriesOffsetInSection, pStateTableHeader->entryCount, pStateTableHeader->entrySize);
            NVTRACE_LOG_ERR_RETURN_IF(pStateTableHeader->entryCount == 0, false, 50, "StateTableReader: entryCount must be non-zero");

            const uint8_t* const pEntries = pStateTableSection + pStateTableHeader->entriesOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pEntries) % alignof(uint64_t)) != 0 || (pStateTableHeader->entrySize % alignof(uint64_t)) != 0, false, 50, "StateTableReader: entries must be %zu-byte aligned (pEntries %p, entrySize %u)", alignof(uint64_t), pEntries, pStateTableHeader->entrySize);

            NVTRACE_LOG_ERR_RETURN_IF(pStateTableHeader->stateTableEntryHeaderMinorVersion < NvTrace_StateTableEntryHeader_MINOR_VERSION_A || pStateTableHeader->stateTableEntryHeaderSize < pStateTableHeader->stateTableEntryHeaderMinorVersion || (pStateTableHeader->stateTableEntryHeaderSize % alignof(uint64_t)) != 0, false, 50, "StateTableReader: invalid entry-header struct (stateTableEntryHeaderSize %u, stateTableEntryHeaderMinorVersion %u)", pStateTableHeader->stateTableEntryHeaderSize, pStateTableHeader->stateTableEntryHeaderMinorVersion);

            const uint32_t variableRecordDescOffsetInEntry = pStateTableHeader->variableRecordDescOffsetInEntry;
            const bool recordDescFits = variableRecordDescOffsetInEntry == NVTRACE_INVALID_OFFSET32 || (pStateTableHeader->stateTableEntryHeaderSize <= variableRecordDescOffsetInEntry && (variableRecordDescOffsetInEntry % alignof(NvTrace_RecordDesc)) == 0 && variableRecordDescOffsetInEntry <= pStateTableHeader->entrySize && pStateTableHeader->recordDescSize <= pStateTableHeader->entrySize - variableRecordDescOffsetInEntry);
            NVTRACE_LOG_ERR_RETURN_IF(!recordDescFits, false, 50, "StateTableReader: invalid record-desc offset (variableRecordDescOffsetInEntry %u, stateTableEntryHeaderSize %u, entrySize %u)", variableRecordDescOffsetInEntry, pStateTableHeader->stateTableEntryHeaderSize, pStateTableHeader->entrySize);

            const uint32_t nextChunkRefOffsetInEntry = pStateTableHeader->nextChunkRefOffsetInEntry;
            const bool nextChunkRefFits = nextChunkRefOffsetInEntry == NVTRACE_INVALID_OFFSET32 || (pStateTableHeader->stateTableEntryHeaderSize <= nextChunkRefOffsetInEntry && (nextChunkRefOffsetInEntry % alignof(NvTrace_RevisionedRef)) == 0 && nextChunkRefOffsetInEntry <= pStateTableHeader->entrySize && sizeof(NvTrace_RevisionedRef) <= pStateTableHeader->entrySize - nextChunkRefOffsetInEntry);
            NVTRACE_LOG_ERR_RETURN_IF(!nextChunkRefFits, false, 50, "StateTableReader: invalid next-chunk-ref offset (nextChunkRefOffsetInEntry %u, stateTableEntryHeaderSize %u, entrySize %u)", nextChunkRefOffsetInEntry, pStateTableHeader->stateTableEntryHeaderSize, pStateTableHeader->entrySize);

            NVTRACE_LOG_ERR_RETURN_IF(pStateTableHeader->payloadOffsetInEntry < pStateTableHeader->stateTableEntryHeaderSize || (pStateTableHeader->payloadOffsetInEntry % alignof(uint64_t)) != 0, false, 50, "StateTableReader: invalid payload offset (payloadOffsetInEntry %u, stateTableEntryHeaderSize %u)", pStateTableHeader->payloadOffsetInEntry, pStateTableHeader->stateTableEntryHeaderSize);
            NVTRACE_LOG_ERR_RETURN_IF(pStateTableHeader->entrySize < pStateTableHeader->payloadOffsetInEntry || pStateTableHeader->entrySize - pStateTableHeader->payloadOffsetInEntry < pStateTableHeader->maxPayloadSizeDiv8 * sizeof(uint64_t), false, 50, "StateTableReader: payload area does not fit the entry (entrySize %u, payloadOffsetInEntry %u, maxPayloadSizeDiv8 %u)", pStateTableHeader->entrySize, pStateTableHeader->payloadOffsetInEntry, pStateTableHeader->maxPayloadSizeDiv8);

            const bool hasFixedRecordDesc = pStateTableHeader->fixedRecordDescOffsetInSection != NVTRACE_INVALID_OFFSET64;
            NVTRACE_LOG_ERR_RETURN_IF(hasFixedRecordDesc == (variableRecordDescOffsetInEntry != NVTRACE_INVALID_OFFSET32), false, 50, "StateTableReader: exactly one of a per-entry record desc and a section-wide fixed one must be published (variableRecordDescOffsetInEntry %u, fixedRecordDescOffsetInSection %" PRIu64 ")", variableRecordDescOffsetInEntry, pStateTableHeader->fixedRecordDescOffsetInSection);

            NvTrace_RecordDesc fixedRecordDesc = {};
            if (hasFixedRecordDesc)
            {
                succeeded = stateTableSectionValidator.AddAndValidateSection(pStateTableHeader->fixedRecordDescOffsetInSection, pStateTableHeader->recordDescSize, pStateTableHeader->recordDescMinorVersion, NvTrace_RecordDesc_MINOR_VERSION_A);
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateTableReader: invalid fixed record desc layout (offset %" PRIu64 ", size %u, minorVersion %u)", pStateTableHeader->fixedRecordDescOffsetInSection, pStateTableHeader->recordDescSize, pStateTableHeader->recordDescMinorVersion);

                const NvTrace_RecordDesc* const pFixedRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(pStateTableSection + pStateTableHeader->fixedRecordDescOffsetInSection);
                NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pFixedRecordDesc) % alignof(NvTrace_RecordDesc)) != 0, false, 50, "StateTableReader: fixed record desc must be %zu-byte aligned (pFixedRecordDesc %p)", alignof(NvTrace_RecordDesc), pFixedRecordDesc);

                // A section-wide desc is written before any consumer attaches, so bounding its payload
                // here is what spares every entry read the check.
                fixedRecordDesc = *pFixedRecordDesc;
                NVTRACE_LOG_ERR_RETURN_IF(pStateTableHeader->maxPayloadSizeDiv8 < fixedRecordDesc.payloadEndOffsetDiv8, false, 50, "StateTableReader: fixed record desc payloadEndOffsetDiv8 (%u) exceeds maxPayloadSizeDiv8 (%u)", fixedRecordDesc.payloadEndOffsetDiv8, pStateTableHeader->maxPayloadSizeDiv8);
            }
            else
            {
                NVTRACE_LOG_ERR_RETURN_IF(pStateTableHeader->recordDescMinorVersion < NvTrace_RecordDesc_MINOR_VERSION_A || pStateTableHeader->recordDescSize < sizeof(NvTrace_RecordDesc), false, 50, "StateTableReader: per-entry record desc is too small to read (recordDescSize %u, recordDescMinorVersion %u, required %zu)", pStateTableHeader->recordDescSize, pStateTableHeader->recordDescMinorVersion, sizeof(NvTrace_RecordDesc));
            }

            AdaptiveChunkPoolReader chunkPool;
            const bool hasChunkPool = pStateTableHeader->adaptiveChunkPoolSectionIndex != NVTRACE_INVALID_SECTION_INDEX;
            NVTRACE_LOG_ERR_RETURN_IF(hasChunkPool != (nextChunkRefOffsetInEntry != NVTRACE_INVALID_OFFSET32), false, 50, "StateTableReader: a next-chunk-ref offset and a chunk pool must be published together (nextChunkRefOffsetInEntry %u, adaptiveChunkPoolSectionIndex %u)", nextChunkRefOffsetInEntry, pStateTableHeader->adaptiveChunkPoolSectionIndex);
            if (hasChunkPool)
            {
                succeeded = chunkPool.Initialize(pStateTableHeader->adaptiveChunkPoolSectionIndex, fnResolveSection);
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "StateTableReader: chunk-pool geometry validation failed (adaptiveChunkPoolSectionIndex %u)", pStateTableHeader->adaptiveChunkPoolSectionIndex);
            }

            m_pEntries = pEntries;
            m_entryCount = pStateTableHeader->entryCount;
            m_entrySize = pStateTableHeader->entrySize;
            m_refCodec.Initialize(pStateTableHeader->entryCount);
            m_hasFixedRecordDesc = hasFixedRecordDesc;
            m_fixedRecordDesc = fixedRecordDesc;
            m_variableRecordDescOffsetInEntry = variableRecordDescOffsetInEntry;
            m_payloadOffsetInEntry = pStateTableHeader->payloadOffsetInEntry;
            m_maxPayloadSizeDiv8 = pStateTableHeader->maxPayloadSizeDiv8;
            m_nextChunkRefOffsetInEntry = nextChunkRefOffsetInEntry;
            m_chunkPoolReader = std::move(chunkPool);

            return true;
        }

        uint32_t EntryCount() const
        {
            return m_entryCount;
        }

        NvTrace_RevisionedRef EncodeEntryRef(uint64_t entryIndex, uint64_t revision) const
        {
            return m_refCodec.EncodeItemRef(entryIndex, revision);
        }

        // Unpacks an entry ref into the (entryIndex, revision) pair the codec encoded it from.
        std::pair<uint32_t, uint64_t> DecodeEntryRef(NvTrace_RevisionedRef entryRef) const
        {
            const auto [entryIndex, revision] = m_refCodec.DecodeItemRef(entryRef);
            return {static_cast<uint32_t>(entryIndex), revision};
        }

        const NvTrace_RecordDesc& FixedRecordDesc() const
        {
            return m_fixedRecordDesc;
        }

        // A point read of one entry by the ref that names it. Delivers the entry's group -- its record,
        // then its chain records -- each followed by fnCommitRecord(bool); a stale ref, an
        // out-of-range index or a malformed desc delivers nothing and so commits nothing. With
        // TFollowChain false only the entry's own record is delivered, so a caller after one typed
        // record never sees a chain record.
        template <bool TFollowChain = true, typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ReadEntry(NvTrace_RevisionedRef entryRef, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            const auto [entryIndex, expectedEntryRevision] = m_refCodec.DecodeItemRef(entryRef);
            if (m_entryCount <= entryIndex)
            {
                return;
            }

            const uint8_t* const pEntry = m_pEntries + static_cast<size_t>(entryIndex) * m_entrySize;
            const NvTrace_StateTableEntryHeader* const pEntryHeader = reinterpret_cast<const NvTrace_StateTableEntryHeader*>(pEntry);

            NvTrace_RecordDesc recordDesc;
            if (m_hasFixedRecordDesc)
            {
                recordDesc = m_fixedRecordDesc;
            }
            else
            {
                recordDesc = *reinterpret_cast<const NvTrace_RecordDesc*>(pEntry + m_variableRecordDescOffsetInEntry);
                if (m_maxPayloadSizeDiv8 < recordDesc.payloadEndOffsetDiv8)
                {
                    return;
                }
            }
            const size_t payloadSize = static_cast<size_t>(recordDesc.payloadEndOffsetDiv8) * sizeof(uint64_t);

            fnReadRecord(entryRef, recordDesc.recordType, pEntry + m_payloadOffsetInEntry, payloadSize);

            if constexpr (!TFollowChain)
            {
                ReadFence();
                fnCommitRecord(m_refCodec.TruncateItemRevision(LoadRelaxed(&pEntryHeader->revision)) == expectedEntryRevision);
            }
            else
            {
                // Read under the entry's verdict too, so a chain the slot's next occupant hangs off it
                // is never followed under this reference.
                const NvTrace_RevisionedRef chainHeadRef = m_nextChunkRefOffsetInEntry == NVTRACE_INVALID_OFFSET32 ? NVTRACE_INVALID_REVISIONED_REF : LoadRelaxed(reinterpret_cast<const volatile NvTrace_RevisionedRef*>(pEntry + m_nextChunkRefOffsetInEntry));

                ReadFence();
                const bool entryConsistencyCheckPassed = m_refCodec.TruncateItemRevision(LoadRelaxed(&pEntryHeader->revision)) == expectedEntryRevision;
                fnCommitRecord(entryConsistencyCheckPassed);
                if (entryConsistencyCheckPassed)
                {
                    ReadEntryChain(chainHeadRef, fnReadRecord, fnCommitRecord);
                }
            }
        }

        // Drains the table once, in full: every published entry's group (or, with TFollowChain false,
        // its record alone), in entry order. An entry a writer holds mid-update is skipped -- a table
        // is drained once, so it is lost for good.
        template <bool TFollowChain = true, typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeAttachRecords(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            for (uint32_t entryIndex = 0; entryIndex < m_entryCount; ++entryIndex)
            {
                const NvTrace_StateTableEntryHeader* const pEntryHeader = reinterpret_cast<const NvTrace_StateTableEntryHeader*>(m_pEntries + static_cast<size_t>(entryIndex) * m_entrySize);

                const uint64_t expectedRevision = LoadAcquire(&pEntryHeader->revision);
                if (expectedRevision == 0 || (expectedRevision & NVTRACE_STATE_TABLE_ENTRY_REVISION_MODIFYING_BIT) != 0)
                {
                    continue;
                }

                ReadEntry<TFollowChain>(m_refCodec.EncodeItemRef(entryIndex, expectedRevision), fnReadRecord, fnCommitRecord);
            }
        }

    private:
        // Delivers a chain's records in order, each under its chunk's verdict; a chunk recycled or
        // malformed mid-walk ends the walk there.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ReadEntryChain(NvTrace_RevisionedRef chainHeadRef, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            NvTrace_RevisionedRef nextChunkRef = chainHeadRef;
            while (nextChunkRef != NVTRACE_INVALID_REVISIONED_REF)
            {
                const NvTrace_RevisionedRef chunkRef = nextChunkRef;
                const auto [pChunkHeader, expectedChunkRevision] = m_chunkPoolReader.ResolveChunkRef(chunkRef);
                if (!pChunkHeader)
                {
                    return;
                }

                nextChunkRef = LoadAcquire(&pChunkHeader->nextChunkRef);
                const uint32_t recordCount = LoadAcquire(&pChunkHeader->recordCount);
                if (m_chunkPoolReader.MaxRecordCountPerChunk() < recordCount)
                {
                    return;
                }

                ReadFence();
                if (m_chunkPoolReader.TruncateChunkRevision(LoadRelaxed(&pChunkHeader->revision)) != expectedChunkRevision)
                {
                    return;
                }

                const auto [pBeginRecordPos, pEndRecordPos, pBeginPayloadPos, maxPayloadOffset] = m_chunkPoolReader.ResolveChunkBoundaries(pChunkHeader, recordCount);
                size_t prevPayloadOffset = 0;
                uint64_t recordCorrelationId = m_chunkPoolReader.CalculateChunkBaseCorrelationId(chunkRef);

                for (const uint8_t* pRecordPos = pBeginRecordPos; pRecordPos < pEndRecordPos; pRecordPos += m_chunkPoolReader.RecordSize())
                {
                    const NvTrace_RecordDesc* const pRecord = reinterpret_cast<const NvTrace_RecordDesc*>(pRecordPos);
                    const size_t payloadOffset = static_cast<size_t>(pRecord->payloadEndOffsetDiv8) * sizeof(uint64_t);
                    const size_t payloadSize = payloadOffset - prevPayloadOffset;

                    // One subtraction check covers range and monotonicity via unsigned wrap (see NvTraceAdaptiveBufferReader.h).
                    if (maxPayloadOffset - prevPayloadOffset < payloadSize)
                    {
                        return;
                    }

                    fnReadRecord(recordCorrelationId, pRecord->recordType, pBeginPayloadPos - payloadOffset, payloadSize);

                    prevPayloadOffset = payloadOffset;
                    recordCorrelationId += 1;

                    ReadFence();
                    const bool chunkConsistencyCheckPassed = m_chunkPoolReader.TruncateChunkRevision(LoadRelaxed(&pChunkHeader->revision)) == expectedChunkRevision;
                    fnCommitRecord(chunkConsistencyCheckPassed);
                    if (!chunkConsistencyCheckPassed)
                    {
                        return;
                    }
                }
            }
        }

        const uint8_t* m_pEntries = nullptr;
        uint32_t m_entryCount = 0;
        uint32_t m_entrySize = 0;
        RevisionedRefCodec m_refCodec;

        bool m_hasFixedRecordDesc = false;
        NvTrace_RecordDesc m_fixedRecordDesc = {};
        uint32_t m_variableRecordDescOffsetInEntry = NVTRACE_INVALID_OFFSET32;

        uint32_t m_payloadOffsetInEntry = 0;
        uint32_t m_maxPayloadSizeDiv8 = 0;

        uint32_t m_nextChunkRefOffsetInEntry = NVTRACE_INVALID_OFFSET32;

        AdaptiveChunkPoolReader m_chunkPoolReader;
    };

} // namespace nv::trace
