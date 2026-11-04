/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceRevisionedRefCodec.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nv::trace {

    // Reader of an Adaptive Chunk Pool section, shared by every reader that walks chunks.
    // Initialize validates before publishing any field, so a populated instance is safe to
    // index with.
    //
    // Resolving a ref is geometry only: it bounds-checks the chunk index and hands back pointers.
    // The revision the ref carries stays the caller's to re-check against the chunk header once the
    // chunk has been read, which is what detects a recycle during the read.
    class AdaptiveChunkPoolReader
    {
    public:
        ~AdaptiveChunkPoolReader() = default;
        AdaptiveChunkPoolReader() = default;

        AdaptiveChunkPoolReader(const AdaptiveChunkPoolReader&) = delete;
        AdaptiveChunkPoolReader(AdaptiveChunkPoolReader&& rhs) noexcept
            : m_pChunks(std::exchange(rhs.m_pChunks, nullptr))
            , m_chunkCount(std::exchange(rhs.m_chunkCount, 0))
            , m_chunkSize(std::exchange(rhs.m_chunkSize, 0))
            , m_chunkHeaderSize(std::exchange(rhs.m_chunkHeaderSize, 0))
            , m_recordSize(std::exchange(rhs.m_recordSize, 0))
            , m_maxRecordCountPerChunk(std::exchange(rhs.m_maxRecordCountPerChunk, 0))
            , m_maxRecordPayloadSize(std::exchange(rhs.m_maxRecordPayloadSize, 0))
            , m_refCodec(std::move(rhs.m_refCodec))
        {
        }

        AdaptiveChunkPoolReader& operator=(const AdaptiveChunkPoolReader&) = delete;
        AdaptiveChunkPoolReader& operator=(AdaptiveChunkPoolReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pChunks = std::exchange(rhs.m_pChunks, nullptr);
                m_chunkCount = std::exchange(rhs.m_chunkCount, 0);
                m_chunkSize = std::exchange(rhs.m_chunkSize, 0);
                m_chunkHeaderSize = std::exchange(rhs.m_chunkHeaderSize, 0);
                m_recordSize = std::exchange(rhs.m_recordSize, 0);
                m_maxRecordCountPerChunk = std::exchange(rhs.m_maxRecordCountPerChunk, 0);
                m_maxRecordPayloadSize = std::exchange(rhs.m_maxRecordPayloadSize, 0);
                m_refCodec = std::move(rhs.m_refCodec);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_refCodec.Reset();
            m_maxRecordPayloadSize = 0;
            m_maxRecordCountPerChunk = 0;
            m_recordSize = 0;
            m_chunkHeaderSize = 0;
            m_chunkSize = 0;
            m_chunkCount = 0;
            m_pChunks = nullptr;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t adaptiveChunkPoolSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            const auto [pChunkPoolSection, chunkPoolSectionSize, chunkPoolSectionHeaderSize, chunkPoolSectionHeaderMinorVersion] = fnResolveSection(adaptiveChunkPoolSectionIndex, NvTrace_ProviderBufferSectionType_AdaptiveChunkPool);
            NVTRACE_LOG_ERR_RETURN_IF(!pChunkPoolSection, false, 50, "AdaptiveChunkPoolReader: failed to resolve chunk-pool section (adaptiveChunkPoolSectionIndex %u)", adaptiveChunkPoolSectionIndex);

            SectionLayoutValidator chunkPoolSectionValidator(chunkPoolSectionSize);
            bool succeeded = chunkPoolSectionValidator.AddAndValidateSection(0, chunkPoolSectionHeaderSize, chunkPoolSectionHeaderMinorVersion, NvTrace_AdaptiveChunkPoolSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "AdaptiveChunkPoolReader: invalid chunk-pool header layout (size %u, minorVersion %u)", chunkPoolSectionHeaderSize, chunkPoolSectionHeaderMinorVersion);

            const NvTrace_AdaptiveChunkPoolSectionHeader* const pChunkPoolHeader = reinterpret_cast<const NvTrace_AdaptiveChunkPoolSectionHeader*>(pChunkPoolSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pChunkPoolHeader) % alignof(NvTrace_AdaptiveChunkPoolSectionHeader)) != 0, false, 50, "AdaptiveChunkPoolReader: chunk-pool header must be %zu-byte aligned (pChunkPoolHeader %p)", alignof(NvTrace_AdaptiveChunkPoolSectionHeader), pChunkPoolHeader);

            succeeded = chunkPoolSectionValidator.AddAndValidateSection(pChunkPoolHeader->chunksOffsetInSection, pChunkPoolHeader->chunkCount, pChunkPoolHeader->chunkSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "AdaptiveChunkPoolReader: invalid chunks array layout (offset %" PRIu64 ", chunkCount %u, chunkSize %u)", pChunkPoolHeader->chunksOffsetInSection, pChunkPoolHeader->chunkCount, pChunkPoolHeader->chunkSize);

            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->chunkCount == 0, false, 50, "AdaptiveChunkPoolReader: chunkCount must be non-zero.");
            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->adaptiveChunkHeaderMinorVersion < NvTrace_AdaptiveChunkHeader_MINOR_VERSION_A || pChunkPoolHeader->adaptiveChunkHeaderSize < pChunkPoolHeader->adaptiveChunkHeaderMinorVersion, false, 50, "AdaptiveChunkPoolReader: invalid chunk-header struct (adaptiveChunkHeaderSize %u, adaptiveChunkHeaderMinorVersion %u)", pChunkPoolHeader->adaptiveChunkHeaderSize, pChunkPoolHeader->adaptiveChunkHeaderMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->recordDescMinorVersion < NvTrace_RecordDesc_MINOR_VERSION_A || pChunkPoolHeader->recordDescSize < pChunkPoolHeader->recordDescMinorVersion, false, 50, "AdaptiveChunkPoolReader: invalid record struct (recordDescSize %u, recordDescMinorVersion %u)", pChunkPoolHeader->recordDescSize, pChunkPoolHeader->recordDescMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->chunkSize < pChunkPoolHeader->adaptiveChunkHeaderSize || pChunkPoolHeader->chunkSize - pChunkPoolHeader->adaptiveChunkHeaderSize < pChunkPoolHeader->recordDescSize, false, 50, "AdaptiveChunkPoolReader: chunk too small for at least one record (chunkSize %u, adaptiveChunkHeaderSize %u, recordDescSize %u)", pChunkPoolHeader->chunkSize, pChunkPoolHeader->adaptiveChunkHeaderSize, pChunkPoolHeader->recordDescSize);
            NVTRACE_LOG_ERR_RETURN_IF((pChunkPoolHeader->chunkSize % alignof(uint64_t)) != 0 || (pChunkPoolHeader->adaptiveChunkHeaderSize % alignof(uint64_t)) != 0 || (pChunkPoolHeader->recordDescSize % alignof(uint64_t)) != 0, false, 50, "AdaptiveChunkPoolReader: chunkSize/adaptiveChunkHeaderSize/recordDescSize must each be a multiple of %zu (chunkSize %u, adaptiveChunkHeaderSize %u, recordDescSize %u)", alignof(uint64_t), pChunkPoolHeader->chunkSize, pChunkPoolHeader->adaptiveChunkHeaderSize, pChunkPoolHeader->recordDescSize);

            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->maxRecordCountPerChunk == 0 || (pChunkPoolHeader->chunkSize - pChunkPoolHeader->adaptiveChunkHeaderSize) / pChunkPoolHeader->recordDescSize < pChunkPoolHeader->maxRecordCountPerChunk, false, 50, "AdaptiveChunkPoolReader: maxRecordCountPerChunk (%u) does not fit a chunk (chunkSize %u, adaptiveChunkHeaderSize %u, recordDescSize %u)", pChunkPoolHeader->maxRecordCountPerChunk, pChunkPoolHeader->chunkSize, pChunkPoolHeader->adaptiveChunkHeaderSize, pChunkPoolHeader->recordDescSize);

            const uint8_t* const pValidatedChunks = pChunkPoolSection + pChunkPoolHeader->chunksOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pValidatedChunks) % alignof(uint64_t)) != 0, false, 50, "AdaptiveChunkPoolReader: chunks must be %zu-byte aligned (pChunks %p)", alignof(uint64_t), pValidatedChunks);

            m_pChunks = pValidatedChunks;
            m_chunkCount = pChunkPoolHeader->chunkCount;
            m_chunkSize = pChunkPoolHeader->chunkSize;
            m_chunkHeaderSize = pChunkPoolHeader->adaptiveChunkHeaderSize;
            m_recordSize = pChunkPoolHeader->recordDescSize;
            m_maxRecordCountPerChunk = pChunkPoolHeader->maxRecordCountPerChunk;
            // The chunk-too-small guard above proves chunkHeaderSize + recordSize <= chunkSize, so this cannot underflow.
            m_maxRecordPayloadSize = AlignDnP2(m_chunkSize - (m_chunkHeaderSize + m_recordSize), alignof(uint64_t));
            m_refCodec.Initialize(m_chunkCount, m_maxRecordCountPerChunk);

            return true;
        }

        uint64_t ChunkCount() const
        {
            return m_chunkCount;
        }

        size_t RecordSize() const
        {
            return m_recordSize;
        }

        uint32_t MaxRecordCountPerChunk() const
        {
            return m_maxRecordCountPerChunk;
        }

        size_t MaxRecordPayloadSize() const
        {
            return m_maxRecordPayloadSize;
        }

        uint64_t TruncateChunkRevision(uint64_t revision) const
        {
            return m_refCodec.TruncateItemRevision(revision);
        }

        uint64_t CalculateChunkBaseCorrelationId(NvTrace_RevisionedRef chunkRef) const
        {
            return m_refCodec.EncodeRecordRef(chunkRef, 0);
        }

        struct ResolvedChunk
        {
            const NvTrace_AdaptiveChunkHeader* pChunkHeader;
            uint64_t expectedRevision;
        };

        // A null pChunkHeader means the ref names no chunk of this pool.
        ResolvedChunk ResolveChunkRef(NvTrace_RevisionedRef chunkRef) const
        {
            const auto [chunkIndex, expectedRevision] = m_refCodec.DecodeItemRef(chunkRef);
            if (m_chunkCount <= chunkIndex)
            {
                return {};
            }

            return {reinterpret_cast<const NvTrace_AdaptiveChunkHeader*>(m_pChunks + chunkIndex * m_chunkSize), expectedRevision};
        }

        struct ChunkBoundaries
        {
            const uint8_t* pBeginRecordDescPos;
            const uint8_t* pEndRecordDescPos;
            const uint8_t* pBeginPayloadPos;
            size_t maxPayloadEndOffset;
        };

        // recordCount must already be bounded by MaxRecordCountPerChunk(); the records-end subtraction
        // below wraps otherwise.
        ChunkBoundaries ResolveChunkBoundaries(const NvTrace_AdaptiveChunkHeader* pChunkHeader, uint32_t recordCount) const
        {
            const uint8_t* const pChunk = reinterpret_cast<const uint8_t*>(pChunkHeader);
            const size_t recordDescsEndOffsetInChunk = m_chunkHeaderSize + recordCount * m_recordSize;
            return {pChunk + m_chunkHeaderSize, pChunk + recordDescsEndOffsetInChunk, pChunk + m_chunkSize, m_chunkSize - recordDescsEndOffsetInChunk};
        }

        // A point read of one record by its record ref (== the correlationId AdaptiveBufferReader hands
        // out). fnCommitRecord pairs 1:1 with fnReadRecord, as in ConsumeRecords: a stale or malformed
        // ref delivers nothing and so commits nothing.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ReadRecord(NvTrace_RevisionedRef recordRef, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord) const
        {
            const auto [recordIndex, chunkIndex, expectedRevision] = m_refCodec.DecodeRecordRef(recordRef);
            if (m_chunkCount <= chunkIndex)
            {
                return;
            }

            const NvTrace_AdaptiveChunkHeader* const pChunkHeader = reinterpret_cast<const NvTrace_AdaptiveChunkHeader*>(m_pChunks + chunkIndex * m_chunkSize);

            // The chunk the ref names must still be the one it was minted against, checked before
            // anything is read out of it: a chunk recycled since resolves to nothing rather than
            // delivering its new occupant's record and rejecting it only at the commit.
            if (m_refCodec.TruncateRecordRevision(LoadAcquire(&pChunkHeader->revision)) != expectedRevision)
            {
                return;
            }

            const uint32_t recordCount = LoadAcquire(&pChunkHeader->recordCount);
            if (m_maxRecordCountPerChunk < recordCount || recordCount <= recordIndex)
            {
                return;
            }

            const auto [pBeginRecordDescPos, pEndRecordDescPos, pBeginPayloadPos, maxPayloadEndOffset] = ResolveChunkBoundaries(pChunkHeader, recordCount);

            const uint8_t* const pRecordDescPos = pBeginRecordDescPos + recordIndex * m_recordSize;
            const NvTrace_RecordDesc* const pRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(pRecordDescPos);
            const uint32_t recordType = pRecordDesc->recordType;
            const size_t payloadEndOffset = pRecordDesc->payloadEndOffsetDiv8 * sizeof(uint64_t);
            const size_t prevPayloadEndOffset = recordIndex == 0 ? 0 : reinterpret_cast<const NvTrace_RecordDesc*>(pRecordDescPos - m_recordSize)->payloadEndOffsetDiv8 * sizeof(uint64_t);
            const size_t payloadSize = payloadEndOffset - prevPayloadEndOffset;

            // Both offsets are untrusted bytes read fresh, so unlike the sequential consumer -- whose
            // previous offset was validated on the previous iteration -- a point read must bound
            // prevPayloadEndOffset itself first; the second subtraction then covers a payload reaching into
            // the record descs area and a non-monotone pair alike (which wraps payloadSize huge).
            if (maxPayloadEndOffset < prevPayloadEndOffset || maxPayloadEndOffset - prevPayloadEndOffset < payloadSize)
            {
                return;
            }

            fnReadRecord(recordRef, recordType, pBeginPayloadPos - payloadEndOffset, payloadSize);

            ReadFence();
            fnCommitRecord(m_refCodec.TruncateRecordRevision(LoadRelaxed(&pChunkHeader->revision)) == expectedRevision);
        }

    private:
        const uint8_t* m_pChunks = nullptr;
        uint64_t m_chunkCount = 0;
        size_t m_chunkSize = 0;
        size_t m_chunkHeaderSize = 0;
        size_t m_recordSize = 0;
        uint32_t m_maxRecordCountPerChunk = 0;
        size_t m_maxRecordPayloadSize = 0;

        RevisionedRefCodec m_refCodec;
    };

} // namespace nv::trace
