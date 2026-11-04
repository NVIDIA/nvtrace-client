/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/Observer/NvTraceAdaptiveChunkPoolReader.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nv::trace {

    // Reader of an Adaptive Buffer (cells -> chunk chains -> records + opaque payloads).
    //
    // Each cell points to a singly-linked chain of variable-sized chunks. Each chunk holds
    // a forward-growing array of NvTrace_AdaptiveBufferRecord followed by a backward-growing
    // array of opaque payload bytes (each payload uint64_t-aligned). The tail chunk in a
    // chain may still be open for writing; the reader speculatively reads from it and
    // re-checks for new records on the next ConsumeRecords call.
    //
    // Producer/consumer contract: (a) per-cell time gate (NvTrace_AdaptiveBufferCell::readableAfterTsc)
    // gives the producer enough headroom to finish in-flight writes before the reader looks,
    // and (b) per-chunk revision check detects if the producer recycled a chunk during a read.
    //
    // Thread safety: the reader is not internally synchronized and must be used from a single
    // consumer thread. The producer side may be concurrent.
    //
    // ConsumeRecords(readEndpointTsc, fnReadRecord, fnCommitRecord):
    //   - Walks cells whose readableAfterTsc <= readEndpointTsc, calling
    //     fnReadRecord(recordType, pPayload, payloadSize) for every record. Accepted payload bytes
    //     are tentative until the fnCommitRecord(bool valid) that follows:
    //       - valid == true:  the record is consistent and may be acted upon irreversibly.
    //       - valid == false: the producer overran the chunk; the tentative record must be
    //                         discarded, along with the rest of the cell -- loss is at cell-tail
    //                         granularity, never a hole in the middle of one.
    //   - fnReadRecord may optionally return a bool: false is an early-exit, re-delivering the record
    //     on the next call; its fnCommitRecord covers no record.
    class AdaptiveBufferReader
    {
    public:
        ~AdaptiveBufferReader() = default;
        AdaptiveBufferReader() noexcept = default;

        AdaptiveBufferReader(const AdaptiveBufferReader&) = delete;
        AdaptiveBufferReader(AdaptiveBufferReader&& rhs) noexcept
            : m_pState(std::exchange(rhs.m_pState, nullptr))
            , m_pCells(std::exchange(rhs.m_pCells, nullptr))
            , m_cellCount(std::exchange(rhs.m_cellCount, 0))
            , m_threeQuartersCellCount(std::exchange(rhs.m_threeQuartersCellCount, 0))
            , m_cellSize(std::exchange(rhs.m_cellSize, 0))
            , m_chunkPoolReader(std::move(rhs.m_chunkPoolReader))
            , m_nextCellReadSequenceNumber(std::exchange(rhs.m_nextCellReadSequenceNumber, 0))
            , m_nextCellReadIndex(std::exchange(rhs.m_nextCellReadIndex, 0))
            , m_currentCellNextChunkRef(std::exchange(rhs.m_currentCellNextChunkRef, NVTRACE_INVALID_REVISIONED_REF))
            , m_pCurrentChunkHeader(std::exchange(rhs.m_pCurrentChunkHeader, nullptr))
            , m_pCurrentChunkBeginPayloadPos(std::exchange(rhs.m_pCurrentChunkBeginPayloadPos, nullptr))
            , m_currentChunkExpectedRevision(std::exchange(rhs.m_currentChunkExpectedRevision, 0))
            , m_currentChunkIncomplete(std::exchange(rhs.m_currentChunkIncomplete, false))
            , m_pCurrentRecordDescPos(std::exchange(rhs.m_pCurrentRecordDescPos, nullptr))
            , m_pEndRecordDescPos(std::exchange(rhs.m_pEndRecordDescPos, nullptr))
            , m_prevPayloadEndOffset(std::exchange(rhs.m_prevPayloadEndOffset, 0))
            , m_maxPayloadEndOffset(std::exchange(rhs.m_maxPayloadEndOffset, 0))
            , m_currentRecordCorrelationId(std::exchange(rhs.m_currentRecordCorrelationId, 0))
            , m_tentativelyConsumedRecordCount(std::exchange(rhs.m_tentativelyConsumedRecordCount, 0))
            , m_totalRecordCount(std::exchange(rhs.m_totalRecordCount, 0))
            , m_droppedRecordCount(std::exchange(rhs.m_droppedRecordCount, 0))
        {
        }

        AdaptiveBufferReader& operator=(const AdaptiveBufferReader&) = delete;
        AdaptiveBufferReader& operator=(AdaptiveBufferReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pState = std::exchange(rhs.m_pState, nullptr);
                m_pCells = std::exchange(rhs.m_pCells, nullptr);
                m_cellCount = std::exchange(rhs.m_cellCount, 0);
                m_threeQuartersCellCount = std::exchange(rhs.m_threeQuartersCellCount, 0);
                m_cellSize = std::exchange(rhs.m_cellSize, 0);
                m_chunkPoolReader = std::move(rhs.m_chunkPoolReader);
                m_nextCellReadSequenceNumber = std::exchange(rhs.m_nextCellReadSequenceNumber, 0);
                m_nextCellReadIndex = std::exchange(rhs.m_nextCellReadIndex, 0);
                m_currentCellNextChunkRef = std::exchange(rhs.m_currentCellNextChunkRef, NVTRACE_INVALID_REVISIONED_REF);
                m_pCurrentChunkHeader = std::exchange(rhs.m_pCurrentChunkHeader, nullptr);
                m_pCurrentChunkBeginPayloadPos = std::exchange(rhs.m_pCurrentChunkBeginPayloadPos, nullptr);
                m_currentChunkExpectedRevision = std::exchange(rhs.m_currentChunkExpectedRevision, 0);
                m_currentChunkIncomplete = std::exchange(rhs.m_currentChunkIncomplete, false);
                m_pCurrentRecordDescPos = std::exchange(rhs.m_pCurrentRecordDescPos, nullptr);
                m_pEndRecordDescPos = std::exchange(rhs.m_pEndRecordDescPos, nullptr);
                m_prevPayloadEndOffset = std::exchange(rhs.m_prevPayloadEndOffset, 0);
                m_maxPayloadEndOffset = std::exchange(rhs.m_maxPayloadEndOffset, 0);
                m_currentRecordCorrelationId = std::exchange(rhs.m_currentRecordCorrelationId, 0);
                m_tentativelyConsumedRecordCount = std::exchange(rhs.m_tentativelyConsumedRecordCount, 0);
                m_totalRecordCount = std::exchange(rhs.m_totalRecordCount, 0);
                m_droppedRecordCount = std::exchange(rhs.m_droppedRecordCount, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_droppedRecordCount = 0;
            m_totalRecordCount = 0;
            m_tentativelyConsumedRecordCount = 0;
            m_currentRecordCorrelationId = 0;
            m_maxPayloadEndOffset = 0;
            m_prevPayloadEndOffset = 0;
            m_pEndRecordDescPos = nullptr;
            m_pCurrentRecordDescPos = nullptr;
            m_currentChunkIncomplete = false;
            m_currentChunkExpectedRevision = 0;
            m_pCurrentChunkBeginPayloadPos = nullptr;
            m_pCurrentChunkHeader = nullptr;
            m_currentCellNextChunkRef = NVTRACE_INVALID_REVISIONED_REF;
            m_nextCellReadIndex = 0;
            m_nextCellReadSequenceNumber = 0;
            m_chunkPoolReader.Reset();
            m_cellSize = 0;
            m_threeQuartersCellCount = 0;
            m_cellCount = 0;
            m_pCells = nullptr;
            m_pState = nullptr;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t adaptiveBufferSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            // Adaptive Buffer section layout
            const auto [pAdaptiveBufferSection, adaptiveBufferSectionSize, adaptiveBufferSectionHeaderSize, adaptiveBufferSectionHeaderMinorVersion] = fnResolveSection(adaptiveBufferSectionIndex, NvTrace_ProviderBufferSectionType_AdaptiveBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(!pAdaptiveBufferSection, false, 50, "AdaptiveBufferReader: failed to resolve adaptive-buffer section (adaptiveBufferSectionIndex %u)", adaptiveBufferSectionIndex);

            SectionLayoutValidator adaptiveBufferSectionValidator(adaptiveBufferSectionSize);
            bool succeeded = adaptiveBufferSectionValidator.AddAndValidateSection(0, adaptiveBufferSectionHeaderSize, adaptiveBufferSectionHeaderMinorVersion, NvTrace_AdaptiveBufferSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "AdaptiveBufferReader: invalid section header layout (size %u, minorVersion %u)", adaptiveBufferSectionHeaderSize, adaptiveBufferSectionHeaderMinorVersion);

            const NvTrace_AdaptiveBufferSectionHeader* pAdaptiveBufferHeader = reinterpret_cast<const NvTrace_AdaptiveBufferSectionHeader*>(pAdaptiveBufferSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pAdaptiveBufferHeader) % alignof(NvTrace_AdaptiveBufferSectionHeader)) != 0, false, 50, "AdaptiveBufferReader: section header must be %zu-byte aligned (pAdaptiveBufferHeader %p)", alignof(NvTrace_AdaptiveBufferSectionHeader), pAdaptiveBufferHeader);

            succeeded = adaptiveBufferSectionValidator.AddAndValidateSection(pAdaptiveBufferHeader->adaptiveBufferStateOffsetInSection, pAdaptiveBufferHeader->adaptiveBufferStateSize, pAdaptiveBufferHeader->adaptiveBufferStateMinorVersion, NvTrace_AdaptiveBufferState_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "AdaptiveBufferReader: invalid state layout (offset %llu, size %u, minorVersion %u)", pAdaptiveBufferHeader->adaptiveBufferStateOffsetInSection, pAdaptiveBufferHeader->adaptiveBufferStateSize, pAdaptiveBufferHeader->adaptiveBufferStateMinorVersion);

            succeeded = adaptiveBufferSectionValidator.AddAndValidateSection(pAdaptiveBufferHeader->adaptiveBufferCellsOffsetInSection, pAdaptiveBufferHeader->adaptiveBufferCellCount, pAdaptiveBufferHeader->adaptiveBufferCellSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "AdaptiveBufferReader: invalid cells array layout (offset %llu, adaptiveBufferCellCount %u, adaptiveBufferCellSize %u)", pAdaptiveBufferHeader->adaptiveBufferCellsOffsetInSection, pAdaptiveBufferHeader->adaptiveBufferCellCount, pAdaptiveBufferHeader->adaptiveBufferCellSize);

            const NvTrace_AdaptiveBufferState* pState = reinterpret_cast<const NvTrace_AdaptiveBufferState*>(pAdaptiveBufferSection + pAdaptiveBufferHeader->adaptiveBufferStateOffsetInSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pState) % alignof(NvTrace_AdaptiveBufferState)) != 0, false, 50, "AdaptiveBufferReader: state must be %zu-byte aligned (pState %p)", alignof(NvTrace_AdaptiveBufferState), pState);

            NVTRACE_LOG_ERR_RETURN_IF(pAdaptiveBufferHeader->adaptiveBufferCellCount <= 1, false, 50, "AdaptiveBufferReader: cellCount (%u) must be greater than 1", pAdaptiveBufferHeader->adaptiveBufferCellCount);
            NVTRACE_LOG_ERR_RETURN_IF(pAdaptiveBufferHeader->adaptiveBufferCellMinorVersion < NvTrace_AdaptiveBufferCell_MINOR_VERSION_A || pAdaptiveBufferHeader->adaptiveBufferCellSize < pAdaptiveBufferHeader->adaptiveBufferCellMinorVersion, false, 50, "AdaptiveBufferReader: invalid cell struct (adaptiveBufferCellSize %u, adaptiveBufferCellMinorVersion %u)", pAdaptiveBufferHeader->adaptiveBufferCellSize, pAdaptiveBufferHeader->adaptiveBufferCellMinorVersion);

            const uint8_t* pCells = pAdaptiveBufferSection + pAdaptiveBufferHeader->adaptiveBufferCellsOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pCells) % alignof(uint64_t)) != 0 || (pAdaptiveBufferHeader->adaptiveBufferCellSize % alignof(uint64_t)) != 0, false, 50, "AdaptiveBufferReader: cells must be %zu-byte aligned (pCells %p, cellSize %u)", alignof(uint64_t), pCells, pAdaptiveBufferHeader->adaptiveBufferCellSize);

            AdaptiveChunkPoolReader chunkPoolReader;
            succeeded = chunkPoolReader.Initialize(pAdaptiveBufferHeader->adaptiveChunkPoolSectionIndex, fnResolveSection);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "AdaptiveBufferReader: chunk-pool geometry validation failed (adaptiveChunkPoolSectionIndex %u)", pAdaptiveBufferHeader->adaptiveChunkPoolSectionIndex);

            m_pState = pState;
            m_pCells = pCells;
            m_cellCount = pAdaptiveBufferHeader->adaptiveBufferCellCount;
            m_threeQuartersCellCount = m_cellCount < 4 ? 1 : static_cast<uint32_t>((static_cast<uint64_t>(m_cellCount) * 3) / 4);
            m_cellSize = pAdaptiveBufferHeader->adaptiveBufferCellSize;
            m_chunkPoolReader = std::move(chunkPoolReader);

            // Start reading from a safe oldest valid cell (~ 75% behind the write pointer == 25% ahead).
            const uint64_t nextCellWriteSequenceNumber = LoadRelaxed(&m_pState->nextCellWriteSequenceNumber);
            m_nextCellReadSequenceNumber = m_threeQuartersCellCount < nextCellWriteSequenceNumber ? nextCellWriteSequenceNumber - m_threeQuartersCellCount : 0;
            m_nextCellReadIndex = static_cast<uint32_t>(m_nextCellReadSequenceNumber % m_cellCount);

            return true;
        }

        uint64_t DroppedRecordCount() const
        {
            return m_droppedRecordCount;
        }

        size_t MaxRecordPayloadSize() const
        {
            return m_chunkPoolReader.MaxRecordPayloadSize();
        }

        uint32_t CellCount() const
        {
            return m_cellCount;
        }

        // Advances only the cell ring, without resolving chunk references, and leaves the first
        // cell whose readable boundary straddles timestamp unread.
        bool SkipCellsBeforeTimestamp(uint64_t timestamp)
        {
            const uint64_t cellReadEndSequenceNumber = LoadAcquire(&m_pState->nextCellWriteSequenceNumber);
            if (m_nextCellReadSequenceNumber + m_cellCount <= cellReadEndSequenceNumber)
            {
                RecoverFromCellBufferOverrun(cellReadEndSequenceNumber);
            }

            const uint64_t initialCellReadSequenceNumber = m_nextCellReadSequenceNumber;
            uint64_t lowerSequenceNumber = initialCellReadSequenceNumber;
            uint64_t upperSequenceNumber = cellReadEndSequenceNumber;
            while (lowerSequenceNumber < upperSequenceNumber)
            {
                const uint64_t middleSequenceNumber = lowerSequenceNumber + (upperSequenceNumber - lowerSequenceNumber) / 2;
                const uint32_t middleCellIndex = static_cast<uint32_t>(middleSequenceNumber % m_cellCount);
                const auto* const pCell = reinterpret_cast<const NvTrace_AdaptiveBufferCell*>(m_pCells + middleCellIndex * m_cellSize);
                if (pCell->readableAfterTsc < timestamp)
                {
                    lowerSequenceNumber = middleSequenceNumber + 1;
                }
                else
                {
                    upperSequenceNumber = middleSequenceNumber;
                }
            }

            ReadFence();
            if (initialCellReadSequenceNumber + m_cellCount <= LoadRelaxed(&m_pState->nextCellWriteSequenceNumber))
            {
                return false;
            }

            m_nextCellReadSequenceNumber = lowerSequenceNumber;
            m_nextCellReadIndex = static_cast<uint32_t>(m_nextCellReadSequenceNumber % m_cellCount);
            return true;
        }

        //   void fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //     Accepts every delivered record.
        //
        //   bool fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //     Return true to accept the record and advance the cursor.
        //     Return false to retain the record and stop until the next ConsumeRecords().
        //
        //   void fnCommitRecord(bool consistencyCheckPassed)
        //     Called once per accepted record. If false, roll back state staged by fnReadRecord.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeRecords(uint64_t readEndpointTsc, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            uint64_t nextCellWriteSequenceNumber = LoadAcquire(&m_pState->nextCellWriteSequenceNumber);
            if (m_nextCellReadSequenceNumber + m_cellCount <= nextCellWriteSequenceNumber)
            {
                RecoverFromCellBufferOverrun(nextCellWriteSequenceNumber);
            }

            const uint64_t cellReadEndSequenceNumber = nextCellWriteSequenceNumber;

            if (m_currentChunkIncomplete)
            {
                RefreshCurrentIncompleteChunk(cellReadEndSequenceNumber);
            }

            do
            {
                while (m_pCurrentRecordDescPos < m_pEndRecordDescPos)
                {
                    const NvTrace_RecordDesc* const pRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(m_pCurrentRecordDescPos);
                    const uint32_t recordType = pRecordDesc->recordType;
                    const size_t payloadEndOffset = pRecordDesc->payloadEndOffsetDiv8 * sizeof(uint64_t);
                    const size_t payloadSize = payloadEndOffset - m_prevPayloadEndOffset;

                    // A single check covers all failure modes via unsigned-wrap of the payloadSize subtraction above:
                    //   - payload offset past the record descs area (m_maxPayloadEndOffset < payloadEndOffset)
                    //   - non-monotone payload offset (payloadEndOffset < m_prevPayloadEndOffset, wraps payloadSize huge)
                    // In both cases payloadSize ends up exceeding m_maxPayloadEndOffset - m_prevPayloadEndOffset; abandon the cell.
                    if (m_maxPayloadEndOffset - m_prevPayloadEndOffset < payloadSize)
                    {
                        AbandonCurrentCell();
                        break;
                    }

                    if constexpr (std::is_same_v<std::invoke_result_t<TFN_ReadRecord, uint32_t, const uint8_t*, size_t>, void>)
                    {
                        fnReadRecord(recordType, m_pCurrentChunkBeginPayloadPos - payloadEndOffset, payloadSize);
                    }
                    else
                    {
                        if (!fnReadRecord(recordType, m_pCurrentChunkBeginPayloadPos - payloadEndOffset, payloadSize))
                        {
                            // A rejected record was not accepted by the consumer, so there is no
                            // tentative consumer state for fnCommitRecord to resolve. Still perform
                            // the consistency check so an overrun abandons the source cell normally.
                            CheckConsistency();
                            return;
                        }
                    }

                    m_pCurrentRecordDescPos += m_chunkPoolReader.RecordSize();
                    m_prevPayloadEndOffset = payloadEndOffset;
                    m_currentRecordCorrelationId += 1;
                    m_tentativelyConsumedRecordCount += 1;

                    fnCommitRecord(CheckConsistency());
                }
            } while (!m_currentChunkIncomplete && TryAdvanceToNextChunk(cellReadEndSequenceNumber, readEndpointTsc));
        }

        // Must only be called from within fnReadRecord()
        uint64_t CurrentRecordCorrelationId() const
        {
            return m_currentRecordCorrelationId;
        }

    private:
        bool CheckConsistency()
        {
            ReadFence();
            if (m_chunkPoolReader.TruncateChunkRevision(LoadRelaxed(&m_pCurrentChunkHeader->revision)) == m_currentChunkExpectedRevision)
            {
                m_totalRecordCount += std::exchange(m_tentativelyConsumedRecordCount, 0);
                return true;
            }
            else
            {
                AbandonCurrentCell();
                return false;
            }
        }

        void RecoverFromCellBufferOverrun(uint64_t nextCellWriteSequenceNumber)
        {
            m_nextCellReadSequenceNumber = nextCellWriteSequenceNumber - m_threeQuartersCellCount;
            m_nextCellReadIndex = static_cast<uint32_t>(m_nextCellReadSequenceNumber % m_cellCount);
        }

        // Drops the rest of the cell, not just this chunk: a successor is reached through a ref read
        // from a chunk that has since been recycled, and one still sitting unrecycled in the free list
        // would pass its own revision check and deliver a torn cell's tail.
        void AbandonCurrentCell()
        {
            m_currentChunkIncomplete = false;
            m_pCurrentRecordDescPos = m_pEndRecordDescPos;
            m_currentCellNextChunkRef = NVTRACE_INVALID_REVISIONED_REF;
        }

        bool TryAdvanceToNextCell(uint64_t cellReadEndSequenceNumber, uint64_t readEndpointTsc)
        {
            if (cellReadEndSequenceNumber <= m_nextCellReadSequenceNumber)
            {
                return false;
            }

            NvTrace_AdaptiveBufferCell cell = *reinterpret_cast<const NvTrace_AdaptiveBufferCell*>(m_pCells + m_nextCellReadIndex * m_cellSize);

            ReadFence();
            if (m_nextCellReadSequenceNumber + m_cellCount <= LoadRelaxed(&m_pState->nextCellWriteSequenceNumber))
            {
                return false;
            }

            if (readEndpointTsc < cell.readableAfterTsc)
            {
                return false;
            }

            // totalRecordCountBefore counts records written before this cell's write window opened.
            // On the very first cell we read, treat its records as the baseline rather than drops.
            if (m_totalRecordCount < cell.totalRecordCountBefore)
            {
                if (0 < m_totalRecordCount)
                {
                    m_droppedRecordCount += cell.totalRecordCountBefore - m_totalRecordCount;
                }
                m_totalRecordCount = cell.totalRecordCountBefore;
            }

            m_currentCellNextChunkRef = cell.firstChunkRef;

            m_nextCellReadSequenceNumber += 1;
            m_nextCellReadIndex = CircularIncrement(m_nextCellReadIndex, m_cellCount);

            return true;
        }

        bool TryAdvanceToNextChunk(uint64_t cellReadEndSequenceNumber, uint64_t readEndpointTsc)
        {
            while (true)
            {
                if (m_currentCellNextChunkRef == NVTRACE_INVALID_REVISIONED_REF && !TryAdvanceToNextCell(cellReadEndSequenceNumber, readEndpointTsc))
                {
                    return false;
                }

                const NvTrace_RevisionedRef chunkRef = m_currentCellNextChunkRef;
                const auto [pChunkHeader, expectedRevision] = m_chunkPoolReader.ResolveChunkRef(chunkRef);
                if (!pChunkHeader)
                {
                    m_currentCellNextChunkRef = NVTRACE_INVALID_REVISIONED_REF;
                    continue;
                }

                const NvTrace_RevisionedRef nextChunkRef = LoadAcquire(&pChunkHeader->nextChunkRef);
                const uint32_t recordCount = pChunkHeader->recordCount;
                if (m_chunkPoolReader.MaxRecordCountPerChunk() < recordCount)
                {
                    m_currentCellNextChunkRef = NVTRACE_INVALID_REVISIONED_REF;
                    continue;
                }

                ReadFence();
                if (m_chunkPoolReader.TruncateChunkRevision(LoadRelaxed(&pChunkHeader->revision)) != expectedRevision)
                {
                    m_currentCellNextChunkRef = NVTRACE_INVALID_REVISIONED_REF;
                    continue;
                }

                const auto [pBeginRecordDescPos, pEndRecordDescPos, pBeginPayloadPos, maxPayloadEndOffset] = m_chunkPoolReader.ResolveChunkBoundaries(pChunkHeader, recordCount);

                m_currentCellNextChunkRef = nextChunkRef;
                m_pCurrentChunkHeader = pChunkHeader;
                m_pCurrentChunkBeginPayloadPos = pBeginPayloadPos;
                m_currentChunkExpectedRevision = expectedRevision;
                m_currentChunkIncomplete = nextChunkRef == NVTRACE_INVALID_REVISIONED_REF && cellReadEndSequenceNumber <= m_nextCellReadSequenceNumber;

                m_pCurrentRecordDescPos = pBeginRecordDescPos;
                m_pEndRecordDescPos = pEndRecordDescPos;
                m_prevPayloadEndOffset = 0;
                m_maxPayloadEndOffset = maxPayloadEndOffset;
                m_currentRecordCorrelationId = m_chunkPoolReader.CalculateChunkBaseCorrelationId(chunkRef);
                m_tentativelyConsumedRecordCount = 0;

                break;
            }

            return true;
        }

        void RefreshCurrentIncompleteChunk(uint64_t cellReadEndSequenceNumber)
        {
            const NvTrace_RevisionedRef nextChunkRef = LoadAcquire(&m_pCurrentChunkHeader->nextChunkRef);
            const uint32_t recordCount = m_pCurrentChunkHeader->recordCount;
            if (m_chunkPoolReader.MaxRecordCountPerChunk() < recordCount)
            {
                AbandonCurrentCell();
                return;
            }

            if (!CheckConsistency())
            {
                return;
            }

            m_currentCellNextChunkRef = nextChunkRef;
            m_currentChunkIncomplete = nextChunkRef == NVTRACE_INVALID_REVISIONED_REF && cellReadEndSequenceNumber <= m_nextCellReadSequenceNumber;

            const AdaptiveChunkPoolReader::ChunkBoundaries chunkBoundaries = m_chunkPoolReader.ResolveChunkBoundaries(m_pCurrentChunkHeader, recordCount);
            m_pEndRecordDescPos = chunkBoundaries.pEndRecordDescPos;
            m_maxPayloadEndOffset = chunkBoundaries.maxPayloadEndOffset;

            if (m_maxPayloadEndOffset < m_prevPayloadEndOffset)
            {
                AbandonCurrentCell();
                return;
            }
        }

    private:
        const NvTrace_AdaptiveBufferState* m_pState = nullptr;
        const uint8_t* m_pCells = nullptr;
        uint32_t m_cellCount = 0;
        uint32_t m_threeQuartersCellCount = 0;
        size_t m_cellSize = 0;

        AdaptiveChunkPoolReader m_chunkPoolReader;

        uint64_t m_nextCellReadSequenceNumber = 0;
        uint32_t m_nextCellReadIndex = 0;

        NvTrace_RevisionedRef m_currentCellNextChunkRef = NVTRACE_INVALID_REVISIONED_REF;

        const NvTrace_AdaptiveChunkHeader* m_pCurrentChunkHeader = nullptr;
        const uint8_t* m_pCurrentChunkBeginPayloadPos = nullptr;
        uint64_t m_currentChunkExpectedRevision = 0;
        bool m_currentChunkIncomplete = false;

        const uint8_t* m_pCurrentRecordDescPos = nullptr;
        const uint8_t* m_pEndRecordDescPos = nullptr;
        size_t m_prevPayloadEndOffset = 0;
        size_t m_maxPayloadEndOffset = 0;
        uint64_t m_currentRecordCorrelationId = 0;

        uint64_t m_tentativelyConsumedRecordCount = 0;
        uint64_t m_totalRecordCount = 0;
        uint64_t m_droppedRecordCount = 0;
    };

} // namespace nv::trace
