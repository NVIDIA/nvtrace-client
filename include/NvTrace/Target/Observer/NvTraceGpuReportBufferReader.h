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

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nv::trace {

    // Reader of a GPU Report Buffer section: fixed-size records written by a GPU work stream into
    // chunks, named by a seq-protected circular buffer of cells. Report readiness -- the ref
    // derivation and the three-way comparison against it -- is the format's, in
    // nvtrace_provider_buffer.h; it is per record, so no time gate. The cell copy taken at first read
    // stays valid (a published cell never changes), and a lap while parked on it is caught by the
    // ring's sequence re-check.
    //
    // A ready record's bytes are valid because the work stream stores them before its ref store (for the
    // GPU semaphore report the ESC IAS orders the timestamp write before the payload write, and both
    // land in one 64-byte line, so one L2 slice keeps the order) and the ref is read with acquire.
    // That ordering alone cannot cover a *recycled* chunk -- the new work's early stores could pair
    // with a slot's still-unwritten old ref -- so reuse is gated by the pool's per-chunk revision
    // instead: the producing CPU bumps it before the chunk's new work is enqueued, and the per-record
    // verdict below compares the chunk's revision against the one the cell was built with, after a
    // read fence that orders every copied byte of the record before the comparison. The one benign
    // residual is a pad stored in place within the same lap (no revision move), which a racing read
    // can deliver torn -- a producer's pad encoding must make a torn pad harmless. The CUDA provider
    // zeroes a pad's ptimer before its releasing ref store, and the DataModel already drops a
    // zero-ptimer report as a pad.
    //
    // ConsumeRecords(fnReadRecord, fnCommitRecord): drains cells oldest-first, calling
    // fnReadRecord(recordType, pPayload, payloadSize) per ready report. A void callback preserves
    // the original consume-on-delivery behavior; a bool callback may return false to retain the
    // current report for the next poll. pPayload points into the live buffer, so accepted payload
    // bytes are tentative until the fnCommitRecord(bool) that follows: false means the chunk was
    // recycled or the ring overran the reader mid-read, so the tentative report must be discarded,
    // along with the rest of its cell.
    //
    // Thread safety: not internally synchronized; one consumer thread. The producers (CPU and GPU)
    // may be concurrent.
    class GpuReportBufferReader
    {
    public:
        ~GpuReportBufferReader() = default;
        GpuReportBufferReader() = default;

        GpuReportBufferReader(const GpuReportBufferReader&) = delete;
        GpuReportBufferReader(GpuReportBufferReader&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        GpuReportBufferReader& operator=(const GpuReportBufferReader&) = delete;
        GpuReportBufferReader& operator=(GpuReportBufferReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pState = std::exchange(rhs.m_pState, nullptr);
                m_pCells = std::exchange(rhs.m_pCells, nullptr);
                m_cellCount = std::exchange(rhs.m_cellCount, 0);
                m_threeQuartersCellCount = std::exchange(rhs.m_threeQuartersCellCount, 0);
                m_cellSize = std::exchange(rhs.m_cellSize, 0);
                m_recordDesc = std::exchange(rhs.m_recordDesc, {});
                m_reportSize = std::exchange(rhs.m_reportSize, 0);
                m_pChunks = std::exchange(rhs.m_pChunks, nullptr);
                m_chunkCount = std::exchange(rhs.m_chunkCount, 0);
                m_chunkSize = std::exchange(rhs.m_chunkSize, 0);
                m_pChunkStates = std::exchange(rhs.m_pChunkStates, nullptr);
                m_chunkStateSize = std::exchange(rhs.m_chunkStateSize, 0);
                m_reportsPerChunk = std::exchange(rhs.m_reportsPerChunk, 0);
                m_refCodec = std::move(rhs.m_refCodec);
                m_nextReadSequenceNumber = std::exchange(rhs.m_nextReadSequenceNumber, 0);
                m_nextReadCellIndex = std::exchange(rhs.m_nextReadCellIndex, 0);
                m_hasCurrentCell = std::exchange(rhs.m_hasCurrentCell, false);
                m_currentCell = std::exchange(rhs.m_currentCell, {});
                m_nextReportIndex = std::exchange(rhs.m_nextReportIndex, 0);
                m_droppedCellCount = std::exchange(rhs.m_droppedCellCount, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_droppedCellCount = 0;
            m_nextReportIndex = 0;
            m_currentCell = {};
            m_hasCurrentCell = false;
            m_nextReadCellIndex = 0;
            m_nextReadSequenceNumber = 0;
            m_refCodec.Reset();
            m_reportsPerChunk = 0;
            m_chunkStateSize = 0;
            m_pChunkStates = nullptr;
            m_chunkSize = 0;
            m_chunkCount = 0;
            m_pChunks = nullptr;
            m_reportSize = 0;
            m_recordDesc = {};
            m_cellSize = 0;
            m_threeQuartersCellCount = 0;
            m_cellCount = 0;
            m_pCells = nullptr;
            m_pState = nullptr;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t gpuReportBufferSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            const auto [pSection, sectionSize, sectionHeaderSize, sectionHeaderMinorVersion] = fnResolveSection(gpuReportBufferSectionIndex, NvTrace_ProviderBufferSectionType_GpuReportBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, false, 50, "GpuReportBufferReader: failed to resolve GPU-report-buffer section (gpuReportBufferSectionIndex %u)", gpuReportBufferSectionIndex);

            SectionLayoutValidator layoutValidator(sectionSize);

            bool succeeded = layoutValidator.AddAndValidateSection(0, sectionHeaderSize, sectionHeaderMinorVersion, NvTrace_GpuReportBufferSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid section header layout (size %u, minorVersion %u)", sectionHeaderSize, sectionHeaderMinorVersion);

            const NvTrace_GpuReportBufferSectionHeader* const pSectionHeader = reinterpret_cast<const NvTrace_GpuReportBufferSectionHeader*>(pSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pSectionHeader) % alignof(NvTrace_GpuReportBufferSectionHeader)) != 0, false, 50, "GpuReportBufferReader: section header must be %zu-byte aligned (pSectionHeader %p)", alignof(NvTrace_GpuReportBufferSectionHeader), pSectionHeader);

            succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->gpuReportBufferStateOffsetInSection, pSectionHeader->gpuReportBufferStateSize, pSectionHeader->gpuReportBufferStateMinorVersion, NvTrace_GpuReportBufferState_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid state layout (offset %" PRIu64 ", size %u, minorVersion %u)", pSectionHeader->gpuReportBufferStateOffsetInSection, pSectionHeader->gpuReportBufferStateSize, pSectionHeader->gpuReportBufferStateMinorVersion);

            const NvTrace_GpuReportBufferState* const pState = reinterpret_cast<const NvTrace_GpuReportBufferState*>(pSection + pSectionHeader->gpuReportBufferStateOffsetInSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pState) % alignof(NvTrace_GpuReportBufferState)) != 0, false, 50, "GpuReportBufferReader: state must be %zu-byte aligned (pState %p)", alignof(NvTrace_GpuReportBufferState), pState);

            succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->gpuReportBufferCellsOffsetInSection, pSectionHeader->gpuReportBufferCellCount, pSectionHeader->gpuReportBufferCellSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid cells array layout (offset %" PRIu64 ", cellCount %u, cellSize %u)", pSectionHeader->gpuReportBufferCellsOffsetInSection, pSectionHeader->gpuReportBufferCellCount, pSectionHeader->gpuReportBufferCellSize);
            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->gpuReportBufferCellCount == 0, false, 50, "GpuReportBufferReader: cellCount must be greater than 0");
            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->gpuReportBufferCellMinorVersion < NvTrace_GpuReportBufferCell_MINOR_VERSION_A || pSectionHeader->gpuReportBufferCellSize < pSectionHeader->gpuReportBufferCellMinorVersion, false, 50, "GpuReportBufferReader: invalid cell struct (gpuReportBufferCellSize %u, gpuReportBufferCellMinorVersion %u)", pSectionHeader->gpuReportBufferCellSize, pSectionHeader->gpuReportBufferCellMinorVersion);

            const uint8_t* const pCells = pSection + pSectionHeader->gpuReportBufferCellsOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pCells) % alignof(uint64_t)) != 0 || (pSectionHeader->gpuReportBufferCellSize % alignof(uint64_t)) != 0, false, 50, "GpuReportBufferReader: cells must be %zu-byte aligned (pCells %p, cellSize %u)", alignof(uint64_t), pCells, pSectionHeader->gpuReportBufferCellSize);

            const auto [pChunkPoolSection, chunkPoolSectionSize, chunkPoolSectionHeaderSize, chunkPoolSectionHeaderMinorVersion] = fnResolveSection(pSectionHeader->gpuReportChunkPoolSectionIndex, NvTrace_ProviderBufferSectionType_GpuReportChunkPool);
            NVTRACE_LOG_ERR_RETURN_IF(!pChunkPoolSection, false, 50, "GpuReportBufferReader: failed to resolve GPU-report-chunk-pool section (gpuReportChunkPoolSectionIndex %u)", pSectionHeader->gpuReportChunkPoolSectionIndex);

            SectionLayoutValidator chunkPoolValidator(chunkPoolSectionSize);
            succeeded = chunkPoolValidator.AddAndValidateSection(0, chunkPoolSectionHeaderSize, chunkPoolSectionHeaderMinorVersion, NvTrace_GpuReportChunkPoolSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid chunk-pool section header layout (size %u, minorVersion %u)", chunkPoolSectionHeaderSize, chunkPoolSectionHeaderMinorVersion);

            const NvTrace_GpuReportChunkPoolSectionHeader* const pChunkPoolHeader = reinterpret_cast<const NvTrace_GpuReportChunkPoolSectionHeader*>(pChunkPoolSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pChunkPoolHeader) % alignof(NvTrace_GpuReportChunkPoolSectionHeader)) != 0, false, 50, "GpuReportBufferReader: chunk-pool section header must be %zu-byte aligned (pChunkPoolHeader %p)", alignof(NvTrace_GpuReportChunkPoolSectionHeader), pChunkPoolHeader);

            succeeded = chunkPoolValidator.AddAndValidateSection(pChunkPoolHeader->chunksOffsetInSection, pChunkPoolHeader->chunkCount, pChunkPoolHeader->chunkSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid chunks array layout (offset %" PRIu64 ", chunkCount %u, chunkSize %u)", pChunkPoolHeader->chunksOffsetInSection, pChunkPoolHeader->chunkCount, pChunkPoolHeader->chunkSize);

            const uint8_t* const pChunks = pChunkPoolSection + pChunkPoolHeader->chunksOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pChunks) % alignof(uint64_t)) != 0 || (pChunkPoolHeader->chunkSize % alignof(uint64_t)) != 0, false, 50, "GpuReportBufferReader: chunks must be %zu-byte aligned (pChunks %p, chunkSize %u)", alignof(uint64_t), pChunks, pChunkPoolHeader->chunkSize);

            succeeded = chunkPoolValidator.AddAndValidateSection(pChunkPoolHeader->recordDescOffsetInSection, pChunkPoolHeader->recordDescSize, pChunkPoolHeader->recordDescMinorVersion, NvTrace_RecordDesc_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid record desc layout (offset %" PRIu64 ", size %u, minorVersion %u)", pChunkPoolHeader->recordDescOffsetInSection, pChunkPoolHeader->recordDescSize, pChunkPoolHeader->recordDescMinorVersion);

            const NvTrace_RecordDesc* const pRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(pChunkPoolSection + pChunkPoolHeader->recordDescOffsetInSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pRecordDesc) % alignof(NvTrace_RecordDesc)) != 0, false, 50, "GpuReportBufferReader: record desc must be %zu-byte aligned (pRecordDesc %p)", alignof(NvTrace_RecordDesc), pRecordDesc);

            // A chunk is a dense array of reportSize-byte records, and the leading word of every record
            // is its report ref.
            const NvTrace_RecordDesc recordDesc = *pRecordDesc;
            NVTRACE_LOG_ERR_RETURN_IF(recordDesc.payloadEndOffsetDiv8 == 0, false, 50, "GpuReportBufferReader: report payload must cover the leading report ref");
            const size_t reportSize = recordDesc.payloadEndOffsetDiv8 * sizeof(uint64_t);

            // Division, not a multiply, so an overstated count cannot overflow its way past the check.
            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->recordsPerChunk == 0 || pChunkPoolHeader->chunkSize / reportSize < pChunkPoolHeader->recordsPerChunk, false, 50, "GpuReportBufferReader: recordsPerChunk (%u) does not fit a chunk (chunkSize %u, reportSize %zu)", pChunkPoolHeader->recordsPerChunk, pChunkPoolHeader->chunkSize, reportSize);

            NVTRACE_LOG_ERR_RETURN_IF(pChunkPoolHeader->gpuReportChunkStateMinorVersion < NvTrace_GpuReportChunkState_MINOR_VERSION_A || pChunkPoolHeader->gpuReportChunkStateSize < pChunkPoolHeader->gpuReportChunkStateMinorVersion, false, 50, "GpuReportBufferReader: invalid chunk state struct (gpuReportChunkStateSize %u, gpuReportChunkStateMinorVersion %u)", pChunkPoolHeader->gpuReportChunkStateSize, pChunkPoolHeader->gpuReportChunkStateMinorVersion);

            succeeded = chunkPoolValidator.AddAndValidateSection(pChunkPoolHeader->gpuReportChunkStatesOffsetInSection, pChunkPoolHeader->chunkCount, pChunkPoolHeader->gpuReportChunkStateSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "GpuReportBufferReader: invalid chunk states array layout (offset %" PRIu64 ", chunkCount %u, gpuReportChunkStateSize %u)", pChunkPoolHeader->gpuReportChunkStatesOffsetInSection, pChunkPoolHeader->chunkCount, pChunkPoolHeader->gpuReportChunkStateSize);

            const uint8_t* const pChunkStates = pChunkPoolSection + pChunkPoolHeader->gpuReportChunkStatesOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pChunkStates) % alignof(NvTrace_GpuReportChunkState)) != 0 || (pChunkPoolHeader->gpuReportChunkStateSize % alignof(NvTrace_GpuReportChunkState)) != 0, false, 50, "GpuReportBufferReader: chunk states must be %zu-byte aligned (pChunkStates %p, gpuReportChunkStateSize %u)", alignof(NvTrace_GpuReportChunkState), pChunkStates, pChunkPoolHeader->gpuReportChunkStateSize);

            m_pState = pState;
            m_pCells = pCells;
            m_cellCount = pSectionHeader->gpuReportBufferCellCount;
            m_threeQuartersCellCount = m_cellCount < 4 ? 1 : static_cast<uint32_t>((static_cast<uint64_t>(m_cellCount) * 3) / 4);
            m_cellSize = pSectionHeader->gpuReportBufferCellSize;
            m_recordDesc = recordDesc;
            m_reportSize = reportSize;
            m_pChunks = pChunks;
            m_chunkCount = pChunkPoolHeader->chunkCount;
            m_chunkSize = static_cast<size_t>(pChunkPoolHeader->chunkSize);
            m_pChunkStates = pChunkStates;
            m_chunkStateSize = pChunkPoolHeader->gpuReportChunkStateSize;
            m_reportsPerChunk = pChunkPoolHeader->recordsPerChunk;
            m_refCodec.Initialize(pChunkPoolHeader->chunkCount, pChunkPoolHeader->recordsPerChunk);

            // Start reading from a safe oldest valid element, ~75% behind the write pointer.
            const uint64_t nextCellWriteSequenceNumber = LoadRelaxed(&m_pState->nextCellWriteSequenceNumber);
            m_nextReadSequenceNumber = m_threeQuartersCellCount < nextCellWriteSequenceNumber ? nextCellWriteSequenceNumber - m_threeQuartersCellCount : 0;
            m_nextReadCellIndex = static_cast<uint32_t>(m_nextReadSequenceNumber % m_cellCount);

            return true;
        }

        bool IsInitialized() const
        {
            return m_pState != nullptr;
        }

        uint32_t CellCount() const
        {
            return m_cellCount;
        }

        uint64_t DroppedCellCount() const
        {
            return m_droppedCellCount;
        }

        //   void fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //     Accepts every delivered report.
        //
        //   bool fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //     Return true to accept the report and advance the cursor.
        //     Return false to retain the report and stop until the next ConsumeRecords().
        //
        //   void fnCommitRecord(bool consistencyCheckPassed)
        //     Called once per accepted report. If false, roll back state staged by fnReadRecord.
        // A retained report remains available for a later call unless the
        // producer laps the reader first.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeRecords(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            uint64_t nextCellWriteSequenceNumber = LoadAcquire(&m_pState->nextCellWriteSequenceNumber);
            if (m_nextReadSequenceNumber + m_cellCount <= nextCellWriteSequenceNumber)
            {
                RecoverFromBufferOverrun(nextCellWriteSequenceNumber);
            }

            while (true)
            {
                if (!m_hasCurrentCell)
                {
                    if (nextCellWriteSequenceNumber <= m_nextReadSequenceNumber)
                    {
                        return;
                    }

                    // Clamped to the published cellSize: a producer built against an older cell
                    // struct publishes a smaller one, and a whole-struct read would run past its cell.
                    m_currentCell = {};
                    Memcpy64(&m_currentCell, m_pCells + m_nextReadCellIndex * m_cellSize, std::min(m_cellSize, sizeof(m_currentCell)));
                    m_hasCurrentCell = true;
                    m_nextReportIndex = 0;
                }

                const CellDrainResult drainResult = DrainCurrentCell(fnReadRecord, fnCommitRecord);
                if (drainResult == CellDrainResult::Blocked)
                {
                    return;
                }

                m_hasCurrentCell = false;
                m_nextReadSequenceNumber += 1;
                m_nextReadCellIndex = CircularIncrement(m_nextReadCellIndex);

                if (drainResult == CellDrainResult::Discarded)
                {
                    return;
                }

                ReadFence();
                nextCellWriteSequenceNumber = LoadRelaxed(&m_pState->nextCellWriteSequenceNumber);
            }
        }

    private:
        enum class CellDrainResult
        {
            Drained,  // The cell is finished; anything delivered this call was committed.
            Blocked,  // A report is not ready or was rejected; the cell is parked and resumed on a later poll.
            Discarded // The chunk was recycled or the ring overran the reader mid-read; the tentative record was rolled back.
        };

        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        CellDrainResult DrainCurrentCell(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            const uint64_t chunkIndex = m_refCodec.DecodeItemRef(m_currentCell.chunkRef).itemIndex;

            // An out-of-range chunk under a consistent ring is stable garbage, not a race: drop the cell
            // before any report is interpreted.
            if (m_chunkCount <= chunkIndex)
            {
                if (CheckConsistency())
                {
                    NVTRACE_LOG_ERR(50, "GpuReportBufferReader: cell names an out-of-range chunk (chunkIndex %" PRIu64 ", chunkCount %" PRIu64 ")", chunkIndex, m_chunkCount);
                    return CellDrainResult::Drained;
                }
                return CellDrainResult::Discarded;
            }

            const uint8_t* const pChunk = m_pChunks + static_cast<size_t>(chunkIndex) * m_chunkSize;
            const uint64_t firstReportRef = m_refCodec.EncodeRecordRef(m_currentCell.chunkRef, 0);

            CellDrainResult drainResult = CellDrainResult::Drained;
            while (m_nextReportIndex < m_reportsPerChunk)
            {
                const uint8_t* const pReport = pChunk + static_cast<size_t>(m_nextReportIndex) * m_reportSize;
                const uint64_t expectedReportRef = firstReportRef + m_nextReportIndex;
                const uint64_t storedReportRef = LoadAcquire(reinterpret_cast<const volatile uint64_t*>(pReport));

                if (storedReportRef < expectedReportRef)
                {
                    drainResult = CellDrainResult::Blocked;
                    break;
                }
                if (expectedReportRef < storedReportRef)
                {
                    m_nextReportIndex = m_reportsPerChunk;
                    break;
                }

                bool recordAccepted = true;
                if constexpr (std::is_same_v<std::invoke_result_t<TFN_ReadRecord, uint32_t, const uint8_t*, size_t>, void>)
                {
                    fnReadRecord(m_recordDesc.recordType, pReport, m_reportSize);
                }
                else
                {
                    recordAccepted = static_cast<bool>(fnReadRecord(m_recordDesc.recordType, pReport, m_reportSize));
                }

                // One fence orders every byte the record copied before both validity loads: the chunk
                // revision proves no recycle touched the chunk mid-read, and the ring sequence proves
                // the cell itself was not lapped.
                ReadFence();
                const bool chunkIntact = m_refCodec.EncodeItemRef(chunkIndex, LoadRelaxed(ChunkRevisionAt(chunkIndex))) == m_currentCell.chunkRef;
                const bool cellIntact = LoadRelaxed(&m_pState->nextCellWriteSequenceNumber) < m_nextReadSequenceNumber + m_cellCount;
                if (!chunkIntact || !cellIntact)
                {
                    if (recordAccepted)
                    {
                        fnCommitRecord(false);
                    }
                    return CellDrainResult::Discarded;
                }

                if (!recordAccepted)
                {
                    return CellDrainResult::Blocked;
                }

                fnCommitRecord(true);
                m_nextReportIndex += 1;
            }

            return drainResult;
        }

        const volatile uint64_t* ChunkRevisionAt(uint64_t chunkIndex) const
        {
            return &reinterpret_cast<const NvTrace_GpuReportChunkState*>(m_pChunkStates + static_cast<size_t>(chunkIndex) * m_chunkStateSize)->revision;
        }

        bool CheckConsistency()
        {
            ReadFence();
            return LoadRelaxed(&m_pState->nextCellWriteSequenceNumber) < m_nextReadSequenceNumber + m_cellCount;
        }

        uint32_t CircularIncrement(uint32_t index) const
        {
            const uint32_t nextIndex = index + 1;
            return nextIndex == m_cellCount ? 0 : nextIndex;
        }

        void RecoverFromBufferOverrun(uint64_t nextCellWriteSequenceNumber)
        {
            const uint64_t recoveredReadSequenceNumber = nextCellWriteSequenceNumber - m_threeQuartersCellCount;
            m_droppedCellCount += (recoveredReadSequenceNumber - m_nextReadSequenceNumber);

            m_nextReadSequenceNumber = recoveredReadSequenceNumber;
            m_nextReadCellIndex = static_cast<uint32_t>(m_nextReadSequenceNumber % m_cellCount);
            m_hasCurrentCell = false;
            m_nextReportIndex = 0;
        }

    private:
        const NvTrace_GpuReportBufferState* m_pState = nullptr;
        const uint8_t* m_pCells = nullptr;
        uint32_t m_cellCount = 0;
        uint32_t m_threeQuartersCellCount = 0;
        size_t m_cellSize = 0;

        NvTrace_RecordDesc m_recordDesc = {};
        size_t m_reportSize = 0;

        const uint8_t* m_pChunks = nullptr;
        uint64_t m_chunkCount = 0;
        size_t m_chunkSize = 0;
        const uint8_t* m_pChunkStates = nullptr;
        size_t m_chunkStateSize = 0;
        uint32_t m_reportsPerChunk = 0;
        RevisionedRefCodec m_refCodec;

        uint64_t m_nextReadSequenceNumber = 0;
        uint32_t m_nextReadCellIndex = 0;

        // The cell being drained, copied out once; a cell outliving a poll resumes here.
        bool m_hasCurrentCell = false;
        NvTrace_GpuReportBufferCell m_currentCell = {};
        uint32_t m_nextReportIndex = 0;

        uint64_t m_droppedCellCount = 0;
    };

} // namespace nv::trace
