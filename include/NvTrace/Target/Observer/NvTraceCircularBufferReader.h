/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/Observer/NvTraceCallableTraits.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nv::trace {

    // Reader of a circular buffer that exposes payloads as opaque byte pointers. In fixed mode every
    // cell carries the section's one NvTrace_RecordDesc (fixedRecordDescOffsetInSection is valid); in
    // variable mode each cell carries its own at variableRecordDescOffsetInCell. Initialize rejects a
    // header that violates the mutual-exclusion invariant (exactly one sentinel is INVALID).
    //
    // Producer/consumer contract: (a) a per-record timestamp gate
    // (NvTrace_CircularBufferSectionHeader::recordTimestampOffsetInCell) lets the reader skip
    // records that are too recent, and (b) sequence-number consistency check detects if the
    // writer lapped the reader during a read.
    //
    // The gate timestamp's clock domain is declared by NvTrace_CircularBufferSectionHeader::recordTimestampType
    // (Tsc / GpuPtimer); the caller supplies a read endpoint in the same domain (RecordTimestampType() reports it).
    //
    // Thread safety: the reader is not internally synchronized and must be used from a single
    // consumer thread. The producer side may be concurrent.
    //
    // ConsumeRecords(readEndpointTimestamp, fnReadRecord, fnCommitRecord):
    //   - Walks records up to the writer's current position whose gate timestamp <=
    //     readEndpointTimestamp, calling fnReadRecord(sequenceNumber, recordType, pPayload, payloadSize)
    //     for each. Pass UINT64_MAX as readEndpointTimestamp to ignore the timestamp gate. Accepted
    //     payload bytes are tentative until the fnCommitRecord(bool valid) that follows:
    //       - valid == true:  the record is consistent and may be acted upon irreversibly.
    //       - valid == false: the writer overran the reader; the tentative record must be
    //                         discarded. The reader re-anchors on the next call.
    //   - fnReadRecord may optionally return a bool: returning false signals an early-exit; the
    //     rejected record is re-delivered on the next call (unless the buffer is then found
    //     overrun). The early-exit's fnCommitRecord covers no record.
    //
    // PeekLatestRecord(fnReadRecord): snapshots the most recently written record without advancing the
    //   consume cursor; returns whether the snapshot was consistent (writer did not lap it during the read).
    class CircularBufferReader
    {
    public:
        ~CircularBufferReader() = default;
        CircularBufferReader() noexcept = default;

        CircularBufferReader(const CircularBufferReader&) = delete;
        CircularBufferReader(CircularBufferReader&& rhs) noexcept
            : m_pState(std::exchange(rhs.m_pState, nullptr))
            , m_pCells(std::exchange(rhs.m_pCells, nullptr))
            , m_cellCount(std::exchange(rhs.m_cellCount, 0))
            , m_threeQuartersCellCount(std::exchange(rhs.m_threeQuartersCellCount, 0))
            , m_cellSize(std::exchange(rhs.m_cellSize, 0))
            , m_hasFixedRecordDesc(std::exchange(rhs.m_hasFixedRecordDesc, false))
            , m_fixedRecordDesc(std::exchange(rhs.m_fixedRecordDesc, {}))
            , m_variableRecordDescOffsetInCell(std::exchange(rhs.m_variableRecordDescOffsetInCell, NVTRACE_INVALID_OFFSET32))
            , m_payloadOffsetInCell(std::exchange(rhs.m_payloadOffsetInCell, 0))
            , m_maxPayloadSizeDiv8(std::exchange(rhs.m_maxPayloadSizeDiv8, 0))
            , m_recordTimestampType(std::exchange(rhs.m_recordTimestampType, NvTrace_CircularBufferRecordTimestampType_Invalid))
            , m_recordTimestampOffsetInCell(std::exchange(rhs.m_recordTimestampOffsetInCell, 0))
            , m_nextReadSequenceNumber(std::exchange(rhs.m_nextReadSequenceNumber, 0))
            , m_nextReadCellIndex(std::exchange(rhs.m_nextReadCellIndex, 0))
            , m_unvalidatedReadSequenceNumber(std::exchange(rhs.m_unvalidatedReadSequenceNumber, 0))
            , m_observationStartTimestamp(std::exchange(rhs.m_observationStartTimestamp, 0))
            , m_retainAtLeastOneRecord(std::exchange(rhs.m_retainAtLeastOneRecord, false))
            , m_droppedRecordCount(std::exchange(rhs.m_droppedRecordCount, 0))
        {
        }

        CircularBufferReader& operator=(const CircularBufferReader&) = delete;
        CircularBufferReader& operator=(CircularBufferReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pState = std::exchange(rhs.m_pState, nullptr);
                m_pCells = std::exchange(rhs.m_pCells, nullptr);
                m_cellCount = std::exchange(rhs.m_cellCount, 0);
                m_threeQuartersCellCount = std::exchange(rhs.m_threeQuartersCellCount, 0);
                m_cellSize = std::exchange(rhs.m_cellSize, 0);
                m_hasFixedRecordDesc = std::exchange(rhs.m_hasFixedRecordDesc, false);
                m_fixedRecordDesc = std::exchange(rhs.m_fixedRecordDesc, {});
                m_variableRecordDescOffsetInCell = std::exchange(rhs.m_variableRecordDescOffsetInCell, NVTRACE_INVALID_OFFSET32);
                m_payloadOffsetInCell = std::exchange(rhs.m_payloadOffsetInCell, 0);
                m_maxPayloadSizeDiv8 = std::exchange(rhs.m_maxPayloadSizeDiv8, 0);
                m_recordTimestampType = std::exchange(rhs.m_recordTimestampType, NvTrace_CircularBufferRecordTimestampType_Invalid);
                m_recordTimestampOffsetInCell = std::exchange(rhs.m_recordTimestampOffsetInCell, 0);
                m_nextReadSequenceNumber = std::exchange(rhs.m_nextReadSequenceNumber, 0);
                m_nextReadCellIndex = std::exchange(rhs.m_nextReadCellIndex, 0);
                m_unvalidatedReadSequenceNumber = std::exchange(rhs.m_unvalidatedReadSequenceNumber, 0);
                m_observationStartTimestamp = std::exchange(rhs.m_observationStartTimestamp, 0);
                m_retainAtLeastOneRecord = std::exchange(rhs.m_retainAtLeastOneRecord, false);
                m_droppedRecordCount = std::exchange(rhs.m_droppedRecordCount, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_droppedRecordCount = 0;
            m_retainAtLeastOneRecord = false;
            m_observationStartTimestamp = 0;
            m_unvalidatedReadSequenceNumber = 0;
            m_nextReadCellIndex = 0;
            m_nextReadSequenceNumber = 0;
            m_recordTimestampOffsetInCell = 0;
            m_recordTimestampType = NvTrace_CircularBufferRecordTimestampType_Invalid;
            m_maxPayloadSizeDiv8 = 0;
            m_payloadOffsetInCell = 0;
            m_variableRecordDescOffsetInCell = NVTRACE_INVALID_OFFSET32;
            m_fixedRecordDesc = {};
            m_hasFixedRecordDesc = false;
            m_cellSize = 0;
            m_threeQuartersCellCount = 0;
            m_cellCount = 0;
            m_pCells = nullptr;
            m_pState = nullptr;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t circularBufferSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            const auto [pSection, sectionSize, sectionHeaderSize, sectionHeaderMinorVersion] = fnResolveSection(circularBufferSectionIndex, NvTrace_ProviderBufferSectionType_CircularBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, false, 50, "CircularBufferReader: failed to resolve circular-buffer section (circularBufferSectionIndex %u)", circularBufferSectionIndex);

            SectionLayoutValidator layoutValidator(sectionSize);

            bool succeeded = layoutValidator.AddAndValidateSection(0, sectionHeaderSize, sectionHeaderMinorVersion, NvTrace_CircularBufferSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "CircularBufferReader: invalid section header layout (size %u, minorVersion %u)", sectionHeaderSize, sectionHeaderMinorVersion);

            const NvTrace_CircularBufferSectionHeader* pSectionHeader = reinterpret_cast<const NvTrace_CircularBufferSectionHeader*>(pSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pSectionHeader) % alignof(NvTrace_CircularBufferSectionHeader)) != 0, false, 50, "CircularBufferReader: section header must be %zu-byte aligned (pSectionHeader %p)", alignof(NvTrace_CircularBufferSectionHeader), pSectionHeader);

            const bool hasFixedRecordDesc = pSectionHeader->fixedRecordDescOffsetInSection != NVTRACE_INVALID_OFFSET64;
            const bool hasVariableRecordDesc = pSectionHeader->variableRecordDescOffsetInCell != NVTRACE_INVALID_OFFSET32;
            NVTRACE_LOG_ERR_RETURN_IF(hasFixedRecordDesc == hasVariableRecordDesc, false, 50, "CircularBufferReader: exactly one of a per-cell record desc and a section-wide fixed one must be published (variableRecordDescOffsetInCell %u, fixedRecordDescOffsetInSection %" PRIu64 ")", pSectionHeader->variableRecordDescOffsetInCell, pSectionHeader->fixedRecordDescOffsetInSection);

            succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->circularBufferStateOffsetInSection, pSectionHeader->circularBufferStateSize, pSectionHeader->circularBufferStateMinorVersion, NvTrace_CircularBufferState2_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "CircularBufferReader: invalid circular buffer state layout (offset %" PRIu64 ", size %u, minorVersion %u)", pSectionHeader->circularBufferStateOffsetInSection, pSectionHeader->circularBufferStateSize, pSectionHeader->circularBufferStateMinorVersion);

            succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->cellsOffsetInSection, pSectionHeader->cellCount, pSectionHeader->cellSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "CircularBufferReader: invalid cells array layout (offset %" PRIu64 ", cellCount %u, cellSize %u)", pSectionHeader->cellsOffsetInSection, pSectionHeader->cellCount, pSectionHeader->cellSize);

            const NvTrace_CircularBufferState2* pState = reinterpret_cast<const NvTrace_CircularBufferState2*>(pSection + pSectionHeader->circularBufferStateOffsetInSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pState) % alignof(NvTrace_CircularBufferState2)) != 0, false, 50, "CircularBufferReader: state must be %zu-byte aligned (pState %p)", alignof(NvTrace_CircularBufferState2), pState);

            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->cellCount == 0, false, 50, "CircularBufferReader: cellCount must be greater than 0");
            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->cellSize < pSectionHeader->payloadOffsetInCell || pSectionHeader->cellSize - pSectionHeader->payloadOffsetInCell < pSectionHeader->maxPayloadSizeDiv8 * sizeof(uint64_t), false, 50, "CircularBufferReader: payload does not fit within cell (cellSize %u, payloadOffsetInCell %u, maxPayloadSizeDiv8 %u)", pSectionHeader->cellSize, pSectionHeader->payloadOffsetInCell, pSectionHeader->maxPayloadSizeDiv8);

            const uint8_t* pCells = pSection + pSectionHeader->cellsOffsetInSection;
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pCells) % alignof(uint64_t)) != 0 || (pSectionHeader->cellSize % alignof(uint64_t)) != 0, false, 50, "CircularBufferReader: cells must be %zu-byte aligned (pCells %p, cellSize %u)", alignof(uint64_t), pCells, pSectionHeader->cellSize);

            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->recordTimestampType == NvTrace_CircularBufferRecordTimestampType_Invalid, false, 50, "CircularBufferReader: recordTimestampType must declare a clock domain (Tsc or GpuPtimer)");
            NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->cellSize < sizeof(uint64_t) || pSectionHeader->cellSize - sizeof(uint64_t) < pSectionHeader->recordTimestampOffsetInCell, false, 50, "CircularBufferReader: recordTimestamp does not fit within cell (cellSize %u, recordTimestampOffsetInCell %u)", pSectionHeader->cellSize, pSectionHeader->recordTimestampOffsetInCell);
            NVTRACE_LOG_ERR_RETURN_IF((pSectionHeader->recordTimestampOffsetInCell % alignof(uint64_t)) != 0, false, 50, "CircularBufferReader: recordTimestampOffsetInCell (%u) must be %zu-byte aligned", pSectionHeader->recordTimestampOffsetInCell, alignof(uint64_t));

            NvTrace_RecordDesc fixedRecordDesc = {};
            uint32_t variableRecordDescOffsetInCell = NVTRACE_INVALID_OFFSET32;
            if (hasFixedRecordDesc)
            {
                succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->fixedRecordDescOffsetInSection, pSectionHeader->recordDescSize, pSectionHeader->recordDescMinorVersion, NvTrace_RecordDesc_MINOR_VERSION_A);
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "CircularBufferReader: invalid fixed record desc layout (offset %" PRIu64 ", size %u, minorVersion %u)", pSectionHeader->fixedRecordDescOffsetInSection, pSectionHeader->recordDescSize, pSectionHeader->recordDescMinorVersion);

                const NvTrace_RecordDesc* const pFixedRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(pSection + pSectionHeader->fixedRecordDescOffsetInSection);
                NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pFixedRecordDesc) % alignof(NvTrace_RecordDesc)) != 0, false, 50, "CircularBufferReader: fixed record desc must be %zu-byte aligned (pFixedRecordDesc %p)", alignof(NvTrace_RecordDesc), pFixedRecordDesc);

                // A section-wide desc is written before any consumer attaches, so bounding its payload
                // here is what spares every cell read the check.
                fixedRecordDesc = *pFixedRecordDesc;
                NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->maxPayloadSizeDiv8 < fixedRecordDesc.payloadEndOffsetDiv8, false, 50, "CircularBufferReader: fixed record desc payloadEndOffsetDiv8 (%u) exceeds maxPayloadSizeDiv8 (%u)", fixedRecordDesc.payloadEndOffsetDiv8, pSectionHeader->maxPayloadSizeDiv8);
            }
            else // if (hasVariableRecordDesc)
            {
                NVTRACE_LOG_ERR_RETURN_IF(pSectionHeader->recordDescMinorVersion < NvTrace_RecordDesc_MINOR_VERSION_A, false, 50, "CircularBufferReader: recordDescMinorVersion (%u) is below the required minimum (%u)", pSectionHeader->recordDescMinorVersion, NvTrace_RecordDesc_MINOR_VERSION_A);
                NVTRACE_LOG_ERR_RETURN_IF((pSectionHeader->variableRecordDescOffsetInCell % alignof(NvTrace_RecordDesc)) != 0 || pSectionHeader->cellSize < pSectionHeader->variableRecordDescOffsetInCell || pSectionHeader->recordDescSize < pSectionHeader->recordDescMinorVersion || pSectionHeader->cellSize - pSectionHeader->variableRecordDescOffsetInCell < pSectionHeader->recordDescSize, false, 50, "CircularBufferReader: record desc does not fit within the cell (variableRecordDescOffsetInCell %u, recordDescSize %u, cellSize %u)", pSectionHeader->variableRecordDescOffsetInCell, pSectionHeader->recordDescSize, pSectionHeader->cellSize);

                variableRecordDescOffsetInCell = pSectionHeader->variableRecordDescOffsetInCell;
            }

            m_pState = pState;
            m_pCells = pCells;
            m_cellCount = pSectionHeader->cellCount;
            m_threeQuartersCellCount = m_cellCount < 4 ? 1 : static_cast<uint32_t>((static_cast<uint64_t>(m_cellCount) * 3) / 4);
            m_cellSize = pSectionHeader->cellSize;
            m_hasFixedRecordDesc = hasFixedRecordDesc;
            m_fixedRecordDesc = fixedRecordDesc;
            m_variableRecordDescOffsetInCell = variableRecordDescOffsetInCell;
            m_payloadOffsetInCell = pSectionHeader->payloadOffsetInCell;
            m_maxPayloadSizeDiv8 = pSectionHeader->maxPayloadSizeDiv8;
            m_recordTimestampType = pSectionHeader->recordTimestampType;
            m_recordTimestampOffsetInCell = pSectionHeader->recordTimestampOffsetInCell;

            // Start reading from a safe oldest valid element.
            // A simple heuristic is ~ 75% behind the write pointer == 25% ahead.
            const uint64_t nextWriteSequenceNumber = LoadRelaxed(&m_pState->nextWriteSequenceNumber);
            m_nextReadSequenceNumber = m_threeQuartersCellCount < nextWriteSequenceNumber ? nextWriteSequenceNumber - m_threeQuartersCellCount : 0;
            m_nextReadCellIndex = static_cast<uint32_t>(m_nextReadSequenceNumber % m_cellCount);
            m_unvalidatedReadSequenceNumber = m_nextReadSequenceNumber;

            return true;
        }

        uint32_t CellCount() const
        {
            return m_cellCount;
        }

        NvTrace_CircularBufferRecordTimestampType RecordTimestampType() const
        {
            return m_recordTimestampType;
        }

        uint64_t DroppedRecordCount() const
        {
            return m_droppedRecordCount;
        }

        uint64_t UnreadRecordCount() const
        {
            return LoadAcquire(&m_pState->nextWriteSequenceNumber) - m_nextReadSequenceNumber;
        }

        void SetObservationStartTimestamp(uint64_t observationStartTimestamp, bool retainAtLeastOneRecord = false)
        {
            m_observationStartTimestamp = observationStartTimestamp;
            m_retainAtLeastOneRecord = retainAtLeastOneRecord;
        }

        // fnReadRecord may use either argument list below and may return void or bool:
        //   [void|bool] fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //   [void|bool] fnReadRecord(uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //
        // A void callback accepts every record.
        // A bool callback returns true to accept the record or false to retain it and stop until the next ConsumeRecords().
        //
        //   void fnCommitRecord(bool consistencyCheckPassed)
        //     Called once per accepted record. If false, roll back state staged by fnReadRecord.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeRecords(uint64_t readEndpointTimestamp, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            const uint64_t readEndSequenceNumber = LoadAcquire(&m_pState->nextWriteSequenceNumber);
            if (m_unvalidatedReadSequenceNumber + m_cellCount <= readEndSequenceNumber)
            {
                RecoverFromBufferOverrun(readEndSequenceNumber);
            }
            if (!ObservationStartReached(readEndpointTimestamp, readEndSequenceNumber))
            {
                return;
            }

            while (m_nextReadSequenceNumber < readEndSequenceNumber)
            {
                const uint8_t* const pCell = m_pCells + m_nextReadCellIndex * m_cellSize;
                // Note: this reads the timestamp before the record's consistency check, so on a cell the writer is
                // concurrently overwriting *pCellTimestamp may be garbage. That is harmless -- a bad timestamp only
                // causes an early break or an extra staged record, and the record is replayed only if the
                // CheckConsistency() below confirms the writer did not overrun this cell.
                const uint64_t* const pCellTimestamp = reinterpret_cast<const uint64_t*>(pCell + m_recordTimestampOffsetInCell);
                if (readEndpointTimestamp < *pCellTimestamp)
                {
                    CheckConsistency();
                    return;
                }
                if (!InvokeReadRecord(fnReadRecord, pCell, m_nextReadSequenceNumber))
                {
                    // A rejected record remains at the cursor and owns no tentative consumer
                    // state, so only update the reader's consistency bookkeeping here.
                    CheckConsistency();
                    return;
                }

                m_nextReadSequenceNumber += 1;
                m_nextReadCellIndex = CircularIncrement(m_nextReadCellIndex);

                ReadFence();
                const uint64_t nextWriteSequenceNumber = LoadRelaxed(&m_pState->nextWriteSequenceNumber);
                if (nextWriteSequenceNumber < m_unvalidatedReadSequenceNumber + m_cellCount)
                {
                    m_unvalidatedReadSequenceNumber = m_nextReadSequenceNumber;
                    fnCommitRecord(true);
                }
                else
                {
                    fnCommitRecord(false);
                    break;
                }
            }
        }

        // Returns false if the buffer is empty or the writer overran the cell during the read
        template <typename TFN_ReadRecord>
        bool PeekLatestRecord(TFN_ReadRecord&& fnReadRecord) const
        {
            // An empty buffer (nextWriteSequenceNumber == 0) underflows to UINT64_MAX here.
            const uint64_t latestSequenceNumber = LoadAcquire(&m_pState->nextWriteSequenceNumber) - 1;
            if (latestSequenceNumber == UINT64_MAX)
            {
                return false;
            }

            const uint32_t cellIndex = static_cast<uint32_t>(latestSequenceNumber % m_cellCount);
            const uint8_t* const pCell = m_pCells + cellIndex * m_cellSize;

            if (!InvokeReadRecord(fnReadRecord, pCell, latestSequenceNumber))
            {
                return false;
            }

            ReadFence();
            return LoadRelaxed(&m_pState->nextWriteSequenceNumber) < latestSequenceNumber + m_cellCount;
        }

    private:
        bool ObservationStartReached(uint64_t readEndpointTimestamp, uint64_t readEndSequenceNumber)
        {
            if (m_observationStartTimestamp == 0)
            {
                return true;
            }
            if (readEndpointTimestamp < m_observationStartTimestamp)
            {
                return false;
            }

            // Record timestamps are monotonic, so find the first record at or after the observation start without
            // burdening the recurring consume loop with a per-record lower-bound check. The logical range wraps over
            // runtime-strided cells; adapting it to std::lower_bound would require a larger custom iterator than this
            // bounded binary search.
            uint64_t lowerSequenceNumber = m_nextReadSequenceNumber;
            uint64_t upperSequenceNumber = readEndSequenceNumber;
            while (lowerSequenceNumber < upperSequenceNumber)
            {
                const uint64_t middleSequenceNumber = lowerSequenceNumber + (upperSequenceNumber - lowerSequenceNumber) / 2;
                const uint32_t middleCellIndex = static_cast<uint32_t>(middleSequenceNumber % m_cellCount);
                const uint8_t* const pCell = m_pCells + middleCellIndex * m_cellSize;
                const uint64_t* const pCellTimestamp = reinterpret_cast<const uint64_t*>(pCell + m_recordTimestampOffsetInCell);
                if (*pCellTimestamp < m_observationStartTimestamp)
                {
                    lowerSequenceNumber = middleSequenceNumber + 1;
                }
                else
                {
                    upperSequenceNumber = middleSequenceNumber;
                }
            }

            if (m_retainAtLeastOneRecord && lowerSequenceNumber == readEndSequenceNumber && m_nextReadSequenceNumber < lowerSequenceNumber)
            {
                --lowerSequenceNumber;
            }

            m_nextReadSequenceNumber = lowerSequenceNumber;
            m_nextReadCellIndex = static_cast<uint32_t>(m_nextReadSequenceNumber % m_cellCount);
            if (!CheckConsistency())
            {
                return false;
            }
            if (m_nextReadSequenceNumber == readEndSequenceNumber)
            {
                return false;
            }

            m_observationStartTimestamp = 0;
            m_retainAtLeastOneRecord = false;
            return true;
        }

        bool CheckConsistency()
        {
            ReadFence();
            if (LoadRelaxed(&m_pState->nextWriteSequenceNumber) < m_unvalidatedReadSequenceNumber + m_cellCount)
            {
                m_unvalidatedReadSequenceNumber = m_nextReadSequenceNumber;
                return true;
            }
            else
            {
                return false;
            }
        }

        uint32_t CircularIncrement(uint32_t index) const
        {
            uint32_t nextIndex = index + 1;
            return nextIndex == m_cellCount ? 0 : nextIndex;
        }

        void RecoverFromBufferOverrun(uint64_t nextWriteSequenceNumber)
        {
            m_nextReadSequenceNumber = nextWriteSequenceNumber - m_threeQuartersCellCount;
            m_nextReadCellIndex = static_cast<uint32_t>(m_nextReadSequenceNumber % m_cellCount);

            m_droppedRecordCount += (m_nextReadSequenceNumber - m_unvalidatedReadSequenceNumber);

            m_unvalidatedReadSequenceNumber = m_nextReadSequenceNumber;
        }

        template <typename TFN_ReadRecord>
        inline static auto InvokeReadRecord(TFN_ReadRecord&& fnReadRecord, uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        {
            using Traits = CallableTraits<std::remove_reference_t<TFN_ReadRecord>>;
            if constexpr (std::is_same_v<typename Traits::ArgTypes, std::tuple<uint32_t, const uint8_t*, size_t>>)
            {
                return fnReadRecord(recordType, pPayload, payloadSize);
            }
            else if constexpr (std::is_same_v<typename Traits::ArgTypes, std::tuple<uint64_t, uint32_t, const uint8_t*, size_t>>)
            {
                return fnReadRecord(sequenceNumber, recordType, pPayload, payloadSize);
            }
            else
            {
                static_assert(sizeof(TFN_ReadRecord*) == 0, "TFN_ReadRecord does not match the expected signature: use (uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) or (uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)");
            }
        }

        template <typename TFN_ReadRecord>
        inline bool InvokeReadRecord(TFN_ReadRecord&& fnReadRecord, const uint8_t* pCell, uint64_t sequenceNumber) const
        {
            NvTrace_RecordDesc recordDesc;
            if (m_hasFixedRecordDesc)
            {
                recordDesc = m_fixedRecordDesc;
            }
            else
            {
                recordDesc = *reinterpret_cast<const NvTrace_RecordDesc*>(pCell + m_variableRecordDescOffsetInCell);
                if (m_maxPayloadSizeDiv8 < recordDesc.payloadEndOffsetDiv8)
                {
                    return false;
                }
            }
            const size_t payloadSize = static_cast<size_t>(recordDesc.payloadEndOffsetDiv8) * sizeof(uint64_t);

            const uint8_t* const pPayload = pCell + m_payloadOffsetInCell;

            if constexpr (std::is_same_v<typename CallableTraits<std::remove_reference_t<TFN_ReadRecord>>::ReturnType, void>)
            {
                InvokeReadRecord(fnReadRecord, sequenceNumber, recordDesc.recordType, pPayload, payloadSize);
                return true;
            }
            else
            {
                return static_cast<bool>(InvokeReadRecord(fnReadRecord, sequenceNumber, recordDesc.recordType, pPayload, payloadSize));
            }
        }

    private:
        const NvTrace_CircularBufferState2* m_pState = nullptr;
        const uint8_t* m_pCells = nullptr;
        uint32_t m_cellCount = 0;
        uint32_t m_threeQuartersCellCount = 0;
        size_t m_cellSize = 0;

        bool m_hasFixedRecordDesc = false;
        NvTrace_RecordDesc m_fixedRecordDesc = {};
        uint32_t m_variableRecordDescOffsetInCell = NVTRACE_INVALID_OFFSET32;

        size_t m_payloadOffsetInCell = 0;
        uint32_t m_maxPayloadSizeDiv8 = 0;

        NvTrace_CircularBufferRecordTimestampType m_recordTimestampType = NvTrace_CircularBufferRecordTimestampType_Invalid;
        size_t m_recordTimestampOffsetInCell = 0;

        uint64_t m_nextReadSequenceNumber = 0;
        uint32_t m_nextReadCellIndex = 0;

        uint64_t m_unvalidatedReadSequenceNumber = 0;
        uint64_t m_observationStartTimestamp = 0;
        bool m_retainAtLeastOneRecord = false;

        uint64_t m_droppedRecordCount = 0;
    };

} // namespace nv::trace
