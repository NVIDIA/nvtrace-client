/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceStreamDatabaseWriter.h>
#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>
#include <NvTrace/Target/NvTraceClientLogger.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

namespace nv::trace {

    class TsfSnapshotHistoryBuffer
    {
    public:
        struct DataBlockInfo
        {
            uint64_t sequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t lowerBoundTimestampUtcNsec = 0;
            uint64_t upperBoundTimestampUtcNsec = 0;
        };

        using FnOnDataBlockAgedIn = void (*)(void* pUserData, TraceStreamDataBlockWriter& dataBlockWriter);
        using FnOnDataBlockAgedOut = void (*)(void* pUserData, const DataBlockInfo& dataBlockInfo);

        bool Initialize(
            size_t writeSlotCount,
            FnOnDataBlockAgedIn fnOnDataBlockAgedIn = nullptr,
            void* pUserDataOnDataBlockAgedIn = nullptr,
            FnOnDataBlockAgedOut fnOnDataBlockAgedOut = nullptr,
            void* pUserDataOnDataBlockAgedOut = nullptr)
        {
            if (!writeSlotCount)
            {
                return false;
            }
            try
            {
                m_historySlots.resize(writeSlotCount);
                m_ancientDataBlockWriters.resize(writeSlotCount);
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to allocate TSF snapshot FIFO: %s.", e.what());
                return false;
            }
            m_maxHistorySlots = writeSlotCount;
            m_fnOnDataBlockAgedIn = fnOnDataBlockAgedIn;
            m_pUserDataOnDataBlockAgedIn = pUserDataOnDataBlockAgedIn;
            m_fnOnDataBlockAgedOut = fnOnDataBlockAgedOut;
            m_pUserDataOnDataBlockAgedOut = pUserDataOnDataBlockAgedOut;
            return true;
        }

        void Reset() noexcept
        {
            m_historySlots.clear();
            m_historySize = 0;
            m_earliestHistorySlotIndex = 0;
            m_ancientDataBlockWriters.clear();
            m_numAncientDataBlockWriters = 0;
            m_fnOnDataBlockAgedIn = nullptr;
            m_pUserDataOnDataBlockAgedIn = nullptr;
            m_fnOnDataBlockAgedOut = nullptr;
            m_pUserDataOnDataBlockAgedOut = nullptr;
        }

        bool GetEarliestDataBlockInfo(DataBlockInfo& out_dataBlockInfo)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_historySize == 0)
            {
                return false;
            }
            out_dataBlockInfo = m_historySlots[m_earliestHistorySlotIndex].dataBlockInfo;
            return true;
        }

        bool PushPublishedDataBlock(TraceStreamDataBlockWriter&& dataBlockWriter)
        {
            if (!dataBlockWriter.IsInitialized())
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotHistoryBuffer::PushPublishedDataBlock() received uninitialized data block writer.");
                return false;
            }
            const DataBlockInfo agedInDataBlockInfo = {
                dataBlockWriter.DataBlockSequenceNumber(),
                dataBlockWriter.RecordTimestampLowerBoundUtcNsec(),
                dataBlockWriter.RecordTimestampUpperBoundUtcNsec()};
            DataBlockInfo agedOutDataBlockInfo;
            bool hasAgedOutDataBlock = false;
            bool succeeded = false;
            TraceStreamDataBlockWriter* pAgedInDataBlockWriter = nullptr;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_maxHistorySlots == 0)
                {
                    NVTRACE_LOG_ERR(50, "TsfSnapshotHistoryBuffer::PushPublishedDataBlock() called before Initialize().");
                    return false;
                }
                if (m_historySize > 0)
                {
                    const uint64_t earliestSequenceNumber = m_historySlots[m_earliestHistorySlotIndex].dataBlockInfo.sequenceNumber;
                    const uint64_t expectedSequenceNumber = earliestSequenceNumber + m_historySize;
                    if (agedInDataBlockInfo.sequenceNumber != expectedSequenceNumber)
                    {
                        NVTRACE_LOG_ERR(50, "TsfSnapshotHistoryBuffer::PushPublishedDataBlock() data block sequence number must be 1 higher than the most recent history data block. (dataBlockSequenceNumber=%" PRIu64 ", expectedSequenceNumber=%" PRIu64 ").", agedInDataBlockInfo.sequenceNumber, expectedSequenceNumber);
                        return false;
                    }
                }
                if (m_historySize == m_maxHistorySlots)
                {
                    HistorySlot& earliestSlot = m_historySlots[m_earliestHistorySlotIndex];
                    if (earliestSlot.dataBlockWriter.IsInitialized())
                    {
                        if (m_numAncientDataBlockWriters >= m_maxHistorySlots)
                        {
                            NVTRACE_LOG_ERR(50, "TsfSnapshotHistoryBuffer::PushPublishedDataBlock() ancient data block writer capacity is full.");
                            return false;
                        }
                        m_ancientDataBlockWriters[m_numAncientDataBlockWriters++] = std::move(earliestSlot.dataBlockWriter);
                    }
                    agedOutDataBlockInfo = earliestSlot.dataBlockInfo;
                    hasAgedOutDataBlock = true;
                    earliestSlot = {agedInDataBlockInfo, std::move(dataBlockWriter)};
                    pAgedInDataBlockWriter = &earliestSlot.dataBlockWriter;
                    m_earliestHistorySlotIndex++;
                    if (m_earliestHistorySlotIndex >= m_maxHistorySlots)
                    {
                        m_earliestHistorySlotIndex -= m_maxHistorySlots;
                    }
                    succeeded = true;
                }
                else
                {
                    size_t indexToInsert = m_earliestHistorySlotIndex + m_historySize;
                    if (indexToInsert >= m_maxHistorySlots)
                    {
                        indexToInsert -= m_maxHistorySlots;
                    }
                    m_historySlots[indexToInsert] = {agedInDataBlockInfo, std::move(dataBlockWriter)};
                    pAgedInDataBlockWriter = &m_historySlots[indexToInsert].dataBlockWriter;
                    m_historySize++;
                    succeeded = true;
                }
                if (succeeded && m_fnOnDataBlockAgedIn)
                {
                    m_fnOnDataBlockAgedIn(m_pUserDataOnDataBlockAgedIn, *pAgedInDataBlockWriter);
                }
            }
            if (succeeded)
            {
                if (hasAgedOutDataBlock && m_fnOnDataBlockAgedOut)
                {
                    m_fnOnDataBlockAgedOut(m_pUserDataOnDataBlockAgedOut, agedOutDataBlockInfo);
                }
            }
            return succeeded;
        }

        // Return an initialized data block writer if available
        bool RelinquishDataBlock(TraceStreamDataBlockWriter& out_dataBlockWriter)
        {
            if (out_dataBlockWriter.IsInitialized())
            {
                NVTRACE_LOG_ERR(50, "TsfSnapshotHistoryBuffer::RelinquishDataBlock() output data block writer is already initialized.");
                return false;
            }
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_numAncientDataBlockWriters > 0)
            {
                out_dataBlockWriter = std::move(m_ancientDataBlockWriters[--m_numAncientDataBlockWriters]);
                return true;
            }
            for (size_t indexToOffset = 0; indexToOffset < m_historySize; ++indexToOffset)
            {
                size_t index = m_earliestHistorySlotIndex + indexToOffset;
                if (index >= m_maxHistorySlots)
                {
                    index -= m_maxHistorySlots;
                }
                if (m_historySlots[index].dataBlockWriter.IsInitialized())
                {
                    out_dataBlockWriter = std::move(m_historySlots[index].dataBlockWriter);
                    return true;
                }
            }
            return false;
        }

        // Calls onIterationBegin before the first data block while holding the same lock.
        // Return false from onIteratedDataBlock to stop.
        template <typename TOnIterationBegin, typename TOnIteratedDataBlock>
        void ForEachDataBlockLatestFirst(
            TOnIterationBegin&& onIterationBegin,
            TOnIteratedDataBlock&& onIteratedDataBlock)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            const DataBlockInfo earliestDataBlockInfo = m_historySize > 0 ? m_historySlots[m_earliestHistorySlotIndex].dataBlockInfo : DataBlockInfo();
            onIterationBegin(m_historySize, earliestDataBlockInfo);

            size_t numIterations = m_historySize;
            while (numIterations > 0)
            {
                numIterations--;
                size_t index = m_earliestHistorySlotIndex + numIterations;
                if (index >= m_maxHistorySlots)
                {
                    index -= m_maxHistorySlots;
                }
                auto& historySlot = m_historySlots[index];
                if (!onIteratedDataBlock(historySlot.dataBlockInfo, historySlot.dataBlockWriter))
                {
                    break;
                }
            }
        }

    private:
        struct HistorySlot
        {
            DataBlockInfo dataBlockInfo;
            // Check if the data block writer is initialized for ownership
            TraceStreamDataBlockWriter dataBlockWriter;
        };

        std::mutex m_mutex;
        size_t m_maxHistorySlots = 0;
        // Fixed size FIFO with contiguous data block sequence numbers
        std::vector<HistorySlot> m_historySlots;
        size_t m_historySize = 0;
        size_t m_earliestHistorySlotIndex = 0;
        std::vector<TraceStreamDataBlockWriter> m_ancientDataBlockWriters;
        size_t m_numAncientDataBlockWriters = 0;
        FnOnDataBlockAgedIn m_fnOnDataBlockAgedIn = nullptr;
        void* m_pUserDataOnDataBlockAgedIn = nullptr;
        FnOnDataBlockAgedOut m_fnOnDataBlockAgedOut = nullptr;
        void* m_pUserDataOnDataBlockAgedOut = nullptr;
    };

} // namespace nv::trace
