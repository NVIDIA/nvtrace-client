/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer_v1.h>

#include <NvTrace/Target/NvTraceMemoryFence.h>

#include <atomic>
#include <cstdint>
#include <utility>

namespace nv::trace {

    // Policy-free reader of a circular buffer.
    // TBufferAccessor is a class that abstracts cell address calculations, allowing for cell data to be stored in several separate parallel arrays.
    template <typename TBufferAccessor>
    class CircularBufferReaderV1
    {
    public:
        ~CircularBufferReaderV1() = default;
        CircularBufferReaderV1() noexcept = default;

        CircularBufferReaderV1(const CircularBufferReaderV1&) = delete;
        CircularBufferReaderV1(CircularBufferReaderV1&& rhs) noexcept
            : m_pState(std::exchange(rhs.m_pState, nullptr))
            , m_bufferAccessor(std::exchange(rhs.m_bufferAccessor, {}))
            , m_cellCount(std::exchange(rhs.m_cellCount, 0))
            , m_readCellIndex(std::exchange(rhs.m_readCellIndex, 0))
            , m_readSequenceNumber(std::exchange(rhs.m_readSequenceNumber, 0))
            , m_droppedRecordCount(std::exchange(rhs.m_droppedRecordCount, 0))
        {
        }

        CircularBufferReaderV1& operator=(const CircularBufferReaderV1&) = delete;
        CircularBufferReaderV1& operator=(CircularBufferReaderV1&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pState = std::exchange(rhs.m_pState, nullptr);
                m_bufferAccessor = std::exchange(rhs.m_bufferAccessor, {});
                m_cellCount = std::exchange(rhs.m_cellCount, 0);
                m_readCellIndex = std::exchange(rhs.m_readCellIndex, 0);
                m_readSequenceNumber = std::exchange(rhs.m_readSequenceNumber, 0);
                m_droppedRecordCount = std::exchange(rhs.m_droppedRecordCount, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_droppedRecordCount = 0;
            m_readSequenceNumber = 0;
            m_readCellIndex = 0;
            m_cellCount = 0;
            m_bufferAccessor = {};
            m_pState = nullptr;
        }

        void Initialize(const NvTrace_CircularBufferState* pState, const TBufferAccessor& bufferAccessor, uint32_t cellCount)
        {
            m_pState = pState;
            m_bufferAccessor = bufferAccessor;
            m_cellCount = cellCount;

            // Start reading from a safe oldest valid element.
            // A simple heuristic is ~ 75% behind the write pointer == 25% ahead.
            uint32_t threeQuartersCellCount = m_cellCount < 4 ? 1 : (m_cellCount / 4) * 3;
            uint64_t currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
            m_readSequenceNumber = threeQuartersCellCount < currentEndSequenceNumber ? currentEndSequenceNumber - threeQuartersCellCount : 0;
            m_readCellIndex = m_readSequenceNumber % m_cellCount;
        }

        uint32_t CellCount() const
        {
            return m_cellCount;
        }

        uint64_t DroppedRecordCount() const
        {
            return m_droppedRecordCount;
        }

        // Returns the number of records that were written since the last Consume*() call.
        // If there was a buffer overrun, the returned value will exceed CellCount().
        uint64_t UnreadRecordCount() const
        {
            return m_pState->currentEndSequenceNumber - m_readSequenceNumber;
        }

        uint64_t LastUpdateTsc() const
        {
            return m_pState->lastUpdateTsc;
        }

        uint64_t AcquireCurrentEndSequenceNumber()
        {
            uint64_t currentEndSequenceNumber = LoadAcquire(&m_pState->currentEndSequenceNumber);
            // one-way barrier: subsequent loads must occur after the above statement. ARM perf: faster than std::atomic_thread_fence(std::memory_order_acquire);
            if (m_readSequenceNumber + m_cellCount <= currentEndSequenceNumber)
            {
                RecoverFromBufferOverrun(currentEndSequenceNumber);
            }

            return currentEndSequenceNumber;
        }

        // Advances past complete records whose timestamp precedes timestamp, leaving the first
        // straddling record unread. The search is bounded by one captured writer watermark.
        bool SkipRecordsBeforeTimestamp(uint64_t timestamp)
        {
            const uint64_t readEndSequenceNumber = AcquireCurrentEndSequenceNumber();
            const uint64_t initialReadSequenceNumber = m_readSequenceNumber;
            uint64_t lowerSequenceNumber = initialReadSequenceNumber;
            uint64_t upperSequenceNumber = readEndSequenceNumber;
            while (lowerSequenceNumber < upperSequenceNumber)
            {
                const uint64_t middleSequenceNumber = lowerSequenceNumber + (upperSequenceNumber - lowerSequenceNumber) / 2;
                const uint32_t middleCellIndex = static_cast<uint32_t>(middleSequenceNumber % m_cellCount);
                if (m_bufferAccessor.ReadRecordTimestamp(middleCellIndex) < timestamp)
                {
                    lowerSequenceNumber = middleSequenceNumber + 1;
                }
                else
                {
                    upperSequenceNumber = middleSequenceNumber;
                }
            }

            ReadFence();
            if (initialReadSequenceNumber + m_cellCount <= LoadRelaxed(&m_pState->currentEndSequenceNumber))
            {
                return false;
            }

            m_readSequenceNumber = lowerSequenceNumber;
            m_readCellIndex = static_cast<uint32_t>(m_readSequenceNumber % m_cellCount);
            return true;
        }

        template <typename TReadRecordArg>
        bool ConsumeOneRecord(uint64_t readEndSequenceNumber, TReadRecordArg&& readRecordArg)
        {
            if (readEndSequenceNumber <= m_readSequenceNumber)
            {
                return false;
            }

            m_bufferAccessor.ReadRecord(m_readCellIndex, m_readSequenceNumber, readRecordArg);
            ReadFence(); // Ensure ReadRecord()'s loads complete before currentEndSequenceNumber is read. ARM perf: this is faster than std::atomic_thread_fence(std::memory_order_acquire), which lowers to `dmb ish`.

            uint64_t currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
            bool consistencyCheckPassed = currentEndSequenceNumber < m_readSequenceNumber + m_cellCount;
            if (consistencyCheckPassed)
            {
                m_readCellIndex = CircularIncrement(m_readCellIndex);
                ++m_readSequenceNumber;
            }

            return consistencyCheckPassed;
        }

        template <typename TReadRecordArg>
        bool ConsumeOneRecord(uint64_t readEndSequenceNumber, uint64_t readEndpointTsc, TReadRecordArg&& readRecordArg)
        {
            if (readEndSequenceNumber <= m_readSequenceNumber)
            {
                return false;
            }

            uint64_t recordTimestampTsc = m_bufferAccessor.ReadRecordTimestamp(m_readCellIndex);
            if (readEndpointTsc < recordTimestampTsc)
            {
                // Since we didn't perform a consistency check, it's possible that recordTimestampTsc has a garbage value.
                // However, in that case, we would return false anyways, so we can just do it earlier here by "accident" as well.
                return false;
            }

            m_bufferAccessor.ReadRecord(m_readCellIndex, m_readSequenceNumber, readRecordArg);
            ReadFence(); // Ensure ReadRecord()'s loads complete before currentEndSequenceNumber is read. ARM perf: this is faster than std::atomic_thread_fence(std::memory_order_acquire), which lowers to `dmb ish`.

            uint64_t currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
            bool consistencyCheckPassed = currentEndSequenceNumber < m_readSequenceNumber + m_cellCount;
            if (consistencyCheckPassed)
            {
                m_readCellIndex = CircularIncrement(m_readCellIndex);
                ++m_readSequenceNumber;
            }

            return consistencyCheckPassed;
        }

        // Consumes available records; this function advances the state machine.
        // Calls fnReadRecord(), then performs a consistency check, then calls either fnCommitRecord(consistencyCheckPassed).
        template <typename TReadRecordArg, typename TFN_CommitRecord>
        void ConsumeRecords(TReadRecordArg&& readRecordArg, TFN_CommitRecord&& fnCommitRecord)
        {
            uint64_t readEndSequenceNumber = AcquireCurrentEndSequenceNumber();

            while (m_readSequenceNumber < readEndSequenceNumber)
            {
                m_bufferAccessor.ReadRecord(m_readCellIndex, m_readSequenceNumber, readRecordArg);
                ReadFence(); // Ensure ReadRecord()'s loads complete before currentEndSequenceNumber is read. ARM perf: this is faster than std::atomic_thread_fence(std::memory_order_acquire), which lowers to `dmb ish`.

                uint64_t currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
                bool consistencyCheckPassed = currentEndSequenceNumber < m_readSequenceNumber + m_cellCount;
                fnCommitRecord(consistencyCheckPassed);

                if (consistencyCheckPassed)
                {
                    m_readCellIndex = CircularIncrement(m_readCellIndex);
                    ++m_readSequenceNumber;
                }
                else
                {
                    break;
                }
            }
        }

        // Consumes available records with timestamps <= readEndpointTsc; this function advances the state machine.
        // Calls fnReadRecord(), then performs a consistency check, then calls either fnCommitRecord(consistencyCheckPassed).
        template <typename TReadRecordArg, typename TFN_CommitRecord>
        void ConsumeRecords(uint64_t readEndpointTsc, TReadRecordArg&& readRecordArg, TFN_CommitRecord&& fnCommitRecord)
        {
            uint64_t readEndSequenceNumber = AcquireCurrentEndSequenceNumber();

            while (m_readSequenceNumber < readEndSequenceNumber)
            {
                uint64_t recordTimestampTsc = m_bufferAccessor.ReadRecordTimestamp(m_readCellIndex);
                if (readEndpointTsc < recordTimestampTsc)
                {
                    // Since we didn't perform a consistency check, it's possible that recordTimestampTsc has a garbage value.
                    // However, in that case, we would break out of the loop at the end anyway, so we can just do it earlier here by "accident" as well.
                    break;
                }

                m_bufferAccessor.ReadRecord(m_readCellIndex, m_readSequenceNumber, readRecordArg);
                ReadFence(); // Ensure ReadRecord()'s loads complete before currentEndSequenceNumber is read. ARM perf: this is faster than std::atomic_thread_fence(std::memory_order_acquire), which lowers to `dmb ish`.

                uint64_t currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
                bool consistencyCheckPassed = currentEndSequenceNumber < m_readSequenceNumber + m_cellCount;
                fnCommitRecord(consistencyCheckPassed);

                if (consistencyCheckPassed)
                {
                    m_readCellIndex = CircularIncrement(m_readCellIndex);
                    ++m_readSequenceNumber;
                }
                else
                {
                    break;
                }
            }
        }

        // Reads the latest record without advancing the state machine.
        // This function retries until a consistent result has been read.
        template <typename TReadRecordArg>
        void ReadLatestRecord(TReadRecordArg&& readRecordArg) const
        {
            uint64_t currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (currentEndSequenceNumber == 0)
            {
                // Circular buffer is empty
                return;
            }

            while (true)
            {
                uint64_t readSequenceNumber = currentEndSequenceNumber - 1;
                uint32_t readCellIndex = readSequenceNumber % m_cellCount;

                m_bufferAccessor.ReadRecord(readCellIndex, readSequenceNumber, readRecordArg);
                ReadFence(); // Ensure ReadRecord()'s loads complete before currentEndSequenceNumber is read. ARM perf: this is faster than std::atomic_thread_fence(std::memory_order_acquire), which lowers to `dmb ish`.

                currentEndSequenceNumber = m_pState->currentEndSequenceNumber;
                bool consistencyCheckPassed = currentEndSequenceNumber < readSequenceNumber + m_cellCount;

                if (consistencyCheckPassed)
                {
                    break;
                }
                else
                {
                    std::atomic_thread_fence(std::memory_order_acquire); // Turn the relaxed load of currentEndSequenceNumber into an acquire load
                }
            }
        }

    private:
        uint32_t CircularIncrement(uint32_t index) const
        {
            uint32_t nextIndex = index + 1;
            return nextIndex == m_cellCount ? 0 : nextIndex;
        }

        void RecoverFromBufferOverrun(uint64_t currentEndSequenceNumber)
        {
            // Buffer overrun - advance m_readSequenceNumber to a safe oldest valid element.
            // A simple heuristic is ~ 75% behind the write pointer == 25% ahead.
            uint32_t threeQuartersCellCount = m_cellCount < 4 ? 1 : (m_cellCount / 4) * 3;
            uint64_t newReadSequenceNumber = currentEndSequenceNumber - threeQuartersCellCount;
            m_droppedRecordCount += (newReadSequenceNumber - m_readSequenceNumber);
            m_readSequenceNumber = newReadSequenceNumber;
            m_readCellIndex = m_readSequenceNumber % m_cellCount;
        }

    private:
        const NvTrace_CircularBufferState* m_pState = nullptr;
        TBufferAccessor m_bufferAccessor = {};
        uint32_t m_cellCount = 0;

        uint32_t m_readCellIndex = 0;
        uint64_t m_readSequenceNumber = 0;

        uint64_t m_droppedRecordCount = 0;
    };

} // namespace nv::trace
