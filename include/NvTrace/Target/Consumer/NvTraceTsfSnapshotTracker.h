/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceStreamDatabaseWriter.h>
#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nv::trace {

    template <typename TStorageWriter>
    class TsfSnapshotCommand
    {
    public:
        // Must align with the bitset size used in TsfSnapshotCommand
        // Either attach data blocks or data blocks can not exceed this limit
        static constexpr size_t MaxDataBlockCountPerSnapshot = 64;

        TsfSnapshotCommand() = default;

        bool Initialize(
            uint64_t triggerTimestampUtcNsec,
            uint64_t historyDurationNsec,
            uint64_t futureDurationNsec,
            std::unique_ptr<TStorageWriter> storageWriter,
            uint64_t initialDataBlockSequenceNumber)
        {
            if (!storageWriter)
            {
                return false;
            }
            m_triggerTimestampUtcNsec = triggerTimestampUtcNsec;
            m_historyDurationNsec = historyDurationNsec;
            m_futureDurationNsec = futureDurationNsec;
            m_storageWriter = std::move(storageWriter);
            m_initialDataBlockSequenceNumber = initialDataBlockSequenceNumber;
            return true;
        }

        void Reset()
        {
            m_triggerTimestampUtcNsec = 0;
            m_historyDurationNsec = 0;
            m_futureDurationNsec = 0;
            m_storageWriter.reset();
            m_isFinalized = false;
            m_lowestExpectedAttachDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_highestExpectedAttachDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_expectedAttachDataBlockBitset = 0;
            m_completedAttachDataBlockBitset = 0;
            m_succeededAttachDataBlockBitset = 0;
            m_initialDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_expectedDataBlockBitset = 0;
            m_completedDataBlockBitset = 0;
            m_succeededDataBlockBitset = 0;
            m_lowestExpectedDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_highestExpectedDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        }

        TStorageWriter& StorageWriter()
        {
            return *m_storageWriter;
        }

        uint64_t BeginDataBlockSequenceNumber() const
        {
            return m_lowestExpectedDataBlockSequenceNumber;
        }

        uint64_t EndDataBlockSequenceNumber() const
        {
            return m_highestExpectedDataBlockSequenceNumber;
        }

        uint64_t BeginAttachDataBlockSequenceNumber() const
        {
            return m_lowestExpectedAttachDataBlockSequenceNumber;
        }

        uint64_t EndAttachDataBlockSequenceNumber() const
        {
            return m_highestExpectedAttachDataBlockSequenceNumber;
        }

        bool IsInitialized() const
        {
            return m_storageWriter != nullptr;
        }

        bool IsFinalized() const
        {
            return m_isFinalized;
        }

        // Require both attach data blocks and data blocks to be completed
        bool IsCompleted(bool& out_isCompletedSuccessfully) const
        {
            if (!IsFinalized())
            {
                return false;
            }
            out_isCompletedSuccessfully = ((m_succeededDataBlockBitset & m_expectedDataBlockBitset) == m_expectedDataBlockBitset) && ((m_succeededAttachDataBlockBitset & m_expectedAttachDataBlockBitset) == m_expectedAttachDataBlockBitset);
            return ((m_completedDataBlockBitset & m_expectedDataBlockBitset) == m_expectedDataBlockBitset) && ((m_completedAttachDataBlockBitset & m_expectedAttachDataBlockBitset) == m_expectedAttachDataBlockBitset);
        }

        // Not for attach data blocks
        bool WantsDataBlock(uint64_t dataBlockSequenceNumber, uint64_t lowerBoundTimestampUtcNsec, uint64_t upperBoundTimestampUtcNsec) const
        {
            if (dataBlockSequenceNumber < m_initialDataBlockSequenceNumber)
            {
                return false;
            }
            if (dataBlockSequenceNumber >= m_initialDataBlockSequenceNumber + MaxDataBlockCountPerSnapshot)
            {
                // [TODO] Inform developer to increase MaxDataBlockCountPerSnapshot
                return false;
            }
            if (m_isFinalized)
            {
                return false;
            }
            const uint64_t wantedLowerBoundTimestampUtcNsec = m_triggerTimestampUtcNsec - m_historyDurationNsec;
            const uint64_t wantedUpperBoundTimestampUtcNsec = m_triggerTimestampUtcNsec + m_futureDurationNsec;
            return lowerBoundTimestampUtcNsec <= wantedUpperBoundTimestampUtcNsec && upperBoundTimestampUtcNsec >= wantedLowerBoundTimestampUtcNsec;
        }

        // Not for attach data blocks
        bool ExpectsDataBlock(uint64_t dataBlockSequenceNumber) const
        {
            return dataBlockSequenceNumber >= m_lowestExpectedDataBlockSequenceNumber && dataBlockSequenceNumber <= m_highestExpectedDataBlockSequenceNumber;
        }

        // Attach data blocks must be added before they are sent to the storage writer, so that a
        // completion arriving before Finalize() can still be indexed against the expected bitset.
        bool AddAttachDataBlock(uint64_t dataBlockSequenceNumber)
        {
            if (m_isFinalized || dataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
            {
                return false;
            }
            if (m_lowestExpectedAttachDataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
            {
                m_lowestExpectedAttachDataBlockSequenceNumber = dataBlockSequenceNumber;
                m_highestExpectedAttachDataBlockSequenceNumber = dataBlockSequenceNumber;
                m_expectedAttachDataBlockBitset = uint64_t{1};
                return true;
            }
            // Attach data blocks are staged in ascending sequence number. Lowering the origin
            // would shift already-recorded completion bits, so reject a sequence number below the current lowest.
            if (dataBlockSequenceNumber < m_lowestExpectedAttachDataBlockSequenceNumber)
            {
                return false;
            }
            if (dataBlockSequenceNumber >= m_lowestExpectedAttachDataBlockSequenceNumber + MaxDataBlockCountPerSnapshot)
            {
                // [TODO] Inform developer to increase MaxDataBlockCountPerSnapshot
                return false;
            }
            if (m_highestExpectedAttachDataBlockSequenceNumber < dataBlockSequenceNumber)
            {
                m_highestExpectedAttachDataBlockSequenceNumber = dataBlockSequenceNumber;
                // Set expected attach data block bitset [m_lowestExpectedAttachDataBlockSequenceNumber, m_highestExpectedAttachDataBlockSequenceNumber]
                const uint64_t numAttachDataBlocks = m_highestExpectedAttachDataBlockSequenceNumber - m_lowestExpectedAttachDataBlockSequenceNumber + 1;
                m_expectedAttachDataBlockBitset = (numAttachDataBlocks == MaxDataBlockCountPerSnapshot) ? ~uint64_t{0} : ((uint64_t{1} << numAttachDataBlocks) - 1);
            }
            return true;
        }

        // Stop accepting published data blocks and attach data blocks
        bool Finalize()
        {
            if (m_isFinalized)
            {
                return false;
            }
            m_isFinalized = true;
            return true;
        }

        // Update expected data blocks
        // Not for attach data blocks
        bool AcknowledgePublishedDataBlock(uint64_t dataBlockSequenceNumber, uint64_t lowerBoundTimestampUtcNsec, uint64_t upperBoundTimestampUtcNsec)
        {
            if (!WantsDataBlock(dataBlockSequenceNumber, lowerBoundTimestampUtcNsec, upperBoundTimestampUtcNsec))
            {
                return false;
            }
            if (dataBlockSequenceNumber >= m_initialDataBlockSequenceNumber + MaxDataBlockCountPerSnapshot)
            {
                // [TODO] Inform developer to increase MaxDataBlockCountPerSnapshot
                return false;
            }
            auto updateExpectedDataBlockBitset = [this]() {
                // Set expected data block bitset [m_lowestExpectedDataBlockSequenceNumber, m_highestExpectedDataBlockSequenceNumber]
                const uint64_t lowBitPos = m_lowestExpectedDataBlockSequenceNumber - m_initialDataBlockSequenceNumber;
                const uint64_t highBitPos = m_highestExpectedDataBlockSequenceNumber - m_initialDataBlockSequenceNumber;
                const uint64_t numDataBlocks = highBitPos - lowBitPos + 1;
                const uint64_t rangeMask = (numDataBlocks == MaxDataBlockCountPerSnapshot) ? ~uint64_t{0} : ((uint64_t{1} << numDataBlocks) - 1);
                m_expectedDataBlockBitset = rangeMask << lowBitPos;
            };
            if (m_lowestExpectedDataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
            {
                m_lowestExpectedDataBlockSequenceNumber = dataBlockSequenceNumber;
                m_highestExpectedDataBlockSequenceNumber = dataBlockSequenceNumber;
                m_expectedDataBlockBitset = uint64_t{1} << (dataBlockSequenceNumber - m_initialDataBlockSequenceNumber);
            }
            else if (m_lowestExpectedDataBlockSequenceNumber > dataBlockSequenceNumber)
            {
                m_lowestExpectedDataBlockSequenceNumber = dataBlockSequenceNumber;
                updateExpectedDataBlockBitset();
            }
            else if (m_highestExpectedDataBlockSequenceNumber < dataBlockSequenceNumber)
            {
                m_highestExpectedDataBlockSequenceNumber = dataBlockSequenceNumber;
                updateExpectedDataBlockBitset();
            }
            return true;
        }

        // For both data blocks and attach data blocks
        bool AcknowledgeCompletedDataBlock(uint64_t dataBlockSequenceNumber, bool isCompletedSuccessfully, bool isAttachDataBlock)
        {
            if (isAttachDataBlock)
            {
                if (m_lowestExpectedAttachDataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER || dataBlockSequenceNumber < m_lowestExpectedAttachDataBlockSequenceNumber)
                {
                    return false;
                }
                // Update completed data block bitset and succeeded data block bitset
                // Ok to set bits outside the expected range
                const uint64_t bitPos = dataBlockSequenceNumber - m_lowestExpectedAttachDataBlockSequenceNumber;
                if (bitPos >= MaxDataBlockCountPerSnapshot)
                {
                    return false;
                }
                m_completedAttachDataBlockBitset |= uint64_t{1} << bitPos;
                // Success is sticky but failure is not sticky (later failures are ignored)
                m_succeededAttachDataBlockBitset |= isCompletedSuccessfully ? uint64_t{1} << bitPos : 0;
                return true;
            }
            // Update completed data block bitset and succeeded data block bitset
            // Ok to set bits outside the expected range
            const uint64_t bitPos = dataBlockSequenceNumber - m_initialDataBlockSequenceNumber;
            if (bitPos >= MaxDataBlockCountPerSnapshot)
            {
                return false;
            }
            m_completedDataBlockBitset |= uint64_t{1} << bitPos;
            // Success is sticky but failure is not sticky (later failures are ignored)
            m_succeededDataBlockBitset |= isCompletedSuccessfully ? uint64_t{1} << bitPos : 0;
            return true;
        }

    private:
        uint64_t m_triggerTimestampUtcNsec = 0;
        uint64_t m_historyDurationNsec = 0;
        uint64_t m_futureDurationNsec = 0;
        std::unique_ptr<TStorageWriter> m_storageWriter;

        // If set, stop acknowledging published data blocks
        bool m_isFinalized = false;

        // Used to index the bitset and as the boundary of the mask
        uint64_t m_lowestExpectedAttachDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        uint64_t m_highestExpectedAttachDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        // Used as a mask for the bitset
        uint64_t m_expectedAttachDataBlockBitset = 0;
        uint64_t m_completedAttachDataBlockBitset = 0;
        uint64_t m_succeededAttachDataBlockBitset = 0;

        // Used to index the bitset
        uint64_t m_initialDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        // Used as a mask for the bitset
        uint64_t m_expectedDataBlockBitset = 0;
        uint64_t m_completedDataBlockBitset = 0;
        uint64_t m_succeededDataBlockBitset = 0;
        uint64_t m_lowestExpectedDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        uint64_t m_highestExpectedDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
    };

    // Two stage tracking:
    // 1. Listen to HistoryBuffer for published data blocks
    // 2. Listen to StorageWriter for completed data blocks
    template <typename TStorageWriter>
    class TsfSnapshotTracker
    {
    public:
        static constexpr uint32_t INVALID_SNAPSHOT_NUMBER = 0;
        static constexpr uint32_t MAX_NUM_CONCURRENT_SNAPSHOTS = 32;

        bool Initialize(
            void (*fnOnSnapshotCompleted)(void* pUserData, uint32_t snapshotNumber, bool succeeded),
            void* pUserDataOnSnapshotCompleted)
        {
            m_fnOnSnapshotCompleted = fnOnSnapshotCompleted;
            m_pUserDataOnSnapshotCompleted = pUserDataOnSnapshotCompleted;
            return true;
        }

        void Reset()
        {
            m_snapshotNumberGenerator = INVALID_SNAPSHOT_NUMBER;
            m_activeSnapshotCommands.clear();
            m_finalizedSnapshotCommands.clear();
            m_completedSnapshotCommands.clear();
            m_dataBlockWriters.clear();
            m_fnOnSnapshotCompleted = nullptr;
            m_pUserDataOnSnapshotCompleted = nullptr;
        }

        // Caller reports memory contention
        // Callee applies best effort to relinquish the data block
        // Current implementation is to relinquish the data block:
        // 1. The unwanted data block writer with the zero ref count
        // 2. The earliest initialized data block writer
        // [TODO] Define the best effort and avoid starvation
        bool RelinquishDataBlock(TraceStreamDataBlockWriter& out_dataBlockWriter)
        {
            if (out_dataBlockWriter.IsInitialized())
            {
                return false;
            }

            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            bool hasRefCount = false;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto itrUnwantedDataBlockWriter = m_dataBlockWriters.end();
                auto itrEarliestDataBlockWriter = m_dataBlockWriters.end();
                for (auto itrDataBlockWriter = m_dataBlockWriters.begin(); itrDataBlockWriter != m_dataBlockWriters.end(); ++itrDataBlockWriter)
                {
                    auto& [dataBlockWriter, refCount] = itrDataBlockWriter->second;
                    if (!dataBlockWriter.IsInitialized())
                    {
                        continue;
                    }
                    if (itrEarliestDataBlockWriter == m_dataBlockWriters.end())
                    {
                        itrEarliestDataBlockWriter = itrDataBlockWriter;
                    }
                    if (refCount == 0)
                    {
                        itrUnwantedDataBlockWriter = itrDataBlockWriter;
                        break;
                    }
                }
                auto itrDataBlockWriterToRelinquish = itrUnwantedDataBlockWriter != m_dataBlockWriters.end()
                                                        ? itrUnwantedDataBlockWriter
                                                        : itrEarliestDataBlockWriter;
                if (itrDataBlockWriterToRelinquish == m_dataBlockWriters.end())
                {
                    return false;
                }
                dataBlockSequenceNumber = itrDataBlockWriterToRelinquish->first;
                auto& [dataBlockWriter, refCount] = itrDataBlockWriterToRelinquish->second;
                hasRefCount = refCount != 0;
                out_dataBlockWriter = std::move(dataBlockWriter);
                m_dataBlockWriters.erase(itrDataBlockWriterToRelinquish);
            }
            if (hasRefCount)
            {
                AcknowledgeCompletedDataBlock(dataBlockSequenceNumber, false, false);
            }
            return true;
        }

        uint32_t CreateSnapshot(
            uint64_t triggerTimestampUtcNsec,
            uint64_t historyDurationNsec,
            uint64_t futureDurationNsec,
            std::unique_ptr<TStorageWriter> storageWriter,
            uint64_t initialDataBlockSequenceNumber)
        {
            TsfSnapshotCommand<TStorageWriter> snapshotCommand;
            if (!snapshotCommand.Initialize(
                    triggerTimestampUtcNsec,
                    historyDurationNsec,
                    futureDurationNsec,
                    std::move(storageWriter),
                    initialDataBlockSequenceNumber))
            {
                return INVALID_SNAPSHOT_NUMBER;
            }

            std::lock_guard<std::mutex> lock(m_mutex);
            const uint32_t newSnapshotNumber = ++m_snapshotNumberGenerator;
            const auto [_, inserted] = m_activeSnapshotCommands.emplace(newSnapshotNumber, std::move(snapshotCommand));
            if (!inserted)
            {
                return INVALID_SNAPSHOT_NUMBER;
            }
            return newSnapshotNumber;
        }

        // Returns true if this snapshot wanted the data block.
        bool AcknowledgePublishedDataBlock(
            uint32_t snapshotNumber,
            TraceStreamDataBlockWriter& dataBlockWriter)
        {
            if (!dataBlockWriter.IsInitialized())
            {
                return false;
            }
            return AcknowledgePublishedDataBlock(
                snapshotNumber,
                dataBlockWriter.DataBlockSequenceNumber(),
                dataBlockWriter.RecordTimestampLowerBoundUtcNsec(),
                dataBlockWriter.RecordTimestampUpperBoundUtcNsec(),
                &dataBlockWriter,
                false);
        }

        // Returns true if this snapshot wanted the data block.
        bool AcknowledgePublishedDataBlock(
            uint32_t snapshotNumber,
            uint64_t dataBlockSequenceNumber,
            uint64_t lowerBoundTimestampUtcNsec,
            uint64_t upperBoundTimestampUtcNsec,
            bool isInStorage)
        {
            return AcknowledgePublishedDataBlock(
                snapshotNumber,
                dataBlockSequenceNumber,
                lowerBoundTimestampUtcNsec,
                upperBoundTimestampUtcNsec,
                nullptr,
                isInStorage);
        }

        bool AddAttachDataBlock(uint32_t snapshotNumber, uint64_t dataBlockSequenceNumber)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto itrSnapshotCommand = m_activeSnapshotCommands.find(snapshotNumber);
            if (itrSnapshotCommand == m_activeSnapshotCommands.end())
            {
                return false;
            }
            return itrSnapshotCommand->second.AddAttachDataBlock(dataBlockSequenceNumber);
        }

        template <typename TFinalizeDataBlock>
        bool FinalizeSnapshot(uint32_t snapshotNumber, TFinalizeDataBlock&& fnFinalizeDataBlock)
        {
            bool isSnapshotCompleted = false;
            bool isSnapshotSucceeded = false;
            uint64_t failedDataBlockSequenceNumbers[TsfSnapshotCommand<TStorageWriter>::MaxDataBlockCountPerSnapshot];
            size_t failedDataBlockCount = 0;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto itrSnapshotCommand = m_activeSnapshotCommands.find(snapshotNumber);
                if (itrSnapshotCommand == m_activeSnapshotCommands.end())
                {
                    return false;
                }
                TsfSnapshotCommand<TStorageWriter>& snapshotCommand = itrSnapshotCommand->second;
                if (!snapshotCommand.Finalize())
                {
                    return false;
                }
                TraceStreamDataBlockWriter* dataBlockWritersToFinalize[TsfSnapshotCommand<TStorageWriter>::MaxDataBlockCountPerSnapshot];
                size_t dataBlockWriterCount = 0;
                const uint64_t beginDataBlockSequenceNumber = snapshotCommand.BeginDataBlockSequenceNumber();
                const uint64_t endDataBlockSequenceNumber = snapshotCommand.EndDataBlockSequenceNumber();
                if (beginDataBlockSequenceNumber != TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER && endDataBlockSequenceNumber != TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
                {
                    for (uint64_t dataBlockSequenceNumber = beginDataBlockSequenceNumber;; ++dataBlockSequenceNumber)
                    {
                        auto itrDataBlockWriter = m_dataBlockWriters.find(dataBlockSequenceNumber);
                        if (itrDataBlockWriter != m_dataBlockWriters.end() && itrDataBlockWriter->second.first.IsInitialized())
                        {
                            dataBlockWritersToFinalize[dataBlockWriterCount++] = &itrDataBlockWriter->second.first;
                        }
                        if (dataBlockSequenceNumber == endDataBlockSequenceNumber)
                        {
                            break;
                        }
                    }
                }
                // All the expected data blocks may have already been completed before the snapshot command was finalized
                isSnapshotCompleted = snapshotCommand.IsCompleted(isSnapshotSucceeded);
                if (isSnapshotCompleted)
                {
                    m_completedSnapshotCommands.emplace(snapshotNumber, std::move(snapshotCommand));
                }
                else
                {
                    m_finalizedSnapshotCommands.emplace(snapshotNumber, std::move(snapshotCommand));
                }
                m_activeSnapshotCommands.erase(itrSnapshotCommand);

                for (size_t dataBlockWriterIndex = 0; dataBlockWriterIndex < dataBlockWriterCount; ++dataBlockWriterIndex)
                {
                    TraceStreamDataBlockWriter* const pDataBlockWriterToFinalize = dataBlockWritersToFinalize[dataBlockWriterIndex];
                    const uint64_t dataBlockSequenceNumber = pDataBlockWriterToFinalize->DataBlockSequenceNumber();
                    const bool finalizeSucceeded = fnFinalizeDataBlock(*pDataBlockWriterToFinalize);
                    if (!finalizeSucceeded || pDataBlockWriterToFinalize->IsInitialized())
                    {
                        failedDataBlockSequenceNumbers[failedDataBlockCount++] = dataBlockSequenceNumber;
                    }
                }
            }
            for (size_t failedDataBlockIndex = 0; failedDataBlockIndex < failedDataBlockCount; ++failedDataBlockIndex)
            {
                AcknowledgeCompletedDataBlock(failedDataBlockSequenceNumbers[failedDataBlockIndex], false, false);
            }
            if (isSnapshotCompleted && m_fnOnSnapshotCompleted)
            {
                m_fnOnSnapshotCompleted(m_pUserDataOnSnapshotCompleted, snapshotNumber, isSnapshotSucceeded);
            }
            return true;
        }

        bool CancelSnapshot(uint32_t snapshotNumber)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto itrSnapshotCommand = m_activeSnapshotCommands.find(snapshotNumber);
            if (itrSnapshotCommand == m_activeSnapshotCommands.end())
            {
                return false;
            }
            TsfSnapshotCommand<TStorageWriter> removedSnapshotCommand = std::move(itrSnapshotCommand->second);
            m_activeSnapshotCommands.erase(itrSnapshotCommand);

            const uint64_t beginDataBlockSequenceNumber = removedSnapshotCommand.BeginDataBlockSequenceNumber();
            const uint64_t endDataBlockSequenceNumber = removedSnapshotCommand.EndDataBlockSequenceNumber();
            if (beginDataBlockSequenceNumber != TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER && endDataBlockSequenceNumber != TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER)
            {
                for (uint64_t dataBlockSequenceNumber = beginDataBlockSequenceNumber;; ++dataBlockSequenceNumber)
                {
                    auto itrOwnedDataBlockWriter = m_dataBlockWriters.find(dataBlockSequenceNumber);
                    if (itrOwnedDataBlockWriter != m_dataBlockWriters.end())
                    {
                        auto& [dataBlockWriter, refCount] = itrOwnedDataBlockWriter->second;
                        if (refCount > 0)
                        {
                            refCount -= 1;
                        }
                        if (refCount == 0 && !dataBlockWriter.IsInitialized())
                        {
                            m_dataBlockWriters.erase(itrOwnedDataBlockWriter);
                        }
                    }
                    if (dataBlockSequenceNumber == endDataBlockSequenceNumber)
                    {
                        break;
                    }
                }
            }
            return true;
        }

        bool RemoveCompletedSnapshot(uint32_t snapshotNumber, TsfSnapshotCommand<TStorageWriter>& out_removedSnapshotCommand)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto itrSnapshotCommand = m_completedSnapshotCommands.find(snapshotNumber);
            if (itrSnapshotCommand == m_completedSnapshotCommands.end())
            {
                return false;
            }
            out_removedSnapshotCommand = std::move(itrSnapshotCommand->second);
            m_completedSnapshotCommands.erase(itrSnapshotCommand);
            return true;
        }

        uint32_t AcknowledgePublishedDataBlock(TraceStreamDataBlockWriter& dataBlockWriter)
        {
            if (!dataBlockWriter.IsInitialized())
            {
                return 0;
            }
            const uint64_t dataBlockSequenceNumber = dataBlockWriter.DataBlockSequenceNumber();
            const uint64_t lowerBoundTimestampUtcNsec = dataBlockWriter.RecordTimestampLowerBoundUtcNsec();
            const uint64_t upperBoundTimestampUtcNsec = dataBlockWriter.RecordTimestampUpperBoundUtcNsec();
            std::lock_guard<std::mutex> lock(m_mutex);
            uint32_t acknowledgedSnapshotCount = 0;
            for (auto& [_, snapshotCommand] : m_activeSnapshotCommands)
            {
                if (snapshotCommand.AcknowledgePublishedDataBlock(dataBlockSequenceNumber, lowerBoundTimestampUtcNsec, upperBoundTimestampUtcNsec))
                {
                    ++acknowledgedSnapshotCount;
                }
            }
            if (acknowledgedSnapshotCount > 0)
            {
                m_dataBlockWriters[dataBlockSequenceNumber] = std::pair<TraceStreamDataBlockWriter, uint32_t>{std::move(dataBlockWriter), acknowledgedSnapshotCount};
            }
            return acknowledgedSnapshotCount;
        }

        uint32_t AcknowledgeCompletedDataBlock(uint64_t dataBlockSequenceNumber, bool isSucceeded, bool isAttachDataBlock)
        {
            uint32_t acknowledgedSnapshotCount = 0;
            std::vector<std::pair<uint32_t, bool>> completedSnapshots;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!isAttachDataBlock)
                {
                    auto itrDataBlockWriter = m_dataBlockWriters.find(dataBlockSequenceNumber);
                    if (itrDataBlockWriter != m_dataBlockWriters.end())
                    {
                        auto& [dataBlockWriter, refCount] = itrDataBlockWriter->second;
                        if (dataBlockWriter.IsInitialized())
                        {
                            refCount = 0;
                        }
                        else
                        {
                            m_dataBlockWriters.erase(itrDataBlockWriter);
                        }
                    }
                }
                for (auto& [_, snapshotCommand] : m_activeSnapshotCommands)
                {
                    if (snapshotCommand.AcknowledgeCompletedDataBlock(dataBlockSequenceNumber, isSucceeded, isAttachDataBlock))
                    {
                        ++acknowledgedSnapshotCount;
                    }
                }
                for (auto itrSnapshotCommand = m_finalizedSnapshotCommands.begin(); itrSnapshotCommand != m_finalizedSnapshotCommands.end();)
                {
                    TsfSnapshotCommand<TStorageWriter>& snapshotCommand = itrSnapshotCommand->second;
                    if (snapshotCommand.AcknowledgeCompletedDataBlock(dataBlockSequenceNumber, isSucceeded, isAttachDataBlock))
                    {
                        ++acknowledgedSnapshotCount;
                    }
                    bool isSnapshotSucceeded = false;
                    if (snapshotCommand.IsCompleted(isSnapshotSucceeded))
                    {
                        completedSnapshots.emplace_back(itrSnapshotCommand->first, isSnapshotSucceeded);
                        m_completedSnapshotCommands.emplace(itrSnapshotCommand->first, std::move(snapshotCommand));
                        itrSnapshotCommand = m_finalizedSnapshotCommands.erase(itrSnapshotCommand);
                        continue;
                    }
                    ++itrSnapshotCommand;
                }
            }
            if (m_fnOnSnapshotCompleted)
            {
                for (const auto& completedSnapshot : completedSnapshots)
                {
                    m_fnOnSnapshotCompleted(m_pUserDataOnSnapshotCompleted, completedSnapshot.first, completedSnapshot.second);
                }
            }
            return acknowledgedSnapshotCount;
        }

    private:
        bool AcknowledgePublishedDataBlock(
            uint32_t snapshotNumber,
            uint64_t dataBlockSequenceNumber,
            uint64_t lowerBoundTimestampUtcNsec,
            uint64_t upperBoundTimestampUtcNsec,
            TraceStreamDataBlockWriter* pIncomingDataBlockWriter,
            bool isInStorage)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto itrSnapshotCommand = m_activeSnapshotCommands.find(snapshotNumber);
            if (itrSnapshotCommand == m_activeSnapshotCommands.end())
            {
                return false;
            }
            TsfSnapshotCommand<TStorageWriter>& snapshotCommand = itrSnapshotCommand->second;
            if (!snapshotCommand.AcknowledgePublishedDataBlock(
                    dataBlockSequenceNumber,
                    lowerBoundTimestampUtcNsec,
                    upperBoundTimestampUtcNsec))
            {
                return false;
            }

            const auto itrDataBlockWriter = m_dataBlockWriters.lower_bound(dataBlockSequenceNumber);
            if (itrDataBlockWriter != m_dataBlockWriters.end() && itrDataBlockWriter->first == dataBlockSequenceNumber)
            {
                auto& [dataBlockWriter, refCount] = itrDataBlockWriter->second;
                refCount++;
                if (pIncomingDataBlockWriter && pIncomingDataBlockWriter->IsInitialized())
                {
                    dataBlockWriter = std::move(*pIncomingDataBlockWriter);
                }
                return true;
            }
            if (pIncomingDataBlockWriter && pIncomingDataBlockWriter->IsInitialized())
            {
                m_dataBlockWriters.emplace_hint(itrDataBlockWriter, dataBlockSequenceNumber, std::pair<TraceStreamDataBlockWriter, uint32_t>{std::move(*pIncomingDataBlockWriter), 1});
                return true;
            }
            snapshotCommand.AcknowledgeCompletedDataBlock(dataBlockSequenceNumber, isInStorage, false);
            return true;
        }

        std::mutex m_mutex;
        uint32_t m_snapshotNumberGenerator = INVALID_SNAPSHOT_NUMBER;
        // [TODO] Change to non-throwing data structure based on MAX_NUM_CONCURRENT_SNAPSHOTS
        // <SnapshotNumber, TsfSnapshotCommand>
        std::unordered_map<uint32_t, TsfSnapshotCommand<TStorageWriter>> m_activeSnapshotCommands;
        std::unordered_map<uint32_t, TsfSnapshotCommand<TStorageWriter>> m_finalizedSnapshotCommands;
        std::unordered_map<uint32_t, TsfSnapshotCommand<TStorageWriter>> m_completedSnapshotCommands;
        // One writer is shared by all snapshots that expect the same data block.
        // RefCount 0 means initialized but unwanted, eligible to RelinquishDataBlock first.
        // <DataBlockSequenceNumber, {DataBlockWriter, RefCount}>
        std::map<uint64_t, std::pair<TraceStreamDataBlockWriter, uint32_t>> m_dataBlockWriters;
        void (*m_fnOnSnapshotCompleted)(void* pUserData, uint32_t snapshotNumber, bool succeeded) = nullptr;
        void* m_pUserDataOnSnapshotCompleted = nullptr;
    };

} //namespace nv::trace
