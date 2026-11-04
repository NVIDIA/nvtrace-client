/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceConsumerBase.h>
#include <NvTrace/Target/Consumer/NvTraceCudaApiTraceSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceNvtxTraceSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceRecordWriterUtilities.h>
#include <NvTrace/Target/Consumer/NvTraceStreamDatabaseWriter.h>
#include <NvTrace/Target/Consumer/NvTraceThreadEventSerialization.h>
#include <NvTrace/Target/Consumer/NvTraceTsfBufferEventEncoders.h>
#include <NvTrace/Target/Consumer/NvTraceTsfSnapshotRecordWriter.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceSectionLayoutBuilder.h>
#include <NvTrace/Target/NvTraceWaitableObjects.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    struct ContinuousSnapshotDataBlockFile
    {
        std::filesystem::path filePath;
        uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
    };

    struct ContinuousSnapshotManifest
    {
        uint64_t snapshotNumber = 0;
        uint64_t requestedBeginTimestampUtcNsec = 0;
        uint64_t cutoffTimestampUtcNsec = 0;
        uint64_t actualBeginTimestampUtcNsec = 0;
        uint64_t actualEndTimestampUtcNsec = 0;
        // Records encountered while all latency-buffer slots were occupied. Their observer
        // cursors remain in place unless the provider laps them first.
        uint64_t backpressuredRecords = 0;
        bool historyTruncated = false;
        // The export callback still runs when this is false.
        bool attachComplete = false;
        uint64_t droppedAttachRecordCount = 0;
        uint64_t droppedAttachBytes = 0;
        uint64_t minimumRequiredAttachSizeBytes = 0;
        uint64_t minimumRequiredDataBlockSizeBytes = 0;
        std::vector<ContinuousSnapshotDataBlockFile> dataBlockFiles;
    };

    namespace Detail {

        constexpr size_t ContinuousSnapshotOverflowPreventionReserveBytes = 10 * 1024 * 1024;

        inline size_t CalculateContinuousSnapshotOverflowPreventionReserveDataBlockCount(size_t dataBlockSize)
        {
            return ContinuousSnapshotOverflowPreventionReserveBytes / dataBlockSize
                 + (ContinuousSnapshotOverflowPreventionReserveBytes % dataBlockSize != 0 ? 1 : 0);
        }

        inline bool CalculateContinuousSnapshotHistoryDataBlockCount(
            uint64_t historyDurationNsec,
            uint64_t provisionedDataRateBytesPerSec,
            size_t dataBlockSize,
            size_t& out_dataBlockCount)
        {
            constexpr uint64_t nsecPerSec = 1'000'000'000;
            if (!historyDurationNsec || !provisionedDataRateBytesPerSec || !dataBlockSize)
            {
                return false;
            }

            const uint64_t wholeSeconds = historyDurationNsec / nsecPerSec;
            const uint64_t remainderNsec = historyDurationNsec % nsecPerSec;
            if (wholeSeconds && provisionedDataRateBytesPerSec > std::numeric_limits<uint64_t>::max() / wholeSeconds)
            {
                return false;
            }
            uint64_t provisionedBytes = provisionedDataRateBytesPerSec * wholeSeconds;
            if (remainderNsec)
            {
                if (provisionedDataRateBytesPerSec > std::numeric_limits<uint64_t>::max() / remainderNsec)
                {
                    return false;
                }
                const uint64_t remainderProduct = provisionedDataRateBytesPerSec * remainderNsec;
                const uint64_t remainderBytes = remainderProduct / nsecPerSec + (remainderProduct % nsecPerSec != 0 ? 1 : 0);
                if (provisionedBytes > std::numeric_limits<uint64_t>::max() - remainderBytes)
                {
                    return false;
                }
                provisionedBytes += remainderBytes;
            }

            const uint64_t roundedCount = provisionedBytes / dataBlockSize + (provisionedBytes % dataBlockSize != 0 ? 1 : 0);
            const size_t overflowPreventionReserveDataBlockCount = CalculateContinuousSnapshotOverflowPreventionReserveDataBlockCount(dataBlockSize);
            if (roundedCount >= std::numeric_limits<size_t>::max() - overflowPreventionReserveDataBlockCount)
            {
                return false;
            }
            // One extra block covers the partially filled block at a snapshot boundary. The
            // overflow-prevention reserve absorbs a one-cycle spike above the measured rate.
            out_dataBlockCount = std::max<size_t>(8, static_cast<size_t>(roundedCount + 1)) + overflowPreventionReserveDataBlockCount;
            return true;
        }

        class ContinuousSnapshotStorageState
        {
        public:
            static constexpr size_t INVALID_BANK_INDEX = std::numeric_limits<size_t>::max();

            struct ExportJob
            {
                size_t bankIndex = INVALID_BANK_INDEX;
                ContinuousSnapshotManifest manifest;
                uint64_t maxConsumeDataDurationNsec = 0;
                bool storageSucceeded = true;
            };

            struct StorageCompletion
            {
                size_t bankIndex = INVALID_BANK_INDEX;
                size_t slotIndex = 0;
                uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                uint64_t lowerTimestampUtcNsec = 0;
                uint64_t upperTimestampUtcNsec = 0;
                bool succeeded = false;
            };

            ~ContinuousSnapshotStorageState()
            {
                Reset();
            }

            ContinuousSnapshotStorageState() = default;
            ContinuousSnapshotStorageState(const ContinuousSnapshotStorageState&) = delete;
            ContinuousSnapshotStorageState& operator=(const ContinuousSnapshotStorageState&) = delete;

            void Reset() noexcept
            {
                std::filesystem::path sessionDirectory = std::move(m_sessionDirectory);
                const bool deleteFilesOnReset = m_deleteFilesOnReset;
                m_banks[0] = {};
                m_banks[1] = {};
                m_historyFiles[0].clear();
                m_historyFiles[1].clear();
                m_currentBankIndex = 0;
                m_historyDurationNsec = 0;
                m_dataBlockSize = 0;
                m_nextSnapshotNumber = 1;
                m_attachFilePaths.clear();
                m_deleteFilesOnReset = false;
                m_initialized = false;
                if (deleteFilesOnReset && !sessionDirectory.empty())
                {
                    std::error_code errorCode;
                    std::filesystem::remove_all(sessionDirectory, errorCode);
                    NVTRACE_LOG_WRN_IF(errorCode != std::error_code{}, 20, "Failed to remove continuous snapshot temporary directory at %s", sessionDirectory.string().c_str());
                }
            }

            bool Initialize(
                const std::filesystem::path& outputDirectory,
                size_t dataBlockSize,
                size_t dataBlockCountPerBank,
                size_t maxAttachSize,
                uint64_t historyDurationNsec,
                bool deleteFilesOnReset)
            {
                Reset();
                NVTRACE_LOG_ERR_RETURN_IF(outputDirectory.empty(), false, 20, "Continuous snapshot output directory is empty");
                NVTRACE_LOG_ERR_RETURN_IF(dataBlockSize < sizeof(TsfDataBlockHeader), false, 20, "Continuous snapshot DataBlock size is too small");
                NVTRACE_LOG_ERR_RETURN_IF(dataBlockCountPerBank < 2, false, 20, "Continuous snapshot history must contain at least two DataBlocks per bank");
                NVTRACE_LOG_ERR_RETURN_IF(maxAttachSize < dataBlockSize, false, 20, "Continuous snapshot maximum attachment size must be at least one DataBlock");

                const size_t attachDataBlockCount = maxAttachSize / dataBlockSize + (maxAttachSize % dataBlockSize != 0 ? 1 : 0);
                constexpr size_t maxReservedDataBlockCount = TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH - TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW + 1;
                NVTRACE_LOG_ERR_RETURN_IF(maxReservedDataBlockCount < attachDataBlockCount, false, 20, "Continuous snapshot maximum attachment size requires too many reserved DataBlocks");

                std::error_code errorCode;
                std::filesystem::create_directories(outputDirectory, errorCode);
                NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to create continuous snapshot output directory at %s", outputDirectory.string().c_str());

                static std::atomic<uint64_t> nextDirectoryId = 0;
                const uint64_t directoryId = nextDirectoryId.fetch_add(1, std::memory_order_relaxed);
                std::stringstream directoryName;
                directoryName << "continuous-snapshot-" << std::hex << ReadCpuTscTimestamp() << '-' << directoryId;
                m_sessionDirectory = outputDirectory / directoryName.str();
                if (!std::filesystem::create_directory(m_sessionDirectory, errorCode) || errorCode != std::error_code{})
                {
                    NVTRACE_LOG_ERR(20, "Failed to create continuous snapshot session directory at %s", m_sessionDirectory.string().c_str());
                    return false;
                }

                // Incomplete initialization always owns and removes its partially constructed
                // session directory, irrespective of the successful-lifetime cleanup policy.
                m_deleteFilesOnReset = true;
                auto cleanupGuard = ScopeExitGuard([this]() { Reset(); });

                for (size_t bankIndex = 0; bankIndex < 2; ++bankIndex)
                {
                    m_banks[bankIndex].slots.resize(dataBlockCountPerBank);
                    m_historyFiles[bankIndex].resize(dataBlockCountPerBank);
                    for (size_t slotIndex = 0; slotIndex < dataBlockCountPerBank; ++slotIndex)
                    {
                        std::stringstream fileName;
                        fileName << "history-" << (bankIndex == 0 ? 'Y' : 'Z') << '-' << std::setfill('0') << std::setw(6) << slotIndex << ".nvtsf";
                        m_banks[bankIndex].slots[slotIndex].filePath = m_sessionDirectory / fileName.str();
                        if (!PreallocateFile(m_banks[bankIndex].slots[slotIndex].filePath, dataBlockSize))
                        {
                            return false;
                        }
                        m_historyFiles[bankIndex][slotIndex].file.open(
                            m_banks[bankIndex].slots[slotIndex].filePath,
                            std::ios::binary | std::ios::in | std::ios::out);
                        NVTRACE_LOG_ERR_RETURN_IF(
                            m_historyFiles[bankIndex][slotIndex].file.fail(),
                            false,
                            20,
                            "Failed to open preallocated continuous snapshot file at %s",
                            m_banks[bankIndex].slots[slotIndex].filePath.string().c_str());
                    }
                }

                m_attachFilePaths.reserve(attachDataBlockCount);
                for (size_t attachDataBlockIndex = 0; attachDataBlockIndex < attachDataBlockCount; ++attachDataBlockIndex)
                {
                    std::stringstream fileName;
                    fileName << "attach-" << std::setfill('0') << std::setw(6) << attachDataBlockIndex << ".nvtsf";
                    m_attachFilePaths.push_back(m_sessionDirectory / fileName.str());
                    if (!PreallocateFile(m_attachFilePaths.back(), dataBlockSize))
                    {
                        return false;
                    }
                }

                m_banks[0].status = BankStatus::Live;
                m_banks[1].status = BankStatus::Free;
                m_currentBankIndex = 0;
                m_historyDurationNsec = historyDurationNsec;
                m_dataBlockSize = dataBlockSize;
                m_deleteFilesOnReset = deleteFilesOnReset;
                m_initialized = true;
                cleanupGuard.Dismiss();
                return true;
            }

            bool WriteDataBlock(
                size_t bankIndex,
                size_t slotIndex,
                const uint8_t* pDataBlock,
                size_t dataBlockSize,
                uint64_t& out_lowerTimestampUtcNsec,
                uint64_t& out_upperTimestampUtcNsec)
            {
                NVTRACE_LOG_ERR_RETURN_IF(!pDataBlock || dataBlockSize != m_dataBlockSize, false, 20, "Continuous snapshot storage received an invalid DataBlock");
                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex || m_historyFiles[bankIndex].size() <= slotIndex, false, 20, "Continuous snapshot storage received an invalid history destination");
                const auto* const pHeader = reinterpret_cast<const TsfDataBlockHeader*>(pDataBlock);
                NVTRACE_LOG_ERR_RETURN_IF(pHeader->offsetofDataBlockState + sizeof(TsfDataBlockState) > dataBlockSize, false, 20, "Continuous snapshot storage received a malformed DataBlock");
                const auto* const pState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + pHeader->offsetofDataBlockState);
                out_lowerTimestampUtcNsec = pState->recordTimestampLowerBoundUtcNsec;
                out_upperTimestampUtcNsec = pState->recordTimestampUpperBoundUtcNsec;
                return OverwriteFile(m_historyFiles[bankIndex][slotIndex].file, pDataBlock, dataBlockSize);
            }

            bool CanReserveDataBlock() const
            {
                return m_initialized && m_banks[m_currentBankIndex].nextSlotIndex < m_banks[m_currentBankIndex].slots.size();
            }

            bool ReserveDataBlock(uint64_t dataBlockSequenceNumber, size_t& out_bankIndex, size_t& out_slotIndex)
            {
                NVTRACE_LOG_ERR_RETURN_IF(!m_initialized, false, 20, "Continuous snapshot storage is not initialized");
                Bank& bank = m_banks[m_currentBankIndex];
                NVTRACE_LOG_ERR_RETURN_IF(bank.status != BankStatus::Live || bank.nextSlotIndex >= bank.slots.size(), false, 20, "Continuous snapshot history bank is full or unavailable");
                out_bankIndex = m_currentBankIndex;
                out_slotIndex = bank.nextSlotIndex++;
                Slot& slot = bank.slots[out_slotIndex];
                slot.sequenceNumber = dataBlockSequenceNumber;
                slot.lowerTimestampUtcNsec = 0;
                slot.upperTimestampUtcNsec = 0;
                slot.valid = false;
                slot.writeCompleted = false;
                bank.lastDataBlockSequenceNumber = dataBlockSequenceNumber;
                bank.pendingWriteCount += 1;
                return true;
            }

            // Defensive rollback for an invariant failure after ReserveDataBlock(). Normal
            // operation always binds the reserved slot to its storage-writer destination.
            void CancelDataBlockReservation(size_t bankIndex, size_t slotIndex)
            {
                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex, /*void*/, 20, "Continuous snapshot tried to cancel an invalid bank reservation");
                Bank& bank = m_banks[bankIndex];
                NVTRACE_LOG_ERR_RETURN_IF(bank.nextSlotIndex == 0 || slotIndex + 1 != bank.nextSlotIndex || bank.pendingWriteCount == 0, /*void*/, 20, "Continuous snapshot tried to cancel a reservation that is not the newest slot");
                bank.nextSlotIndex -= 1;
                bank.pendingWriteCount -= 1;
                Slot& slot = bank.slots[slotIndex];
                slot.sequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                slot.lowerTimestampUtcNsec = 0;
                slot.upperTimestampUtcNsec = 0;
                slot.valid = false;
                slot.writeCompleted = false;
                bank.lastDataBlockSequenceNumber = bank.nextSlotIndex == 0 ? TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER : bank.slots[bank.nextSlotIndex - 1].sequenceNumber;
            }

            size_t RemainingDataBlockCount() const
            {
                if (!m_initialized)
                {
                    return 0;
                }
                const Bank& bank = m_banks[m_currentBankIndex];
                return bank.slots.size() - bank.nextSlotIndex;
            }

            bool ApplyStorageCompletion(const StorageCompletion& completion, size_t& out_readyBankIndex)
            {
                out_readyBankIndex = INVALID_BANK_INDEX;
                NVTRACE_LOG_ERR_RETURN_IF(2 <= completion.bankIndex, false, 20, "Continuous snapshot storage completion has an invalid bank index");
                Bank& bank = m_banks[completion.bankIndex];
                NVTRACE_LOG_ERR_RETURN_IF(bank.nextSlotIndex <= completion.slotIndex, false, 20, "Continuous snapshot storage completion has an invalid slot index");
                Slot& slot = bank.slots[completion.slotIndex];
                NVTRACE_LOG_ERR_RETURN_IF(slot.sequenceNumber != completion.dataBlockSequenceNumber || slot.writeCompleted, false, 20, "Continuous snapshot storage completion does not match its reserved slot");
                slot.lowerTimestampUtcNsec = completion.lowerTimestampUtcNsec;
                slot.upperTimestampUtcNsec = completion.upperTimestampUtcNsec;
                slot.valid = completion.succeeded;
                slot.writeCompleted = true;
                bank.storageSucceeded = bank.storageSucceeded && completion.succeeded;
                NVTRACE_LOG_ERR_RETURN_IF(bank.pendingWriteCount == 0, false, 20, "Continuous snapshot bank has no pending storage writes");
                bank.pendingWriteCount -= 1;
                if (bank.status == BankStatus::SealedWriting && bank.pendingWriteCount == 0)
                {
                    bank.status = BankStatus::Ready;
                    out_readyBankIndex = completion.bankIndex;
                }
                return true;
            }

            bool CanChangeover() const
            {
                return m_initialized && m_banks[1 - m_currentBankIndex].status == BankStatus::Free;
            }

            bool SetLiveBankBeginTimestamp(uint64_t beginTimestampUtcNsec)
            {
                NVTRACE_LOG_ERR_RETURN_IF(!m_initialized, false, 20, "Continuous snapshot storage is not initialized");
                Bank& bank = m_banks[m_currentBankIndex];
                NVTRACE_LOG_ERR_RETURN_IF(bank.status != BankStatus::Live || bank.nextSlotIndex != 0, false, 20, "Continuous snapshot live bank has already started");
                bank.beginTimestampUtcNsec = beginTimestampUtcNsec;
                return true;
            }

            void RecordBackpressuredRecords(uint64_t backpressuredRecords)
            {
                if (m_initialized)
                {
                    m_banks[m_currentBankIndex].backpressuredRecords += backpressuredRecords;
                }
            }

            void RecordConsumeDataDuration(uint64_t durationNsec)
            {
                if (m_initialized)
                {
                    Bank& bank = m_banks[m_currentBankIndex];
                    bank.maxConsumeDataDurationNsec = std::max(bank.maxConsumeDataDurationNsec, durationNsec);
                }
            }

            bool SealAndSwitch(uint64_t lastDataBlockSequenceNumber, uint64_t cutoffTimestampUtcNsec, bool hadDataBlock, bool succeeded, size_t& out_readyBankIndex)
            {
                out_readyBankIndex = INVALID_BANK_INDEX;
                NVTRACE_LOG_ERR_RETURN_IF(!m_initialized, false, 20, "Continuous snapshot storage is not initialized");
                const size_t sealedBankIndex = m_currentBankIndex;
                const size_t standbyBankIndex = 1 - sealedBankIndex;
                NVTRACE_LOG_ERR_RETURN_IF(m_banks[standbyBankIndex].status != BankStatus::Free, false, 20, "Continuous snapshot changeover attempted while the standby bank is unavailable");
                SealBank(sealedBankIndex, lastDataBlockSequenceNumber, cutoffTimestampUtcNsec, hadDataBlock, succeeded, out_readyBankIndex);
                m_banks[standbyBankIndex].status = BankStatus::Live;
                m_banks[standbyBankIndex].beginTimestampUtcNsec = cutoffTimestampUtcNsec;
                m_currentBankIndex = standbyBankIndex;
                return true;
            }

            bool SealForStop(uint64_t lastDataBlockSequenceNumber, uint64_t cutoffTimestampUtcNsec, bool hadDataBlock, bool succeeded, size_t& out_readyBankIndex)
            {
                out_readyBankIndex = INVALID_BANK_INDEX;
                NVTRACE_LOG_ERR_RETURN_IF(!m_initialized, false, 20, "Continuous snapshot storage is not initialized");
                Bank& bank = m_banks[m_currentBankIndex];
                NVTRACE_LOG_ERR_RETURN_IF(bank.status != BankStatus::Live, false, 20, "Continuous snapshot final changeover requires a live bank");
                if (!hadDataBlock && bank.nextSlotIndex == 0 && cutoffTimestampUtcNsec <= bank.beginTimestampUtcNsec)
                {
                    PrepareBankForLiveUse(bank);
                    bank.status = BankStatus::Free;
                    return succeeded;
                }
                SealBank(m_currentBankIndex, lastDataBlockSequenceNumber, cutoffTimestampUtcNsec, hadDataBlock, succeeded, out_readyBankIndex);
                return true;
            }

            bool IsDrained() const
            {
                return m_initialized
                    && m_banks[0].status == BankStatus::Free
                    && m_banks[1].status == BankStatus::Free;
            }

            bool TransferReadyBankToExport(size_t bankIndex)
            {
                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex || m_banks[bankIndex].status != BankStatus::Ready, false, 20, "Continuous snapshot tried to export a bank that is not ready");
                m_banks[bankIndex].status = BankStatus::Exporting;
                return true;
            }

            bool BuildExportJob(size_t bankIndex, ExportJob& out_job) const
            {
                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex || m_banks[bankIndex].status != BankStatus::Exporting, false, 20, "Continuous snapshot export thread received a bank it does not own");
                const Bank& bank = m_banks[bankIndex];
                out_job = {};
                out_job.bankIndex = bankIndex;
                out_job.maxConsumeDataDurationNsec = bank.maxConsumeDataDurationNsec;
                out_job.storageSucceeded = bank.storageSucceeded;
                out_job.manifest.snapshotNumber = bank.snapshotNumber;
                out_job.manifest.cutoffTimestampUtcNsec = bank.cutoffTimestampUtcNsec;
                out_job.manifest.requestedBeginTimestampUtcNsec = bank.cutoffTimestampUtcNsec > m_historyDurationNsec ? bank.cutoffTimestampUtcNsec - m_historyDurationNsec : 0;
                out_job.manifest.backpressuredRecords = bank.backpressuredRecords;

                bool hasTimestamps = false;
                for (const Slot& slot : bank.slots)
                {
                    if (!slot.valid)
                    {
                        continue;
                    }
                    out_job.manifest.dataBlockFiles.push_back({slot.filePath, slot.sequenceNumber});
                    if (!hasTimestamps)
                    {
                        out_job.manifest.actualBeginTimestampUtcNsec = slot.lowerTimestampUtcNsec;
                        out_job.manifest.actualEndTimestampUtcNsec = slot.upperTimestampUtcNsec;
                        hasTimestamps = true;
                    }
                    else
                    {
                        out_job.manifest.actualBeginTimestampUtcNsec = std::min(out_job.manifest.actualBeginTimestampUtcNsec, slot.lowerTimestampUtcNsec);
                        out_job.manifest.actualEndTimestampUtcNsec = std::max(out_job.manifest.actualEndTimestampUtcNsec, slot.upperTimestampUtcNsec);
                    }
                }
                if (!hasTimestamps)
                {
                    out_job.manifest.actualBeginTimestampUtcNsec = bank.cutoffTimestampUtcNsec;
                    out_job.manifest.actualEndTimestampUtcNsec = bank.cutoffTimestampUtcNsec;
                }
                out_job.manifest.historyTruncated = out_job.manifest.actualBeginTimestampUtcNsec > out_job.manifest.requestedBeginTimestampUtcNsec;
                return true;
            }

            bool ReclaimReturnedBank(size_t bankIndex)
            {
                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex || m_banks[bankIndex].status != BankStatus::Exporting, false, 20, "Continuous snapshot returned an invalid export bank");
                PrepareBankForLiveUse(m_banks[bankIndex]);
                m_banks[bankIndex].status = BankStatus::Free;
                return true;
            }

            const std::vector<std::filesystem::path>& AttachFilePaths() const
            {
                return m_attachFilePaths;
            }

        private:
            enum class BankStatus
            {
                Free,
                Live,
                SealedWriting,
                Ready,
                Exporting,
            };

            struct Slot
            {
                std::filesystem::path filePath;
                uint64_t sequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                uint64_t lowerTimestampUtcNsec = 0;
                uint64_t upperTimestampUtcNsec = 0;
                bool valid = false;
                bool writeCompleted = false;
            };

            struct HistoryFile
            {
                std::fstream file;
            };

            struct Bank
            {
                BankStatus status = BankStatus::Free;
                std::vector<Slot> slots;
                size_t nextSlotIndex = 0;
                uint64_t beginTimestampUtcNsec = 0;
                uint64_t lastDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                uint64_t cutoffTimestampUtcNsec = 0;
                uint64_t snapshotNumber = 0;
                uint64_t backpressuredRecords = 0;
                uint64_t maxConsumeDataDurationNsec = 0;
                size_t pendingWriteCount = 0;
                bool storageSucceeded = true;
            };

            static bool PreallocateFile(const std::filesystem::path& filePath, size_t fileSize)
            {
                std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to create continuous snapshot file at %s", filePath.string().c_str());
                constexpr size_t zeroBufferSize = 1024 * 1024;
                const std::vector<char> zeros(std::min(fileSize, zeroBufferSize), 0);
                size_t remainingSize = fileSize;
                while (remainingSize)
                {
                    const size_t writeSize = std::min(remainingSize, zeros.size());
                    file.write(zeros.data(), static_cast<std::streamsize>(writeSize));
                    NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to preallocate continuous snapshot file at %s", filePath.string().c_str());
                    remainingSize -= writeSize;
                }
                file.close();
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to close continuous snapshot file at %s", filePath.string().c_str());
                return true;
            }

            static bool OverwriteFile(std::fstream& file, const uint8_t* pData, size_t dataSize)
            {
                file.clear();
                file.seekp(0);
                file.write(reinterpret_cast<const char*>(pData), static_cast<std::streamsize>(dataSize));
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to overwrite a preallocated continuous snapshot file");
                file.flush();
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to flush a preallocated continuous snapshot file");
                return true;
            }

            static void PrepareBankForLiveUse(Bank& bank)
            {
                bank.status = BankStatus::Live;
                bank.nextSlotIndex = 0;
                bank.beginTimestampUtcNsec = 0;
                bank.lastDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                bank.cutoffTimestampUtcNsec = 0;
                bank.snapshotNumber = 0;
                bank.backpressuredRecords = 0;
                bank.maxConsumeDataDurationNsec = 0;
                bank.pendingWriteCount = 0;
                bank.storageSucceeded = true;
                for (Slot& slot : bank.slots)
                {
                    slot.sequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                    slot.lowerTimestampUtcNsec = 0;
                    slot.upperTimestampUtcNsec = 0;
                    slot.valid = false;
                    slot.writeCompleted = false;
                }
            }

            void SealBank(
                size_t bankIndex,
                uint64_t lastDataBlockSequenceNumber,
                uint64_t cutoffTimestampUtcNsec,
                bool hadDataBlock,
                bool succeeded,
                size_t& out_readyBankIndex)
            {
                Bank& bank = m_banks[bankIndex];
                bank.cutoffTimestampUtcNsec = cutoffTimestampUtcNsec;
                if (bank.nextSlotIndex != 0)
                {
                    NVTRACE_LOG_ERR_IF(hadDataBlock && bank.lastDataBlockSequenceNumber != lastDataBlockSequenceNumber, 20, "Continuous snapshot checkpoint sequence does not match the reserved history slot");
                }
                bank.storageSucceeded = bank.storageSucceeded && succeeded;
                bank.snapshotNumber = m_nextSnapshotNumber++;
                bank.status = bank.pendingWriteCount == 0 ? BankStatus::Ready : BankStatus::SealedWriting;
                if (bank.status == BankStatus::Ready)
                {
                    out_readyBankIndex = bankIndex;
                }
            }

        private:
            Bank m_banks[2];
            std::vector<HistoryFile> m_historyFiles[2];
            size_t m_currentBankIndex = 0;
            size_t m_dataBlockSize = 0;
            uint64_t m_historyDurationNsec = 0;
            uint64_t m_nextSnapshotNumber = 1;
            std::filesystem::path m_sessionDirectory;
            std::vector<std::filesystem::path> m_attachFilePaths;
            bool m_deleteFilesOnReset = false;
            bool m_initialized = false;
        };

        class ContinuousSnapshotStorageWriter
        {
        public:
            struct WriteMetadata
            {
                size_t bankIndex = ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
                size_t slotIndex = 0;
                uint64_t expectedDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                uint64_t lowerTimestampUtcNsec = 0;
                uint64_t upperTimestampUtcNsec = 0;
            };

            void Reset() noexcept
            {
                m_writeMetadata.clear();
                m_pState = nullptr;
            }

            bool Initialize(ContinuousSnapshotStorageState* pState, uint8_t writeSlotCount)
            {
                m_pState = pState;
                m_writeMetadata.resize(writeSlotCount);
                return m_pState && !m_writeMetadata.empty();
            }

            bool SetWriteTarget(uint8_t writeSlotIndex, size_t bankIndex, size_t slotIndex, uint64_t dataBlockSequenceNumber)
            {
                NVTRACE_LOG_ERR_RETURN_IF(m_writeMetadata.size() <= writeSlotIndex, false, 20, "Continuous snapshot received an invalid write-slot index");
                WriteMetadata& metadata = m_writeMetadata[writeSlotIndex];
                NVTRACE_LOG_ERR_RETURN_IF(metadata.bankIndex != ContinuousSnapshotStorageState::INVALID_BANK_INDEX, false, 20, "Continuous snapshot write slot already has a history destination");
                metadata.bankIndex = bankIndex;
                metadata.slotIndex = slotIndex;
                metadata.expectedDataBlockSequenceNumber = dataBlockSequenceNumber;
                return true;
            }

            // ContinuousSnapshotExporter has no Snapshot() command path. This
            // method only satisfies TraceStreamDatabaseWriter's storage-writer
            // contract; history banks are transferred by SealAndSwitch().
            bool HardlinkDataBlock(ContinuousSnapshotStorageWriter&, uint64_t) const
            {
                return false;
            }

            bool WriteDataBlock(uint8_t writeSlotIndex, const uint8_t* pDataBlock, size_t dataBlockSize, uint64_t dataBlockSequenceNumber)
            {
                NVTRACE_LOG_ERR_RETURN_IF(!m_pState || m_writeMetadata.size() <= writeSlotIndex, false, 20, "Continuous snapshot storage received an invalid write-slot index");
                WriteMetadata& metadata = m_writeMetadata[writeSlotIndex];
                NVTRACE_LOG_ERR_RETURN_IF(metadata.bankIndex == ContinuousSnapshotStorageState::INVALID_BANK_INDEX || metadata.expectedDataBlockSequenceNumber != dataBlockSequenceNumber, false, 20, "Continuous snapshot write slot has no matching history destination");
                return m_pState->WriteDataBlock(
                    metadata.bankIndex,
                    metadata.slotIndex,
                    pDataBlock,
                    dataBlockSize,
                    metadata.lowerTimestampUtcNsec,
                    metadata.upperTimestampUtcNsec);
            }

            bool TakeStorageCompletion(uint8_t writeSlotIndex, uint64_t dataBlockSequenceNumber, bool succeeded, ContinuousSnapshotStorageState::StorageCompletion& out_completion)
            {
                NVTRACE_LOG_ERR_RETURN_IF(m_writeMetadata.size() <= writeSlotIndex, false, 20, "Continuous snapshot completion received an invalid write-slot index");
                WriteMetadata& metadata = m_writeMetadata[writeSlotIndex];
                NVTRACE_LOG_ERR_RETURN_IF(metadata.bankIndex == ContinuousSnapshotStorageState::INVALID_BANK_INDEX || metadata.expectedDataBlockSequenceNumber != dataBlockSequenceNumber, false, 20, "Continuous snapshot completion has no matching history destination");
                out_completion.bankIndex = metadata.bankIndex;
                out_completion.slotIndex = metadata.slotIndex;
                out_completion.dataBlockSequenceNumber = dataBlockSequenceNumber;
                out_completion.lowerTimestampUtcNsec = metadata.lowerTimestampUtcNsec;
                out_completion.upperTimestampUtcNsec = metadata.upperTimestampUtcNsec;
                out_completion.succeeded = succeeded;
                metadata = {};
                return true;
            }

        private:
            ContinuousSnapshotStorageState* m_pState = nullptr;
            std::vector<WriteMetadata> m_writeMetadata;
        };

    } // namespace Detail
} // namespace nv::trace

namespace nv::trace {

    struct ContinuousSnapshotExporterObjectModel : ConsumerObjectModelBase
    {
        enum class ZombieParentType
        {
            Process,
            GpuPartition
        };

        struct ZombieParentQueueEntry
        {
            ZombieParentType type = ZombieParentType::Process;
            void* pState = nullptr;
            uint64_t expirationTimestampUtcNsec = 0;
            size_t retainedBytes = 0;
        };

        using ZombieParentQueue = std::deque<ZombieParentQueueEntry>;

        template <typename TNotification, NvTrace_RouterObserverClient_NotificationType TNotificationType>
        struct PendingDestroyState
        {
            static constexpr NvTrace_RouterObserverClient_NotificationType notificationType = TNotificationType;

            TNotification notification = {};
            bool isPending = false;

            void Reset() noexcept
            {
                notification = {};
                isPending = false;
            }

            explicit operator bool() const
            {
                return isPending;
            }
        };

        using PendingProviderDestroy = PendingDestroyState<NvTrace_RouterObserverClient_ProviderDestroyedNotification, NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed>;
        using PendingProcessDestroy = PendingDestroyState<NvTrace_RouterObserverClient_ProcessDestroyedNotification, NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed>;
        using PendingGpuPartitionDestroy = PendingDestroyState<NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification, NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed>;

        struct ProviderState : ConsumerObjectModelBase::ProviderState
        {
            PendingProviderDestroy pendingDestroy;
            std::vector<uint64_t> onAttachRecordLocations;
            uint64_t providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint32_t providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            bool hasProviderCreatedRecord = false;
            bool isReadyToConsume = true;

            void Reset() noexcept
            {
                pendingDestroy.Reset();
                threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                onAttachRecordLocations.clear();
                providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
                providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
                hasProviderCreatedRecord = false;
                isReadyToConsume = true;
                ConsumerObjectModelBase::ProviderState::Reset();
            }
        };
        using CpuClockCalibrationProviderState = TypedProviderState<CpuClockCalibrationObserver, ProviderState>;
        using GpuClockCalibrationProviderState = TypedProviderState<GpuClockCalibrationObserver, ProviderState>;
        struct GpuPmCountersProviderState : TypedProviderState<GpuPmCounterObserver, ProviderState>
        {
        };
        using GpuHwTraceProviderState = TypedProviderState<GpuHwTraceObserver, ProviderState>;
        using GpuGrCtxTraceProviderState = TypedProviderState<GpuGrCtxTraceObserver, ProviderState>;
        using CudaApiTraceProviderState = TypedCudaApiTraceProviderState<ProviderState>;
        using NvtxTraceProviderState = TypedNvtxTraceProviderState<ProviderState>;
        using ThreadEventProviderState = TypedThreadEventProviderState<ProviderState>;

        template <typename TGpuClockCalibrationProviderState, typename TGpuPmCountersProviderState, typename TGpuHwTraceProviderState, typename TGpuGrCtxTraceProviderState>
        struct GpuPartitionState : ConsumerObjectModelBase::GpuPartitionState<TGpuClockCalibrationProviderState, TGpuPmCountersProviderState, TGpuHwTraceProviderState, TGpuGrCtxTraceProviderState>
        {
            PendingGpuPartitionDestroy pendingDestroy;
            uint64_t gpuPartitionCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint64_t observationStartGpuNsec = 0;
            bool isObservationStartResolved = true;
        };

        template <typename TCudaApiTraceProviderState, typename TNvtxTraceProviderState, typename TThreadEventProviderState>
        struct ProcessState : ConsumerObjectModelBase::ProcessState<TCudaApiTraceProviderState, TNvtxTraceProviderState, TThreadEventProviderState>
        {
            PendingProcessDestroy pendingDestroy;
            uint64_t processCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        };

        template <typename TCpuClockCalibrationProviderState, typename TProviderState, typename TGpuPartitionState, typename TProcessState>
        struct SystemState : ConsumerObjectModelBase::SystemState<TCpuClockCalibrationProviderState, TProviderState, TGpuPartitionState, TProcessState>
        {
            uint64_t systemCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
        };
    };

} // namespace nv::trace

namespace nv::trace {

    template <typename TRecordWriter, typename TFN_ConsumeThreadEventSourceProviderRegistrations>
    class TsfAttachmentSerializer
    {
    public:
        TsfAttachmentSerializer(TRecordWriter& recordWriter, TFN_ConsumeThreadEventSourceProviderRegistrations& fnConsumeThreadEventSourceProviderRegistrations)
            : m_recordWriter(recordWriter)
            , m_fnConsumeThreadEventSourceProviderRegistrations(fnConsumeThreadEventSourceProviderRegistrations)
        {
        }

        template <typename TSystemState>
        void Serialize(TSystemState& systemState, uint64_t attachTimestampUtcNsec, bool includeZombies)
        {
            m_recordWriter.SetLastConsumeTimestampUtcNsec(attachTimestampUtcNsec);
            PrepareProviderAttachState(systemState);
            AttachRouterMessages(systemState, includeZombies);
            AttachStateMessages(systemState, includeZombies);
            AttachStateTables(systemState, includeZombies);
        }

    private:
        template <typename TProviderState>
        static bool ShouldAttachProvider(const TProviderState& providerState, bool includeZombies)
        {
            return providerState.IsInitialized() && (includeZombies || !providerState.pendingDestroy);
        }

        template <typename TSystemState>
        void PrepareProviderAttachState(TSystemState& systemState)
        {
            // Every attach DataBlock reuses the same reserved sequence number, so the per-DataBlock ref caches
            // left by the previous attach would otherwise suppress the refs this attach DataBlock needs.
            for (auto& gpuPartitionEntry : systemState.gpuPartitions)
            {
                gpuPartitionEntry.second.dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            }
            for (auto& processEntry : systemState.processes)
            {
                processEntry.second.dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            }
            if (systemState.cpuClockCalibrationProvider.IsInitialized())
            {
                PrepareProviderAttach(systemState.cpuClockCalibrationProvider);
            }
            for (auto& gpuPartitionEntry : systemState.gpuPartitions)
            {
                auto& gpuPartitionState = gpuPartitionEntry.second;
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                {
                    PrepareProviderAttach(gpuPartitionState.gpuClockCalibrationProvider);
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                {
                    PrepareProviderAttach(gpuPartitionState.gpuPmCountersProvider);
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                {
                    PrepareProviderAttach(gpuPartitionState.gpuHwTraceProvider);
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                {
                    PrepareProviderAttach(gpuPartitionState.gpuGrCtxTraceProvider);
                }
            }
            for (auto& processEntry : systemState.processes)
            {
                auto& processState = processEntry.second;
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    if (threadEventProviderEntry.second.IsInitialized())
                    {
                        PrepareProviderAttach(threadEventProviderEntry.second);
                    }
                }
                if (processState.cudaApiTraceProvider.IsInitialized())
                {
                    PrepareProviderAttach(processState.cudaApiTraceProvider);
                }
                if (processState.nvtxTraceProvider.IsInitialized())
                {
                    PrepareProviderAttach(processState.nvtxTraceProvider);
                }
            }
        }

        template <typename TSystemState>
        void AttachRouterMessages(TSystemState& systemState, bool includeZombies)
        {
            bool systemAttached = true;
            if (systemState.pSystemInfoMemory)
            {
                systemAttached = AttachSystem(systemState);
            }

            if (systemAttached && ShouldAttachProvider(systemState.cpuClockCalibrationProvider, includeZombies))
            {
                AttachProviderCreated_CpuClockCalibration(systemState.cpuClockCalibrationProvider);
            }

            for (auto& gpuPartitionEntry : systemState.gpuPartitions)
            {
                auto& gpuPartitionState = gpuPartitionEntry.second;
                if ((!includeZombies && gpuPartitionState.pendingDestroy) || !AttachGpuPartition(gpuPartitionState))
                {
                    continue;
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuClockCalibrationProvider, includeZombies))
                {
                    AttachProviderCreated_Gpu(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState, NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, "GPU Clock Calibration");
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuPmCountersProvider, includeZombies))
                {
                    AttachProviderCreated_Gpu(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState, NvTrace_GPU_PmCounters_ProviderTypeUuid, "GPU PM Counters");
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuHwTraceProvider, includeZombies))
                {
                    AttachProviderCreated_Gpu(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState, NvTrace_GPU_HwTrace_ProviderTypeUuid, "GPU HW Trace");
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuGrCtxTraceProvider, includeZombies))
                {
                    AttachProviderCreated_Gpu(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState, NvTrace_GPU_GrCtxTrace_ProviderTypeUuid, "GPU GR Context Trace");
                }
            }

            for (auto& processEntry : systemState.processes)
            {
                auto& processState = processEntry.second;
                if ((!includeZombies && processState.pendingDestroy) || !AttachProcess(processState))
                {
                    continue;
                }
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    auto& threadEventProviderState = threadEventProviderEntry.second;
                    if (ShouldAttachProvider(threadEventProviderState, includeZombies))
                    {
                        AttachProviderCreated_ThreadEvent(threadEventProviderState, processState);
                    }
                }
                if (ShouldAttachProvider(processState.cudaApiTraceProvider, includeZombies))
                {
                    AttachProviderCreated_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                }
                if (ShouldAttachProvider(processState.nvtxTraceProvider, includeZombies))
                {
                    AttachProviderCreated_NvtxTrace(processState.nvtxTraceProvider, processState);
                }
            }
        }

        template <typename TSystemState>
        void AttachStateMessages(TSystemState& systemState, bool includeZombies)
        {
            if (ShouldAttachProvider(systemState.cpuClockCalibrationProvider, includeZombies))
            {
                AttachStateMessages_CpuClockCalibration(systemState.cpuClockCalibrationProvider);
            }

            for (auto& gpuPartitionEntry : systemState.gpuPartitions)
            {
                auto& gpuPartitionState = gpuPartitionEntry.second;
                if (!includeZombies && gpuPartitionState.pendingDestroy)
                {
                    continue;
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuClockCalibrationProvider, includeZombies))
                {
                    AttachStateMessages_Gpu<Codec::ProviderTypeId::GpuClockCalibration>(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuPmCountersProvider, includeZombies))
                {
                    AttachStateMessages_Gpu<Codec::ProviderTypeId::GpuPmCounters>(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuHwTraceProvider, includeZombies))
                {
                    AttachStateMessages_Gpu<Codec::ProviderTypeId::GpuHwTrace>(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                }
                if (ShouldAttachProvider(gpuPartitionState.gpuGrCtxTraceProvider, includeZombies))
                {
                    AttachStateMessages_Gpu<Codec::ProviderTypeId::GpuGrCtxTrace>(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                }
            }

            for (auto& processEntry : systemState.processes)
            {
                auto& processState = processEntry.second;
                if (!includeZombies && processState.pendingDestroy)
                {
                    continue;
                }
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    auto& threadEventProviderState = threadEventProviderEntry.second;
                    if (ShouldAttachProvider(threadEventProviderState, includeZombies))
                    {
                        AttachStateMessages_ThreadEvent(threadEventProviderState, processState);
                    }
                }
                if (ShouldAttachProvider(processState.cudaApiTraceProvider, includeZombies))
                {
                    AttachStateMessages_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                }
                if (ShouldAttachProvider(processState.nvtxTraceProvider, includeZombies))
                {
                    AttachStateMessages_NvtxTrace(processState.nvtxTraceProvider, processState);
                }
            }
        }

        template <typename TSystemState>
        void AttachStateTables(TSystemState& systemState, bool includeZombies)
        {
            for (auto& processEntry : systemState.processes)
            {
                auto& processState = processEntry.second;
                if (!includeZombies && processState.pendingDestroy)
                {
                    continue;
                }
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    auto& threadEventProviderState = threadEventProviderEntry.second;
                    if (ShouldAttachProvider(threadEventProviderState, includeZombies))
                    {
                        AttachStateTables_ThreadEvent(threadEventProviderState, processState);
                    }
                }
                if (ShouldAttachProvider(processState.cudaApiTraceProvider, includeZombies))
                {
                    AttachStateTables_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                }
                if (ShouldAttachProvider(processState.nvtxTraceProvider, includeZombies))
                {
                    AttachStateTables_NvtxTrace(processState.nvtxTraceProvider, processState);
                }
            }
        }

        template <typename TSystemState>
        bool AttachSystem(TSystemState& systemState)
        {
            systemState.systemCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteSystemCreatedRecord(m_recordWriter, systemState.systemInfo))
            {
                return false;
            }
            m_recordWriter.Commit();
            systemState.systemCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            m_recordWriter.SetSystemInfoRecordLocation(systemState.systemCreatedRecordLocation);
            return true;
        }

        template <typename TGpuPartitionState>
        bool AttachGpuPartition(TGpuPartitionState& gpuPartitionState)
        {
            gpuPartitionState.gpuPartitionCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteGpuPartitionCreatedRecord(m_recordWriter, gpuPartitionState.luid, gpuPartitionState.gpuUuid, gpuPartitionState.creationTimestampTsc, gpuPartitionState.gpuPartitionInfo))
            {
                return false;
            }
            m_recordWriter.Commit();
            gpuPartitionState.gpuPartitionCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            return true;
        }

        template <typename TProcessState>
        bool AttachProcess(TProcessState& processState)
        {
            processState.processCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProcessCreatedRecord(m_recordWriter, processState.luid, processState.namespacePid, processState.globalPid, processState.pidNamespaceId, processState.path.c_str(), processState.creationTimestampTsc))
            {
                return false;
            }
            m_recordWriter.Commit();
            processState.processCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            return true;
        }

        template <Codec::ProviderTypeId TProviderTypeId, typename TProviderState>
        void ConsumeOnAttachRecords(TProviderState& providerState)
        {
            providerState.observer.ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(
                [&](uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (!Codec::EventEncoders::WriteRecord<TProviderTypeId>(m_recordWriter, recordType, NVTRACE_NO_CORRELATION_ID, pPayload, payloadSize))
                    {
                        return;
                    }
                    m_recordWriter.Commit();
                    providerState.onAttachRecordLocations.push_back(m_recordWriter.LastCommittedRecordLocation());
                });
        }

        template <typename TProviderState>
        void EmitProviderAttachedRecord(TProviderState& providerState)
        {
            if (!Codec::NvTraceServiceEncoders::WriteProviderAttachedRecord(m_recordWriter, providerState.luid, providerState.onAttachRecordLocations))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerAttachedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
        }

        template <typename TProviderState>
        void PrepareProviderAttach(TProviderState& providerState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.providerCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.providerAttachedRecordLocation = TSF_INVALID_RECORD_LOCATION;
            providerState.onAttachRecordLocations.clear();
            providerState.providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            providerState.dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            providerState.threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        }

        template <typename TProviderState>
        void AttachProviderCreated_CpuClockCalibration(TProviderState& providerState)
        {
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, NVTRACE_NIL_LUID, NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid, "CPU Clock Calibration"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            providerState.hasProviderCreatedRecord = true;
        }

        template <typename TProviderState, typename TGpuPartitionState>
        void AttachProviderCreated_Gpu(TProviderState& providerState, TGpuPartitionState& parentState, const NVTRACE_UUID& providerTypeUuid, const char* pProviderName)
        {
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, providerTypeUuid, pProviderName))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            providerState.hasProviderCreatedRecord = true;
        }

        template <typename TProviderState, typename TProcessState>
        void AttachProviderCreated_CudaApiTrace(TProviderState& providerState, TProcessState& parentState)
        {
            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_CUDA_ApiTrace_ProviderTypeUuid, "CUDA API Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            providerState.hasProviderCreatedRecord = true;
        }

        template <typename TProviderState, typename TProcessState>
        void AttachProviderCreated_NvtxTrace(TProviderState& providerState, TProcessState& parentState)
        {
            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_NvtxTrace_ProviderTypeUuid, "NVTX Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            providerState.hasProviderCreatedRecord = true;
        }

        template <typename TProviderState, typename TProcessState>
        void AttachProviderCreated_ThreadEvent(TProviderState& providerState, TProcessState& parentState)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_ThreadEvent_ProviderTypeUuid, "Thread Event"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.providerCreatedRecordLocation = m_recordWriter.LastCommittedRecordLocation();
            providerState.hasProviderCreatedRecord = true;
        }

        template <typename TProviderState>
        void AttachStateMessages_CpuClockCalibration(TProviderState& providerState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetSystemProviderRef(providerState);
            ConsumeOnAttachRecords<Codec::ProviderTypeId::CpuClockCalibration>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        template <Codec::ProviderTypeId TProviderTypeId, typename TProviderState, typename TGpuPartitionState>
        void AttachStateMessages_Gpu(TProviderState& providerState, TGpuPartitionState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            ConsumeOnAttachRecords<TProviderTypeId>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        template <typename TProviderState, typename TProcessState>
        void AttachStateMessages_CudaApiTrace(TProviderState& providerState, TProcessState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            CudaApiTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_recordWriter, [&] { providerState.onAttachRecordLocations.push_back(m_recordWriter.LastCommittedRecordLocation()); });
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        template <typename TProviderState, typename TProcessState>
        void AttachStateMessages_NvtxTrace(TProviderState& providerState, TProcessState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            NvtxTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_recordWriter, [&] { providerState.onAttachRecordLocations.push_back(m_recordWriter.LastCommittedRecordLocation()); });
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        template <typename TProviderState, typename TProcessState>
        void AttachStateMessages_ThreadEvent(TProviderState& providerState, TProcessState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_fnConsumeThreadEventSourceProviderRegistrations(parentState, providerState);
        }

        template <typename TProviderState, typename TProcessState>
        void AttachStateTables_CudaApiTrace(TProviderState& providerState, TProcessState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            CudaApiTraceSerialization::SerializeStateRecords(providerState.observer, m_recordWriter);
        }

        template <typename TProviderState, typename TProcessState>
        void AttachStateTables_NvtxTrace(TProviderState& providerState, TProcessState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            NvtxTraceSerialization::SerializeStateRecords(providerState.observer, m_recordWriter);
        }

        template <typename TProviderState, typename TProcessState>
        void AttachStateTables_ThreadEvent(TProviderState& providerState, TProcessState& parentState)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return;
            }
            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            ThreadEventSerialization::SerializeThreadStateEvents(providerState.observer, m_recordWriter);
        }

    private:
        TRecordWriter& m_recordWriter;
        TFN_ConsumeThreadEventSourceProviderRegistrations& m_fnConsumeThreadEventSourceProviderRegistrations;
    };

    template <typename TRecordWriter, typename TSystemState, typename TFN_ConsumeThreadEventSourceProviderRegistrations>
    void SerializeTsfAttachment(TRecordWriter& recordWriter, TSystemState& systemState, uint64_t attachTimestampUtcNsec, bool includeZombies, TFN_ConsumeThreadEventSourceProviderRegistrations&& fnConsumeThreadEventSourceProviderRegistrations)
    {
        TsfAttachmentSerializer<TRecordWriter, std::remove_reference_t<TFN_ConsumeThreadEventSourceProviderRegistrations>> serializer(recordWriter, fnConsumeThreadEventSourceProviderRegistrations);
        serializer.Serialize(systemState, attachTimestampUtcNsec, includeZombies);
    }

} // namespace nv::trace

namespace nv::trace {

    struct ContinuousSnapshotExportResult
    {
        uint64_t snapshotNumber = 0;
        uint64_t maxConsumeDataDurationNsec = 0;
        uint64_t attachDurationNsec = 0;
        uint64_t exportDurationNsec = 0;
        uint64_t totalWorkerDurationNsec = 0;
        bool exceededSnapshotCadence = false;
        bool storageSucceeded = false;
        bool attachComplete = false;
        bool exportSucceeded = false;
    };

    namespace Detail {

        template <typename TProviderState, typename TGpuPartitionState, typename TProcessState, typename TThreadEventProviderState>
        class ContinuousSnapshotAttachDataBlockWriter
        {
        public:
            using RecordWriterBase = TsfSnapshotRecordWriter<TProviderState, TGpuPartitionState, TProcessState, TThreadEventProviderState, true>;

            class RecordWriter : public RecordWriterBase
            {
            public:
                void SetFailureSink(ContinuousSnapshotAttachDataBlockWriter* pFailureSink)
                {
                    m_pFailureSink = pFailureSink;
                }

                template <typename TFN_WritePayload>
                bool WriteRecord(uint32_t recordType, uint64_t correlationId, size_t majorPayloadSize, TFN_WritePayload&& fnWritePayload)
                {
                    if (RecordWriterBase::WriteRecord(recordType, correlationId, majorPayloadSize, std::forward<TFN_WritePayload>(fnWritePayload)))
                    {
                        return true;
                    }
                    if (m_pFailureSink)
                    {
                        m_pFailureSink->RecordWriteFailed(majorPayloadSize);
                    }
                    return false;
                }

            private:
                ContinuousSnapshotAttachDataBlockWriter* m_pFailureSink = nullptr;
            };

            ContinuousSnapshotAttachDataBlockWriter()
            {
                m_recordWriter.SetFailureSink(this);
            }

            void Reset() noexcept
            {
                m_recordWriter.Reset();
                m_dataBlockWriter.Reset();
                for (File& file : m_files)
                {
                    if (file.stream.is_open())
                    {
                        file.stream.close();
                    }
                }
                m_files.clear();
                m_pDataBlock.reset();
                m_dataBlockHeader = {};
                m_recordLocationEncoder.Reset();
                m_dataBlockSize = 0;
                m_maxMajorPayloadSize = 0;
                m_chunkCapacityBytes = 0;
                m_nextDataBlockIndex = 0;
                m_currentDataBlockIndex = SIZE_MAX;
                m_stagedDataBlockFiles.clear();
                m_overflowed = false;
                m_capacityExceeded = false;
                m_recordWriteFailed = false;
                m_droppedRecordCount = 0;
                m_droppedBytes = 0;
                m_requiredAdditionalDataBlockCount = 0;
                m_requiredAdditionalChunkBytes = 0;
                m_minimumRequiredDataBlockSizeBytes = 0;
            }

            bool Initialize(
                size_t dataBlockSize,
                const NvTrace_RouterObserverClient_RouterConfig& routerConfig,
                const std::vector<std::filesystem::path>& filePaths)
            {
                Reset();
                NVTRACE_LOG_ERR_RETURN_IF(filePaths.empty(), false, 20, "Continuous snapshot attachment requires at least one preallocated DataBlock file");

                SectionLayoutBuilder layout;
                layout.AppendSection<TsfDataBlockHeader>();

                m_dataBlockHeader.version.magic = TRACE_STREAM_FORMAT_MAGIC;
                m_dataBlockHeader.version.majorVersion = TRACE_STREAM_FORMAT_MAJOR_VERSION;
                m_dataBlockHeader.version.sizeofDataBlockHeader = sizeof(TsfDataBlockHeader);
                m_dataBlockHeader.sizeofDataBlock = dataBlockSize;
                m_dataBlockHeader.offsetofDataBlockState = layout.AppendSection<TsfDataBlockState>();
                m_dataBlockHeader.sizeofDataBlockState = sizeof(TsfDataBlockState);

                const uint32_t maxGpuPartitionCount = routerConfig.maxGpuPartitionCount * 5 / 4;
                const uint32_t maxProcessCount = routerConfig.maxProcessCount * 5 / 4;
                const uint32_t maxProviderCount = routerConfig.maxProviderCount * 5 / 4;
                m_dataBlockHeader.offsetofGpuPartitionRefs = layout.AppendSection<TsfGpuPartitionRef>(maxGpuPartitionCount);
                m_dataBlockHeader.sizeofGpuPartitionRef = sizeof(TsfGpuPartitionRef);
                m_dataBlockHeader.maxGpuPartitionRefCount = maxGpuPartitionCount;
                m_dataBlockHeader.offsetofProcessRefs = layout.AppendSection<TsfProcessRef>(maxProcessCount);
                m_dataBlockHeader.sizeofProcessRef = sizeof(TsfProcessRef);
                m_dataBlockHeader.maxProcessRefCount = maxProcessCount;
                m_dataBlockHeader.offsetofProviderRefs = layout.AppendSection<TsfProviderRef>(maxProviderCount);
                m_dataBlockHeader.sizeofProviderRef = sizeof(TsfProviderRef);
                m_dataBlockHeader.maxProviderRefCount = maxProviderCount;
                m_dataBlockHeader.offsetofThreadEventProviderMappingRefs = layout.AppendSection<TsfThreadEventProviderMappingRef>(maxProviderCount);
                m_dataBlockHeader.sizeofThreadEventProviderMappingRef = sizeof(TsfThreadEventProviderMappingRef);
                m_dataBlockHeader.maxThreadEventProviderMappingRefCount = maxProviderCount;
                m_dataBlockHeader.offsetofChunkStates = layout.AppendSection<TsfChunkState>();
                m_dataBlockHeader.sizeofChunkState = sizeof(TsfChunkState);

                constexpr size_t minChunkSize = 1024;
                constexpr size_t maxChunkSize = (uint32_t(1) << NVTRACE_FIELD_RANGE_SIZE(NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8)) * sizeof(uint64_t);
                const size_t remainingDataBlockSize = layout.GetEndOffset() < dataBlockSize ? dataBlockSize - layout.GetEndOffset() : 0;
                const size_t chunkSize = AlignDnP2(std::clamp(remainingDataBlockSize, minChunkSize, maxChunkSize), sizeof(uint64_t));
                m_dataBlockHeader.offsetofChunks = layout.AppendSection(chunkSize, sizeof(uint64_t));
                m_dataBlockHeader.sizeofChunk = static_cast<uint32_t>(chunkSize);
                m_dataBlockHeader.maxChunkCount = 1;
                m_dataBlockHeader.sizeofRecord = sizeof(TsfRecord);
                m_dataBlockHeader.sizeofInlinePayload = TSF_RECORD_INLINE_PAYLOAD_SIZE;
                m_dataBlockHeader.maxRecordCountPerChunk = m_dataBlockHeader.sizeofChunk / sizeof(TsfRecordWithInlinePayload);

                NVTRACE_LOG_ERR_RETURN_IF(dataBlockSize < layout.GetEndOffset(), false, 20, "Attach DataBlock size (%zu) is too small; at least %zu bytes are required", dataBlockSize, layout.GetEndOffset());
                NVTRACE_LOG_ERR_RETURN_IF(dataBlockSize > std::numeric_limits<uint32_t>::max(), false, 20, "Attach DataBlock size exceeds the TSF format limit");
                m_dataBlockSize = dataBlockSize;
                m_maxMajorPayloadSize = AlignDnP2(chunkSize - sizeof(TsfRecordWithInlinePayload), sizeof(uint64_t));
                m_chunkCapacityBytes = chunkSize;
                NVTRACE_LOG_ERR_RETURN_IF(!m_recordLocationEncoder.Initialize(1, m_dataBlockHeader.maxRecordCountPerChunk), false, 20, "Failed to initialize attach record location encoder");

                try
                {
                    m_pDataBlock = std::make_unique<uint8_t[]>(dataBlockSize);
                }
                catch (const std::exception& exception)
                {
                    NVTRACE_LOG_ERR(20, "Failed to allocate attach DataBlock: %s", exception.what());
                    return false;
                }
                m_files.reserve(filePaths.size());
                for (const std::filesystem::path& filePath : filePaths)
                {
                    m_files.emplace_back();
                    File& file = m_files.back();
                    file.path = filePath;
                    file.stream.open(file.path, std::ios::binary | std::ios::in | std::ios::out);
                    NVTRACE_LOG_ERR_RETURN_IF(file.stream.fail(), false, 20, "Failed to open preallocated attach DataBlock at %s", file.path.string().c_str());
                }
                m_recordWriter.Initialize(
                    m_maxMajorPayloadSize,
                    &ContinuousSnapshotAttachDataBlockWriter::EncodeRecordLocation,
                    &ContinuousSnapshotAttachDataBlockWriter::StageDataBlock,
                    &ContinuousSnapshotAttachDataBlockWriter::ObtainDataBlock,
                    this);
                return true;
            }

            void Begin(uint64_t timestampUtcNsec)
            {
                m_recordWriter.Reset();
                m_dataBlockWriter.Reset();
                m_nextDataBlockIndex = 0;
                m_currentDataBlockIndex = SIZE_MAX;
                m_stagedDataBlockFiles.clear();
                m_overflowed = false;
                m_capacityExceeded = false;
                m_recordWriteFailed = false;
                m_droppedRecordCount = 0;
                m_droppedBytes = 0;
                m_requiredAdditionalDataBlockCount = 0;
                m_requiredAdditionalChunkBytes = 0;
                m_minimumRequiredDataBlockSizeBytes = 0;
                m_recordWriter.SetLastConsumeTimestampUtcNsec(timestampUtcNsec);
            }

            RecordWriter& GetRecordWriter()
            {
                return m_recordWriter;
            }

            bool Finish()
            {
                const bool checkpointSucceeded = m_recordWriter.Checkpoint();
                if (!checkpointSucceeded || m_overflowed || m_recordWriteFailed || m_stagedDataBlockFiles.empty())
                {
                    return false;
                }
                return true;
            }

            const std::vector<ContinuousSnapshotDataBlockFile>& GetStagedDataBlockFiles() const
            {
                return m_stagedDataBlockFiles;
            }

            uint64_t DroppedRecordCount() const
            {
                return m_droppedRecordCount;
            }

            uint64_t DroppedBytes() const
            {
                return m_droppedBytes;
            }

            uint64_t MinimumRequiredAttachSizeBytes() const
            {
                if (!m_capacityExceeded || !m_dataBlockSize || !m_requiredAdditionalDataBlockCount)
                {
                    return 0;
                }

                if (std::numeric_limits<uint64_t>::max() - m_files.size() < m_requiredAdditionalDataBlockCount)
                {
                    return std::numeric_limits<uint64_t>::max();
                }
                const uint64_t requiredDataBlockCount = m_files.size() + m_requiredAdditionalDataBlockCount;
                if (std::numeric_limits<uint64_t>::max() / m_dataBlockSize < requiredDataBlockCount)
                {
                    return std::numeric_limits<uint64_t>::max();
                }
                return requiredDataBlockCount * m_dataBlockSize;
            }

            uint64_t MinimumRequiredDataBlockSizeBytes() const
            {
                return m_minimumRequiredDataBlockSizeBytes;
            }

        private:
            void RecordWriteFailed(size_t majorPayloadSize)
            {
                m_recordWriteFailed = true;
                if (m_droppedRecordCount != std::numeric_limits<uint64_t>::max())
                {
                    m_droppedRecordCount += 1;
                }

                const uint64_t alignedPayloadSize = majorPayloadSize <= std::numeric_limits<uint64_t>::max() - (sizeof(uint64_t) - 1)
                                                      ? AlignUpP2(static_cast<uint64_t>(majorPayloadSize), sizeof(uint64_t))
                                                      : std::numeric_limits<uint64_t>::max();
                const uint64_t recordSize = alignedPayloadSize <= std::numeric_limits<uint64_t>::max() - sizeof(TsfRecordWithInlinePayload)
                                              ? sizeof(TsfRecordWithInlinePayload) + alignedPayloadSize
                                              : std::numeric_limits<uint64_t>::max();
                m_droppedBytes = recordSize <= std::numeric_limits<uint64_t>::max() - m_droppedBytes
                                   ? m_droppedBytes + recordSize
                                   : std::numeric_limits<uint64_t>::max();

                if (m_capacityExceeded && recordSize <= m_chunkCapacityBytes)
                {
                    if (!m_requiredAdditionalDataBlockCount || m_chunkCapacityBytes - m_requiredAdditionalChunkBytes < recordSize)
                    {
                        m_requiredAdditionalDataBlockCount += 1;
                        m_requiredAdditionalChunkBytes = 0;
                    }
                    m_requiredAdditionalChunkBytes += recordSize;
                }
                else if (m_maxMajorPayloadSize < majorPayloadSize)
                {
                    const uint64_t minimumDataBlockSize = recordSize <= std::numeric_limits<uint64_t>::max() - m_dataBlockHeader.offsetofChunks
                                                            ? m_dataBlockHeader.offsetofChunks + recordSize
                                                            : std::numeric_limits<uint64_t>::max();
                    m_minimumRequiredDataBlockSizeBytes = std::max(m_minimumRequiredDataBlockSizeBytes, minimumDataBlockSize);
                }
            }

            static uint64_t EncodeRecordLocation(void* pUserData, uint64_t dataBlockSequenceNumber, uint32_t chunkIndex, uint32_t recordIndex)
            {
                auto& writer = *static_cast<ContinuousSnapshotAttachDataBlockWriter*>(pUserData);
                return writer.m_recordLocationEncoder.EncodeRecordLocation(dataBlockSequenceNumber, chunkIndex, recordIndex);
            }

            static bool StageDataBlock(void* pUserData, TraceStreamDataBlockWriter&& dataBlockWriter)
            {
                auto& writer = *static_cast<ContinuousSnapshotAttachDataBlockWriter*>(pUserData);
                if (!dataBlockWriter.IsInitialized())
                {
                    return true;
                }
                if (writer.m_files.size() <= writer.m_currentDataBlockIndex)
                {
                    writer.m_overflowed = true;
                    return false;
                }

                File& file = writer.m_files[writer.m_currentDataBlockIndex];
                file.stream.clear();
                file.stream.seekp(0);
                file.stream.write(reinterpret_cast<const char*>(writer.m_pDataBlock.get()), static_cast<std::streamsize>(writer.m_dataBlockHeader.sizeofDataBlock));
                if (file.stream.fail())
                {
                    NVTRACE_LOG_ERR(20, "Failed to write attach DataBlock at %s", file.path.string().c_str());
                    writer.m_overflowed = true;
                    return false;
                }
                file.stream.flush();
                if (file.stream.fail())
                {
                    NVTRACE_LOG_ERR(20, "Failed to flush attach DataBlock at %s", file.path.string().c_str());
                    writer.m_overflowed = true;
                    return false;
                }
                writer.m_stagedDataBlockFiles.push_back({file.path, dataBlockWriter.DataBlockSequenceNumber()});
                writer.m_currentDataBlockIndex = SIZE_MAX;
                dataBlockWriter.Reset();
                return true;
            }

            static bool ObtainDataBlock(void* pUserData, TraceStreamDataBlockWriter& dataBlockWriter)
            {
                auto& writer = *static_cast<ContinuousSnapshotAttachDataBlockWriter*>(pUserData);
                if (writer.m_currentDataBlockIndex != SIZE_MAX)
                {
                    writer.m_overflowed = true;
                    return false;
                }
                if (writer.m_files.size() <= writer.m_nextDataBlockIndex)
                {
                    writer.m_overflowed = true;
                    writer.m_capacityExceeded = true;
                    return false;
                }
                const size_t dataBlockIndex = writer.m_nextDataBlockIndex++;
                const uint64_t dataBlockSequenceNumber = TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW + dataBlockIndex;
                std::memcpy(writer.m_pDataBlock.get(), &writer.m_dataBlockHeader, sizeof(TsfDataBlockHeader));
                dataBlockWriter.Initialize(
                    0,
                    dataBlockSequenceNumber,
                    writer.m_pDataBlock.get(),
                    0,
                    0);
                writer.m_currentDataBlockIndex = dataBlockIndex;
                return true;
            }

        private:
            struct File
            {
                std::filesystem::path path;
                std::fstream stream;
            };

            TsfDataBlockHeader m_dataBlockHeader = {};
            TraceStreamRecordLocationEncoder m_recordLocationEncoder;
            std::unique_ptr<uint8_t[]> m_pDataBlock;
            TraceStreamDataBlockWriter m_dataBlockWriter;
            RecordWriter m_recordWriter;
            std::vector<File> m_files;
            std::vector<ContinuousSnapshotDataBlockFile> m_stagedDataBlockFiles;
            size_t m_dataBlockSize = 0;
            size_t m_maxMajorPayloadSize = 0;
            size_t m_chunkCapacityBytes = 0;
            size_t m_nextDataBlockIndex = 0;
            size_t m_currentDataBlockIndex = SIZE_MAX;
            bool m_overflowed = false;
            bool m_capacityExceeded = false;
            bool m_recordWriteFailed = false;
            uint64_t m_droppedRecordCount = 0;
            uint64_t m_droppedBytes = 0;
            uint64_t m_requiredAdditionalDataBlockCount = 0;
            uint64_t m_requiredAdditionalChunkBytes = 0;
            uint64_t m_minimumRequiredDataBlockSizeBytes = 0;
        };

        class ContinuousSnapshotAttachWorker : protected ConsumerBase<ContinuousSnapshotAttachWorker, ContinuousSnapshotExporterObjectModel>
        {
            using Super = ConsumerBase<ContinuousSnapshotAttachWorker, ContinuousSnapshotExporterObjectModel>;
            friend Super;

            using typename Super::GpuPartitionState;
            using typename Super::ProcessState;
            using typename Super::ProviderState;
            using typename Super::SystemState;
            using typename Super::ThreadEventProviderState;

            using ZombieParentType = ContinuousSnapshotExporterObjectModel::ZombieParentType;
            using ZombieParentQueue = ContinuousSnapshotExporterObjectModel::ZombieParentQueue;
            using PendingProviderDestroy = ContinuousSnapshotExporterObjectModel::PendingProviderDestroy;
            using PendingProcessDestroy = ContinuousSnapshotExporterObjectModel::PendingProcessDestroy;
            using PendingGpuPartitionDestroy = ContinuousSnapshotExporterObjectModel::PendingGpuPartitionDestroy;
            using AttachWriter = ContinuousSnapshotAttachDataBlockWriter<ProviderState, GpuPartitionState, ProcessState, ThreadEventProviderState>;

        public:
            using FnExportSnapshot = std::function<bool(const ContinuousSnapshotManifest& manifest)>;
            using FnOnExportCompleted = std::function<void(const ContinuousSnapshotManifest& manifest, const ContinuousSnapshotExportResult& result)>;
            using FnTryTakeExportBank = bool (*)(void* pUserData, size_t& out_bankIndex);
            using FnReturnExportBank = void (*)(void* pUserData, size_t bankIndex);

            ~ContinuousSnapshotAttachWorker()
            {
                Reset();
            }

            ContinuousSnapshotAttachWorker() = default;
            ContinuousSnapshotAttachWorker(const ContinuousSnapshotAttachWorker&) = delete;
            ContinuousSnapshotAttachWorker& operator=(const ContinuousSnapshotAttachWorker&) = delete;

            void Reset() noexcept
            {
                Terminate();
                m_waitMultiplexer.Reset();
                m_jobReadyEvent.Reset();
                m_attachWriter.Reset();
                m_zombieParentQueue.clear();
                m_zombieObjectBytes = 0;
                m_maxZombieObjectBytes = 0;
                m_historyDurationNsec = 0;
                m_fnExportSnapshot = {};
                m_fnOnExportCompleted = {};
                m_fnTryTakeExportBank = nullptr;
                m_fnReturnExportBank = nullptr;
                m_pStorageState = nullptr;
                m_pUserData = nullptr;
                m_terminateRequested.store(false, std::memory_order_relaxed);
                m_threadExited.store(true, std::memory_order_relaxed);
                Super::Reset();
            }

            template <typename TConsumerOptions>
            bool Initialize(
                const TConsumerOptions& options,
                size_t dataBlockSize,
                uint64_t historyDurationNsec,
                const std::vector<std::filesystem::path>& attachFilePaths,
                ContinuousSnapshotStorageState* pStorageState,
                const FnExportSnapshot& fnExportSnapshot,
                const FnOnExportCompleted& fnOnExportCompleted,
                FnTryTakeExportBank fnTryTakeExportBank,
                FnReturnExportBank fnReturnExportBank,
                void* pUserData)
            {
                typename Super::Options consumerOptions;
                consumerOptions.pGpuPartitionFilter = options.pGpuPartitionFilter;
                consumerOptions.pProcessFilter = options.pProcessFilter;
                consumerOptions.isProviderTypeIgnored = options.isProviderTypeIgnored;
                consumerOptions.consumeIntervalMsec = options.consumeIntervalMsec;
                consumerOptions.threadEventHoldTimeoutNsec = options.threadEventHoldTimeoutNsec;

                NvTrace_RouterObserverClient_RouterConfig routerConfig = {};
                NVTRACE_LOG_ERR_RETURN_IF(!Super::Initialize(consumerOptions, &routerConfig), false, 50, "Failed to initialize continuous snapshot attach object model");
                auto resetGuard = ScopeExitGuard([this]() { Reset(); });

                m_historyDurationNsec = historyDurationNsec;
                m_maxZombieObjectBytes = options.maxZombieObjectBytes;
                try
                {
                    m_fnExportSnapshot = fnExportSnapshot;
                    m_fnOnExportCompleted = fnOnExportCompleted;
                }
                catch (const std::exception& exception)
                {
                    NVTRACE_LOG_ERR(50, "Failed to copy continuous snapshot callbacks: %s", exception.what());
                    return false;
                }
                m_fnTryTakeExportBank = fnTryTakeExportBank;
                m_fnReturnExportBank = fnReturnExportBank;
                m_pStorageState = pStorageState;
                m_pUserData = pUserData;
                NVTRACE_LOG_ERR_RETURN_IF(!m_fnTryTakeExportBank || !m_fnReturnExportBank || !m_pStorageState || !m_pUserData, false, 50, "Continuous snapshot export handoff callbacks are invalid");
                NVTRACE_LOG_ERR_RETURN_IF(!m_attachWriter.Initialize(dataBlockSize, routerConfig, attachFilePaths), false, 50, "Failed to initialize continuous snapshot attach writer");
                NVTRACE_LOG_ERR_RETURN_IF(!m_jobReadyEvent.Initialize(), false, 50, "Failed to initialize continuous snapshot job event");
                NVTRACE_LOG_ERR_RETURN_IF(!m_waitMultiplexer.Initialize(2), false, 50, "Failed to initialize continuous snapshot worker wait multiplexer");
                NVTRACE_LOG_ERR_RETURN_IF(!m_waitMultiplexer.AddObject(EventIdentifier::JobReady, m_jobReadyEvent.Desc()), false, 50, "Failed to add continuous snapshot job event");
                NVTRACE_LOG_ERR_RETURN_IF(!m_waitMultiplexer.AddObject(EventIdentifier::RouterNotification, m_pRouterObserverClient->GetWaitForPendingNotificationHandle()), false, 50, "Failed to add continuous snapshot Router event");

                m_threadExited.store(false, std::memory_order_relaxed);
                try
                {
                    m_thread = std::thread(&ContinuousSnapshotAttachWorker::ThreadMain, this);
                }
                catch (const std::system_error& exception)
                {
                    m_threadExited.store(true, std::memory_order_relaxed);
                    NVTRACE_LOG_ERR(50, "Failed to create continuous snapshot export thread: %s", exception.what());
                    return false;
                }
                resetGuard.Dismiss();
                return true;
            }

            bool SignalJobReady()
            {
                if (m_threadExited.load(std::memory_order_acquire))
                {
                    return false;
                }
                return m_jobReadyEvent.Signal();
            }

            bool IsRunning() const
            {
                return !m_threadExited.load(std::memory_order_acquire);
            }

        private:
            enum class EventIdentifier
            {
                JobReady,
                RouterNotification,
            };

            void ThreadMain() noexcept
            {
                auto exitGuard = ScopeExitGuard([this]() {
                    m_threadExited.store(true, std::memory_order_release);

                    // Never leave a bank owned by a worker that can no longer service it.
                    // The active job's release guard runs before this guard; this loop returns
                    // any jobs that were still queued when the worker stopped.
                    size_t bankIndex = ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
                    while (m_fnTryTakeExportBank(m_pUserData, bankIndex))
                    {
                        m_fnReturnExportBank(m_pUserData, bankIndex);
                    }
                });
                try
                {
                    while (!m_terminateRequested.load(std::memory_order_acquire))
                    {
                        EventIdentifier identifier;
                        if (!m_waitMultiplexer.Wait(identifier))
                        {
                            return;
                        }
                        if (m_terminateRequested.load(std::memory_order_acquire))
                        {
                            return;
                        }

                        if (identifier == EventIdentifier::RouterNotification)
                        {
                            const bool succeeded = m_pRouterObserverClient->ReceiveNotificationIfPending([this](NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) {
                                return HandleRouterNotification(notificationType, pNotification);
                            });
                            if (!succeeded)
                            {
                                NVTRACE_LOG_WRN(50, "Continuous snapshot attach worker lost its Router observer connection");
                                if (!EnterRouterlessMode())
                                {
                                    NVTRACE_LOG_ERR(50, "Continuous snapshot attach worker failed to continue without Router notifications");
                                    return;
                                }
                                NVTRACE_LOG_WRN(50, "Continuous snapshot attach worker is continuing in routerless mode with its known object model");
                                continue;
                            }
                            PurgeExpiredZombieObjects(CurrentTimestampUtcNsec());
                        }
                        else
                        {
                            size_t bankIndex = ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
                            while (m_fnTryTakeExportBank(m_pUserData, bankIndex))
                            {
                                auto releaseBankGuard = ScopeExitGuard([this, bankIndex]() {
                                    m_fnReturnExportBank(m_pUserData, bankIndex);
                                });
                                ContinuousSnapshotStorageState::ExportJob job;
                                if (m_pStorageState->BuildExportJob(bankIndex, job))
                                {
                                    ProcessExportJob(job);
                                }
                            }
                        }
                    }
                }
                catch (const std::exception& exception)
                {
                    NVTRACE_LOG_ERR(50, "Continuous snapshot export thread terminated after an exception: %s", exception.what());
                }
                catch (...)
                {
                    NVTRACE_LOG_ERR(50, "Continuous snapshot export thread terminated after an unknown exception");
                }
            }

            void ProcessExportJob(ContinuousSnapshotStorageState::ExportJob& job)
            {
                const uint64_t workerBeginTsc = ReadCpuTscTimestamp();
                PurgeExpiredZombieObjects(job.manifest.cutoffTimestampUtcNsec);

                const uint64_t attachBeginTsc = ReadCpuTscTimestamp();
                m_attachWriter.Begin(job.manifest.cutoffTimestampUtcNsec);
                const auto fnConsumeThreadEventSourceProviderRegistrations = [this](ProcessState& parentState, ThreadEventProviderState& providerState) {
                    Super::ConsumeThreadEventSourceProviderRegistrations(parentState, providerState, ReadCpuTscTimestamp());
                };
                SerializeTsfAttachment(
                    m_attachWriter.GetRecordWriter(),
                    m_systemState,
                    job.manifest.cutoffTimestampUtcNsec,
                    true,
                    fnConsumeThreadEventSourceProviderRegistrations);
                const bool attachComplete = m_attachWriter.Finish();
                const uint64_t attachEndTsc = ReadCpuTscTimestamp();
                job.manifest.attachComplete = attachComplete;
                job.manifest.droppedAttachRecordCount = m_attachWriter.DroppedRecordCount();
                job.manifest.droppedAttachBytes = m_attachWriter.DroppedBytes();
                job.manifest.minimumRequiredAttachSizeBytes = m_attachWriter.MinimumRequiredAttachSizeBytes();
                job.manifest.minimumRequiredDataBlockSizeBytes = m_attachWriter.MinimumRequiredDataBlockSizeBytes();

                const auto& attachDataBlockFiles = m_attachWriter.GetStagedDataBlockFiles();
                job.manifest.dataBlockFiles.insert(job.manifest.dataBlockFiles.end(), attachDataBlockFiles.begin(), attachDataBlockFiles.end());
                std::sort(job.manifest.dataBlockFiles.begin(), job.manifest.dataBlockFiles.end(), [](const ContinuousSnapshotDataBlockFile& lhs, const ContinuousSnapshotDataBlockFile& rhs) {
                    return lhs.dataBlockSequenceNumber < rhs.dataBlockSequenceNumber;
                });

                bool exportSucceeded = false;
                const uint64_t exportBeginTsc = ReadCpuTscTimestamp();
                if (job.storageSucceeded && m_fnExportSnapshot)
                {
                    try
                    {
                        exportSucceeded = m_fnExportSnapshot(job.manifest);
                    }
                    catch (const std::exception& exception)
                    {
                        NVTRACE_LOG_ERR(50, "Continuous snapshot export callback threw: %s", exception.what());
                    }
                    catch (...)
                    {
                        NVTRACE_LOG_ERR(50, "Continuous snapshot export callback threw an unknown exception");
                    }
                }
                const uint64_t exportEndTsc = ReadCpuTscTimestamp();

                ContinuousSnapshotExportResult result;
                result.snapshotNumber = job.manifest.snapshotNumber;
                result.maxConsumeDataDurationNsec = job.maxConsumeDataDurationNsec;
                const TscRtcCalibrationPoint calibrationPoint = this->GetLatestCpuClockCalibrationPoint();
                result.attachDurationNsec = static_cast<uint64_t>(static_cast<double>(attachEndTsc - attachBeginTsc) * calibrationPoint.tscToUtcFactor);
                result.exportDurationNsec = static_cast<uint64_t>(static_cast<double>(exportEndTsc - exportBeginTsc) * calibrationPoint.tscToUtcFactor);
                result.totalWorkerDurationNsec = static_cast<uint64_t>(static_cast<double>(exportEndTsc - workerBeginTsc) * calibrationPoint.tscToUtcFactor);
                result.exceededSnapshotCadence = result.totalWorkerDurationNsec > m_historyDurationNsec;
                result.storageSucceeded = job.storageSucceeded;
                result.attachComplete = attachComplete;
                result.exportSucceeded = exportSucceeded;

                if (m_fnOnExportCompleted)
                {
                    try
                    {
                        m_fnOnExportCompleted(job.manifest, result);
                    }
                    catch (const std::exception& exception)
                    {
                        NVTRACE_LOG_ERR(50, "Continuous snapshot completion callback threw: %s", exception.what());
                    }
                    catch (...)
                    {
                        NVTRACE_LOG_ERR(50, "Continuous snapshot completion callback threw an unknown exception");
                    }
                }
            }

            void Terminate() noexcept
            {
                if (!m_thread.joinable())
                {
                    return;
                }
                m_terminateRequested.store(true, std::memory_order_release);
                const bool succeeded = m_jobReadyEvent.Signal();
                NVTRACE_LOG_ERR_IF(!succeeded, 50, "Failed to signal continuous snapshot export thread termination");
                m_thread.join();
            }

            bool EnterRouterlessMode()
            {
                const bool succeeded = m_waitMultiplexer.RemoveLastSignaledObject();
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Continuous snapshot attach worker failed to remove its Router wait object");
                m_pRouterObserverClient.Reset();
                return true;
            }

            uint64_t CurrentTimestampUtcNsec() const
            {
                return Super::ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
            }

            bool HandleRouterNotification(NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification)
            {
                switch (notificationType)
                {
                    case NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed:
                        return DelayGpuPartitionDestroyedNotification(*reinterpret_cast<const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification*>(pNotification));
                    case NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed:
                        return DelayProcessDestroyedNotification(*reinterpret_cast<const NvTrace_RouterObserverClient_ProcessDestroyedNotification*>(pNotification));
                    case NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed:
                    {
                        const auto& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProviderDestroyedNotification*>(pNotification);
                        if (ResolveProviderTypeUuidAlias(notification.providerTypeUuid) == NvTrace_ThreadEvent_ProviderTypeUuid)
                        {
                            return Super::HandleRouterNotification(notificationType, pNotification);
                        }
                        if (m_systemState.processes.find(notification.parentLuid) != m_systemState.processes.end()
                            || m_systemState.gpuPartitions.find(notification.parentLuid) != m_systemState.gpuPartitions.end())
                        {
                            return DelayProviderDestroyedNotification(notification);
                        }
                        return Super::HandleRouterNotification(notificationType, pNotification);
                    }
                    default:
                        return Super::HandleRouterNotification(notificationType, pNotification);
                }
            }

            bool DelayProviderDestroyedNotification(const NvTrace_RouterObserverClient_ProviderDestroyedNotification& notification)
            {
                ProviderState* const pProviderState = m_systemState.GetProviderState(notification.providerLuid);
                NVTRACE_LOG_ERR_RETURN_IF(!pProviderState || pProviderState->pendingDestroy, false, 50, "Invalid duplicate or unknown ProviderDestroyed notification in continuous snapshot attach worker");
                PendingProviderDestroy& pendingDestroy = pProviderState->pendingDestroy;
                pendingDestroy.notification = notification;
                pendingDestroy.isPending = true;
                return true;
            }

            bool DelayProcessDestroyedNotification(const NvTrace_RouterObserverClient_ProcessDestroyedNotification& notification)
            {
                auto itr = m_systemState.processes.find(notification.processLuid);
                NVTRACE_LOG_ERR_RETURN_IF(itr == m_systemState.processes.end() || itr->second.pendingDestroy, false, 50, "Invalid duplicate or unknown ProcessDestroyed notification in continuous snapshot attach worker");
                PendingProcessDestroy& pendingDestroy = itr->second.pendingDestroy;
                pendingDestroy.notification = notification;
                pendingDestroy.isPending = true;
                return QueueZombieParent(ZombieParentType::Process, &itr->second, GetProcessRetainedBytes(itr->second));
            }

            bool DelayGpuPartitionDestroyedNotification(const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification& notification)
            {
                auto itr = m_systemState.gpuPartitions.find(notification.gpuPartitionLuid);
                NVTRACE_LOG_ERR_RETURN_IF(itr == m_systemState.gpuPartitions.end() || itr->second.pendingDestroy, false, 50, "Invalid duplicate or unknown GpuPartitionDestroyed notification in continuous snapshot attach worker");
                PendingGpuPartitionDestroy& pendingDestroy = itr->second.pendingDestroy;
                pendingDestroy.notification = notification;
                pendingDestroy.isPending = true;
                return QueueZombieParent(ZombieParentType::GpuPartition, &itr->second, GetGpuPartitionRetainedBytes(itr->second));
            }

            bool QueueZombieParent(ZombieParentType type, void* pState, size_t retainedBytes)
            {
                if (m_maxZombieObjectBytes == 0 || retainedBytes > m_maxZombieObjectBytes)
                {
                    NVTRACE_LOG_WRN(50, "Forwarding a continuous snapshot zombie immediately because it exceeds maxZombieObjectBytes=%zu", m_maxZombieObjectBytes);
                    return ReplayZombieParent({type, pState, 0, retainedBytes});
                }

                while (m_zombieObjectBytes > m_maxZombieObjectBytes - retainedBytes && !m_zombieParentQueue.empty())
                {
                    NVTRACE_LOG_WRN(50, "Evicting a continuous snapshot zombie before its time window expires to satisfy maxZombieObjectBytes=%zu", m_maxZombieObjectBytes);
                    if (!ReplayOldestZombieParent())
                    {
                        ReplayZombieParent({type, pState, 0, retainedBytes});
                        return false;
                    }
                }

                ContinuousSnapshotExporterObjectModel::ZombieParentQueueEntry entry;
                entry.type = type;
                entry.pState = pState;
                entry.retainedBytes = retainedBytes;
                const uint64_t nowUtcNsec = CurrentTimestampUtcNsec();
                entry.expirationTimestampUtcNsec = nowUtcNsec > std::numeric_limits<uint64_t>::max() - m_historyDurationNsec ? std::numeric_limits<uint64_t>::max() : nowUtcNsec + m_historyDurationNsec;
                m_zombieParentQueue.push_back(entry);
                m_zombieObjectBytes += retainedBytes;
                return true;
            }

            void PurgeExpiredZombieObjects(uint64_t timestampUtcNsec)
            {
                while (!m_zombieParentQueue.empty() && m_zombieParentQueue.front().expirationTimestampUtcNsec <= timestampUtcNsec)
                {
                    if (!ReplayOldestZombieParent())
                    {
                        return;
                    }
                }
            }

            bool ReplayOldestZombieParent()
            {
                NVTRACE_LOG_ERR_RETURN_IF(m_zombieParentQueue.empty(), false, 50, "Continuous snapshot tried to replay an empty zombie queue");
                const auto entry = m_zombieParentQueue.front();
                m_zombieParentQueue.pop_front();
                m_zombieObjectBytes = m_zombieObjectBytes > entry.retainedBytes ? m_zombieObjectBytes - entry.retainedBytes : 0;
                return ReplayZombieParent(entry);
            }

            bool ReplayZombieParent(const ContinuousSnapshotExporterObjectModel::ZombieParentQueueEntry& entry)
            {
                if (entry.type == ZombieParentType::Process)
                {
                    auto* const pProcessState = static_cast<ProcessState*>(entry.pState);
                    NVTRACE_LOG_ERR_RETURN_IF(!pProcessState || !pProcessState->pendingDestroy, false, 50, "Continuous snapshot zombie process has no pending destroy notification");
                    const auto notification = pProcessState->pendingDestroy.notification;
                    pProcessState->pendingDestroy.Reset();
                    return Super::HandleRouterNotification(PendingProcessDestroy::notificationType, &notification);
                }
                if (entry.type == ZombieParentType::GpuPartition)
                {
                    auto* const pGpuPartitionState = static_cast<GpuPartitionState*>(entry.pState);
                    NVTRACE_LOG_ERR_RETURN_IF(!pGpuPartitionState || !pGpuPartitionState->pendingDestroy, false, 50, "Continuous snapshot zombie GPU partition has no pending destroy notification");
                    const auto notification = pGpuPartitionState->pendingDestroy.notification;
                    pGpuPartitionState->pendingDestroy.Reset();
                    return Super::HandleRouterNotification(PendingGpuPartitionDestroy::notificationType, &notification);
                }
                NVTRACE_LOG_ERR(50, "Continuous snapshot zombie parent has an invalid type");
                return false;
            }

            size_t GetProviderRetainedBytes(const ProviderState& providerState) const
            {
                return providerState.providerBufferSize;
            }

            template <typename TProviderState>
            size_t GetProviderRetainedBytesIfInitialized(const TProviderState& providerState) const
            {
                return providerState.IsInitialized() ? GetProviderRetainedBytes(providerState) : 0;
            }

            size_t GetProcessRetainedBytes(const ProcessState& processState) const
            {
                size_t retainedBytes = GetProviderRetainedBytesIfInitialized(processState.cudaApiTraceProvider) + GetProviderRetainedBytesIfInitialized(processState.nvtxTraceProvider);
                for (const auto& entry : processState.threadEventProviders)
                {
                    retainedBytes += GetProviderRetainedBytesIfInitialized(entry.second);
                }
                return retainedBytes;
            }

            size_t GetGpuPartitionRetainedBytes(const GpuPartitionState& gpuPartitionState) const
            {
                return gpuPartitionState.gpuPartitionInfo.size
                     + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuClockCalibrationProvider)
                     + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuPmCountersProvider)
                     + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuHwTraceProvider)
                     + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuGrCtxTraceProvider);
            }

            void OnThreadEventSourceProviderBound(ThreadEventProviderState&, ProviderState& providerState)
            {
                providerState.threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            }

        private:
            using Super::m_pRouterObserverClient;
            using Super::m_systemState;

            AttachWriter m_attachWriter;
            AutoResetEvent m_jobReadyEvent;
            WaitMultiplexer m_waitMultiplexer;
            std::thread m_thread;
            std::atomic<bool> m_terminateRequested = false;
            std::atomic<bool> m_threadExited = true;

            ZombieParentQueue m_zombieParentQueue;
            size_t m_zombieObjectBytes = 0;
            size_t m_maxZombieObjectBytes = 0;
            uint64_t m_historyDurationNsec = 0;

            FnExportSnapshot m_fnExportSnapshot = nullptr;
            FnOnExportCompleted m_fnOnExportCompleted = nullptr;
            FnTryTakeExportBank m_fnTryTakeExportBank = nullptr;
            FnReturnExportBank m_fnReturnExportBank = nullptr;
            ContinuousSnapshotStorageState* m_pStorageState = nullptr;
            void* m_pUserData = nullptr;
        };

    } // namespace Detail
} // namespace nv::trace

namespace nv::trace {

    class ContinuousSnapshotExporter : private ConsumerBase<ContinuousSnapshotExporter, ContinuousSnapshotExporterObjectModel>
    {
        using Super = ConsumerBase<ContinuousSnapshotExporter, ContinuousSnapshotExporterObjectModel>;
        friend Super;

        using typename Super::CpuClockCalibrationProviderState;
        using typename Super::CudaApiTraceProviderState;
        using typename Super::GpuClockCalibrationProviderState;
        using typename Super::GpuGrCtxTraceProviderState;
        using typename Super::GpuHwTraceProviderState;
        using typename Super::GpuPartitionState;
        using typename Super::GpuPmCountersProviderState;
        using typename Super::NvtxTraceProviderState;
        using typename Super::ProcessState;
        using typename Super::ProviderState;
        using typename Super::SystemState;
        using typename Super::ThreadEventProviderState;
        using ZombieParentType = ContinuousSnapshotExporterObjectModel::ZombieParentType;
        using ZombieParentQueue = ContinuousSnapshotExporterObjectModel::ZombieParentQueue;
        using PendingProviderDestroy = ContinuousSnapshotExporterObjectModel::PendingProviderDestroy;
        using PendingProcessDestroy = ContinuousSnapshotExporterObjectModel::PendingProcessDestroy;
        using PendingGpuPartitionDestroy = ContinuousSnapshotExporterObjectModel::PendingGpuPartitionDestroy;

    public:
        // Both callbacks run serially on the export thread.
        // FnExportSnapshot runs for every storage-complete snapshot, even when manifest.attachComplete is false.
        // Files remain stable through both callbacks and are then eligible for reuse.
        using FnExportSnapshot = Detail::ContinuousSnapshotAttachWorker::FnExportSnapshot;
        using FnOnExportCompleted = Detail::ContinuousSnapshotAttachWorker::FnOnExportCompleted;

        struct Configuration
        {
            size_t dataBlockSizeBytes = 0;
            uint8_t latencyBufferDataBlockCount = 0;
            size_t historyBufferDataBlockCount = 0;
            uint64_t latencyBufferSizeBytes = 0;
            uint64_t historyBufferSizeBytesPerBank = 0;
            uint64_t overflowPreventionReserveSizeBytesPerBank = 0;
            uint64_t consumeIntervalNsec = 0;
            uint64_t snapshotCadenceNsec = 0;
        };

        struct LiveStatistics
        {
            uint64_t sampleTimestampUtcNsec = 0;
            uint64_t totalCommittedBytes = 0;
            uint64_t dataRateBytesPerSec = 0;
        };

        struct Options : Super::Options
        {
            uint32_t dataBlockSizeMiB = 10;
            // TraceStreamDatabaseWriter preserves already queued blocks when
            // these fixed staging slots fill. New record serializations are
            // retained at their observer cursor until a slot becomes available
            // or the provider's SHM ring laps that cursor.
            uint8_t latencyBufferDataBlockCount = 8;
            uint64_t historyDurationNsec = 0;
            // This is the provisioned data rate used to size each of the two
            // preallocated on-disk history banks.
            uint64_t provisionedDataRateBytesPerSec = 0;
            size_t maxAttachSize = 0;
            size_t maxZombieObjectBytes = 0;
            std::filesystem::path outputDirectory;
            FnExportSnapshot fnExportSnapshot;
            FnOnExportCompleted fnOnExportCompleted;
            // Scoped cleanup is the portable fallback for platforms where the
            // pre-opened files cannot also use delete-on-close semantics.
            bool deleteFilesOnReset = true;
        };

    public:
        using Super::ConvertTscTickToUtcNsec;

        ~ContinuousSnapshotExporter()
        {
            Reset();
        }

        ContinuousSnapshotExporter()
            : m_recordWriter(*this)
        {
        }

        ContinuousSnapshotExporter(const ContinuousSnapshotExporter&) = delete;
        ContinuousSnapshotExporter& operator=(const ContinuousSnapshotExporter&) = delete;

        void Reset() noexcept
        {
            TerminateConsumeThread();
            m_isConsuming = false;
            m_consumeThreadExited = false;
            m_consumeThreadCommand = ConsumeThreadCommand::Stop;
            m_recordWriter.Reset();
            m_recordTimestampUpperBoundUtcNsec = 0;
            m_recordTimestampLowerBoundUtcNsec = 0;
            m_observationStartTsc = 0;
            m_observationStartUtcNsec = 0;
            m_recordTimestampLimitUtcNsec.store(UINT64_MAX, std::memory_order_relaxed);
            m_dataBlockWriter.Reset();
            m_databaseWriter.Reset();
            m_attachWorker.Reset();
            m_waitMultiplexer.Reset();
            m_returnedBankEvent.Reset();
            m_storageWriteCompletedEvent.Reset();
            m_consumeThreadCommandCompletedEvent.Reset();
            m_consumeThreadCommandPendingEvent.Reset();
            {
                std::lock_guard<std::mutex> lock(m_storageCompletionsMutex);
                m_storageCompletions.clear();
                m_storageCompletionScratch.clear();
            }
            {
                std::lock_guard<std::mutex> lock(m_exportBanksMutex);
                m_exportBankHead = 0;
                m_exportBankCount = 0;
            }
            {
                std::lock_guard<std::mutex> lock(m_returnedBanksMutex);
                m_returnedBankHead = 0;
                m_returnedBankCount = 0;
            }
            m_storageState.Reset();
            m_snapshotCadenceTsc = 0;
            m_historyDurationNsec = 0;
            m_nextSnapshotDeadlineTsc = 0;
            m_lastConsumeTimestampUtcNsec = 0;
            m_lastConsumeIntervalNsec = 0;
            m_lastRateSampleTimestampUtcNsec = 0;
            m_latestDataRateBytesPerSec = 0;
            m_lastRateSampleBytes = 0;
            m_zombieParentQueue.clear();
            m_zombieObjectBytes = 0;
            m_maxZombieObjectBytes = 0;
            m_isReplayingZombieDestroy = false;
            {
                std::lock_guard<std::mutex> lock(m_liveStatisticsMutex);
                m_liveStatistics = {};
            }
            m_configuration = {};
            m_consumeThreadCommandSucceeded = true;
            m_stopCommandPending = false;
            m_changeoverPending = false;
            Super::Reset();
        }

        const Configuration& GetConfiguration() const
        {
            return m_configuration;
        }

        // This intentionally uses try_lock: statistics must never make the
        // consume thread wait behind a UI reader.
        bool TryGetLiveStatistics(LiveStatistics& out_statistics) const
        {
            std::unique_lock<std::mutex> lock(m_liveStatisticsMutex, std::try_to_lock);
            if (!lock.owns_lock())
            {
                return false;
            }
            out_statistics = m_liveStatistics;
            return true;
        }

        bool Initialize(const Options& options)
        {
            Reset();
            constexpr size_t bytesPerMiB = 1024 * 1024;
            NVTRACE_LOG_ERR_RETURN_IF(options.historyDurationNsec == 0, false, 50, "ContinuousSnapshotExporter historyDurationNsec must be non-zero");
            NVTRACE_LOG_ERR_RETURN_IF(options.provisionedDataRateBytesPerSec == 0, false, 50, "ContinuousSnapshotExporter provisionedDataRateBytesPerSec must be non-zero");
            NVTRACE_LOG_ERR_RETURN_IF(options.maxAttachSize == 0, false, 50, "ContinuousSnapshotExporter maxAttachSize must be non-zero");
            NVTRACE_LOG_ERR_RETURN_IF(options.outputDirectory.empty(), false, 50, "ContinuousSnapshotExporter outputDirectory must be non-empty");
            NVTRACE_LOG_ERR_RETURN_IF(!options.fnExportSnapshot, false, 50, "ContinuousSnapshotExporter requires an export callback");

            const uint32_t normalizedDataBlockSizeMiB = std::max<uint32_t>(1, options.dataBlockSizeMiB);
            NVTRACE_LOG_ERR_RETURN_IF(normalizedDataBlockSizeMiB > std::numeric_limits<size_t>::max() / bytesPerMiB, false, 50, "ContinuousSnapshotExporter dataBlockSizeMiB is too large");
            const size_t dataBlockSize = static_cast<size_t>(normalizedDataBlockSizeMiB) * bytesPerMiB;
            const size_t overflowPreventionReserveDataBlockCount = Detail::CalculateContinuousSnapshotOverflowPreventionReserveDataBlockCount(dataBlockSize);
            const uint8_t latencyBufferDataBlockCount = std::max<uint8_t>(4, options.latencyBufferDataBlockCount);
            m_historyDurationNsec = options.historyDurationNsec;
            m_maxZombieObjectBytes = options.maxZombieObjectBytes;

            NvTrace_RouterObserverClient_RouterConfig routerConfig = {};
            if (!Super::Initialize(options, &routerConfig))
            {
                return false;
            }

            auto resetGuard = ScopeExitGuard([this]() { Reset(); });

            size_t dataBlockCountPerBank = 0;
            NVTRACE_LOG_ERR_RETURN_IF(
                !Detail::CalculateContinuousSnapshotHistoryDataBlockCount(
                    options.historyDurationNsec,
                    options.provisionedDataRateBytesPerSec,
                    dataBlockSize,
                    dataBlockCountPerBank),
                false,
                50,
                "ContinuousSnapshotExporter failed to calculate a valid history bank size");

            NVTRACE_LOG_ERR_RETURN_IF(
                !m_storageState.Initialize(
                    options.outputDirectory,
                    dataBlockSize,
                    dataBlockCountPerBank,
                    options.maxAttachSize,
                    options.historyDurationNsec,
                    options.deleteFilesOnReset),
                false,
                50,
                "ContinuousSnapshotExporter failed to initialize storage");

            NVTRACE_LOG_ERR_RETURN_IF(
                !m_attachWorker.Initialize(
                    options,
                    dataBlockSize,
                    options.historyDurationNsec,
                    m_storageState.AttachFilePaths(),
                    &m_storageState,
                    options.fnExportSnapshot,
                    options.fnOnExportCompleted,
                    &ContinuousSnapshotExporter::TryTakeExportBank,
                    &ContinuousSnapshotExporter::ReturnExportBank,
                    this),
                false,
                50,
                "ContinuousSnapshotExporter failed to initialize Attach/export worker");

            NVTRACE_LOG_ERR_RETURN_IF(dataBlockCountPerBank > std::numeric_limits<size_t>::max() / 2, false, 50, "ContinuousSnapshotExporter completion queue capacity overflows size_t");
            try
            {
                m_storageCompletions.reserve(dataBlockCountPerBank * 2);
                m_storageCompletionScratch.reserve(dataBlockCountPerBank * 2);
            }
            catch (const std::exception& exception)
            {
                NVTRACE_LOG_ERR(50, "Failed to allocate continuous snapshot handoff queues: %s", exception.what());
                return false;
            }

            typename TraceStreamDatabaseWriter<Detail::ContinuousSnapshotStorageWriter>::Options databaseWriterOptions;
            databaseWriterOptions.dataBlockSize = dataBlockSize;
            databaseWriterOptions.maxGpuPartitionCountPerDataBlock = routerConfig.maxGpuPartitionCount * 5 / 4;
            databaseWriterOptions.maxProcessCountPerDataBlock = routerConfig.maxProcessCount * 5 / 4;
            databaseWriterOptions.maxProviderCountPerDataBlock = routerConfig.maxProviderCount * 5 / 4;
            databaseWriterOptions.maxThreadEventProviderMappingCountPerDataBlock = routerConfig.maxProviderCount * 5 / 4;
            databaseWriterOptions.chunkCountPerDataBlock = 1;
            databaseWriterOptions.writeSlotCount = latencyBufferDataBlockCount;
            databaseWriterOptions.fnOnStorageWriteCompleted = &ContinuousSnapshotExporter::OnStorageWriteCompleted;
            databaseWriterOptions.pUserDataOnStorageWriteCompleted = this;
            if (!m_databaseWriter.Initialize(databaseWriterOptions, [this, latencyBufferDataBlockCount](Detail::ContinuousSnapshotStorageWriter& storageWriter) {
                    return storageWriter.Initialize(&m_storageState, latencyBufferDataBlockCount);
                }))
            {
                return false;
            }

            const TscRtcCalibrationPoint& calibrationPoint = this->GetLatestCpuClockCalibrationPoint();
            m_snapshotCadenceTsc = static_cast<uint64_t>(static_cast<double>(options.historyDurationNsec) * calibrationPoint.utcToTscFactor);
            NVTRACE_LOG_ERR_RETURN_IF(m_snapshotCadenceTsc == 0, false, 50, "ContinuousSnapshotExporter snapshot cadence rounds to zero TSC ticks");

            NVTRACE_LOG_ERR_RETURN_IF(dataBlockCountPerBank > std::numeric_limits<uint64_t>::max() / dataBlockSize, false, 50, "ContinuousSnapshotExporter history bank size overflows uint64_t");
            m_configuration.dataBlockSizeBytes = dataBlockSize;
            m_configuration.latencyBufferDataBlockCount = latencyBufferDataBlockCount;
            m_configuration.historyBufferDataBlockCount = dataBlockCountPerBank;
            m_configuration.latencyBufferSizeBytes = static_cast<uint64_t>(latencyBufferDataBlockCount) * dataBlockSize;
            m_configuration.historyBufferSizeBytesPerBank = static_cast<uint64_t>(dataBlockCountPerBank) * dataBlockSize;
            m_configuration.overflowPreventionReserveSizeBytesPerBank = static_cast<uint64_t>(overflowPreventionReserveDataBlockCount) * dataBlockSize;
            m_configuration.consumeIntervalNsec = options.consumeIntervalMsec * 1'000'000;
            m_configuration.snapshotCadenceNsec = options.historyDurationNsec;

            bool succeeded = m_consumeThreadCommandPendingEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: AutoResetEvent::Initialize failed for consumeThreadCommandPendingEvent");
            succeeded = m_consumeThreadCommandCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: AutoResetEvent::Initialize failed for consumeThreadCommandCompletedEvent");
            succeeded = m_returnedBankEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: AutoResetEvent::Initialize failed for returnedBankEvent");
            succeeded = m_storageWriteCompletedEvent.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: AutoResetEvent::Initialize failed for storageWriteCompletedEvent");

            succeeded = BuildWaitMultiplexer();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: BuildWaitMultiplexer failed");

            try
            {
                m_consumeThread = std::thread(&ContinuousSnapshotExporter::ConsumeLoop, this);
            }
            catch (const std::system_error& exception)
            {
                NVTRACE_LOG_ERR(50, "Failed to create consume thread (std::system_error %d: %s).", exception.code().value(), exception.what());
                return false;
            }

            resetGuard.Dismiss();
            return true;
        }

        bool Start(uint64_t observationStartUtcNsec = 0)
        {
            if (m_consumeThreadCommand != ConsumeThreadCommand::Stop || m_consumeThreadExited)
            {
                return false;
            }

            m_recordTimestampLimitUtcNsec.store(UINT64_MAX, std::memory_order_release);
            m_observationStartUtcNsec = observationStartUtcNsec;
            m_consumeThreadCommand = ConsumeThreadCommand::Start;
            std::atomic_thread_fence(std::memory_order_release);

            bool succeeded = m_consumeThreadCommandPendingEvent.Signal();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter::Start: AutoResetEvent::Signal failed (consumeThreadCommandPendingEvent)");
            succeeded = m_consumeThreadCommandCompletedEvent.Wait();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter::Start: AutoResetEvent::Wait failed (consumeThreadCommandCompletedEvent)");

            std::atomic_thread_fence(std::memory_order_acquire);
            if (m_consumeThreadExited)
            {
                return false;
            }

            return true;
        }

        bool Stop()
        {
            if (m_consumeThreadCommand != ConsumeThreadCommand::Start || m_consumeThreadExited)
            {
                return false;
            }

            m_consumeThreadCommand = ConsumeThreadCommand::Stop;
            m_consumeThreadCommandSucceeded = true;
            std::atomic_thread_fence(std::memory_order_release);

            bool succeeded = m_consumeThreadCommandPendingEvent.Signal();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter::Stop: AutoResetEvent::Signal failed (consumeThreadCommandPendingEvent)");
            succeeded = m_consumeThreadCommandCompletedEvent.Wait();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter::Stop: AutoResetEvent::Wait failed (consumeThreadCommandCompletedEvent)");

            std::atomic_thread_fence(std::memory_order_acquire);
            if (m_consumeThreadExited || !m_consumeThreadCommandSucceeded)
            {
                return false;
            }

            m_isConsuming = false;
            return true;
        }

        bool StopAtTsc(uint64_t targetTsc)
        {
            if (targetTsc == 0)
            {
                targetTsc = ReadCpuTscTimestamp();
            }
            return StopAtUtc(ConvertTscTickToUtcNsec(targetTsc));
        }

        bool StopAtUtc(uint64_t targetUtcNsec)
        {
            if (m_consumeThreadCommand != ConsumeThreadCommand::Start || m_consumeThreadExited)
            {
                return false;
            }

            // Continue consuming through the grace period without extending the requested recording timespan.
            m_recordTimestampLimitUtcNsec.store(targetUtcNsec, std::memory_order_release);
            this->ConsumeUpToUtc(targetUtcNsec);
            return Stop();
        }

    private:
        using Super::ConsumeData;
        using Super::ScheduleImmediateConsumeIteration;
        using Super::ScheduleNextConsumeIteration;

        static bool TryTakeExportBank(void* pUserData, size_t& out_bankIndex)
        {
            auto& exporter = *static_cast<ContinuousSnapshotExporter*>(pUserData);
            std::lock_guard<std::mutex> lock(exporter.m_exportBanksMutex);
            if (exporter.m_exportBankCount == 0)
            {
                return false;
            }
            out_bankIndex = exporter.m_exportBanks[exporter.m_exportBankHead];
            exporter.m_exportBankHead = (exporter.m_exportBankHead + 1) % exporter.m_exportBanks.size();
            exporter.m_exportBankCount -= 1;
            return true;
        }

        static void ReturnExportBank(void* pUserData, size_t bankIndex)
        {
            auto& exporter = *static_cast<ContinuousSnapshotExporter*>(pUserData);
            {
                std::lock_guard<std::mutex> lock(exporter.m_returnedBanksMutex);
                NVTRACE_LOG_ERR_RETURN_IF(exporter.m_returnedBanks.size() <= exporter.m_returnedBankCount, /*void*/, 50, "ContinuousSnapshotExporter returned-bank queue overflowed");
                const size_t tailIndex = (exporter.m_returnedBankHead + exporter.m_returnedBankCount) % exporter.m_returnedBanks.size();
                exporter.m_returnedBanks[tailIndex] = bankIndex;
                exporter.m_returnedBankCount += 1;
            }
            const bool succeeded = exporter.m_returnedBankEvent.Signal();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "ContinuousSnapshotExporter failed to signal a returned export bank");
        }

        static void OnStorageWriteCompleted(void* pUserData, uint8_t writeSlotIndex, uint64_t dataBlockSequenceNumber, bool succeeded)
        {
            auto& exporter = *static_cast<ContinuousSnapshotExporter*>(pUserData);
            Detail::ContinuousSnapshotStorageState::StorageCompletion completion;
            if (!exporter.m_databaseWriter.StorageWriter().TakeStorageCompletion(writeSlotIndex, dataBlockSequenceNumber, succeeded, completion))
            {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(exporter.m_storageCompletionsMutex);
                NVTRACE_LOG_ERR_RETURN_IF(exporter.m_storageCompletions.capacity() <= exporter.m_storageCompletions.size(), /*void*/, 50, "ContinuousSnapshotExporter storage-completion queue overflowed");
                exporter.m_storageCompletions.push_back(completion);
            }
            const bool signaled = exporter.m_storageWriteCompletedEvent.Signal();
            NVTRACE_LOG_ERR_IF(!signaled, 50, "ContinuousSnapshotExporter failed to signal a storage completion");
        }

        bool HandleRouterNotification(NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification)
        {
            switch (notificationType)
            {
                case NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed:
                    return DelayGpuPartitionDestroyedNotification(*reinterpret_cast<const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification*>(pNotification));
                case NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed:
                    return DelayProcessDestroyedNotification(*reinterpret_cast<const NvTrace_RouterObserverClient_ProcessDestroyedNotification*>(pNotification));
                case NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed:
                {
                    const auto& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProviderDestroyedNotification*>(pNotification);
                    if (ResolveProviderTypeUuidAlias(notification.providerTypeUuid) == NvTrace_ThreadEvent_ProviderTypeUuid)
                    {
                        return Super::HandleRouterNotification(notificationType, pNotification);
                    }
                    if (m_systemState.processes.find(notification.parentLuid) != m_systemState.processes.end()
                        || m_systemState.gpuPartitions.find(notification.parentLuid) != m_systemState.gpuPartitions.end())
                    {
                        return DelayProviderDestroyedNotification(notification);
                    }
                    return Super::HandleRouterNotification(notificationType, pNotification);
                }
                default:
                    return Super::HandleRouterNotification(notificationType, pNotification);
            }
        }

        bool DelayProviderDestroyedNotification(const NvTrace_RouterObserverClient_ProviderDestroyedNotification& notification)
        {
            ProviderState* const pProviderState = m_systemState.GetProviderState(notification.providerLuid);
            NVTRACE_LOG_ERR_RETURN_IF(!pProviderState || pProviderState->pendingDestroy, false, 50, "Invalid duplicate or unknown ProviderDestroyed notification in continuous snapshot exporter");
            OnProviderDestroyed(*pProviderState, ProviderDestroyReason::Explicit);
            pProviderState->pendingDestroy.notification = notification;
            pProviderState->pendingDestroy.isPending = true;
            return true;
        }

        bool DelayProcessDestroyedNotification(const NvTrace_RouterObserverClient_ProcessDestroyedNotification& notification)
        {
            auto itr = m_systemState.processes.find(notification.processLuid);
            NVTRACE_LOG_ERR_RETURN_IF(itr == m_systemState.processes.end() || itr->second.pendingDestroy, false, 50, "Invalid duplicate or unknown ProcessDestroyed notification in continuous snapshot exporter");
            OnProcessDestroyed(itr->second, notification.destructionTimestampTsc);
            itr->second.pendingDestroy.notification = notification;
            itr->second.pendingDestroy.isPending = true;
            return QueueZombieParent(ZombieParentType::Process, &itr->second, GetProcessRetainedBytes(itr->second));
        }

        bool DelayGpuPartitionDestroyedNotification(const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification& notification)
        {
            auto itr = m_systemState.gpuPartitions.find(notification.gpuPartitionLuid);
            NVTRACE_LOG_ERR_RETURN_IF(itr == m_systemState.gpuPartitions.end() || itr->second.pendingDestroy, false, 50, "Invalid duplicate or unknown GpuPartitionDestroyed notification in continuous snapshot exporter");
            OnGpuPartitionDestroyed(itr->second, notification.destructionTimestampTsc);
            itr->second.pendingDestroy.notification = notification;
            itr->second.pendingDestroy.isPending = true;
            return QueueZombieParent(ZombieParentType::GpuPartition, &itr->second, GetGpuPartitionRetainedBytes(itr->second));
        }

        bool QueueZombieParent(ZombieParentType type, void* pState, size_t retainedBytes)
        {
            if (m_maxZombieObjectBytes == 0 || retainedBytes > m_maxZombieObjectBytes)
            {
                NVTRACE_LOG_WRN(50, "Forwarding a continuous snapshot zombie immediately because it exceeds maxZombieObjectBytes=%zu", m_maxZombieObjectBytes);
                return ReplayZombieParent({type, pState, 0, retainedBytes});
            }
            while (m_zombieObjectBytes > m_maxZombieObjectBytes - retainedBytes && !m_zombieParentQueue.empty())
            {
                NVTRACE_LOG_WRN(50, "Evicting a continuous snapshot zombie before its time window expires to satisfy maxZombieObjectBytes=%zu", m_maxZombieObjectBytes);
                if (!ReplayOldestZombieParent())
                {
                    ReplayZombieParent({type, pState, 0, retainedBytes});
                    return false;
                }
            }

            ContinuousSnapshotExporterObjectModel::ZombieParentQueueEntry entry;
            entry.type = type;
            entry.pState = pState;
            entry.retainedBytes = retainedBytes;
            const uint64_t nowUtcNsec = ConvertTscTickToUtcNsecForConsume(ReadCpuTscTimestamp());
            entry.expirationTimestampUtcNsec = nowUtcNsec > std::numeric_limits<uint64_t>::max() - m_historyDurationNsec ? std::numeric_limits<uint64_t>::max() : nowUtcNsec + m_historyDurationNsec;
            m_zombieParentQueue.push_back(entry);
            m_zombieObjectBytes += retainedBytes;
            return true;
        }

        void PurgeExpiredZombieParents(uint64_t timestampUtcNsec)
        {
            while (!m_zombieParentQueue.empty() && m_zombieParentQueue.front().expirationTimestampUtcNsec <= timestampUtcNsec)
            {
                if (!ReplayOldestZombieParent())
                {
                    return;
                }
            }
        }

        bool ReplayOldestZombieParent()
        {
            NVTRACE_LOG_ERR_RETURN_IF(m_zombieParentQueue.empty(), false, 50, "Continuous snapshot tried to replay an empty zombie queue");
            const auto entry = m_zombieParentQueue.front();
            m_zombieParentQueue.pop_front();
            m_zombieObjectBytes = m_zombieObjectBytes > entry.retainedBytes ? m_zombieObjectBytes - entry.retainedBytes : 0;
            return ReplayZombieParent(entry);
        }

        bool ReplayZombieParent(const ContinuousSnapshotExporterObjectModel::ZombieParentQueueEntry& entry)
        {
            m_isReplayingZombieDestroy = true;
            auto replayGuard = ScopeExitGuard([this]() { m_isReplayingZombieDestroy = false; });
            if (entry.type == ZombieParentType::Process)
            {
                auto* const pProcessState = static_cast<ProcessState*>(entry.pState);
                NVTRACE_LOG_ERR_RETURN_IF(!pProcessState || !pProcessState->pendingDestroy, false, 50, "Continuous snapshot zombie process has no pending destroy notification");
                const auto notification = pProcessState->pendingDestroy.notification;
                pProcessState->pendingDestroy.Reset();
                return Super::HandleRouterNotification(PendingProcessDestroy::notificationType, &notification);
            }
            if (entry.type == ZombieParentType::GpuPartition)
            {
                auto* const pGpuPartitionState = static_cast<GpuPartitionState*>(entry.pState);
                NVTRACE_LOG_ERR_RETURN_IF(!pGpuPartitionState || !pGpuPartitionState->pendingDestroy, false, 50, "Continuous snapshot zombie GPU partition has no pending destroy notification");
                const auto notification = pGpuPartitionState->pendingDestroy.notification;
                pGpuPartitionState->pendingDestroy.Reset();
                return Super::HandleRouterNotification(PendingGpuPartitionDestroy::notificationType, &notification);
            }
            NVTRACE_LOG_ERR(50, "Continuous snapshot zombie parent has an invalid type");
            return false;
        }

        size_t GetProviderRetainedBytes(const ProviderState& providerState) const
        {
            return providerState.providerBufferSize;
        }

        template <typename TProviderState>
        size_t GetProviderRetainedBytesIfInitialized(const TProviderState& providerState) const
        {
            return providerState.IsInitialized() ? GetProviderRetainedBytes(providerState) : 0;
        }

        size_t GetProcessRetainedBytes(const ProcessState& processState) const
        {
            size_t retainedBytes = GetProviderRetainedBytesIfInitialized(processState.cudaApiTraceProvider) + GetProviderRetainedBytesIfInitialized(processState.nvtxTraceProvider);
            for (const auto& entry : processState.threadEventProviders)
            {
                retainedBytes += GetProviderRetainedBytesIfInitialized(entry.second);
            }
            return retainedBytes;
        }

        size_t GetGpuPartitionRetainedBytes(const GpuPartitionState& gpuPartitionState) const
        {
            return gpuPartitionState.gpuPartitionInfo.size
                 + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuClockCalibrationProvider)
                 + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuPmCountersProvider)
                 + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuHwTraceProvider)
                 + GetProviderRetainedBytesIfInitialized(gpuPartitionState.gpuGrCtxTraceProvider);
        }

        void OnSystemCreated(SystemState& systemState)
        {
            if (m_isConsuming)
            {
                AttachSystem(systemState);
            }
        }
        void OnGpuPartitionCreated(GpuPartitionState& gpuPartitionState)
        {
            if (m_isConsuming)
            {
                gpuPartitionState.observationStartGpuNsec = 0;
                gpuPartitionState.isObservationStartResolved = m_observationStartTsc == 0;
                AttachGpuPartition(gpuPartitionState);
            }
        }
        void OnProcessCreated(ProcessState& processState)
        {
            if (m_isConsuming)
            {
                AttachProcess(processState);
            }
        }
        void OnProviderCreated_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            if (m_isConsuming)
            {
                ConfigureCpuObservationStart(providerState);
                AttachProvider_CpuClockCalibration(providerState);
            }
        }
        void OnProviderCreated_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                ConfigureGpuClockObservationStart(providerState);
                AttachProvider_GpuClockCalibration(providerState, parentState);
            }
        }
        void OnProviderCreated_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareGpuPmCounters(providerState, parentState);
            }
        }
        void OnProviderCreated_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareGpuHwTrace(providerState, parentState);
            }
        }
        void OnProviderCreated_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareGpuGrCtxTrace(providerState, parentState);
            }
        }
        void OnProviderCreated_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                AttachProvider_CudaApiTrace(providerState, parentState);
            }
        }
        void OnProviderCreated_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                AttachProvider_NvtxTrace(providerState, parentState);
            }
        }

        void OnProviderCreated_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                providerState.isReadyToConsume = false;
                TryPrepareThreadEvent(providerState, parentState, ReadCpuTscTimestamp());
            }
        }

        void OnGpuPartitionDestroyed(GpuPartitionState& gpuPartitionState, uint64_t destructionTimestampTsc)
        {
            if (m_isConsuming && !m_isReplayingZombieDestroy)
            {
                AttachGpuPartitionDestroyed(gpuPartitionState, destructionTimestampTsc);
            }
        }
        void OnProcessDestroyed(ProcessState& processState, uint64_t destructionTimestampTsc)
        {
            if (m_isConsuming && !m_isReplayingZombieDestroy)
            {
                AttachProcessDestroyed(processState, destructionTimestampTsc);
            }
        }
        void OnProviderDestroyed(ProviderState& providerState, ProviderDestroyReason reason)
        {
            auto itrProcessState = m_systemState.processes.find(providerState.processLuid);
            if (m_isConsuming && itrProcessState != m_systemState.processes.end())
            {
                ProcessState& parentState = itrProcessState->second;
                if (reason == ProviderDestroyReason::Explicit)
                {
                    if (ThreadEventProviderState* const pThreadEventProviderState = parentState.FindThreadEventProvider(providerState.threadEventProviderLuid))
                    {
                        m_recordWriter.SetProcessProviderRef(providerState, parentState);
                        ConsumeData_ThreadEvent(*pThreadEventProviderState, parentState, ReadCpuTscTimestamp());
                    }
                }
            }
        }
        void OnProviderDestroyed_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                ConsumeData_CpuClockCalibration(providerState, ReadEndpoints{ReadCpuTscTimestamp(), 0});
            }
        }
        void OnProviderDestroyed_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuClockCalibration(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuPmCounters(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuHwTrace(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_GpuGrCtxTrace(providerState, parentState, ReadEndpoints{readEndpointTsc, this->ResolveGpuPtimerReadEndpoint(parentState, readEndpointTsc)});
            }
        }
        void OnProviderDestroyed_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                const uint64_t readEndpointTsc = ReadCpuTscTimestamp();
                ConsumeData_CudaApiTrace(providerState, parentState, readEndpointTsc);
            }
        }
        void OnProviderDestroyed_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            (void)providerState;
            (void)parentState;
        }
        void OnProviderDestroyed_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            if (m_isConsuming)
            {
                // Ensure any pending data is consumed before destruction
                ConsumeData_ThreadEvent(providerState, parentState, ReadCpuTscTimestamp());
            }
        }

        void ConfigureExistingObservationStarts()
        {
            if (m_systemState.cpuClockCalibrationProvider.IsInitialized())
            {
                ConfigureCpuObservationStart(m_systemState.cpuClockCalibrationProvider);
            }
            for (auto& [_, gpuPartitionState] : m_systemState.gpuPartitions)
            {
                gpuPartitionState.observationStartGpuNsec = 0;
                gpuPartitionState.isObservationStartResolved = m_observationStartTsc == 0;
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                {
                    ConfigureGpuClockObservationStart(gpuPartitionState.gpuClockCalibrationProvider);
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                {
                    gpuPartitionState.gpuPmCountersProvider.isReadyToConsume = false;
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                {
                    gpuPartitionState.gpuHwTraceProvider.isReadyToConsume = false;
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                {
                    gpuPartitionState.gpuGrCtxTraceProvider.isReadyToConsume = false;
                }
            }
            for (auto& [_, processState] : m_systemState.processes)
            {
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    threadEventProviderEntry.second.isReadyToConsume = false;
                }
            }
        }

        void ConfigureCpuObservationStart(CpuClockCalibrationProviderState& providerState)
        {
            providerState.observer.SetObservationStart(ReadEndpoints{m_observationStartTsc, 0});
        }

        void ConfigureGpuClockObservationStart(GpuClockCalibrationProviderState& providerState)
        {
            providerState.observer.SetObservationStart(ReadEndpoints{m_observationStartTsc, 0});
        }

        bool TryResolveGpuObservationStart(GpuPartitionState& parentState)
        {
            if (parentState.isObservationStartResolved)
            {
                return true;
            }

            TscGpuCalibrationPoint calibrationPoint;
            if (!parentState.gpuClockCalibrationProvider.IsInitialized() || !parentState.gpuClockCalibrationProvider.observer.GetLatestCalibrationPoint(calibrationPoint))
            {
                return false;
            }
            parentState.observationStartGpuNsec = calibrationPoint.ToGpu(m_observationStartTsc);
            parentState.isObservationStartResolved = true;
            return true;
        }

        template <typename TProviderState>
        bool TryPrepareGpuProvider(
            TProviderState& providerState,
            GpuPartitionState& parentState,
            void (ContinuousSnapshotExporter::*fnAttachProvider)(TProviderState&, GpuPartitionState&))
        {
            if (providerState.isReadyToConsume)
            {
                return true;
            }
            if (!TryResolveGpuObservationStart(parentState))
            {
                return false;
            }

            providerState.observer.SetObservationStart(ReadEndpoints{m_observationStartTsc, parentState.observationStartGpuNsec});
            (this->*fnAttachProvider)(providerState, parentState);
            providerState.isReadyToConsume = providerState.hasProviderCreatedRecord;
            return providerState.isReadyToConsume;
        }

        bool TryPrepareGpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            return TryPrepareGpuProvider(providerState, parentState, &ContinuousSnapshotExporter::AttachProvider_GpuPmCounters);
        }

        bool TryPrepareGpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            return TryPrepareGpuProvider(providerState, parentState, &ContinuousSnapshotExporter::AttachProvider_GpuHwTrace);
        }

        bool TryPrepareGpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            return TryPrepareGpuProvider(providerState, parentState, &ContinuousSnapshotExporter::AttachProvider_GpuGrCtxTrace);
        }

        void ConsumeData_CpuClockCalibration(CpuClockCalibrationProviderState& providerState, const ReadEndpoints& readEndpoints)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return; // Not an error.
            }

            m_recordWriter.SetSystemProviderRef(providerState);

            // Retry only while an on-attach record remains pending. If another record becomes Ready, publish a new
            // ProviderAttached record so the forward scan observes the expanded attachment set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::CpuClockCalibration>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                    m_recordWriter.SetSystemProviderRef(providerState);
                }
            }

            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) -> bool {
                    const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CpuClockCalibration>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!written && !m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write CPU clock calibration record.");
                    }
                    return AcceptSerializedRecord(written);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!providerState.hasProviderCreatedRecord)
            {
                return; // Not an error.
            }
            m_recordWriter.SetGpuProviderRef(providerState, parentState);

            // Retry only while an on-attach record remains pending. If another record becomes Ready, publish a new
            // ProviderAttached record so the forward scan observes the expanded attachment set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuClockCalibration>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                    m_recordWriter.SetGpuProviderRef(providerState, parentState);
                }
            }

            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) -> bool {
                    const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuClockCalibration>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!written && !m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU clock calibration record.");
                    }
                    return AcceptSerializedRecord(written);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!TryPrepareGpuPmCounters(providerState, parentState))
            {
                return;
            }
            // Retry only while an on-attach DataBlob remains pending. If another record becomes Ready, publish a new
            // ProviderAttached record so the forward scan observes the expanded attachment set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                m_recordWriter.SetGpuProviderRef(providerState, parentState);
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuPmCounters>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                }
            }

            m_recordWriter.SetGpuProviderRef(providerState, parentState);

            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) -> bool {
                    const bool succeeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuPmCounters>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!succeeded && !m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU PM counters recurring record (providerLuid=%" PRIu64 " sequenceNumber=%" PRIu64 " recordType=%" PRIu32 ").", providerState.luid, sequenceNumber, recordType);
                    }
                    return AcceptSerializedRecord(succeeded);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!TryPrepareGpuHwTrace(providerState, parentState))
            {
                return;
            }
            // Retry only while an on-attach DataBlob remains pending. Publish a new ProviderAttached record when the
            // attachment set grows so the forward scan observes the complete set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                m_recordWriter.SetGpuProviderRef(providerState, parentState);
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuHwTrace>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                }
            }

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) -> bool {
                    const bool succeeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuHwTrace>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!succeeded && !m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU HW trace recurring record (providerLuid=%" PRIu64 " sequenceNumber=%" PRIu64 " recordType=%" PRIu32 ").", providerState.luid, sequenceNumber, recordType);
                    }
                    return AcceptSerializedRecord(succeeded);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints)
        {
            if (!TryPrepareGpuGrCtxTrace(providerState, parentState))
            {
                return;
            }
            // Retry only while an on-attach DataBlob remains pending. Publish a new ProviderAttached record when the
            // attachment set grows so the forward scan observes the complete set.
            if (providerState.onAttachRecordLocations.size() != providerState.observer.GetOnAttachRecordsTotalCount())
            {
                m_recordWriter.SetGpuProviderRef(providerState, parentState);
                if (TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuGrCtxTrace>(providerState))
                {
                    m_recordWriter.SetNullProviderRef();
                    EmitProviderAttachedRecord(providerState);
                }
            }

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            providerState.observer.ConsumeCircularBufferRecords(
                readEndpoints,
                [&](uint64_t sequenceNumber, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) -> bool {
                    const bool succeeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::GpuGrCtxTrace>(
                        m_recordWriter,
                        recordType,
                        sequenceNumber,
                        pPayload,
                        payloadSize);
                    if (!succeeded && !m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write GPU GR context trace recurring record (providerLuid=%" PRIu64 " sequenceNumber=%" PRIu64 " recordType=%" PRIu32 ").", providerState.luid, sequenceNumber, recordType);
                    }
                    return AcceptSerializedRecord(succeeded);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc)
        {
            (void)readEndpointTsc;
            if (!providerState.hasProviderCreatedRecord)
            {
                return; // Not an error.
            }

            m_recordWriter.SetProcessProviderRef(providerState, parentState);
            providerState.observer.ConsumeGpuTimestampEvents(
                [&](uint64_t gpuTimestampRef, uint64_t ptimer) -> bool {
                    const NvTrace_CUDA_ApiTrace_GpuTimestampReport report = {gpuTimestampRef, ptimer};
                    const bool reportWritten = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CudaApiTrace>(m_recordWriter, NvTrace_CUDA_ApiTrace_RecordType_GpuTimestampReport, NVTRACE_NO_CORRELATION_ID, reinterpret_cast<const uint8_t*>(&report), sizeof(report));
                    if (!reportWritten && !m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write CUDA GPU timestamp record.");
                    }
                    return AcceptSerializedRecord(reportWritten);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        void ConsumeData_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc)
        {
            if (!TryPrepareThreadEvent(providerState, parentState, readEndpointTsc))
            {
                return;
            }

            Super::ConsumeThreadEventSourceProviderRegistrations(parentState, providerState, readEndpointTsc);

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            // Ensure sufficient inline payload capacity; this assert is specific to TraceStreamDatabaseWriter and should be removed when dependence on the inline payload is removed.
            static_assert(sizeof(Codec::NvtxTraceEvents::ThreadEventInlinePayload) <= TSF_RECORD_INLINE_PAYLOAD_SIZE, "NVTX Trace event inline payload exceeds TSF inline payload capacity");

            providerState.observer.ConsumeThreadEvents(
                readEndpointTsc,
                [&](uint32_t sourceProviderIndex, const ThreadEventRecord& record) -> bool {
                    return Super::ConsumeThreadEvent(parentState, providerState, sourceProviderIndex, record, readEndpointTsc);
                },
                [&](bool consistencyCheckPassed) {
                    if (consistencyCheckPassed)
                    {
                        m_recordWriter.Commit();
                    }
                    else
                    {
                        m_recordWriter.Rollback();
                    }
                });
        }

        bool TryPrepareThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc)
        {
            if (providerState.isReadyToConsume)
            {
                return true;
            }
            if (!providerState.hasProviderCreatedRecord)
            {
                AttachProvider_ThreadEvent(providerState, parentState);
                if (!providerState.hasProviderCreatedRecord)
                {
                    return false;
                }
            }
            if (m_observationStartTsc == 0)
            {
                providerState.isReadyToConsume = true;
                return true;
            }

            const uint64_t warmupEndpointTsc = std::min(readEndpointTsc, m_observationStartTsc - 1);

            // Consume complete FIFO cells preceding the observation window so object state remains
            // reconstructible, but leave the first straddling cell for the normal consume path. Nothing
            // is held here: the skip below would discard a held record and everything behind it.
            {
                const auto dropUnboundGuard = Super::ScopeGuardDropUnboundThreadEvents();
                providerState.observer.ConsumeThreadEvents(
                    warmupEndpointTsc,
                    [&](uint32_t sourceProviderIndex, const ThreadEventRecord& record) -> bool {
                        if (!Super::IsObjectLifetimeThreadEvent(providerState, sourceProviderIndex, record))
                        {
                            return true;
                        }
                        return Super::ConsumeThreadEvent(parentState, providerState, sourceProviderIndex, record, warmupEndpointTsc);
                    },
                    [&](bool consistencyCheckPassed) {
                        if (consistencyCheckPassed)
                        {
                            m_recordWriter.Commit();
                        }
                        else
                        {
                            m_recordWriter.Rollback();
                        }
                    });
            }

            if (readEndpointTsc < m_observationStartTsc || !providerState.observer.SkipEventBufferEntriesBefore(m_observationStartTsc))
            {
                return false;
            }
            providerState.isReadyToConsume = true;
            return true;
        }

        void RegisterThreadEventConsumeCallbacks()
        {
            Super::RegisterThreadEventConsumeCallback(NvTrace_ThreadEvent_ProviderTypeUuid, &ContinuousSnapshotExporter::ConsumeThreadEvent_ThreadEventCallback);
            Super::RegisterThreadEventConsumeCallback(NvTrace_CUDA_ApiTrace_ProviderTypeUuid, &ContinuousSnapshotExporter::ConsumeThreadEvent_CudaApiTraceCallback);
            Super::RegisterThreadEventConsumeCallback(NvTrace_NvtxTrace_ProviderTypeUuid, &ContinuousSnapshotExporter::ConsumeThreadEvent_NvtxTraceCallback);
        }

        bool AcceptSerializedRecord(bool writeSucceeded) const
        {
            // Only temporary storage pressure is retryable. A format/configuration error cannot be
            // repaired by pinning the provider cursor on the same record forever.
            return writeSucceeded || !m_recordWriter.IsStorageBackpressured();
        }

        static bool ConsumeThreadEvent_ThreadEventCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            ContinuousSnapshotExporter& consumer = *reinterpret_cast<ContinuousSnapshotExporter*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            ThreadEventProviderState& providerState = *static_cast<ThreadEventProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            switch (record.recordType)
            {
                case NvTrace_ReservedRecordType_RecordReference:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < sizeof(NvTrace_RecordReference), true, 50, "Invalid record reference ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

                    NvTrace_RecordReference reference;
                    memcpy(&reference, record.pPayload, sizeof(reference));

                    bool eventResolved = false;
                    bool eventSucceeded = false;
                    providerState.observer.ResolveThreadStateReference(
                        reference,
                        [&](const NvTrace_ThreadEvent_ThreadCreated& threadCreated) {
                            eventResolved = true;
                            eventSucceeded = Codec::ThreadEventEncoders::WriteThreadCreatedRecord(consumer.m_recordWriter, threadCreated);
                        },
                        [&](bool consistencyCheckPassed) { CommitOrRollback(consumer.m_recordWriter, consistencyCheckPassed && eventSucceeded); });
                    return !eventResolved || consumer.AcceptSerializedRecord(eventSucceeded);
                }
                case NvTrace_ThreadEvent_RecordType_ThreadCreated:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < sizeof(NvTrace_ThreadEvent_ThreadCreated), true, 50, "Invalid thread created ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

                    NvTrace_ThreadEvent_ThreadCreated threadCreated;
                    memcpy(&threadCreated, record.pPayload, sizeof(threadCreated));
                    return consumer.AcceptSerializedRecord(Codec::ThreadEventEncoders::WriteThreadCreatedRecord(consumer.m_recordWriter, threadCreated));
                }
                case NvTrace_ThreadEvent_RecordType_ThreadDestroyed:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < sizeof(NvTrace_ThreadEvent_ThreadDestroyed), true, 50, "Invalid thread destroyed ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

                    NvTrace_ThreadEvent_ThreadDestroyed threadDestroyed;
                    memcpy(&threadDestroyed, record.pPayload, sizeof(threadDestroyed));
                    return consumer.AcceptSerializedRecord(Codec::ThreadEventEncoders::WriteThreadDestroyedRecord(consumer.m_recordWriter, threadDestroyed, record.uniqueThreadId));
                }
                default:
                    return true;
            }
        }

        static bool ConsumeThreadEvent_CudaApiTraceCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            ContinuousSnapshotExporter& consumer = *reinterpret_cast<ContinuousSnapshotExporter*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            CudaApiTraceProviderState& providerState = *static_cast<CudaApiTraceProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            CudaApiTraceObserver::ThreadEventRecordStorage recordStorage;
            const ThreadEventRecord* const pRecord = providerState.observer.TranslateThreadEventRecord(record, recordStorage);
            if (!pRecord)
            {
                return true;
            }

            if (pRecord->recordType == NvTrace_ReservedRecordType_RecordReference)
            {
                NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < sizeof(NvTrace_RecordReference), true, 50, "Invalid record reference ThreadEvent record (recordType=%u, payloadSize=%zu).", pRecord->recordType, pRecord->payloadSize);

                NvTrace_RecordReference reference;
                memcpy(&reference, pRecord->pPayload, sizeof(reference));

                const bool serialized = CudaApiTraceSerialization::SerializeReferencedStateEvent(providerState.observer, consumer.m_recordWriter, reference);
                return consumer.AcceptSerializedRecord(serialized);
            }

            const auto recordType = static_cast<NvTrace_CUDA_ApiTrace_RecordType>(pRecord->recordType);
            // The records written below ride the delivering thread event's own commit; committing
            // here would publish them before that record's consistency verdict.
            switch (recordType)
            {
                case NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA context-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA stream-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A, true, 50, "Invalid CUDA module-unloaded ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                case NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed:
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A, true, 50, "Invalid CUDA graph-exec-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);
                    break;
                default:
                    return consumer.AcceptSerializedRecord(Codec::CudaApiTraceEventEncoders::WriteApiTraceEventRecord(consumer.m_recordWriter, *pRecord));
            }

            const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CudaApiTrace>(consumer.m_recordWriter, recordType, pRecord->correlationId, pRecord->pPayload, pRecord->payloadSize);
            NVTRACE_LOG_ERR_IF(!written && !consumer.m_recordWriter.IsStorageBackpressured(), 50, "Failed WriteRecord() for CUDA object-destroyed record (recordType=%u).", static_cast<uint32_t>(recordType));
            return consumer.AcceptSerializedRecord(written);
        }

        static bool ConsumeThreadEvent_NvtxTraceCallback(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record)
        {
            ContinuousSnapshotExporter& consumer = *reinterpret_cast<ContinuousSnapshotExporter*>(pConsumer);
            ProcessState& processState = *reinterpret_cast<ProcessState*>(pProcessState);
            NvtxTraceProviderState& providerState = *static_cast<NvtxTraceProviderState*>(static_cast<ProviderState*>(pProviderState));
            consumer.m_recordWriter.SetProcessProviderRef(providerState, processState);

            NvtxTraceObserver::ThreadEventRecordStorage recordStorage;
            const ThreadEventRecord* const pRecord = providerState.observer.TranslateThreadEventRecord(record, recordStorage);
            if (!pRecord)
            {
                return true;
            }

            if (pRecord->recordType == NvTrace_ReservedRecordType_RecordReference)
            {
                NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < sizeof(NvTrace_RecordReference), true, 50, "Invalid record reference ThreadEvent record (recordType=%u, payloadSize=%zu).", pRecord->recordType, pRecord->payloadSize);

                NvTrace_RecordReference reference;
                memcpy(&reference, pRecord->pPayload, sizeof(reference));

                const bool serialized = NvtxTraceSerialization::SerializeReferencedStateEvent(providerState.observer, consumer.m_recordWriter, reference);
                return consumer.AcceptSerializedRecord(serialized);
            }

            const auto recordType = static_cast<NvTrace_NvtxTrace_RecordType>(pRecord->recordType);
            // The records written below ride the delivering thread event's own commit; committing
            // here would publish them before that record's consistency verdict.
            switch (recordType)
            {
                case NvTrace_NvtxTrace_RecordType_StackGroupDestroyed:
                {
                    NVTRACE_LOG_ERR_RETURN_IF(pRecord->pPayload == nullptr || pRecord->payloadSize < NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A, true, 50, "Invalid NVTX stack-group-destroyed ThreadEvent record (payloadSize=%zu).", pRecord->payloadSize);

                    const bool written = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::NvtxTrace>(consumer.m_recordWriter, recordType, pRecord->correlationId, pRecord->pPayload, pRecord->payloadSize);
                    NVTRACE_LOG_ERR_IF(!written && !consumer.m_recordWriter.IsStorageBackpressured(), 50, "Failed WriteRecord() for NVTX stack-group-destroyed record.");
                    return consumer.AcceptSerializedRecord(written);
                }
                case NvTrace_NvtxTrace_RecordType_Mark:
                case NvTrace_NvtxTrace_RecordType_RangePop:
                case NvTrace_NvtxTrace_RecordType_DroppedRangePush:
                case NvTrace_NvtxTrace_RecordType_DroppedRangePop:
                    return consumer.AcceptSerializedRecord(Codec::NvtxTraceEventEncoders::WriteThreadEventRecord(consumer.m_recordWriter, *pRecord));
                default:
                    NVTRACE_LOG_ERR(50, "Unknown NVTX Trace ThreadEvent record type %u.", pRecord->recordType);
                    return true;
            }
        }

    private:
        void AttachSystem(SystemState& systemState)
        {
            m_systemState.systemCreatedRecordLocation = TSF_INVALID_RECORD_LOCATION;

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteSystemCreatedRecord(m_recordWriter, systemState.systemInfo))
            {
                return;
            }
            m_recordWriter.Commit();
            m_systemState.systemCreatedRecordLocation = m_recordWriter.GetLastCommittedRecordLocation();
            // History DataBlocks carry no record locations: this record lives only in the first bank, so each
            // snapshot's attach DataBlock provides the system info location instead.
        }

        void AttachGpuPartition(GpuPartitionState& gpuPartitionState)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteGpuPartitionCreatedRecord(m_recordWriter, gpuPartitionState.luid, gpuPartitionState.gpuUuid, gpuPartitionState.creationTimestampTsc, gpuPartitionState.gpuPartitionInfo))
            {
                return;
            }
            m_recordWriter.Commit();
        }
        void AttachGpuPartitionDestroyed(GpuPartitionState& gpuPartitionState, uint64_t destructionTimestampTsc)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteGpuPartitionDestroyedRecord(m_recordWriter, gpuPartitionState.luid, destructionTimestampTsc))
            {
                return;
            }
            m_recordWriter.Commit();
        }


        void AttachProcess(ProcessState& processState)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProcessCreatedRecord(m_recordWriter, processState.luid, processState.namespacePid, processState.globalPid, processState.pidNamespaceId, processState.path.c_str(), processState.creationTimestampTsc))
            {
                return;
            }
            m_recordWriter.Commit();
        }

        void AttachProcessDestroyed(ProcessState& processState, uint64_t destructionTimestampTsc)
        {
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProcessDestroyedRecord(m_recordWriter, processState.luid, destructionTimestampTsc))
            {
                return;
            }
            m_recordWriter.Commit();
        }

        template <Codec::ProviderTypeId TProviderTypeId, typename TProviderState>
        bool TryConsumeOnAttachRecords(TProviderState& providerState)
        {
            const size_t previousRecordCount = providerState.onAttachRecordLocations.size();

            providerState.observer.ConsumeOnAttachDataBlobRecords([&](uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) -> bool {
                const bool written = Codec::EventEncoders::WriteRecord<TProviderTypeId>(m_recordWriter, recordType, NVTRACE_NO_CORRELATION_ID, pPayload, payloadSize);
                if (!written)
                {
                    if (!m_recordWriter.IsStorageBackpressured())
                    {
                        NVTRACE_LOG_ERR(50, "Failed to write %s on-attach record (providerLuid=%" PRIu64 " recordType=%" PRIu32 ").", Codec::ToCString(TProviderTypeId), providerState.luid, recordType);
                    }
                    return AcceptSerializedRecord(false);
                }

                m_recordWriter.Commit();
                providerState.onAttachRecordLocations.push_back(m_recordWriter.GetLastCommittedRecordLocation());
                return true;
            });

            return previousRecordCount != providerState.onAttachRecordLocations.size();
        }

        template <typename TProviderState>
        void EmitProviderAttachedRecord(TProviderState& providerState)
        {
            if (!Codec::NvTraceServiceEncoders::WriteProviderAttachedRecord(m_recordWriter, providerState.luid, providerState.onAttachRecordLocations))
            {
                return;
            }

            m_recordWriter.Commit();
        }

        void AttachProvider_CpuClockCalibration(CpuClockCalibrationProviderState& providerState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, NVTRACE_NIL_LUID, NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid, "CPU Clock Calibration"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetSystemProviderRef(providerState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::CpuClockCalibration>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, "GPU Clock Calibration"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuClockCalibration>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_PmCounters_ProviderTypeUuid, "GPU PM Counters"))
            {
                return;
            }
            m_recordWriter.Commit();

            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuPmCounters>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_HwTrace_ProviderTypeUuid, "GPU HW Trace"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuHwTrace>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();
            providerState.onAttachRecordLocations.reserve(providerState.observer.GetOnAttachRecordsTotalCount());

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_GPU_GrCtxTrace_ProviderTypeUuid, "GPU GR Context Trace"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetGpuProviderRef(providerState, parentState);
            TryConsumeOnAttachRecords<Codec::ProviderTypeId::GpuGrCtxTrace>(providerState);
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
        }

        void AttachProvider_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();

            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_CUDA_ApiTrace_ProviderTypeUuid, "CUDA API Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            // The on-attach records lead every other state record: the DataModel needs the devices to
            // place the resources that follow under them, and a partial load reaches them through the
            // ProviderAttached record alone.
            CudaApiTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_recordWriter, [&] {
                providerState.onAttachRecordLocations.push_back(m_recordWriter.GetLastCommittedRecordLocation());
            });

            // CUDA keeps writing provider-owned records after ProviderAttached, which the null
            // provider ref carries, so the process provider ref is re-set behind it.
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            CudaApiTraceSerialization::SerializeStateRecords(providerState.observer, m_recordWriter);
        }

        void AttachProvider_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;
            providerState.onAttachRecordLocations.clear();

            const auto configView = providerState.observer.GetProviderConfigurationFlatBufferView();
            const auto recordVersionInfoSpan = providerState.observer.GetRecordVersionInfoByteSpan();
            const uint32_t recordVersionInfoMinorVersion = providerState.observer.GetRecordVersionInfo().recordVersionInfoMinorVersion;
            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_NvtxTrace_ProviderTypeUuid, "NVTX Trace", configView.pData, configView.size, recordVersionInfoSpan.pData, recordVersionInfoSpan.size, recordVersionInfoMinorVersion))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            NvtxTraceSerialization::SerializeOnAttachRecords(providerState.observer, m_recordWriter, [&] {
                providerState.onAttachRecordLocations.push_back(m_recordWriter.GetLastCommittedRecordLocation());
            });

            // NVTX keeps writing provider-owned records after ProviderAttached, which the null
            // provider ref carries, so the process provider ref is re-set behind it.
            m_recordWriter.SetNullProviderRef();
            EmitProviderAttachedRecord(providerState);
            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            NvtxTraceSerialization::SerializeStateRecords(providerState.observer, m_recordWriter);
        }

        void OnThreadEventSourceProviderBound(ThreadEventProviderState&, ProviderState& providerState)
        {
            providerState.threadEventProviderMappingRefDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            if (!m_isConsuming || !providerState.hasProviderCreatedRecord)
            {
                return;
            }

            const auto itrProcessState = m_systemState.processes.find(providerState.processLuid);
            if (itrProcessState != m_systemState.processes.end())
            {
                m_recordWriter.SetProcessProviderRef(providerState, itrProcessState->second);
            }
        }

        void AttachProvider_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState)
        {
            providerState.hasProviderCreatedRecord = false;

            m_recordWriter.SetNullProviderRef();
            if (!Codec::NvTraceServiceEncoders::WriteProviderCreatedRecord(m_recordWriter, providerState.luid, parentState.luid, NvTrace_ThreadEvent_ProviderTypeUuid, "Thread Event"))
            {
                return;
            }
            m_recordWriter.Commit();
            providerState.hasProviderCreatedRecord = true;

            Super::ConsumeThreadEventSourceProviderRegistrations(parentState, providerState, ReadCpuTscTimestamp());

            m_recordWriter.SetProcessProviderRef(providerState, parentState);

            ThreadEventSerialization::SerializeThreadStateEvents(providerState.observer, m_recordWriter);
        }

        void AttachExistingObjects()
        {
            // Attach System created event
            if (m_systemState.pSystemInfoMemory)
            {
                AttachSystem(m_systemState);
            }

            // Attach System providers
            {
                if (m_systemState.cpuClockCalibrationProvider.IsInitialized())
                {
                    AttachProvider_CpuClockCalibration(m_systemState.cpuClockCalibrationProvider);
                }
            }

            // Attach GpuPartitions and their providers
            for (auto& [_, gpuPartitionState] : m_systemState.gpuPartitions)
            {
                AttachGpuPartition(gpuPartitionState);
                if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                {
                    AttachProvider_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                {
                    TryPrepareGpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                {
                    TryPrepareGpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                }
                if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                {
                    TryPrepareGpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                }
            }

            // Attach Processes and their providers
            for (auto& [_, processState] : m_systemState.processes)
            {
                AttachProcess(processState);
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    AttachProvider_ThreadEvent(threadEventProviderEntry.second, processState);
                }
                if (processState.cudaApiTraceProvider.IsInitialized())
                {
                    AttachProvider_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                }
                if (processState.nvtxTraceProvider.IsInitialized())
                {
                    AttachProvider_NvtxTrace(processState.nvtxTraceProvider, processState);
                }
                for (auto& threadEventProviderEntry : processState.threadEventProviders)
                {
                    TryPrepareThreadEvent(threadEventProviderEntry.second, processState, ReadCpuTscTimestamp());
                }
            }
        }

    private:
        using Super::ConvertTscTickToUtcNsecForConsume;
        using Super::UpdateConsumeCpuClockCalibration;

        bool BuildWaitMultiplexer()
        {
            constexpr uint32_t eventCount = 5;
            bool succeeded = m_waitMultiplexer.Initialize(eventCount);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::Initialize failed (maxObjectCount=%" PRIu32 ")", eventCount);
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::Command, m_consumeThreadCommandPendingEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::AddObject failed (identifier=Command)");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::RouterNotification, m_pRouterObserverClient->GetWaitForPendingNotificationHandle());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::AddObject failed (identifier=RouterNotification)");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::ConsumeData, m_consumeTimer.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::AddObject failed (identifier=ConsumeData)");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::StorageWriteCompleted, m_storageWriteCompletedEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::AddObject failed (identifier=StorageWriteCompleted)");
            succeeded = m_waitMultiplexer.AddObject(EventIdentifier::ReturnedBank, m_returnedBankEvent.Desc());
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::AddObject failed (identifier=ReturnedBank)");
            return true;
        }

        bool QueueReadyBank(size_t bankIndex)
        {
            if (bankIndex == Detail::ContinuousSnapshotStorageState::INVALID_BANK_INDEX)
            {
                return true;
            }

            {
                std::lock_guard<std::mutex> lock(m_exportBanksMutex);
                NVTRACE_LOG_ERR_RETURN_IF(m_exportBanks.size() <= m_exportBankCount, false, 50, "ContinuousSnapshotExporter export-bank queue overflowed");
                NVTRACE_LOG_ERR_RETURN_IF(!m_storageState.TransferReadyBankToExport(bankIndex), false, 50, "ContinuousSnapshotExporter failed to transfer ownership of a ready bank");
                if (!m_attachWorker.IsRunning())
                {
                    m_consumeThreadCommandSucceeded = false;
                    const bool reclaimed = m_storageState.ReclaimReturnedBank(bankIndex);
                    NVTRACE_LOG_ERR_IF(!reclaimed, 50, "ContinuousSnapshotExporter failed to reclaim a bank after its export worker exited");
                    return false;
                }
                const size_t tailIndex = (m_exportBankHead + m_exportBankCount) % m_exportBanks.size();
                m_exportBanks[tailIndex] = bankIndex;
                m_exportBankCount += 1;
            }
            const bool succeeded = m_attachWorker.SignalJobReady();
            if (!succeeded)
            {
                m_consumeThreadCommandSucceeded = false;
            }
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "ContinuousSnapshotExporter failed to signal an export job");
            return succeeded;
        }

        void HandleStorageCompletions()
        {
            {
                std::lock_guard<std::mutex> lock(m_storageCompletionsMutex);
                m_storageCompletionScratch.assign(m_storageCompletions.begin(), m_storageCompletions.end());
                m_storageCompletions.clear();
            }
            for (const auto& completion : m_storageCompletionScratch)
            {
                size_t readyBankIndex = Detail::ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
                if (m_storageState.ApplyStorageCompletion(completion, readyBankIndex))
                {
                    QueueReadyBank(readyBankIndex);
                }
            }
            m_storageCompletionScratch.clear();
            TryCompleteStopCommand();
        }

        void HandleReturnedBanks()
        {
            std::array<size_t, 2> returnedBanks = {};
            size_t returnedBankCount = 0;
            {
                std::lock_guard<std::mutex> lock(m_returnedBanksMutex);
                returnedBankCount = m_returnedBankCount;
                for (size_t returnedBankOffset = 0; returnedBankOffset < returnedBankCount; ++returnedBankOffset)
                {
                    returnedBanks[returnedBankOffset] = m_returnedBanks[(m_returnedBankHead + returnedBankOffset) % m_returnedBanks.size()];
                }
                m_returnedBankHead = 0;
                m_returnedBankCount = 0;
            }
            for (size_t returnedBankIndex = 0; returnedBankIndex < returnedBankCount; ++returnedBankIndex)
            {
                m_storageState.ReclaimReturnedBank(returnedBanks[returnedBankIndex]);
            }
            TryCompleteStopCommand();
        }

        void TryCompleteStopCommand()
        {
            if (!m_stopCommandPending || !m_storageState.IsDrained())
            {
                return;
            }
            m_stopCommandPending = false;
            const bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
            NVTRACE_LOG_ERR_IF(!succeeded, 50, "ContinuousSnapshotExporter failed to signal completion of a drained stop command");
        }

        bool EnterRouterlessMode()
        {
            bool succeeded = m_waitMultiplexer.RemoveLastSignaledObject();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "ContinuousSnapshotExporter: WaitMultiplexer::RemoveLastSignaledObject failed");
            m_pRouterObserverClient.Reset();
            return true;
        }

        void ConsumeData(uint64_t readEndpointTsc = ReadCpuTscTimestamp())
        {
            if (m_changeoverPending && !TryCompletePendingChangeover(readEndpointTsc))
            {
                return;
            }

            bool changeoverRequested = false;
            bool immediateRetryRequested = false;
            {
                const uint64_t consumeDataBeginTsc = ReadCpuTscTimestamp();
                auto consumeDataDurationGuard = ScopeExitGuard([this, consumeDataBeginTsc]() {
                    const uint64_t consumeDataEndTsc = ReadCpuTscTimestamp();
                    const TscRtcCalibrationPoint calibrationPoint = this->GetLatestCpuClockCalibrationPoint();
                    const uint64_t durationNsec = static_cast<uint64_t>(static_cast<double>(consumeDataEndTsc - consumeDataBeginTsc) * calibrationPoint.tscToUtcFactor);
                    m_storageState.RecordConsumeDataDuration(durationNsec);
                });

                m_recordWriter.BeginConsumeCycle();
                auto consumeCycleGuard = ScopeExitGuard([this]() { m_recordWriter.EndConsumeCycle(); });

                UpdateConsumeCpuClockCalibration();
                const uint64_t readEndpointUtcNsec = ConvertTscTickToUtcNsecForConsume(readEndpointTsc);
                PurgeExpiredZombieParents(readEndpointUtcNsec);
                const uint64_t recordTimestampLimitUtcNsec = m_recordTimestampLimitUtcNsec.load(std::memory_order_acquire);
                const uint64_t recordTimestampEndpointUtcNsec = std::max(m_recordTimestampLowerBoundUtcNsec, std::min(readEndpointUtcNsec, recordTimestampLimitUtcNsec));

                m_recordTimestampUpperBoundUtcNsec = recordTimestampEndpointUtcNsec;
                if (m_dataBlockWriter.IsInitialized())
                {
                    m_dataBlockWriter.SetRecordTimestampUpperBoundUtcNsec(m_recordTimestampUpperBoundUtcNsec);
                }

                Super::ConsumeData(readEndpointTsc);

                const bool storageBackpressured = m_recordWriter.IsStorageBackpressured();
                if (!storageBackpressured)
                {
                    m_recordTimestampLowerBoundUtcNsec = recordTimestampEndpointUtcNsec;
                }
                PublishLiveStatistics(recordTimestampEndpointUtcNsec);

                if (m_recordWriter.IsHistoryBankFull())
                {
                    changeoverRequested = true;
                    immediateRetryRequested = true;
                }
                else if (!storageBackpressured && m_isConsuming)
                {
                    // Once StopAtUtc's requested endpoint has passed, keep the current
                    // bank open throughout the grace period. Stop seals it exactly once
                    // after late-arriving records through that endpoint are consumed.
                    const bool recordTimestampLimitReached = recordTimestampLimitUtcNsec <= readEndpointUtcNsec;
                    const bool snapshotDeadlineReached = !recordTimestampLimitReached && m_nextSnapshotDeadlineTsc <= readEndpointTsc;
                    changeoverRequested = snapshotDeadlineReached || ShouldChangeoverAfterConsume();
                }
            }

            if (changeoverRequested)
            {
                m_changeoverPending = true;
                if (TryCompletePendingChangeover(readEndpointTsc) && immediateRetryRequested)
                {
                    const bool scheduled = ScheduleImmediateConsumeIteration();
                    NVTRACE_LOG_ERR_IF(!scheduled, 50, "ContinuousSnapshotExporter failed to schedule an immediate retry after filling a history bank");
                }
            }
        }

        bool TryCompletePendingChangeover(uint64_t readEndpointTsc)
        {
            if (!m_changeoverPending || !TryChangeover(readEndpointTsc, m_recordTimestampLowerBoundUtcNsec))
            {
                return !m_changeoverPending;
            }
            m_changeoverPending = false;
            return true;
        }

        bool TryChangeover(uint64_t readEndpointTsc, uint64_t cutoffTimestampUtcNsec)
        {
            if (!m_storageState.CanChangeover())
            {
                return false;
            }

            uint64_t lastDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            bool hadDataBlock = false;
            const bool checkpointSucceeded = m_recordWriter.Checkpoint(lastDataBlockSequenceNumber, hadDataBlock);
            size_t readyBankIndex = Detail::ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
            const bool changeoverSucceeded = m_storageState.SealAndSwitch(
                lastDataBlockSequenceNumber,
                cutoffTimestampUtcNsec,
                hadDataBlock,
                checkpointSucceeded,
                readyBankIndex);
            if (changeoverSucceeded)
            {
                QueueReadyBank(readyBankIndex);
                m_nextSnapshotDeadlineTsc = readEndpointTsc > UINT64_MAX - m_snapshotCadenceTsc
                                              ? UINT64_MAX
                                              : readEndpointTsc + m_snapshotCadenceTsc;
            }
            return changeoverSucceeded;
        }

        bool ShouldChangeoverAfterConsume() const
        {
            const uint64_t chunkSize = m_databaseWriter.ChunkSize();
            uint64_t availableBytes = static_cast<uint64_t>(m_storageState.RemainingDataBlockCount()) * chunkSize;
            const size_t currentChunkCommittedBytes = m_recordWriter.CurrentChunkCommittedByteCount();
            if (m_dataBlockWriter.IsInitialized() && currentChunkCommittedBytes < chunkSize)
            {
                availableBytes += chunkSize - currentChunkCommittedBytes;
            }
            uint64_t predictedNextConsumeBytes = 0;
            if (m_latestDataRateBytesPerSec != 0 && m_lastConsumeIntervalNsec != 0)
            {
                predictedNextConsumeBytes = m_latestDataRateBytesPerSec > UINT64_MAX / m_lastConsumeIntervalNsec
                                              ? UINT64_MAX
                                              : m_latestDataRateBytesPerSec * m_lastConsumeIntervalNsec / 1'000'000'000ULL;
            }
            const uint64_t overflowPreventionReserveBytes = m_configuration.overflowPreventionReserveSizeBytesPerBank;
            const uint64_t requiredHeadroomBytes = predictedNextConsumeBytes > UINT64_MAX - overflowPreventionReserveBytes
                                                     ? UINT64_MAX
                                                     : predictedNextConsumeBytes + overflowPreventionReserveBytes;
            return availableBytes <= requiredHeadroomBytes;
        }

        void PublishLiveStatistics(uint64_t sampleTimestampUtcNsec)
        {
            const uint64_t totalCommittedBytes = m_recordWriter.CommittedByteCount();
            if (m_lastConsumeTimestampUtcNsec != 0 && m_lastConsumeTimestampUtcNsec < sampleTimestampUtcNsec)
            {
                m_lastConsumeIntervalNsec = sampleTimestampUtcNsec - m_lastConsumeTimestampUtcNsec;
            }
            m_lastConsumeTimestampUtcNsec = sampleTimestampUtcNsec;

            constexpr uint64_t rateSampleIntervalNsec = 100'000'000;
            if (m_lastRateSampleTimestampUtcNsec == 0)
            {
                m_lastRateSampleTimestampUtcNsec = sampleTimestampUtcNsec;
                m_lastRateSampleBytes = totalCommittedBytes;
            }
            else if (m_lastRateSampleTimestampUtcNsec + rateSampleIntervalNsec <= sampleTimestampUtcNsec)
            {
                const uint64_t sampleDurationNsec = sampleTimestampUtcNsec - m_lastRateSampleTimestampUtcNsec;
                const uint64_t byteDelta = totalCommittedBytes - m_lastRateSampleBytes;
                m_latestDataRateBytesPerSec = byteDelta > UINT64_MAX / 1'000'000'000ULL
                                                ? UINT64_MAX
                                                : byteDelta * 1'000'000'000ULL / sampleDurationNsec;
                m_lastRateSampleTimestampUtcNsec = sampleTimestampUtcNsec;
                m_lastRateSampleBytes = totalCommittedBytes;
            }

            std::unique_lock<std::mutex> lock(m_liveStatisticsMutex, std::try_to_lock);
            if (lock.owns_lock())
            {
                m_liveStatistics.sampleTimestampUtcNsec = sampleTimestampUtcNsec;
                m_liveStatistics.totalCommittedBytes = totalCommittedBytes;
                m_liveStatistics.dataRateBytesPerSec = m_latestDataRateBytesPerSec;
            }
        }

        void ConsumeLoop()
        {
            auto exitGuard = ScopeExitGuard([this]() {
                m_consumeThreadExited = true;
                std::atomic_thread_fence(std::memory_order_release);

                bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                NVTRACE_LOG_ERR_IF(!succeeded, 50, "ConsumeLoop exit: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");
            });

            while (true)
            {
                EventIdentifier signaledIdentifier;
                if (!m_waitMultiplexer.Wait(signaledIdentifier))
                {
                    NVTRACE_LOG_ERR(50, "ConsumeLoop: WaitMultiplexer::Wait returned false (see preceding WaitMultiplexer log for errno or Win32 error).");
                    return;
                }

                switch (signaledIdentifier)
                {
                    case EventIdentifier::Command:
                    {
                        std::atomic_thread_fence(std::memory_order_acquire);
                        switch (m_consumeThreadCommand)
                        {
                            case ConsumeThreadCommand::Start:
                            {
                                const uint64_t startTimestampUtcNsec = m_observationStartUtcNsec != 0
                                                                         ? m_observationStartUtcNsec
                                                                         : ConvertTscTickToUtcNsec(ReadCpuTscTimestamp());
                                m_recordTimestampLowerBoundUtcNsec = startTimestampUtcNsec;
                                m_recordTimestampUpperBoundUtcNsec = startTimestampUtcNsec;
                                m_observationStartTsc = m_observationStartUtcNsec != 0
                                                          ? this->GetLatestCpuClockCalibrationPoint().ToTsc(m_observationStartUtcNsec)
                                                          : 0;
                                NVTRACE_LOG_ERR_RETURN_IF(
                                    !m_storageState.SetLiveBankBeginTimestamp(startTimestampUtcNsec),
                                    /*void*/,
                                    50,
                                    "ConsumeLoop Start: failed to initialize the live history interval");
                                ConfigureExistingObservationStarts();
                                m_isConsuming = true;

                                // Release the thread that called Start before Attaching
                                bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "ConsumeLoop Start: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");

                                m_recordWriter.BeginConsumeCycle();
                                {
                                    auto consumeCycleGuard = ScopeExitGuard([this]() { m_recordWriter.EndConsumeCycle(); });
                                    AttachExistingObjects();
                                }

                                const uint64_t snapshotCadenceBeginTsc = ReadCpuTscTimestamp();
                                m_nextSnapshotDeadlineTsc = snapshotCadenceBeginTsc > UINT64_MAX - m_snapshotCadenceTsc
                                                              ? UINT64_MAX
                                                              : snapshotCadenceBeginTsc + m_snapshotCadenceTsc;

                                if (!ScheduleImmediateConsumeIteration())
                                {
                                    return;
                                }

                                break;
                            }

                            case ConsumeThreadCommand::Stop:
                            {
                                // Ensure at least one iteration of data consumption is performed before stopping
                                {
                                    const auto dropUnboundGuard = Super::ScopeGuardDropUnboundThreadEvents();
                                    ConsumeData();
                                }
                                m_isConsuming = false;

                                uint64_t lastDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                                bool hadDataBlock = false;
                                const bool checkpointSucceeded = m_recordWriter.Checkpoint(lastDataBlockSequenceNumber, hadDataBlock);
                                size_t readyBankIndex = Detail::ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
                                m_consumeThreadCommandSucceeded = m_storageState.SealForStop(
                                    lastDataBlockSequenceNumber,
                                    m_recordTimestampLowerBoundUtcNsec,
                                    hadDataBlock,
                                    checkpointSucceeded,
                                    readyBankIndex);
                                m_consumeThreadCommandSucceeded = QueueReadyBank(readyBankIndex) && m_consumeThreadCommandSucceeded;
                                if (!m_consumeThreadCommandSucceeded)
                                {
                                    const bool succeeded = m_consumeThreadCommandCompletedEvent.Signal();
                                    NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "ConsumeLoop Stop failure: AutoResetEvent::Signal failed (consumeThreadCommandCompletedEvent)");
                                }
                                else
                                {
                                    m_stopCommandPending = true;
                                    TryCompleteStopCommand();
                                }

                                break;
                            }

                            case ConsumeThreadCommand::Terminate:
                                return;
                        }

                        break;
                    }

                    case EventIdentifier::RouterNotification:
                    {
                        bool succeeded = m_pRouterObserverClient->ReceiveNotificationIfPending([this](NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) {
                            return HandleRouterNotification(notificationType, pNotification);
                        });
                        if (!succeeded)
                        {
                            NVTRACE_LOG_WRN(50, "ConsumeLoop: Router observer connection lost");
                            if (!EnterRouterlessMode())
                            {
                                NVTRACE_LOG_ERR(50, "ConsumeLoop: Failed to continue without router notifications.");
                                return;
                            }
                            NVTRACE_LOG_WRN(50, "ConsumeLoop: Continuing in routerless mode with known provider buffers.");
                        }

                        break;
                    }

                    case EventIdentifier::ConsumeData:
                    {
                        if (m_isConsuming)
                        {
                            const bool nextIterationScheduled = ScheduleNextConsumeIteration();
                            ConsumeData();
                            if (!nextIterationScheduled)
                            {
                                return;
                            }
                        }

                        break;
                    }

                    case EventIdentifier::StorageWriteCompleted:
                    {
                        HandleStorageCompletions();
                        break;
                    }

                    case EventIdentifier::ReturnedBank:
                    {
                        HandleReturnedBanks();
                        if (m_isConsuming)
                        {
                            ConsumeData();
                        }
                        break;
                    }
                }
            }
        }

        void TerminateConsumeThread() noexcept
        {
            if (m_consumeThread.joinable())
            {
                m_consumeThreadCommand = ConsumeThreadCommand::Terminate;
                std::atomic_thread_fence(std::memory_order_release);

                bool succeeded = m_consumeThreadCommandPendingEvent.Signal();
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, /*void*/, 50, "TerminateConsumeThread: AutoResetEvent::Signal failed (consumeThreadCommandPendingEvent)");

                m_consumeThread.join();
            }
        }

    private:
        class TsfRecordWriter
        {
        public:
            TsfRecordWriter(ContinuousSnapshotExporter& consumer)
                : m_consumer(consumer)
            {
            }

            /// void fnWritePayload(TInlinePayload* pInlinePayload, TMajorPayload* pPayload);
            template <typename TFN_WritePayload>
            bool WriteRecord(uint32_t recordType, uint64_t correlationId, size_t majorPayloadSize, TFN_WritePayload&& fnWritePayload)
            {
                if (m_storageBackpressuredThisConsumeCycle)
                {
                    ++m_backpressuredRecords;
                    return false;
                }

                if (!m_chunkWriter.HasSpaceForRecord(majorPayloadSize))
                {
                    if (m_consumer.m_databaseWriter.MaxMajorPayloadSize() < majorPayloadSize)
                    {
                        return false;
                    }

                    if (!AcquireNewChunk())
                    {
                        m_storageBackpressuredThisConsumeCycle = true;
                        ++m_backpressuredRecords;
                        return false;
                    }
                }

                m_chunkWriter.WriteRecord(m_providerRefIndex, recordType, correlationId, majorPayloadSize, std::forward<TFN_WritePayload>(fnWritePayload));
                return true;
            }

            /// void fnWritePayload(TInlinePayload* pInlinePayload, TMajorPayload* pPayload);
            template <typename TFN_WritePayload>
            bool WriteRecord(uint32_t recordType, TFN_WritePayload&& fnWritePayload)
            {
                using MajorPayloadType = typename GetWritePayloadTypes<TFN_WritePayload>::MajorPayloadType;
                return WriteRecord(recordType, sizeof(MajorPayloadType), std::forward<TFN_WritePayload>(fnWritePayload));
            }

            void BeginConsumeCycle()
            {
                m_storageBackpressuredThisConsumeCycle = false;
                m_historyBankFullThisConsumeCycle = false;
                m_backpressuredRecords = 0;
            }

            void EndConsumeCycle()
            {
                if (m_backpressuredRecords)
                {
                    m_consumer.m_storageState.RecordBackpressuredRecords(m_backpressuredRecords);
                }
            }

            bool IsStorageBackpressured() const
            {
                return m_storageBackpressuredThisConsumeCycle;
            }

            bool IsHistoryBankFull() const
            {
                return m_historyBankFullThisConsumeCycle;
            }

            uint64_t CommittedByteCount() const
            {
                return m_totalCommittedBytes + (m_chunkWriter.IsInitialized() ? m_chunkWriter.CommittedByteCount() : 0);
            }

            size_t CurrentChunkCommittedByteCount() const
            {
                return m_chunkWriter.IsInitialized() ? m_chunkWriter.CommittedByteCount() : 0;
            }

            void Commit()
            {
                m_chunkWriter.Commit();
            }

            void Rollback()
            {
                m_chunkWriter.Rollback();
            }

            bool Checkpoint(uint64_t& out_lastDataBlockSequenceNumber, bool& out_hadDataBlock)
            {
                if (m_chunkWriter.IsInitialized())
                {
                    m_totalCommittedBytes += m_chunkWriter.CommittedByteCount();
                }
                m_chunkWriter.Reset();
                out_hadDataBlock = m_consumer.m_dataBlockWriter.IsInitialized();
                out_lastDataBlockSequenceNumber = out_hadDataBlock
                                                    ? m_consumer.m_dataBlockWriter.DataBlockSequenceNumber()
                                                    : TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                if (!out_hadDataBlock)
                {
                    return true;
                }
                return m_consumer.m_databaseWriter.FinalizeDataBlock(m_consumer.m_dataBlockWriter);
            }

            void Reset() noexcept
            {
                m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                m_pProcessState = nullptr;
                m_pGpuPartitionState = nullptr;
                m_pProviderState = nullptr;
                m_chunkWriter.Reset();
                m_storageBackpressuredThisConsumeCycle = false;
                m_historyBankFullThisConsumeCycle = false;
                m_backpressuredRecords = 0;
                m_totalCommittedBytes = 0;
            }

            uint64_t GetLastCommittedRecordLocation() const
            {
                return m_consumer.m_databaseWriter.EncodeRecordLocation(m_consumer.m_dataBlockWriter.DataBlockSequenceNumber(), m_chunkWriter.ChunkIndex(), m_chunkWriter.GetLastCommittedRecordIndex());
            }

            void SetNullProviderRef()
            {
                m_pProviderState = nullptr;
                m_pGpuPartitionState = nullptr;
                m_pProcessState = nullptr;
                m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            }
            void SetSystemProviderRef(ProviderState& providerState)
            {
                m_pProviderState = &providerState;
                m_pGpuPartitionState = nullptr;
                m_pProcessState = nullptr;
                WriteProviderRef();
            }
            void SetGpuProviderRef(ProviderState& providerState, GpuPartitionState& gpuPartitionState)
            {
                m_pProviderState = &providerState;
                m_pGpuPartitionState = &gpuPartitionState;
                m_pProcessState = nullptr;
                WriteProviderRef();
            }
            void SetProcessProviderRef(ProviderState& providerState, ProcessState& processState)
            {
                m_pProviderState = &providerState;
                m_pGpuPartitionState = nullptr;
                m_pProcessState = &processState;
                WriteProviderRef();
            }

        private:
            bool AcquireNewDataBlock()
            {
                if (m_consumer.m_dataBlockWriter.IsInitialized())
                {
                    m_totalCommittedBytes += m_consumer.m_databaseWriter.ChunkSize();
                    m_consumer.m_databaseWriter.FinalizeDataBlock(m_consumer.m_dataBlockWriter);
                }

                if (!m_consumer.m_storageState.CanReserveDataBlock())
                {
                    m_historyBankFullThisConsumeCycle = true;
                    return false;
                }
                if (!m_consumer.m_databaseWriter.AcquireDataBlock(m_consumer.m_dataBlockWriter, m_consumer.m_recordTimestampLowerBoundUtcNsec, m_consumer.m_recordTimestampUpperBoundUtcNsec))
                {
                    return false;
                }
                size_t bankIndex = Detail::ContinuousSnapshotStorageState::INVALID_BANK_INDEX;
                size_t historySlotIndex = 0;
                if (!m_consumer.m_storageState.ReserveDataBlock(m_consumer.m_dataBlockWriter.DataBlockSequenceNumber(), bankIndex, historySlotIndex))
                {
                    m_consumer.m_databaseWriter.RecycleDataBlock(m_consumer.m_dataBlockWriter);
                    m_historyBankFullThisConsumeCycle = true;
                    return false;
                }
                const uint8_t writeSlotIndex = m_consumer.m_dataBlockWriter.WriteSlotIndex();
                if (!m_consumer.m_databaseWriter.StorageWriter().SetWriteTarget(
                        writeSlotIndex,
                        bankIndex,
                        historySlotIndex,
                        m_consumer.m_dataBlockWriter.DataBlockSequenceNumber()))
                {
                    m_consumer.m_storageState.CancelDataBlockReservation(bankIndex, historySlotIndex);
                    m_consumer.m_databaseWriter.RecycleDataBlock(m_consumer.m_dataBlockWriter);
                    return false;
                }

                WriteProviderRef();
                return true;
            }

            bool AcquireNewChunk()
            {
                m_chunkWriter.Reset();

                if (!m_consumer.m_dataBlockWriter.IsInitialized() || !m_consumer.m_dataBlockWriter.HasSpaceForChunk())
                {
                    if (!AcquireNewDataBlock())
                    {
                        return false;
                    }
                }

                m_consumer.m_dataBlockWriter.AcquireChunk(m_chunkWriter);
                return true;
            }

            void WriteProviderRef()
            {
                if (!m_consumer.m_dataBlockWriter.IsInitialized() || !m_pProviderState)
                {
                    return;
                }

                const uint64_t dataBlockSequenceNumber = m_consumer.m_dataBlockWriter.DataBlockSequenceNumber();

                // Determine which parent refs need to be written
                const bool requiresNewProviderRef = m_pProviderState->dataBlockSequenceNumber != dataBlockSequenceNumber;
                const bool requiresNewGpuPartitionRef = m_pGpuPartitionState && m_pGpuPartitionState->dataBlockSequenceNumber != dataBlockSequenceNumber;
                const bool requiresNewProcessRef = m_pProcessState && m_pProcessState->dataBlockSequenceNumber != dataBlockSequenceNumber;

                // ThreadEventProvider
                ThreadEventProviderState* const pThreadEventProviderState = m_pProcessState ? m_pProcessState->FindThreadEventProvider(m_pProviderState->threadEventProviderLuid) : nullptr;
                const bool hasThreadEventProviderMappingRef = pThreadEventProviderState && m_pProviderState->threadEventSourceProviderIndex != ProviderState::kInvalidThreadEventSourceProviderIndex;
                const bool requiresNewThreadEventProviderMappingRef = hasThreadEventProviderMappingRef && m_pProviderState->threadEventProviderMappingRefDataBlockSequenceNumber != dataBlockSequenceNumber;
                const bool requiresNewThreadEventProviderRef = requiresNewThreadEventProviderMappingRef && pThreadEventProviderState->dataBlockSequenceNumber != dataBlockSequenceNumber;

                if (!requiresNewProviderRef && !requiresNewThreadEventProviderMappingRef)
                {
                    m_providerRefIndex = m_pProviderState->providerRefIndex;
                    return;
                }
                const uint32_t requiredProviderRefCount = (requiresNewProviderRef ? 1 : 0) + (requiresNewThreadEventProviderRef ? 1 : 0);

                // Check if there's space for all required refs
                // clang-format off
                if (!m_consumer.m_dataBlockWriter.HasSpaceForProviderRef(requiredProviderRefCount) ||
                    (requiresNewGpuPartitionRef && !m_consumer.m_dataBlockWriter.HasSpaceForGpuPartitionRef()) ||
                    (requiresNewProcessRef && !m_consumer.m_dataBlockWriter.HasSpaceForProcessRef()) ||
                    (requiresNewThreadEventProviderMappingRef && !m_consumer.m_dataBlockWriter.HasSpaceForThreadEventProviderMappingRef()))
                // clang-format on
                {
                    if (m_chunkWriter.IsInitialized())
                    {
                        m_totalCommittedBytes += m_chunkWriter.CommittedByteCount();
                    }
                    m_chunkWriter.Reset();
                    m_consumer.m_databaseWriter.FinalizeDataBlock(m_consumer.m_dataBlockWriter);
                    m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                    return;
                }

                // Write parent refs if needed
                if (requiresNewGpuPartitionRef)
                {
                    m_consumer.m_dataBlockWriter.WriteGpuPartitionRef(m_pGpuPartitionState->luid, TSF_INVALID_RECORD_LOCATION);
                    m_pGpuPartitionState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                if (requiresNewProcessRef)
                {
                    m_consumer.m_dataBlockWriter.WriteProcessRef(m_pProcessState->luid, TSF_INVALID_RECORD_LOCATION);
                    m_pProcessState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                if (requiresNewThreadEventProviderRef)
                {
                    pThreadEventProviderState->providerRefIndex = m_consumer.m_dataBlockWriter.WriteProviderRef(pThreadEventProviderState->luid, TSF_INVALID_RECORD_LOCATION, TSF_INVALID_RECORD_LOCATION);
                    pThreadEventProviderState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                if (requiresNewThreadEventProviderMappingRef)
                {
                    m_consumer.m_dataBlockWriter.WriteThreadEventProviderMappingRef(pThreadEventProviderState->luid, m_pProviderState->luid, m_pProviderState->threadEventSourceProviderIndex);
                    m_pProviderState->threadEventProviderMappingRefDataBlockSequenceNumber = dataBlockSequenceNumber;
                }

                if (requiresNewProviderRef)
                {
                    m_pProviderState->providerRefIndex = m_consumer.m_dataBlockWriter.WriteProviderRef(m_pProviderState->luid, TSF_INVALID_RECORD_LOCATION, TSF_INVALID_RECORD_LOCATION);
                    m_pProviderState->dataBlockSequenceNumber = dataBlockSequenceNumber;
                }
                m_providerRefIndex = m_pProviderState->providerRefIndex;
            }

        private:
            ContinuousSnapshotExporter& m_consumer;
            TraceStreamChunkWriter m_chunkWriter;

            ProviderState* m_pProviderState = nullptr;
            GpuPartitionState* m_pGpuPartitionState = nullptr;
            ProcessState* m_pProcessState = nullptr;

            uint32_t m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            bool m_storageBackpressuredThisConsumeCycle = false;
            bool m_historyBankFullThisConsumeCycle = false;
            uint64_t m_backpressuredRecords = 0;
            uint64_t m_totalCommittedBytes = 0;
        };

    private:
        using Super::m_consumeTimer;
        using Super::m_pRouterObserverClient;
        using Super::m_systemState;

        Detail::ContinuousSnapshotStorageState m_storageState;
        Detail::ContinuousSnapshotAttachWorker m_attachWorker;
        TraceStreamDatabaseWriter<Detail::ContinuousSnapshotStorageWriter> m_databaseWriter;
        TraceStreamDataBlockWriter m_dataBlockWriter;
        Configuration m_configuration;

        uint64_t m_recordTimestampLowerBoundUtcNsec = 0;
        uint64_t m_observationStartTsc = 0;
        uint64_t m_observationStartUtcNsec = 0;
        uint64_t m_recordTimestampUpperBoundUtcNsec = 0;
        uint64_t m_snapshotCadenceTsc = 0;
        uint64_t m_historyDurationNsec = 0;
        uint64_t m_nextSnapshotDeadlineTsc = 0;
        uint64_t m_lastConsumeTimestampUtcNsec = 0;
        uint64_t m_lastConsumeIntervalNsec = 0;
        uint64_t m_lastRateSampleTimestampUtcNsec = 0;
        uint64_t m_latestDataRateBytesPerSec = 0;
        uint64_t m_lastRateSampleBytes = 0;
        std::atomic<uint64_t> m_recordTimestampLimitUtcNsec = UINT64_MAX;

        ZombieParentQueue m_zombieParentQueue;
        size_t m_zombieObjectBytes = 0;
        size_t m_maxZombieObjectBytes = 0;
        bool m_isReplayingZombieDestroy = false;

        mutable std::mutex m_liveStatisticsMutex;
        LiveStatistics m_liveStatistics;

        TsfRecordWriter m_recordWriter;

        enum class ConsumeThreadCommand
        {
            Start,
            Stop,
            Terminate
        };
        ConsumeThreadCommand m_consumeThreadCommand = ConsumeThreadCommand::Stop;
        AutoResetEvent m_consumeThreadCommandPendingEvent;
        AutoResetEvent m_consumeThreadCommandCompletedEvent;
        AutoResetEvent m_storageWriteCompletedEvent;
        AutoResetEvent m_returnedBankEvent;

        std::mutex m_storageCompletionsMutex;
        std::vector<Detail::ContinuousSnapshotStorageState::StorageCompletion> m_storageCompletions;
        std::vector<Detail::ContinuousSnapshotStorageState::StorageCompletion> m_storageCompletionScratch;
        std::mutex m_exportBanksMutex;
        std::array<size_t, 2> m_exportBanks = {};
        size_t m_exportBankHead = 0;
        size_t m_exportBankCount = 0;
        std::mutex m_returnedBanksMutex;
        std::array<size_t, 2> m_returnedBanks = {};
        size_t m_returnedBankHead = 0;
        size_t m_returnedBankCount = 0;

        enum class EventIdentifier
        {
            Command,
            RouterNotification,
            ConsumeData,
            StorageWriteCompleted,
            ReturnedBank
        };
        WaitMultiplexer m_waitMultiplexer;

        std::thread m_consumeThread;
        bool m_consumeThreadExited = false;
        bool m_consumeThreadCommandSucceeded = true;
        bool m_stopCommandPending = false;
        bool m_changeoverPending = false;

        bool m_isConsuming = false;
    };

} // namespace nv::trace
