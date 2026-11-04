/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>
#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>
#include <NvTrace/Target/Consumer/NvTraceStreamUtilities.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

namespace nv::trace {

    struct TsfDataBlockRange
    {
        size_t beginDataBlockIndex = 0;
        size_t endDataBlockIndex = 0;

        bool Empty() const noexcept
        {
            return beginDataBlockIndex >= endDataBlockIndex;
        }

        size_t Size() const noexcept
        {
            return Empty() ? 0 : endDataBlockIndex - beginDataBlockIndex;
        }
    };

    template <typename TStorageReader>
    class TraceStreamDatabaseReader
    {
    public:
        struct Options
        {
            uint8_t maxDataBlockLoadSlots = 16;
        };

        // Minimum of 4 load slots required, to be able to have at least 4 data blocks loaded at the same time
        static constexpr uint8_t MIN_DATA_BLOCK_LOAD_SLOT_COUNT = 4;

    public:
        ~TraceStreamDatabaseReader() = default;
        TraceStreamDatabaseReader() = default;
        TraceStreamDatabaseReader(const TraceStreamDatabaseReader&) = delete;
        TraceStreamDatabaseReader(TraceStreamDatabaseReader&& rhs) noexcept = default;
        TraceStreamDatabaseReader& operator=(const TraceStreamDatabaseReader&) = delete;
        TraceStreamDatabaseReader& operator=(TraceStreamDatabaseReader&& rhs) noexcept = default;

        void Reset() noexcept
        {
            m_relativeDataBlockSequenceNumberToLoadSlotIndexMap.clear();
            m_mruLoadSlotIndex = UINT8_MAX;
            m_loadSlots.clear();
            m_loadSlotsBackingMemory.reset();
            m_baseDataBlockSequenceNumber = UINT64_MAX;
            m_recordLocationEncoder.Reset();
            m_sizeofRecordWithInlinePayload = 0;
            m_dataBlockHeader = {};
            m_systemInfoRecordLocation = TSF_INVALID_RECORD_LOCATION;
            m_storageReader.Reset();
        }

        template <typename TFN_InitializeStorageReader>
        bool Initialize(TFN_InitializeStorageReader&& fnInitializeStorageReader)
        {
            Options options = {};
            return Initialize(options, std::forward<TFN_InitializeStorageReader>(fnInitializeStorageReader));
        }

        template <typename TFN_InitializeStorageReader>
        bool Initialize(const Options& options, TFN_InitializeStorageReader&& fnInitializeStorageReader)
        {
            NVTRACE_LOG_ERR_RETURN_IF(options.maxDataBlockLoadSlots < MIN_DATA_BLOCK_LOAD_SLOT_COUNT, false, 50, "maxDataBlockLoadSlots is too small, it must be at least 4");

            auto resetGuard = ScopeExitGuard([this]() { Reset(); });

            bool succeeded = fnInitializeStorageReader(m_storageReader);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize storage reader");
            NVTRACE_LOG_ERR_RETURN_IF(m_storageReader.DataBlockCount() == 0, false, 50, "No data blocks found in storage reader");

            for (size_t dataBlockIndex = 1; dataBlockIndex < m_storageReader.DataBlockCount(); ++dataBlockIndex)
            {
                const uint64_t previousSequenceNumber = m_storageReader.GetDataBlockSequenceNumber(dataBlockIndex - 1);
                const uint64_t sequenceNumber = m_storageReader.GetDataBlockSequenceNumber(dataBlockIndex);
                NVTRACE_LOG_ERR_RETURN_IF(sequenceNumber <= previousSequenceNumber, false, 50, "Data block sequence numbers must be strictly increasing, but sequence number %llu at index %zu follows %llu", sequenceNumber, dataBlockIndex, previousSequenceNumber);
            }

            // Read and verify the header of the first data block, expected to be common to all data blocks in the database
            {
                typename TStorageReader::DataBlockStorageReader dataBlockStorageReader;
                succeeded = m_storageReader.OpenDataBlock(0, dataBlockStorageReader);
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to open data block with index 0");

                // Read the header base, common to all versions
                succeeded = dataBlockStorageReader.ReadSection(&m_dataBlockHeader, 0, sizeof(TsfDataBlockHeaderVersion));
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to read data block header");

                // Verify magic number
                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.version.magic != TRACE_STREAM_FORMAT_MAGIC, false, 50, "Invalid data block header magic number %016llX", m_dataBlockHeader.version.magic);

                // Check that the version is supported
                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.version.majorVersion != TRACE_STREAM_FORMAT_MAJOR_VERSION, false, 50, "Unsupported data block header version %u, highest supported version is %u", m_dataBlockHeader.version.majorVersion, TRACE_STREAM_FORMAT_MAJOR_VERSION);

                succeeded = dataBlockStorageReader.ReadSection(&m_dataBlockHeader, sizeof(TsfDataBlockHeaderVersion), std::min((size_t)m_dataBlockHeader.version.sizeofDataBlockHeader, sizeof(TsfDataBlockHeader)));
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to read data block header");

                // Verify that the sections are not overlapping and appear in increasing offsets
                size_t currentOffset = m_dataBlockHeader.sizeofDataBlockState;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofDataBlockState < currentOffset, false, 50, "DataBlockState section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofDataBlockState, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofDataBlockState;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofGpuPartitionRefs < currentOffset, false, 50, "GpuPartitionRefs section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofGpuPartitionRefs, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofGpuPartitionRef * m_dataBlockHeader.maxGpuPartitionRefCount;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofProcessRefs < currentOffset, false, 50, "ProcessRefs section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofProcessRefs, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofProcessRef * m_dataBlockHeader.maxProcessRefCount;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofProviderRefs < currentOffset, false, 50, "ProviderRefs section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofProviderRefs, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofProviderRef * m_dataBlockHeader.maxProviderRefCount;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofThreadEventProviderMappingRefs < currentOffset, false, 50, "ThreadEventProviderMappingRefs section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofThreadEventProviderMappingRefs, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofThreadEventProviderMappingRef * m_dataBlockHeader.maxThreadEventProviderMappingRefCount;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofChunkStates < currentOffset, false, 50, "ChunkStates section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofChunkStates, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofChunkState * m_dataBlockHeader.maxChunkCount;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.offsetofChunks < currentOffset, false, 50, "Chunks section is overlapping or out-of-order (offset: %zu, expected >= %zu)", m_dataBlockHeader.offsetofChunks, currentOffset);
                currentOffset += m_dataBlockHeader.sizeofChunk * m_dataBlockHeader.maxChunkCount;

                NVTRACE_LOG_ERR_RETURN_IF(m_dataBlockHeader.sizeofDataBlock < currentOffset, false, 50, "Data block size is too small to fit all sections (size: %zu, required >= %zu)", m_dataBlockHeader.sizeofDataBlock, currentOffset);
            }

            m_sizeofRecordWithInlinePayload = m_dataBlockHeader.sizeofRecord + m_dataBlockHeader.sizeofInlinePayload;

            succeeded = m_recordLocationEncoder.Initialize(m_dataBlockHeader.maxChunkCount, m_dataBlockHeader.maxRecordCountPerChunk);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "TraceStreamRecordLocationEncoder::Initialize() failed");

            m_baseDataBlockSequenceNumber = m_storageReader.GetDataBlockSequenceNumber(0);

            m_loadSlotsBackingMemory = std::make_unique<uint8_t[]>(options.maxDataBlockLoadSlots * m_dataBlockHeader.sizeofDataBlock);

            m_loadSlots.resize(options.maxDataBlockLoadSlots);
            for (uint8_t loadSlotIndex = 0; loadSlotIndex < options.maxDataBlockLoadSlots; ++loadSlotIndex)
            {
                DataBlockLoadSlot& dataBlockLoadSlot = m_loadSlots[loadSlotIndex];
                dataBlockLoadSlot.pBackingMemory = m_loadSlotsBackingMemory.get() + loadSlotIndex * m_dataBlockHeader.sizeofDataBlock;
                dataBlockLoadSlot.nextMruSlotIndex = loadSlotIndex + 1;
                dataBlockLoadSlot.prevMruSlotIndex = loadSlotIndex - 1;
            }
            m_loadSlots.back().nextMruSlotIndex = 0;
            m_loadSlots.front().prevMruSlotIndex = options.maxDataBlockLoadSlots - 1;
            m_mruLoadSlotIndex = 0;

            uint64_t lastDataBlockSequenceNumber = m_storageReader.GetDataBlockSequenceNumber(m_storageReader.DataBlockCount() - 1);
            m_relativeDataBlockSequenceNumberToLoadSlotIndexMap.resize(static_cast<size_t>(lastDataBlockSequenceNumber - m_baseDataBlockSequenceNumber) + 1, UINT8_MAX);

            resetGuard.Dismiss();
            return true;
        }

        size_t DataBlockCount() const
        {
            return m_storageReader.DataBlockCount();
        }

        uint64_t DataBlockSize() const
        {
            return m_dataBlockHeader.sizeofDataBlock;
        }

        uint32_t RecordHeaderAndInlinePayloadSize() const
        {
            return m_sizeofRecordWithInlinePayload;
        }

        uint64_t GetDataBlockSequenceNumber(size_t dataBlockIndex) const
        {
            return m_storageReader.GetDataBlockSequenceNumber(dataBlockIndex);
        }

        TsfDataBlockRange GetDataBlockRange() const noexcept
        {
            return {0, m_storageReader.DataBlockCount()};
        }

        bool IsSnapshotDatabase() const
        {
            return m_storageReader.DataBlockCount() != 0 && IsAttachDataBlock(0);
        }

        bool IsAttachDataBlock(size_t dataBlockIndex) const
        {
            if (m_storageReader.DataBlockCount() <= dataBlockIndex)
            {
                return false;
            }
            const uint64_t dataBlockSequenceNumber = m_storageReader.GetDataBlockSequenceNumber(dataBlockIndex);
            return TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW <= dataBlockSequenceNumber
                && dataBlockSequenceNumber <= TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH;
        }

        TsfDataBlockRange GetHistoryDataBlockRange(const TsfDataBlockRange& dataBlockRange) const
        {
            TsfDataBlockRange historyDataBlockRange = {
                std::min(dataBlockRange.beginDataBlockIndex, m_storageReader.DataBlockCount()),
                std::min(dataBlockRange.endDataBlockIndex, m_storageReader.DataBlockCount()),
            };
            historyDataBlockRange.beginDataBlockIndex =
                std::min(historyDataBlockRange.beginDataBlockIndex, historyDataBlockRange.endDataBlockIndex);
            while (historyDataBlockRange.beginDataBlockIndex < historyDataBlockRange.endDataBlockIndex
                   && IsAttachDataBlock(historyDataBlockRange.beginDataBlockIndex))
            {
                ++historyDataBlockRange.beginDataBlockIndex;
            }
            return historyDataBlockRange;
        }

        bool GetDataBlockRangeTimeBoundsUtcNsec(
            const TsfDataBlockRange& dataBlockRange,
            uint64_t& out_lowerBoundUtcNsec,
            uint64_t& out_upperBoundUtcNsec)
        {
            const TsfDataBlockRange clampedDataBlockRange = {
                std::min(dataBlockRange.beginDataBlockIndex, m_storageReader.DataBlockCount()),
                std::min(dataBlockRange.endDataBlockIndex, m_storageReader.DataBlockCount()),
            };
            if (clampedDataBlockRange.beginDataBlockIndex >= clampedDataBlockRange.endDataBlockIndex)
            {
                return false;
            }

            TsfDataBlockState firstDataBlockState;
            TsfDataBlockState lastDataBlockState;
            if (!GetDataBlockState(clampedDataBlockRange.beginDataBlockIndex, firstDataBlockState)
                || !GetDataBlockState(clampedDataBlockRange.endDataBlockIndex - 1, lastDataBlockState))
            {
                return false;
            }

            out_lowerBoundUtcNsec = firstDataBlockState.recordTimestampLowerBoundUtcNsec;
            out_upperBoundUtcNsec = lastDataBlockState.recordTimestampUpperBoundUtcNsec;
            return out_lowerBoundUtcNsec <= out_upperBoundUtcNsec;
        }

        bool GetDataBlockState(size_t dataBlockIndex, TsfDataBlockState& out_dataBlockState)
        {
            if (dataBlockIndex >= m_storageReader.DataBlockCount())
            {
                return false;
            }

            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::DataBlockState);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            out_dataBlockState = *pDataBlockState;
            return true;
        }

        size_t FindDataBlockIndexBySequenceNumber(uint64_t dataBlockSequenceNumber) const
        {
            return m_storageReader.FindDataBlockIndexBySequenceNumber(dataBlockSequenceNumber);
        }

        uint64_t GetSystemInfoRecordLocation() const
        {
            return m_systemInfoRecordLocation;
        }

        bool FindDataBlocksInTimespan(uint64_t startTimeUtcNsec, uint64_t endTimeUtcNsec, size_t& out_firstDataBlockIndex, size_t& out_endDataBlockIndex)
        {
            // Binary search for first and past-the-end data block indices that contain records within the timespan simultaneously.
            // This way we guarantee to not load any data block twice

            std::pair<size_t, size_t> firstDataBlockIndexRange = {0, m_storageReader.DataBlockCount()};
            std::pair<size_t, size_t> endDataBlockIndexRange = {0, m_storageReader.DataBlockCount()};

            while (firstDataBlockIndexRange.first < firstDataBlockIndexRange.second)
            {
                size_t dataBlockIndex = (firstDataBlockIndexRange.first + firstDataBlockIndexRange.second) / 2;
                const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::DataBlockState);
                if (!pDataBlock)
                {
                    return false;
                }

                const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
                if (pDataBlockState->recordTimestampUpperBoundUtcNsec < startTimeUtcNsec)
                {
                    firstDataBlockIndexRange.first = endDataBlockIndexRange.first = dataBlockIndex + 1;
                }
                else
                {
                    firstDataBlockIndexRange.second = dataBlockIndex;

                    if (endTimeUtcNsec < pDataBlockState->recordTimestampLowerBoundUtcNsec)
                    {
                        endDataBlockIndexRange.second = dataBlockIndex;
                    }
                    else if (endDataBlockIndexRange.first <= dataBlockIndex)
                    {
                        endDataBlockIndexRange.first = dataBlockIndex + 1;
                    }
                }
            }

            while (endDataBlockIndexRange.first < endDataBlockIndexRange.second)
            {
                size_t dataBlockIndex = (endDataBlockIndexRange.first + endDataBlockIndexRange.second) / 2;
                const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::DataBlockState);
                if (!pDataBlock)
                {
                    return false;
                }

                const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
                if (endTimeUtcNsec < pDataBlockState->recordTimestampLowerBoundUtcNsec)
                {
                    endDataBlockIndexRange.second = dataBlockIndex;
                }
                else
                {
                    endDataBlockIndexRange.first = dataBlockIndex + 1;
                }
            }

            out_firstDataBlockIndex = firstDataBlockIndexRange.first;
            out_endDataBlockIndex = endDataBlockIndexRange.first;
            return true;
        }

        bool FindDataBlocksInTimespan(uint64_t startTimeUtcNsec, uint64_t endTimeUtcNsec, TsfDataBlockRange& dataBlockRange)
        {
            return FindDataBlocksInTimespan(startTimeUtcNsec, endTimeUtcNsec, dataBlockRange.beginDataBlockIndex, dataBlockRange.endDataBlockIndex);
        }

        template <typename TFN_OnGpuPartitionRef>
        bool ForEachGpuPartitionRefInDataBlock(size_t dataBlockIndex, TFN_OnGpuPartitionRef&& fnOnGpuPartitionRef)
        {
            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::RouterObjectRefs);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            const uint8_t* pGpuPartitionRefs = pDataBlock + m_dataBlockHeader.offsetofGpuPartitionRefs;

            for (uint32_t gpuPartitionRefIndex = 0; gpuPartitionRefIndex < pDataBlockState->validGpuPartitionRefCount; ++gpuPartitionRefIndex)
            {
                fnOnGpuPartitionRef(*reinterpret_cast<const TsfGpuPartitionRef*>(pGpuPartitionRefs + gpuPartitionRefIndex * m_dataBlockHeader.sizeofGpuPartitionRef));
            }

            return true;
        }

        template <typename TFN_OnProcessRef>
        bool ForEachProcessRefInDataBlock(size_t dataBlockIndex, TFN_OnProcessRef&& fnOnProcessRef)
        {
            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::RouterObjectRefs);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            const uint8_t* pProcessRefs = pDataBlock + m_dataBlockHeader.offsetofProcessRefs;

            for (uint32_t processRefIndex = 0; processRefIndex < pDataBlockState->validProcessRefCount; ++processRefIndex)
            {
                fnOnProcessRef(*reinterpret_cast<const TsfProcessRef*>(pProcessRefs + processRefIndex * m_dataBlockHeader.sizeofProcessRef));
            }

            return true;
        }

        template <typename TFN_OnProviderRef>
        bool ForEachProviderRefInDataBlock(size_t dataBlockIndex, TFN_OnProviderRef&& fnOnProviderRef)
        {
            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::RouterObjectRefs);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            const uint8_t* pProviderRefs = pDataBlock + m_dataBlockHeader.offsetofProviderRefs;

            for (uint32_t providerRefIndex = 0; providerRefIndex < pDataBlockState->validProviderRefCount; ++providerRefIndex)
            {
                fnOnProviderRef(*reinterpret_cast<const TsfProviderRef*>(pProviderRefs + providerRefIndex * m_dataBlockHeader.sizeofProviderRef));
            }

            return true;
        }

        template <typename TFN_OnThreadEventProviderMappingRef>
        bool ForEachThreadEventProviderMappingRefInDataBlock(size_t dataBlockIndex, TFN_OnThreadEventProviderMappingRef&& fnOnThreadEventProviderMappingRef)
        {
            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::RouterObjectRefs);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            const uint8_t* pThreadEventProviderMappingRefs = pDataBlock + m_dataBlockHeader.offsetofThreadEventProviderMappingRefs;

            for (uint32_t mappingRefIndex = 0; mappingRefIndex < pDataBlockState->validThreadEventProviderMappingRefCount; ++mappingRefIndex)
            {
                const TsfThreadEventProviderMappingRef& mappingRef = *reinterpret_cast<const TsfThreadEventProviderMappingRef*>(pThreadEventProviderMappingRefs + mappingRefIndex * m_dataBlockHeader.sizeofThreadEventProviderMappingRef);
                fnOnThreadEventProviderMappingRef(mappingRef);
            }

            return true;
        }

        template <typename TFN_OnRecord>
        bool ForEachRecordInDataBlock(size_t dataBlockIndex, TFN_OnRecord&& fnOnRecord)
        {
            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::Chunks);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            const uint8_t* pProviderRefs = pDataBlock + m_dataBlockHeader.offsetofProviderRefs;
            const uint8_t* pChunkStates = pDataBlock + m_dataBlockHeader.offsetofChunkStates;
            const uint8_t* pChunks = pDataBlock + m_dataBlockHeader.offsetofChunks;

            for (uint32_t chunkIndex = 0; chunkIndex < pDataBlockState->validChunkCount; ++chunkIndex)
            {
                const TsfChunkState* pChunkState = reinterpret_cast<const TsfChunkState*>(pChunkStates + chunkIndex * m_dataBlockHeader.sizeofChunkState);
                const uint8_t* pChunk = pChunks + chunkIndex * m_dataBlockHeader.sizeofChunk;
                uint32_t prevMajorPayloadOffset = 0;

                for (uint32_t recordIndex = 0; recordIndex < pChunkState->validRecordCount; ++recordIndex)
                {
                    uint64_t recordLocation = m_recordLocationEncoder.EncodeRecordLocation(pDataBlockState->dataBlockSequenceNumber, chunkIndex, recordIndex);
                    const TsfRecord* pRecord = reinterpret_cast<const TsfRecord*>(pChunk + recordIndex * m_sizeofRecordWithInlinePayload);
                    const uint8_t* pInlinePayload = reinterpret_cast<const uint8_t*>(pRecord) + m_dataBlockHeader.sizeofRecord;

                    uint32_t providerRefIndex = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_PROVIDER_REF_INDEX);
                    NVTRACE_LUID providerLuid = providerRefIndex != TSF_INVALID_OBJECT_REF_INDEX ? reinterpret_cast<const TsfProviderRef*>(pProviderRefs + providerRefIndex * m_dataBlockHeader.sizeofProviderRef)->providerLuid : NVTRACE_NIL_LUID;

                    uint32_t recordType = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_RECORD_TYPE);

                    uint32_t majorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
                    if (majorPayloadOffset > m_dataBlockHeader.sizeofChunk || majorPayloadOffset < prevMajorPayloadOffset)
                    {
                        NVTRACE_LOG_ERR(50, "Invalid major payload offsets in chunk %u record %u", chunkIndex, recordIndex);
                        return false;
                    }
                    const uint8_t* pMajorPayload = pChunk + m_dataBlockHeader.sizeofChunk - majorPayloadOffset;

                    size_t maxMajorPayloadSize = static_cast<size_t>(majorPayloadOffset - prevMajorPayloadOffset);
                    prevMajorPayloadOffset = majorPayloadOffset;

                    fnOnRecord(recordLocation, providerLuid, recordType, pRecord->correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
                }
            }

            return true;
        }

        template <typename TFN_OnRecord>
        bool ForEachRecordInDataBlockBackwards(size_t dataBlockIndex, TFN_OnRecord&& fnOnRecord)
        {
            const uint8_t* pDataBlock = LoadDataBlockByIndex(dataBlockIndex, DataBlockLoadLevel::Chunks);
            if (!pDataBlock)
            {
                return false;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);
            const uint8_t* pProviderRefs = pDataBlock + m_dataBlockHeader.offsetofProviderRefs;
            const uint8_t* pChunkStates = pDataBlock + m_dataBlockHeader.offsetofChunkStates;
            const uint8_t* pChunks = pDataBlock + m_dataBlockHeader.offsetofChunks;

            for (uint32_t chunkIndex = pDataBlockState->validChunkCount; 0 < chunkIndex;)
            {
                --chunkIndex;
                const TsfChunkState* pChunkState = reinterpret_cast<const TsfChunkState*>(pChunkStates + chunkIndex * m_dataBlockHeader.sizeofChunkState);
                const uint8_t* pChunk = pChunks + chunkIndex * m_dataBlockHeader.sizeofChunk;

                for (uint32_t recordIndex = pChunkState->validRecordCount; 0 < recordIndex;)
                {
                    --recordIndex;
                    uint64_t recordLocation = m_recordLocationEncoder.EncodeRecordLocation(pDataBlockState->dataBlockSequenceNumber, chunkIndex, recordIndex);
                    const TsfRecord* pRecord = reinterpret_cast<const TsfRecord*>(pChunk + recordIndex * m_sizeofRecordWithInlinePayload);
                    const uint8_t* pInlinePayload = reinterpret_cast<const uint8_t*>(pRecord) + m_dataBlockHeader.sizeofRecord;

                    uint32_t providerRefIndex = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_PROVIDER_REF_INDEX);
                    NVTRACE_LUID providerLuid = providerRefIndex != TSF_INVALID_OBJECT_REF_INDEX ? reinterpret_cast<const TsfProviderRef*>(pProviderRefs + providerRefIndex * m_dataBlockHeader.sizeofProviderRef)->providerLuid : NVTRACE_NIL_LUID;

                    uint32_t recordType = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_RECORD_TYPE);

                    uint32_t majorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
                    const uint8_t* pMajorPayload = pChunk + m_dataBlockHeader.sizeofChunk - majorPayloadOffset;

                    uint32_t prevMajorPayloadOffset = 0;
                    if (0 < recordIndex)
                    {
                        const TsfRecord* pPrevRecord = reinterpret_cast<const TsfRecord*>(reinterpret_cast<const uint8_t*>(pRecord) - m_sizeofRecordWithInlinePayload);
                        prevMajorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pPrevRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
                    }
                    size_t maxMajorPayloadSize = static_cast<size_t>(majorPayloadOffset - prevMajorPayloadOffset);

                    fnOnRecord(recordLocation, providerLuid, recordType, pRecord->correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
                }
            }

            return true;
        }

        size_t GetRecordDataBlockIndex(uint64_t recordLocation)
        {
            const uint64_t dataBlockSequenceNumber = m_recordLocationEncoder.DecodeRecordLocation(recordLocation).dataBlockSequenceNumber;
            return m_storageReader.FindDataBlockIndexBySequenceNumber(dataBlockSequenceNumber);
        }

        uint64_t GetRecordDataBlockSequenceNumber(uint64_t recordLocation) const
        {
            return m_recordLocationEncoder.DecodeRecordLocation(recordLocation).dataBlockSequenceNumber;
        }

        struct RecordMetadata
        {
            uint32_t recordType = Codec::INVALID_RECORD_TYPE;
            NvTrace_CorrelationId correlationId = NVTRACE_NO_CORRELATION_ID;
        };
        RecordMetadata GetRecordMetadata(uint64_t recordLocation)
        {
            auto [dataBlockSequenceNumber, chunkIndex, recordIndex] = m_recordLocationEncoder.DecodeRecordLocation(recordLocation);
            const uint8_t* pDataBlock = LoadDataBlockBySequenceNumber(dataBlockSequenceNumber, DataBlockLoadLevel::Chunks);
            if (!pDataBlock)
            {
                return {};
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);

            if (pDataBlockState->validChunkCount <= chunkIndex)
            {
                NVTRACE_LOG_ERR(50, "Out-of-bounds chunk index %u", chunkIndex);
                return {};
            }
            const TsfChunkState* pChunkState = reinterpret_cast<const TsfChunkState*>(pDataBlock + m_dataBlockHeader.offsetofChunkStates + chunkIndex * m_dataBlockHeader.sizeofChunkState);
            const uint8_t* pChunk = reinterpret_cast<const uint8_t*>(pDataBlock + m_dataBlockHeader.offsetofChunks + chunkIndex * m_dataBlockHeader.sizeofChunk);

            if (pChunkState->validRecordCount <= recordIndex)
            {
                NVTRACE_LOG_ERR(50, "Out-of-bounds record index %u", recordIndex);
                return {};
            }
            const TsfRecord* pRecord = reinterpret_cast<const TsfRecord*>(pChunk + recordIndex * m_sizeofRecordWithInlinePayload);

            RecordMetadata recordMetadata;
            recordMetadata.recordType = static_cast<uint32_t>(NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_RECORD_TYPE));
            recordMetadata.correlationId = pRecord->correlationId;

            return recordMetadata;
        }

        struct RecordPayloads
        {
            const uint8_t* pInlinePayload = nullptr;
            const uint8_t* pMajorPayload = nullptr;
            size_t maxMajorPayloadSize = 0;
        };

        RecordPayloads GetRecordPayloads(uint64_t recordLocation)
        {
            auto [dataBlockSequenceNumber, chunkIndex, recordIndex] = m_recordLocationEncoder.DecodeRecordLocation(recordLocation);
            RecordPayloads recordPayloads;

            const uint8_t* pDataBlock = LoadDataBlockBySequenceNumber(dataBlockSequenceNumber, DataBlockLoadLevel::Chunks);
            if (!pDataBlock)
            {
                return recordPayloads;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);

            if (pDataBlockState->validChunkCount <= chunkIndex)
            {
                NVTRACE_LOG_ERR(50, "Out-of-bounds chunk index %u", chunkIndex);
                return recordPayloads;
            }
            const TsfChunkState* pChunkState = reinterpret_cast<const TsfChunkState*>(pDataBlock + m_dataBlockHeader.offsetofChunkStates + chunkIndex * m_dataBlockHeader.sizeofChunkState);
            const uint8_t* pChunk = reinterpret_cast<const uint8_t*>(pDataBlock + m_dataBlockHeader.offsetofChunks + chunkIndex * m_dataBlockHeader.sizeofChunk);

            if (pChunkState->validRecordCount <= recordIndex)
            {
                NVTRACE_LOG_ERR(50, "Out-of-bounds record index %u", recordIndex);
                return recordPayloads;
            }
            const TsfRecord* pRecord = reinterpret_cast<const TsfRecord*>(pChunk + recordIndex * m_sizeofRecordWithInlinePayload);
            recordPayloads.pInlinePayload = reinterpret_cast<const uint8_t*>(pRecord) + m_dataBlockHeader.sizeofRecord;

            const uint32_t majorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
            recordPayloads.pMajorPayload = pChunk + m_dataBlockHeader.sizeofChunk - majorPayloadOffset;

            uint32_t prevMajorPayloadOffset = 0;
            if (0 < recordIndex)
            {
                const TsfRecord* pPrevRecord = reinterpret_cast<const TsfRecord*>(reinterpret_cast<const uint8_t*>(pRecord) - m_sizeofRecordWithInlinePayload);
                prevMajorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pPrevRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
            }
            recordPayloads.maxMajorPayloadSize = static_cast<size_t>(majorPayloadOffset - prevMajorPayloadOffset);

            return recordPayloads;
        }

        RecordPayloads GetRecordPayloads(uint64_t recordLocation, NVTRACE_LUID expectedProviderLuid, uint32_t expectedRecordType)
        {
            auto [dataBlockSequenceNumber, chunkIndex, recordIndex] = m_recordLocationEncoder.DecodeRecordLocation(recordLocation);
            RecordPayloads recordPayloads;

            const uint8_t* pDataBlock = LoadDataBlockBySequenceNumber(dataBlockSequenceNumber, DataBlockLoadLevel::Chunks);
            if (!pDataBlock)
            {
                return recordPayloads;
            }

            const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(pDataBlock + m_dataBlockHeader.offsetofDataBlockState);

            if (pDataBlockState->validChunkCount <= chunkIndex)
            {
                NVTRACE_LOG_ERR(50, "Out-of-bounds chunk index %u", chunkIndex);
                return recordPayloads;
            }
            const TsfChunkState* pChunkState = reinterpret_cast<const TsfChunkState*>(pDataBlock + m_dataBlockHeader.offsetofChunkStates + chunkIndex * m_dataBlockHeader.sizeofChunkState);
            const uint8_t* pChunk = reinterpret_cast<const uint8_t*>(pDataBlock + m_dataBlockHeader.offsetofChunks + chunkIndex * m_dataBlockHeader.sizeofChunk);

            if (pChunkState->validRecordCount <= recordIndex)
            {
                NVTRACE_LOG_ERR(50, "Out-of-bounds record index %u", recordIndex);
                return recordPayloads;
            }
            const TsfRecord* pRecord = reinterpret_cast<const TsfRecord*>(pChunk + recordIndex * m_sizeofRecordWithInlinePayload);
            recordPayloads.pInlinePayload = reinterpret_cast<const uint8_t*>(pRecord) + m_dataBlockHeader.sizeofRecord;

            const uint32_t providerRefIndex = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_PROVIDER_REF_INDEX);
            const NVTRACE_LUID providerLuid = providerRefIndex != TSF_INVALID_OBJECT_REF_INDEX ? reinterpret_cast<const TsfProviderRef*>(pDataBlock + m_dataBlockHeader.offsetofProviderRefs + providerRefIndex * m_dataBlockHeader.sizeofProviderRef)->providerLuid : NVTRACE_NIL_LUID;
            if (providerLuid != expectedProviderLuid)
            {
                NVTRACE_LOG_ERR(50, "Expected provider LUID %llu but got %llu for record at location %llu", expectedProviderLuid, providerLuid, recordLocation);
                return recordPayloads;
            }

            const uint32_t recordType = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_RECORD_TYPE);
            if (recordType != expectedRecordType)
            {
                NVTRACE_LOG_ERR(50, "Expected record type %u but got %u for record at location %llu", expectedRecordType, recordType, recordLocation);
                return recordPayloads;
            }

            const uint32_t majorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
            recordPayloads.pMajorPayload = pChunk + m_dataBlockHeader.sizeofChunk - majorPayloadOffset;

            uint32_t prevMajorPayloadOffset = 0;
            if (0 < recordIndex)
            {
                const TsfRecord* pPrevRecord = reinterpret_cast<const TsfRecord*>(reinterpret_cast<const uint8_t*>(pRecord) - m_sizeofRecordWithInlinePayload);
                prevMajorPayloadOffset = (uint32_t)NVTRACE_GETFIELD64(pPrevRecord->packed, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8) * 8;
            }
            recordPayloads.maxMajorPayloadSize = static_cast<size_t>(majorPayloadOffset - prevMajorPayloadOffset);

            return recordPayloads;
        }

    private:
        enum class DataBlockLoadLevel
        {
            Header = 0,
            DataBlockState,
            RouterObjectRefs,
            Chunks,
            Everything = Chunks
        };

        struct DataBlockLoadSlot
        {
            uint8_t* pBackingMemory = nullptr;

            size_t relativeDataBlockSequenceNumber = SIZE_MAX;
            typename TStorageReader::DataBlockStorageReader storageReader;
            DataBlockLoadLevel loadLevel = DataBlockLoadLevel::Header;

            uint8_t nextMruSlotIndex = UINT8_MAX;
            uint8_t prevMruSlotIndex = UINT8_MAX;
        };

        void UpdateSystemInfoRecordLocation(uint64_t systemInfoRecordLocation)
        {
            if (systemInfoRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                return;
            }

            if (m_systemInfoRecordLocation == TSF_INVALID_RECORD_LOCATION || m_recordLocationEncoder.DecodeRecordLocation(m_systemInfoRecordLocation).dataBlockSequenceNumber < m_recordLocationEncoder.DecodeRecordLocation(systemInfoRecordLocation).dataBlockSequenceNumber)
            {
                m_systemInfoRecordLocation = systemInfoRecordLocation;
            }
        }

        bool LoadDataBlockHeader(size_t dataBlockIndex, size_t relativeDataBlockSequenceNumber, uint8_t loadSlotIndex)
        {
            DataBlockLoadSlot& loadSlot = m_loadSlots[loadSlotIndex];

            if (loadSlot.relativeDataBlockSequenceNumber != SIZE_MAX)
            {
                // Remove the load slot's sequence number from the map and reset the slot to an empty state
                m_relativeDataBlockSequenceNumberToLoadSlotIndexMap[loadSlot.relativeDataBlockSequenceNumber] = UINT8_MAX;
                loadSlot.relativeDataBlockSequenceNumber = SIZE_MAX;
                loadSlot.storageReader.Reset();
            }

            // Open the data block
            bool succeeded = m_storageReader.OpenDataBlock(dataBlockIndex, loadSlot.storageReader);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to open data block with sequence number %llu", m_baseDataBlockSequenceNumber + relativeDataBlockSequenceNumber);

            auto resetDataBlockStorageReaderGuard = ScopeExitGuard([&loadSlot]() { loadSlot.storageReader.Reset(); });

            // Read and verify the header of the data block
            succeeded = loadSlot.storageReader.ReadSection(loadSlot.pBackingMemory, 0, m_dataBlockHeader.version.sizeofDataBlockHeader);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to read data block header for data block with sequence number %llu", m_baseDataBlockSequenceNumber + relativeDataBlockSequenceNumber);
            bool headerMatches = std::memcmp(loadSlot.pBackingMemory, &m_dataBlockHeader, std::min((size_t)m_dataBlockHeader.version.sizeofDataBlockHeader, sizeof(TsfDataBlockHeaderVersion))) == 0;
            NVTRACE_LOG_ERR_RETURN_IF(!headerMatches, false, 50, "Data block header for data block with sequence number %llu does not match expected header", m_baseDataBlockSequenceNumber + relativeDataBlockSequenceNumber);

            resetDataBlockStorageReaderGuard.Dismiss();

            loadSlot.relativeDataBlockSequenceNumber = relativeDataBlockSequenceNumber;
            loadSlot.loadLevel = DataBlockLoadLevel::Header;
            m_relativeDataBlockSequenceNumberToLoadSlotIndexMap[relativeDataBlockSequenceNumber] = loadSlotIndex;

            return true;
        }

        bool IncreaseDataBlockLoadLevel(DataBlockLoadSlot& loadSlot, DataBlockLoadLevel loadLevel)
        {
            auto fnGetLoadLevelEndOffset = [this](DataBlockLoadLevel loadLevel) -> size_t {
                switch (loadLevel)
                {
                    case DataBlockLoadLevel::Header:
                        return static_cast<size_t>(m_dataBlockHeader.version.sizeofDataBlockHeader);
                    case DataBlockLoadLevel::DataBlockState:
                        return static_cast<size_t>(m_dataBlockHeader.offsetofDataBlockState + m_dataBlockHeader.sizeofDataBlockState);
                    case DataBlockLoadLevel::RouterObjectRefs:
                        return static_cast<size_t>(m_dataBlockHeader.offsetofThreadEventProviderMappingRefs + m_dataBlockHeader.maxThreadEventProviderMappingRefCount * m_dataBlockHeader.sizeofThreadEventProviderMappingRef);
                    case DataBlockLoadLevel::Chunks:
                        return static_cast<size_t>(m_dataBlockHeader.offsetofChunks + m_dataBlockHeader.maxChunkCount * m_dataBlockHeader.sizeofChunk);
                }
                return 0;
            };

            size_t currentLoadedEndOffset = fnGetLoadLevelEndOffset(loadSlot.loadLevel);
            size_t requiredLoadedEndOffset = fnGetLoadLevelEndOffset(loadLevel);
            bool succeeded = loadSlot.storageReader.ReadSection(loadSlot.pBackingMemory, currentLoadedEndOffset, requiredLoadedEndOffset);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to read data block with sequence number %llu from %zu to %zu", m_baseDataBlockSequenceNumber + loadSlot.relativeDataBlockSequenceNumber, currentLoadedEndOffset, requiredLoadedEndOffset);

            if (loadSlot.loadLevel < DataBlockLoadLevel::DataBlockState && loadLevel >= DataBlockLoadLevel::DataBlockState)
            {
                const TsfDataBlockState* pDataBlockState = reinterpret_cast<const TsfDataBlockState*>(loadSlot.pBackingMemory + m_dataBlockHeader.offsetofDataBlockState);
                UpdateSystemInfoRecordLocation(pDataBlockState->systemInfoRecordLocation);
            }

            loadSlot.loadLevel = loadLevel;

            // If the load level is now Everything, we will no longer need the storage reader, so reset it to free any resources it might be holding
            if (loadLevel == DataBlockLoadLevel::Everything)
            {
                loadSlot.storageReader.Reset();
            }

            return true;
        }

        // Precondition: loadSlotIndex != m_mruLoadSlotIndex
        void RelocateLoadSlotBeforeMruSlot(uint8_t loadSlotIndex)
        {
            DataBlockLoadSlot& loadSlot = m_loadSlots[loadSlotIndex];

            // Check if the load slot is already before the MRU slot, in which case there is nothing to do
            if (loadSlot.nextMruSlotIndex == m_mruLoadSlotIndex)
            {
                return;
            }

            // Remove the load slot from the MRU slot list
            m_loadSlots[loadSlot.nextMruSlotIndex].prevMruSlotIndex = loadSlot.prevMruSlotIndex;
            m_loadSlots[loadSlot.prevMruSlotIndex].nextMruSlotIndex = loadSlot.nextMruSlotIndex;

            // Add the load slot back right before the MRU slot
            loadSlot.nextMruSlotIndex = m_mruLoadSlotIndex;
            loadSlot.prevMruSlotIndex = m_loadSlots[m_mruLoadSlotIndex].prevMruSlotIndex;
            m_loadSlots[loadSlot.prevMruSlotIndex].nextMruSlotIndex = loadSlotIndex;
            m_loadSlots[m_mruLoadSlotIndex].prevMruSlotIndex = loadSlotIndex;
        }

        const uint8_t* LoadDataBlockByIndex(size_t dataBlockIndex, DataBlockLoadLevel loadLevel)
        {
            uint64_t dataBlockSequenceNumber = m_storageReader.GetDataBlockSequenceNumber(dataBlockIndex);
            size_t relativeDataBlockSequenceNumber = static_cast<size_t>(dataBlockSequenceNumber - m_baseDataBlockSequenceNumber);

            uint8_t loadSlotIndex = m_relativeDataBlockSequenceNumberToLoadSlotIndexMap[relativeDataBlockSequenceNumber];
            if (loadSlotIndex == UINT8_MAX)
            {
                // Replace the data block in the LRU load slot with the new data block
                loadSlotIndex = m_loadSlots[m_mruLoadSlotIndex].prevMruSlotIndex;
                if (!LoadDataBlockHeader(dataBlockIndex, relativeDataBlockSequenceNumber, loadSlotIndex))
                {
                    return nullptr;
                }

                m_mruLoadSlotIndex = loadSlotIndex;
            }
            else if (loadSlotIndex != m_mruLoadSlotIndex)
            {
                RelocateLoadSlotBeforeMruSlot(loadSlotIndex);
                m_mruLoadSlotIndex = loadSlotIndex;
            }

            DataBlockLoadSlot& loadSlot = m_loadSlots[loadSlotIndex];
            if (loadSlot.loadLevel < loadLevel && !IncreaseDataBlockLoadLevel(loadSlot, loadLevel))
            {
                return nullptr;
            }

            return loadSlot.pBackingMemory;
        }

        const uint8_t* LoadDataBlockBySequenceNumber(uint64_t dataBlockSequenceNumber, DataBlockLoadLevel loadLevel)
        {
            size_t relativeDataBlockSequenceNumber = static_cast<size_t>(dataBlockSequenceNumber - m_baseDataBlockSequenceNumber);
            NVTRACE_LOG_ERR_RETURN_IF(m_relativeDataBlockSequenceNumberToLoadSlotIndexMap.size() <= relativeDataBlockSequenceNumber, nullptr, 50, "Data block with sequence number %llu not found", dataBlockSequenceNumber);

            uint8_t loadSlotIndex = m_relativeDataBlockSequenceNumberToLoadSlotIndexMap[relativeDataBlockSequenceNumber];
            if (loadSlotIndex == UINT8_MAX)
            {
                size_t dataBlockIndex = m_storageReader.FindDataBlockIndexBySequenceNumber(dataBlockSequenceNumber);
                NVTRACE_LOG_ERR_RETURN_IF(dataBlockIndex == SIZE_MAX, nullptr, 50, "Data block with sequence number %llu not found", dataBlockSequenceNumber);

                // Replace the data block in the LRU load slot with the new data block
                loadSlotIndex = m_loadSlots[m_mruLoadSlotIndex].prevMruSlotIndex;
                if (!LoadDataBlockHeader(dataBlockIndex, relativeDataBlockSequenceNumber, loadSlotIndex))
                {
                    return nullptr;
                }

                m_mruLoadSlotIndex = loadSlotIndex;
            }
            else if (loadSlotIndex != m_mruLoadSlotIndex)
            {
                RelocateLoadSlotBeforeMruSlot(loadSlotIndex);
                m_mruLoadSlotIndex = loadSlotIndex;
            }

            DataBlockLoadSlot& loadSlot = m_loadSlots[loadSlotIndex];
            if (loadSlot.loadLevel < loadLevel && !IncreaseDataBlockLoadLevel(loadSlot, loadLevel))
            {
                return nullptr;
            }

            return loadSlot.pBackingMemory;
        }

    private:
        TStorageReader m_storageReader;

        TsfDataBlockHeader m_dataBlockHeader = {};
        uint64_t m_systemInfoRecordLocation = TSF_INVALID_RECORD_LOCATION;
        uint32_t m_sizeofRecordWithInlinePayload = 0;
        TraceStreamRecordLocationEncoder m_recordLocationEncoder;

        uint64_t m_baseDataBlockSequenceNumber = UINT64_MAX;

        std::unique_ptr<uint8_t[]> m_loadSlotsBackingMemory;
        std::vector<DataBlockLoadSlot> m_loadSlots;
        uint8_t m_mruLoadSlotIndex = UINT8_MAX;

        std::vector<uint8_t> m_relativeDataBlockSequenceNumberToLoadSlotIndexMap;
    };


    class TsfDiskDatabaseBufferedStorageReader
    {
    public:
        class DataBlockStorageReader
        {
        public:
            ~DataBlockStorageReader() = default;
            DataBlockStorageReader() = default;
            DataBlockStorageReader(const DataBlockStorageReader&) = delete;
            DataBlockStorageReader(DataBlockStorageReader&& rhs) noexcept = default;
            DataBlockStorageReader& operator=(const DataBlockStorageReader&) = delete;
            DataBlockStorageReader& operator=(DataBlockStorageReader&& rhs) noexcept = default;

            void Reset() noexcept
            {
                m_file.close();
            }

            bool Initialize(const std::filesystem::path& filePath)
            {
                m_file.open(filePath, std::ios::binary);
                return !m_file.fail();
            }

            bool ReadSection(void* pBuffer, size_t sectionBeginOffset, size_t sectionEndOffset)
            {
                m_file.seekg(sectionBeginOffset, std::ios::beg);
                m_file.read(static_cast<char*>(pBuffer) + sectionBeginOffset, sectionEndOffset - sectionBeginOffset);
                return !m_file.fail();
            }

        private:
            std::ifstream m_file;
        };

    public:
        ~TsfDiskDatabaseBufferedStorageReader() = default;
        TsfDiskDatabaseBufferedStorageReader() = default;
        TsfDiskDatabaseBufferedStorageReader(const TsfDiskDatabaseBufferedStorageReader&) = delete;
        TsfDiskDatabaseBufferedStorageReader(TsfDiskDatabaseBufferedStorageReader&& rhs) noexcept = default;
        TsfDiskDatabaseBufferedStorageReader& operator=(const TsfDiskDatabaseBufferedStorageReader&) = delete;
        TsfDiskDatabaseBufferedStorageReader& operator=(TsfDiskDatabaseBufferedStorageReader&& rhs) noexcept = default;

        void Reset() noexcept
        {
            m_dataBlockInfos.clear();
        }

        bool Initialize(const std::filesystem::path& databaseDirectoryPath)
        {
            for (const auto& entry : std::filesystem::directory_iterator(databaseDirectoryPath))
            {
                if (!entry.is_regular_file() || entry.path().extension() != ".nvtsf")
                {
                    continue;
                }

                // Extract dataBlockSequenceNumber from filename (hex after last '-')
                std::string stem = entry.path().stem().string();
                size_t lastDashPos = stem.find_last_of('-');
                size_t dataBlockSequenceNumberBeginPos = lastDashPos == std::string::npos ? 0 : lastDashPos + 1;

                uint64_t dataBlockSequenceNumber = 0;
                auto [_, errorCode] = std::from_chars(stem.data() + dataBlockSequenceNumberBeginPos, stem.data() + stem.size(), dataBlockSequenceNumber, 16);
                if (errorCode != std::errc{})
                {
                    continue; // Skip files with invalid hex sequence
                }

                TsfDataBlockInfo& dataBlockInfo = m_dataBlockInfos.emplace_back();
                dataBlockInfo.filePath = entry.path();
                dataBlockInfo.dataBlockSequenceNumber = dataBlockSequenceNumber;
            }

            std::sort(m_dataBlockInfos.begin(), m_dataBlockInfos.end(), [](const TsfDataBlockInfo& lhs, const TsfDataBlockInfo& rhs) { return lhs.dataBlockSequenceNumber < rhs.dataBlockSequenceNumber; });

            return true;
        }

        size_t DataBlockCount() const
        {
            return m_dataBlockInfos.size();
        }

        uint64_t GetDataBlockSequenceNumber(size_t dataBlockIndex) const
        {
            return m_dataBlockInfos[dataBlockIndex].dataBlockSequenceNumber;
        }

        size_t FindDataBlockIndexBySequenceNumber(uint64_t dataBlockSequenceNumber) const
        {
            auto itr = std::lower_bound(m_dataBlockInfos.begin(), m_dataBlockInfos.end(), dataBlockSequenceNumber, [](const TsfDataBlockInfo& lhs, uint64_t rhs) { return lhs.dataBlockSequenceNumber < rhs; });
            return (itr != m_dataBlockInfos.end() && itr->dataBlockSequenceNumber == dataBlockSequenceNumber) ? (itr - m_dataBlockInfos.begin()) : SIZE_MAX;
        }

        bool OpenDataBlock(size_t dataBlockIndex, DataBlockStorageReader& dataBlockStorageReader)
        {
            return dataBlockStorageReader.Initialize(m_dataBlockInfos[dataBlockIndex].filePath);
        }

    private:
        struct TsfDataBlockInfo
        {
            std::filesystem::path filePath;
            uint64_t dataBlockSequenceNumber = 0;
        };
        std::vector<TsfDataBlockInfo> m_dataBlockInfos;
    };

} // namespace nv::trace
