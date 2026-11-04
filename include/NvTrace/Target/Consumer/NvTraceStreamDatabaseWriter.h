/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>
#include <NvTrace/Target/Consumer/NvTraceStreamUtilities.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceSectionLayoutBuilder.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    template <typename TFN_WritePayloads, typename = void>
    struct GetWritePayloadTypes
    {
    };
    template <typename TInlinePayload, typename TMajorPayload>
    struct GetWritePayloadTypes<void (*)(TInlinePayload*, TMajorPayload*)>
    {
        using InlinePayloadType = TInlinePayload;
        using MajorPayloadType = TMajorPayload;
    };
    template <typename TLambda, typename TInlinePayload, typename TMajorPayload>
    struct GetWritePayloadTypes<void (TLambda::*)(TInlinePayload*, TMajorPayload*) const>
    {
        using InlinePayloadType = TInlinePayload;
        using MajorPayloadType = TMajorPayload;
    };
    template <typename TFN_WritePayloads>
    struct GetWritePayloadTypes<TFN_WritePayloads, std::void_t<decltype(&std::remove_reference_t<TFN_WritePayloads>::operator())>>
        : GetWritePayloadTypes<decltype(&std::remove_reference_t<TFN_WritePayloads>::operator())>
    {
    };


    class TraceStreamChunkWriter
    {
    public:
        ~TraceStreamChunkWriter() = default;
        TraceStreamChunkWriter() = default;
        TraceStreamChunkWriter(const TraceStreamChunkWriter&) = default;
        TraceStreamChunkWriter& operator=(const TraceStreamChunkWriter&) = default;

        void Reset() noexcept
        {
            m_committedMajorPayloadOffsetDiv8 = 0;
            m_lastMajorPayloadOffsetDiv8 = 0;
            m_pCommittedMajorPayloadPos = nullptr;
            m_pLastMajorPayloadWritePos = nullptr;

            m_writtenRecordCount = 0;
            m_pCommittedRecordNextPos = nullptr;
            m_pRecordWritePos = nullptr;

            m_pChunkState = &s_dummyChunkState;
            m_chunkIndex = TSF_INVALID_CHUNK_INDEX;
        }

        void Initialize(uint32_t chunkIndex, TsfChunkState* pChunkState, uint8_t* pChunk, size_t chunkSize)
        {
            m_chunkIndex = chunkIndex;
            m_pChunkState = pChunkState;

            m_pRecordWritePos = reinterpret_cast<TsfRecordWithInlinePayload*>(pChunk);
            m_pCommittedRecordNextPos = m_pRecordWritePos;

            m_pLastMajorPayloadWritePos = pChunk + chunkSize;
            m_pCommittedMajorPayloadPos = m_pLastMajorPayloadWritePos;

            m_pChunkState->validRecordCount = 0;
        }

        bool IsInitialized() const
        {
            return m_chunkIndex != TSF_INVALID_CHUNK_INDEX;
        }

        uint32_t ChunkIndex() const
        {
            return m_chunkIndex;
        }

        uint32_t GetLastCommittedRecordIndex() const
        {
            return m_pChunkState->validRecordCount - 1;
        }

        size_t CommittedByteCount() const
        {
            return static_cast<size_t>(m_pChunkState->validRecordCount) * sizeof(TsfRecordWithInlinePayload)
                 + static_cast<size_t>(m_committedMajorPayloadOffsetDiv8) * sizeof(uint64_t);
        }

        bool HasStagedRecords() const
        {
            return m_pCommittedRecordNextPos != m_pRecordWritePos;
        }

        bool HasSpaceForRecord(size_t majorPayloadSize) const
        {
            return reinterpret_cast<uint8_t*>(m_pRecordWritePos + 1) + majorPayloadSize <= m_pLastMajorPayloadWritePos;
        }

        // Precondition: HasSpaceForRecord(majorPayloadSize)
        template <typename TFN_WritePayloads>
        void WriteRecord(uint32_t providerRefIndex, uint32_t recordType, uint64_t correlationId, size_t majorPayloadSize, TFN_WritePayloads&& fnWritePayloads)
        {
            using PayloadTypes = GetWritePayloadTypes<TFN_WritePayloads>;
            using InlinePayloadType = typename PayloadTypes::InlinePayloadType;
            using MajorPayloadType = typename PayloadTypes::MajorPayloadType;

            const uint32_t majorPayloadSizeDiv8 = static_cast<uint32_t>((majorPayloadSize + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            const uint32_t majorPayloadOffsetDiv8 = m_lastMajorPayloadOffsetDiv8 + majorPayloadSizeDiv8;
            uint8_t* const pMajorPayloadWritePos = m_pLastMajorPayloadWritePos - majorPayloadSizeDiv8 * sizeof(uint64_t);

            m_pRecordWritePos->record.packed = NVTRACE_SETFIELD64_UNCHECKED(0, NVTRACE_TSF_RECORD_PROVIDER_REF_INDEX, providerRefIndex) | NVTRACE_SETFIELD64_UNCHECKED(0, NVTRACE_TSF_RECORD_RECORD_TYPE, recordType) | NVTRACE_SETFIELD64_UNCHECKED(0, NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8, majorPayloadOffsetDiv8);
            m_pRecordWritePos->record.correlationId = correlationId;
            fnWritePayloads(reinterpret_cast<InlinePayloadType*>(m_pRecordWritePos->inlinePayload), reinterpret_cast<MajorPayloadType*>(pMajorPayloadWritePos));

            m_pRecordWritePos += 1;
            m_writtenRecordCount += 1;

            m_pLastMajorPayloadWritePos = pMajorPayloadWritePos;
            m_lastMajorPayloadOffsetDiv8 = majorPayloadOffsetDiv8;
        }

        void Commit()
        {
            m_pCommittedRecordNextPos = m_pRecordWritePos;
            m_pChunkState->validRecordCount = m_writtenRecordCount;

            m_pCommittedMajorPayloadPos = m_pLastMajorPayloadWritePos;
            m_committedMajorPayloadOffsetDiv8 = m_lastMajorPayloadOffsetDiv8;
        }

        void Rollback()
        {
            m_pRecordWritePos = m_pCommittedRecordNextPos;
            m_writtenRecordCount = m_pChunkState->validRecordCount;

            m_pLastMajorPayloadWritePos = m_pCommittedMajorPayloadPos;
            m_lastMajorPayloadOffsetDiv8 = m_committedMajorPayloadOffsetDiv8;
        }

    private:
        // A dummy chunk state is used when the chunk writer is not initialized.
        // Its purpose is to avoid null pointer dereferences when Commit() or Rollback() are called on an uninitialized chunk writer.
        inline static TsfChunkState s_dummyChunkState = {};

        uint32_t m_chunkIndex = TSF_INVALID_CHUNK_INDEX;
        TsfChunkState* m_pChunkState = &s_dummyChunkState;

        TsfRecordWithInlinePayload* m_pRecordWritePos = nullptr;
        TsfRecordWithInlinePayload* m_pCommittedRecordNextPos = nullptr;
        uint32_t m_writtenRecordCount = 0;

        uint8_t* m_pLastMajorPayloadWritePos = nullptr;
        uint8_t* m_pCommittedMajorPayloadPos = nullptr;
        uint32_t m_lastMajorPayloadOffsetDiv8 = 0;
        uint32_t m_committedMajorPayloadOffsetDiv8 = 0;
    };

    template <typename TStorageWriter>
    class TsfSnapshotConsumer;

    class TraceStreamDataBlockWriter
    {
        // TsfSnapshotConsumer needs access to m_sections for patching
        // Could make m_sections public
        template <typename TStorageWriter>
        friend class TsfSnapshotConsumer;

    public:
        ~TraceStreamDataBlockWriter() = default;
        TraceStreamDataBlockWriter() = default;
        TraceStreamDataBlockWriter(const TraceStreamDataBlockWriter&) = default;
        TraceStreamDataBlockWriter& operator=(const TraceStreamDataBlockWriter&) = default;
        TraceStreamDataBlockWriter(TraceStreamDataBlockWriter&& rhs) noexcept
        {
            *this = std::move(rhs);
        }
        TraceStreamDataBlockWriter& operator=(TraceStreamDataBlockWriter&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_writeSlotIndex = rhs.m_writeSlotIndex;
                m_dataBlockSequenceNumber = rhs.m_dataBlockSequenceNumber;
                m_sections = rhs.m_sections;
                rhs.Reset();
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_sections = {};
            m_dataBlockSequenceNumber = 0;
            m_writeSlotIndex = UINT8_MAX;
        }

        void Initialize(uint8_t writeSlotIndex, uint64_t dataBlockSequenceNumber, uint8_t* pDataBlock, uint64_t recordTimestampLowerBoundUtcNsec, uint64_t recordTimestampUpperBoundUtcNsec)
        {
            m_writeSlotIndex = writeSlotIndex;
            m_dataBlockSequenceNumber = dataBlockSequenceNumber;

            m_sections.pDataBlockHeader = reinterpret_cast<TsfDataBlockHeader*>(pDataBlock);
            m_sections.pDataBlockState = reinterpret_cast<TsfDataBlockState*>(pDataBlock + m_sections.pDataBlockHeader->offsetofDataBlockState);
            m_sections.pGpuPartitionRefs = reinterpret_cast<TsfGpuPartitionRef*>(pDataBlock + m_sections.pDataBlockHeader->offsetofGpuPartitionRefs);
            m_sections.pProcessRefs = reinterpret_cast<TsfProcessRef*>(pDataBlock + m_sections.pDataBlockHeader->offsetofProcessRefs);
            m_sections.pProviderRefs = reinterpret_cast<TsfProviderRef*>(pDataBlock + m_sections.pDataBlockHeader->offsetofProviderRefs);
            m_sections.pThreadEventProviderMappingRefs = reinterpret_cast<TsfThreadEventProviderMappingRef*>(pDataBlock + m_sections.pDataBlockHeader->offsetofThreadEventProviderMappingRefs);
            m_sections.pChunkStates = reinterpret_cast<TsfChunkState*>(pDataBlock + m_sections.pDataBlockHeader->offsetofChunkStates);
            m_sections.pChunks = reinterpret_cast<uint8_t*>(pDataBlock + m_sections.pDataBlockHeader->offsetofChunks);

            std::memset(m_sections.pDataBlockState, 0, sizeof(TsfDataBlockState));
            m_sections.pDataBlockState->dataBlockSequenceNumber = dataBlockSequenceNumber;
            m_sections.pDataBlockState->recordTimestampLowerBoundUtcNsec = recordTimestampLowerBoundUtcNsec;
            m_sections.pDataBlockState->recordTimestampUpperBoundUtcNsec = recordTimestampUpperBoundUtcNsec;
        }

        bool IsInitialized() const
        {
            return m_sections.pDataBlockHeader != nullptr;
        }

        uint8_t WriteSlotIndex() const
        {
            return m_writeSlotIndex;
        }

        uint64_t DataBlockSequenceNumber() const
        {
            return m_dataBlockSequenceNumber;
        }

        uint64_t RecordTimestampLowerBoundUtcNsec() const
        {
            return m_sections.pDataBlockState->recordTimestampLowerBoundUtcNsec;
        }

        uint64_t RecordTimestampUpperBoundUtcNsec() const
        {
            return m_sections.pDataBlockState->recordTimestampUpperBoundUtcNsec;
        }

        void SetRecordTimestampLowerBoundUtcNsec(uint64_t recordTimestampLowerBoundUtcNsec)
        {
            m_sections.pDataBlockState->recordTimestampLowerBoundUtcNsec = recordTimestampLowerBoundUtcNsec;
        }

        void SetRecordTimestampUpperBoundUtcNsec(uint64_t recordTimestampUpperBoundUtcNsec)
        {
            m_sections.pDataBlockState->recordTimestampUpperBoundUtcNsec = recordTimestampUpperBoundUtcNsec;
        }

        void SetSystemInfoRecordLocation(uint64_t systemInfoRecordLocation)
        {
            m_sections.pDataBlockState->systemInfoRecordLocation = systemInfoRecordLocation;
        }

        bool HasSpaceForGpuPartitionRef() const
        {
            return m_sections.pDataBlockState->validGpuPartitionRefCount < m_sections.pDataBlockHeader->maxGpuPartitionRefCount;
        }

        // Precondition: HasSpaceForGpuPartitionRef()
        uint32_t WriteGpuPartitionRef(NVTRACE_LUID gpuPartitionLuid, uint64_t gpuPartitionCreatedRecordLocation)
        {
            const uint32_t gpuPartitionRefIndex = m_sections.pDataBlockState->validGpuPartitionRefCount;
            m_sections.pGpuPartitionRefs[gpuPartitionRefIndex] = {gpuPartitionLuid, gpuPartitionCreatedRecordLocation};
            m_sections.pDataBlockState->validGpuPartitionRefCount += 1;

            return gpuPartitionRefIndex;
        }

        bool HasSpaceForProcessRef() const
        {
            return m_sections.pDataBlockState->validProcessRefCount < m_sections.pDataBlockHeader->maxProcessRefCount;
        }

        // Precondition: HasSpaceForProcessRef()
        uint32_t WriteProcessRef(NVTRACE_LUID processLuid, uint64_t processCreatedRecordLocation)
        {
            const uint32_t processRefIndex = m_sections.pDataBlockState->validProcessRefCount;
            m_sections.pProcessRefs[processRefIndex] = {processLuid, processCreatedRecordLocation};
            m_sections.pDataBlockState->validProcessRefCount += 1;

            return processRefIndex;
        }

        bool HasSpaceForProviderRef(uint32_t requiredCount = 1) const
        {
            return m_sections.pDataBlockState->validProviderRefCount + requiredCount <= m_sections.pDataBlockHeader->maxProviderRefCount;
        }

        // Precondition: HasSpaceForProviderRef()
        uint32_t WriteProviderRef(NVTRACE_LUID providerLuid, uint64_t providerCreatedRecordLocation, uint64_t providerAttachedRecordLocation)
        {
            const uint32_t providerRefIndex = m_sections.pDataBlockState->validProviderRefCount;
            m_sections.pProviderRefs[providerRefIndex] = {providerLuid, providerCreatedRecordLocation, providerAttachedRecordLocation};
            m_sections.pDataBlockState->validProviderRefCount += 1;

            return providerRefIndex;
        }

        bool HasSpaceForThreadEventProviderMappingRef() const
        {
            return m_sections.pDataBlockState->validThreadEventProviderMappingRefCount < m_sections.pDataBlockHeader->maxThreadEventProviderMappingRefCount;
        }

        // Precondition: HasSpaceForThreadEventProviderMappingRef()
        uint32_t WriteThreadEventProviderMappingRef(NVTRACE_LUID threadEventProviderLuid, NVTRACE_LUID mappedProviderLuid, uint32_t providerMappingId)
        {
            const uint32_t mappingRefIndex = m_sections.pDataBlockState->validThreadEventProviderMappingRefCount;
            m_sections.pThreadEventProviderMappingRefs[mappingRefIndex] = {
                threadEventProviderLuid,
                mappedProviderLuid,
                providerMappingId};
            m_sections.pDataBlockState->validThreadEventProviderMappingRefCount += 1;

            return mappingRefIndex;
        }

        bool HasSpaceForChunk() const
        {
            return m_sections.pDataBlockState->validChunkCount < m_sections.pDataBlockHeader->maxChunkCount;
        }

        // Precondition: HasSpaceForChunk() && !chunkWriter.IsInitialized()
        void AcquireChunk(TraceStreamChunkWriter& chunkWriter)
        {
            const uint32_t chunkIndex = m_sections.pDataBlockState->validChunkCount;
            chunkWriter.Initialize(chunkIndex, m_sections.pChunkStates + chunkIndex, m_sections.pChunks + chunkIndex * m_sections.pDataBlockHeader->sizeofChunk, m_sections.pDataBlockHeader->sizeofChunk);
            m_sections.pDataBlockState->validChunkCount += 1;
        }

    private:
        uint8_t m_writeSlotIndex = UINT8_MAX;
        uint64_t m_dataBlockSequenceNumber = 0;

        struct Sections
        {
            TsfDataBlockHeader* pDataBlockHeader = nullptr;
            TsfDataBlockState* pDataBlockState = nullptr;
            TsfGpuPartitionRef* pGpuPartitionRefs = nullptr;
            TsfProcessRef* pProcessRefs = nullptr;
            TsfProviderRef* pProviderRefs = nullptr;
            TsfThreadEventProviderMappingRef* pThreadEventProviderMappingRefs = nullptr;
            TsfChunkState* pChunkStates = nullptr;
            uint8_t* pChunks = nullptr;
        };
        Sections m_sections;
    };

    template <typename TStorageWriter>
    class TraceStreamDatabaseWriter
    {
    public:
        using OnStorageWriteCompleted = void (*)(void* pUserData, uint8_t writeSlotIndex, uint64_t dataBlockSequenceNumber, bool succeeded);

        struct Options
        {
            size_t dataBlockSize = 10 * 1024 * 1024; // 10 MiB
            uint32_t maxGpuPartitionCountPerDataBlock = 32;
            uint32_t maxProcessCountPerDataBlock = 32;
            uint32_t maxProviderCountPerDataBlock = 64;
            uint32_t maxThreadEventProviderMappingCountPerDataBlock = 64;
            uint32_t chunkCountPerDataBlock = 1;
            uint8_t writeSlotCount = 2;
            uint8_t reservedSlotCount = 0;
            void (*fnOnStorageWriteCompleted)(void* pUserData, uint8_t writeSlotIndex, uint64_t dataBlockSequenceNumber, bool isStorageWriteCompletedSuccessfully) = nullptr;
            void* pUserDataOnStorageWriteCompleted = nullptr;
        };

    public:
        ~TraceStreamDatabaseWriter()
        {
            TerminateStorageWriteThread();
        }

        TraceStreamDatabaseWriter() = default;

        TraceStreamDatabaseWriter(const TraceStreamDatabaseWriter&) = delete;
        TraceStreamDatabaseWriter& operator=(const TraceStreamDatabaseWriter&) = delete;

        void Reset() noexcept
        {
            TerminateStorageWriteThread();
            m_terminateStorageWriteThreadRequested = false;
            m_pendingStorageWriteSlots = {};
            m_availableWriteSlots = {};
            m_availableReservedWriteSlots = {};
            m_reservedSlotCount = 0;
            m_writeSlots.clear();
            m_pWriteSlotsBackingMemory.reset();
            m_nextDataBlockSequenceNumber = TSF_DATA_BLOCK_SEQUENCE_NUMBER_LOW;
            m_nextReservedDataBlockSequenceNumber = TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW;
            m_recordLocationEncoder.Reset();
            m_maxMajorPayloadSize = 0;
            m_dataBlockHeader = {};
            m_systemInfoRecordLocation = TSF_INVALID_RECORD_LOCATION;
            m_onStorageWriteCompleted = nullptr;
            m_pUserDataOnStorageWriteCompleted = nullptr;
            m_storageWriter.Reset();
        }

        template <typename TFN_InitializeStorageWriter>
        bool Initialize(TFN_InitializeStorageWriter&& fnInitializeStorageWriter)
        {
            return Initialize(Options(), std::forward<TFN_InitializeStorageWriter>(fnInitializeStorageWriter));
        }

        template <typename TFN_InitializeStorageWriter>
        bool Initialize(const Options& options, TFN_InitializeStorageWriter&& fnInitializeStorageWriter)
        {
            auto resetGuard = ScopeExitGuard([this]() { Reset(); });

            if (!fnInitializeStorageWriter(m_storageWriter))
            {
                return false;
            }

            SectionLayoutBuilder dataBlockLayoutBuilder;
            dataBlockLayoutBuilder.AppendSection<TsfDataBlockHeader>();

            m_dataBlockHeader.version.magic = TRACE_STREAM_FORMAT_MAGIC;
            m_dataBlockHeader.version.majorVersion = TRACE_STREAM_FORMAT_MAJOR_VERSION;
            m_dataBlockHeader.version.sizeofDataBlockHeader = sizeof(TsfDataBlockHeader);

            m_dataBlockHeader.sizeofDataBlock = options.dataBlockSize;

            m_dataBlockHeader.offsetofDataBlockState = dataBlockLayoutBuilder.AppendSection<TsfDataBlockState>();
            m_dataBlockHeader.sizeofDataBlockState = sizeof(TsfDataBlockState);
            m_dataBlockHeader._reserved0 = 0;

            m_dataBlockHeader.offsetofGpuPartitionRefs = dataBlockLayoutBuilder.AppendSection<TsfGpuPartitionRef>(options.maxGpuPartitionCountPerDataBlock);
            m_dataBlockHeader.sizeofGpuPartitionRef = sizeof(TsfGpuPartitionRef);
            m_dataBlockHeader.maxGpuPartitionRefCount = options.maxGpuPartitionCountPerDataBlock;

            m_dataBlockHeader.offsetofProcessRefs = dataBlockLayoutBuilder.AppendSection<TsfProcessRef>(options.maxProcessCountPerDataBlock);
            m_dataBlockHeader.sizeofProcessRef = sizeof(TsfProcessRef);
            m_dataBlockHeader.maxProcessRefCount = options.maxProcessCountPerDataBlock;

            m_dataBlockHeader.offsetofProviderRefs = dataBlockLayoutBuilder.AppendSection<TsfProviderRef>(options.maxProviderCountPerDataBlock);
            m_dataBlockHeader.sizeofProviderRef = sizeof(TsfProviderRef);
            m_dataBlockHeader.maxProviderRefCount = options.maxProviderCountPerDataBlock;

            m_dataBlockHeader.offsetofThreadEventProviderMappingRefs = dataBlockLayoutBuilder.AppendSection<TsfThreadEventProviderMappingRef>(options.maxThreadEventProviderMappingCountPerDataBlock);
            m_dataBlockHeader.sizeofThreadEventProviderMappingRef = sizeof(TsfThreadEventProviderMappingRef);
            m_dataBlockHeader.maxThreadEventProviderMappingRefCount = options.maxThreadEventProviderMappingCountPerDataBlock;

            m_dataBlockHeader.offsetofChunkStates = dataBlockLayoutBuilder.AppendSection<TsfChunkState>(options.chunkCountPerDataBlock);
            m_dataBlockHeader.sizeofChunkState = sizeof(TsfChunkState);

            constexpr size_t minChunkSize = 1024; // 1 KiB
            constexpr size_t maxChunkSize = (uint32_t(1) << NVTRACE_FIELD_RANGE_SIZE(NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8)) * sizeof(uint64_t);
            const size_t remainingDataBlockSize = dataBlockLayoutBuilder.GetEndOffset() < options.dataBlockSize ? options.dataBlockSize - dataBlockLayoutBuilder.GetEndOffset() : 0;
            const size_t chunkSize = AlignDnP2(std::clamp(remainingDataBlockSize / options.chunkCountPerDataBlock, minChunkSize, maxChunkSize), sizeof(uint64_t));
            m_dataBlockHeader.offsetofChunks = dataBlockLayoutBuilder.AppendSection(options.chunkCountPerDataBlock * chunkSize, sizeof(uint64_t));
            m_dataBlockHeader.sizeofChunk = static_cast<uint32_t>(chunkSize);
            m_dataBlockHeader.maxChunkCount = options.chunkCountPerDataBlock;

            m_dataBlockHeader.sizeofRecord = sizeof(TsfRecord);
            m_dataBlockHeader.sizeofInlinePayload = TSF_RECORD_INLINE_PAYLOAD_SIZE;
            m_dataBlockHeader.maxRecordCountPerChunk = m_dataBlockHeader.sizeofChunk / sizeof(TsfRecordWithInlinePayload);

            NVTRACE_LOG_ERR_RETURN_IF(options.dataBlockSize < dataBlockLayoutBuilder.GetEndOffset(), false, 20, "Data block size (%zu bytes) is too small for the requested maximum resource and chunkcounts (at least %zu bytes required)", options.dataBlockSize, dataBlockLayoutBuilder.GetEndOffset());
            NVTRACE_LOG_WRN_IF(dataBlockLayoutBuilder.GetEndOffset() < options.dataBlockSize, 20, "Data block is not fully utilized, %zu bytes are wasted at the end", options.dataBlockSize - dataBlockLayoutBuilder.GetEndOffset());

            m_maxMajorPayloadSize = AlignDnP2(chunkSize - sizeof(TsfRecordWithInlinePayload), sizeof(uint64_t));

            bool succeeded = m_recordLocationEncoder.Initialize(options.chunkCountPerDataBlock, m_dataBlockHeader.maxRecordCountPerChunk);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 20, "TraceStreamRecordLocationEncoder::Initialize() failed");

            NVTRACE_LOG_ERR_RETURN_IF(options.writeSlotCount < 2, false, 20, "writeSlotCount must be at least 2");
            NVTRACE_LOG_ERR_RETURN_IF(static_cast<uint16_t>(options.writeSlotCount) + options.reservedSlotCount > UINT8_MAX, false, 20, "writeSlotCount (%u) + reservedSlotCount (%u) exceeds %" PRIu8, static_cast<unsigned int>(options.writeSlotCount), static_cast<unsigned int>(options.reservedSlotCount), UINT8_MAX);

            const uint8_t totalSlotCount = static_cast<uint8_t>(options.writeSlotCount + options.reservedSlotCount);
            m_reservedSlotCount = options.reservedSlotCount;

            m_pWriteSlotsBackingMemory = std::make_unique<uint8_t[]>(totalSlotCount * m_dataBlockHeader.sizeofDataBlock);
            m_writeSlots.resize(totalSlotCount);

            auto initializeSlotRange = [&](uint8_t firstSlotIndex, uint8_t slotCount, bool isReserved, WriteSlotQueue& availableQueue) {
                if (!slotCount)
                {
                    availableQueue = {};
                    return;
                }

                const uint8_t lastSlotIndex = static_cast<uint8_t>(static_cast<uint16_t>(firstSlotIndex) + slotCount - 1);
                for (uint8_t slotOffset = 0; slotOffset < slotCount; ++slotOffset)
                {
                    const uint8_t writeSlotIndex = static_cast<uint8_t>(static_cast<uint16_t>(firstSlotIndex) + slotOffset);
                    DataBlockWriteSlot& writeSlot = m_writeSlots[writeSlotIndex];
                    writeSlot.pBackingMemory = m_pWriteSlotsBackingMemory.get() + writeSlotIndex * m_dataBlockHeader.sizeofDataBlock;
                    std::memcpy(writeSlot.pBackingMemory, &m_dataBlockHeader, sizeof(TsfDataBlockHeader));
                    writeSlot.isReserved = isReserved;
                    writeSlot.nextSlotIndex = static_cast<uint8_t>(writeSlotIndex + 1);
                }
                m_writeSlots[lastSlotIndex].nextSlotIndex = UINT8_MAX;
                availableQueue.firstSlotIndex = firstSlotIndex;
                availableQueue.lastSlotIndex = lastSlotIndex;
            };

            initializeSlotRange(0, options.writeSlotCount, false, m_availableWriteSlots);
            initializeSlotRange(options.writeSlotCount, options.reservedSlotCount, true, m_availableReservedWriteSlots);

            m_pUserDataOnStorageWriteCompleted = options.pUserDataOnStorageWriteCompleted;
            m_onStorageWriteCompleted = options.fnOnStorageWriteCompleted;

            try
            {
                m_storageWriteThread = std::thread(&TraceStreamDatabaseWriter::StorageWriteLoop, this);
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(20, "Failed to create storage write thread: %s", e.what());
                return false;
            }

            resetGuard.Dismiss();
            return true;
        }

        bool IsReservedDataBlock(uint64_t dataBlockSequenceNumber) const
        {
            return dataBlockSequenceNumber >= TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW && dataBlockSequenceNumber <= TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH;
        }

        size_t MaxMajorPayloadSize() const
        {
            return m_maxMajorPayloadSize;
        }

        size_t ChunkSize() const
        {
            return m_dataBlockHeader.sizeofChunk;
        }

        uint64_t EncodeRecordLocation(uint64_t dataBlockSequenceNumber, uint32_t chunkIndex, uint32_t recordIndex) const
        {
            return m_recordLocationEncoder.EncodeRecordLocation(dataBlockSequenceNumber, chunkIndex, recordIndex);
        }

        void SetSystemInfoRecordLocation(uint64_t systemInfoRecordLocation)
        {
            m_systemInfoRecordLocation = systemInfoRecordLocation;
        }

        // Precondition: !dataBlockWriter.IsInitialized()
        bool AcquireDataBlock(TraceStreamDataBlockWriter& dataBlockWriter, uint64_t recordTimestampLowerBoundUtcNsec = 0, uint64_t recordTimestampUpperBoundUtcNsec = 0, bool isReserved = false)
        {
            NVTRACE_LOG_ERR_RETURN_IF(isReserved && m_nextReservedDataBlockSequenceNumber > TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH, false, 20, "Reserved data block sequence numbers exhausted");

            WriteSlotQueue& availableQueue = isReserved ? m_availableReservedWriteSlots : m_availableWriteSlots;
            const uint8_t writeSlotIndex = DequeueWriteSlot(availableQueue);
            if (writeSlotIndex == UINT8_MAX)
            {
                if (isReserved)
                {
                    NVTRACE_LOG_WRN(20, "No available reserved write slots (reservedSlotCount=%u)", static_cast<unsigned int>(m_reservedSlotCount));
                }
                else
                {
                    NVTRACE_LOG_WRN(20, "Not an error for TSF snapshot consumer as it will always try to acquire a new data block from here first");
                    NVTRACE_LOG_WRN(20, "No available write slots - storage writes may be too slow or writeSlotCount too small (writeSlotCount=%zu reservedSlotCount=%u)", m_writeSlots.size() - m_reservedSlotCount, static_cast<unsigned int>(m_reservedSlotCount));
                }
                return false;
            }
            DataBlockWriteSlot& writeSlot = m_writeSlots[writeSlotIndex];

            uint64_t dataBlockSequenceNumber;
            if (isReserved)
            {
                dataBlockSequenceNumber = m_nextReservedDataBlockSequenceNumber;
                m_nextReservedDataBlockSequenceNumber += 1;
            }
            else
            {
                dataBlockSequenceNumber = m_nextDataBlockSequenceNumber;
                m_nextDataBlockSequenceNumber += 1; // Not checking for overflow because TSF_DATA_BLOCK_SEQUENCE_NUMBER_HIGH is UINT64_MAX
            }
            writeSlot.dataBlockSequenceNumber = dataBlockSequenceNumber;
            dataBlockWriter.Initialize(writeSlotIndex, dataBlockSequenceNumber, writeSlot.pBackingMemory, recordTimestampLowerBoundUtcNsec, recordTimestampUpperBoundUtcNsec);
            dataBlockWriter.SetSystemInfoRecordLocation(m_systemInfoRecordLocation);

            return true;
        }

        // Returns the write slot owned by dataBlockWriter to the available pool
        // without writing it to storage. Precondition: dataBlockWriter.IsInitialized().
        bool RecycleDataBlock(TraceStreamDataBlockWriter& dataBlockWriter)
        {
            const uint8_t writeSlotIndex = dataBlockWriter.WriteSlotIndex();
            NVTRACE_LOG_ERR_RETURN_IF(writeSlotIndex >= m_writeSlots.size(), false, 20, "Write slot index %u is out of range (writeSlotCount=%zu)", static_cast<unsigned int>(writeSlotIndex), m_writeSlots.size());

            DataBlockWriteSlot& writeSlot = m_writeSlots[writeSlotIndex];
            NVTRACE_LOG_ERR_RETURN_IF(writeSlot.dataBlockSequenceNumber == TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER, false, 20, "Write slot %u does not contain a retained DataBlock", static_cast<unsigned int>(writeSlotIndex));

            writeSlot.dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            dataBlockWriter.Reset();
            EnqueueWriteSlot(AvailableQueue(writeSlot.isReserved), writeSlotIndex);
            return true;
        }

        bool HasDataBlockInStorage(uint64_t dataBlockSequenceNumber) const
        {
            return m_storageWriter.ExistsDataBlock(dataBlockSequenceNumber);
        }

        // Precondition: dataBlockWriter.IsInitialized()
        bool FinalizeDataBlock(TraceStreamDataBlockWriter& dataBlockWriter)
        {
            const uint8_t writeSlotIndex = dataBlockWriter.WriteSlotIndex();
            NVTRACE_LOG_ERR_RETURN_IF(writeSlotIndex >= m_writeSlots.size(), false, 20, "Write slot index %u is out of range (writeSlotCount=%zu)", static_cast<unsigned int>(writeSlotIndex), m_writeSlots.size());

            dataBlockWriter.Reset();

            EnqueueWriteSlot(m_pendingStorageWriteSlots, writeSlotIndex);
            m_pendingStorageWriteCV.notify_one();

            return true;
        }

        uint32_t StorageWriteFailureCount() const
        {
            return m_storageWriteFailureCount;
        }

        // Callers must synchronize any operation that is not safe to perform
        // concurrently with the storage write thread.
        TStorageWriter& StorageWriter()
        {
            return m_storageWriter;
        }

        const TStorageWriter& StorageWriter() const
        {
            return m_storageWriter;
        }

    private:
        struct DataBlockWriteSlot
        {
            uint8_t* pBackingMemory = nullptr;
            uint64_t dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            uint8_t nextSlotIndex = 0;
            bool isReserved = false;
        };

        struct WriteSlotQueue
        {
            uint8_t firstSlotIndex = UINT8_MAX;
            uint8_t lastSlotIndex = UINT8_MAX;
        };

        void EnqueueWriteSlot(WriteSlotQueue& queue, uint8_t slotIndex)
        {
            std::lock_guard<std::mutex> lock(m_writeSlotQueuesMutex);

            m_writeSlots[slotIndex].nextSlotIndex = UINT8_MAX;

            if (queue.lastSlotIndex != UINT8_MAX)
            {
                m_writeSlots[queue.lastSlotIndex].nextSlotIndex = slotIndex;
            }
            else
            {
                queue.firstSlotIndex = slotIndex;
            }
            queue.lastSlotIndex = slotIndex;
        }

        // Returns UINT8_MAX if queue is empty
        uint8_t DequeueWriteSlot(WriteSlotQueue& queue)
        {
            std::lock_guard<std::mutex> lock(m_writeSlotQueuesMutex);
            return DequeueWriteSlotUnderLock(queue);
        }

        // Returns UINT8_MAX if queue is empty. Precondition: m_writeSlotQueuesMutex is held
        uint8_t DequeueWriteSlotUnderLock(WriteSlotQueue& queue)
        {
            const uint8_t slotIndex = queue.firstSlotIndex;
            if (slotIndex == UINT8_MAX)
            {
                return UINT8_MAX;
            }
            DataBlockWriteSlot& slot = m_writeSlots[slotIndex];

            queue.firstSlotIndex = slot.nextSlotIndex;
            if (queue.firstSlotIndex == UINT8_MAX)
            {
                queue.lastSlotIndex = UINT8_MAX;
            }

            return slotIndex;
        }

        WriteSlotQueue& AvailableQueue(bool isReserved)
        {
            return isReserved ? m_availableReservedWriteSlots : m_availableWriteSlots;
        }

    private:
        void StorageWriteLoop()
        {
            while (true)
            {
                uint8_t writeSlotIndex;
                {
                    std::unique_lock<std::mutex> lock(m_writeSlotQueuesMutex);
                    m_pendingStorageWriteCV.wait(lock, [this]() { return m_pendingStorageWriteSlots.firstSlotIndex != UINT8_MAX || m_terminateStorageWriteThreadRequested; });
                    if (m_pendingStorageWriteSlots.firstSlotIndex == UINT8_MAX)
                    {
                        break;
                    }

                    writeSlotIndex = DequeueWriteSlotUnderLock(m_pendingStorageWriteSlots);
                }
                DataBlockWriteSlot& writeSlot = m_writeSlots[writeSlotIndex];
                const uint64_t dataBlockSequenceNumber = writeSlot.dataBlockSequenceNumber;

                // Write outside lock
                bool succeeded = false;
                try
                {
                    succeeded = m_storageWriter.WriteDataBlock(writeSlotIndex, writeSlot.pBackingMemory, m_dataBlockHeader.sizeofDataBlock, dataBlockSequenceNumber);
                }
                catch (const std::exception& e)
                {
                    NVTRACE_LOG_ERR(20, "Storage write threw for data block %llu: %s", dataBlockSequenceNumber, e.what());
                }
                catch (...)
                {
                    NVTRACE_LOG_ERR(20, "Storage write threw an unknown exception for data block %llu", dataBlockSequenceNumber);
                }
                if (!succeeded)
                {
                    NVTRACE_LOG_ERR(20, "Failed to write data block to storage: %llu", dataBlockSequenceNumber);
                    m_storageWriteFailureCount += 1;
                }

                if (m_onStorageWriteCompleted)
                {
                    try
                    {
                        m_onStorageWriteCompleted(m_pUserDataOnStorageWriteCompleted, writeSlotIndex, dataBlockSequenceNumber, succeeded);
                    }
                    catch (const std::exception& e)
                    {
                        NVTRACE_LOG_ERR(20, "Storage write completion callback threw: %s", e.what());
                    }
                    catch (...)
                    {
                        NVTRACE_LOG_ERR(20, "Storage write completion callback threw an unknown exception");
                    }
                }

                // Enqueue to the matching available pool
                writeSlot.dataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
                EnqueueWriteSlot(AvailableQueue(writeSlot.isReserved), writeSlotIndex);
            }
        }

        void TerminateStorageWriteThread()
        {
            if (m_storageWriteThread.joinable())
            {
                {
                    std::lock_guard<std::mutex> lock(m_writeSlotQueuesMutex);
                    m_terminateStorageWriteThreadRequested = true;
                }
                m_pendingStorageWriteCV.notify_one();
                m_storageWriteThread.join();
            }
        }

    private:
        TStorageWriter m_storageWriter;

        TsfDataBlockHeader m_dataBlockHeader = {};
        size_t m_maxMajorPayloadSize = 0;
        TraceStreamRecordLocationEncoder m_recordLocationEncoder;

        uint64_t m_systemInfoRecordLocation = TSF_INVALID_RECORD_LOCATION;

        uint64_t m_nextDataBlockSequenceNumber = TSF_DATA_BLOCK_SEQUENCE_NUMBER_LOW;
        uint64_t m_nextReservedDataBlockSequenceNumber = TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW;

        std::unique_ptr<uint8_t[]> m_pWriteSlotsBackingMemory = nullptr;
        std::vector<DataBlockWriteSlot> m_writeSlots;

        // Accessed by both main thread and storage write thread
        // All accesses protected by m_writeSlotQueuesMutex
        WriteSlotQueue m_availableWriteSlots;
        WriteSlotQueue m_availableReservedWriteSlots;
        WriteSlotQueue m_pendingStorageWriteSlots;
        uint8_t m_reservedSlotCount = 0;

        std::mutex m_writeSlotQueuesMutex;
        std::condition_variable m_pendingStorageWriteCV;

        OnStorageWriteCompleted m_onStorageWriteCompleted;
        void* m_pUserDataOnStorageWriteCompleted = nullptr;

        std::thread m_storageWriteThread;
        bool m_terminateStorageWriteThreadRequested = false;

        uint32_t m_storageWriteFailureCount = 0;
    };
    class TsfDiskDatabaseBufferedStorageWriter
    {
    public:
        ~TsfDiskDatabaseBufferedStorageWriter() = default;
        TsfDiskDatabaseBufferedStorageWriter() = default;
        TsfDiskDatabaseBufferedStorageWriter(const TsfDiskDatabaseBufferedStorageWriter&) = delete;
        TsfDiskDatabaseBufferedStorageWriter(TsfDiskDatabaseBufferedStorageWriter&& rhs) noexcept = default;
        TsfDiskDatabaseBufferedStorageWriter& operator=(const TsfDiskDatabaseBufferedStorageWriter&) = delete;
        TsfDiskDatabaseBufferedStorageWriter& operator=(TsfDiskDatabaseBufferedStorageWriter&& rhs) noexcept = default;

        static bool HardlinkDataBlock(
            TsfDiskDatabaseBufferedStorageWriter& dstStorageWriter,
            const TsfDiskDatabaseBufferedStorageWriter& srcStorageWriter,
            uint64_t dataBlockSequenceNumber)
        {
            std::filesystem::path srcDataBlockFilePath;
            NVTRACE_LOG_ERR_RETURN_IF(!srcStorageWriter.GetDataBlockFilePath(srcDataBlockFilePath, dataBlockSequenceNumber), false, 20, "Failed to get source data block file path");

            std::filesystem::path dstDataBlockFilePath;
            NVTRACE_LOG_ERR_RETURN_IF(!dstStorageWriter.GetDataBlockFilePath(dstDataBlockFilePath, dataBlockSequenceNumber), false, 20, "Failed to get destination data block file path");

            std::error_code errorCode;
            const bool sourceExists = std::filesystem::exists(srcDataBlockFilePath, errorCode);
            NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to check source data block file at %s", srcDataBlockFilePath.c_str());
            NVTRACE_LOG_ERR_RETURN_IF(!sourceExists, false, 20, "Source data block file does not exist: %s", srcDataBlockFilePath.c_str());

            const bool destinationExists = std::filesystem::exists(dstDataBlockFilePath, errorCode);
            NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to check destination data block file at %s", dstDataBlockFilePath.c_str());
            if (destinationExists)
            {
                return true;
            }

            std::filesystem::create_hard_link(srcDataBlockFilePath, dstDataBlockFilePath, errorCode);
            NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to hardlink data block %" PRIu64 " from %s to %s", dataBlockSequenceNumber, srcDataBlockFilePath.c_str(), dstDataBlockFilePath.c_str());
            return true;
        }

        void Reset() noexcept
        {
            m_databaseDirectoryPath.clear();
        }

        bool Initialize(const std::filesystem::path& databaseDirectoryPath)
        {
            std::error_code errorCode;
            std::filesystem::create_directories(databaseDirectoryPath, errorCode);
            NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to create database directory at %s", databaseDirectoryPath.c_str());

            bool isEmpty = std::filesystem::is_empty(databaseDirectoryPath, errorCode);
            NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to check if database directory at %s is empty", databaseDirectoryPath.c_str());
            NVTRACE_LOG_ERR_RETURN_IF(!isEmpty, false, 20, "Database directory at %s is not empty", databaseDirectoryPath.c_str());

            m_databaseDirectoryPath = databaseDirectoryPath;

            return true;
        }

        bool ExistsDataBlock(uint64_t dataBlockSequenceNumber) const
        {
            std::filesystem::path dataBlockFilePath;
            if (!GetDataBlockFilePath(dataBlockFilePath, dataBlockSequenceNumber))
            {
                return false;
            }
            return std::filesystem::exists(dataBlockFilePath);
        }

        bool HardlinkDataBlock(TsfDiskDatabaseBufferedStorageWriter& dstStorageWriter, uint64_t dataBlockSequenceNumber) const
        {
            return HardlinkDataBlock(dstStorageWriter, *this, dataBlockSequenceNumber);
        }

        // Does not overwrite if data block files already exist
        bool WriteDataBlock(uint8_t, const uint8_t* pDataBlock, size_t dataBlockSize, uint64_t dataBlockSequenceNumber)
        {
            std::filesystem::path dataBlockFilePath;
            if (!GetDataBlockFilePath(dataBlockFilePath, dataBlockSequenceNumber))
            {
                return false;
            }
            std::error_code errorCode;
            const bool dataBlockFileExists = std::filesystem::exists(dataBlockFilePath, errorCode);
            NVTRACE_LOG_ERR_RETURN_IF(errorCode != std::error_code{}, false, 20, "Failed to check data block file at %s", dataBlockFilePath.c_str());
            if (!dataBlockFileExists)
            {
                std::ofstream file(dataBlockFilePath, std::ios::binary);
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to open file for writing: %s", dataBlockFilePath.c_str());

                file.write(reinterpret_cast<const char*>(pDataBlock), dataBlockSize);
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to write data block to file: %s", dataBlockFilePath.c_str());
                file.close();
                NVTRACE_LOG_ERR_RETURN_IF(file.fail(), false, 20, "Failed to close data block file: %s", dataBlockFilePath.c_str());
            }
            return true;
        }

    private:
        bool GetDataBlockFilePath(std::filesystem::path& out_dataBlockFilePath, uint64_t dataBlockSequenceNumber) const
        {
            std::stringstream dataBlockSequenceNumberString;
            dataBlockSequenceNumberString << std::hex << dataBlockSequenceNumber;
            out_dataBlockFilePath = m_databaseDirectoryPath / (dataBlockSequenceNumberString.str() + ".nvtsf");
            return true;
        }

        std::filesystem::path m_databaseDirectoryPath;
    };
} // namespace nv::trace
