/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceStreamDatabaseWriter.h>
#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>
#include <NvTrace/Target/NvTraceClientLogger.h>

#include <cstdint>
#include <utility>

namespace nv::trace {

    template <
        typename TProviderState,
        typename TGpuPartitionState,
        typename TProcessState,
        typename TThreadEventProviderState,
        bool TWriteRecordLocations = false>
    class TsfSnapshotRecordWriter
    {
    public:
        using FnStageDataBlock = bool (*)(void* pUserData, TraceStreamDataBlockWriter&& dataBlockWriter);
        using FnObtainDataBlock = bool (*)(void* pUserData, TraceStreamDataBlockWriter& dataBlockWriter);
        using FnEncodeRecordLocation = uint64_t (*)(void* pUserData, uint64_t dataBlockSequenceNumber, uint32_t chunkIndex, uint32_t recordIndex);

        TsfSnapshotRecordWriter() = default;

        void Initialize(
            size_t maxMajorPayloadSize,
            FnEncodeRecordLocation fnEncodeRecordLocation,
            FnStageDataBlock fnStageDataBlock,
            FnObtainDataBlock fnObtainDataBlock,
            void* pUserData)
        {
            m_maxMajorPayloadSize = maxMajorPayloadSize;
            m_fnEncodeRecordLocation = fnEncodeRecordLocation;
            m_fnStageDataBlock = fnStageDataBlock;
            m_fnObtainDataBlock = fnObtainDataBlock;
            m_pUserData = pUserData;
            Reset();
        }

        void Reset() noexcept
        {
            m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            m_pProcessState = nullptr;
            m_pGpuPartitionState = nullptr;
            m_pProviderState = nullptr;
            m_chunkWriter.Reset();
            m_recordTimestampUpperBoundUtcNsec = 0;
            m_recordTimestampLowerBoundUtcNsec = 0;
            m_lastWrittenDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
            m_dataBlockWriter.Reset();
        }

        uint64_t LastCommittedRecordLocation() const
        {
            return m_fnEncodeRecordLocation(m_pUserData, m_dataBlockWriter.DataBlockSequenceNumber(), m_chunkWriter.ChunkIndex(), m_chunkWriter.GetLastCommittedRecordIndex());
        }

        uint64_t LastWrittenDataBlockSequenceNumber() const
        {
            if (m_dataBlockWriter.IsInitialized())
            {
                return m_dataBlockWriter.DataBlockSequenceNumber();
            }
            return m_lastWrittenDataBlockSequenceNumber;
        }

        void SetLastConsumeTimestampUtcNsec(uint64_t lastConsumeTimestampUtcNsec)
        {
            m_recordTimestampLowerBoundUtcNsec = m_recordTimestampUpperBoundUtcNsec;
            m_recordTimestampUpperBoundUtcNsec = lastConsumeTimestampUtcNsec;
            if (m_dataBlockWriter.IsInitialized())
            {
                m_dataBlockWriter.SetRecordTimestampUpperBoundUtcNsec(m_recordTimestampUpperBoundUtcNsec);
            }
        }

        void SetSystemInfoRecordLocation(uint64_t recordLocation)
        {
            if (m_dataBlockWriter.IsInitialized())
            {
                m_dataBlockWriter.SetSystemInfoRecordLocation(TWriteRecordLocations ? recordLocation : TSF_INVALID_RECORD_LOCATION);
            }
        }

        void SetNullProviderRef()
        {
            m_pProviderState = nullptr;
            m_pGpuPartitionState = nullptr;
            m_pProcessState = nullptr;
            m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
        }
        void SetSystemProviderRef(TProviderState& providerState)
        {
            m_pProviderState = &providerState;
            m_pGpuPartitionState = nullptr;
            m_pProcessState = nullptr;
            WriteProviderRef();
        }
        void SetGpuProviderRef(TProviderState& providerState, TGpuPartitionState& gpuPartitionState)
        {
            m_pProviderState = &providerState;
            m_pGpuPartitionState = &gpuPartitionState;
            m_pProcessState = nullptr;
            WriteProviderRef();
        }
        void SetProcessProviderRef(TProviderState& providerState, TProcessState& processState)
        {
            m_pProviderState = &providerState;
            m_pGpuPartitionState = nullptr;
            m_pProcessState = &processState;
            WriteProviderRef();
        }

        /// void fnWritePayload(TInlinePayload* pInlinePayload, TMajorPayload* pPayload);
        template <typename TFN_WritePayload>
        bool WriteRecord(uint32_t recordType, uint64_t correlationId, size_t majorPayloadSize, TFN_WritePayload&& fnWritePayload)
        {
            // [TODO] Avoid IsInitialized check by relying on the nullptr arithmetic check in HasSpaceForChunk()
            if (!m_chunkWriter.IsInitialized() || !m_chunkWriter.HasSpaceForRecord(majorPayloadSize))
            {
                if (m_maxMajorPayloadSize < majorPayloadSize)
                {
                    return false;
                }

                // Records staged across a chunk switch cannot cross it and roll back here; the next
                // record opens clean in the fresh chunk.
                if (m_chunkWriter.HasStagedRecords())
                {
                    NVTRACE_LOG_ERR(50, "Records were staged across a TSF chunk switch; rolling them back and refusing the record (recordType %u).", recordType);
                    m_chunkWriter.Rollback();
                    return false;
                }

                if (!AcquireNewChunk())
                {
                    return false;
                }
            }

            m_chunkWriter.WriteRecord(m_providerRefIndex, recordType, correlationId, majorPayloadSize, fnWritePayload);
            return true;
        }

        void Commit()
        {
            m_chunkWriter.Commit();
        }

        void Rollback()
        {
            m_chunkWriter.Rollback();
        }

        bool Checkpoint()
        {
            if (m_dataBlockWriter.IsInitialized())
            {
                SetLastWrittenDataBlockSequenceNumber(m_dataBlockWriter.DataBlockSequenceNumber());
            }
            if (!m_fnStageDataBlock(m_pUserData, std::move(m_dataBlockWriter)))
            {
                return false;
            }
            m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
            m_pProcessState = nullptr;
            m_pGpuPartitionState = nullptr;
            m_pProviderState = nullptr;
            m_chunkWriter.Reset();
            m_dataBlockWriter.Reset();
            return true;
        }

    private:
        bool AcquireNewDataBlock()
        {
            m_chunkWriter.Reset();
            m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;

            if (m_dataBlockWriter.IsInitialized())
            {
                SetLastWrittenDataBlockSequenceNumber(m_dataBlockWriter.DataBlockSequenceNumber());
            }
            if (!m_fnStageDataBlock(m_pUserData, std::move(m_dataBlockWriter)))
            {
                return false;
            }
            if (!m_fnObtainDataBlock(m_pUserData, m_dataBlockWriter))
            {
                return false;
            }

            m_dataBlockWriter.SetRecordTimestampLowerBoundUtcNsec(m_recordTimestampLowerBoundUtcNsec);
            m_dataBlockWriter.SetRecordTimestampUpperBoundUtcNsec(m_recordTimestampUpperBoundUtcNsec);

            WriteProviderRef();
            return true;
        }

        bool AcquireNewChunk()
        {
            m_chunkWriter.Reset();
            if (!m_dataBlockWriter.IsInitialized() || !m_dataBlockWriter.HasSpaceForChunk())
            {
                if (!AcquireNewDataBlock())
                {
                    return false;
                }
            }

            m_dataBlockWriter.AcquireChunk(m_chunkWriter);
            return true;
        }

        void WriteProviderRef()
        {
            if (!m_dataBlockWriter.IsInitialized() || !m_pProviderState)
            {
                return;
            }

            const uint64_t dataBlockSequenceNumber = m_dataBlockWriter.DataBlockSequenceNumber();

            // Determine which parent refs need to be written
            const bool requiresNewProviderRef = m_pProviderState->dataBlockSequenceNumber != dataBlockSequenceNumber;
            const bool requiresNewGpuPartitionRef = m_pGpuPartitionState && m_pGpuPartitionState->dataBlockSequenceNumber != dataBlockSequenceNumber;
            const bool requiresNewProcessRef = m_pProcessState && m_pProcessState->dataBlockSequenceNumber != dataBlockSequenceNumber;

            // ThreadEventProvider
            TThreadEventProviderState* const pThreadEventProviderState = m_pProcessState ? m_pProcessState->FindThreadEventProvider(m_pProviderState->threadEventProviderLuid) : nullptr;
            const bool hasThreadEventProviderMappingRef = pThreadEventProviderState && m_pProviderState->threadEventSourceProviderIndex != TProviderState::kInvalidThreadEventSourceProviderIndex;
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
            if (!m_dataBlockWriter.HasSpaceForProviderRef(requiredProviderRefCount) ||
                (requiresNewGpuPartitionRef && !m_dataBlockWriter.HasSpaceForGpuPartitionRef()) ||
                (requiresNewProcessRef && !m_dataBlockWriter.HasSpaceForProcessRef()) ||
                (requiresNewThreadEventProviderMappingRef && !m_dataBlockWriter.HasSpaceForThreadEventProviderMappingRef()))
            // clang-format on
            {
                if (m_dataBlockWriter.IsInitialized())
                {
                    SetLastWrittenDataBlockSequenceNumber(m_dataBlockWriter.DataBlockSequenceNumber());
                }
                if (!m_fnStageDataBlock(m_pUserData, std::move(m_dataBlockWriter)))
                {
                    return;
                }
                m_chunkWriter.Reset();
                m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
                return;
            }

            // Write parent refs if needed
            if (requiresNewGpuPartitionRef)
            {
                m_dataBlockWriter.WriteGpuPartitionRef(m_pGpuPartitionState->luid, GetGpuPartitionCreatedRecordLocation());
                m_pGpuPartitionState->dataBlockSequenceNumber = dataBlockSequenceNumber;
            }
            if (requiresNewProcessRef)
            {
                m_dataBlockWriter.WriteProcessRef(m_pProcessState->luid, GetProcessCreatedRecordLocation());
                m_pProcessState->dataBlockSequenceNumber = dataBlockSequenceNumber;
            }
            if (requiresNewThreadEventProviderRef)
            {
                pThreadEventProviderState->providerRefIndex = m_dataBlockWriter.WriteProviderRef(
                    pThreadEventProviderState->luid,
                    GetProviderCreatedRecordLocation(*pThreadEventProviderState),
                    GetProviderAttachedRecordLocation(*pThreadEventProviderState));
                pThreadEventProviderState->dataBlockSequenceNumber = dataBlockSequenceNumber;
            }
            if (requiresNewThreadEventProviderMappingRef)
            {
                m_dataBlockWriter.WriteThreadEventProviderMappingRef(
                    pThreadEventProviderState->luid,
                    m_pProviderState->luid,
                    m_pProviderState->threadEventSourceProviderIndex);
                m_pProviderState->threadEventProviderMappingRefDataBlockSequenceNumber = dataBlockSequenceNumber;
            }

            if (requiresNewProviderRef)
            {
                m_pProviderState->providerRefIndex = m_dataBlockWriter.WriteProviderRef(
                    m_pProviderState->luid,
                    GetProviderCreatedRecordLocation(*m_pProviderState),
                    GetProviderAttachedRecordLocation(*m_pProviderState));
                m_pProviderState->dataBlockSequenceNumber = dataBlockSequenceNumber;
            }
            m_providerRefIndex = m_pProviderState->providerRefIndex;
        }

    private:
        uint64_t GetGpuPartitionCreatedRecordLocation() const
        {
            if constexpr (TWriteRecordLocations)
            {
                return m_pGpuPartitionState->gpuPartitionCreatedRecordLocation;
            }
            else
            {
                return TSF_INVALID_RECORD_LOCATION;
            }
        }

        uint64_t GetProcessCreatedRecordLocation() const
        {
            if constexpr (TWriteRecordLocations)
            {
                return m_pProcessState->processCreatedRecordLocation;
            }
            else
            {
                return TSF_INVALID_RECORD_LOCATION;
            }
        }

        template <typename TState>
        static uint64_t GetProviderCreatedRecordLocation(const TState& state)
        {
            if constexpr (TWriteRecordLocations)
            {
                return state.providerCreatedRecordLocation;
            }
            else
            {
                return TSF_INVALID_RECORD_LOCATION;
            }
        }

        template <typename TState>
        static uint64_t GetProviderAttachedRecordLocation(const TState& state)
        {
            if constexpr (TWriteRecordLocations)
            {
                return state.providerAttachedRecordLocation;
            }
            else
            {
                return TSF_INVALID_RECORD_LOCATION;
            }
        }

        void SetLastWrittenDataBlockSequenceNumber(uint64_t dataBlockSequenceNumber)
        {
            m_lastWrittenDataBlockSequenceNumber = dataBlockSequenceNumber;
        }

        size_t m_maxMajorPayloadSize = 0;
        TraceStreamDataBlockWriter m_dataBlockWriter;
        uint64_t m_lastWrittenDataBlockSequenceNumber = TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER;
        uint64_t m_recordTimestampLowerBoundUtcNsec = 0;
        uint64_t m_recordTimestampUpperBoundUtcNsec = 0;
        FnEncodeRecordLocation m_fnEncodeRecordLocation = nullptr;
        FnStageDataBlock m_fnStageDataBlock = nullptr;
        FnObtainDataBlock m_fnObtainDataBlock = nullptr;
        void* m_pUserData = nullptr;
        TraceStreamChunkWriter m_chunkWriter;

        TProviderState* m_pProviderState = nullptr;
        TGpuPartitionState* m_pGpuPartitionState = nullptr;
        TProcessState* m_pProcessState = nullptr;

        uint32_t m_providerRefIndex = TSF_INVALID_OBJECT_REF_INDEX;
    };

} // namespace nv::trace
