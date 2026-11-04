/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Offline/DataModel/NvTraceDataModelBuilderBase.h>
#include <NvTrace/Offline/DataModel/NvTraceStreamDatabaseReader.h>
#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>

#include <cinttypes>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace nv::trace::Codec {

    template <typename TStorageReader>
    class DataModelBuilder<TraceStreamDatabaseReader<TStorageReader>> : public DataModelBuilderBase<TraceStreamDatabaseReader<TStorageReader>, DataModelBuilder<TraceStreamDatabaseReader<TStorageReader>>>
    {
        using Super = DataModelBuilderBase<TraceStreamDatabaseReader<TStorageReader>, DataModelBuilder>;
        friend Super;

        using Super::m_systemBuilderData;
        using typename Super::CudaApiTraceBuilderData;
        using typename Super::GpuPmCountersBuilderData;
        using typename Super::NvtxTraceBuilderData;
        using typename Super::ObjectType;
        using typename Super::ProviderBuilderData;
        using typename Super::ThreadEventProviderBuilderData;

        struct OrphanRecord
        {
            uint64_t recordLocation = 0;
            uint64_t providerHistoryByteCount = 0;
            uint32_t recordType = 0;
        };

    public:
        using ProviderInfo = typename Super::ProviderInfo;

        struct Options : Super::Options
        {
            TsfDataBlockRange dataBlockRange = {0, SIZE_MAX};
        };

        struct TraceVolume
        {
            uint64_t recordCount = 0;
            uint64_t byteCount = 0;

            TraceVolume& operator+=(const TraceVolume& rhs)
            {
                recordCount += rhs.recordCount;
                byteCount += rhs.byteCount;
                return *this;
            }
        };

        struct TraceVolumeStatistics
        {
            ProviderTypeTable<TraceVolume> historyByProviderType = {};
            TraceVolume totalHistory;
            TraceVolume routerHistory;
            TraceVolume gpuPmSamples;
            TraceVolume attach;
        };

    public:
        using Super::Super;

        const TraceVolumeStatistics& GetTraceVolumeStatistics() const
        {
            return m_traceVolumeStatistics;
        }

        // Returns false if a data block can't be read. The model is still built from the readable data blocks, but it is
        // missing data.
        bool Build(TraceStreamDatabaseReader<TStorageReader>& databaseReader, const Options& options = Options{})
        {
            m_traceVolumeStatistics = {};
            const uint64_t startTimeUtcNsec = options.observationStartUtcNsec != InvalidTimestamp ? options.observationStartUtcNsec : 0;
            const uint64_t endTimeUtcNsec = options.observationEndUtcNsec != InvalidTimestamp ? options.observationEndUtcNsec : UINT64_MAX;

            TsfDataBlockRange dataBlockRange;
            if (!databaseReader.FindDataBlocksInTimespan(startTimeUtcNsec, endTimeUtcNsec, dataBlockRange))
            {
                NVTRACE_LOG_ERR(50, "Failed to find the data blocks of the observation window");
                return false;
                }

            bool allDataBlocksRead = true;
            const auto checkDataBlockRead = [&allDataBlocksRead](bool dataBlockRead, size_t dataBlockIndex) {
                if (!dataBlockRead)
                {
                    NVTRACE_LOG_ERR(50, "Failed to read data block %zu", dataBlockIndex);
                    allDataBlocksRead = false;
                }
            };

            const TsfDataBlockRange availableDataBlockRange = databaseReader.GetDataBlockRange();
            dataBlockRange.beginDataBlockIndex = std::max(dataBlockRange.beginDataBlockIndex, std::min(options.dataBlockRange.beginDataBlockIndex, availableDataBlockRange.endDataBlockIndex));
            dataBlockRange.endDataBlockIndex = std::min(dataBlockRange.endDataBlockIndex, std::min(options.dataBlockRange.endDataBlockIndex, availableDataBlockRange.endDataBlockIndex));
            dataBlockRange.beginDataBlockIndex = std::min(dataBlockRange.beginDataBlockIndex, dataBlockRange.endDataBlockIndex);

            const bool isSnapshotDatabase = databaseReader.IsSnapshotDatabase();
            const TsfDataBlockRange selectedHistoryDataBlockRange =
                isSnapshotDatabase ? databaseReader.GetHistoryDataBlockRange(dataBlockRange) : dataBlockRange;
            Options effectiveOptions = options;
            // Adjust observation window if this is a snapshot database
            uint64_t snapshotObservationStartUtcNsec = 0;
            uint64_t snapshotObservationEndUtcNsec = 0;
            if (isSnapshotDatabase
                && (effectiveOptions.observationStartUtcNsec == InvalidTimestamp || effectiveOptions.observationEndUtcNsec == InvalidTimestamp)
                && databaseReader.GetDataBlockRangeTimeBoundsUtcNsec(
                    selectedHistoryDataBlockRange,
                    snapshotObservationStartUtcNsec,
                    snapshotObservationEndUtcNsec))
            {
                if (effectiveOptions.observationStartUtcNsec == InvalidTimestamp)
                {
                    effectiveOptions.observationStartUtcNsec = snapshotObservationStartUtcNsec;
                }
                if (effectiveOptions.observationEndUtcNsec == InvalidTimestamp)
                {
                    effectiveOptions.observationEndUtcNsec = snapshotObservationEndUtcNsec;
                }
            }

            Super::BeginBuilding(databaseReader, effectiveOptions);

            const uint64_t systemInfoRecordLocation = databaseReader.GetSystemInfoRecordLocation();
            if (systemInfoRecordLocation != TSF_INVALID_RECORD_LOCATION)
            {
                auto recordPayloads = databaseReader.GetRecordPayloads(systemInfoRecordLocation, NVTRACE_NIL_LUID, FB::Codec::NvTraceServiceEvents::EventTypeId_SystemCreated);
                if (recordPayloads.pMajorPayload)
                {
                    Super::AddSystemCreatedRecord(recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize);
                }
            }

            size_t firstDataBlockIndexForReferenceChasing = SIZE_MAX;
            std::map<NVTRACE_LUID, std::vector<OrphanRecord>> providerLuidToOrphanRecords;
            std::set<NVTRACE_LUID> rejectedProviderLuids;
            const bool readSnapshotDependencies =
                isSnapshotDatabase && !selectedHistoryDataBlockRange.Empty();
            const size_t firstDataBlockIndexToRead = readSnapshotDependencies ? 0 : dataBlockRange.beginDataBlockIndex;
            const size_t endDataBlockIndexToRead = readSnapshotDependencies ? availableDataBlockRange.endDataBlockIndex : dataBlockRange.endDataBlockIndex;
            for (size_t dataBlockIndex = firstDataBlockIndexToRead; dataBlockIndex < endDataBlockIndexToRead; ++dataBlockIndex)
            {
                if (readSnapshotDependencies
                    && !databaseReader.IsAttachDataBlock(dataBlockIndex)
                    && dataBlockIndex < selectedHistoryDataBlockRange.beginDataBlockIndex)
                {
                    // Advance directly to the first relevant history data block after finishing the attach blocks
                    dataBlockIndex = selectedHistoryDataBlockRange.beginDataBlockIndex;
                }

                const uint64_t dataBlockSequenceNumber = databaseReader.GetDataBlockSequenceNumber(dataBlockIndex);
                const bool gpuPartitionRefsRead = databaseReader.ForEachGpuPartitionRefInDataBlock(dataBlockIndex, [this, &databaseReader](const TsfGpuPartitionRef& gpuPartitionRef) {
                    if (m_systemBuilderData.gpuPartitions.find(gpuPartitionRef.gpuPartitionLuid) != m_systemBuilderData.gpuPartitions.end())
                    {
                        // Already added
                        return;
                    }

                    const auto recordPayloads = databaseReader.GetRecordPayloads(gpuPartitionRef.gpuPartitionCreatedRecordLocation, NVTRACE_NIL_LUID, FB::Codec::NvTraceServiceEvents::EventTypeId_GpuPartitionCreated);
                    if (!recordPayloads.pMajorPayload || !Super::AddGpuPartitionCreatedRecord(recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize))
                    {
                        return; // DFD
                    }
                });

                const bool processRefsRead = databaseReader.ForEachProcessRefInDataBlock(dataBlockIndex, [this, &databaseReader](const TsfProcessRef& processRef) {
                    if (m_systemBuilderData.processes.find(processRef.processLuid) != m_systemBuilderData.processes.end())
                    {
                        // Already added
                        return;
                    }

                    const auto recordPayloads = databaseReader.GetRecordPayloads(processRef.processCreatedRecordLocation, NVTRACE_NIL_LUID, FB::Codec::NvTraceServiceEvents::EventTypeId_ProcessCreated);
                    if (!recordPayloads.pMajorPayload || !Super::AddProcessCreatedRecord(recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize))
                    {
                        return; // DFD
                    }
                });

                /// Scan the DataBlock's ProviderRefs to add ProviderCreated, ProviderAttached, and on-attach records.
                const bool providerRefsRead = databaseReader.ForEachProviderRefInDataBlock(dataBlockIndex, [this, &databaseReader, firstDataBlockIndex = dataBlockRange.beginDataBlockIndex, &firstDataBlockIndexForReferenceChasing, &rejectedProviderLuids, &options](const TsfProviderRef& providerRef) {
                    if (m_systemBuilderData.providers.find(providerRef.providerLuid) != m_systemBuilderData.providers.end())
                    {
                        // The first ProviderRef restores the ProviderCreated and cumulative on-attach state that predates
                        // the selected range. Later on-attach records are processed by the normal forward DataBlock scan.
                        return;
                    }
                    if (rejectedProviderLuids.find(providerRef.providerLuid) != rejectedProviderLuids.end())
                    {
                        return;
                    }

                    const auto recordPayloads = databaseReader.GetRecordPayloads(providerRef.providerCreatedRecordLocation, NVTRACE_NIL_LUID, FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderCreated);
                    ProviderInfo providerInfo;
                    if (!recordPayloads.pMajorPayload || !Super::GetProviderInfoFromProviderCreated(recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize, providerInfo) || providerInfo.providerLuid != providerRef.providerLuid)
                    {
                        return;
                    }
                    ProviderBuilderData* const pProviderBuilderData = TryIngestNewProvider(providerInfo, recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize, options, rejectedProviderLuids);
                    if (!pProviderBuilderData)
                    {
                        return;
                    }

                    ProviderBuilderData& providerBuilderData = *pProviderBuilderData;

                    if (!AddProviderOnAttachRecords(databaseReader, providerRef, providerBuilderData, firstDataBlockIndex))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to add provider on-attach records (providerLuid=%" PRIu64 ")", providerRef.providerLuid);
                        return;
                    }

                    const ProviderType providerType = providerBuilderData.providerInfo.providerType;
                    switch (providerType)
                    {
                        case ProviderType::Invalid:
                        case ProviderType::Unknown:
                        case ProviderType::Count:
                            return;
                        case ProviderType::CudaApiTrace:
                        case ProviderType::NvtxTrace:
                        case ProviderType::ThreadEvent:
                            firstDataBlockIndexForReferenceChasing = std::min(firstDataBlockIndexForReferenceChasing, databaseReader.GetRecordDataBlockIndex(providerRef.providerCreatedRecordLocation));
                            break;

                        case ProviderType::GpuPmCounters:
                        case ProviderType::CpuClockCalibration:
                        case ProviderType::GpuClockCalibration:
                        case ProviderType::GpuHwTrace:
                        case ProviderType::GpuGrCtxTrace:
                            break;
                    }
                });

                const bool threadEventProviderMappingRefsRead = databaseReader.ForEachThreadEventProviderMappingRefInDataBlock(dataBlockIndex, [this](const TsfThreadEventProviderMappingRef& providerMappingRef) {
                    const auto itrProvider = m_systemBuilderData.providers.find(providerMappingRef.threadEventProviderLuid);
                    if (itrProvider == m_systemBuilderData.providers.end() || itrProvider->second->providerInfo.providerType != ProviderType::ThreadEvent)
                    {
                        return;
                    }

                    ThreadEventProviderBuilderData& threadEventProviderBuilderData = static_cast<ThreadEventProviderBuilderData&>(*itrProvider->second);
                    Super::ApplyThreadEventProviderMapping(
                        threadEventProviderBuilderData,
                        providerMappingRef.providerMappingId,
                        providerMappingRef.mappedProviderLuid);
                });

                const bool recordsRead = databaseReader.ForEachRecordInDataBlock(dataBlockIndex, [this, &databaseReader, &providerLuidToOrphanRecords, &rejectedProviderLuids, &options, dataBlockSequenceNumber](uint64_t recordLocation, NVTRACE_LUID providerLuid, uint32_t recordType, uint64_t correlationId, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize) {
                    const uint64_t providerHistoryByteCount = TallyTraceRecord(
                        dataBlockSequenceNumber,
                        providerLuid,
                        static_cast<uint64_t>(databaseReader.RecordHeaderAndInlinePayloadSize()) + maxMajorPayloadSize);
                    if (providerLuid == NVTRACE_NIL_LUID && recordType == FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderCreated)
                    {
                        ProviderInfo providerInfo;
                        if (!Super::GetProviderInfoFromProviderCreated(pMajorPayload, maxMajorPayloadSize, providerInfo))
                        {
                            return;
                        }

                        bool providerRejected = false;
                        ProviderBuilderData* const pProviderBuilderData = TryIngestProvider(providerInfo, pMajorPayload, maxMajorPayloadSize, options, rejectedProviderLuids, providerRejected);
                        if (!pProviderBuilderData)
                        {
                            if (providerRejected)
                            {
                                TallyAndEraseOrphanRecordsAsUnknown(providerInfo.providerLuid, providerLuidToOrphanRecords);
                            }
                            return;
                        }

                        ReplayOrphanRecords(databaseReader, *pProviderBuilderData, providerLuidToOrphanRecords);
                        return;
                    }

                    ProviderBuilderData* pProviderBuilderData = nullptr;
                    if (providerLuid != NVTRACE_NIL_LUID)
                    {
                        const auto itrProviderBuilderData = m_systemBuilderData.providers.find(providerLuid);
                        if (itrProviderBuilderData == m_systemBuilderData.providers.end())
                        {
                            if (rejectedProviderLuids.find(providerLuid) != rejectedProviderLuids.end())
                            {
                                TallyProviderHistoryRecord(ProviderType::Unknown, recordType, providerHistoryByteCount);
                                return;
                            }
                            providerLuidToOrphanRecords[providerLuid].push_back({recordLocation, providerHistoryByteCount, recordType});
                            return;
                        }
                        pProviderBuilderData = itrProviderBuilderData->second;
                        TallyProviderHistoryRecord(pProviderBuilderData->providerInfo.providerType, recordType, providerHistoryByteCount);
                    }

                    const bool recordAdded = pProviderBuilderData
                                               ? Super::AddProviderRecord(*pProviderBuilderData, recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize)
                                               : Super::AddRecord(recordLocation, providerLuid, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize);
                    if (!recordAdded)
                    {
                        return; // DFD
                    }

                    if (Super::CanRecordTypeEnableOrphanReplay(recordType))
                    {
                        if (pProviderBuilderData)
                        {
                            ReplayOrphanRecords(databaseReader, *pProviderBuilderData, providerLuidToOrphanRecords);
                        }
                    }
                });
                checkDataBlockRead(gpuPartitionRefsRead && processRefsRead && providerRefsRead && threadEventProviderMappingRefsRead && recordsRead, dataBlockIndex);
            }

            for (const auto& providerOrphanRecords : providerLuidToOrphanRecords)
            {
                TallyOrphanRecordsAsUnknown(providerOrphanRecords.second);
            }

            Super::FinalizeProviderOrphanRecords(providerLuidToOrphanRecords);

            // Search for referenced but undefined objects (reference chasing)
            for (size_t dataBlockIndex = dataBlockRange.beginDataBlockIndex; dataBlockIndex > firstDataBlockIndexForReferenceChasing && !m_referencedUndefinedObjects.empty();)
            {
                --dataBlockIndex;

                const bool recordsRead = databaseReader.ForEachRecordInDataBlockBackwards(dataBlockIndex, [this](uint64_t recordLocation, NVTRACE_LUID providerLuid, uint32_t recordType, uint64_t correlationId, const uint8_t* pInlinePayload, const uint8_t* pMajorPayload, size_t maxMajorPayloadSize) {
                    auto itrProvider = m_systemBuilderData.providers.find(providerLuid);
                    if (itrProvider == m_systemBuilderData.providers.end())
                    {
                        return;
                    }
                    ProviderBuilderData& providerBuilderData = *itrProvider->second;

                    bool definesReferencedObject = false;
                    const auto checkObjectIdentifier = [&](ObjectType objectType, uint64_t objectId) {
                        const ObjectIdentifier objectIdentifier = {providerLuid, objectType, objectId};
                        definesReferencedObject |= m_referencedUndefinedObjects.find(objectIdentifier) != m_referencedUndefinedObjects.end();
                    };
                    switch (providerBuilderData.providerInfo.providerType)
                    {
                        case ProviderType::Invalid:
                        case ProviderType::Unknown:
                        case ProviderType::Count:
                            return;
                        case ProviderType::CudaApiTrace:
                        {
                            const bool handled = Super::TryCheckCudaApiTraceObjectIdentifiers(recordType, correlationId, pMajorPayload, maxMajorPayloadSize, checkObjectIdentifier);
                            if (!handled)
                            {
                                return;
                            }

                            if (definesReferencedObject)
                            {
                                if (!Super::AddCudaApiTraceRecord(static_cast<CudaApiTraceBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize))
                                {
                                    return; // DFD
                                }
                            }

                            break;
                        }

                        case ProviderType::NvtxTrace:
                        {
                            const bool handled = Super::TryCheckNvtxTraceObjectIdentifiers(recordType, correlationId, pMajorPayload, maxMajorPayloadSize, checkObjectIdentifier);
                            if (!handled)
                            {
                                return;
                            }

                            if (definesReferencedObject)
                            {
                                if (!Super::AddNvtxTraceRecord(static_cast<NvtxTraceBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize))
                                {
                                    return;
                                }
                            }

                            break;
                        }

                        case ProviderType::ThreadEvent:
                        {
                            switch (static_cast<FB::Codec::ThreadLifetimeEvents::EventTypeId>(recordType))
                            {
                                case FB::Codec::ThreadLifetimeEvents::EventTypeId_ProviderMapped:
                                    NVTRACE_LOG_ERR_RETURN_IF(!flatbuffers::Verifier(pMajorPayload, maxMajorPayloadSize).VerifyBuffer<FB::Codec::ThreadLifetimeEvents::ProviderMapped>(), /*void*/, 50, "Invalid ThreadEvent ProviderMapped record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);
                                    checkObjectIdentifier(ObjectType::ThreadEventProviderMapping, flatbuffers::GetRoot<FB::Codec::ThreadLifetimeEvents::ProviderMapped>(pMajorPayload)->providerMappingId());
                                    break;

                                case FB::Codec::ThreadLifetimeEvents::EventTypeId_ThreadCreated:
                                    NVTRACE_LOG_ERR_RETURN_IF(maxMajorPayloadSize < NvTrace_ThreadEvent_ThreadCreated_MINOR_VERSION_A, /*void*/, 50, "Invalid ThreadEvent ThreadCreated record (payload=%p size=%zu)", static_cast<const void*>(pMajorPayload), maxMajorPayloadSize);
                                    checkObjectIdentifier(ObjectType::ThreadEventThread, reinterpret_cast<const NvTrace_ThreadEvent_ThreadCreated*>(pMajorPayload)->uniqueThreadId);
                                    break;

                                default:
                                    return;
                            }

                            if (definesReferencedObject)
                            {
                                if (!Super::AddThreadEventRecord(static_cast<ThreadEventProviderBuilderData&>(providerBuilderData), recordLocation, recordType, correlationId, pInlinePayload, pMajorPayload, maxMajorPayloadSize))
                                {
                                    return; // DFD
                                }
                            }

                            break;
                        }

                        case ProviderType::CpuClockCalibration:
                        case ProviderType::GpuClockCalibration:
                        case ProviderType::GpuPmCounters:
                        case ProviderType::GpuHwTrace:
                        case ProviderType::GpuGrCtxTrace:
                            return;
                    }
                });
                checkDataBlockRead(recordsRead, dataBlockIndex);
            }

            Super::EndBuilding();
            return allDataBlocksRead;
        }

    private:
        // Tallies the categories that do not require provider metadata and returns the history byte count
        // that still needs to be classified by provider type. Zero means no provider tally is needed.
        uint64_t TallyTraceRecord(uint64_t dataBlockSequenceNumber, NVTRACE_LUID providerLuid, uint64_t byteCount)
        {
            const TraceVolume volume = {1, byteCount};
            if (TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW <= dataBlockSequenceNumber && dataBlockSequenceNumber <= TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH)
            {
                m_traceVolumeStatistics.attach += volume;
                return 0;
            }

            m_traceVolumeStatistics.totalHistory += volume;
            if (providerLuid == NVTRACE_NIL_LUID)
            {
                m_traceVolumeStatistics.routerHistory += volume;
                return 0;
            }
            return byteCount;
        }

        void TallyProviderHistoryRecord(ProviderType providerType, uint32_t recordType, uint64_t byteCount)
        {
            if (!byteCount)
            {
                return;
            }
            if (providerType == ProviderType::Invalid || providerType == ProviderType::Count)
            {
                providerType = ProviderType::Unknown;
            }

            const TraceVolume volume = {1, byteCount};
            m_traceVolumeStatistics.historyByProviderType[static_cast<size_t>(providerType)] += volume;
            if (providerType == ProviderType::GpuPmCounters && recordType == NvTrace_GPU_PmCounters_RecordType_CounterSample)
            {
                m_traceVolumeStatistics.gpuPmSamples += volume;
            }
        }

        void TallyOrphanRecordsAsUnknown(const std::vector<OrphanRecord>& orphanRecords)
        {
            for (const OrphanRecord& orphanRecord : orphanRecords)
            {
                TallyProviderHistoryRecord(ProviderType::Unknown, orphanRecord.recordType, orphanRecord.providerHistoryByteCount);
            }
        }

        void TallyAndEraseOrphanRecordsAsUnknown(
            NVTRACE_LUID providerLuid,
            std::map<NVTRACE_LUID, std::vector<OrphanRecord>>& providerLuidToOrphanRecords)
        {
            const auto itrOrphanRecords = providerLuidToOrphanRecords.find(providerLuid);
            if (itrOrphanRecords == providerLuidToOrphanRecords.end())
            {
                return;
            }

            TallyOrphanRecordsAsUnknown(itrOrphanRecords->second);
            providerLuidToOrphanRecords.erase(itrOrphanRecords);
        }

        ProviderBuilderData* TryIngestProvider(
            const ProviderInfo& providerInfo,
            const uint8_t* pMajorPayload,
            size_t maxMajorPayloadSize,
            const Options& options,
            std::set<NVTRACE_LUID>& rejectedProviderLuids,
            bool& out_providerRejected)
        {
            out_providerRejected = false;
            const auto itrProviderBuilderData = m_systemBuilderData.providers.find(providerInfo.providerLuid);
            if (itrProviderBuilderData != m_systemBuilderData.providers.end())
            {
                return itrProviderBuilderData->second;
            }
            if (rejectedProviderLuids.find(providerInfo.providerLuid) != rejectedProviderLuids.end())
            {
                out_providerRejected = true;
                return nullptr;
            }

            return TryIngestNewProvider(providerInfo, pMajorPayload, maxMajorPayloadSize, options, rejectedProviderLuids, out_providerRejected);
        }

        ProviderBuilderData* TryIngestNewProvider(
            const ProviderInfo& providerInfo,
            const uint8_t* pMajorPayload,
            size_t maxMajorPayloadSize,
            const Options& options,
            std::set<NVTRACE_LUID>& rejectedProviderLuids,
            bool& out_providerRejected)
        {
            out_providerRejected = false;
            if (options.fnShouldIngestProvider && !options.fnShouldIngestProvider(providerInfo))
            {
                rejectedProviderLuids.insert(providerInfo.providerLuid);
                out_providerRejected = true;
                return nullptr;
            }

            ProviderBuilderData* pProviderBuilderData = nullptr;
            if (!Super::AddProviderCreatedRecord(pMajorPayload, maxMajorPayloadSize, providerInfo, pProviderBuilderData))
            {
                return nullptr;
            }
            return pProviderBuilderData;
        }

        ProviderBuilderData* TryIngestNewProvider(
            const ProviderInfo& providerInfo,
            const uint8_t* pMajorPayload,
            size_t maxMajorPayloadSize,
            const Options& options,
            std::set<NVTRACE_LUID>& rejectedProviderLuids)
        {
            bool providerRejected = false;
            return TryIngestNewProvider(providerInfo, pMajorPayload, maxMajorPayloadSize, options, rejectedProviderLuids, providerRejected);
        }

        template <typename TReader>
        bool AddProviderOnAttachRecords(TReader& reader, const TsfProviderRef& providerRef, ProviderBuilderData& providerBuilderData, size_t firstDataBlockIndex)
        {
            if (providerRef.providerAttachedRecordLocation == TSF_INVALID_RECORD_LOCATION)
            {
                return true;
            }

            const auto providerAttachedPayloads = reader.GetRecordPayloads(providerRef.providerAttachedRecordLocation, NVTRACE_NIL_LUID, FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderAttached);
            if (!providerAttachedPayloads.pMajorPayload || !flatbuffers::Verifier(providerAttachedPayloads.pMajorPayload, providerAttachedPayloads.maxMajorPayloadSize).VerifyBuffer<FB::Codec::NvTraceServiceEvents::ProviderAttached>())
            {
                NVTRACE_LOG_ERR(50, "Invalid ProviderAttached record (payload=%p size=%zu)", static_cast<const void*>(providerAttachedPayloads.pMajorPayload), providerAttachedPayloads.maxMajorPayloadSize);
                return false;
            }
            const auto* pProviderAttached = flatbuffers::GetRoot<FB::Codec::NvTraceServiceEvents::ProviderAttached>(providerAttachedPayloads.pMajorPayload);
            const auto* pProviderLuid = pProviderAttached->providerLuid();
            if (!pProviderLuid)
            {
                NVTRACE_LOG_ERR(50, "ProviderAttached record is missing its provider LUID");
                return false;
            }
            const NVTRACE_LUID providerLuid = *reinterpret_cast<const NVTRACE_LUID*>(pProviderLuid);
            if (providerLuid != providerRef.providerLuid)
            {
                NVTRACE_LOG_ERR(50, "ProviderAttached record LUID mismatch (expected=%" PRIu64 " actual=%" PRIu64 ")", providerRef.providerLuid, providerLuid);
                return false;
            }
            const auto* pRecordLocations = pProviderAttached->onAttachRecordLocations();
            if (!pRecordLocations)
            {
                return true;
            }

            for (const uint64_t recordLocation : *pRecordLocations)
            {
                const size_t recordDataBlockIndex = reader.GetRecordDataBlockIndex(recordLocation);
                if (recordDataBlockIndex == SIZE_MAX)
                {
                    NVTRACE_LOG_ERR(50, "Invalid on-attach record location (recordLocation=%" PRIu64 ")", recordLocation);
                    return false;
                }
                if (firstDataBlockIndex <= recordDataBlockIndex)
                {
                    // This record is inside the continuously scanned range and will be processed by the normal record scan
                    continue;
                }

                const typename TReader::RecordMetadata recordMetadata = reader.GetRecordMetadata(recordLocation);
                const auto recordPayloads = reader.GetRecordPayloads(recordLocation, providerRef.providerLuid, recordMetadata.recordType);
                if (!recordPayloads.pMajorPayload || !Super::AddProviderRecord(providerBuilderData, recordLocation, recordMetadata.recordType, recordMetadata.correlationId, recordPayloads.pInlinePayload, recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize))
                {
                    NVTRACE_LOG_ERR(50, "Failed to add on-attach record (recordLocation=%" PRIu64 " recordType=%" PRIu32 ")", recordLocation, recordMetadata.recordType);
                    return false;
                }
            }

            return true;
        }

        void ReplayOrphanRecords(
            TraceStreamDatabaseReader<TStorageReader>& databaseReader,
            ProviderBuilderData& providerBuilderData,
            std::map<NVTRACE_LUID, std::vector<OrphanRecord>>& providerLuidToOrphanRecords)
        {
            Super::ReplayProviderOrphanRecords(providerBuilderData, providerLuidToOrphanRecords, [this, &databaseReader, &providerBuilderData](const OrphanRecord& orphanRecord) {
                TallyProviderHistoryRecord(providerBuilderData.providerInfo.providerType, orphanRecord.recordType, orphanRecord.providerHistoryByteCount);
                const auto recordPayloads = databaseReader.GetRecordPayloads(orphanRecord.recordLocation);
                Super::AddProviderRecord(providerBuilderData, orphanRecord.recordLocation, orphanRecord.recordType, databaseReader.GetRecordMetadata(orphanRecord.recordLocation).correlationId, recordPayloads.pInlinePayload, recordPayloads.pMajorPayload, recordPayloads.maxMajorPayloadSize);
            });
        }

        void OnObjectDefined(NVTRACE_LUID providerLuid, ObjectType objectType, uint64_t objectId)
        {
            ObjectIdentifier objectIdentifier = {providerLuid, objectType, objectId};
            if (m_definedObjects.emplace(objectIdentifier).second)
            {
                m_referencedUndefinedObjects.erase(objectIdentifier);
            }
        }

        void OnObjectReferenced(NVTRACE_LUID providerLuid, ObjectType objectType, uint64_t objectId)
        {
            ObjectIdentifier objectIdentifier = {providerLuid, objectType, objectId};
            if (m_definedObjects.find(objectIdentifier) == m_definedObjects.end())
            {
                m_referencedUndefinedObjects.insert(objectIdentifier);
            }
        }

        struct ObjectIdentifier
        {
            NVTRACE_LUID providerLuid;
            ObjectType objectType;
            uint64_t objectId;

            bool operator<(const ObjectIdentifier& other) const
            {
                return providerLuid < other.providerLuid || (providerLuid == other.providerLuid && ((int)objectType < (int)other.objectType || (objectType == other.objectType && objectId < other.objectId)));
            }
        };
        std::set<ObjectIdentifier> m_definedObjects;
        std::set<ObjectIdentifier> m_referencedUndefinedObjects;
        TraceVolumeStatistics m_traceVolumeStatistics;
    };

} // namespace nv::trace::Codec
