/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceRecordWriterUtilities.h>
#include <NvTrace/Target/Consumer/NvTraceTsfBufferEventEncoders.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Observer/NvTraceCudaApiTraceObserver.h>

#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace nv::trace::CudaApiTraceSerialization {

    // Consumers write these ahead of every other state record: a partial load resolves them by
    // location, and the DataModel keys the records that follow on the devices.
    template <typename TWriter, typename TFN_OnCommittedRecord>
    void SerializeOnAttachRecords(CudaApiTraceObserver& observer, TWriter& writer, TFN_OnCommittedRecord&& fnOnCommittedRecord)
    {
        const auto configurationView = observer.GetProviderConfigurationFlatBufferView();
        if (configurationView.size == 0)
        {
            NVTRACE_LOG_ERR(50, "The CUDA API Trace provider published no configuration blob.");
        }
        else
        {
            const bool configurationWritten = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::CudaApiTrace>(writer, NvTrace_CUDA_ApiTrace_RecordType_Configuration, NVTRACE_NO_CORRELATION_ID, configurationView.pData, configurationView.size);
            if (configurationWritten)
            {
                writer.Commit();
                fnOnCommittedRecord();
            }
        }

        bool deviceWritten = false;
        observer.ConsumeAttachDeviceRecordBytes(
            [&](uint32_t /* deviceIndex */, const uint8_t* pRecordBytes, size_t recordByteCount) {
                deviceWritten = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::CudaApiTrace>(writer, NvTrace_CUDA_ApiTrace_RecordType_DeviceCreated, NVTRACE_NO_CORRELATION_ID, pRecordBytes, recordByteCount);
            },
            [&](bool consistencyCheckPassed) {
                NVTRACE_LOG_ERR_IF(!consistencyCheckPassed, 50, "Non-fatal failure draining the CUDA API Trace device records.");
                const bool deviceCommitted = CommitOrRollback(writer, consistencyCheckPassed && deviceWritten);
                deviceWritten = false;
                if (deviceCommitted)
                {
                    fnOnCommittedRecord();
                }
            });
    }

    // - Each record is written tentatively and committed only under the reader's verdict, so a record a
    //   racing writer disturbed is rolled back rather than emitted.
    // - Each record keeps its observer-supplied correlation ID and names its owner in its payload, so
    //   one blanket drain serves every state section but graph nodes.
    // - Graph nodes are drained after every other section, so in the TSF each exec precedes its nodes,
    //   whatever their order in the provider buffer. A partial load chases references backwards and
    //   stops collecting an exec's nodes once it reaches the exec's record.
    template <typename TWriter>
    void SerializeStateRecords(CudaApiTraceObserver& observer, TWriter& writer)
    {
        bool recordWritten = false;
        observer.ConsumeAttachStateRecords(
            [&](NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount) {
                if (recordType != NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated)
                {
                    recordWritten = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::CudaApiTrace>(writer, recordType, recordCorrelationId, pRecordBytes, recordByteCount);
                }
            },
            [&](bool consistencyCheckPassed) {
                const bool written = std::exchange(recordWritten, false);
                NVTRACE_LOG_ERR_IF(written && !consistencyCheckPassed, 50, "Non-fatal failure draining CUDA API Trace state records.");
                CommitOrRollback(writer, consistencyCheckPassed && written);
            });

        bool graphNodeWritten = false;
        observer.ConsumeAttachGraphNodeRecordBytes(
            [&](NvTrace_CorrelationId graphNodeCorrelationId, const uint8_t* pRecordBytes, size_t recordByteCount) {
                graphNodeWritten = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::CudaApiTrace>(writer, NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated, graphNodeCorrelationId, pRecordBytes, recordByteCount);
            },
            [&](bool consistencyCheckPassed) {
                const bool written = std::exchange(graphNodeWritten, false);
                NVTRACE_LOG_ERR_IF(written && !consistencyCheckPassed, 50, "Non-fatal failure draining CUDA API Trace graph node state.");
                CommitOrRollback(writer, consistencyCheckPassed && written);
            });
    }

    // A reference names an entry; the resolve delivers its whole group -- the module's functions ride
    // behind it, and an exec's nodes, if the observer serves them apart, follow in a second resolve.
    // The entry's own record, the first delivered, decides the group: an unknown entry record type
    // skips the group whole. Each record is its own transaction, committed under the state table's
    // verdict alone.
    template <typename TWriter>
    bool SerializeReferencedStateEvent(CudaApiTraceObserver& observer, TWriter& writer, const NvTrace_RecordReference& reference)
    {
        bool entryRecordDelivered = false;
        bool entryIsGraphExec = false;
        bool groupAccepted = false;
        bool eventSucceeded = false;
        bool graphExecCommitted = false;
        bool serializationSucceeded = true;
        observer.ResolveRecordReference(
            reference,
            [&](uint64_t /* entryRevision */, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                const bool isEntryRecord = !std::exchange(entryRecordDelivered, true);

                NvTrace_CorrelationId correlationId = NVTRACE_NO_CORRELATION_ID;
                if (isEntryRecord)
                {
                    switch (recordType)
                    {
                        case NvTrace_CUDA_ApiTrace_RecordType_ContextCreated:
                        case NvTrace_CUDA_ApiTrace_RecordType_StreamCreated:
                        case NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated:
                        case NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded:
                            break;
                        case NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated:
                            correlationId = observer.EncodeGraphExecCorrelationId(reference);
                            entryIsGraphExec = true;
                            break;
                        case NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated:
                            correlationId = observer.EncodeGraphNodeCorrelationId(reference);
                            break;
                        default:
                            NVTRACE_LOG_ERR(50, "Unknown CUDA state table record type %u.", recordType);
                            return;
                    }

                    groupAccepted = true;
                }

                if (!groupAccepted)
                {
                    return;
                }

                eventSucceeded = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::CudaApiTrace>(writer, recordType, correlationId, pPayload, payloadSize);
                serializationSucceeded = serializationSucceeded && eventSucceeded;
            },
            [&](bool consistencyCheckPassed) {
                // A failed verdict is not logged: routine after a destroy.
                if (std::exchange(eventSucceeded, false))
                {
                    const bool committed = CommitOrRollback(writer, consistencyCheckPassed);
                    graphExecCommitted = committed && entryIsGraphExec;
                }
            });

        if (!graphExecCommitted)
        {
            return serializationSucceeded;
        }

        // The exec's record leads its nodes on the wire, as at attach.
        observer.ResolveGraphExecNodeRecordBytes(
            reference,
            [&](uint32_t entryIndex, uint64_t entryRevision, const uint8_t* pRecordBytes, size_t recordByteCount) {
                const NvTrace_CorrelationId graphNodeCorrelationId = observer.EncodeGraphNodeCorrelationId(entryRevision, entryIndex);
                eventSucceeded = Codec::EventEncoders::WriteRecord<Codec::ProviderTypeId::CudaApiTrace>(writer, NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated, graphNodeCorrelationId, pRecordBytes, recordByteCount);
                serializationSucceeded = serializationSucceeded && eventSucceeded;
            },
            [&](bool consistencyCheckPassed) {
                const bool written = std::exchange(eventSucceeded, false);
                CommitOrRollback(writer, consistencyCheckPassed && written);
            });
        return serializationSucceeded;
    }

} // namespace nv::trace::CudaApiTraceSerialization
