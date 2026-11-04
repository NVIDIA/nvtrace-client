/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/Consumer/NvTraceRecordWriterUtilities.h>
#include <NvTrace/Target/Consumer/NvTraceTsfBufferEventEncoders.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Observer/NvTraceNvtxTraceObserver.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace nv::trace::NvtxTraceSerialization {

    // Consumers write this ahead of every other state record, so a partial load resolves it by location.
    template <typename TWriter, typename TFN_OnCommittedRecord>
    void SerializeOnAttachRecords(NvtxTraceObserver& observer, TWriter& writer, TFN_OnCommittedRecord&& fnOnCommittedRecord)
    {
        const auto configurationView = observer.GetProviderConfigurationFlatBufferView();
        if (configurationView.size == 0)
        {
            NVTRACE_LOG_ERR(50, "The NVTX Trace provider published no configuration blob.");
            return;
        }

        const bool configurationWritten = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::NvtxTrace>(writer, NvTrace_NvtxTrace_RecordType_Configuration, NVTRACE_NO_CORRELATION_ID, configurationView.pData, configurationView.size);
        if (configurationWritten)
        {
            writer.Commit();
            fnOnCommittedRecord();
        }
    }

    // Each record is written into the writer while still tentative and committed only under the
    // reader's verdict for it, so a record a racing writer disturbed is rolled back rather than emitted.
    // Every record goes out under its own address and names what it belongs to in its payload, so one
    // drain serves every state section and needs no per-record-type handling.
    //
    // Strings reach the wire before the domains and ranges that name them, and ranges before the stack
    // groups that name them, because that is the order the section tags -- and so the section
    // descriptors the drain walks -- already place them in.
    template <typename TWriter>
    void SerializeStateRecords(NvtxTraceObserver& observer, TWriter& writer)
    {
        bool recordWritten = false;
        observer.ConsumeAttachStateRecords(
            [&](NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount) {
                recordWritten = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::NvtxTrace>(writer, recordType, recordCorrelationId, pRecordBytes, recordByteCount);
            },
            [&](bool consistencyCheckPassed) {
                NVTRACE_LOG_ERR_IF(!consistencyCheckPassed, 50, "Non-fatal failure draining an NVTX Trace state section.");
                CommitOrRollback(writer, consistencyCheckPassed && recordWritten);
                recordWritten = false;
            });
    }

    // The referenced record, the first delivered, decides: an unreferenceable record type is skipped,
    // and so is one fnAcceptLeadingRecordType declines. A state table entry brings its chain records
    // along behind it and they ride its decision, while an arena record is referenced on its own. Each
    // record is its own transaction, committed under the state section's verdict alone.
    template <typename TWriter, typename TFN_AcceptLeadingRecordType>
    bool SerializeReferencedStateEvent(NvtxTraceObserver& observer, TWriter& writer, const NvTrace_RecordReference& reference, TFN_AcceptLeadingRecordType&& fnAcceptLeadingRecordType)
    {
        bool leadingRecordDelivered = false;
        bool recordsAccepted = false;
        bool eventSucceeded = false;
        bool serializationSucceeded = true;
        observer.ResolveRecordReference(
            reference,
            [&](NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                const bool isLeadingRecord = !std::exchange(leadingRecordDelivered, true);
                if (isLeadingRecord)
                {
                    switch (recordType)
                    {
                        case NvTrace_NvtxTrace_RecordType_StringRegistered:
                        case NvTrace_NvtxTrace_RecordType_StringSegment:
                        case NvTrace_NvtxTrace_RecordType_DomainCreated:
                        case NvTrace_NvtxTrace_RecordType_PushPopRangeCreated:
                        case NvTrace_NvtxTrace_RecordType_StackGroupCreated:
                            break;
                        default:
                            NVTRACE_LOG_ERR(50, "Unknown NVTX state record type %u.", recordType);
                            return;
                    }

                    recordsAccepted = fnAcceptLeadingRecordType(recordType);
                }

                if (!recordsAccepted)
                {
                    return;
                }

                eventSucceeded = Codec::EventEncoders::WriteRecordFromMappedBuffer<Codec::ProviderTypeId::NvtxTrace>(writer, recordType, recordCorrelationId, pPayload, payloadSize);
                serializationSucceeded = serializationSucceeded && eventSucceeded;
            },
            [&](bool consistencyCheckPassed) {
                // A failed verdict is not logged: routine after a pop.
                if (std::exchange(eventSucceeded, false))
                {
                    CommitOrRollback(writer, consistencyCheckPassed);
                }
            });
        return serializationSucceeded;
    }

    template <typename TWriter>
    bool SerializeReferencedStateEvent(NvtxTraceObserver& observer, TWriter& writer, const NvTrace_RecordReference& reference)
    {
        return SerializeReferencedStateEvent(observer, writer, reference, [](uint32_t /* leadingRecordType */) { return true; });
    }

} // namespace nv::trace::NvtxTraceSerialization
