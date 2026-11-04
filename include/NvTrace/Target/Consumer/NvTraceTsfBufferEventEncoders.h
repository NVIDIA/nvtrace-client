/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>
#include <NvTrace/Target/Consumer/NvTraceCudaApiTraceEventEncoders.h>
#include <NvTrace/Target/Consumer/NvTraceNvtxTraceEventEncoders.h>
#include <NvTrace/Target/Consumer/NvTraceStreamFormat.h>
#include <NvTrace/Target/Consumer/NvTraceThreadEventEncoders.h>
#include <NvTrace/Target/NvTraceCommonSerializationTypes_generated.h>
#include <NvTrace/Target/NvTraceGpuPartitionInfo_generated.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceSystemInfo_generated.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <cstring>
#include <vector>

#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/Target/NvTraceCpuCalibrationPointTypes.h>
#include <NvTrace/Target/NvTraceGpuCalibrationPointTypes.h>
#include <NvTrace/Target/Observer/NvTraceCudaApiTraceObserver.h>
#include <NvTrace/Target/Observer/NvTraceGpuGrCtxTraceObserver.h>
#include <NvTrace/Target/Observer/NvTraceGpuPmCounterObserver.h>

namespace nv::trace::Codec::NvTraceServiceEncoders {

    template <typename TRecordWriter>
    inline bool WriteSystemCreatedRecord(TRecordWriter& recordWriter, const FlatBufferView<FB::SystemInfo>& event)
    {
        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_SystemCreated, NVTRACE_NO_CORRELATION_ID, event.size, [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            memcpy(pPayload, event.pData, event.size);
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for SystemCreated event.");
            return false;
        }
        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteGpuPartitionCreatedRecord(TRecordWriter& recordWriter, NVTRACE_LUID gpuPartitionLuid, NVTRACE_UUID gpuUuid, uint64_t creationTimestampTsc, const FlatBufferView<FB::GpuPartitionInfo>& event)
    {
        flatbuffers::FlatBufferBuilder fbb;
        auto gpuPartitionInfoOffset = fbb.CreateVector(event.pData, event.size);
        auto fbGpuPartitionCreated = FB::Codec::NvTraceServiceEvents::CreateGpuPartitionCreated(
            fbb,
            reinterpret_cast<const FB::LUID*>(&gpuPartitionLuid),
            reinterpret_cast<const FB::UUID*>(&gpuUuid),
            creationTimestampTsc,
            gpuPartitionInfoOffset);
        fbb.Finish(fbGpuPartitionCreated);

        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_GpuPartitionCreated, NVTRACE_NO_CORRELATION_ID, fbb.GetSize(), [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            memcpy(pPayload, fbb.GetBufferPointer(), fbb.GetSize());
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for GpuPartitionCreated event.");
            return false;
        }

        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteGpuPartitionDestroyedRecord(TRecordWriter& recordWriter, NVTRACE_LUID gpuPartitionLuid, uint64_t destructionTimestampTsc)
    {
        flatbuffers::FlatBufferBuilder fbb;
        auto fbGpuPartitionDestroyed = FB::Codec::NvTraceServiceEvents::CreateGpuPartitionDestroyed(fbb, reinterpret_cast<const FB::LUID*>(&gpuPartitionLuid), destructionTimestampTsc);
        fbb.Finish(fbGpuPartitionDestroyed);

        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_GpuPartitionDestroyed, NVTRACE_NO_CORRELATION_ID, fbb.GetSize(), [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            memcpy(pPayload, fbb.GetBufferPointer(), fbb.GetSize());
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for GpuPartitionDestroyed event.");
            return false;
        }

        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteProcessCreatedRecord(TRecordWriter& recordWriter, NVTRACE_LUID processLuid, NVTRACE_PID pid, NVTRACE_PID globalPid, NvTrace_LinuxNamespaceId pidNamespaceId, const char* pPath = "", uint64_t creationTimestampTsc = 1)
    {
        flatbuffers::FlatBufferBuilder fbb;
        auto& fbProcessLuid = reinterpret_cast<FB::LUID&>(processLuid);
        int32_t fbNamespacePid = pid;
        int32_t fbNamespaceParentPid = 0; // TODO
        int32_t fbGlobalPid = globalPid;
        auto fbPath = fbb.CreateString(pPath != nullptr ? pPath : "");
        const FB::LinuxNamespaceId fbPidNamespaceId(pidNamespaceId.inode, pidNamespaceId.device);
        auto fbProcessCreated = FB::Codec::NvTraceServiceEvents::CreateProcessCreated(fbb, &fbProcessLuid, fbNamespacePid, fbNamespaceParentPid, fbPath, fbGlobalPid, &fbPidNamespaceId, creationTimestampTsc);
        fbb.Finish(fbProcessCreated);

        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_ProcessCreated, NVTRACE_NO_CORRELATION_ID, fbb.GetSize(), [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            memcpy(pPayload, fbb.GetBufferPointer(), fbb.GetSize());
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for ProcessCreated event.");
            return false;
        }

        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteProcessDestroyedRecord(TRecordWriter& recordWriter, NVTRACE_LUID processLuid, uint64_t destructionTimestampTsc)
    {
        flatbuffers::FlatBufferBuilder fbb;
        auto fbProcessDestroyed = FB::Codec::NvTraceServiceEvents::CreateProcessDestroyed(fbb, reinterpret_cast<const FB::LUID*>(&processLuid), destructionTimestampTsc);
        fbb.Finish(fbProcessDestroyed);

        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_ProcessDestroyed, NVTRACE_NO_CORRELATION_ID, fbb.GetSize(), [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            memcpy(pPayload, fbb.GetBufferPointer(), fbb.GetSize());
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for ProcessDestroyed event.");
            return false;
        }

        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteProviderCreatedRecord(TRecordWriter& recordWriter, NVTRACE_LUID providerLuid, NVTRACE_LUID parentLuid, NVTRACE_UUID providerTypeUuid, const char* pProviderName, const uint8_t* pConfiguration = nullptr, size_t configurationSize = 0, const uint8_t* pEventVersionInfo = nullptr, size_t eventVersionInfoSize = 0, uint32_t eventVersionInfoMinorVersion = 0)
    {
        flatbuffers::FlatBufferBuilder fbb;
        auto& fbProviderLuid = reinterpret_cast<FB::LUID&>(providerLuid);
        auto& fbParentLuid = reinterpret_cast<FB::LUID&>(parentLuid);
        auto& fbProviderTypeUuid = reinterpret_cast<FB::UUID&>(providerTypeUuid);
        auto fbProviderName = fbb.CreateString(pProviderName);
        auto fbConfiguration = pConfiguration ? fbb.CreateVector<uint8_t>(pConfiguration, configurationSize) : 0;
        auto fbEventVersionInfo = pEventVersionInfo ? fbb.CreateVector<uint8_t>(pEventVersionInfo, eventVersionInfoSize) : 0;
        auto fbProviderCreated = FB::Codec::NvTraceServiceEvents::CreateProviderCreated(fbb, &fbProviderLuid, &fbParentLuid, &fbProviderTypeUuid, fbProviderName, fbConfiguration, fbEventVersionInfo, eventVersionInfoMinorVersion);
        fbb.Finish(fbProviderCreated);

        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderCreated, NVTRACE_NO_CORRELATION_ID, fbb.GetSize(), [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            memcpy(pPayload, fbb.GetBufferPointer(), fbb.GetSize());
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for ProviderCreated event.");
            return false;
        }

        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteProviderAttachedRecord(TRecordWriter& recordWriter, NVTRACE_LUID providerLuid, const std::vector<uint64_t>& onAttachRecordLocations)
    {
        flatbuffers::FlatBufferBuilder fbb;
        const auto& fbProviderLuid = reinterpret_cast<const FB::LUID&>(providerLuid);
        const auto fbOnAttachRecordLocations = fbb.CreateVector(onAttachRecordLocations);
        const auto fbProviderAttached = FB::Codec::NvTraceServiceEvents::CreateProviderAttached(fbb, &fbProviderLuid, fbOnAttachRecordLocations);
        fbb.Finish(fbProviderAttached);

        const bool succeeded = recordWriter.WriteRecord(FB::Codec::NvTraceServiceEvents::EventTypeId_ProviderAttached, NVTRACE_NO_CORRELATION_ID, fbb.GetSize(), [&](uint8_t* /* pInlinePayload */, uint8_t* pPayload) {
            memcpy(pPayload, fbb.GetBufferPointer(), fbb.GetSize());
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvTraceService: Failed WriteRecord() for ProviderAttached event.");
            return false;
        }

        return true;
    }

} // namespace nv::trace::Codec::NvTraceServiceEncoders


namespace nv::trace::Codec::EventEncoders {

    template <ProviderTypeId TProviderTypeId, typename TRecordWriter>
    inline bool WriteRecord(TRecordWriter& recordWriter, uint32_t recordType, NvTrace_CorrelationId correlationId, const uint8_t* pPayload, size_t payloadSize)
    {
        const bool succeeded = recordWriter.WriteRecord(recordType, correlationId, payloadSize, [&](uint8_t*, uint8_t* pPayloadDest) {
            if (payloadSize)
            {
                memcpy(pPayloadDest, pPayload, payloadSize);
            }
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "%s: Failed WriteRecord() for record.", Codec::ToCString(TProviderTypeId));
            return false;
        }

        return true;
    }

    // WriteRecord for a payload delivered where it lives in the mapped provider buffer -- a state table
    // entry or a data blob; the StateTableReader contract says why it is copied with Memcpy64.
    template <ProviderTypeId TProviderTypeId, typename TRecordWriter>
    inline bool WriteRecordFromMappedBuffer(TRecordWriter& recordWriter, uint32_t recordType, NvTrace_CorrelationId correlationId, const uint8_t* pPayload, size_t payloadSize)
    {
        NVTRACE_LOG_ERR_RETURN_IF((payloadSize % sizeof(uint64_t)) != 0, false, 20, "%s: A mapped-buffer payload must be a whole number of words (recordType=%u, payloadSize=%zu).", Codec::ToCString(TProviderTypeId), recordType, payloadSize);

        const bool succeeded = recordWriter.WriteRecord(recordType, correlationId, payloadSize, [&](uint8_t*, uint8_t* pPayloadDest) {
            Memcpy64(pPayloadDest, pPayload, payloadSize);
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "%s: Failed WriteRecord() for record.", Codec::ToCString(TProviderTypeId));
            return false;
        }

        return true;
    }

} // namespace nv::trace::Codec::EventEncoders


namespace nv::trace::Codec::CudaApiTraceEventEncoders {

    static_assert(sizeof(Codec::CudaApiTraceEvents::ThreadEventInlinePayload) <= TSF_RECORD_INLINE_PAYLOAD_SIZE, "Inline payload size exceeds the TSF maximum allowed size");

} // namespace nv::trace::Codec::CudaApiTraceEventEncoders


namespace nv::trace::Codec::ThreadEventEncoders {

    static_assert(sizeof(Codec::ThreadLifetimeEvents::ThreadEventInlinePayload) <= TSF_RECORD_INLINE_PAYLOAD_SIZE, "Inline payload size exceeds the TSF maximum allowed size");

} // namespace nv::trace::Codec::ThreadEventEncoders
