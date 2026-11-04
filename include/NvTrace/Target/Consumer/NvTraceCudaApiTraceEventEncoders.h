/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <cstring>

// The thread event is the one CUDA-specific encoder, because its inline payload names the producing thread.
namespace nv::trace::Codec::CudaApiTraceEventEncoders {

    template <typename TWriter>
    inline bool WriteApiTraceEventRecord(TWriter& writer, const ThreadEventRecord& record)
    {
        const bool succeeded = writer.WriteRecord(record.recordType, record.correlationId, record.payloadSize, [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            auto* pThreadEventInlinePayload = reinterpret_cast<Codec::CudaApiTraceEvents::ThreadEventInlinePayload*>(pInlinePayload);
            pThreadEventInlinePayload->uniqueThreadId = record.uniqueThreadId;
            if (record.payloadSize != 0)
            {
                memcpy(pPayload, record.pPayload, record.payloadSize);
            }
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "CudaApiTrace: Failed WriteRecord() for ApiTraceEvent record.");
            return false;
        }

        return true;
    }

} // namespace nv::trace::Codec::CudaApiTraceEventEncoders
