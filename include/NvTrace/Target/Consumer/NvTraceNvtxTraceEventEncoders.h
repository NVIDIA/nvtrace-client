/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <cstring>

namespace nv::trace::Codec::NvtxTraceEventEncoders {

    template <typename TWriter>
    inline bool WriteThreadEventRecord(TWriter& writer, const ThreadEventRecord& record)
    {
        const bool succeeded = writer.WriteRecord(record.recordType, record.correlationId, record.payloadSize, [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            auto* pThreadEventInlinePayload = reinterpret_cast<Codec::NvtxTraceEvents::ThreadEventInlinePayload*>(pInlinePayload);
            pThreadEventInlinePayload->uniqueThreadId = record.uniqueThreadId;
            if (record.payloadSize != 0)
            {
                memcpy(pPayload, record.pPayload, record.payloadSize);
            }
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "NvtxTrace: Failed WriteRecord() for thread event.");
            return false;
        }

        return true;
    }

} // namespace nv::trace::Codec::NvtxTraceEventEncoders
