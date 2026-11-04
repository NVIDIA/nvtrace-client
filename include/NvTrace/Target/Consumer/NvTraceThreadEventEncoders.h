/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>
#include <NvTrace/Target/NvTraceClientLogger.h>

#include <NvTrace/DataFormats/nvtrace_thread_event.h>

#include <cstring>

namespace nv::trace::Codec::ThreadEventEncoders {

    static_assert(static_cast<uint32_t>(FB::Codec::ThreadLifetimeEvents::EventTypeId_ThreadCreated) == static_cast<uint32_t>(NvTrace_ThreadEvent_RecordType_ThreadCreated), "ThreadLifetimeEvents::EventTypeId_ThreadCreated must mirror NvTrace_ThreadEvent_RecordType_ThreadCreated");
    static_assert(static_cast<uint32_t>(FB::Codec::ThreadLifetimeEvents::EventTypeId_ThreadDestroyed) == static_cast<uint32_t>(NvTrace_ThreadEvent_RecordType_ThreadDestroyed), "ThreadLifetimeEvents::EventTypeId_ThreadDestroyed must mirror NvTrace_ThreadEvent_RecordType_ThreadDestroyed");

    template <typename TRecordWriter>
    inline bool WriteThreadCreatedRecord(TRecordWriter& recordWriter, const NvTrace_ThreadEvent_ThreadCreated& threadCreated)
    {
        const bool succeeded = recordWriter.WriteRecord(NvTrace_ThreadEvent_RecordType_ThreadCreated, NVTRACE_NO_CORRELATION_ID, sizeof(threadCreated), [&](uint8_t* /* pInlinePayload */, uint8_t* pPayload) {
            memcpy(pPayload, &threadCreated, sizeof(threadCreated));
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "ThreadEvent: Failed WriteRecord() for ThreadCreated event.");
            return false;
        }

        return true;
    }

    template <typename TRecordWriter>
    inline bool WriteThreadDestroyedRecord(TRecordWriter& recordWriter, const NvTrace_ThreadEvent_ThreadDestroyed& threadDestroyed, uint64_t uniqueThreadId)
    {
        const bool succeeded = recordWriter.WriteRecord(NvTrace_ThreadEvent_RecordType_ThreadDestroyed, NVTRACE_NO_CORRELATION_ID, sizeof(threadDestroyed), [&](uint8_t* pInlinePayload, uint8_t* pPayload) {
            auto* const pThreadEventInlinePayload = reinterpret_cast<Codec::ThreadLifetimeEvents::ThreadEventInlinePayload*>(pInlinePayload);
            pThreadEventInlinePayload->uniqueThreadId = uniqueThreadId;
            memcpy(pPayload, &threadDestroyed, sizeof(threadDestroyed));
        });
        if (!succeeded)
        {
            NVTRACE_LOG_ERR(20, "ThreadEvent: Failed WriteRecord() for ThreadDestroyed event.");
            return false;
        }

        return true;
    }

} // namespace nv::trace::Codec::ThreadEventEncoders
