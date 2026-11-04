/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Consumer/NvTraceRecordWriterUtilities.h>
#include <NvTrace/Target/Consumer/NvTraceThreadEventEncoders.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventObserver.h>

#include <cstdint>

namespace nv::trace::ThreadEventSerialization {

    // Each record is written into the writer while still tentative and committed only under the
    // reader's verdict, so a record a racing writer disturbed is rolled back rather than emitted.
    template <typename TWriter>
    void SerializeThreadStateEvents(ThreadEventObserver& observer, TWriter& writer)
    {
        bool eventSucceeded = false;
        observer.ConsumeAttachThreadStates(
            [&](uint32_t /* entryIndex */, uint64_t /* entryRevision */, const NvTrace_ThreadEvent_ThreadCreated& threadCreated) {
                eventSucceeded = Codec::ThreadEventEncoders::WriteThreadCreatedRecord(writer, threadCreated);
            },
            [&](bool consistencyCheckPassed) {
                NVTRACE_LOG_ERR_IF(!consistencyCheckPassed, 50, "Non-fatal failure in threadEvent.observer.ConsumeAttachThreadStates().");
                CommitOrRollback(writer, eventSucceeded && consistencyCheckPassed);
                eventSucceeded = false;
            });
    }

} // namespace nv::trace::ThreadEventSerialization
