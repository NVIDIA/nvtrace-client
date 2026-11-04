/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

namespace nv::trace {

    struct ThreadEventRecord
    {
        uint32_t recordType = 0;
        uint64_t uniqueThreadId = 0;
        uint64_t correlationId = 0;
        const uint8_t* pPayload = nullptr;
        size_t payloadSize = 0;
    };

    // Delivers one record of a bound source provider: pProviderState is that source provider's consumer
    // state, never null. Return true to consume the record, false to retain it and retry on the next pass.
    using ThreadEventConsumeCallback = bool (*)(void* pConsumer, void* pProcessState, void* pProviderState, const ThreadEventRecord& record);

} // namespace nv::trace
