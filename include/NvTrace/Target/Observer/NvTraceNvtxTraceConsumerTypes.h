/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>

#include <cstddef>
#include <cstdint>

namespace nv::trace {

    // A V1 object-lifetime record translated into the V2 vocabulary: a created announce becomes the
    // reference its resolve goes through, a destroyed announce becomes the self-contained record.
    struct NvtxTraceTranslatedRecordV1
    {
        uint32_t recordType = 0;
        size_t payloadSize = 0;
        union
        {
            NvTrace_RecordReference reference;
            NvTrace_NvtxTrace_StackGroupDestroyed stackGroupDestroyed;
        } payload = {};
    };

} // namespace nv::trace
