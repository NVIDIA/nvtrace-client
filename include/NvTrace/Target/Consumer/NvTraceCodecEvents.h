/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
// Event types not defined in flatbuffers are defined here.

#include <stdint.h>

#include <NvTrace/Target/Consumer/NvTraceCodecEvents_generated.h>
#include <NvTrace/Target/Consumer/NvTraceCodecTypes.h>
#include <NvTrace/Target/NvTraceCpuCalibrationPointTypes.h>
#include <NvTrace/Target/NvTraceGpuCalibrationPointTypes.h>

namespace nv::trace::Codec::CudaApiTraceEvents {

    struct ThreadEventInlinePayload
    {
        uint64_t uniqueThreadId;
    };

} // namespace nv::trace::Codec::CudaApiTraceEvents

namespace nv::trace::Codec::NvtxTraceEvents {

    struct ThreadEventInlinePayload
    {
        uint64_t uniqueThreadId;
    };

} // namespace nv::trace::Codec::NvtxTraceEvents

namespace nv::trace::Codec::ThreadLifetimeEvents {

    struct ThreadEventInlinePayload
    {
        uint64_t uniqueThreadId;
    };

} // namespace nv::trace::Codec::ThreadLifetimeEvents
