/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

namespace nv::trace::Codec {

    /// Codec provider-type tag shared by IB and TSF records.
    enum class ProviderTypeId
    {
        Invalid = 0,
        /// Provider type for metadata events like provider creation
        NvTraceService,
        CpuClockCalibration,
        GpuClockCalibration,
        GpuPmCounters,
        GpuHwTrace,
        GpuGrCtxTrace,
        CudaApiTrace,
        ThreadEvent,
        NvtxTrace,
    };

    inline const char* ToCString(ProviderTypeId providerTypeId)
    {
        switch (providerTypeId)
        {
            case ProviderTypeId::Invalid:
                return "Invalid";
            case ProviderTypeId::NvTraceService:
                return "NvTraceService";
            case ProviderTypeId::CpuClockCalibration:
                return "CpuClockCalibration";
            case ProviderTypeId::GpuClockCalibration:
                return "GpuClockCalibration";
            case ProviderTypeId::GpuPmCounters:
                return "GpuPmCounters";
            case ProviderTypeId::GpuHwTrace:
                return "GpuHwTrace";
            case ProviderTypeId::GpuGrCtxTrace:
                return "GpuGrCtxTrace";
            case ProviderTypeId::CudaApiTrace:
                return "CudaApiTrace";
            case ProviderTypeId::ThreadEvent:
                return "ThreadEvent";
            case ProviderTypeId::NvtxTrace:
                return "NvtxTrace";
        }

        return "Unknown";
    }

    inline constexpr uint32_t INVALID_RECORD_TYPE = 0;

} // namespace nv::trace::Codec
