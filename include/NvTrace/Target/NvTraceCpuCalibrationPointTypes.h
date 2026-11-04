/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cpu_clock_calibration.h>

#include <cmath>

namespace nv::trace {

    // This struct only adds convenience methods, without introducing additional state.
    // The C and C++ types can be cast between each other, to avoid data copies.
    struct TscRtcCalibrationPoint : public NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint
    {
        uint64_t ToTsc(uint64_t utcIn) const
        {
            return tscTick + int64_t(std::round(utcToTscFactor * int64_t(utcIn - utcNsec)));
        }
        uint64_t ToUtc(uint64_t tscIn) const
        {
            return utcNsec + int64_t(std::round(tscToUtcFactor * int64_t(tscIn - tscTick)));
        }
    };

} // namespace nv::trace
