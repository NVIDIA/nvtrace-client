/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_gpu_clock_calibration.h>

#include <cmath>

namespace nv::trace {

    // This struct only adds convenience methods, without introducing additional state.
    // The C and C++ types can be cast between each other, to avoid data copies.
    struct TscGpuCalibrationPoint : public NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint
    {
        uint64_t ToTsc(uint64_t gpuIn) const
        {
            return tscTick + int64_t(std::round(gpuToTscFactor * int64_t(gpuIn - gpuNsec)));
        }
        uint64_t ToGpu(uint64_t tscIn) const
        {
            return gpuNsec + int64_t(std::round(tscToGpuFactor * int64_t(tscIn - tscTick)));
        }
    };

} // namespace nv::trace
