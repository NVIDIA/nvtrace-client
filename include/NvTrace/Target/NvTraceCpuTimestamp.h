/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

// clang-format off
// Make the __rdtsc intrinsic available on x64.
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    #if defined(__linux__)
        #include <x86intrin.h>
    #else
        #error Unsupported platform
    #endif
#endif
// clang-format on

namespace nv::trace {

    const uint64_t NanosecondsPerSecond = 1000'000'000;

    /// Returns a CPU timestamp using a native instruction.
    inline uint64_t ReadCpuTscTimestamp()
    {
#if (defined(__x86_64) || defined(AMD64)) && defined(__linux__)
        // We intentionally choose rdtsc in favor of rdtscp
        // for lower overhead in exchange for greater uncertainty in timings.
        return __rdtsc();
#elif defined(__arm__) || defined(__aarch64__)
        uint64_t timerCounter;
        asm volatile("mrs %0, CNTVCT_EL0" : "=r"(timerCounter)::);
        return timerCounter;
#else
#error Unsupported platform
#endif
    }

} // namespace nv::trace
