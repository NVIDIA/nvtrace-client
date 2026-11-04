/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

#include <cstdint>

#if defined(__linux__)
#include <NvTrace/Target/NvTraceLinuxUtilities.h>

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#endif

namespace nv::trace {

// Only Linux gives a GPU an OS object whose permissions decide who may reach it, so only there can a client prove
// access and only there does the Router enforce it. Elsewhere the security policy alone confines a session.
#if defined(__linux__)
    // Max physical GPUs supported by RM = NV_MAX_DEVICES = NV0080_MAX_DEVICES, the same limit the service bounds
    // its GPU partitions by, and so the capacity of one client's accessible GPU set.
    constexpr uint32_t MAX_ACCESSIBLE_GPU_COUNT = 32;

    // Opens the object of every GPU this process can reach, for a client to declare to the Router as its proof. Writes
    // and returns at most min(maxGpuIdentifierCount, MAX_ACCESSIBLE_GPU_COUNT) identifiers -- a system holds no more
    // GPUs than that, so asking for more than the buffer holds is safe only up to that bound. A declaration hands the
    // Router its own copies, so the caller needs these only until the send returns.
    inline uint32_t OpenAccessibleGpuIdentifiers(uint32_t maxGpuIdentifierCount, UniqueLinuxFd* out_pGpuIdentifiers)
    {
        maxGpuIdentifierCount = std::min(maxGpuIdentifierCount, MAX_ACCESSIBLE_GPU_COUNT);

        uint32_t gpuIdentifierCount = 0;
        for (uint32_t deviceMinor = 0; deviceMinor < MAX_ACCESSIBLE_GPU_COUNT && gpuIdentifierCount < maxGpuIdentifierCount; ++deviceMinor)
        {
            char deviceNodePath[32];
            std::snprintf(deviceNodePath, sizeof(deviceNodePath), "/dev/nvidia%u", deviceMinor);

            out_pGpuIdentifiers[gpuIdentifierCount].reset(::open(deviceNodePath, O_RDWR | O_CLOEXEC));
            if (out_pGpuIdentifiers[gpuIdentifierCount])
            {
                ++gpuIdentifierCount;
            }
        }

        return gpuIdentifierCount;
    }
#else
    // No OS proof of access exists, so the Router holds no accessible GPU set.
    constexpr uint32_t MAX_ACCESSIBLE_GPU_COUNT = 0;
#endif

} // namespace nv::trace
