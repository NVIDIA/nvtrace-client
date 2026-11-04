/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceUniqueHandle.h>

#include <cstddef>
#include <cstdint>

extern "C" int close(int __fd);
extern "C" int munmap(void* __addr, size_t __len) throw();

namespace nv::trace {

    inline constexpr int INFINITE_TIMEOUT = -1;

    struct LinuxFdDeleter
    {
        void operator()(int fd) const
        {
            ::close(fd);
        }
    };
    using UniqueLinuxFd = UniqueHandle<int, -1, LinuxFdDeleter>;

    struct LinuxMappedAddrDeleter
    {
        size_t len = 0;

        void operator()(void* addr) const
        {
            ::munmap(addr, len);
        }
    };
    using UniqueLinuxMappedAddr = UniqueHandle<void*, uintptr_t{0}, LinuxMappedAddrDeleter>;

} //namespace nv::trace
