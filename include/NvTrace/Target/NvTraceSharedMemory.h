/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

#include <NvTrace/Target/NvTraceMemory.h>

#if defined(__linux__)
#include <NvTrace/Target/NvTraceLinuxUtilities.h>
#endif

#include <cstddef>
#include <memory>

namespace nv::trace {

    class SharedMemory
    {
    public:
        ~SharedMemory();
        SharedMemory();

        SharedMemory(const SharedMemory&) = delete;
        SharedMemory(SharedMemory&& other) noexcept;

        SharedMemory& operator=(const SharedMemory&) = delete;
        SharedMemory& operator=(SharedMemory&& other) noexcept;

        void Reset() noexcept;

        bool AllocateWritable(size_t size);

        bool ImportReadOnly(NvTrace_OsObjectIdentifier identifier, size_t size, bool map = true);

        void Unmap();

        NvTrace_OsObjectIdentifier GetIdentifier() const
        {
#if defined(__linux__)
            return m_memoryFd;
#endif
        }

        void* MappedAddress()
        {
            return m_pMappedAddress;
        }
        const void* MappedAddress() const
        {
            return m_pMappedAddress;
        }

        size_t Size() const
        {
            return m_size;
        }

    private:
#if defined(__linux__)
        UniqueLinuxFd m_memoryFd;
        UniqueLinuxMappedAddr m_pMappedAddress;
#endif
        size_t m_size = 0;
    };

    struct SharedMemoryDeleter
    {
        void operator()(SharedMemory* pSharedMemory) const
        {
            Delete(pSharedMemory);
        }
    };
    using UniqueSharedMemoryPtr = std::unique_ptr<SharedMemory, SharedMemoryDeleter>;

    inline bool SharedMemory_ImportReadOnly(NvTrace_OsObjectIdentifier identifier, size_t size, UniqueSharedMemoryPtr& out_pSharedMemory, const void*& out_pMappedAddress)
    {
        UniquePtr<SharedMemory> pSharedMemory(New<SharedMemory>());
        if (!pSharedMemory)
        {
            return false;
        }

        bool succeeded = pSharedMemory->ImportReadOnly(identifier, size);
        if (!succeeded)
        {
            return false;
        }

        out_pMappedAddress = pSharedMemory->MappedAddress();
        out_pSharedMemory.reset(pSharedMemory.Release());
        return true;
    }

} // namespace nv::trace

#if defined(__linux__)
#include <NvTrace/Target/NvTraceSharedMemoryLinux.h>
#endif
