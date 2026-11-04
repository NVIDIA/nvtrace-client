/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceSharedMemory.h>


#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace nv::trace {


    inline SharedMemory::~SharedMemory() = default;
    inline SharedMemory::SharedMemory() = default;

    inline SharedMemory::SharedMemory(SharedMemory&& other) noexcept
        : m_memoryFd(std::move(other.m_memoryFd))
        , m_pMappedAddress(std::move(other.m_pMappedAddress))
        , m_size(std::exchange(other.m_size, 0))
    {
    }

    inline SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept
    {
        if (this != &other)
        {
            m_memoryFd = std::move(other.m_memoryFd);
            m_pMappedAddress = std::move(other.m_pMappedAddress);
            m_size = std::exchange(other.m_size, 0);
        }

        return *this;
    }

    inline void SharedMemory::Reset() noexcept
    {
        m_size = 0;
        m_pMappedAddress.reset();
        m_memoryFd.reset();
    }

    inline bool SharedMemory::AllocateWritable(size_t size)
    {
        // Create a memory-backed file
        UniqueLinuxFd memoryFd(::memfd_create("", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        if (!memoryFd)
        {
            return false;
        }

        // Resize it
        int result = ::ftruncate(memoryFd, static_cast<off_t>(size));
        if (result == -1)
        {
            return false;
        }

        // Create a writable view of the file
        void* pAddr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, memoryFd, 0);
        if (pAddr == MAP_FAILED)
        {
            return false;
        }
        else if (!pAddr)
        {
            // We don't use MAP_FIXED, so mmap should not return 0 on success. Defensively reject and unmap such a mapping.
            ::munmap(pAddr, size);
            return false;
        }
        UniqueLinuxMappedAddr pMappedAddress(pAddr, LinuxMappedAddrDeleter{size});

        // Seal the file to prevent other processes from writing to it
        result = ::fcntl(memoryFd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | 0x10 /* F_SEAL_FUTURE_WRITE */);
        if (result == -1)
        {
            return false;
        }

        m_memoryFd = std::move(memoryFd);
        m_pMappedAddress = std::move(pMappedAddress);
        m_size = size;
        return true;
    }

    inline bool SharedMemory::ImportReadOnly(NvTrace_OsObjectIdentifier identifier, size_t size, bool map)
    {
        // Verify that the file cannot be shrunk concurrently by another process, as that can cause us to crash
        int result = ::fcntl(identifier, F_GET_SEALS);
        if (result == -1 || (result & F_SEAL_SHRINK) == 0)
        {
            return false;
        }

        // Verify that size is <= the actual file size
        struct stat fileStats;
        result = ::fstat(identifier, &fileStats);
        if (result == -1 || size > static_cast<size_t>(fileStats.st_size))
        {
            return false;
        }

        // Duplicate the handle
        UniqueLinuxFd memoryFd(::fcntl(identifier, F_DUPFD_CLOEXEC, 0));
        if (!memoryFd)
        {
            return false;
        }

        // Map the file
        UniqueLinuxMappedAddr pMappedAddress;
        if (map)
        {
            void* pAddr = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, memoryFd, 0);
            if (pAddr == MAP_FAILED)
            {
                return false;
            }
            else if (!pAddr)
            {
                // We don't use MAP_FIXED, so mmap should not return 0 on success. Defensively reject and unmap such a mapping.
                ::munmap(pAddr, size);
                return false;
            }
            pMappedAddress = UniqueLinuxMappedAddr(pAddr, LinuxMappedAddrDeleter{size});
        }

        m_memoryFd = std::move(memoryFd);
        m_pMappedAddress = std::move(pMappedAddress);
        m_size = size;
        return true;
    }

    inline void SharedMemory::Unmap()
    {
        m_pMappedAddress.reset();
    }

} // namespace nv::trace
