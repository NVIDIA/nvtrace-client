/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

#include <cstdint>
#include <flatbuffers/flatbuffers.h>
#include <string.h>

namespace nv::trace {

    /// A sectionOffset of 0 indicates an absent section, deprecated by the provider.
    template <class TSection>
    void AssignSectionPointer(
        const uint8_t* pMappedProviderBufferMemory,
        size_t sectionOffset,
        const TSection*& pSection)
    {
        if (sectionOffset == 0)
        {
            pSection = nullptr;
        }
        else
        {
            pSection = reinterpret_cast<const TSection*>(pMappedProviderBufferMemory + sectionOffset);
        }
    }

    template <class TFlatbuffer>
    void AssignFlatbufferSectionPointer(
        const uint8_t* pMappedProviderBufferMemory,
        size_t sectionOffset,
        const TFlatbuffer*& pSection)
    {
        if (sectionOffset == 0)
        {
            pSection = nullptr;
        }
        else
        {
            const uint8_t* pSectionBase = pMappedProviderBufferMemory + sectionOffset;
            pSection = flatbuffers::GetRoot<TFlatbuffer>(pSectionBase);
        }
    }

    // A V1 seqlock state mid-read: the revision sampled before the read, re-checked after.
    struct RevisionCursor
    {
        const volatile uint64_t* pRevision = nullptr;
        uint64_t expectedRevision = 0;

        bool CheckConsistency() const
        {
            ReadFence();
            return LoadRelaxed(pRevision) == expectedRevision;
        }
    };

    struct CStringComparator
    {
        bool operator()(const char* pLhs, const char* pRhs) const
        {
            return strcmp(pLhs, pRhs) < 0;
        }
    };

} // namespace nv::trace
