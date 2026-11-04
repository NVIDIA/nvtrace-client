/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceMathUtilities.h>

#include <cstddef>

namespace nv::trace {

    /// Helper class to build the offset-table for a buffer
    class SectionLayoutBuilder
    {
    public:
        ~SectionLayoutBuilder() = default;
        SectionLayoutBuilder() = default;
        SectionLayoutBuilder(const SectionLayoutBuilder&) = default;
        SectionLayoutBuilder& operator=(const SectionLayoutBuilder&) = default;

        uint64_t AppendSection(uint64_t size, uint32_t alignment)
        {
            const uint64_t sectionOffset = AlignUpP2(m_endOffset, alignment);
            m_endOffset = sectionOffset + size;
            return sectionOffset;
        }

        template <typename TStruct>
        uint64_t AppendSection()
        {
            return AppendSection(sizeof(TStruct), alignof(TStruct));
        }

        template <typename TStruct>
        uint64_t AppendSection(uint32_t count)
        {
            return AppendSection(count * sizeof(TStruct), alignof(TStruct));
        }

        uint64_t GetEndOffset() const
        {
            return m_endOffset;
        }

    private:
        uint64_t m_endOffset = 0;
    };

} // namespace nv::trace
