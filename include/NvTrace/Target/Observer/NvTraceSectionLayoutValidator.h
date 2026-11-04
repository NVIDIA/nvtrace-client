/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace nv::trace {

    /// Helper class to validate that a set of sections form a valid non-overlapping layout within a buffer.
    /// Sections are kept sorted by offset so that each addition only checks the two immediate neighbors.
    class SectionLayoutValidator
    {
    public:
        ~SectionLayoutValidator() = default;

        explicit SectionLayoutValidator(uint64_t totalBufferSize)
            : m_totalBufferSize(totalBufferSize)
        {
        }

        SectionLayoutValidator(const SectionLayoutValidator&) = default;
        SectionLayoutValidator& operator=(const SectionLayoutValidator&) = default;

        // Attempts to add a section.
        // Returns true if the section is valid (contained within the buffer and non-overlapping), otherwise returns false
        bool AddAndValidateSection(uint64_t offset, uint64_t size)
        {
            if (size == 0)
            {
                return true;
            }
            if (m_totalBufferSize < size || m_totalBufferSize - size < offset)
            {
                return false;
            }

            const Section newSection = {offset, offset + size};

            auto itr = std::upper_bound(m_sections.begin(), m_sections.end(), newSection.beginOffset, [](uint64_t lhs, const Section& rhs) {
                return lhs < rhs.endOffset;
            });
            if (itr != m_sections.end() && itr->beginOffset < newSection.endOffset)
            {
                return false;
            }

            m_sections.insert(itr, newSection);
            return true;
        }

        bool AddAndValidateSection(uint64_t offset, uint32_t elementCount, uint32_t elementSize)
        {
            // No overflow guard needed: the product of two 32-bit values is at most (2^32-1)^2 < 2^64, so the
            // uint64_t multiply is always exact. (The uint64_t/uint64_t overload below does need an explicit check.)
            return AddAndValidateSection(offset, elementCount * uint64_t(elementSize));
        }

        bool AddAndValidateSection(uint64_t offset, uint64_t elementCount, uint64_t elementSize)
        {
            // Overflow check
            if (elementSize != 0 && UINT64_MAX / elementSize < elementCount)
            {
                return false;
            }

            return AddAndValidateSection(offset, elementCount * elementSize);
        }

        bool AddAndValidateSection(uint64_t offset, uint32_t structSize, uint32_t structMinorVersion, uint32_t minimumRequiredMinorVersion)
        {
            return minimumRequiredMinorVersion <= structMinorVersion && structMinorVersion <= structSize && AddAndValidateSection(offset, structSize);
        }

        bool AddAndValidateSection(uint64_t offset, uint32_t structCount, uint32_t structSize, uint32_t structMinorVersion, uint32_t minimumRequiredMinorVersion)
        {
            return minimumRequiredMinorVersion <= structMinorVersion && structMinorVersion <= structSize && AddAndValidateSection(offset, structCount, structSize);
        }

    private:
        uint64_t m_totalBufferSize;

        struct Section
        {
            uint64_t beginOffset;
            uint64_t endOffset;
        };
        std::vector<Section> m_sections;
    };

} // namespace nv::trace
