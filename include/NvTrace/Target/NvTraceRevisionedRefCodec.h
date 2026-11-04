/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>

#include <cstdint>
#include <utility>

namespace nv::trace {

    // Packs a pool item's (itemIndex, revision) into a 64-bit item ref, refined by an intra-item
    // record index in the low CeilLog2(maxRecordCountPerItem) bits into a record ref -- zero-width
    // for single-record items, so their record ref IS their item ref. The revision travels
    // truncated, in the high bits at both levels, so staleness checks compare in ref space and a
    // same-item ref compares greater until the truncated revision wraps.
    class RevisionedRefCodec
    {
    public:
        ~RevisionedRefCodec() = default;
        RevisionedRefCodec() = default;

        RevisionedRefCodec(const RevisionedRefCodec&) = default;
        RevisionedRefCodec(RevisionedRefCodec&& rhs) noexcept
            : m_itemIndexBitCount(std::exchange(rhs.m_itemIndexBitCount, 0))
            , m_itemIndexMask(std::exchange(rhs.m_itemIndexMask, 0))
            , m_recordIndexBitCount(std::exchange(rhs.m_recordIndexBitCount, 0))
            , m_recordIndexMask(std::exchange(rhs.m_recordIndexMask, 0))
            , m_itemAndRecordIndexBitCount(std::exchange(rhs.m_itemAndRecordIndexBitCount, 0))
            , m_itemRevisionMask(std::exchange(rhs.m_itemRevisionMask, 0))
            , m_recordRevisionMask(std::exchange(rhs.m_recordRevisionMask, 0))
        {
        }

        RevisionedRefCodec& operator=(const RevisionedRefCodec&) = default;
        RevisionedRefCodec& operator=(RevisionedRefCodec&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_itemIndexBitCount = std::exchange(rhs.m_itemIndexBitCount, 0);
                m_itemIndexMask = std::exchange(rhs.m_itemIndexMask, 0);
                m_recordIndexBitCount = std::exchange(rhs.m_recordIndexBitCount, 0);
                m_recordIndexMask = std::exchange(rhs.m_recordIndexMask, 0);
                m_itemAndRecordIndexBitCount = std::exchange(rhs.m_itemAndRecordIndexBitCount, 0);
                m_itemRevisionMask = std::exchange(rhs.m_itemRevisionMask, 0);
                m_recordRevisionMask = std::exchange(rhs.m_recordRevisionMask, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_recordRevisionMask = 0;
            m_itemRevisionMask = 0;
            m_itemAndRecordIndexBitCount = 0;
            m_recordIndexMask = 0;
            m_recordIndexBitCount = 0;
            m_itemIndexMask = 0;
            m_itemIndexBitCount = 0;
        }

        // Both counts tally memory-backed items, so the bit counts -- and their sum, within two
        // bits of the log of the memory-backed record total -- stay far below 64, and the
        // encode/decode shifts omit shift-by-width guards on that proof; BitMask still guards,
        // since it costs nothing off the encode path.
        void Initialize(uint64_t itemCount, uint64_t maxRecordCountPerItem = 1)
        {
            m_itemIndexBitCount = CeilLog2(itemCount);
            m_itemIndexMask = BitMask(m_itemIndexBitCount);
            m_recordIndexBitCount = CeilLog2(maxRecordCountPerItem);
            m_recordIndexMask = BitMask(m_recordIndexBitCount);
            m_itemAndRecordIndexBitCount = m_itemIndexBitCount + m_recordIndexBitCount;

            m_itemRevisionMask = BitMask(64 - m_itemIndexBitCount);
            m_recordRevisionMask = BitMask(64 - m_itemAndRecordIndexBitCount);
        }

        NvTrace_RevisionedRef EncodeItemRef(uint64_t itemIndex, uint64_t revision) const
        {
            return (revision << m_itemIndexBitCount) | itemIndex;
        }

        NvTrace_RevisionedRef EncodeRecordRef(NvTrace_RevisionedRef itemRef, uint64_t recordIndex) const
        {
            return (itemRef << m_recordIndexBitCount) | recordIndex;
        }

        NvTrace_RevisionedRef EncodeRecordRef(uint64_t recordIndex, uint64_t itemIndex, uint64_t revision) const
        {
            return (revision << m_itemAndRecordIndexBitCount) | (itemIndex << m_recordIndexBitCount) | recordIndex;
        }

        NvTrace_RevisionedRef ItemRefOfRecordRef(NvTrace_RevisionedRef recordRef) const
        {
            return recordRef >> m_recordIndexBitCount;
        }

        struct DecodedItemRef
        {
            uint64_t itemIndex;
            uint64_t revision;
        };
        DecodedItemRef DecodeItemRef(NvTrace_RevisionedRef itemRef) const
        {
            return {itemRef & m_itemIndexMask, itemRef >> m_itemIndexBitCount};
        }

        struct DecodedRecordRef
        {
            uint64_t recordIndex;
            uint64_t itemIndex;
            uint64_t revision;
        };
        DecodedRecordRef DecodeRecordRef(NvTrace_RevisionedRef recordRef) const
        {
            return {recordRef & m_recordIndexMask, (recordRef >> m_recordIndexBitCount) & m_itemIndexMask, recordRef >> m_itemAndRecordIndexBitCount};
        }

        // A ref carries its revision truncated to the bits left above its index fields, so a revision
        // read fresh from an item must be truncated the same way before it is compared against one a
        // ref carried.
        uint64_t TruncateItemRevision(uint64_t revision) const
        {
            return revision & m_itemRevisionMask;
        }
        uint64_t TruncateRecordRevision(uint64_t revision) const
        {
            return revision & m_recordRevisionMask;
        }

    private:
        static uint64_t BitMask(uint32_t bitCount)
        {
            return bitCount < 64 ? (UINT64_C(1) << bitCount) - 1 : UINT64_MAX;
        }

    private:
        uint32_t m_itemIndexBitCount = 0;
        uint64_t m_itemIndexMask = 0;

        uint32_t m_recordIndexBitCount = 0;
        uint64_t m_recordIndexMask = 0;

        uint32_t m_itemAndRecordIndexBitCount = 0;

        uint64_t m_itemRevisionMask = 0;
        uint64_t m_recordRevisionMask = 0;
    };

} // namespace nv::trace
