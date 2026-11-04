/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>

namespace nv::trace {

    class TraceStreamRecordLocationEncoder
    {
    public:
        ~TraceStreamRecordLocationEncoder() = default;
        TraceStreamRecordLocationEncoder() = default;

        void Reset() noexcept
        {
            m_recordIndexMask = 0;
            m_chunkIndexShift = 0;
            m_chunkIndexMask = 0;
            m_dataBlockSequenceNumberShift = 0;
        }

        bool Initialize(uint32_t maxChunkCountPerDataBlock, uint32_t maxRecordCountPerChunk)
        {
            NVTRACE_LOG_ERR_RETURN_IF(maxChunkCountPerDataBlock == 0, false, 50, "maxChunkCountPerDataBlock is 0");
            NVTRACE_LOG_ERR_RETURN_IF(maxRecordCountPerChunk == 0, false, 50, "maxRecordCountPerChunk is 0");

            uint32_t chunkIndexBitCount = CeilLog2(maxChunkCountPerDataBlock);
            uint32_t recordIndexBitCount = CeilLog2(maxRecordCountPerChunk);
            m_dataBlockSequenceNumberShift = chunkIndexBitCount + recordIndexBitCount;
            m_chunkIndexMask = chunkIndexBitCount < 32 ? ((1 << chunkIndexBitCount) - 1) : 0xFFFFFFFF;
            m_chunkIndexShift = recordIndexBitCount;
            m_recordIndexMask = recordIndexBitCount < 32 ? ((1 << recordIndexBitCount) - 1) : 0xFFFFFFFF;

            return true;
        }

        uint64_t EncodeRecordLocation(uint64_t dataBlockSequenceNumber, uint32_t chunkIndex, uint32_t recordIndex) const
        {
            return (dataBlockSequenceNumber << m_dataBlockSequenceNumberShift) | (uint64_t(chunkIndex) << m_chunkIndexShift) | uint64_t(recordIndex);
        }

        struct DecodedRecordLocation
        {
            uint64_t dataBlockSequenceNumber = 0;
            uint32_t chunkIndex = 0;
            uint32_t recordIndex = 0;
        };
        DecodedRecordLocation DecodeRecordLocation(uint64_t recordLocation) const
        {
            return {
                recordLocation >> m_dataBlockSequenceNumberShift,
                static_cast<uint32_t>(recordLocation >> m_chunkIndexShift) & m_chunkIndexMask,
                static_cast<uint32_t>(recordLocation) & m_recordIndexMask};
        }

    private:
        uint32_t m_dataBlockSequenceNumberShift = 0;
        uint32_t m_chunkIndexMask = 0;
        uint32_t m_chunkIndexShift = 0;
        uint32_t m_recordIndexMask = 0;
    };

} // namespace nv::trace
