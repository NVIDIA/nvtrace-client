/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

namespace nv::trace {

    inline constexpr uint64_t TRACE_STREAM_FORMAT_MAGIC = 0x4653544E4F41564E; // 'NVTRACETSF'
    inline constexpr uint32_t TRACE_STREAM_FORMAT_MAJOR_VERSION = 1;

    // This struct is common to all versions of the data block header, no matter the major version.
    // As such, it can never be changed without breaking backwards-compatibility.
    struct TsfDataBlockHeaderVersion
    {
        uint64_t magic;
        uint32_t majorVersion;
        uint32_t sizeofDataBlockHeader;
    };

    struct TsfDataBlockHeader
    {
        /// This is a COFF-style scheme, where each offset is relative to the pointer of "this" struct

        TsfDataBlockHeaderVersion version;

        uint64_t sizeofDataBlock;

        uint64_t offsetofDataBlockState;
        uint32_t sizeofDataBlockState;
        uint32_t _reserved0;

        uint64_t offsetofGpuPartitionRefs;
        uint32_t sizeofGpuPartitionRef;
        uint32_t maxGpuPartitionRefCount;

        uint64_t offsetofProcessRefs;
        uint32_t sizeofProcessRef;
        uint32_t maxProcessRefCount;

        uint64_t offsetofProviderRefs;
        uint32_t sizeofProviderRef;
        uint32_t maxProviderRefCount;

        uint64_t offsetofThreadEventProviderMappingRefs;
        uint32_t sizeofThreadEventProviderMappingRef;
        uint32_t maxThreadEventProviderMappingRefCount;

        uint64_t offsetofChunkStates;
        uint64_t offsetofChunks;
        uint32_t sizeofChunkState;
        uint32_t sizeofChunk;
        uint32_t maxChunkCount;

        uint32_t sizeofRecord;
        uint32_t sizeofInlinePayload;
        uint32_t maxRecordCountPerChunk;
    };

    struct TsfDataBlockState
    {
        uint64_t dataBlockSequenceNumber;

        uint64_t recordTimestampLowerBoundUtcNsec;
        uint64_t recordTimestampUpperBoundUtcNsec;

        uint64_t systemInfoRecordLocation;

        uint32_t validGpuPartitionRefCount;
        uint32_t validProcessRefCount;
        uint32_t validProviderRefCount;
        uint32_t validThreadEventProviderMappingRefCount;
        uint32_t validChunkCount;
    };

    struct TsfGpuPartitionRef
    {
        NVTRACE_LUID gpuPartitionLuid;
        uint64_t gpuPartitionCreatedRecordLocation;
    };

    struct TsfProcessRef
    {
        NVTRACE_LUID processLuid;
        uint64_t processCreatedRecordLocation;
    };

    struct TsfProviderRef
    {
        NVTRACE_LUID providerLuid;
        uint64_t providerCreatedRecordLocation;
        uint64_t providerAttachedRecordLocation;
    };

    struct TsfThreadEventProviderMappingRef
    {
        NVTRACE_LUID threadEventProviderLuid;
        NVTRACE_LUID mappedProviderLuid;
        uint32_t providerMappingId;
    };

    struct TsfChunkState
    {
        uint32_t validRecordCount;
    };

    struct TsfRecord
    {
        // uint32_t providerRefIndex        : 16;
        // uint32_t recordType              : 20;
        // uint32_t majorPayloadOffsetDiv8  : 28; // Subtractive offset, relative to the end of the chunk, in units of 8 bytes
        uint64_t packed;

        // The address of this record within the provider buffer it was read from, in whatever terms
        // the owning section addresses its records: {sequenceId} for a circular buffer record,
        // {chunk revision, chunk index, record index} for a thread event or a state chunk chain
        // record, {revision, entryIndex} for a state table entry, {revision, recordIndex} for a
        // state arena record.
        //
        // It identifies this record and nothing else. What a record belongs to is a named field in
        // its payload, never the owner's address borrowed into this one -- so a chain record carries
        // its own address here and its owner's ref in its payload.
        NvTrace_CorrelationId correlationId;
    };

    inline constexpr size_t TSF_RECORD_INLINE_PAYLOAD_SIZE = 32;

    struct TsfRecordWithInlinePayload
    {
        TsfRecord record;
        uint8_t inlinePayload[TSF_RECORD_INLINE_PAYLOAD_SIZE];
    };
    static_assert(sizeof(TsfRecordWithInlinePayload) == sizeof(TsfRecord) + TSF_RECORD_INLINE_PAYLOAD_SIZE, "TsfRecordWithInlinePayload size is not equal to the sum of TsfRecord and TSF_RECORD_INLINE_PAYLOAD_SIZE");

    // clang-format off
#define NVTRACE_TSF_RECORD_PROVIDER_REF_INDEX         15: 0
#define NVTRACE_TSF_RECORD_RECORD_TYPE                35:16
#define NVTRACE_TSF_RECORD_MAJOR_PAYLOAD_OFFSET_DIV8  63:36
    // clang-format on

    inline constexpr uint64_t TSF_INVALID_RECORD_LOCATION = 0;
    inline constexpr uint64_t TSF_INVALID_DATA_BLOCK_SEQUENCE_NUMBER = 0;
    inline constexpr uint64_t TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_LOW = 1;
    inline constexpr uint64_t TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH = 0x0FFFF; // Should be enough for attach data blocks
    inline constexpr uint64_t TSF_DATA_BLOCK_SEQUENCE_NUMBER_LOW = TSF_RESERVED_DATA_BLOCK_SEQUENCE_NUMBER_HIGH + 1;
    inline constexpr uint64_t TSF_DATA_BLOCK_SEQUENCE_NUMBER_HIGH = UINT64_MAX;
    inline constexpr uint32_t TSF_INVALID_OBJECT_REF_INDEX = 0xFFFF;
    inline constexpr uint32_t TSF_INVALID_CHUNK_INDEX = 0xFFFFFFFF;

} // namespace nv::trace
