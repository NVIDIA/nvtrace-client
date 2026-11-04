/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


// Types of the V1 provider buffer format, whose buffers are described by NvTrace_ProviderBufferHeader_V1.
// The successor format in nvtrace_provider_buffer.h declares a parallel, differently-shaped set of
// adaptive buffer types under the same concept names; the _V1 suffix keeps the two families apart.
// NvTrace_ProviderStatus stays in nvtrace_common_types.h, since both families' headers carry one.


////////////////////////////////////////////////////////////
// Provider Buffer Header

/// \brief The buffer header for all V1 NvTrace provider buffers.
///
/// The ProviderBufferHeader contains identifying information.
/// Contains an offset to provider-specific information.
struct NvTrace_ProviderBufferHeader_V1
{
    /// Uniquely identifies this provider.
    NVTRACE_UUID providerTypeUuid;

    uint32_t minorVersion;

    /// Short name for this provider.
    char providerName[64];

    /// Indicates the operational state of this provider.
    volatile enum NvTrace_ProviderStatus providerStatus;

    /// An offset from 'this' struct to the Provider-specific Layout.
    uint64_t providerBufferLayoutOffset;
    uint32_t providerBufferLayoutSize;
    uint32_t providerBufferLayoutMinorVersion;
};
#define NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ProviderBufferHeader_V1, providerBufferLayoutMinorVersion)

// This format predates NvTrace_ProviderBufferVersion, so no V1 buffer carries the major version; it is
// declared here for the successor format's major version to be numbered against.
#define NvTrace_ProviderBuffer_MAJOR_VERSION_1 1


////////////////////////////////////////////////////////////
// Circular Buffer

/// \brief Circular Buffer control data. The successor format's counterpart is NvTrace_CircularBufferState2.
struct NvTrace_CircularBufferState
{
    /// The current past-the-end sequence number.
    /// A record with sequence number N is stored in the cell of index 'N % cellCount' when 'currentEndSequenceNumber - cellCount < N < currentEndSequenceNumber'.
    /// The cell of index 'currentEndSequenceNumber % cellCount' may be concurrently written to by the producer and is not valid to read from.
    alignas(NVTRACE_CPU_CACHELINE_SIZE) volatile uint64_t currentEndSequenceNumber;

    /// The CPU timestamp of the most recent update.
    /// Useful for retiming Observers.
    volatile uint64_t lastUpdateTsc;
};
#define NvTrace_CircularBufferState_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CircularBufferState, lastUpdateTsc)


////////////////////////////////////////////////////////////
// Chunk

union NvTrace_ChunkRef_V1
{
    uint64_t value;

    struct
    {
        uint32_t chunkIndex;
        uint32_t chunkRevision;
    };
};

NVTRACE_WEAK_CONST uint32_t NVTRACE_INVALID_CHUNK_INDEX_V1 = UINT32_MAX;
NVTRACE_WEAK_CONST uint64_t NVTRACE_INVALID_CHUNK_REF_VALUE_V1 = UINT64_MAX;

struct NvTrace_ChunkHeader_V1
{
    volatile uint32_t chunkRevision;
};


////////////////////////////////////////////////////////////
// Adaptive Buffer

struct NvTrace_AdaptiveBufferState_V1
{
    struct NvTrace_CircularBufferState circularBufferState;
};
#define NvTrace_AdaptiveBufferState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveBufferState_V1, circularBufferState)

struct NvTrace_AdaptiveBufferCell_V1
{
    union NvTrace_ChunkRef_V1 firstChunkRef;
    uint64_t readableAfterTsc;
    uint64_t totalRecordCountBefore;
};
#define NvTrace_AdaptiveBufferCell_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveBufferCell_V1, totalRecordCountBefore)

// No alignment specifier: nextChunkRef already forces 8-byte alignment, and `struct alignas(...)` is
// C++-only syntax that would not compile for a C consumer of this header.
struct NvTrace_AdaptiveBufferChunkHeader_V1
{
    struct NvTrace_ChunkHeader_V1 chunkHeader;
    /// While the Chunk is still held by a thread, recordCount=-trueRecordCount (negative value or zero). Upon Chunk finalization, it will flip to its positive value.
    volatile int32_t recordCount;
    volatile union NvTrace_ChunkRef_V1 nextChunkRef;
};
#define NvTrace_AdaptiveBufferChunkHeader_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveBufferChunkHeader_V1, nextChunkRef)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
