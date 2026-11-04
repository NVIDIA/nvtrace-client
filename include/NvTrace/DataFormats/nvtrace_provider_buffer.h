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


// This struct must never change
struct NvTrace_ProviderBufferVersion
{
    uint32_t majorVersion;

    uint32_t providerBufferHeaderSize;
    uint32_t providerBufferHeaderMinorVersion;
};
#define NvTrace_ProviderBuffer_MAJOR_VERSION_2 2


struct NvTrace_ProviderBufferHeader
{
    struct NvTrace_ProviderBufferVersion version;

    NVTRACE_UUID providerTypeUuid;

    char providerName[64];

    volatile enum NvTrace_ProviderStatus providerStatus;

    uint64_t sectionDescsOffset;
    uint32_t sectionDescCount;
    uint32_t sectionDescSize;
    uint32_t sectionDescMinorVersion;

    // NvTrace_RecordReference below is published once here instead of in each provider's own record
    // version info.
    uint32_t recordReferenceSize;
    uint32_t recordReferenceMinorVersion;

    // If the provider emits ThreadEvents, its source provider index, registered via NvTrace_ThreadEvent_SourceProviderRegistered.
    // Otherwise 0.
    uint32_t threadEventSourceProviderIndex;
};
#define NvTrace_ProviderBufferHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ProviderBufferHeader, recordReferenceMinorVersion)


enum NvTrace_ProviderBufferSectionType
{
    NvTrace_ProviderBufferSectionType_Invalid = 0,
    NvTrace_ProviderBufferSectionType_DataBlob = 1,
    NvTrace_ProviderBufferSectionType_CircularBuffer = 2,
    NvTrace_ProviderBufferSectionType_AdaptiveChunkPool = 3,
    NvTrace_ProviderBufferSectionType_AdaptiveBuffer = 4,
    NvTrace_ProviderBufferSectionType_StateTable = 5,
    NvTrace_ProviderBufferSectionType_GpuReportChunkPool = 6,
    NvTrace_ProviderBufferSectionType_GpuReportBuffer = 7,
    NvTrace_ProviderBufferSectionType_StateArena = 8,

    NvTrace_ProviderBufferSectionType_FORCE_INT = 0x7FFFFFFF
};

struct NvTrace_ProviderBufferSectionDesc
{
    enum NvTrace_ProviderBufferSectionType sectionType;
    uint32_t sectionTag;

    uint64_t sectionOffset;
    uint64_t sectionSize;

    uint32_t sectionHeaderSize;
    uint32_t sectionHeaderMinorVersion;
};
#define NvTrace_ProviderBufferSectionDesc_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ProviderBufferSectionDesc, sectionHeaderMinorVersion)


////////////////////////////////////////////////////////////
// Shared across section types

NVTRACE_WEAK_CONST uint32_t NVTRACE_INVALID_SECTION_INDEX = 0xFFFFFFFF;
NVTRACE_WEAK_CONST uint32_t NVTRACE_INVALID_OFFSET32 = 0xFFFFFFFF;
NVTRACE_WEAK_CONST uint64_t NVTRACE_INVALID_OFFSET64 = 0xFFFFFFFFFFFFFFFFull;

// A revisioned reference to a pool item: the item's index packed with the revision that was
// current when the ref was taken (revision in the high bits), optionally refined by an intra-item
// record index in the low bits. Staleness is detected by comparing against the item's current
// revision in ref space.
typedef uint64_t NvTrace_RevisionedRef;
NVTRACE_WEAK_CONST NvTrace_RevisionedRef NVTRACE_INVALID_REVISIONED_REF = 0;

// Every record stands on its own, so a record read out of a group -- a state table entry's chunk
// chain, an arena run reserved together -- never takes meaning from the group it arrived in, and
// never borrows another record's identity:
//
//   - A record introducing an object is named either by its own address, which is that object's (a
//     state table entry's ref), or by an id its payload already carries (a stringHandle, a
//     cufunctionId).
//   - A record continuing one is named by its own address, and names its owner in a payload field.
//
// So what a record belongs to is always something the record itself carries, never the address of
// whichever record happened to precede it.

// One record's type and where its payload ends, in whole words from the base its section defines:
// consecutive descs bracket each payload, so a section holding exactly one -- a data blob, or a state
// table entry -- reads the value as that payload's size, its predecessor's end being zero. That is why
// NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8 still names a size. A section that reserves a desc before its payload
// is written publishes recordType 0 until the record commits.
struct NvTrace_RecordDesc
{
    uint32_t recordType;
    uint32_t payloadEndOffsetDiv8;
};
#define NvTrace_RecordDesc_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_RecordDesc, payloadEndOffsetDiv8)

#define NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8(payloadSize) ((uint32_t)(((uint64_t)(payloadSize) + sizeof(uint64_t) - 1) / sizeof(uint64_t)))

// Record types every provider shares, at the top of the space so a provider's own never collide. One
// is encoded exactly as a provider's own, so it still carries its emitter through the usual decode.
NVTRACE_WEAK_CONST uint32_t NvTrace_ReservedRecordType_RecordReference = 0xFFFFF;

// Names one record in this provider buffer by the section that owns its storage and gates its
// lifetime. recordRef packs (revision, itemIndex, recordIndex) high-to-low, the two index widths
// derived from counts the owning section's header publishes and the record field zero-width for
// single-record items. The revision travels truncated, so a resolve compares in ref space: a
// stale reference -- the item modified or recycled since -- resolves to nothing.
struct NvTrace_RecordReference
{
    uint32_t sectionIndex;
    uint32_t rsvd0004;

    NvTrace_RevisionedRef recordRef;
};
#define NvTrace_RecordReference_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_RecordReference, recordRef)


////////////////////////////////////////////////////////////
// Data Blob Section

enum NvTrace_DataBlobDrainFlag
{
    NvTrace_DataBlobDrainFlag_OnAttach = 1u << 0,
    NvTrace_DataBlobDrainFlag_OnDetach = 1u << 1,

    NvTrace_DataBlobDrainFlag_FORCE_INT = 0x7FFFFFFF
};

enum NvTrace_DataBlobStatusFlag
{
    NvTrace_DataBlobStatusFlag_Ready = 1u << 0,
    NvTrace_DataBlobStatusFlag_Pending = 1u << 1,

    NvTrace_DataBlobStatusFlag_FORCE_INT = 0x7FFFFFFF
};

struct NvTrace_DataBlobSectionHeader
{
    uint64_t recordDescOffsetInSection;
    uint32_t recordDescSize;
    uint32_t recordDescMinorVersion;

    uint32_t drainFlags;
    uint32_t statusFlags;

    uint64_t payloadOffsetInSection;
};
#define NvTrace_DataBlobSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_DataBlobSectionHeader, payloadOffsetInSection)


////////////////////////////////////////////////////////////
// Circular Buffer Section

// Clock domain of a cell's gate timestamp (located at recordTimestampOffsetInCell); the consumer compares it against
// its read endpoint to decide which records are ready. _Invalid means the section carries no gate timestamp.
enum NvTrace_CircularBufferRecordTimestampType
{
    NvTrace_CircularBufferRecordTimestampType_Invalid = 0,
    NvTrace_CircularBufferRecordTimestampType_Tsc = 1,
    NvTrace_CircularBufferRecordTimestampType_GpuPtimer = 2,

    NvTrace_CircularBufferRecordTimestampType_FORCE_INT = 0x7FFFFFFF
};

struct NvTrace_CircularBufferSectionHeader
{
    uint64_t circularBufferStateOffsetInSection;
    uint32_t circularBufferStateSize;
    uint32_t circularBufferStateMinorVersion;

    uint64_t cellsOffsetInSection;
    uint32_t cellCount;
    uint32_t cellSize;

    // Exactly one offset is valid; the size and minor version describe the struct either way
    uint64_t fixedRecordDescOffsetInSection;
    uint32_t variableRecordDescOffsetInCell;
    uint32_t recordDescSize;
    uint32_t recordDescMinorVersion;

    uint32_t payloadOffsetInCell;
    uint32_t maxPayloadSizeDiv8;

    enum NvTrace_CircularBufferRecordTimestampType recordTimestampType;
    uint32_t recordTimestampOffsetInCell;
};
#define NvTrace_CircularBufferSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CircularBufferSectionHeader, recordTimestampOffsetInCell)

struct NvTrace_CircularBufferState2
{
    alignas(NVTRACE_CPU_CACHELINE_SIZE) volatile uint64_t nextWriteSequenceNumber;
};
#define NvTrace_CircularBufferState2_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_CircularBufferState2, nextWriteSequenceNumber)


////////////////////////////////////////////////////////////
// Adaptive Chunk Pool Section

// Chunks backing adaptive buffer sections and state table entry chains -- each an
// NvTrace_AdaptiveChunkHeader followed by NvTrace_RecordDescs, written against the chunk-header and
// record versions the pool publishes once for every referencing section.
struct NvTrace_AdaptiveChunkPoolSectionHeader
{
    uint64_t chunksOffsetInSection;
    uint32_t chunkCount;
    uint32_t chunkSize;

    uint32_t adaptiveChunkHeaderSize;
    uint32_t adaptiveChunkHeaderMinorVersion;

    uint32_t recordDescSize;
    uint32_t recordDescMinorVersion;

    uint32_t maxRecordCountPerChunk;
};
#define NvTrace_AdaptiveChunkPoolSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveChunkPoolSectionHeader, maxRecordCountPerChunk)

struct NvTrace_AdaptiveChunkHeader
{
    volatile uint64_t revision;
    volatile NvTrace_RevisionedRef nextChunkRef;
    uint32_t recordCount;
};
#define NvTrace_AdaptiveChunkHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveChunkHeader, recordCount)


////////////////////////////////////////////////////////////
// Adaptive Buffer Section

struct NvTrace_AdaptiveBufferSectionHeader
{
    uint64_t adaptiveBufferStateOffsetInSection;
    uint32_t adaptiveBufferStateSize;
    uint32_t adaptiveBufferStateMinorVersion;

    uint64_t adaptiveBufferCellsOffsetInSection;
    uint32_t adaptiveBufferCellCount;
    uint32_t adaptiveBufferCellSize;
    uint32_t adaptiveBufferCellMinorVersion;

    uint32_t adaptiveChunkPoolSectionIndex;

    uint8_t mayEmitRecordReferences;
};
#define NvTrace_AdaptiveBufferSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveBufferSectionHeader, mayEmitRecordReferences)

struct NvTrace_AdaptiveBufferState
{
    alignas(NVTRACE_CPU_CACHELINE_SIZE) volatile uint64_t nextCellWriteSequenceNumber;
};
#define NvTrace_AdaptiveBufferState_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveBufferState, nextCellWriteSequenceNumber)

struct NvTrace_AdaptiveBufferCell
{
    NvTrace_RevisionedRef firstChunkRef;
    uint64_t readableAfterTsc;
    uint64_t totalRecordCountBefore;
};
#define NvTrace_AdaptiveBufferCell_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_AdaptiveBufferCell, totalRecordCountBefore)


////////////////////////////////////////////////////////////
// State Table Section
//
// Drained in full at attach, exactly once; every later change arrives as a record instead.

struct NvTrace_StateTableSectionHeader
{
    uint64_t entriesOffsetInSection;
    uint32_t entryCount;
    uint32_t entrySize;

    uint32_t stateTableEntryHeaderSize;
    uint32_t stateTableEntryHeaderMinorVersion;

    // Exactly one offset is valid; the size and minor version describe the struct either way.
    uint64_t fixedRecordDescOffsetInSection;
    uint32_t variableRecordDescOffsetInEntry;
    uint32_t recordDescSize;
    uint32_t recordDescMinorVersion;

    uint32_t payloadOffsetInEntry;
    uint32_t maxPayloadSizeDiv8;

    // Heads the entry's chain of additional records; NVTRACE_INVALID_OFFSET32 when entries carry no chains.
    uint32_t nextChunkRefOffsetInEntry;

    uint32_t adaptiveChunkPoolSectionIndex;

    uint8_t mayEmitRecordReferences;
};
#define NvTrace_StateTableSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_StateTableSectionHeader, mayEmitRecordReferences)

struct NvTrace_StateTableEntryHeader
{
    volatile uint64_t revision;
};
#define NvTrace_StateTableEntryHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_StateTableEntryHeader, revision)

NVTRACE_WEAK_CONST uint64_t NVTRACE_STATE_TABLE_ENTRY_REVISION_MODIFYING_BIT = UINT64_C(0x8000000000000000);


////////////////////////////////////////////////////////////
// State Arena Section
//
// A monotonic bump arena of variable-size records that are never individually freed, only reset as a
// whole; drained in full at attach, exactly once, with every later addition arriving as a record
// reference instead. Shaped like a single adaptive chunk: NvTrace_RecordDescs grow forward from the
// record arena's low edge and payloads grow backward from its high edge, so a payload is bounded only
// by the space the two cursors have not closed on and nothing is ever chained.
//
// A record is published by its recordType, which a producer writes last: recordCount is released when
// a record's payload extent is reserved, so every record below the watermark can be strided over
// safely, and a recordType still 0 marks one whose payload is not yet written -- a drain skips it and
// keeps walking. Neither allocation cursor is published; the payload cursor is the last record's
// payloadEndOffsetDiv8.

struct NvTrace_StateArenaSectionHeader
{
    uint64_t stateArenaStateOffsetInSection;
    uint32_t stateArenaStateSize;
    uint32_t stateArenaStateMinorVersion;

    uint64_t recordArenaOffsetInSection;
    uint64_t recordArenaSize;

    uint32_t recordDescSize;
    uint32_t recordDescMinorVersion;

    uint32_t maxPayloadSizeDiv8;
    uint32_t maxRecordCount;

    uint8_t mayEmitRecordReferences;
};
#define NvTrace_StateArenaSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_StateArenaSectionHeader, mayEmitRecordReferences)

// revision gates the whole arena the way an entry's revision gates one state table entry: a reset
// moves it, so every ref taken before the reset resolves to nothing. Zero means never initialized.
// The two fields share a cache line deliberately: recordCount is the hot one, released by every
// reservation, revision is written only by a reset, and a reader loads both together.
struct NvTrace_StateArenaState
{
    alignas(NVTRACE_CPU_CACHELINE_SIZE) volatile uint64_t revision;
    volatile uint32_t recordCount;
};
#define NvTrace_StateArenaState_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_StateArenaState, recordCount)

NVTRACE_WEAK_CONST uint64_t NVTRACE_STATE_ARENA_REVISION_RESETTING_BIT = UINT64_C(0x8000000000000000);


////////////////////////////////////////////////////////////
// GPU Report Chunk Pool Section
//
// Chunks backing GPU report buffer sections -- a distinct section type from AdaptiveChunkPool, whose
// chunks carry in-chunk headers: a GPU report chunk is bare records, so its mutable state
// lives beside the chunk memory, in the pool's NvTrace_GpuReportChunkState array. The states belong
// to the pool rather than a referencing section because several GPU report buffer sections may
// share one pool.

struct NvTrace_GpuReportChunkPoolSectionHeader
{
    uint64_t chunksOffsetInSection;
    uint32_t chunkCount;
    uint32_t chunkSize;

    // Per-chunk NvTrace_GpuReportChunkState array (chunkCount entries of gpuReportChunkStateSize each)
    uint64_t gpuReportChunkStatesOffsetInSection;
    uint32_t gpuReportChunkStateSize;
    uint32_t gpuReportChunkStateMinorVersion;

    // One desc for the whole pool: reports are homogeneous, and a chunk is a dense array of records of the desc's payload size
    uint64_t recordDescOffsetInSection;
    uint32_t recordDescSize;
    uint32_t recordDescMinorVersion;

    uint32_t recordsPerChunk;
};
#define NvTrace_GpuReportChunkPoolSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GpuReportChunkPoolSectionHeader, recordsPerChunk)

struct NvTrace_GpuReportChunkState
{
    volatile uint64_t revision;
};
#define NvTrace_GpuReportChunkState_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GpuReportChunkState, revision)


////////////////////////////////////////////////////////////
// GPU Report Buffer Section
//
// Fixed-size records written by a GPU work stream into chunks drawn from a GPU report chunk pool; a
// seq-protected circular buffer of cells names the chunks. A record's first 8 bytes are its report
// ref -- identity and readiness in one: record k of a cell is expected at ref
// (chunkRef << CeilLog2(recordsPerChunk)) + k, and a stored ref equal to that means written, lower
// means not written yet (resume on a later poll), higher means the chunk was overwritten by newer
// work (abandon the rest of the cell) -- higher by construction, since the recycling revision is
// the ref's high bits. A cell names a whole chunk of the pool's recordsPerChunk records; a
// producer that stops mid-chunk pads the remaining records with the refs the consumer expects
// there. The cell is published only once fully written and is never modified after.
//
// The GPU writes a record as ordered stores, not one atomic transaction, so a recycled chunk could
// transiently pair a slot's old ref with new bytes. The pool's per-chunk revision closes that: the
// producing CPU bumps it before the chunk's new work is enqueued, the cell carries the revision it
// was built against, and a consumer commits a group of copied records only if the chunk's revision
// still matches the cell's. The revision gates recycling only: the producer must hand a chunk to
// new work only once the previous work's writes are known complete, since a stale write landing
// after the handoff would corrupt a record under an unmoved revision.

struct NvTrace_GpuReportBufferSectionHeader
{
    uint64_t gpuReportBufferStateOffsetInSection;
    uint32_t gpuReportBufferStateSize;
    uint32_t gpuReportBufferStateMinorVersion;

    uint64_t gpuReportBufferCellsOffsetInSection;
    uint32_t gpuReportBufferCellCount;
    uint32_t gpuReportBufferCellSize;
    uint32_t gpuReportBufferCellMinorVersion;

    uint32_t gpuReportChunkPoolSectionIndex;
};
#define NvTrace_GpuReportBufferSectionHeader_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GpuReportBufferSectionHeader, gpuReportChunkPoolSectionIndex)

struct NvTrace_GpuReportBufferState
{
    alignas(NVTRACE_CPU_CACHELINE_SIZE) volatile uint64_t nextCellWriteSequenceNumber;
};
#define NvTrace_GpuReportBufferState_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GpuReportBufferState, nextCellWriteSequenceNumber)

struct NvTrace_GpuReportBufferCell
{
    NvTrace_RevisionedRef chunkRef;
};
#define NvTrace_GpuReportBufferCell_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GpuReportBufferCell, chunkRef)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
