/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_NvtxTrace_ProviderTypeUuid = NVTRACE_UUID_VALUE(0xa4127608, 0x2854, 0x488c, 0x80, 0xe1, 0x18, 0xc5, 0x19, 0x61, 0x50, 0xcd);

enum NvTrace_NvtxTrace_SectionTag
{
    NvTrace_NvtxTrace_SectionTag_INVALID = 0,

    // Data blobs
    NvTrace_NvtxTrace_SectionTag_RecordVersionInfo = 1,
    NvTrace_NvtxTrace_SectionTag_Configuration = 2,
    NvTrace_NvtxTrace_SectionTag_ProviderStatistics = 3,

    // State arena
    NvTrace_NvtxTrace_SectionTag_Strings = 4, // a string's characters past the inline ones are the StringSegment records reserved with it

    // State tables
    NvTrace_NvtxTrace_SectionTag_Domains = 5,
    NvTrace_NvtxTrace_SectionTag_PushPopRanges = 6, // chain-extended: message segments past the inline ones
    NvTrace_NvtxTrace_SectionTag_StackGroups = 7,   // chain-extended: stack refs past the inline ones

    // Chunk pools
    NvTrace_NvtxTrace_SectionTag_StateChunkPool = 8, // backs the three chained tables

    NvTrace_NvtxTrace_SectionTag_FORCE_INT = 0x7fffffff
};

/// The provider's one record-type space: every record in any section carries one of these values.
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_NvtxTrace_RecordType
{
    NvTrace_NvtxTrace_RecordType_INVALID = 0,

    NvTrace_NvtxTrace_RecordType_RecordVersionInfo,
    NvTrace_NvtxTrace_RecordType_Configuration,
    NvTrace_NvtxTrace_RecordType_ProviderStatistics,

    // The object records, each followed by the continuation record that extends it -- which names
    // its owner in its own payload. Strings and domains are never destroyed; a push-pop range ends
    // with its RangePop thread event; a stack group has a destroyed record.
    NvTrace_NvtxTrace_RecordType_StringRegistered,
    NvTrace_NvtxTrace_RecordType_StringSegment,
    NvTrace_NvtxTrace_RecordType_DomainCreated,
    NvTrace_NvtxTrace_RecordType_PushPopRangeCreated,
    NvTrace_NvtxTrace_RecordType_PushPopRangeMessageSegment,
    NvTrace_NvtxTrace_RecordType_StackGroupCreated,
    NvTrace_NvtxTrace_RecordType_StackGroupStackRefs,
    NvTrace_NvtxTrace_RecordType_StackGroupDestroyed,

    // The thread events, likewise.
    NvTrace_NvtxTrace_RecordType_Mark,
    NvTrace_NvtxTrace_RecordType_MarkMessageSegment,
    NvTrace_NvtxTrace_RecordType_RangePop,
    NvTrace_NvtxTrace_RecordType_DroppedRangePush,
    NvTrace_NvtxTrace_RecordType_DroppedRangePop,

    NvTrace_NvtxTrace_RecordType_COUNT,
    NvTrace_NvtxTrace_RecordType_FORCE_INT = 0x7fffffff
};

#define NVTRACE_NVTX_THREAD_EVENT_RECORD_IDENTIFIER(recordType) NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER(NvTrace_ThreadEvent_StaticSourceProviderIndex_NVTX, (recordType))

// Self-describing record-format descriptor: recordVersionInfoSize and recordVersionInfoMinorVersion
// describe this struct itself, so a consumer can version it from the payload alone.
struct NvTrace_NvtxTrace_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;
    uint32_t providerStatisticsSize;
    uint32_t providerStatisticsMinorVersion;

    uint32_t stringRegisteredSize;
    uint32_t stringRegisteredMinorVersion;
    uint32_t stringSegmentSize;
    uint32_t stringSegmentMinorVersion;
    uint32_t domainCreatedSize;
    uint32_t domainCreatedMinorVersion;
    uint32_t pushPopRangeCreatedSize;
    uint32_t pushPopRangeCreatedMinorVersion;
    uint32_t pushPopRangeMessageSegmentSize;
    uint32_t pushPopRangeMessageSegmentMinorVersion;
    uint32_t stackGroupCreatedSize;
    uint32_t stackGroupCreatedMinorVersion;
    uint32_t stackGroupStackRefsSize;
    uint32_t stackGroupStackRefsMinorVersion;
    uint32_t stackGroupDestroyedSize;
    uint32_t stackGroupDestroyedMinorVersion;

    uint32_t eventAttributesSize;
    uint32_t eventAttributesMinorVersion;
    uint32_t stackRefSize;
    uint32_t stackRefMinorVersion;

    uint32_t markSize;
    uint32_t markMinorVersion;
    uint32_t markMessageSegmentSize;
    uint32_t markMessageSegmentMinorVersion;
    uint32_t rangePopSize;
    uint32_t rangePopMinorVersion;
    uint32_t droppedRangePushSize;
    uint32_t droppedRangePushMinorVersion;
    uint32_t droppedRangePopSize;
    uint32_t droppedRangePopMinorVersion;
};
#define NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_RecordVersionInfo, droppedRangePopMinorVersion)

// Independently meaningful 64-bit fields, sampled best-effort: there is no consistency protocol.
struct NvTrace_NvtxTrace_ProviderStatistics
{
    uint64_t registeredStringCount;
    uint64_t domainCount;
    uint64_t pushPopRangeCount;
    uint64_t stackGroupCount;

    uint64_t peakPushPopRangeCount;
    uint64_t peakStackGroupCount;

    // Objects refused because the section holding them was full.
    uint64_t droppedStringCount;
    uint64_t droppedDomainCount;
    uint64_t droppedPushPopRangeCount;
    uint64_t droppedStackGroupCount;

    // Range chains cut by state chunk exhaustion and Mark messages cut by thread-event exhaustion.
    // Registered strings are reserved whole or dropped, so they have no truncation counter.
    uint64_t truncatedMessageCount;

    // Stack-group chains cut by state chunk pool exhaustion.
    uint64_t truncatedStackGroupCount;
};
#define NvTrace_NvtxTrace_ProviderStatistics_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_ProviderStatistics, truncatedStackGroupCount)

/// A registered string, the leading record of a group in the Strings state arena. stringHandle is the
/// nvtxStringHandle_t the application holds and a RegisteredString message carries; never 0.
/// inlineStringSegmentLength UTF-8 bytes (no NUL) follow this struct at the published
/// stringRegisteredSize; the remaining totalStringLength - inlineStringSegmentLength arrive as the
/// NvTrace_NvtxTrace_StringSegment records reserved directly after it, so the group always tiles the
/// whole string.
struct NvTrace_NvtxTrace_StringRegistered
{
    uint64_t creationTimestampTsc;

    uint64_t stringHandle;

    uint32_t totalStringLength;
    uint32_t inlineStringSegmentLength;
};
#define NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StringRegistered, inlineStringSegmentLength)

/// One segment of a registered string, for the part its StringRegistered record did not carry inline:
/// stringSegmentLength bytes following this struct at the published stringSegmentSize, placed at
/// stringOffset into the string. A reader reassembles the string from the records in any order.
struct NvTrace_NvtxTrace_StringSegment
{
    uint64_t stringHandle;
    uint32_t stringOffset;
    uint32_t stringSegmentLength;
};
#define NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StringSegment, stringSegmentLength)

/// A domain, the entry record of a Domains table entry. domainIndex is the entry's index and the
/// nvtxDomainHandle_t; index 0 is the default domain, published at provider initialization with no name.
struct NvTrace_NvtxTrace_DomainCreated
{
    uint64_t creationTimestampTsc;

    /// NvTrace_NvtxTrace_StringRegistered::stringHandle of the name, or 0 for none.
    uint64_t nameStringHandle;

    uint32_t domainIndex;
};
#define NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_DomainCreated, domainIndex)

/// A push-pop range, the entry record of a PushPopRanges table entry.
/// Payload layout, the same shape a Mark thread event uses:
///   [NvTrace_NvtxTrace_PushPopRangeCreated] at the published pushPopRangeCreatedSize
///   [NvTrace_NvtxTrace_EventAttributes] only if hasEventAttributes, at the published eventAttributesSize
///   [inline message bytes]
/// The message bytes, in the encoding messageType names, lead with the inlineMessageSegmentLength bytes
/// that follow the attributes; the rest follow in the entry's chain as NvTrace_NvtxTrace_PushPopRangeMessageSegment
/// records. For RegisteredString, both message lengths are zero and an 8-byte stringHandle follows
/// the optional attributes instead of text. The handle is always inline and has no segments.
struct NvTrace_NvtxTrace_PushPopRangeCreated
{
    uint64_t startTimestampTsc;
    /// 0 while the range is open; stamped in place at the pop, moving no revision.
    uint64_t endTimestampTsc;

    /// Entry ref of the nearest ancestor range on the same stack, or NVTRACE_INVALID_REVISIONED_REF if none.
    NvTrace_RevisionedRef ancestorRef;
    uint64_t uniqueThreadId;
    uint32_t domainIndex;
    uint16_t nestingDepth;
    /// NvTrace_NvtxTrace_MessageType
    uint16_t messageType;

    /// Total text bytes (zero for RegisteredString); inlineMessageSegmentLength are carried here and
    /// NvTrace_NvtxTrace_PushPopRangeMessageSegment records carry the rest. Pool exhaustion can truncate the chain.
    uint32_t totalMessageLength;
    uint32_t inlineMessageSegmentLength;

    uint8_t hasEventAttributes;
};
#define NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_PushPopRangeCreated, hasEventAttributes)

/// One segment of a push-pop range's message, for the part its PushPopRangeCreated record did not
/// carry inline: messageSegmentLength bytes following this struct at the published
/// pushPopRangeMessageSegmentSize, placed at messageOffset into the message. A reader reassembles
/// the message from the records in any order.
struct NvTrace_NvtxTrace_PushPopRangeMessageSegment
{
    NvTrace_RevisionedRef pushPopRangeRef;

    uint32_t messageOffset;
    uint32_t messageSegmentLength;
};
#define NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_PushPopRangeMessageSegment, messageSegmentLength)

/// A stack group, the entry record of a StackGroups table entry: the ranges open on each of a thread's
/// domain stacks at one moment. The inlineStackRefCount NvTrace_NvtxTrace_StackRef entries follow this
/// struct at the published stackGroupCreatedSize; the rest follow in the entry's chain as
/// NvTrace_NvtxTrace_StackGroupStackRefs records.
struct NvTrace_NvtxTrace_StackGroupCreated
{
    uint64_t creationTimestampTsc;

    uint32_t totalStackRefCount;
    uint32_t inlineStackRefCount;
};
#define NvTrace_NvtxTrace_StackGroupCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StackGroupCreated, inlineStackRefCount)

struct NvTrace_NvtxTrace_StackRef
{
    /// Entry ref of the deepest allocated push-pop range on this stack, or NVTRACE_INVALID_REVISIONED_REF if none
    /// (a V1 buffer carries this struct too, with NvTrace_NvtxTrace_INVALID_REF_V1 for none).
    NvTrace_RevisionedRef topRef;
    uint32_t domainIndex;
    /// Logical stack depth; may exceed the depth of topRef's range if pushes were dropped.
    uint16_t nestingDepth;
};
#define NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StackRef, nestingDepth)

/// The stack refs of a stack group past the ones its StackGroupCreated record carried inline, a chain
/// record of its entry: stackRefCount NvTrace_NvtxTrace_StackRef entries follow this struct at the
/// published stackGroupStackRefsSize. Several refs ride one record so that this struct and the record's
/// desc are amortized over them; a chain cut by state chunk pool exhaustion tiles fewer than the
/// group's totalStackRefCount. The StackRef struct itself carries no owner -- it is also the created
/// record's inline trailer, which needs none -- so the owner is named here.
struct NvTrace_NvtxTrace_StackGroupStackRefs
{
    NvTrace_RevisionedRef stackGroupRef;
    uint32_t stackRefCount;
};
#define NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StackGroupStackRefs, stackRefCount)

/// A self-contained thread event: the ref stays meaningful after the entry's slot is recycled, and a
/// consumer joins the destroy to the created record it holds by the ref alone.
struct NvTrace_NvtxTrace_StackGroupDestroyed
{
    uint64_t timestampTsc;
    NvTrace_RevisionedRef stackGroupRef;
};
#define NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StackGroupDestroyed, stackGroupRef)


enum NvTrace_NvtxTrace_MessageType
{
    NvTrace_NvtxTrace_MessageType_Invalid = 0,

    NvTrace_NvtxTrace_MessageType_UTF8,
    NvTrace_NvtxTrace_MessageType_UTF16,
    NvTrace_NvtxTrace_MessageType_UTF32,
    NvTrace_NvtxTrace_MessageType_RegisteredString,

    NvTrace_NvtxTrace_MessageType_COUNT,
    NvTrace_NvtxTrace_MessageType_FORCE_INT = 0x7fffffff
};


enum NvTrace_NvtxTrace_EventColorType
{
    NvTrace_NvtxTrace_EventColorType_Invalid = 0,

    NvTrace_NvtxTrace_EventColorType_ARGB,

    NvTrace_NvtxTrace_EventColorType_COUNT,
    NvTrace_NvtxTrace_EventColorType_FORCE_INT = 0x7fffffff
};


enum NvTrace_NvtxTrace_EventPayloadType
{
    NvTrace_NvtxTrace_EventPayloadType_Invalid = 0,

    NvTrace_NvtxTrace_EventPayloadType_I32,
    NvTrace_NvtxTrace_EventPayloadType_I64,
    NvTrace_NvtxTrace_EventPayloadType_U32,
    NvTrace_NvtxTrace_EventPayloadType_U64,
    NvTrace_NvtxTrace_EventPayloadType_F32,
    NvTrace_NvtxTrace_EventPayloadType_F64,

    NvTrace_NvtxTrace_EventPayloadType_COUNT,
    NvTrace_NvtxTrace_EventPayloadType_FORCE_INT = 0x7fffffff
};


union NvTrace_NvtxTrace_EventPayload
{
    int32_t i32;
    int64_t i64;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;
};


/// The attributes of a mark or a push-pop range, carried inline in the record of the event they belong
/// to -- a Mark thread event or a PushPopRangeCreated entry record -- and so naming no owner of its own.
struct NvTrace_NvtxTrace_EventAttributes
{
    union NvTrace_NvtxTrace_EventPayload payload;
    uint32_t categoryId;
    uint32_t color;
    /// NvTrace_NvtxTrace_EventColorType
    uint8_t colorType;
    /// NvTrace_NvtxTrace_EventPayloadType
    uint8_t payloadType;
};
#define NvTrace_NvtxTrace_EventAttributes_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_EventAttributes, payloadType)


/// Payload layout:
///   [NvTrace_NvtxTrace_Mark] at the published markSize
///   [NvTrace_NvtxTrace_EventAttributes] only if hasEventAttributes, aligned to alignof(NvTrace_NvtxTrace_EventAttributes), at the published eventAttributesSize
///   [inline message bytes]
/// The message bytes lead with the inlineMessageSegmentLength that follow the attributes; the rest
/// follow as NvTrace_NvtxTrace_MarkMessageSegment thread events naming this one.
/// For RegisteredString, both message lengths are zero and an 8-byte stringHandle follows the optional
/// attributes instead of text. The handle is always inline and has no segments.
struct NvTrace_NvtxTrace_Mark
{
    uint64_t timestampTsc;
    uint32_t domainIndex;

    uint32_t totalMessageLength;
    uint32_t inlineMessageSegmentLength;

    /// NvTrace_NvtxTrace_MessageType
    uint16_t messageType;
    uint8_t hasEventAttributes;
};
#define NvTrace_NvtxTrace_Mark_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_Mark, hasEventAttributes)

struct NvTrace_NvtxTrace_MarkMessageSegment
{
    /// Address of the Mark thread event whose message this fills in.
    NvTrace_RevisionedRef markRef;

    uint32_t messageOffset;
    uint32_t messageSegmentLength;
};
#define NvTrace_NvtxTrace_MarkMessageSegment_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_MarkMessageSegment, messageSegmentLength)


struct NvTrace_NvtxTrace_RangePop
{
    NvTrace_RevisionedRef pushPopRangeRef;
    uint64_t endTimestampTsc;
};
#define NvTrace_NvtxTrace_RangePop_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_RangePop, endTimestampTsc)


struct NvTrace_NvtxTrace_DroppedRangePush
{
    uint64_t timestampTsc;
    uint32_t domainIndex;
};
#define NvTrace_NvtxTrace_DroppedRangePush_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_DroppedRangePush, domainIndex)


struct NvTrace_NvtxTrace_DroppedRangePop
{
    uint64_t timestampTsc;
    uint32_t domainIndex;
};
#define NvTrace_NvtxTrace_DroppedRangePop_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_DroppedRangePop, domainIndex)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
