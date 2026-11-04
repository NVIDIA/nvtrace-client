/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <perfetto_trace.pb.h>

#include <NvTrace/Target/NvTraceClientLogger.h>

namespace nv::trace {

    // Writer for native Perfetto protobuf traces (.perfetto-trace).
    //
    // Perfetto Format is defined here:
    //  Documentation: https://perfetto.dev/docs/reference/trace-packet-proto
    //  Proto Definition: (https://github.com/google/perfetto, protos/perfetto/trace/...)
    //  NvTrace thin wrapper is defined in perfetto_trace.proto.
    //
    // Responsibilities:
    //  - Allocate stable TrackDescriptor UUIDs and their parent/child hierarchy.
    //  - Intern event names, categories, debug annotation names, and string values to reduce size.
    //  - Buffer slice/instant/counter events, then emit correctly ordered
    //    TYPE_SLICE_BEGIN/END, TYPE_INSTANT and TYPE_COUNTER packets on serialize.
    //  - Stream each TracePacket individually (length-delimited as field #1 of the
    //    Trace message) so output is not bound by protobuf's 2 GB per-message
    //    serialization limit.
    //
    // The writer is format-only: it has no knowledge of NvTrace data model semantics.
    class PerfettoTraceWriter
    {
    public:
        using TrackUuid = uint64_t;
        using EventId = size_t;
        using ChildOrdering = perfetto::protos::TrackDescriptor::ChildTracksOrdering;

        struct AnnotationIid
        {
            uint64_t iid = 0;
        };

        struct StringValueIid
        {
            uint64_t iid = 0;
        };

        struct EventNameIid
        {
            uint64_t iid = 0;
        };

        struct EventCategoryIid
        {
            uint64_t iid = 0;
        };

        struct Annotation
        {
            // Distinguishes pointer from uint64_t in Value.
            struct Pointer
            {
                uint64_t address = 0;
            };

            // Refers to a dictionary or array in the writer-owned nested annotation storage.
            struct Nested
            {
                size_t index = 0;
            };

            using Value = std::variant<bool, uint64_t, int64_t, double, StringValueIid, Pointer, Nested>;

            // Name and string values are eagerly interned.
            AnnotationIid name;
            Value value;
        };

        PerfettoTraceWriter() = default;
        ~PerfettoTraceWriter() = default;

        // Sets the absolute UTC time represented by timestamp zero in trace events.
        // This must be called before Write() or WriteToFile().
        void SetTimestampOriginUtcNsec(uint64_t timestampOriginUtcNsec)
        {
            m_timestampOriginUtcNsec = timestampOriginUtcNsec;
            m_hasTimestampOriginUtcNsec = true;
        }

        Annotation MakeBool(const std::string& name, bool value)
        {
            return Annotation{InternAnnotationName(name), value};
        }
        Annotation MakeUint(const std::string& name, uint64_t value)
        {
            return Annotation{InternAnnotationName(name), value};
        }
        Annotation MakeInt(const std::string& name, int64_t value)
        {
            return Annotation{InternAnnotationName(name), value};
        }
        Annotation MakeDouble(const std::string& name, double value)
        {
            return Annotation{InternAnnotationName(name), value};
        }
        Annotation MakeString(const std::string& name, const std::string& value)
        {
            return Annotation{InternAnnotationName(name), InternStringValue(value)};
        }
        Annotation MakePointer(const std::string& name, uint64_t value)
        {
            return Annotation{InternAnnotationName(name), Annotation::Pointer{value}};
        }
        Annotation MakeStringValue(const std::string& value)
        {
            return Annotation{{}, InternStringValue(value)};
        }
        Annotation MakeIntValue(int64_t value)
        {
            return Annotation{{}, value};
        }
        Annotation MakeDictionary(const std::string& name, std::vector<Annotation>&& entries)
        {
            return MakeNested(name, NestedAnnotation::Type::Dictionary, std::move(entries));
        }
        Annotation MakeArray(const std::string& name, std::vector<Annotation>&& values)
        {
            return MakeNested(name, NestedAnnotation::Type::Array, std::move(values));
        }

        // --- Track creation ------------------------------------------------

        TrackUuid CreateNamedTrack(const std::string& name, TrackUuid parentUuid = 0)
        {
            return AddTrack(Track::Kind::Named, name, parentUuid).uuid;
        }

        TrackUuid CreateCounterTrack(const std::string& name, TrackUuid parentUuid, const std::string& unitName)
        {
            Track& track = AddTrack(Track::Kind::Counter, name, parentUuid);
            track.unitName = unitName;
            return track.uuid;
        }

        void SetTrackChildOrdering(TrackUuid uuid, ChildOrdering ordering)
        {
            GetTrack(uuid).childOrdering = ordering;
        }

        void SetTrackSiblingOrderRank(TrackUuid uuid, int32_t rank)
        {
            Track& track = GetTrack(uuid);
            track.hasSiblingOrderRank = true;
            track.siblingOrderRank = rank;
        }

        // --- Event emission ------------------------------------------------

        EventId AddSlice(
            TrackUuid trackUuid,
            const std::string& category,
            const std::string& name,
            uint64_t startTimestampNsec,
            uint64_t durationNsec,
            std::vector<Annotation> annotations = {},
            std::vector<uint64_t> flowIds = {},
            std::vector<uint64_t> terminatingFlowIds = {})
        {
            const EventId eventId = m_events.size();
            Event& event = m_events.emplace_back();
            // Slices are a single event that produces two timeline entries
            m_extraTimelineEntryCount += 1;
            event.kind = Event::Kind::Slice;
            event.trackUuid = trackUuid;
            event.startTimestampNsec = startTimestampNsec;
            event.endTimestampNsec = startTimestampNsec + durationNsec;
            event.nameIid = InternEventName(name);
            event.categoryIid = InternEventCategory(category);
            event.annotations = std::move(annotations);
            event.flowIds = std::move(flowIds);
            event.terminatingFlowIds = std::move(terminatingFlowIds);
            return eventId;
        }

        void AddFlowId(EventId eventId, uint64_t flowId)
        {
            m_events[eventId].flowIds.push_back(flowId);
        }

        void SetCorrelationId(EventId eventId, uint64_t correlationId)
        {
            m_events[eventId].correlationId = correlationId;
        }

        void AddInstant(
            TrackUuid trackUuid,
            const std::string& category,
            const std::string& name,
            uint64_t timestampNsec,
            std::vector<Annotation> annotations = {},
            std::vector<uint64_t> flowIds = {})
        {
            Event& event = m_events.emplace_back();
            event.kind = Event::Kind::Instant;
            event.trackUuid = trackUuid;
            event.startTimestampNsec = timestampNsec;
            event.endTimestampNsec = timestampNsec;
            event.nameIid = InternEventName(name);
            event.categoryIid = InternEventCategory(category);
            event.annotations = std::move(annotations);
            event.flowIds = std::move(flowIds);
        }

        void AddCounterSample(TrackUuid trackUuid, uint64_t timestampNsec, double value)
        {
            Event& event = m_events.emplace_back();
            event.kind = Event::Kind::Counter;
            event.trackUuid = trackUuid;
            event.startTimestampNsec = timestampNsec;
            event.endTimestampNsec = timestampNsec;
            event.counterValue = value;
        }

        // --- Serialization -------------------------------------------------

        // Streams the trace to sink. Returns false if the timestamp origin was not set.
        template <typename TByteSink>
        bool Write(TByteSink&& sink)
        {
            if (!m_hasTimestampOriginUtcNsec)
            {
                return false;
            }
            std::string scratch;
            return StreamTrace(sink, scratch);
        }

        // Serializes the trace and streams it to disk as a binary .perfetto-trace.
        // Packets are framed and written incrementally.
        bool WriteToFile(const std::filesystem::path& outputFilePath)
        {
            std::ofstream outFile(outputFilePath, std::ios::binary);
            if (!outFile.is_open())
            {
                NVTRACE_LOG_ERR(100, "Failed to open output file: %s", outputFilePath.string().c_str());
                return false;
            }

            auto sink = [&](const uint8_t* pData, size_t size) {
                outFile.write(reinterpret_cast<const char*>(pData), static_cast<std::streamsize>(size));
                return outFile.good();
            };

            const bool writeSucceeded = Write(sink);
            if (!writeSucceeded)
            {
                NVTRACE_LOG_ERR(100, "Failed to serialize Perfetto trace protobuf");
                return false;
            }
            return outFile.good();
        }

    private:
        // Uniquely identifies a producer + writer combo between packets.
        // Just use an arbitrary constant value here.
        static constexpr uint32_t kTrustedPacketSequenceId = 1;
        static constexpr size_t kMaxVarint64Size = 10;
        // Wire field numbers from Perfetto's full DebugAnnotation schema. Keeping them out of
        // the minimal generated schema avoids adding two repeated-field containers to every
        // scalar DebugAnnotation.
        static constexpr int kDebugAnnotationDictEntriesFieldNumber = 11;
        static constexpr int kDebugAnnotationArrayValuesFieldNumber = 12;

        // Will be converted to a TrackDescriptor message.
        struct Track
        {
            enum class Kind
            {
                Named,
                Counter,
            };

            TrackUuid uuid = 0;
            TrackUuid parentUuid = 0;
            std::string name;
            std::string unitName;
            Kind kind = Kind::Named;
            ChildOrdering childOrdering = perfetto::protos::TrackDescriptor::UNKNOWN;
            bool hasSiblingOrderRank = false;
            int32_t siblingOrderRank = 0;
        };

        // Will be converted to a TrackEvents message.
        struct Event
        {
            enum class Kind
            {
                Slice,
                Instant,
                Counter,
            };

            Kind kind = Kind::Slice;
            TrackUuid trackUuid = 0;
            uint64_t startTimestampNsec = 0;
            uint64_t endTimestampNsec = 0;
            EventNameIid nameIid;
            EventCategoryIid categoryIid;
            double counterValue = 0.0;
            uint64_t correlationId = 0;
            std::vector<Annotation> annotations;
            std::vector<uint64_t> flowIds;
            std::vector<uint64_t> terminatingFlowIds;
        };

        struct NestedAnnotation
        {
            enum class Type
            {
                Dictionary,
                Array,
            };

            Type type = Type::Dictionary;
            std::vector<Annotation> values;
        };

        Annotation MakeNested(const std::string& name, NestedAnnotation::Type type, std::vector<Annotation>&& values)
        {
            const size_t index = m_nestedAnnotations.size();
            m_nestedAnnotations.push_back({type, std::move(values)});
            return Annotation{InternAnnotationName(name), Annotation::Nested{index}};
        }

        // Used to sort Events into chronological order.
        struct TimelineEntry
        {
            enum class Kind
            {
                SliceBegin,
                SliceEnd,
                Instant,
                Counter,
            };

            uint64_t timestampNsec = 0;
            int phaseRank = 0;   // ends (0) < instants/counters (1) < begins (2) at equal timestamp
            uint64_t tieKey = 0; // sorted descending within a phaseRank to enforce nesting
            Kind kind = Kind::SliceBegin;
            const Event* pEvent = nullptr;
        };

        Track& AddTrack(Track::Kind kind, const std::string& name, TrackUuid parentUuid)
        {
            Track& track = m_tracks.emplace_back();
            track.uuid = ++m_nextTrackUuid;
            track.name = name;
            track.kind = kind;
            track.parentUuid = parentUuid;
            return track;
        }

        // Track UUIDs are dense and 1-based in creation order.
        Track& GetTrack(TrackUuid uuid)
        {
            return m_tracks[uuid - 1];
        }

        EventNameIid InternEventName(const std::string& name)
        {
            if (name.empty())
            {
                return {};
            }
            auto it = m_eventNameIids.find(name);
            if (it != m_eventNameIids.end())
            {
                return it->second;
            }
            const EventNameIid iid{static_cast<uint64_t>(m_eventNames.size()) + 1}; // iids are 1-based
            m_eventNameIids.emplace(name, iid);
            m_eventNames.emplace_back(iid.iid, name);
            return iid;
        }

        EventCategoryIid InternEventCategory(const std::string& category)
        {
            if (category.empty())
            {
                return {};
            }
            auto it = m_eventCategoryIids.find(category);
            if (it != m_eventCategoryIids.end())
            {
                return it->second;
            }
            const EventCategoryIid iid{static_cast<uint64_t>(m_eventCategories.size()) + 1}; // iids are 1-based
            m_eventCategoryIids.emplace(category, iid);
            m_eventCategories.emplace_back(iid.iid, category);
            return iid;
        }

        AnnotationIid InternAnnotationName(const std::string& name)
        {
            auto it = m_annotationIids.find(name);
            if (it != m_annotationIids.end())
            {
                return it->second;
            }
            const AnnotationIid iid{static_cast<uint64_t>(m_annotationNames.size()) + 1}; // iids are 1-based
            m_annotationIids.emplace(name, iid);
            m_annotationNames.emplace_back(iid.iid, name);
            return iid;
        }

        StringValueIid InternStringValue(const std::string& value)
        {
            auto it = m_stringValueIids.find(value);
            if (it != m_stringValueIids.end())
            {
                return it->second;
            }
            const StringValueIid iid{static_cast<uint64_t>(m_stringValues.size()) + 1}; // iids are 1-based
            m_stringValueIids.emplace(value, iid);
            m_stringValues.emplace_back(iid.iid, value);
            return iid;
        }

        static size_t EncodeVarint(uint8_t* pBuffer, uint64_t value)
        {
            size_t count = 0;
            while (value >= 0x80)
            {
                pBuffer[count++] = static_cast<uint8_t>(value) | 0x80;
                value >>= 7;
            }
            pBuffer[count++] = static_cast<uint8_t>(value);
            return count;
        }

        // Frames a single TracePacket as field #1 of Trace.
        template <typename TByteSink>
        static bool WriteFramedPacket(TByteSink& sink, const perfetto::protos::TracePacket& packet, std::string& scratch)
        {
            scratch.clear();
            // The payload is prefixed by two varint64_t values: the packet tag and payload size.
            constexpr size_t kPrefixCapacity = 2 * kMaxVarint64Size;
            scratch.resize(kPrefixCapacity);
            const bool serialized = packet.AppendToString(&scratch);
            if (!serialized)
            {
                return false;
            }
            const size_t payloadSize = scratch.size() - kPrefixCapacity;
            constexpr uint64_t kTracePacketTag = (1ULL << 3) | 2ULL; // field 1, wire type 2
            uint8_t* pPrefix = reinterpret_cast<uint8_t*>(scratch.data());
            // Encode the two varint64_t values.
            size_t prefixSize = EncodeVarint(pPrefix, kTracePacketTag);
            prefixSize += EncodeVarint(pPrefix + prefixSize, payloadSize);
            // Right-align the encoded values.
            const size_t prefixOffset = kPrefixCapacity - prefixSize;
            std::move_backward(
                scratch.begin(),
                scratch.begin() + prefixSize,
                scratch.begin() + kPrefixCapacity);
            // Sink the combined prefix and payload.
            return sink(
                reinterpret_cast<const uint8_t*>(scratch.data() + prefixOffset),
                prefixSize + payloadSize);
        }

        template <typename TByteSink>
        bool StreamTrace(TByteSink& sink, std::string& scratch)
        {
            const bool clockSnapshotSucceeded = EmitClockSnapshotPacket(sink, scratch);
            if (!clockSnapshotSucceeded)
            {
                return false;
            }
            const bool internedDataSucceeded = EmitInternedDataPacket(sink, scratch);
            if (!internedDataSucceeded)
            {
                return false;
            }
            const bool trackDescriptorsSucceeded = EmitTrackDescriptorPackets(sink, scratch);
            if (!trackDescriptorsSucceeded)
            {
                return false;
            }
            return EmitEventPackets(sink, scratch);
        }

        template <typename TByteSink>
        bool EmitClockSnapshotPacket(TByteSink& sink, std::string& scratch)
        {
            perfetto::protos::TracePacket packet;
            perfetto::protos::ClockSnapshot* pClockSnapshot = packet.mutable_clock_snapshot();
            pClockSnapshot->set_primary_trace_clock(perfetto::protos::BUILTIN_CLOCK_REALTIME);

            // Perfetto otherwise assigns the file an implicit BOOTTIME clock even
            // though every timestamped NvTrace packet explicitly uses REALTIME.
            perfetto::protos::ClockSnapshot::Clock* pBoottimeClock = pClockSnapshot->add_clocks();
            pBoottimeClock->set_clock_id(perfetto::protos::BUILTIN_CLOCK_BOOTTIME);
            pBoottimeClock->set_timestamp(0);

            perfetto::protos::ClockSnapshot::Clock* pRealtimeClock = pClockSnapshot->add_clocks();
            pRealtimeClock->set_clock_id(perfetto::protos::BUILTIN_CLOCK_REALTIME);
            pRealtimeClock->set_timestamp(0);

            return WriteFramedPacket(sink, packet, scratch);
        }

        template <typename TByteSink>
        bool EmitInternedDataPacket(TByteSink& sink, std::string& scratch)
        {
            perfetto::protos::TracePacket packet;
            packet.set_trusted_packet_sequence_id(kTrustedPacketSequenceId);
            packet.set_sequence_flags(
                perfetto::protos::TracePacket::SEQ_INCREMENTAL_STATE_CLEARED | perfetto::protos::TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);

            perfetto::protos::InternedData* pInterned = packet.mutable_interned_data();
            for (const auto& [iid, name] : m_eventCategories)
            {
                perfetto::protos::EventCategory* pEntry = pInterned->add_event_categories();
                pEntry->set_iid(iid);
                pEntry->set_name(name);
            }
            for (const auto& [iid, name] : m_eventNames)
            {
                perfetto::protos::EventName* pEntry = pInterned->add_event_names();
                pEntry->set_iid(iid);
                pEntry->set_name(name);
            }
            for (const auto& [iid, name] : m_annotationNames)
            {
                perfetto::protos::DebugAnnotationName* pEntry = pInterned->add_debug_annotation_names();
                pEntry->set_iid(iid);
                pEntry->set_name(name);
            }
            for (const auto& [iid, value] : m_stringValues)
            {
                perfetto::protos::InternedString* pEntry = pInterned->add_debug_annotation_string_values();
                pEntry->set_iid(iid);
                pEntry->set_str(value);
            }

            return WriteFramedPacket(sink, packet, scratch);
        }

        template <typename TByteSink>
        bool EmitTrackDescriptorPackets(TByteSink& sink, std::string& scratch)
        {
            perfetto::protos::TracePacket packet;
            for (const Track& track : m_tracks)
            {
                packet.Clear();
                packet.set_trusted_packet_sequence_id(kTrustedPacketSequenceId);

                perfetto::protos::TrackDescriptor* pDescriptor = packet.mutable_track_descriptor();
                pDescriptor->set_uuid(track.uuid);
                if (!track.name.empty())
                {
                    pDescriptor->set_name(track.name);
                }
                if (track.parentUuid != 0)
                {
                    pDescriptor->set_parent_uuid(track.parentUuid);
                }

                if (track.childOrdering != perfetto::protos::TrackDescriptor::UNKNOWN)
                {
                    pDescriptor->set_child_ordering(track.childOrdering);
                }

                if (track.hasSiblingOrderRank)
                {
                    pDescriptor->set_sibling_order_rank(track.siblingOrderRank);
                }

                switch (track.kind)
                {
                    case Track::Kind::Named:
                    {
                        break;
                    }
                    case Track::Kind::Counter:
                    {
                        perfetto::protos::CounterDescriptor* pCounter = pDescriptor->mutable_counter();
                        if (!track.unitName.empty())
                        {
                            pCounter->set_unit_name(track.unitName);
                        }
                        break;
                    }
                }

                if (!WriteFramedPacket(sink, packet, scratch))
                {
                    return false;
                }
            }
            return true;
        }

        template <typename TByteSink>
        bool EmitEventPackets(TByteSink& sink, std::string& scratch)
        {
            std::vector<TimelineEntry> timeline;
            timeline.reserve(m_events.size() + m_extraTimelineEntryCount);

            for (const Event& event : m_events)
            {
                switch (event.kind)
                {
                    case Event::Kind::Slice:
                    {
                        const bool isZeroDuration = event.startTimestampNsec == event.endTimestampNsec;
                        TimelineEntry beginEntry;
                        beginEntry.timestampNsec = event.startTimestampNsec;
                        beginEntry.phaseRank = isZeroDuration ? 1 : 2;
                        beginEntry.tieKey = event.endTimestampNsec; // outer (later end) opens first
                        beginEntry.kind = TimelineEntry::Kind::SliceBegin;
                        beginEntry.pEvent = &event;
                        timeline.push_back(beginEntry);

                        TimelineEntry endEntry;
                        endEntry.timestampNsec = event.endTimestampNsec;
                        endEntry.phaseRank = isZeroDuration ? 1 : 0;
                        endEntry.tieKey = event.startTimestampNsec; // inner (later start) closes first
                        endEntry.kind = TimelineEntry::Kind::SliceEnd;
                        endEntry.pEvent = &event;
                        timeline.push_back(endEntry);
                        break;
                    }
                    case Event::Kind::Instant:
                    {
                        TimelineEntry entry;
                        entry.timestampNsec = event.startTimestampNsec;
                        entry.phaseRank = 1;
                        entry.kind = TimelineEntry::Kind::Instant;
                        entry.pEvent = &event;
                        timeline.push_back(entry);
                        break;
                    }
                    case Event::Kind::Counter:
                    {
                        TimelineEntry entry;
                        entry.timestampNsec = event.startTimestampNsec;
                        entry.phaseRank = 1;
                        entry.kind = TimelineEntry::Kind::Counter;
                        entry.pEvent = &event;
                        timeline.push_back(entry);
                        break;
                    }
                }
            }

            std::stable_sort(timeline.begin(), timeline.end(), [](const TimelineEntry& lhs, const TimelineEntry& rhs) {
                if (lhs.timestampNsec != rhs.timestampNsec)
                {
                    return lhs.timestampNsec < rhs.timestampNsec;
                }
                if (lhs.phaseRank != rhs.phaseRank)
                {
                    return lhs.phaseRank < rhs.phaseRank;
                }
                return lhs.tieKey > rhs.tieKey;
            });

            perfetto::protos::TracePacket packet;
            for (const TimelineEntry& entry : timeline)
            {
                packet.Clear();
                packet.set_trusted_packet_sequence_id(kTrustedPacketSequenceId);
                packet.set_sequence_flags(perfetto::protos::TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);
                packet.set_timestamp(m_timestampOriginUtcNsec + entry.timestampNsec);
                packet.set_timestamp_clock_id(perfetto::protos::BUILTIN_CLOCK_REALTIME);

                perfetto::protos::TrackEvent* pTrackEvent = packet.mutable_track_event();
                pTrackEvent->set_track_uuid(entry.pEvent->trackUuid);

                switch (entry.kind)
                {
                    case TimelineEntry::Kind::SliceBegin:
                    {
                        pTrackEvent->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
                        ApplyNameAndCategory(pTrackEvent, *entry.pEvent);
                        ApplyAnnotations(pTrackEvent, entry.pEvent->annotations);
                        ApplyFlows(pTrackEvent, *entry.pEvent);
                        if (entry.pEvent->correlationId)
                        {
                            pTrackEvent->set_correlation_id(entry.pEvent->correlationId);
                        }
                        break;
                    }
                    case TimelineEntry::Kind::SliceEnd:
                    {
                        pTrackEvent->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
                        break;
                    }
                    case TimelineEntry::Kind::Instant:
                    {
                        pTrackEvent->set_type(perfetto::protos::TrackEvent::TYPE_INSTANT);
                        ApplyNameAndCategory(pTrackEvent, *entry.pEvent);
                        ApplyAnnotations(pTrackEvent, entry.pEvent->annotations);
                        ApplyFlows(pTrackEvent, *entry.pEvent);
                        break;
                    }
                    case TimelineEntry::Kind::Counter:
                    {
                        pTrackEvent->set_type(perfetto::protos::TrackEvent::TYPE_COUNTER);
                        pTrackEvent->set_double_counter_value(entry.pEvent->counterValue);
                        break;
                    }
                }

                if (!WriteFramedPacket(sink, packet, scratch))
                {
                    return false;
                }
            }
            return true;
        }

        static void ApplyNameAndCategory(perfetto::protos::TrackEvent* pTrackEvent, const Event& event)
        {
            if (event.nameIid.iid != 0)
            {
                pTrackEvent->set_name_iid(event.nameIid.iid);
            }
            if (event.categoryIid.iid != 0)
            {
                pTrackEvent->add_category_iids(event.categoryIid.iid);
            }
        }

        static void ApplyFlows(perfetto::protos::TrackEvent* pTrackEvent, const Event& event)
        {
            for (uint64_t flowId : event.flowIds)
            {
                pTrackEvent->add_flow_ids(flowId);
            }
            for (uint64_t flowId : event.terminatingFlowIds)
            {
                pTrackEvent->add_terminating_flow_ids(flowId);
            }
        }

        void ApplyNestedAnnotation(perfetto::protos::DebugAnnotation* pAnnotation, size_t nestedAnnotationIndex) const
        {
            const NestedAnnotation& nested = m_nestedAnnotations[nestedAnnotationIndex];
            const int fieldNumber = nested.type == NestedAnnotation::Type::Dictionary
                                      ? kDebugAnnotationDictEntriesFieldNumber
                                      : kDebugAnnotationArrayValuesFieldNumber;
            for (const Annotation& nestedValue : nested.values)
            {
                perfetto::protos::DebugAnnotation nestedAnnotation;
                ApplyAnnotation(&nestedAnnotation, nestedValue);
                std::string serializedNestedAnnotation;
                nestedAnnotation.SerializeToString(&serializedNestedAnnotation);
                pAnnotation->mutable_unknown_fields()->AddLengthDelimited(fieldNumber, serializedNestedAnnotation);
            }
        }

        void ApplyAnnotation(perfetto::protos::DebugAnnotation* pAnnotation, const Annotation& annotation) const
        {
            if (annotation.name.iid != 0)
            {
                pAnnotation->set_name_iid(annotation.name.iid);
            }
            std::visit(
                [&](const auto& value) {
                    using TValue = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<TValue, bool>)
                    {
                        pAnnotation->set_bool_value(value);
                    }
                    else if constexpr (std::is_same_v<TValue, uint64_t>)
                    {
                        pAnnotation->set_uint_value(value);
                    }
                    else if constexpr (std::is_same_v<TValue, int64_t>)
                    {
                        pAnnotation->set_int_value(value);
                    }
                    else if constexpr (std::is_same_v<TValue, double>)
                    {
                        pAnnotation->set_double_value(value);
                    }
                    else if constexpr (std::is_same_v<TValue, StringValueIid>)
                    {
                        pAnnotation->set_string_value_iid(value.iid);
                    }
                    else if constexpr (std::is_same_v<TValue, Annotation::Pointer>)
                    {
                        pAnnotation->set_pointer_value(value.address);
                    }
                    else if constexpr (std::is_same_v<TValue, Annotation::Nested>)
                    {
                        ApplyNestedAnnotation(pAnnotation, value.index);
                    }
                },
                annotation.value);
        }

        void ApplyAnnotations(perfetto::protos::TrackEvent* pTrackEvent, const std::vector<Annotation>& annotations) const
        {
            for (const Annotation& annotation : annotations)
            {
                perfetto::protos::DebugAnnotation* pAnnotation = pTrackEvent->add_debug_annotations();
                ApplyAnnotation(pAnnotation, annotation);
            }
        }

        TrackUuid m_nextTrackUuid = 0;
        uint64_t m_timestampOriginUtcNsec = 0;
        bool m_hasTimestampOriginUtcNsec = false;
        std::vector<Track> m_tracks;
        std::vector<Event> m_events;
        size_t m_extraTimelineEntryCount = 0;
        std::vector<NestedAnnotation> m_nestedAnnotations;

        std::unordered_map<std::string, EventNameIid> m_eventNameIids;
        std::unordered_map<std::string, EventCategoryIid> m_eventCategoryIids;
        std::unordered_map<std::string, AnnotationIid> m_annotationIids;
        std::unordered_map<std::string, StringValueIid> m_stringValueIids;

        std::vector<std::pair<uint64_t, std::string>> m_eventNames;
        std::vector<std::pair<uint64_t, std::string>> m_eventCategories;
        std::vector<std::pair<uint64_t, std::string>> m_annotationNames;
        std::vector<std::pair<uint64_t, std::string>> m_stringValues;
    };

} // namespace nv::trace
