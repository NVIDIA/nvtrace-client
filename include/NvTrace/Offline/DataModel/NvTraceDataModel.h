/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_grctx_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_hw_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_pm_counters.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/Target/Consumer/NvTraceCodecEvents.h>
#include <NvTrace/Target/Consumer/NvTraceCodecEvents_generated.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConstants.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderFunctionIds_generated.h>
#include <NvTrace/Target/NvTraceCudaApiTraceRecordUtilities.h>
#include <NvTrace/Target/NvTraceGpuPartitionInfo_generated.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceSystemInfo_generated.h>
#include <NvTrace/Target/NvTraceUtf8Conversion.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace nv::trace::Codec {

    // A timestamp of zero is invalid, regardless of which units it's denominated in. (units as in TSC, GPU nsec, UTC nsec)
    inline constexpr uint64_t InvalidTimestamp = 0;
    inline constexpr uint64_t MinTimestampTsc = 1;
    inline constexpr uint64_t MaxTimestampTsc = UINT64_MAX;
    inline constexpr uint64_t InvalidCorrelationId = UINT64_MAX;
    inline constexpr uint64_t InvalidRecordLocation = 0;

    struct ProviderData
    {
        NVTRACE_LUID providerLuid = NVTRACE_NIL_LUID;
    };

    struct CpuClockCalibrationData : ProviderData
    {
        std::vector<TscRtcCalibrationPoint> calibrationPoints;
    };

    struct GpuClockCalibrationData : ProviderData
    {
        uint32_t circularBufferCellCount = 0;
        uint64_t updateIntervalNsec = 0;
        std::vector<TscGpuCalibrationPoint> calibrationPoints;
    };

    struct ClockCalibrationInterval
    {
        uint64_t startUtcNsec = InvalidTimestamp;
        uint64_t endUtcNsec = InvalidTimestamp;
        uint64_t clampedStartUtcNsec = InvalidTimestamp;
        uint64_t clampedEndUtcNsec = InvalidTimestamp;
        bool isExtrapolated = false;
    };

    struct MetricToConfigure
    {
        std::string metricName;
        bool keepInstances = false;
        /// How the metric set named and grouped this metric. Empty for a metric from architectures[], and
        /// for the peak that backs a sub-metric's maxValue, neither of which is a row of its own.
        std::string displayName;
        std::string category;
    };

    struct ThreadEventProviderData : ProviderData
    {
        struct ThreadData
        {
            uint64_t creationTimestampTsc = MinTimestampTsc;
            uint64_t destructionTimestampTsc = MaxTimestampTsc;
            uint32_t nativeThreadId = 0;
            uint64_t uniqueThreadId = 0;
        };

        /// Map ThreadEvent providerMappingId -> mapped producer provider.
        std::map<uint32_t, NVTRACE_LUID> providerMappings;
        /// Map uniqueThreadId -> Thread
        std::map<uint64_t, ThreadData> threads;

        ThreadData& EnsureThread(uint64_t uniqueThreadId)
        {
            auto [itrThreadData, inserted] = threads.try_emplace(uniqueThreadId);
            if (inserted)
            {
                itrThreadData->second.uniqueThreadId = uniqueThreadId;
            }
            return itrThreadData->second;
        }
    };

    struct GpuProviderLifecycleEvent
    {
        uint64_t timestampTsc = InvalidTimestamp;
        uint64_t endTimestampTsc = MaxTimestampTsc;

        NvTrace_GPU_Provider_LifecycleEventType lifecycleEventType = NvTrace_GPU_Provider_LifecycleEventType_Invalid;
        NvTrace_GPU_Provider_PauseReason pauseReason = NvTrace_GPU_Provider_PauseReason_None;
        NvTrace_GPU_Provider_ResumePhase resumePhase = NvTrace_GPU_Provider_ResumePhase_Invalid;
        uint32_t retryCount = 0;
        NvTrace_GPU_Provider_ResumeFailureReason resumeFailureReason = NvTrace_GPU_Provider_ResumeFailureReason_None;
    };

    struct GpuPmCountersData : ProviderData
    {
        // provider configuration
        uint32_t pmSampleRateHz = 0;
        uint64_t updateIntervalNsec = 0;
        uint64_t readbackHoldoffNsec = 0;
        bool keepLatest = false;
        /// One dimensional quantity of a timeline metric, carrying the label the set gave it.
        struct TimelineSubMetric
        {
            std::string name;
            std::string suffix;
            /// The counter this resolves to. Empty in traces written before the field, where the parent's
            /// baseId joined with the suffix is the best available reconstruction.
            std::string fullId;
            double multiplier = 1.0;
            double maxValue = 0.0;
        };

        /// A metric as the timeline draws it: one display name covering every sub-metric beneath it.
        struct TimelineMetric
        {
            std::string name;
            std::string baseId;
            std::string description;
            std::string colorString;
            std::vector<TimelineSubMetric> subMetrics;
        };

        /// What a row draws, named so rows can reference it rather than repeat its members.
        struct TimelineMetricGroup
        {
            std::string name;
            std::string description;
            std::vector<std::string> metricNames;
            /// Which sub-metrics of those metrics this group draws. Empty means all of them.
            std::vector<std::string> subMetricNames;
            bool sumAcrossMetrics = false;
        };

        /// A row, nesting to whatever depth the set declared.
        struct TimelineRow
        {
            std::string metricGroupName;
            bool stacked = false;
            double heightScale = 1.0;
            std::vector<TimelineRow> children;
        };

        struct TimelineView
        {
            std::string name;
            std::vector<TimelineRow> rows;
        };

        /// The row hierarchy the metric set declared. Empty for architectures[], and for traces captured
        /// before it was written, which fall back to the per-metric names below.
        struct TimelineConfig
        {
            uint32_t version = 0;
            std::vector<TimelineView> views;
            std::vector<TimelineMetricGroup> metricGroups;
            std::vector<TimelineMetric> metrics;

            bool Empty() const
            {
                return views.empty();
            }

            /// Where one row draws a metric: the path to the row, outermost first, and which of the metric's
            /// sub-metrics that row shows. An empty selection means all of them.
            struct TimelineRowPlacement
            {
                std::vector<std::string> rowPath;
                std::vector<std::string> subMetricNames;

                /// Only the first sub-metric a group lists is drawn on the timeline; the rest belong to the
                /// tooltip table. The list is ordered and the loader has already flattened any nesting, so
                /// the primary is simply the front. Drawing the others would also spoil the row's own value,
                /// since a max across a percentage, a byte count and a rate is not a quantity.
                bool DrawsSubMetric(const std::string& subMetricName) const
                {
                    return subMetricNames.empty() || subMetricNames.front() == subMetricName;
                }
            };

            /// Row paths in the order the view declares them, outermost first, parents before children.
            /// The exporter creates tracks in this order so a row sits where the set put it rather than
            /// where the first metric that mentioned it happened to fall.
            std::vector<std::vector<std::string>> RowPathsInDeclaredOrder(const std::string& viewName = "Default") const
            {
                std::vector<std::vector<std::string>> rowPaths;
                if (views.empty())
                {
                    return rowPaths;
                }
                const TimelineView* pView = &views.front();
                for (const TimelineView& candidate : views)
                {
                    if (candidate.name == viewName)
                    {
                        pView = &candidate;
                        break;
                    }
                }
                constexpr size_t kMaxTimelineRowDepth = 32;
                std::vector<std::string> path;
                const auto fnWalk = [&](const auto& self, const TimelineRow& row, size_t depth) -> void {
                    if (depth >= kMaxTimelineRowDepth)
                    {
                        return;
                    }
                    path.push_back(row.metricGroupName);
                    rowPaths.push_back(path);
                    for (const TimelineRow& child : row.children)
                    {
                        self(self, child, depth + 1);
                    }
                    path.pop_back();
                };
                for (const TimelineRow& row : pView->rows)
                {
                    fnWalk(fnWalk, row, 0);
                }
                return rowPaths;
            }

            /// Every placement of a metric. Empty for a metric no row names. A metric may be listed by
            /// several groups, so it can own more than one and is drawn once under each. Resolved once per
            /// trace, since every counter of a metric shares its rows.
            /// viewName picks a perspective; the first is used when that name is not there.
            std::map<std::string, std::vector<TimelineRowPlacement>> ResolveRowPathsByMetricName(const std::string& viewName = "Default") const
            {
                std::map<std::string, std::vector<TimelineRowPlacement>> rowPathsByMetricName;
                if (views.empty())
                {
                    return rowPathsByMetricName;
                }
                const TimelineView* pView = &views.front();
                for (const TimelineView& candidate : views)
                {
                    if (candidate.name == viewName)
                    {
                        pView = &candidate;
                        break;
                    }
                }

                std::map<std::string, const TimelineMetricGroup*> groupsByName;
                for (const TimelineMetricGroup& group : metricGroups)
                {
                    groupsByName.emplace(group.name, &group);
                }

                // The rows come from a trace file, so the walk is bounded rather than trusting its depth.
                constexpr size_t kMaxTimelineRowDepth = 32;
                std::vector<std::string> path;
                // Self-passed rather than a std::function: this recurses per row and a std::function would
                // add an allocation and an indirect call to each step for no gain.
                const auto fnWalk = [&](const auto& self, const TimelineRow& row, size_t depth) -> void {
                    if (depth >= kMaxTimelineRowDepth)
                    {
                        return;
                    }
                    path.push_back(row.metricGroupName);
                    if (const auto groupIt = groupsByName.find(row.metricGroupName); groupIt != groupsByName.end())
                    {
                        for (const std::string& metricName : groupIt->second->metricNames)
                        {
                            // Appended, not replaced: a metric named by two rows belongs to both, and the
                            // walk order decides which is drawn first rather than which one survives.
                            rowPathsByMetricName[metricName].push_back({path, groupIt->second->subMetricNames});
                        }
                    }
                    for (const TimelineRow& child : row.children)
                    {
                        self(self, child, depth + 1);
                    }
                    path.pop_back();
                };
                for (const TimelineRow& row : pView->rows)
                {
                    fnWalk(fnWalk, row, 0);
                }
                return rowPathsByMetricName;
            }
        };

        TimelineConfig timeline;
        /// The selected set's custom metric definitions, as the script the counter library takes. Offline
        /// evaluation builds its own evaluator, which resolves no name these introduce until it is
        /// registered there too. Empty when the set defines none, or for a trace written before the field.
        std::string customMetricsScript;

        std::vector<MetricToConfigure> metricsToConfigure;
        std::vector<std::string> metricsToEvaluate;
        /// The MetricConfig document this configuration was resolved from, as authored. Empty for
        /// architectures[] and for traces written before it was carried. The resolved fields are above; this
        /// carries the rest -- groups, timeline views, aggregation -- for a reader without the file.
        std::string metricConfigDocument;

        // counter data info
        // Published size of CounterSample's fixed-field segment. RangeData begins at the 8-byte-aligned end of this segment.
        uint32_t counterSampleFixedFieldSize = 0;
        uint32_t counterDataPreambleSize = 0;
        uint32_t sampleRangeDataSize = 0;
        uint32_t sampleEvaluatedMetricCount = 0;  // in units of doubles (not bytes)
        std::vector<uint8_t> counterDataPreamble; // raw bytes, used when the counter data is decoded in the exporter

        std::vector<uint64_t> sampleRecordLocations;

        using LifecycleEvent = GpuProviderLifecycleEvent;

        struct ProviderDiagnostic
        {
            NvTrace_GPU_PmCounters_RecordType recordType = NvTrace_GPU_PmCounters_RecordType_Invalid;

            union
            {
                struct
                {
                    uint64_t timestampTsc;
                    NvTrace_GPU_PmCounters_ProviderFailureReason failureReason;
                } initializationFailure;
                struct
                {
                    uint64_t updateStartTimestampTsc;
                    uint64_t updateEndTimestampTsc;
                    uint64_t updateThreadCpuTimeNsec;
                    uint64_t updateWallTimeNsec; // Full update duration before clipping; zero means unavailable.
                } updateTimeExceeded;
                struct
                {
                    uint64_t startTimestampTsc;
                    uint64_t endTimestampTsc;
                    NvTrace_GPU_PmCounters_SampleDiscontinuityReason reason;
                } sampleDiscontinuity;
            } payload = {};
        };

        std::vector<LifecycleEvent> lifecycleEvents;
        std::vector<ProviderDiagnostic> providerDiagnostics;
    };

    struct GpuPmCountersSample
    {
        uint64_t startTimestampUtcNsec = InvalidTimestamp;
        uint64_t endTimestampUtcNsec = InvalidTimestamp;

        NvTrace_GPU_PmCounters_CounterSample counterSample = {};

        const uint8_t* pSampleRangeData = nullptr;
        uint32_t sampleRangeDataSize = 0;

        const double* pSampleEvaluatedMetrics = nullptr;
        uint32_t sampleEvaluatedMetricCount = 0;
    };

    struct GpuHwTraceData : ProviderData
    {
        uint32_t circularBufferCellCount = 0;
        uint64_t updateIntervalNsec = 0;
        uint32_t readbackHoldoffNsec = 0;
        bool keepLatest = true;
        bool showInternalEvents = true;
        uint32_t recordBufferSize = 4 * 1024 * 1024;
        bool allowLossyProvider = false;

        struct ContextInterval
        {
            uint32_t contextId = 0;
            uint64_t loadTimestampGpuNsec = InvalidTimestamp;
            uint64_t saveTimestampGpuNsec = InvalidTimestamp;
        };

        struct KernelExecution
        {
            uint64_t startTimestampGpuNsec = InvalidTimestamp;
            uint64_t endTimestampGpuNsec = InvalidTimestamp;
        };

        struct ProviderDiagnostic
        {
            uint64_t timestampTsc = InvalidTimestamp;
            NvTrace_GPU_HwTrace_RecordType recordType = NvTrace_GPU_HwTrace_RecordType_Invalid;
            NvTrace_GPU_HwTrace_ProviderFailureReason initializationFailureReason = NvTrace_GPU_HwTrace_ProviderFailureReason_None;
        };

        using LifecycleEvent = GpuProviderLifecycleEvent;

        /// Vector of context intervals in chronological order
        std::vector<ContextInterval> contextTrace;

        /// Vector of GpuHwTrace provider diagnostics
        std::vector<ProviderDiagnostic> providerDiagnostics;

        /// Vector of GpuHwTrace provider lifecycle events
        std::vector<LifecycleEvent> lifecycleEvents;
    };

    struct GpuGrCtxTraceData : ProviderData
    {
        uint32_t circularBufferCellCount = 0;
        uint64_t updateIntervalNsec = 0;

        struct ProviderDiagnostic
        {
            uint64_t timestampTsc = InvalidTimestamp;
            NvTrace_GPU_GrCtxTrace_RecordType recordType = NvTrace_GPU_GrCtxTrace_RecordType_Invalid;
            NvTrace_GPU_GrCtxTrace_ProviderFailureReason initializationFailureReason = NvTrace_GPU_GrCtxTrace_ProviderFailureReason_None;
        };

        static constexpr uint64_t CtxMinimalTimestampGpuNsec = 0ULL;
        static constexpr uint64_t CtxMaximalTimestampGpuNsec = UINT64_MAX;
        static constexpr uint64_t InvalidSequenceNumber = UINT64_MAX;
        // If an unprivileged user collects FECS trace, the RM only allows that user to see PID values from that same uid.
        // Other users' PIDs will be output as 0x0. It's a security feature.
        static constexpr uint64_t HiddenPid = static_cast<uint64_t>(0);
        static constexpr uint64_t UnknownPid = static_cast<uint64_t>(0xFFFFFFFF);

        // organization phase: organize the grCtx trace events
        struct ResidentContext
        {
            uint64_t ctxLoadTimestampGpuNsec = CtxMinimalTimestampGpuNsec;
            uint64_t ctxSaveTimestampGpuNsec = CtxMaximalTimestampGpuNsec;
            // Sequence numbers of the load and save records in the provider's buffer, which numbers its records in
            // order. InvalidSequenceNumber when the event wasn't observed.
            uint64_t loadSequenceNumber = InvalidSequenceNumber;
            uint64_t saveSequenceNumber = InvalidSequenceNumber;
            uint8_t vmid = 0;
            uint16_t rmSequenceNumber = 0;
            uint32_t rmCid = 0;
            uint64_t pid = 0; // Host PID from RM/FECS (not the in-container PID); matched against ProcessData::globalPid. Recyclable.

            // If multiple events are dropped during the window, we guess a process from the previous or the next event. Reason: HW trace can have better chances to find the corresponding kernel execution (best effort algorithm).
            // However, to prevent misleading users, the context's label will be shown as "??" in the trace to indicate the context is not fully captured.
            bool isMultipleEventsDropped = false;
            // The load or save event was dropped, and its timestamp is a guess next to the neighboring event.
            // Load and save events outside the observation window are CtxMinimalTimestampGpuNsec and CtxMaximalTimestampGpuNsec instead.
            bool isLoadFabricated = false;
            bool isSaveFabricated = false;
            // The corresponding process data, whose globalPid (or namespacePid as a fallback) and its activities' timestamps match the context's pid and timespan.
            // We need this to prevent the confusion when OS recycle the pid and reuse it for a new process.
            NVTRACE_LUID processLuid = NVTRACE_NIL_LUID;

            ResidentContext() = default;
            ResidentContext(uint64_t ctxLoadTimestampGpuNsecIn, uint64_t ctxSaveTimestampGpuNsecIn, const NvTrace_GPU_GrCtxTrace_ContextChange& context, NVTRACE_LUID processLuidIn, bool isMultipleEventsDroppedIn, uint64_t loadSequenceNumberIn = InvalidSequenceNumber, uint64_t saveSequenceNumberIn = InvalidSequenceNumber)
            {
                ctxLoadTimestampGpuNsec = ctxLoadTimestampGpuNsecIn;
                ctxSaveTimestampGpuNsec = ctxSaveTimestampGpuNsecIn;
                loadSequenceNumber = loadSequenceNumberIn;
                saveSequenceNumber = saveSequenceNumberIn;
                vmid = context.vmid;
                rmSequenceNumber = context.rmSequenceNumber;
                rmCid = context.rmCid;
                pid = context.pid;
                processLuid = processLuidIn;
                isMultipleEventsDropped = isMultipleEventsDroppedIn;
            }
        };

        // organization phase: organize the grCtx trace events
        std::vector<ResidentContext> residentContexts;
        std::vector<ProviderDiagnostic> providerDiagnostics;
    };

    struct CudaApiTraceData : ProviderData
    {
        // Key into ProcessData::threadEventProviders
        NVTRACE_LUID threadEventProviderLuid = NVTRACE_NIL_LUID;

        struct ApiCall
        {
            uint64_t startTimestampTsc = InvalidTimestamp;
            uint64_t endTimestampTsc = InvalidTimestamp;

            uint64_t startRecordLocation = InvalidRecordLocation;
            uint64_t endRecordLocation = InvalidRecordLocation;
            // Currently used for {ImmediateKernelLaunch,GraphLaunch,Memcpy,Memset} events; additional internal events will be defined in the future.
            std::vector<uint64_t> innerRecordLocations;
        };

        struct RuntimeApiCall : ApiCall
        {
            FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId runtimeApiFunctionId = FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId_INVALID;
        };

        struct DriverApiCall : ApiCall
        {
            FB::Provider::CudaApiTrace::CudaDriverApiFunctionId driverApiFunctionId = FB::Provider::CudaApiTrace::CudaDriverApiFunctionId_INVALID;

            // Map of NvtxTraceData domainCliqueIndex -> active range index in DomainCliqueThreadData::pushPopRanges
            std::map<uint32_t, uint64_t> nvtxTraceDomainCliquePushPopRanges;
        };

        struct DriverApiCallRef
        {
            uint64_t uniqueThreadId = 0;
            // Index into ThreadData::driverApiCalls
            size_t driverApiCallIdx = SIZE_MAX;

            bool operator==(const DriverApiCallRef& other) const
            {
                return uniqueThreadId == other.uniqueThreadId && driverApiCallIdx == other.driverApiCallIdx;
            }
        };

        struct ApiCallRef : DriverApiCallRef
        {
            // Index into ThreadData::runtimeApiCalls
            size_t runtimeApiCallIdx = SIZE_MAX;

            bool operator==(const ApiCallRef& other) const
            {
                return uniqueThreadId == other.uniqueThreadId && driverApiCallIdx == other.driverApiCallIdx && runtimeApiCallIdx == other.runtimeApiCallIdx;
            }
        };

        enum class GraphLaunchSource
        {
            Unknown,
            Host,
            Device,
        };

        struct GraphLaunch
        {
            struct HostData
            {
                uint64_t apiCorrelationId = 0;
                DriverApiCallRef driverApiCallRef;
                // From the GridNullEnd record that matched this launch on the GPU
                uint64_t timestampGpuNsec = InvalidTimestamp;
                uint32_t cudaDeviceOrdinal = UINT32_MAX;
                uint64_t custreamId = 0;
            };

            struct DeviceData
            {
                uint64_t timestampGpuNsec = InvalidTimestamp;
                uint32_t cudaDeviceOrdinal = UINT32_MAX;
            };

            GraphLaunchSource source = GraphLaunchSource::Unknown;
            uint32_t hwCorrelationId = 0;
            // Key into CudaApiTraceData::graphExecs
            uint64_t cugraphExecId = 0;
            union
            {
                HostData host;
                DeviceData device;
            } payload = {};
        };

        struct ImmediateKernelLaunch
        {
            uint64_t apiCorrelationId = 0;
            uint64_t startTimestampGpuNsec = InvalidTimestamp;
            uint64_t endTimestampGpuNsec = InvalidTimestamp;

            uint64_t recordLocation = InvalidRecordLocation;

            // Used to find the CPU NVTX stacks that should be projected onto this event
            DriverApiCallRef driverApiCallRef;

            uint32_t GetHwCorrelationId() const
            {
                return (static_cast<uint32_t>(apiCorrelationId) & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK) | NVTRACE_CUDA_HW_CORRELATION_ID_IMMEDIATE_MSBS;
            };
        };

        struct GraphNodeLaunch
        {
            uint32_t hwCorrelationId = 0;
            uint64_t startTimestampGpuNsec = InvalidTimestamp;
            uint64_t endTimestampGpuNsec = InvalidTimestamp;
            // Index into CudaApiTraceData::graphLaunches
            size_t graphLaunchIdx = SIZE_MAX;
            // Key into CudaApiTraceData::graphNodes
            uint64_t graphNodeCorrelationId = UINT64_MAX;
        };

        enum class GpuCommandType
        {
            Invalid,
            EagerMemcpy,
            EagerMemset,
            GraphMemcpy,
            GraphMemset,
            MemOp, // applies to eager and graph operations
            HostCallback,
        };

        struct GpuTimestampReport
        {
            uint64_t gpuTimestampRef = 0;
            uint64_t ptimer = InvalidTimestamp;
        };

        // Timings for non-kernel GPU commands.
        struct GpuCommand
        {
            GpuCommandType commandType = GpuCommandType::Invalid;
            uint64_t recordLocation = InvalidRecordLocation;
            uint64_t startTimestampGpuNsec = InvalidTimestamp;
            uint64_t endTimestampGpuNsec = InvalidTimestamp;
            // Key into CudaApiTraceData::graphNodes for commands that execute as graph nodes
            uint64_t graphNodeCorrelationId = UINT64_MAX;
            // Used to find the CPU NVTX stacks that should be projected onto this event
            DriverApiCallRef driverApiCallRef;
            // HostCallback only: index into CudaApiTraceData::hostCallbackInstances (SIZE_MAX otherwise).
            size_t hostCallbackInstanceIndex = SIZE_MAX;
        };

        enum class HostCallbackKind
        {
            Eager,
            Graph,
        };

        // Host-side callback instance represents combined Launch, ExecBegin, and ExecEnd.
        struct HostCallbackInstance
        {
            HostCallbackKind kind = HostCallbackKind::Eager;

            uint64_t startTimestampTsc = InvalidTimestamp;
            uint64_t endTimestampTsc = InvalidTimestamp;

            uint64_t custreamId = 0;
            union
            {
                uint64_t launchApiCorrelationId = 0; // Eager
                uint64_t graphNodeCorrelationId;     // Graph
            };
            uint64_t enclosingCorrelationId = 0;

            uint64_t callback = 0;
            uint64_t userData = 0;

            uint64_t executingUniqueThreadId = 0;
            // Launching driver API call (from Launch's LinkCudaApiTraceToEnclosingApiCalls).
            DriverApiCallRef driverApiCallRef;
        };

        struct ResourceData
        {
            uint64_t creationTimestampTsc = InvalidTimestamp;
            uint64_t destructionTimestampTsc = InvalidTimestamp;
        };

        struct ThreadData
        {
            ThreadEventProviderData::ThreadData* pThreadData = nullptr;
            std::vector<RuntimeApiCall> runtimeApiCalls;
            std::vector<DriverApiCall> driverApiCalls;
        };

        struct CuStreamData : ResourceData
        {
            uint64_t custreamId = 0;
            uint64_t custreamHandle = 0;
            // Index into CudaApiTraceData::devices
            uint32_t cudaDeviceOrdinal = UINT32_MAX;
            std::vector<ApiCallRef> apiCallReferences;
        };

        struct CuFunctionData : ResourceData
        {
            uint64_t cufunctionId = 0;
            uint64_t cufunctionHandle = 0;
            // Key into CudaApiTraceData::modules
            uint64_t cumoduleId = 0;
            std::string mangledName;
            // Empty means not attempted; one NUL byte means demangling failed.
            mutable std::string demangledName;
        };

        struct CuModuleData : ResourceData
        {
            uint64_t contextId = 0;
            uint64_t cumoduleId = 0;
            uint64_t cumoduleHandle = 0;
            // Map of mangledName --> cufunctionId. Functions with an empty (cut-off) mangledName are
            // omitted here to avoid colliding on the "" key -- functionCount below is the true total.
            std::map<std::string, uint64_t> cufunctionIds;
            uint64_t functionCount = 0;
        };

        struct CuGraphExecData : ResourceData
        {
            uint64_t cugraphExecId = 0;
            uint64_t cugraphExecHandle = 0;
        };

        struct CuGraphNodeData : ResourceData
        {
            uint64_t cugraphNodeId = 0;
            uint64_t cugraphNodeHandle = 0;
            // Key into CudaApiTraceData::graphExecs
            uint64_t cugraphExecId = 0;
            uint64_t graphNodeCorrelationId = 0;
            uint64_t recordLocation = InvalidRecordLocation; // to the NvTrace_CUDA_ApiTrace_GraphNodeCreated record
            // Thread that recorded this graph node
            uint64_t uniqueThreadId = 0;
            // Map from NvtxTraceData domainCliqueIndex to index into DomainCliqueThreadData::pushPopRanges
            std::map<uint32_t, size_t> nvtxTraceDomainCliquePushPopRanges;

            uint32_t GetHwCorrelationId() const
            {
                return (static_cast<uint32_t>(graphNodeCorrelationId) & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK) | NVTRACE_CUDA_HW_CORRELATION_ID_GRAPHNODE_MSBS;
            };
        };

        struct CuContextData : ResourceData
        {
            uint64_t cucontextId = 0;
            uint64_t cucontextHandle = 0;
            // Keys into CudaApiTraceData::streams
            std::vector<uint64_t> custreamIds;
            // RM CIDs of this context's channels (from ChannelCreated events). A FECS GrCtx event reports
            // whichever compute channel was switched, so any of these CIDs identifies this context --
            // namespace-independent GR-context-to-process correlation without PIDs. Empty if no channel was
            // reported (correlation then falls back to PID).
            std::vector<uint32_t> channelRmCids;
        };

        struct CuDeviceData
        {
            uint32_t cudaDeviceOrdinal = 0;
            uint64_t cudeviceHandle = 0;
            NVTRACE_UUID gpuUuid = NVTRACE_NIL_UUID;

            // Sorted by startTimestampGpuNsec after building
            std::vector<ImmediateKernelLaunch> immediateKernelLaunches;
            // Map hwCorrelationId -> GPU start and end events of immediate kernels whose launch record is not in this
            // trace, as when the launch record is in an earlier snapshot. Only kept with preserveOutOfWindowTimestamps.
            std::map<uint32_t, std::vector<GpuHwTraceData::KernelExecution>> unmatchedImmediateKernelExecutions;
            // Sorted by startTimestampGpuNsec after building; each GraphNodeLaunch corresponds to a kernel launch on the GPU
            std::vector<GraphNodeLaunch> graphNodeLaunches;
            // GPU end events of graph node launches whose start is not in this trace, as when the start is in an
            // earlier snapshot. Only the end and the resolved graph node are known. Only kept with
            // preserveOutOfWindowTimestamps.
            std::vector<GraphNodeLaunch> unmatchedGraphNodeLaunchEnds;
            // Map hwCorrelationId -> GridNullEnd timestamps of host graph launches whose launch record is not in this
            // trace, as when the launch record is in an earlier snapshot. Only kept with preserveOutOfWindowTimestamps.
            std::map<uint32_t, std::vector<uint64_t>> unmatchedHostGraphLaunchTimestampsGpuNsec;

            // Sorted by { 32-bit hwCorrelationId, creationTimestampTsc } where hwCorrelationId = graphNodeCorrelationId & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK.
            std::vector<CuGraphNodeData*> graphNodes;

            // Sorted by startTimestampGpuNsec after building
            std::vector<GpuCommand> gpuCommands;
        };

        /// Always the published minor versions, never sizeof()/offsetof(), which describe this build rather than
        /// the producer that wrote the trace.
        NvTrace_CUDA_ApiTrace_RecordVersionInfo recordVersionInfo = {};

        uint64_t configurationRecordLocation = InvalidRecordLocation;

        /// Vector of CUDA devices, indexed by cudaDeviceOrdinal
        std::vector<CuDeviceData> devices;
        /// Map uniqueThreadId -> Thread
        std::map<uint64_t, ThreadData> threads;
        /// Map cucontextId -> CuContext
        std::map<uint64_t, CuContextData> contexts;
        /// Map custreamId -> CuStream
        std::map<uint64_t, CuStreamData> streams;
        /// Map cumoduleId -> CuModule
        std::map<uint64_t, CuModuleData> modules;
        /// Map cufunctionId -> CuFunction
        std::map<uint64_t, CuFunctionData> functions;
        /// Map cugraphExecId -> CuGraphExec
        std::map<uint64_t, CuGraphExecData> graphExecs;
        /// Map graphNodeCorrelationId -> CuGraphNodeData
        std::map<uint64_t, CuGraphNodeData> graphNodes;
        /// Graph launch instances referenced by GraphNodeLaunch::graphLaunchIdx
        std::vector<GraphLaunch> graphLaunches;

        /// Host-side callback instances (eager and graph), unordered.
        std::vector<HostCallbackInstance> hostCallbackInstances;

        /// GPU timestamp reports that no GPU command in this trace references, as when the command's begin record is
        /// in an earlier snapshot. Only populated when preserveOutOfWindowTimestamps is true.
        std::vector<GpuTimestampReport> unmatchedGpuTimestampReports;
    };

    struct NvtxTraceData : ProviderData
    {
        // Key into ProcessData::threadEventProviders
        NVTRACE_LUID threadEventProviderLuid = NVTRACE_NIL_LUID;
        NvTrace_NvtxTrace_RecordVersionInfo recordVersionInfo = {};
        uint32_t recordVersionInfoMinorVersion = 0;

        struct StringTableEntryData
        {
            uint64_t stringRegisteredRecordLocation = InvalidRecordLocation;
            std::map<uint32_t /*stringOffset*/, uint64_t /*recordLocation*/> segmentRecordLocations;
            mutable std::string cachedUtf8Value;
        };

        struct DomainData
        {
            uint64_t domainCreatedRecordLocation = InvalidRecordLocation;
        };

        enum FieldMask : uint8_t
        {
            FieldMask_None = 0,
            FieldMask_StartTimestamp = 1 << 0,
            FieldMask_EndTimestamp = 1 << 1,
        };

        // Set in the PushPopRangeData::rangeRef of ranges the builder synthesized; wire refs never carry it.
        static constexpr uint64_t SyntheticPushPopRangeRefBit = uint64_t(1) << 63;

        struct PushPopRangeData
        {
            static constexpr size_t InvalidPushPopRangeIndex = ~size_t(0);

            // Process-unique identity: the wire PushPopRange ref, or a synthesized ref with SyntheticPushPopRangeRefBit set
            uint64_t rangeRef = 0;
            uint64_t startTimestampTsc = InvalidTimestamp;
            uint64_t endTimestampTsc = InvalidTimestamp;
            uint64_t uniqueThreadId = 0;
            uint32_t domainIndex = 0;
            // Depth in the merged domain clique, used for projected stacks.
            uint16_t nestingDepth = 0;
            // Stable source depth within the range's own NVTX domain.
            uint16_t sourceNestingDepth = 0;
            uint16_t messageType = 0;
            // Stable source parent identity; 0 denotes a root range.
            uint64_t parentRangeRef = 0;
            // Index into DomainCliqueThreadData::pushPopRanges
            size_t parentRangeIndex = InvalidPushPopRangeIndex;
            uint8_t approximationMask = FieldMask_None;
            bool isSynthetic = false;
            // Retained for a graph node's record-time NVTX stack.
            bool isRecordTimeRange = false;
            // Eligible for CPU-row export and launch-time association after observation-window culling.
            bool isObservationTimeRange = false;
            uint64_t stateRecordLocation = InvalidRecordLocation;

            /// The message in the encoding messageType names, assembled at build time; a
            /// RegisteredString message carries its 8-byte handle here instead. A byte no record
            /// filled is 0, so a message the trace tiled only partly shows its holes.
            std::string message;
            mutable std::string cachedUtf8Message;
        };

        struct MarkData
        {
            uint64_t timestampTsc = InvalidTimestamp;
            uint64_t uniqueThreadId = 0;
            uint32_t domainIndex = 0;
            uint16_t messageType = 0;
            // The wire Mark ref
            uint64_t correlationId = 0;
            uint64_t recordLocation = InvalidRecordLocation;

            /// Assembled exactly as a push-pop range's message is.
            std::string message;
            mutable std::string cachedUtf8Message;
        };

        struct DroppedRangePushData
        {
            uint64_t timestampTsc = InvalidTimestamp;
            uint64_t uniqueThreadId = 0;
            uint32_t domainIndex = 0;
        };

        struct DroppedRangePopData
        {
            uint64_t timestampTsc = InvalidTimestamp;
            uint64_t uniqueThreadId = 0;
            uint32_t domainIndex = 0;
        };

        struct DomainCliqueThreadData
        {
            std::vector<MarkData> marks;
            std::vector<DroppedRangePushData> droppedRangePushes;
            std::vector<DroppedRangePopData> droppedRangePops;
            std::vector<PushPopRangeData> pushPopRanges;
            // True if any remaining range is in the observation window (record-time-only ranges may still be present).
            bool hasObservationTimeRange = false;

            template <typename TOnPushPopRange>
            void ForEachObservedPushPopRange(TOnPushPopRange&& onPushPopRange) const
            {
                for (const PushPopRangeData& pushPopRange : pushPopRanges)
                {
                    if (pushPopRange.isObservationTimeRange)
                    {
                        onPushPopRange(pushPopRange);
                    }
                }
            }
        };

        struct DomainCliqueData
        {
            // Keys into NvtxTraceData::domains
            std::vector<uint32_t> domainIndices;
            // Map of uniqueThreadId -> events for this domain clique
            std::map<uint64_t, DomainCliqueThreadData> threads;
        };

        std::map<uint64_t, StringTableEntryData> stringTableEntries;
        std::map<uint32_t, DomainData> domains;
        std::vector<DomainCliqueData> domainCliques;
    };

    struct SystemInfo
    {
        std::string hostname;
        uint64_t serviceStartTimestampTsc = 0;
        std::string osVersion;
        FB::OsFamily osFamily;
        std::string cpuModel;
        FB::CpuArch cpuArch;
        uint64_t sysMemTotal;
        NvTrace_SemVer nvidiaRmKernelDriverVersion;
        NvTrace_SemVer nvidiaMrmKernelDriverVersion;
    };

    struct GpuPartitionInfo
    {
        std::string chipName;
        std::string deviceName;
        NvTrace_SemVer driverVersion;
        FB::DriverType driverType;
        std::vector<NVTRACE_UUID> supportedProviders;
        std::vector<NvTrace_CudaAttribute> attributes;
        uint64_t deviceMemorySize;
        bool isProfiled;
    };

    struct GpuPartitionData
    {
        NVTRACE_LUID gpuPartitionLuid;
        NVTRACE_UUID gpuUuid;
        uint64_t creationTimestampTsc;
        uint64_t destructionTimestampTsc = MaxTimestampTsc;
        uint64_t observationStartGpuNsec = GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec;
        uint64_t observationEndGpuNsec = GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec;

        GpuPartitionInfo info;

        GpuClockCalibrationData gpuClockCalibration;
        GpuPmCountersData gpuPmCounters;
        GpuHwTraceData gpuHwTrace;
        GpuGrCtxTraceData gpuGrCtxTrace;
    };

    struct ProcessData
    {
        NVTRACE_LUID processLuid;
        int namespacePid;
        int namespaceParentPid;
        int globalPid = 0;                            // Host PID from the Service (the PID FECS reports). 0 => correlation falls back to namespacePid.
        NvTrace_LinuxNamespaceId pidNamespaceId = {}; // (device, inode) of /proc/<pid>/ns/pid. Both 0 if unavailable. Display only -- disambiguates equal namespace PIDs across containers.
        std::string path;
        uint64_t creationTimestampTsc = MinTimestampTsc;
        uint64_t destructionTimestampTsc = MaxTimestampTsc;
        // TODO add any other Process properties or metadata we want

        CudaApiTraceData cudaApiTrace;
        NvtxTraceData nvtxTrace;
        std::map<NVTRACE_LUID, ThreadEventProviderData> threadEventProviders;
    };

    struct SystemData
    {
        SystemInfo systemInfo;
        CpuClockCalibrationData cpuClockCalibration;
        std::map<NVTRACE_LUID, GpuPartitionData> gpuPartitions;
        std::map<NVTRACE_LUID, ProcessData> processes;
    };

    enum DeviceEventType
    {
        DeviceEventType_INVALID,
        DeviceEventType_ImmediateKernelLaunch,
        DeviceEventType_GraphNodeLaunch,
        DeviceEventType_GpuCommand,
        DeviceEventType_Count
    };

    struct DeviceEventOrigin
    {
        NVTRACE_LUID processLuid = NVTRACE_NIL_LUID;

        const ProcessData* pProcessData = nullptr;
        const CudaApiTraceData* pCudaApiTrace = nullptr;
        const CudaApiTraceData::CuDeviceData* pCuDevice = nullptr;

        explicit operator bool() const
        {
            return processLuid != NVTRACE_NIL_LUID && pProcessData != nullptr && pCudaApiTrace != nullptr && pCuDevice != nullptr;
        }
    };

    struct DeviceEvent
    {
        DeviceEventType eventType;
        union
        {
            const CudaApiTraceData::ImmediateKernelLaunch* pImmediateKernelLaunch;
            const CudaApiTraceData::GraphNodeLaunch* pGraphNodeLaunch;
            const CudaApiTraceData::GpuCommand* pGpuCommand;
        };
        uint64_t startTimestampGpuNsec = InvalidTimestamp;
        uint64_t endTimestampGpuNsec = InvalidTimestamp;

        DeviceEvent()
            : eventType(DeviceEventType_INVALID)
            , pImmediateKernelLaunch(nullptr)
        {
        }

        explicit DeviceEvent(const CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch)
            : eventType(DeviceEventType_ImmediateKernelLaunch)
            , pImmediateKernelLaunch(&immediateKernelLaunch)
            , startTimestampGpuNsec(immediateKernelLaunch.startTimestampGpuNsec)
            , endTimestampGpuNsec(immediateKernelLaunch.endTimestampGpuNsec)
        {
        }

        explicit DeviceEvent(const CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch)
            : eventType(DeviceEventType_GraphNodeLaunch)
            , pGraphNodeLaunch(&graphNodeLaunch)
            , startTimestampGpuNsec(graphNodeLaunch.startTimestampGpuNsec)
            , endTimestampGpuNsec(graphNodeLaunch.endTimestampGpuNsec)
        {
        }

        explicit DeviceEvent(const CudaApiTraceData::GpuCommand& gpuCommand)
            : eventType(DeviceEventType_GpuCommand)
            , pGpuCommand(&gpuCommand)
            , startTimestampGpuNsec(gpuCommand.startTimestampGpuNsec)
            , endTimestampGpuNsec(gpuCommand.endTimestampGpuNsec)
        {
        }

        explicit operator bool() const
        {
            return eventType != DeviceEventType_INVALID;
        }

        const void* GetEventPointer() const
        {
            switch (eventType)
            {
                case DeviceEventType_ImmediateKernelLaunch:
                    return pImmediateKernelLaunch;
                case DeviceEventType_GraphNodeLaunch:
                    return pGraphNodeLaunch;
                case DeviceEventType_GpuCommand:
                    return pGpuCommand;
                default:
                    return nullptr;
            }
        }

        bool operator>(const DeviceEvent& rhs) const
        {
            if (startTimestampGpuNsec != rhs.startTimestampGpuNsec)
            {
                return startTimestampGpuNsec > rhs.startTimestampGpuNsec;
            }
            return eventType > rhs.eventType;
        }
    };

    struct DeviceEventStream
    {
        DeviceEventType eventType;
        union
        {
            const std::vector<CudaApiTraceData::ImmediateKernelLaunch>* pImmediateKernelLaunches;
            const std::vector<CudaApiTraceData::GraphNodeLaunch>* pGraphNodeLaunches;
            const std::vector<CudaApiTraceData::GpuCommand>* pGpuCommands;
        };
        size_t nextEventIdx = 0;

        explicit DeviceEventStream(const std::vector<CudaApiTraceData::ImmediateKernelLaunch>& immediateKernelLaunches)
            : eventType(DeviceEventType_ImmediateKernelLaunch)
            , pImmediateKernelLaunches(&immediateKernelLaunches)
        {
        }

        explicit DeviceEventStream(const std::vector<CudaApiTraceData::GraphNodeLaunch>& graphNodeLaunches)
            : eventType(DeviceEventType_GraphNodeLaunch)
            , pGraphNodeLaunches(&graphNodeLaunches)
        {
        }

        explicit DeviceEventStream(const std::vector<CudaApiTraceData::GpuCommand>& gpuCommands)
            : eventType(DeviceEventType_GpuCommand)
            , pGpuCommands(&gpuCommands)
        {
        }

        DeviceEvent GetNextEvent()
        {
            switch (eventType)
            {
                case DeviceEventType_ImmediateKernelLaunch:
                    if (nextEventIdx < pImmediateKernelLaunches->size())
                    {
                        return DeviceEvent((*pImmediateKernelLaunches)[nextEventIdx++]);
                    }
                    break;
                case DeviceEventType_GraphNodeLaunch:
                    if (nextEventIdx < pGraphNodeLaunches->size())
                    {
                        return DeviceEvent((*pGraphNodeLaunches)[nextEventIdx++]);
                    }
                    break;
                case DeviceEventType_GpuCommand:
                    if (nextEventIdx < pGpuCommands->size())
                    {
                        return DeviceEvent((*pGpuCommands)[nextEventIdx++]);
                    }
                    break;
                default:
                    break;
            }

            eventType = DeviceEventType_INVALID;
            return DeviceEvent();
        }
    };

    using DeviceEventStreams = std::array<DeviceEventStream, DeviceEventType_Count - 1>;

    inline DeviceEventStreams CreateDeviceEventStreams(const CudaApiTraceData::CuDeviceData& cuDevice)
    {
        static_assert(
            DeviceEventType_ImmediateKernelLaunch == 1
                && DeviceEventType_GraphNodeLaunch == 2
                && DeviceEventType_GpuCommand == 3,
            "DeviceEventType values must match the DeviceEventStreams array order");

        return DeviceEventStreams{
            {
             DeviceEventStream(cuDevice.immediateKernelLaunches),
             DeviceEventStream(cuDevice.graphNodeLaunches),
             DeviceEventStream(cuDevice.gpuCommands),
             }
        };
    }

    struct NvtxDeviceProjectionEvent
    {
        const NvtxTraceData::PushPopRangeData* pPushPopRange;
        uint64_t startTimestampGpuNsec;
        uint64_t endTimestampGpuNsec;
        // The depth of the NVTX stack for the first GPU event this range covers
        uint16_t initialStackSize;
        // One-based depth within the combined launch-time and record-time stack
        uint16_t stackDepth;
    };

    enum NvtxAssociationSource : uint8_t
    {
        NvtxAssociationSource_None = 0,
        // On the stack of the thread that launched the work
        NvtxAssociationSource_LaunchTime = 1 << 0,
        // On the stack of the thread that recorded the graph node
        NvtxAssociationSource_RecordTime = 1 << 1,
    };

    struct NvtxAssociationStackEntry
    {
        const NvtxTraceData::PushPopRangeData* pPushPopRange;
        // NvtxAssociationSource bits
        uint8_t associationSources;
    };

    template <typename TTraceReader>
    struct DataModel : public SystemData
    {
        using RecordMetadata = typename TTraceReader::RecordMetadata;

        TTraceReader* m_pTraceReader = nullptr;

        uint64_t m_observationStartTsc = MinTimestampTsc;
        uint64_t m_observationEndTsc = MaxTimestampTsc;
        uint64_t m_observationStartUtcNsec = InvalidTimestamp;
        uint64_t m_observationEndUtcNsec = InvalidTimestamp;
        uint64_t m_timestampStartTsc = MaxTimestampTsc;
        uint64_t m_timestampEndTsc = MinTimestampTsc;

        void Clear()
        {
            processes.clear();
            gpuPartitions.clear();
            cpuClockCalibration.calibrationPoints.clear();
            cpuClockCalibration.providerLuid = NVTRACE_NIL_LUID;

            m_pTraceReader = nullptr;
            m_observationStartTsc = MinTimestampTsc;
            m_observationEndTsc = MaxTimestampTsc;
            m_observationStartUtcNsec = InvalidTimestamp;
            m_observationEndUtcNsec = InvalidTimestamp;
            m_timestampStartTsc = MaxTimestampTsc;
            m_timestampEndTsc = MinTimestampTsc;
        }

        //////////////////////////////////////////////////
        // System
        //////////////////////////////////////////////////

        GpuPartitionData* FindGpuPartition(NVTRACE_LUID gpuPartitionLuid)
        {
            auto itr = gpuPartitions.find(gpuPartitionLuid);
            return itr != gpuPartitions.end() ? &itr->second : nullptr;
        }
        const GpuPartitionData* FindGpuPartition(NVTRACE_LUID gpuPartitionLuid) const
        {
            return const_cast<DataModel*>(this)->FindGpuPartition(gpuPartitionLuid);
        }

        GpuPartitionData* FindGpuPartitionByGpuUuid(NVTRACE_UUID gpuUuid)
        {
            for (auto& [_, gpuPartitionData] : gpuPartitions)
            {
                if (gpuPartitionData.gpuUuid == gpuUuid)
                {
                    return &gpuPartitionData;
                }
            }

            return nullptr;
        }
        const GpuPartitionData* FindGpuPartitionByGpuUuid(NVTRACE_UUID gpuUuid) const
        {
            return const_cast<DataModel*>(this)->FindGpuPartitionByGpuUuid(gpuUuid);
        }

        ProcessData* FindProcess(NVTRACE_LUID processLuid)
        {
            auto itr = processes.find(processLuid);
            return itr != processes.end() ? &itr->second : nullptr;
        }
        const ProcessData* FindProcess(NVTRACE_LUID processLuid) const
        {
            return const_cast<DataModel*>(this)->FindProcess(processLuid);
        }

        //////////////////////////////////////////////////
        // Observation Window
        //////////////////////////////////////////////////

        void RecordObservedTimestampTsc(uint64_t timestampTsc)
        {
            if (timestampTsc == InvalidTimestamp || timestampTsc == MaxTimestampTsc)
            {
                return;
            }

            m_timestampStartTsc = std::min(m_timestampStartTsc, timestampTsc);
            m_timestampEndTsc = std::max(m_timestampEndTsc, timestampTsc);
        }

        void RecordObservedTimestampRangeTsc(uint64_t startTimestampTsc, uint64_t endTimestampTsc)
        {
            RecordObservedTimestampTsc(startTimestampTsc);
            RecordObservedTimestampTsc(endTimestampTsc);
        }

        bool CheckCpuTimestamp(uint64_t timestampTsc, bool updateObservedTimestamps, uint64_t& out_timestampTsc)
        {
            if (timestampTsc == InvalidTimestamp || timestampTsc < m_observationStartTsc || timestampTsc > m_observationEndTsc)
            {
                return false;
            }

            out_timestampTsc = timestampTsc;
            if (updateObservedTimestamps)
            {
                RecordObservedTimestampTsc(out_timestampTsc);
            }
            return true;
        }

        bool CheckCpuTimestamp(uint64_t timestampTsc, bool updateObservedTimestamps)
        {
            uint64_t out_timestampTsc = InvalidTimestamp;
            return CheckCpuTimestamp(timestampTsc, updateObservedTimestamps, out_timestampTsc);
        }

        bool CheckCpuTimestampRange(uint64_t startTimestampTsc, uint64_t endTimestampTsc, bool updateObservedTimestamps, uint64_t& out_startTimestampTsc, uint64_t& out_endTimestampTsc)
        {
            if (startTimestampTsc == InvalidTimestamp || endTimestampTsc == InvalidTimestamp || endTimestampTsc < startTimestampTsc)
            {
                return false;
            }
            if (endTimestampTsc < m_observationStartTsc || startTimestampTsc > m_observationEndTsc)
            {
                return false;
            }

            out_startTimestampTsc = std::max(startTimestampTsc, m_observationStartTsc);
            out_endTimestampTsc = std::min(endTimestampTsc, m_observationEndTsc);
            if (updateObservedTimestamps)
            {
                RecordObservedTimestampRangeTsc(out_startTimestampTsc, out_endTimestampTsc);
            }
            return true;
        }

        bool CheckProcessTimestamp(
            const ProcessData& processData,
            uint64_t timestampTsc,
            bool updateObservedTimestamps,
            uint64_t& out_timestampTsc)
        {
            return timestampTsc >= processData.creationTimestampTsc && timestampTsc <= processData.destructionTimestampTsc && CheckCpuTimestamp(timestampTsc, updateObservedTimestamps, out_timestampTsc);
        }

        bool CheckProcessTimestampRange(
            const ProcessData& processData,
            uint64_t startTimestampTsc,
            uint64_t endTimestampTsc,
            bool updateObservedTimestamps,
            uint64_t& out_startTimestampTsc,
            uint64_t& out_endTimestampTsc)
        {
            return CheckCpuTimestampRange(
                std::max(startTimestampTsc, processData.creationTimestampTsc),
                std::min(endTimestampTsc, processData.destructionTimestampTsc),
                updateObservedTimestamps,
                out_startTimestampTsc,
                out_endTimestampTsc);
        }

        bool CheckThreadTimestamp(
            const ThreadEventProviderData::ThreadData& threadData,
            uint64_t timestampTsc,
            bool updateObservedTimestamps,
            uint64_t& out_timestampTsc)
        {
            return timestampTsc >= threadData.creationTimestampTsc && timestampTsc <= threadData.destructionTimestampTsc && CheckCpuTimestamp(timestampTsc, updateObservedTimestamps, out_timestampTsc);
        }

        bool CheckThreadTimestampRange(
            const ThreadEventProviderData::ThreadData& threadData,
            uint64_t startTimestampTsc,
            uint64_t endTimestampTsc,
            bool updateObservedTimestamps,
            uint64_t& out_startTimestampTsc,
            uint64_t& out_endTimestampTsc)
        {
            return CheckCpuTimestampRange(
                std::max(startTimestampTsc, threadData.creationTimestampTsc),
                std::min(endTimestampTsc, threadData.destructionTimestampTsc),
                updateObservedTimestamps,
                out_startTimestampTsc,
                out_endTimestampTsc);
        }

        bool CheckCpuTimestampRange(uint64_t startTimestampTsc, uint64_t endTimestampTsc, bool updateObservedTimestamps)
        {
            uint64_t out_startTimestampTsc = InvalidTimestamp;
            uint64_t out_endTimestampTsc = InvalidTimestamp;
            return CheckCpuTimestampRange(startTimestampTsc, endTimestampTsc, updateObservedTimestamps, out_startTimestampTsc, out_endTimestampTsc);
        }

        bool CheckGpuTimestamp(const GpuPartitionData& gpuPartitionData, uint64_t timestampGpuNsec, bool updateObservedTimestamps, uint64_t& out_timestampGpuNsec)
        {
            if (timestampGpuNsec == InvalidTimestamp || timestampGpuNsec < gpuPartitionData.observationStartGpuNsec || timestampGpuNsec > gpuPartitionData.observationEndGpuNsec)
            {
                return false;
            }

            out_timestampGpuNsec = timestampGpuNsec;
            if (updateObservedTimestamps)
            {
                RecordObservedTimestampTsc(ConvertGpuNsecToTscTick(gpuPartitionData, out_timestampGpuNsec));
            }
            return true;
        }

        bool CheckGpuTimestampRange(const GpuPartitionData& gpuPartitionData, uint64_t startTimestampGpuNsec, uint64_t endTimestampGpuNsec, bool updateObservedTimestamps, uint64_t& out_startTimestampGpuNsec, uint64_t& out_endTimestampGpuNsec)
        {
            if (endTimestampGpuNsec == InvalidTimestamp || endTimestampGpuNsec < startTimestampGpuNsec)
            {
                return false;
            }
            if (endTimestampGpuNsec < gpuPartitionData.observationStartGpuNsec || startTimestampGpuNsec > gpuPartitionData.observationEndGpuNsec)
            {
                return false;
            }

            out_startTimestampGpuNsec = std::max(startTimestampGpuNsec, gpuPartitionData.observationStartGpuNsec);
            out_endTimestampGpuNsec = std::min(endTimestampGpuNsec, gpuPartitionData.observationEndGpuNsec);
            if (updateObservedTimestamps)
            {
                const uint64_t startTimestampTsc = out_startTimestampGpuNsec != GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec ? ConvertGpuNsecToTscTick(gpuPartitionData, out_startTimestampGpuNsec) : InvalidTimestamp;
                const uint64_t endTimestampTsc = out_endTimestampGpuNsec != GpuGrCtxTraceData::CtxMaximalTimestampGpuNsec ? ConvertGpuNsecToTscTick(gpuPartitionData, out_endTimestampGpuNsec) : MaxTimestampTsc;
                RecordObservedTimestampRangeTsc(startTimestampTsc, endTimestampTsc);
            }
            return true;
        }

        bool CheckGpuTimestampRange(const GpuPartitionData& gpuPartitionData, uint64_t startTimestampGpuNsec, uint64_t endTimestampGpuNsec, bool updateObservedTimestamps)
        {
            uint64_t out_startTimestampGpuNsec = InvalidTimestamp;
            uint64_t out_endTimestampGpuNsec = InvalidTimestamp;
            return CheckGpuTimestampRange(gpuPartitionData, startTimestampGpuNsec, endTimestampGpuNsec, updateObservedTimestamps, out_startTimestampGpuNsec, out_endTimestampGpuNsec);
        }

        //////////////////////////////////////////////////
        // CPU Clock Calibration
        //////////////////////////////////////////////////

        template <typename TCallback>
        void ForEachCpuClockCalibrationInterval(TCallback&& callback) const
        {
            ForEachClockCalibrationInterval(
                cpuClockCalibration.calibrationPoints,
                [](const TscRtcCalibrationPoint& calibrationPoint) {
                    return calibrationPoint.utcNsec;
                },
                std::forward<TCallback>(callback));
        }

        uint64_t ConvertTscTickToUtcNsec(uint64_t tscTick) const
        {
            if (tscTick == InvalidTimestamp || cpuClockCalibration.calibrationPoints.empty())
            {
                return InvalidTimestamp;
            }

            auto itr = std::lower_bound(cpuClockCalibration.calibrationPoints.begin(), cpuClockCalibration.calibrationPoints.end(), tscTick, [&](const TscRtcCalibrationPoint& calibrationPoint, uint64_t tscTick) {
                return calibrationPoint.tscTick < tscTick;
            });

            return itr != cpuClockCalibration.calibrationPoints.end() ? itr->ToUtc(tscTick) : cpuClockCalibration.calibrationPoints.back().ToUtc(tscTick);
        }

        uint64_t ConvertUtcNsecToTscTick(uint64_t utcNsec) const
        {
            if (utcNsec == InvalidTimestamp || cpuClockCalibration.calibrationPoints.empty())
            {
                return InvalidTimestamp;
            }

            auto itr = std::lower_bound(cpuClockCalibration.calibrationPoints.begin(), cpuClockCalibration.calibrationPoints.end(), utcNsec, [&](const TscRtcCalibrationPoint& calibrationPoint, uint64_t utcNsec) {
                return calibrationPoint.utcNsec < utcNsec;
            });

            return itr != cpuClockCalibration.calibrationPoints.end() ? itr->ToTsc(utcNsec) : cpuClockCalibration.calibrationPoints.back().ToTsc(utcNsec);
        }

        //////////////////////////////////////////////////
        // GPU Clock Calibration
        //////////////////////////////////////////////////

        template <typename TCallback>
        void ForEachGpuClockCalibrationInterval(const GpuClockCalibrationData& gpuClockCalibrationData, TCallback&& callback) const
        {
            ForEachClockCalibrationInterval(
                gpuClockCalibrationData.calibrationPoints,
                [&](const TscGpuCalibrationPoint& calibrationPoint) {
                    return ConvertTscTickToUtcNsec(calibrationPoint.tscTick);
                },
                std::forward<TCallback>(callback));
        }

        static uint64_t ConvertTscTickToGpuNsec(const GpuPartitionData& gpuPartitionData, uint64_t tscTick)
        {
            if (tscTick == InvalidTimestamp || gpuPartitionData.gpuClockCalibration.calibrationPoints.empty())
            {
                return InvalidTimestamp;
            }

            auto itr = std::lower_bound(gpuPartitionData.gpuClockCalibration.calibrationPoints.begin(), gpuPartitionData.gpuClockCalibration.calibrationPoints.end(), tscTick, [&](const TscGpuCalibrationPoint& calibrationPoint, uint64_t tscTick) {
                return calibrationPoint.tscTick < tscTick;
            });

            return itr != gpuPartitionData.gpuClockCalibration.calibrationPoints.end() ? itr->ToGpu(tscTick) : gpuPartitionData.gpuClockCalibration.calibrationPoints.back().ToGpu(tscTick);
        }

        static uint64_t ConvertGpuNsecToTscTick(const GpuPartitionData& gpuPartitionData, uint64_t gpuNsec)
        {
            if (gpuNsec == InvalidTimestamp || gpuPartitionData.gpuClockCalibration.calibrationPoints.empty())
            {
                return InvalidTimestamp;
            }

            auto itr = std::lower_bound(gpuPartitionData.gpuClockCalibration.calibrationPoints.begin(), gpuPartitionData.gpuClockCalibration.calibrationPoints.end(), gpuNsec, [&](const TscGpuCalibrationPoint& calibrationPoint, uint64_t gpuNsec) {
                return calibrationPoint.gpuNsec < gpuNsec;
            });

            return itr != gpuPartitionData.gpuClockCalibration.calibrationPoints.end() ? itr->ToTsc(gpuNsec) : gpuPartitionData.gpuClockCalibration.calibrationPoints.back().ToTsc(gpuNsec);
        }

        uint64_t ConvertGpuNsecToUtcNsec(const GpuPartitionData& gpuPartitionData, uint64_t gpuNsec) const
        {
            return ConvertTscTickToUtcNsec(ConvertGpuNsecToTscTick(gpuPartitionData, gpuNsec));
        }

        uint64_t ConvertUtcNsecToGpuNsec(const GpuPartitionData& gpuPartitionData, uint64_t utcNsec) const
        {
            return ConvertTscTickToGpuNsec(gpuPartitionData, ConvertUtcNsecToTscTick(utcNsec));
        }

        //////////////////////////////////////////////////
        // GPU PM Counters
        //////////////////////////////////////////////////

        /// Creates a CounterData with space for the specified number of samples.
        bool InitializeGpuPmCounterData(const GpuPmCountersData& gpuPmCountersData, std::vector<uint8_t>& counterDataImage, const size_t maxSamples) const
        {
            if (!gpuPmCountersData.counterDataPreambleSize || !gpuPmCountersData.sampleRangeDataSize)
            {
                return false;
            }

            const size_t counterDataPreambleSize = static_cast<size_t>(gpuPmCountersData.counterDataPreambleSize);
            const size_t sampleRangeDataSize = static_cast<size_t>(gpuPmCountersData.sampleRangeDataSize);

            counterDataImage.resize(counterDataPreambleSize + sampleRangeDataSize * maxSamples);
            memcpy(counterDataImage.data(), gpuPmCountersData.counterDataPreamble.data(), counterDataPreambleSize);

            return true;
        }

        GpuPmCountersSample GetGpuPmCountersSample(const GpuPartitionData& gpuPartitionData, uint64_t recordLocation) const
        {
            GpuPmCountersSample gpuPmCountersSample = {};
            const GpuPmCountersData& gpuPmCountersData = gpuPartitionData.gpuPmCounters;
            if (gpuPmCountersData.pmSampleRateHz == 0)
            {
                return gpuPmCountersSample;
            }

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(recordLocation);
            const uint8_t* const pMajorPayload = recordPayloads.pMajorPayload;
            const size_t maxMajorPayloadSize = recordPayloads.maxMajorPayloadSize;
            if (!pMajorPayload || gpuPmCountersData.counterSampleFixedFieldSize < NvTrace_GPU_PmCounters_CounterSample_MINOR_VERSION_A || maxMajorPayloadSize < gpuPmCountersData.counterSampleFixedFieldSize)
            {
                return gpuPmCountersSample;
            }

            const size_t sampleRangeDataOffsetInPayload = AlignUpP2(static_cast<size_t>(gpuPmCountersData.counterSampleFixedFieldSize), alignof(uint64_t));
            if (maxMajorPayloadSize < sampleRangeDataOffsetInPayload || maxMajorPayloadSize - sampleRangeDataOffsetInPayload < gpuPmCountersData.sampleRangeDataSize)
            {
                return gpuPmCountersSample;
            }

            const size_t sampleEvaluatedMetricsOffsetInPayload = AlignUpP2(sampleRangeDataOffsetInPayload + gpuPmCountersData.sampleRangeDataSize, alignof(uint64_t));
            const size_t sampleEvaluatedMetricsSize = static_cast<size_t>(gpuPmCountersData.sampleEvaluatedMetricCount) * sizeof(double);
            if (maxMajorPayloadSize < sampleEvaluatedMetricsOffsetInPayload || maxMajorPayloadSize - sampleEvaluatedMetricsOffsetInPayload < sampleEvaluatedMetricsSize)
            {
                return gpuPmCountersSample;
            }

            gpuPmCountersSample.counterSample = *reinterpret_cast<const NvTrace_GPU_PmCounters_CounterSample*>(pMajorPayload);
            const uint64_t samplePeriodNsec = 1'000'000'000 / gpuPmCountersData.pmSampleRateHz;
            const uint64_t endTimestampGpuNsec = gpuPmCountersSample.counterSample.endTimestampPtimer;
            const uint64_t startTimestampGpuNsec = endTimestampGpuNsec > samplePeriodNsec ? endTimestampGpuNsec - samplePeriodNsec : 1;
            gpuPmCountersSample.startTimestampUtcNsec = ConvertGpuNsecToUtcNsec(gpuPartitionData, startTimestampGpuNsec);
            gpuPmCountersSample.endTimestampUtcNsec = ConvertGpuNsecToUtcNsec(gpuPartitionData, endTimestampGpuNsec);
            gpuPmCountersSample.pSampleRangeData = pMajorPayload + sampleRangeDataOffsetInPayload;
            gpuPmCountersSample.sampleRangeDataSize = gpuPmCountersData.sampleRangeDataSize;
            gpuPmCountersSample.pSampleEvaluatedMetrics = reinterpret_cast<const double*>(pMajorPayload + sampleEvaluatedMetricsOffsetInPayload);
            gpuPmCountersSample.sampleEvaluatedMetricCount = gpuPmCountersData.sampleEvaluatedMetricCount;
            return gpuPmCountersSample;
        }

        // Traverse GPU PM counter samples and invoke the provided callback for each sample.
        // onGpuPmCountersSample: void(const GpuPmCountersSample& gpuPmCountersSample)
        template <typename TOnGpuPmCountersSample>
        void ForEachGpuPmCountersSamples(const GpuPartitionData& gpuPartitionData, bool updateObservedTimestamps, TOnGpuPmCountersSample&& onGpuPmCountersSample)
        {
            const GpuPmCountersData& gpuPmCountersData = gpuPartitionData.gpuPmCounters;
            if (gpuPmCountersData.pmSampleRateHz == 0)
            {
                return;
            }

            for (const uint64_t sampleRecordLocation : gpuPmCountersData.sampleRecordLocations)
            {
                GpuPmCountersSample gpuPmCountersSample = GetGpuPmCountersSample(gpuPartitionData, sampleRecordLocation);
                if (!gpuPmCountersSample.pSampleRangeData || !gpuPmCountersSample.pSampleEvaluatedMetrics)
                {
                    continue;
                }

                const uint64_t samplePeriodNsec = 1'000'000'000 / gpuPmCountersData.pmSampleRateHz;
                uint64_t endTimestampGpuNsec = gpuPmCountersSample.counterSample.endTimestampPtimer;
                uint64_t startTimestampGpuNsec = endTimestampGpuNsec > samplePeriodNsec ? endTimestampGpuNsec - samplePeriodNsec : 1;
                if (!CheckGpuTimestampRange(gpuPartitionData, startTimestampGpuNsec, endTimestampGpuNsec, updateObservedTimestamps, startTimestampGpuNsec, endTimestampGpuNsec))
                {
                    continue;
                }

                const uint64_t startTimestampUtcNsec = ConvertGpuNsecToUtcNsec(gpuPartitionData, startTimestampGpuNsec);
                const uint64_t endTimestampUtcNsec = ConvertGpuNsecToUtcNsec(gpuPartitionData, endTimestampGpuNsec);
                if (startTimestampUtcNsec == InvalidTimestamp || endTimestampUtcNsec == InvalidTimestamp)
                {
                    continue;
                }

                gpuPmCountersSample.startTimestampUtcNsec = startTimestampUtcNsec;
                gpuPmCountersSample.endTimestampUtcNsec = endTimestampUtcNsec;

                onGpuPmCountersSample(gpuPmCountersSample);
            }
        }

        // Traverse GPU PM counter samples and invoke the provided callback for each sample.
        // onGpuPmCountersSample: void(const GpuPmCountersSample& gpuPmCountersSample)
        template <typename TOnGpuPmCountersSample>
        void ForEachGpuPmCountersSamples(const GpuPartitionData& gpuPartitionData, TOnGpuPmCountersSample&& onGpuPmCountersSample)
        {
            ForEachGpuPmCountersSamples(gpuPartitionData, false, std::forward<TOnGpuPmCountersSample>(onGpuPmCountersSample));
        }

        /// Copies the sample at recordLocation into counterData.RangeData[sampleIndex].  After this call, the specified rangeIndex can be used with Nsight Perf SDK metrics evaluation calls.
        /// Pre-condition: counterDataImage must have been initialized via InitializeCounterData().
        bool CopyGpuPmCountersSample(GpuPmCountersData& gpuPmCountersData, std::vector<uint8_t>& counterDataImage, const size_t sampleIndex, const uint8_t* pSamplePayload, const size_t samplePayloadSize) const
        {
            const size_t rangeEndOffset = gpuPmCountersData.counterDataPreambleSize + (sampleIndex + 1) * gpuPmCountersData.sampleRangeDataSize;
            if (counterDataImage.size() < rangeEndOffset)
            {
                const size_t maxSampleCount = (counterDataImage.size() - gpuPmCountersData.counterDataPreambleSize) / gpuPmCountersData.sampleRangeDataSize;
                NVTRACE_LOG_ERR(20, "Out-of-range argument: rangeIndex=%zu  exceeds  maxRanges=%zu", sampleIndex, maxSampleCount);
                return false;
            }

            NVTRACE_LOG_ERR_RETURN_IF(!pSamplePayload, false, 20, "Invalid sample payload");
            NVTRACE_LOG_ERR_RETURN_IF(
                gpuPmCountersData.counterSampleFixedFieldSize < NvTrace_GPU_PmCounters_CounterSample_MINOR_VERSION_A || samplePayloadSize < gpuPmCountersData.counterSampleFixedFieldSize,
                false,
                20,
                "Invalid sample payload size");

            const size_t sampleRangeDataOffsetInPayload = AlignUpP2(static_cast<size_t>(gpuPmCountersData.counterSampleFixedFieldSize), alignof(uint64_t));
            if (samplePayloadSize < sampleRangeDataOffsetInPayload || samplePayloadSize - sampleRangeDataOffsetInPayload < gpuPmCountersData.sampleRangeDataSize)
            {
                NVTRACE_LOG_ERR(20, "Out-of-range argument: sampleRangeDataSize=%zu  exceeds  samplePayloadSize=%zu", gpuPmCountersData.sampleRangeDataSize, samplePayloadSize);
                return false;
            }
            uint32_t counterDataImageOffset = (uint32_t)(gpuPmCountersData.counterDataPreambleSize + sampleIndex * gpuPmCountersData.sampleRangeDataSize);
            std::memcpy(counterDataImage.data() + counterDataImageOffset, pSamplePayload + sampleRangeDataOffsetInPayload, gpuPmCountersData.sampleRangeDataSize);
            return true;
        }

        //////////////////////////////////////////////////
        // CUDA API Trace
        //////////////////////////////////////////////////

        const FB::Provider::CudaApiTrace::ProviderConfiguration* GetCudaApiTraceProviderConfiguration(const CudaApiTraceData& cudaApiTraceData) const
        {
            if (cudaApiTraceData.configurationRecordLocation == InvalidRecordLocation)
            {
                return nullptr;
            }

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(cudaApiTraceData.configurationRecordLocation);
            return recordPayloads.pMajorPayload ? flatbuffers::GetRoot<FB::Provider::CudaApiTrace::ProviderConfiguration>(recordPayloads.pMajorPayload) : nullptr;
        }

        static CudaApiTraceData::CuDeviceData* FindCuDeviceByGpuUuid(ProcessData& processData, NVTRACE_UUID gpuUuid)
        {
            for (CudaApiTraceData::CuDeviceData& device : processData.cudaApiTrace.devices)
            {
                if (device.gpuUuid == gpuUuid)
                {
                    return &device;
                }
            }

            return nullptr;
        }
        static const CudaApiTraceData::CuDeviceData* FindCuDeviceByGpuUuid(const ProcessData& processData, NVTRACE_UUID gpuUuid)
        {
            return FindCuDeviceByGpuUuid(const_cast<ProcessData&>(processData), gpuUuid);
        }

        static const CudaApiTraceData::CuStreamData* FindCuStream(const CudaApiTraceData& cudaApiTraceData, uint64_t custreamId)
        {
            auto itr = cudaApiTraceData.streams.find(custreamId);
            return itr != cudaApiTraceData.streams.end() ? &itr->second : nullptr;
        }

        static const CudaApiTraceData::GraphLaunch* FindGraphLaunch(const CudaApiTraceData& cudaApiTraceData, const CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch)
        {
            return graphNodeLaunch.graphLaunchIdx < cudaApiTraceData.graphLaunches.size() ? &cudaApiTraceData.graphLaunches[graphNodeLaunch.graphLaunchIdx] : nullptr;
        }

        const CudaApiTraceData::CuGraphNodeData* FindCuGraphNode(const CudaApiTraceData& cudaApiTraceData, uint64_t graphNodeCorrelationId) const
        {
            auto itr = cudaApiTraceData.graphNodes.find(graphNodeCorrelationId);
            return itr != cudaApiTraceData.graphNodes.end() ? &itr->second : nullptr;
        }

        const CudaApiTraceData::CuGraphNodeData* FindCuGraphNode(const CudaApiTraceData::CuDeviceData& cuDevice, const uint32_t hwCorrelationId, const uint64_t startTimestampTsc) const
        {
            struct Comparator
            {
                uint64_t startTimestampTsc;

                bool operator()(const uint32_t hwCorrelationId, const CudaApiTraceData::CuGraphNodeData* pRhs) const
                {
                    if (hwCorrelationId != pRhs->GetHwCorrelationId())
                    {
                        return hwCorrelationId < pRhs->GetHwCorrelationId();
                    }
                    return startTimestampTsc < pRhs->creationTimestampTsc;
                }
            };

            auto itr = std::upper_bound(cuDevice.graphNodes.begin(), cuDevice.graphNodes.end(), hwCorrelationId, Comparator{startTimestampTsc});
            if (itr == cuDevice.graphNodes.begin())
            {
                return nullptr; // not found
            }
            --itr;
            const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = *itr;
            if (pGraphNodeData->GetHwCorrelationId() != hwCorrelationId)
            {
                return nullptr; // not found
            }
            return pGraphNodeData;
        }

        NvTrace_CUDA_ApiTrace_RecordType GetCudaApiTraceRecordType(uint64_t recordLocation) const
        {
            return static_cast<NvTrace_CUDA_ApiTrace_RecordType>(GetRecordMetadata(recordLocation).recordType);
        }

        RecordMetadata GetRecordMetadata(uint64_t recordLocation) const
        {
            return m_pTraceReader->GetRecordMetadata(recordLocation);
        }

        const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch* GetImmediateKernelLaunchPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_GraphLaunch* GetGraphLaunchPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphLaunch*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_MemcpyBegin* GetMemcpyBeginPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemcpyBegin*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_MemsetBegin* GetMemsetBeginPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemsetBegin*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin* GetGraphMemcpyBeginPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_GraphMemsetBegin* GetGraphMemsetBeginPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphMemsetBegin*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_MemOpBegin* GetMemOpBeginPayload(uint64_t recordLocation, const CudaApiTraceData& cudaApiTraceData) const
        {
            const uint32_t memOpBeginMinorVersion = cudaApiTraceData.recordVersionInfo.memOpBeginMinorVersion;
            if (memOpBeginMinorVersion < NvTrace_CUDA_ApiTrace_MemOpBegin_MINOR_VERSION_A)
            {
                return nullptr;
            }

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(recordLocation);
            if (!recordPayloads.pMajorPayload || recordPayloads.maxMajorPayloadSize < memOpBeginMinorVersion)
            {
                return nullptr;
            }

            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemOpBegin*>(recordPayloads.pMajorPayload);
        }

        /// The caller reads the arm memOpType selects.
        const NvTrace_CUDA_ApiTrace_MemOpParams* GetMemOpParamsPayload(uint64_t recordLocation, const CudaApiTraceData& cudaApiTraceData, NvTrace_CUDA_ApiTrace_MemOpType memOpType) const
        {
            const uint32_t memOpBeginMinorVersion = cudaApiTraceData.recordVersionInfo.memOpBeginMinorVersion;
            if (memOpBeginMinorVersion < NvTrace_CUDA_ApiTrace_MemOpBegin_MINOR_VERSION_A)
            {
                return nullptr;
            }

            const uint32_t floorMinorVersion = CudaApiTraceMemOpParamsFloorMinorVersion(memOpType);
            if (floorMinorVersion == 0 || CudaApiTraceMemOpParamsMinorVersion(cudaApiTraceData.recordVersionInfo, memOpType) < floorMinorVersion)
            {
                return nullptr;
            }

            // The params begin at the published MemOpBegin size. Aligning that offset is the
            // producer's obligation and the observer checks it once on attach, so all this needs of
            // it is that it clears the common segment and leaves the arm inside the record.
            const uint32_t paramsOffset = cudaApiTraceData.recordVersionInfo.memOpBeginSize;
            if (paramsOffset < memOpBeginMinorVersion)
            {
                return nullptr;
            }
            const uint64_t paramsEndOffset = static_cast<uint64_t>(paramsOffset) + floorMinorVersion;

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(recordLocation);
            if (!recordPayloads.pMajorPayload || recordPayloads.maxMajorPayloadSize < paramsEndOffset)
            {
                return nullptr;
            }

            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_MemOpParams*>(recordPayloads.pMajorPayload + paramsOffset);
        }

        NvTrace_CUDA_ApiTrace_GraphNodeType GetCudaApiTraceGraphNodeType(uint64_t recordLocation, const CudaApiTraceData& cudaApiTraceData) const
        {
            const uint32_t graphNodeCreatedMinorVersion = cudaApiTraceData.recordVersionInfo.graphNodeCreatedMinorVersion;
            if (graphNodeCreatedMinorVersion < NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A)
            {
                return NvTrace_CUDA_ApiTrace_GraphNodeType_Invalid;
            }

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(recordLocation);
            if (!recordPayloads.pMajorPayload || recordPayloads.maxMajorPayloadSize < NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A)
            {
                return NvTrace_CUDA_ApiTrace_GraphNodeType_Invalid;
            }

            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphNodeCreated*>(recordPayloads.pMajorPayload)->graphNodeType;
        }

        const NvTrace_CUDA_ApiTrace_HostCallbackLaunch* GetHostCallbackLaunchPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_HostCallbackLaunch*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_HostCallbackExecBegin* GetHostCallbackExecBeginPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_HostCallbackExecBegin*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_HostCallbackExecEnd* GetHostCallbackExecEndPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_HostCallbackExecEnd*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch* GetGraphHostCallbackLaunchPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin* GetGraphHostCallbackExecBeginPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphHostCallbackExecBegin*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        const NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd* GetGraphHostCallbackExecEndPayload(uint64_t recordLocation) const
        {
            return reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphHostCallbackExecEnd*>(m_pTraceReader->GetRecordPayloads(recordLocation).pMajorPayload);
        }

        /// The caller reads the arm graphNodeType selects.
        const CudaApiTraceGraphNodeParams* GetCudaApiTraceGraphNodeParams(uint64_t recordLocation, const CudaApiTraceData& cudaApiTraceData, NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType) const
        {
            const uint32_t graphNodeCreatedMinorVersion = cudaApiTraceData.recordVersionInfo.graphNodeCreatedMinorVersion;
            if (graphNodeCreatedMinorVersion < NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A)
            {
                return nullptr;
            }

            const uint32_t floorMinorVersion = CudaApiTraceGraphNodeParamsFloorMinorVersion(graphNodeType);
            if (floorMinorVersion == 0 || CudaApiTraceGraphNodeParamsMinorVersion(cudaApiTraceData.recordVersionInfo, graphNodeType) < floorMinorVersion)
            {
                return nullptr;
            }

            const uint32_t paramsOffset = cudaApiTraceData.recordVersionInfo.graphNodeCreatedSize;
            if (paramsOffset < graphNodeCreatedMinorVersion)
            {
                return nullptr;
            }
            const uint64_t paramsEndOffset = static_cast<uint64_t>(paramsOffset) + floorMinorVersion;

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(recordLocation);
            if (!recordPayloads.pMajorPayload || recordPayloads.maxMajorPayloadSize < paramsEndOffset)
            {
                return nullptr;
            }

            return reinterpret_cast<const CudaApiTraceGraphNodeParams*>(recordPayloads.pMajorPayload + paramsOffset);
        }

        static const char* GetCuFunctionMangledName(const CudaApiTraceData& cudaApiTraceData, uint64_t cufunctionId)
        {
            auto itr = cudaApiTraceData.functions.find(cufunctionId);
            return itr != cudaApiTraceData.functions.end() && !itr->second.mangledName.empty() ? itr->second.mangledName.c_str() : "??";
        }

        /// A partially observed string still names the object it belongs to, so it is returned as assembled
        /// rather than dropped.
        const std::string& GetNvtxTraceStringTableValue(const NvtxTraceData& nvtxTraceData, uint64_t stringHandle) const
        {
            static const std::string kUnknown = "?? (Unknown string)";

            const auto itrEntry = nvtxTraceData.stringTableEntries.find(stringHandle);
            if (itrEntry == nvtxTraceData.stringTableEntries.end())
            {
                NVTRACE_LOG_WRN(50, "Unknown NvtxTrace stringHandle: %" PRIu64, stringHandle);
                return kUnknown;
            }

            const NvtxTraceData::StringTableEntryData& entryData = itrEntry->second;
            if (entryData.stringRegisteredRecordLocation == InvalidRecordLocation)
            {
                return kUnknown;
            }
            if (!entryData.cachedUtf8Value.empty())
            {
                return entryData.cachedUtf8Value;
            }

            const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo = nvtxTraceData.recordVersionInfo;
            const size_t stringRegisteredSize = recordVersionInfo.stringRegisteredSize;
            const size_t stringSegmentSize = recordVersionInfo.stringSegmentSize;
            const auto registeredRecordPayloads = m_pTraceReader->GetRecordPayloads(entryData.stringRegisteredRecordLocation);
            const bool stringReadable = NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A <= stringRegisteredSize
                                     && NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A <= stringSegmentSize
                                     && stringRegisteredSize <= registeredRecordPayloads.maxMajorPayloadSize;
            if (!stringReadable)
            {
                NVTRACE_LOG_ERR(50, "NvtxTrace StringRegistered record is too small (size=%zu payload=%zu)", stringRegisteredSize, registeredRecordPayloads.maxMajorPayloadSize);
                return kUnknown;
            }

            const auto& stringRegistered = *reinterpret_cast<const NvTrace_NvtxTrace_StringRegistered*>(registeredRecordPayloads.pMajorPayload);
            const uint32_t totalStringLength = stringRegistered.totalStringLength;
            std::string assembledValue(totalStringLength, '\0');

            const size_t maxInlineLength = std::min<size_t>(registeredRecordPayloads.maxMajorPayloadSize - stringRegisteredSize, totalStringLength);
            const size_t inlineLength = std::min<size_t>(stringRegistered.inlineStringSegmentLength, maxInlineLength);
            memcpy(assembledValue.data(), registeredRecordPayloads.pMajorPayload + stringRegisteredSize, inlineLength);

            for (const auto& [stringOffset, segmentRecordLocation] : entryData.segmentRecordLocations)
            {
                const auto segmentRecordPayloads = m_pTraceReader->GetRecordPayloads(segmentRecordLocation);
                const bool segmentReadable = stringSegmentSize <= segmentRecordPayloads.maxMajorPayloadSize && stringOffset < totalStringLength;
                if (!segmentReadable)
                {
                    continue;
                }

                const auto& stringSegment = *reinterpret_cast<const NvTrace_NvtxTrace_StringSegment*>(segmentRecordPayloads.pMajorPayload);
                const size_t maxSegmentLength = std::min<size_t>(segmentRecordPayloads.maxMajorPayloadSize - stringSegmentSize, totalStringLength - stringOffset);
                const size_t segmentLength = std::min<size_t>(stringSegment.stringSegmentLength, maxSegmentLength);
                memcpy(assembledValue.data() + stringOffset, segmentRecordPayloads.pMajorPayload + stringSegmentSize, segmentLength);
            }

            entryData.cachedUtf8Value = std::move(assembledValue);
            return entryData.cachedUtf8Value;
        }

        // An assembled message decoded into UTF-8: a RegisteredString one is the handle it carries,
        // resolved through the string table; anything else is already the message in messageType's
        // encoding, so UTF-8 needs no work at all and storage stays untouched.
        const char* DecodeNvtxTraceMessage(const NvtxTraceData& nvtxTraceData, uint16_t messageType, const std::string& message, std::string& storage, const char* pUnknownMessage) const
        {
            if (messageType == NvTrace_NvtxTrace_MessageType_RegisteredString)
            {
                if (message.size() < sizeof(uint64_t))
                {
                    return pUnknownMessage;
                }

                uint64_t stringHandle;
                memcpy(&stringHandle, message.data(), sizeof(uint64_t));
                return GetNvtxTraceStringTableValue(nvtxTraceData, stringHandle).c_str();
            }

            if (message.empty())
            {
                return pUnknownMessage;
            }

            switch (messageType)
            {
                case NvTrace_NvtxTrace_MessageType_UTF8:
                {
                    return message.c_str();
                }

                case NvTrace_NvtxTrace_MessageType_UTF16:
                {
                    // The message is raw and not NUL-terminated, so its length bounds the conversion.
                    const std::u16string wideMessage(reinterpret_cast<const char16_t*>(message.data()), message.size() / sizeof(char16_t));
                    ConvertUtf16ToUtf8(wideMessage.c_str(), storage);
                    return storage.c_str();
                }

                case NvTrace_NvtxTrace_MessageType_UTF32:
                {
                    const std::u32string wideMessage(reinterpret_cast<const char32_t*>(message.data()), message.size() / sizeof(char32_t));
                    ConvertUtf32ToUtf8(wideMessage.c_str(), storage);
                    return storage.c_str();
                }
            }

            NVTRACE_LOG_ERR(50, "Unexpected NvtxTrace messageType: %u", messageType);
            return pUnknownMessage;
        }

        const char* GetNvtxTraceMarkMessage(const NvtxTraceData& nvtxTraceData, const NvtxTraceData::MarkData& mark, std::string& storage) const
        {
            return DecodeNvtxTraceMessage(nvtxTraceData, mark.messageType, mark.message, storage, "?? (NVTX Mark)");
        }

        const NvTrace_NvtxTrace_EventAttributes* GetNvtxTraceMarkAttributes(const NvtxTraceData& nvtxTraceData, const NvtxTraceData::MarkData& mark) const
        {
            if (mark.recordLocation == InvalidRecordLocation)
            {
                return nullptr;
            }

            const uint8_t* pMajorPayload = m_pTraceReader->GetRecordPayloads(mark.recordLocation).pMajorPayload;
            const auto& markEvent = *reinterpret_cast<const NvTrace_NvtxTrace_Mark*>(pMajorPayload);
            if (!markEvent.hasEventAttributes)
            {
                return nullptr;
            }

            const size_t eventAttributesOffset = AlignUp(static_cast<size_t>(nvtxTraceData.recordVersionInfo.markSize), alignof(NvTrace_NvtxTrace_EventAttributes));
            return reinterpret_cast<const NvTrace_NvtxTrace_EventAttributes*>(pMajorPayload + eventAttributesOffset);
        }

        const char* GetNvtxTracePushPopRangeMessage(const NvtxTraceData& nvtxTraceData, const NvtxTraceData::PushPopRangeData& pushPopRange, std::string& storage) const
        {
            return DecodeNvtxTraceMessage(nvtxTraceData, pushPopRange.messageType, pushPopRange.message, storage, "?? (NVTX Push-Pop Range)");
        }

        // The attributes ride the range's own created record, between it and the inline message bytes,
        // exactly as a Mark carries them. Null when the range has none.
        const NvTrace_NvtxTrace_EventAttributes* GetNvtxTracePushPopRangeAttributes(const NvtxTraceData& nvtxTraceData, const NvtxTraceData::PushPopRangeData& pushPopRange) const
        {
            if (pushPopRange.stateRecordLocation == InvalidRecordLocation)
            {
                return nullptr;
            }

            const NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo = nvtxTraceData.recordVersionInfo;
            const size_t attributesOffset = recordVersionInfo.pushPopRangeCreatedSize;
            const auto createdRecordPayloads = m_pTraceReader->GetRecordPayloads(pushPopRange.stateRecordLocation);
            if (attributesOffset < NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A || createdRecordPayloads.maxMajorPayloadSize < attributesOffset)
            {
                return nullptr;
            }

            if (reinterpret_cast<const NvTrace_NvtxTrace_PushPopRangeCreated*>(createdRecordPayloads.pMajorPayload)->hasEventAttributes == 0)
            {
                return nullptr;
            }

            const bool attributesReadable = NvTrace_NvtxTrace_EventAttributes_MINOR_VERSION_A <= recordVersionInfo.eventAttributesSize
                                         && recordVersionInfo.eventAttributesSize <= createdRecordPayloads.maxMajorPayloadSize - attributesOffset;
            NVTRACE_LOG_ERR_RETURN_IF(!attributesReadable, nullptr, 50, "NvtxTrace PushPopRangeCreated record claims attributes it cannot carry (createdSize=%zu attributesSize=%u payload=%zu)", attributesOffset, recordVersionInfo.eventAttributesSize, createdRecordPayloads.maxMajorPayloadSize);

            return reinterpret_cast<const NvTrace_NvtxTrace_EventAttributes*>(createdRecordPayloads.pMajorPayload + attributesOffset);
        }

        const std::string& GetNvtxTraceDomainName(const NvtxTraceData& nvtxTraceData, uint32_t domainIndex) const
        {
            static const std::string kEmpty;
            static const std::string kUnknown = "?? (Unknown domain)";

            if (!domainIndex)
            {
                return kEmpty;
            }

            const auto itrDomain = nvtxTraceData.domains.find(domainIndex);
            if (itrDomain == nvtxTraceData.domains.end() || itrDomain->second.domainCreatedRecordLocation == InvalidRecordLocation)
            {
                return kUnknown;
            }

            const auto recordPayloads = m_pTraceReader->GetRecordPayloads(itrDomain->second.domainCreatedRecordLocation);
            const bool domainCreatedReadable = NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A <= nvtxTraceData.recordVersionInfo.domainCreatedSize
                                            && NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A <= recordPayloads.maxMajorPayloadSize;
            if (!domainCreatedReadable)
            {
                NVTRACE_LOG_ERR(50, "NvtxTrace DomainCreated record is too small (size=%u payload=%zu)", nvtxTraceData.recordVersionInfo.domainCreatedSize, recordPayloads.maxMajorPayloadSize);
                return kUnknown;
            }

            const auto& domainCreated = *reinterpret_cast<const NvTrace_NvtxTrace_DomainCreated*>(recordPayloads.pMajorPayload);
            if (domainCreated.nameStringHandle == 0)
            {
                return kEmpty;
            }

            return GetNvtxTraceStringTableValue(nvtxTraceData, domainCreated.nameStringHandle);
        }

        // Traverse the GPU events for a device and invoke the provided callback for each event
        // onDeviceEvent: void(const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent)
        template <typename TOnDeviceEvent>
        void TraverseDeviceEventsForGpuPartition(
            const GpuPartitionData& gpuPartition,
            TOnDeviceEvent&& onDeviceEvent)
        {
            struct ProcessTraversalState
            {
                DeviceEventOrigin deviceEventOrigin;
                DeviceEventStreams eventStreams;
                std::priority_queue<DeviceEvent, std::vector<DeviceEvent>, std::greater<DeviceEvent>> deviceEvents;

                ProcessTraversalState(const DeviceEventOrigin& deviceEventOrigin)
                    : deviceEventOrigin(deviceEventOrigin)
                    , eventStreams(CreateDeviceEventStreams(*deviceEventOrigin.pCuDevice))
                {
                    for (auto& eventStream : eventStreams)
                    {
                        DeviceEvent deviceEvent = eventStream.GetNextEvent();
                        if (deviceEvent)
                        {
                            deviceEvents.emplace(deviceEvent);
                        }
                    }
                }

                DeviceEvent NextDeviceEvent()
                {
                    const DeviceEvent deviceEvent = deviceEvents.top();
                    deviceEvents.pop();

                    DeviceEventStream& eventStream = eventStreams[static_cast<size_t>(deviceEvent.eventType) - 1];
                    DeviceEvent nextDeviceEvent = eventStream.GetNextEvent();
                    if (nextDeviceEvent)
                    {
                        deviceEvents.emplace(nextDeviceEvent);
                    }

                    return deviceEvent;
                }

                bool operator>(const ProcessTraversalState& rhs) const
                {
                    const DeviceEvent& lhsEvent = deviceEvents.top();
                    const DeviceEvent& rhsEvent = rhs.deviceEvents.top();
                    if (lhsEvent > rhsEvent)
                    {
                        return true;
                    }
                    if (rhsEvent > lhsEvent)
                    {
                        return false;
                    }
                    return deviceEventOrigin.processLuid > rhs.deviceEventOrigin.processLuid;
                }
            };

            std::vector<ProcessTraversalState> processTraversalStates;

            // Initialize process traversal states
            processTraversalStates.reserve(processes.size());
            for (auto& [processLuid, process] : processes)
            {
                auto& cudaApiTrace = process.cudaApiTrace;
                // Find the correct device
                auto deviceItr = std::find_if(cudaApiTrace.devices.begin(), cudaApiTrace.devices.end(), [&](const CudaApiTraceData::CuDeviceData& cuDevice) {
                    return cuDevice.gpuUuid == gpuPartition.gpuUuid;
                });

                if (deviceItr != cudaApiTrace.devices.end())
                {
                    processTraversalStates.emplace_back(DeviceEventOrigin{processLuid, &process, &cudaApiTrace, &(*deviceItr)});
                }
            }

            auto processTraversalStateComparator = [&](const size_t& lhsIndex, const size_t& rhsIndex) {
                return processTraversalStates[lhsIndex] > processTraversalStates[rhsIndex];
            };
            std::priority_queue<size_t, std::vector<size_t>, decltype(processTraversalStateComparator)> processTraversalStateQueue(processTraversalStateComparator);

            for (size_t processTraversalStateIndex = 0; processTraversalStateIndex < processTraversalStates.size(); processTraversalStateIndex++)
            {
                if (!processTraversalStates[processTraversalStateIndex].deviceEvents.empty())
                {
                    processTraversalStateQueue.push(processTraversalStateIndex);
                }
            }

            while (!processTraversalStateQueue.empty())
            {
                size_t processTraversalStateIndex = processTraversalStateQueue.top();
                auto& processTraversalState = processTraversalStates[processTraversalStateIndex];
                processTraversalStateQueue.pop();

                const DeviceEvent deviceEvent = processTraversalState.NextDeviceEvent();
                onDeviceEvent(processTraversalState.deviceEventOrigin, deviceEvent);

                // Add back into the queue if it has more events
                if (!processTraversalState.deviceEvents.empty())
                {
                    processTraversalStateQueue.push(processTraversalStateIndex);
                }
            }
        }

        // Traverse the GPU events for a device and invoke the provided callback for each event
        // Additionally, splits events based on active context
        // onDeviceEvent: void(const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent)
        template <typename TOnDeviceEvent>
        void TraverseDeviceEventsForGpuPartitionAdvanced(
            const GpuPartitionData& gpuPartition,
            TOnDeviceEvent&& onDeviceEvent)
        {
            struct DeviceProcessState
            {
                DeviceEventOrigin deviceEventOrigin;
                std::deque<DeviceEvent> ongoingDeviceEventsQueue;
            };

            std::map<NVTRACE_LUID, DeviceProcessState> deviceProcessStateMap;

            auto& residentContextVector = gpuPartition.gpuGrCtxTrace.residentContexts;
            if (residentContextVector.empty())
            {
                return;
            }

            bool continueProcessing = true;
            size_t currentContextIndex = 0;
            uint64_t currentContextStartTimestampGpuNsec = residentContextVector[0].ctxLoadTimestampGpuNsec;
            uint64_t currentContextEndTimestampGpuNsec = residentContextVector[0].ctxSaveTimestampGpuNsec;

            auto findNextValidContext = [&]() -> bool {
                currentContextIndex++;
                if (currentContextIndex >= residentContextVector.size())
                {
                    return false;
                }

                currentContextStartTimestampGpuNsec = residentContextVector[currentContextIndex].ctxLoadTimestampGpuNsec;
                currentContextEndTimestampGpuNsec = residentContextVector[currentContextIndex].ctxSaveTimestampGpuNsec;
                return true;
            };

            // Lambda to emit all relevant device events of a DeviceProcessState for the current context
            auto emitDeviceEventsForDeviceProcessState = [&](DeviceProcessState& deviceProcessState) -> void {
                auto deviceEventItr = deviceProcessState.ongoingDeviceEventsQueue.begin();
                while (deviceEventItr != deviceProcessState.ongoingDeviceEventsQueue.end())
                {
                    // Remove stale device events
                    if (deviceEventItr->endTimestampGpuNsec <= currentContextStartTimestampGpuNsec)
                    {
                        deviceEventItr = deviceProcessState.ongoingDeviceEventsQueue.erase(deviceEventItr);
                        continue;
                    }

                    DeviceEvent deviceEvent = *deviceEventItr;

                    if (deviceEvent.startTimestampGpuNsec >= currentContextEndTimestampGpuNsec)
                    {
                        break;
                    }

                    deviceEvent.startTimestampGpuNsec = std::max(deviceEvent.startTimestampGpuNsec, currentContextStartTimestampGpuNsec);
                    deviceEvent.endTimestampGpuNsec = std::min(deviceEvent.endTimestampGpuNsec, currentContextEndTimestampGpuNsec);
                    onDeviceEvent(deviceProcessState.deviceEventOrigin, deviceEvent);
                    deviceEventItr++;
                }
            };

            auto processCurrentContextEntry = [&]() -> bool {
                if (!continueProcessing)
                {
                    return false;
                }

                auto deviceProcessStateItr = deviceProcessStateMap.find(residentContextVector[currentContextIndex].processLuid);
                if (deviceProcessStateItr != deviceProcessStateMap.end())
                {
                    emitDeviceEventsForDeviceProcessState(deviceProcessStateItr->second);
                }

                continueProcessing = findNextValidContext();

                return continueProcessing;
            };

            TraverseDeviceEventsForGpuPartition(
                gpuPartition,
                [&](const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent) {
                    while (deviceEvent.startTimestampGpuNsec >= currentContextEndTimestampGpuNsec && processCurrentContextEntry())
                        ;

                    auto [deviceProcessStateItr, isNew] = deviceProcessStateMap.try_emplace(deviceEventOrigin.processLuid, DeviceProcessState{deviceEventOrigin, {}});
                    deviceProcessStateItr->second.ongoingDeviceEventsQueue.push_back(deviceEvent);
                });

            // Process any remaining contexts
            while (processCurrentContextEntry())
                ;
        }

        // Traverse the GPU events for a device and invoke the provided callback for each event
        // Splits events based on active context
        // Additionally, provides an eventDepth which guarantees that emitted events of the same depth will not collide
        // onDeviceEvent: void(const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent, size_t eventDepth)
        template <typename TOnDeviceEvent>
        void TraverseDeviceEventsForGpuPartitionAdvancedParallel(
            const GpuPartitionData& gpuPartition,
            TOnDeviceEvent&& onDeviceEvent)
        {
            using ActiveDeviceEvent = std::pair<size_t, DeviceEvent>;
            auto activeDeviceEventComparator = [](const ActiveDeviceEvent& lhs, const ActiveDeviceEvent& rhs) {
                return lhs.second.endTimestampGpuNsec > rhs.second.endTimestampGpuNsec;
            };
            std::priority_queue<ActiveDeviceEvent, std::vector<ActiveDeviceEvent>, decltype(activeDeviceEventComparator)> activeDeviceEventsQueue(activeDeviceEventComparator);

            std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> availableSlotsQueue;

            TraverseDeviceEventsForGpuPartitionAdvanced(
                gpuPartition,
                [&](const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent) {
                    while (!activeDeviceEventsQueue.empty())
                    {
                        const ActiveDeviceEvent& activeDeviceEvent = activeDeviceEventsQueue.top();
                        if (activeDeviceEvent.second.endTimestampGpuNsec <= deviceEvent.startTimestampGpuNsec)
                        {
                            availableSlotsQueue.push(activeDeviceEvent.first);
                            activeDeviceEventsQueue.pop();
                        }
                        else
                        {
                            break;
                        }
                    }

                    size_t foundSlotIndex;
                    if (availableSlotsQueue.empty())
                    {
                        foundSlotIndex = activeDeviceEventsQueue.size();
                        activeDeviceEventsQueue.emplace(foundSlotIndex, deviceEvent);
                    }
                    else
                    {
                        foundSlotIndex = availableSlotsQueue.top();
                        availableSlotsQueue.pop();
                        activeDeviceEventsQueue.emplace(foundSlotIndex, deviceEvent);
                    }

                    onDeviceEvent(deviceEventOrigin, deviceEvent, foundSlotIndex);
                });
        }

        std::vector<const NvtxTraceData::PushPopRangeData*> GetNvtxTraceProjectionStack(
            const NvtxTraceData& nvtxTraceData,
            uint32_t domainCliqueIndex,
            uint64_t uniqueThreadId,
            size_t topPushPopRangeIndex) const
        {
            std::vector<const NvtxTraceData::PushPopRangeData*> projectionStack;
            if (domainCliqueIndex >= nvtxTraceData.domainCliques.size())
            {
                return projectionStack;
            }

            const auto& domainClique = nvtxTraceData.domainCliques[domainCliqueIndex];
            const auto threadItr = domainClique.threads.find(uniqueThreadId);
            if (threadItr == domainClique.threads.end() || topPushPopRangeIndex >= threadItr->second.pushPopRanges.size())
            {
                return projectionStack;
            }

            const auto& pushPopRanges = threadItr->second.pushPopRanges;
            for (size_t pushPopRangeIndex = topPushPopRangeIndex;
                 pushPopRangeIndex != NvtxTraceData::PushPopRangeData::InvalidPushPopRangeIndex;
                 pushPopRangeIndex = pushPopRanges[pushPopRangeIndex].parentRangeIndex)
            {
                projectionStack.push_back(&pushPopRanges[pushPopRangeIndex]);
            }
            std::reverse(projectionStack.begin(), projectionStack.end());
            return projectionStack;
        }

        // Finds what links a device event to CPU NVTX stacks: its launching driver API call and, for graph work, its graph node.
        // Returns false for an event type that has neither.
        bool GetDeviceEventNvtxSources(
            const DeviceEventOrigin& deviceEventOrigin,
            const DeviceEvent& deviceEvent,
            CudaApiTraceData::DriverApiCallRef& out_driverApiCallRef,
            const CudaApiTraceData::CuGraphNodeData*& out_pGraphNodeData) const
        {
            out_driverApiCallRef = {};
            out_pGraphNodeData = nullptr;
            switch (deviceEvent.eventType)
            {
                case DeviceEventType_ImmediateKernelLaunch:
                {
                    out_driverApiCallRef = deviceEvent.pImmediateKernelLaunch->driverApiCallRef;
                    return true;
                }
                case DeviceEventType_GraphNodeLaunch:
                {
                    const CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch = *deviceEvent.pGraphNodeLaunch;
                    const CudaApiTraceData::GraphLaunch* pGraphLaunch = deviceEventOrigin ? FindGraphLaunch(*deviceEventOrigin.pCudaApiTrace, graphNodeLaunch) : nullptr;
                    if (pGraphLaunch && pGraphLaunch->source == CudaApiTraceData::GraphLaunchSource::Host)
                    {
                        out_driverApiCallRef = pGraphLaunch->payload.host.driverApiCallRef;
                    }
                    if (deviceEventOrigin.pCudaApiTrace)
                    {
                        out_pGraphNodeData = FindCuGraphNode(*deviceEventOrigin.pCudaApiTrace, graphNodeLaunch.graphNodeCorrelationId);
                    }
                    return true;
                }
                case DeviceEventType_GpuCommand:
                {
                    const CudaApiTraceData::GpuCommand& gpuCommand = *deviceEvent.pGpuCommand;
                    out_driverApiCallRef = gpuCommand.driverApiCallRef;
                    if (deviceEventOrigin.pCudaApiTrace && gpuCommand.graphNodeCorrelationId != UINT64_MAX)
                    {
                        out_pGraphNodeData = FindCuGraphNode(*deviceEventOrigin.pCudaApiTrace, gpuCommand.graphNodeCorrelationId);
                    }
                    return true;
                }
                default:
                    return false;
            }
        }

        // Builds the root-to-leaf NVTX stack associated with a device event in one domain clique: the launch-time stack,
        // followed by the record-time stack entries past their shared prefix.
        // Returns false when no stack can be associated (no NVTX provider, or neither a launch thread nor a record-time stack).
        bool GetNvtxAssociationStack(
            const ProcessData& processData,
            uint32_t domainCliqueIndex,
            const CudaApiTraceData::DriverApiCallRef& driverApiCallRef,
            const CudaApiTraceData::CuGraphNodeData* pGraphNodeData,
            std::vector<NvtxAssociationStackEntry>& out_associationStack,
            size_t* out_launchTimeDepth = nullptr) const
        {
            out_associationStack.clear();
            if (processData.nvtxTrace.providerLuid == NVTRACE_NIL_LUID)
            {
                return false;
            }

            const size_t* pRecordTimeTopPushPopRangeIndex = nullptr;
            if (pGraphNodeData)
            {
                const auto recordStackItr = pGraphNodeData->nvtxTraceDomainCliquePushPopRanges.find(domainCliqueIndex);
                if (recordStackItr != pGraphNodeData->nvtxTraceDomainCliquePushPopRanges.end())
                {
                    pRecordTimeTopPushPopRangeIndex = &recordStackItr->second;
                }
            }

            // A known launch thread is required unless this graph event has a record-time stack
            if (!driverApiCallRef.uniqueThreadId && !pRecordTimeTopPushPopRangeIndex)
            {
                return false;
            }

            const auto cudaThreadItr = processData.cudaApiTrace.threads.find(driverApiCallRef.uniqueThreadId);
            if (cudaThreadItr != processData.cudaApiTrace.threads.end()
                && driverApiCallRef.driverApiCallIdx < cudaThreadItr->second.driverApiCalls.size())
            {
                const CudaApiTraceData::DriverApiCall& driverApiCall = cudaThreadItr->second.driverApiCalls[driverApiCallRef.driverApiCallIdx];
                const auto driverApiCallRangeRefItr = driverApiCall.nvtxTraceDomainCliquePushPopRanges.find(domainCliqueIndex);
                if (driverApiCallRangeRefItr != driverApiCall.nvtxTraceDomainCliquePushPopRanges.end())
                {
                    for (const NvtxTraceData::PushPopRangeData* pPushPopRange : GetNvtxTraceProjectionStack(processData.nvtxTrace, domainCliqueIndex, driverApiCallRef.uniqueThreadId, driverApiCallRangeRefItr->second))
                    {
                        out_associationStack.push_back({pPushPopRange, NvtxAssociationSource_LaunchTime});
                    }
                }
            }

            if (out_launchTimeDepth)
            {
                *out_launchTimeDepth = out_associationStack.size();
            }

            if (pRecordTimeTopPushPopRangeIndex)
            {
                const std::vector<const NvtxTraceData::PushPopRangeData*> recordTimeProjectionStack = GetNvtxTraceProjectionStack(
                    processData.nvtxTrace,
                    domainCliqueIndex,
                    pGraphNodeData->uniqueThreadId,
                    *pRecordTimeTopPushPopRangeIndex);

                size_t sharedDepth = 0;
                while (sharedDepth < out_associationStack.size()
                       && sharedDepth < recordTimeProjectionStack.size()
                       && out_associationStack[sharedDepth].pPushPopRange == recordTimeProjectionStack[sharedDepth])
                {
                    out_associationStack[sharedDepth].associationSources |= NvtxAssociationSource_RecordTime;
                    ++sharedDepth;
                }
                for (size_t recordTimeDepth = sharedDepth; recordTimeDepth < recordTimeProjectionStack.size(); ++recordTimeDepth)
                {
                    out_associationStack.push_back({recordTimeProjectionStack[recordTimeDepth], NvtxAssociationSource_RecordTime});
                }
            }
            return true;
        }

        // Generates all GPU events and NVTX projection events for the given device, process, and domain clique
        // setting processLuid to NVTRACE_NIL_LUID indicates that all processes should be included
        // onNvtxGpuProjectionEvent: void(const DeviceEventOrigin& deviceEventOrigin, const NvtxDeviceProjectionEvent& nvtxDeviceProjectionEvent, size_t eventDepth)
        // onDeviceEvent: void(const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent, size_t eventDepth)
        template <typename TOnNvtxDeviceProjectionEvent, typename TOnDeviceEvent>
        void GenerateGpuPartitionTimelineWithNvtxProjection(
            const GpuPartitionData& gpuPartition,
            NVTRACE_LUID processLuid,
            uint32_t domainCliqueIndex,
            TOnNvtxDeviceProjectionEvent&& onNvtxDeviceProjectionEvent,
            TOnDeviceEvent&& onDeviceEvent)
        {
            struct NvtxOngoingProjectionEvent
            {
                const NvtxTraceData::PushPopRangeData* pPushPopRange;
                uint64_t startTimestampGpuNsec;
                uint16_t initialStackSize;
                uint16_t stackDepth;
            };

            struct NvtxCompletedProjectionEvent
            {
                const NvtxTraceData::PushPopRangeData* pPushPopRange;
                uint64_t startTimestampGpuNsec;
                uint64_t endTimestampGpuNsec;
                uint16_t initialStackSize;
                uint16_t stackDepth;
            };

            struct ProjectionDepthState
            {
                std::vector<NvtxOngoingProjectionEvent> ongoingProjectionEventsStack;
                CudaApiTraceData::DriverApiCallRef ongoingProjectionDriverApiCallRef;
                size_t ongoingProjectionLaunchTimeDepth = 0;

                std::vector<NvtxCompletedProjectionEvent> completedProjectionEventsStack;
                std::vector<DeviceEvent> completedDeviceEventsStack;

                uint64_t lastEndTimestampGpuNsec = 0;
                uint64_t currentUniqueThreadId = 0;

                DeviceEventOrigin currentDeviceEventOrigin;
            };

            std::vector<ProjectionDepthState> projectionDepthStates;

            // Function to emit all pending NVTX projection events
            auto emitAllPendingNvtxProjectionEvents = [&](size_t projectionDepthStateIndex, bool emitOngoingEvents = true) {
                auto& state = projectionDepthStates[projectionDepthStateIndex];

                if (emitOngoingEvents)
                {
                    // Emit ongoing events directly
                    // These would normally be pushed onto the completed stack in reverse, then popped in reverse, so these need to be emitted in forward order
                    for (auto ongoingEventItr = state.ongoingProjectionEventsStack.begin(); ongoingEventItr != state.ongoingProjectionEventsStack.end(); ongoingEventItr++)
                    {
                        auto& ongoingEvent = *ongoingEventItr;
                        onNvtxDeviceProjectionEvent(
                            state.currentDeviceEventOrigin,
                            NvtxDeviceProjectionEvent{ongoingEvent.pPushPopRange, ongoingEvent.startTimestampGpuNsec, state.lastEndTimestampGpuNsec, ongoingEvent.initialStackSize, ongoingEvent.stackDepth},
                            projectionDepthStateIndex);
                    }
                    state.ongoingProjectionEventsStack.clear();
                }

                // Emit all completed events
                // These need to be emitted in reverse order so that they stack correctly (parent before child)
                for (auto completedEventItr = state.completedProjectionEventsStack.rbegin(); completedEventItr != state.completedProjectionEventsStack.rend(); completedEventItr++)
                {
                    auto& completedEvent = *completedEventItr;
                    onNvtxDeviceProjectionEvent(
                        state.currentDeviceEventOrigin,
                        NvtxDeviceProjectionEvent{completedEvent.pPushPopRange, completedEvent.startTimestampGpuNsec, completedEvent.endTimestampGpuNsec, completedEvent.initialStackSize, completedEvent.stackDepth},
                        projectionDepthStateIndex);
                }
                state.completedProjectionEventsStack.clear();

                // Emit all completed device events
                // The order that these are emitted does not matter
                for (auto deviceEventItr = state.completedDeviceEventsStack.begin(); deviceEventItr != state.completedDeviceEventsStack.end(); deviceEventItr++)
                {
                    onDeviceEvent(
                        state.currentDeviceEventOrigin,
                        *deviceEventItr,
                        projectionDepthStateIndex);
                }
                state.completedDeviceEventsStack.clear();
            };

            // Generate the Device Events and NVTX Projection Events
            TraverseDeviceEventsForGpuPartitionAdvancedParallel(
                gpuPartition,
                [&](const DeviceEventOrigin& deviceEventOrigin, const DeviceEvent& deviceEvent, size_t deviceEventDepth) {
                    if (deviceEventDepth >= projectionDepthStates.size())
                    {
                        projectionDepthStates.resize(deviceEventDepth + 1);
                    }

                    auto& state = projectionDepthStates[deviceEventDepth];

                    auto deviceEventProcessLuid = deviceEventOrigin.processLuid;

                    uint64_t startTimestampGpuNsec = deviceEvent.startTimestampGpuNsec;
                    uint64_t endTimestampGpuNsec = deviceEvent.endTimestampGpuNsec;
                    CudaApiTraceData::DriverApiCallRef driverApiCallRef;
                    const CudaApiTraceData::CuGraphNodeData* pGraphNodeData = nullptr;

                    if (!GetDeviceEventNvtxSources(deviceEventOrigin, deviceEvent, driverApiCallRef, pGraphNodeData))
                    {
                        return;
                    }

                    // Ignore events with invalid timestamps
                    if (startTimestampGpuNsec == InvalidTimestamp || endTimestampGpuNsec == InvalidTimestamp)
                    {
                        return;
                    }

                    // Updates that need to occur after this event is processed
                    auto stateUpdateGuard = ScopeExitGuard([&]() {
                        state.completedDeviceEventsStack.push_back(deviceEvent);
                        state.lastEndTimestampGpuNsec = endTimestampGpuNsec;
                        state.currentUniqueThreadId = driverApiCallRef.uniqueThreadId;
                        state.currentDeviceEventOrigin = deviceEventOrigin;
                    });

                    // In all unhappy paths, the new kernel will have no NVTX events
                    auto stackResetGuard = ScopeExitGuard([&]() {
                        emitAllPendingNvtxProjectionEvents(deviceEventDepth);
                    });

                    // Verify that we are processing NVTX events for this process
                    if (processLuid != NVTRACE_NIL_LUID && deviceEventProcessLuid != processLuid)
                    {
                        return;
                    }

                    // Encountering an event from a different process, thread, or unknown launch thread is a hard stack reset
                    if (!deviceEventOrigin
                        || deviceEventProcessLuid != state.currentDeviceEventOrigin.processLuid
                        || !driverApiCallRef.uniqueThreadId
                        || driverApiCallRef.uniqueThreadId != state.currentUniqueThreadId)
                    {
                        emitAllPendingNvtxProjectionEvents(deviceEventDepth);
                    }

                    if (!deviceEventOrigin)
                    {
                        return;
                    }

                    std::vector<NvtxAssociationStackEntry> deviceEventProjectionStack;
                    size_t launchTimeDepth = 0;
                    if (!GetNvtxAssociationStack(
                            *deviceEventOrigin.pProcessData,
                            domainCliqueIndex,
                            driverApiCallRef,
                            pGraphNodeData,
                            deviceEventProjectionStack,
                            &launchTimeDepth))
                    {
                        return;
                    }

                    size_t currentDepth = deviceEventProjectionStack.size();
                    size_t ongoingDepth = state.ongoingProjectionEventsStack.size();
                    const size_t sharedDepth = std::min(currentDepth, ongoingDepth);
                    const size_t minLaunchTimeDepth = std::min(launchTimeDepth, state.ongoingProjectionLaunchTimeDepth);

                    // Convert excessive ongoing events to completed events
                    while (ongoingDepth > sharedDepth)
                    {
                        auto& ongoingEvent = state.ongoingProjectionEventsStack[ongoingDepth - 1];
                        state.completedProjectionEventsStack.push_back({ongoingEvent.pPushPopRange, ongoingEvent.startTimestampGpuNsec, state.lastEndTimestampGpuNsec, ongoingEvent.initialStackSize, ongoingEvent.stackDepth});
                        ongoingDepth--;
                    }

                    // Resize
                    state.ongoingProjectionEventsStack.resize(currentDepth);

                    // Add events from the new stack until we are even with the ongoing event stack depth
                    while (currentDepth > sharedDepth)
                    {
                        state.ongoingProjectionEventsStack[currentDepth - 1] = {
                            deviceEventProjectionStack[currentDepth - 1].pPushPopRange,
                            startTimestampGpuNsec,
                            static_cast<uint16_t>(state.ongoingProjectionEventsStack.size()),
                            static_cast<uint16_t>(currentDepth)};
                        currentDepth--;
                    }

                    // Compare events at each depth of both stacks until we find a shared event
                    while (currentDepth > 0)
                    {
                        auto& ongoingEvent = state.ongoingProjectionEventsStack[currentDepth - 1];
                        if (ongoingEvent.pPushPopRange == deviceEventProjectionStack[currentDepth - 1].pPushPopRange)
                        {
                            if (currentDepth <= minLaunchTimeDepth || state.ongoingProjectionDriverApiCallRef == driverApiCallRef)
                            {
                                // Found a shared NVTX event
                                break;
                            }
                        }
                        currentDepth--;

                        // Complete an event and replace it with the new event
                        state.completedProjectionEventsStack.push_back({ongoingEvent.pPushPopRange, ongoingEvent.startTimestampGpuNsec, state.lastEndTimestampGpuNsec, ongoingEvent.initialStackSize, ongoingEvent.stackDepth});
                        ongoingEvent = {
                            deviceEventProjectionStack[currentDepth].pPushPopRange,
                            startTimestampGpuNsec,
                            static_cast<uint16_t>(state.ongoingProjectionEventsStack.size()),
                            static_cast<uint16_t>(currentDepth + 1)};
                    }

                    state.ongoingProjectionLaunchTimeDepth = launchTimeDepth;
                    state.ongoingProjectionDriverApiCallRef = driverApiCallRef;

                    // Check for no shared events
                    if (currentDepth == 0)
                    {
                        // Emit all pending events, but preserve the ongoing events
                        emitAllPendingNvtxProjectionEvents(deviceEventDepth, false);
                    }

                    stackResetGuard.Dismiss();
                }); // End device-event traversal

            // Emit all pending events
            for (size_t projectionDepthStateIndex = 0; projectionDepthStateIndex < projectionDepthStates.size(); projectionDepthStateIndex++)
            {
                emitAllPendingNvtxProjectionEvents(projectionDepthStateIndex);
            }
        }

    private:
        template <typename TCalibrationPoint, typename TConvertToUtcNsec, typename TCallback>
        void ForEachClockCalibrationInterval(
            const std::vector<TCalibrationPoint>& calibrationPoints,
            TConvertToUtcNsec&& convertToUtcNsec,
            TCallback&& callback) const
        {
            if (calibrationPoints.empty())
            {
                return;
            }

            const auto emitInterval = [&](const TCalibrationPoint& calibrationPoint, uint64_t startUtcNsec, uint64_t endUtcNsec, bool isExtrapolated) {
                if (startUtcNsec == InvalidTimestamp || endUtcNsec == InvalidTimestamp)
                {
                    return;
                }

                ClockCalibrationInterval interval;
                interval.startUtcNsec = startUtcNsec;
                interval.endUtcNsec = endUtcNsec;
                interval.clampedStartUtcNsec = std::max(m_observationStartUtcNsec, startUtcNsec);
                interval.clampedEndUtcNsec = std::min(m_observationEndUtcNsec, endUtcNsec);
                interval.isExtrapolated = isExtrapolated;
                if (interval.clampedEndUtcNsec <= interval.clampedStartUtcNsec)
                {
                    return;
                }

                callback(calibrationPoint, interval);
            };

            uint64_t startUtcNsec = m_observationStartUtcNsec;
            for (const TCalibrationPoint& calibrationPoint : calibrationPoints)
            {
                const uint64_t endUtcNsec = convertToUtcNsec(calibrationPoint);
                emitInterval(calibrationPoint, startUtcNsec, endUtcNsec, false);
                startUtcNsec = endUtcNsec;
            }

            const TCalibrationPoint& lastCalibrationPoint = calibrationPoints.back();
            emitInterval(lastCalibrationPoint, startUtcNsec, m_observationEndUtcNsec, true);
        }
    };

} // namespace nv::trace::Codec
