/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdio.h>
#include <string.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>
#include <NvTrace/Target/NvTraceClientLogger.h>

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Offline/Exporter/NvTraceExporterBase.h>
#include <NvTrace/Offline/Exporter/NvTraceGpuPmOfflineMetricsEvaluator.h>
#include <NvTrace/Offline/Exporter/NvTracePerfettoTraceWriter.h>
#include <NvTrace/Offline/Exporter/NvTraceZipStreamingCompressor.h>
#include <NvTrace/Offline/NvTraceCudaPublicEnumNames.h>
#include <NvTrace/Offline/NvTracePrintHelpers.h>
#include <NvTrace/Offline/NvTraceYamlCodec.h>
#include <NvTrace/Target/Consumer/NvTraceProviderTypes.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>

namespace nv::trace {

    struct PerfettoTraceExporterOptions : TraceExporterOptions
    {
        // Places GPU event rows under a separate parent instead of alongside each GPU's counters.
        bool isolateGpuEventRows = true;

        // Writes a ZIP archive containing the trace and perfetto_manifest instead of a raw trace file.
        bool enableTraceArchive = true;
    };

    /**
     * @brief Exports NvTrace DataModel traces as Perfetto trace archives.

     */
    template <typename TTraceReader>
    class PerfettoTraceExporter : private TraceExporterBase<TTraceReader>
    {
        using RecordMetadata = typename TTraceReader::RecordMetadata;

        using TraceExporterBase<TTraceReader>::m_dataModel;
        using TraceExporterBase<TTraceReader>::kUnknownData;
        using TraceExporterBase<TTraceReader>::kNotApplicableData;
        using TraceExporterBase<TTraceReader>::GetRuntimeApiFunctionName;
        using TraceExporterBase<TTraceReader>::GetDriverApiFunctionName;
        using TraceExporterBase<TTraceReader>::GetDemangledCuFunctionName;
        using TraceExporterBase<TTraceReader>::GetCuFunctionNames;
        using TraceExporterBase<TTraceReader>::EnsureOutputParentDirectoryExists;

    public:
        ~PerfettoTraceExporter() = default;
        PerfettoTraceExporter() = delete;
        PerfettoTraceExporter(Codec::DataModel<TTraceReader>& dataModel, const PerfettoTraceExporterOptions& options = PerfettoTraceExporterOptions{})
            : TraceExporterBase<TTraceReader>(dataModel)
            , m_options(options)
        {
        }

        void Export(const std::filesystem::path& archiveFilePath);

    private:
        PerfettoTraceExporterOptions m_options;

        using Annotation = PerfettoTraceWriter::Annotation;
        using EventId = PerfettoTraceWriter::EventId;
        using TrackUuid = PerfettoTraceWriter::TrackUuid;

        inline uint64_t ExportTimestampNsec(uint64_t timestampUtcNsec) const
        {
            return timestampUtcNsec >= m_dataModel.m_observationStartUtcNsec ? (timestampUtcNsec - m_dataModel.m_observationStartUtcNsec) : 0;
        }

        std::string GetNodeId() const
        {
            return !m_dataModel.systemInfo.hostname.empty()
                     ? m_dataModel.systemInfo.hostname
                     : GetHostName();
        }

        static bool InsertManifestNode(
            YamlCodec::DataNode& mapping,
            const char* pKey,
            YamlCodec::DataNode&& value)
        {
            return mapping.MapInsert(std::string(pKey), std::move(value));
        }

        bool BuildPartialManifest(const std::string& traceName, std::string& manifestStr) const
        {
            manifestStr.clear();
            YamlCodec::DataNode file;
            YamlCodec::DataNode path;
            path.SetString(traceName);
            const bool pathInserted = InsertManifestNode(file, "path", std::move(path));
            if (!pathInserted)
            {
                return false;
            }

            YamlCodec::DataNode machine;
            YamlCodec::DataNode machineName;
            machineName.SetString(GetNodeId());
            const bool machineNameInserted = InsertManifestNode(machine, "name", std::move(machineName));
            if (!machineNameInserted)
            {
                return false;
            }
            const bool machineInserted = InsertManifestNode(file, "machine", std::move(machine));
            if (!machineInserted)
            {
                return false;
            }

            YamlCodec::DataNode files;
            files.UpdateSequence([&](YamlCodec::Vector<YamlCodec::SequenceEntry>& entries) {
                entries.PushBackDefault().value = std::move(file);
            });

            YamlCodec::DataNode manifest;
            YamlCodec::DataNode version;
            version.SetUInt64(1);
            const bool versionInserted = InsertManifestNode(manifest, "version", std::move(version));
            if (!versionInserted)
            {
                return false;
            }
            const bool filesInserted = InsertManifestNode(manifest, "files", std::move(files));
            if (!filesInserted)
            {
                return false;
            }

            YamlCodec::DataNode root;
            const bool manifestInserted = InsertManifestNode(root, "perfetto_manifest", std::move(manifest));
            if (!manifestInserted)
            {
                return false;
            }

            YamlCodec::WriteOptions writeOptions;
            writeOptions.prettyJson = true;
            writeOptions.prettyJsonFirstRootKeyInline = true;
            auto sink = [&](const uint8_t* pData, size_t size) {
                const char* pChars = reinterpret_cast<const char*>(pData);
                manifestStr.insert(manifestStr.end(), pChars, pChars + size);
                return true;
            };
            const bool manifestWritten = YamlCodec::WriteJson(root, writeOptions, sink);
            if (!manifestWritten)
            {
                return false;
            }
            manifestStr += '\n';
            return true;
        }

        bool WriteArchive(const std::filesystem::path& archiveFilePath)
        {
            std::ofstream output(archiveFilePath, std::ios::binary | std::ios::trunc);
            if (!output.is_open())
            {
                return false;
            }

            std::string traceName = archiveFilePath.stem().filename().string();
            if (std::filesystem::path(traceName).extension() != ".perfetto-trace")
            {
                traceName += ".perfetto-trace";
            }
            std::string manifestStr;
            const bool manifestBuilt = BuildPartialManifest(traceName, manifestStr);
            if (!manifestBuilt)
            {
                return false;
            }
            std::vector<uint8_t> stagingBuffer(kArchiveBufferSize, uint8_t{0});

            ZipStreamingCompressor compressor;
            const bool initialized = compressor.Initialize(2);
            if (!initialized)
            {
                return false;
            }

            auto acquireBuffer = [&]() {
                ZipStreamingCompressor::OutputBuffer outputBuffer;
                outputBuffer.pData = stagingBuffer.data();
                outputBuffer.capacity = stagingBuffer.size();
                outputBuffer.slot = 0;
                return outputBuffer;
            };
            auto submitBuffer = [&](const ZipStreamingCompressor::OutputBuffer& outputBuffer, size_t size) {
                if (size != 0)
                {
                    output.write(
                        reinterpret_cast<const char*>(outputBuffer.pData),
                        static_cast<std::streamsize>(size));
                }
                return output.good();
            };
            auto writeArchive = [&](auto& archive) {
                {
                    auto entry = archive.OpenEntry(
                        traceName,
                        ZipStreamingCompressor::CompressionMethod::Deflate,
                        kArchiveBufferSize);
                    auto sink = [&](const uint8_t* pData, size_t size) {
                        return entry.Write(pData, size);
                    };
                    const bool perfettoSucceeded = m_writer.Write(sink);
                    if (!perfettoSucceeded || !entry.Close())
                    {
                        return false;
                    }
                }
                const bool manifestSucceeded = archive.WriteStoredEntry(
                    "perfetto_manifest.json",
                    reinterpret_cast<const uint8_t*>(manifestStr.data()),
                    manifestStr.size());
                if (!manifestSucceeded)
                {
                    return false;
                }
                return true;
            };

            const bool writeSucceeded = compressor.WriteArchive(
                acquireBuffer,
                submitBuffer,
                writeArchive);
            output.flush();
            return writeSucceeded && output.good();
        }

        static constexpr size_t kArchiveBufferSize = 64 * 1024;
        static constexpr int32_t kGroupOrderSystemInfo = 0;
        static constexpr int32_t kGroupOrderGpu = 20000;
        static constexpr int32_t kGroupOrderEventTrace = 40000;
        static constexpr int32_t kGroupOrderOverview = 60000;
        static constexpr int32_t kGroupOrderStreams = 80000;

        TrackUuid GetTraceRootTrack()
        {
            if (m_traceRootTrack != 0)
            {
                return m_traceRootTrack;
            }
            m_traceRootTrack = m_writer.CreateNamedTrack("NvTrace - " + GetNodeId(), 0);
            m_writer.SetTrackChildOrdering(m_traceRootTrack, perfetto::protos::TrackDescriptor::EXPLICIT);
            return m_traceRootTrack;
        }

        TrackUuid GetOrCreateNamedGroup(TrackUuid parentUuid, const std::string& name, int32_t siblingOrderRank = 0)
        {
            const auto key = std::make_pair(parentUuid, name);
            const auto itrTrack = m_namedChildTrack.lower_bound(key);
            if (itrTrack != m_namedChildTrack.end() && itrTrack->first == key)
            {
                return itrTrack->second;
            }
            const TrackUuid uuid = m_writer.CreateNamedTrack(name, parentUuid);
            m_writer.SetTrackChildOrdering(uuid, perfetto::protos::TrackDescriptor::CHRONOLOGICAL);
            m_writer.SetTrackSiblingOrderRank(uuid, siblingOrderRank);
            m_namedChildTrack.emplace_hint(itrTrack, key, uuid);
            return uuid;
        }

        TrackUuid GetGpuDiagnosticsGroup(TrackUuid gpuTrack)
        {
            return GetOrCreateNamedGroup(gpuTrack, "Provider Diagnostics");
        }

        TrackUuid GetSubTrack(TrackUuid groupTrack, const std::string& laneName)
        {
            const auto key = std::make_pair(groupTrack, laneName);
            const auto itrTrack = m_laneTrack.lower_bound(key);
            if (itrTrack != m_laneTrack.end() && itrTrack->first == key)
            {
                return itrTrack->second;
            }
            const TrackUuid uuid = m_writer.CreateNamedTrack(laneName, groupTrack);
            m_laneTrack.emplace_hint(itrTrack, key, uuid);
            return uuid;
        }

        /// siblingOrderRank is honoured for a counter track that is also a row, which needs to sit where the
        /// set placed it rather than wherever the writer happened to emit it.
        TrackUuid GetCounterSubTrack(TrackUuid groupTrack, const std::string& laneName, const std::string& unitName, const std::optional<int32_t> siblingOrderRank = std::nullopt)
        {
            const auto key = std::make_pair(groupTrack, laneName);
            const auto itrTrack = m_counterTrack.lower_bound(key);
            if (itrTrack != m_counterTrack.end() && itrTrack->first == key)
            {
                return itrTrack->second;
            }
            const TrackUuid uuid = m_writer.CreateCounterTrack(laneName, groupTrack, unitName);
            if (siblingOrderRank.has_value())
            {
                m_writer.SetTrackSiblingOrderRank(uuid, *siblingOrderRank);
            }
            m_counterTrack.emplace_hint(itrTrack, key, uuid);
            return uuid;
        }

        std::string GetNameForNvtxTraceCpuTraceRow(uint64_t uniqueThreadId, uint32_t domainCliqueIndex, const Codec::NvtxTraceData::DomainCliqueData& domainClique, const Codec::NvtxTraceData& nvtxTrace, bool isCudaThread = false)
        {
            const char* const threadLabel = isCudaThread ? "CUDA Thread" : "Thread";
            const std::vector<uint32_t>& domainIndices = domainClique.domainIndices;
            if (domainIndices.empty() || (domainIndices.size() == 1 && domainIndices[0] == 0))
            {
                return SprintfToString("%s %" PRIu64, threadLabel, uniqueThreadId);
            }

            if (domainIndices.size() == 1)
            {
                return SprintfToString("%s %" PRIu64 " (%s)", threadLabel, uniqueThreadId, m_dataModel.GetNvtxTraceDomainName(nvtxTrace, domainIndices[0]).c_str());
            }

            return SprintfToString("%s %" PRIu64 " (NVTX Group %" PRIu32 ")", threadLabel, uniqueThreadId, domainCliqueIndex + 1);
        }

        std::string GetNvtxTraceDomainExportName(const Codec::NvtxTraceData& nvtxTrace, uint32_t domainIndex)
        {
            return domainIndex ? m_dataModel.GetNvtxTraceDomainName(nvtxTrace, domainIndex) : "Default";
        }

        std::string GetNvtxTraceCategory(const Codec::NvtxTraceData& nvtxTrace, uint32_t domainIndex, const char* pBaseCategory)
        {
            return SprintfToString("%s: %s", pBaseCategory, GetNvtxTraceDomainExportName(nvtxTrace, domainIndex).c_str());
        }

        // Threads that executed host callbacks are labeled "CUDA Thread" on CPU rows.
        inline std::unordered_set<uint64_t> CollectCudaThreadIds(const Codec::CudaApiTraceData& cudaApiTrace)
        {
            std::unordered_set<uint64_t> cudaThreadIds;
            for (const auto& hostCallbackInstance : cudaApiTrace.hostCallbackInstances)
            {
                if (hostCallbackInstance.executingUniqueThreadId)
                {
                    cudaThreadIds.insert(hostCallbackInstance.executingUniqueThreadId);
                }
            }
            return cudaThreadIds;
        }

        // Maps thread to CPU row names that it should be represented on.
        // * One row per NVTX domain clique that has events on this thread.
        // * CUDA API spans are replicated onto every row so each NVTX row shows the CUDA activity its ranges enclose.
        // * A thread with no NVTX events gets a single row.
        inline std::vector<std::string> GetCpuTraceRowNames(uint64_t uniqueThreadId, bool isCudaThread, const Codec::NvtxTraceData& nvtxTrace)
        {
            std::vector<std::string> rowNames;
            const auto addRowName = [&rowNames](std::string rowName) {
                if (std::find(rowNames.begin(), rowNames.end(), rowName) == rowNames.end())
                {
                    rowNames.push_back(std::move(rowName));
                }
            };

            for (uint32_t domainCliqueIndex = 0; domainCliqueIndex < nvtxTrace.domainCliques.size(); domainCliqueIndex++)
            {
                const Codec::NvtxTraceData::DomainCliqueData& domainClique = nvtxTrace.domainCliques[domainCliqueIndex];
                const auto threadItr = domainClique.threads.find(uniqueThreadId);
                if (threadItr == domainClique.threads.end())
                {
                    continue;
                }

                if (threadItr->second.marks.empty() && !threadItr->second.hasObservationTimeRange)
                {
                    continue;
                }

                addRowName(GetNameForNvtxTraceCpuTraceRow(uniqueThreadId, domainCliqueIndex, domainClique, nvtxTrace, isCudaThread));
            }

            if (rowNames.empty())
            {
                addRowName(SprintfToString("%s %" PRIu64, isCudaThread ? "CUDA Thread" : "Thread", uniqueThreadId));
            }

            return rowNames;
        }

        inline void AddApiCorrelationFlowSource(uint64_t apiCorrelationId, NVTRACE_LUID processLuid, EventId eventId)
        {
            m_apiCorrelationToFlowSourceEventIdsMap[std::make_tuple(apiCorrelationId, processLuid)].push_back(eventId);
        }

        inline std::vector<uint64_t> CreateApiCorrelationFlowIds(uint64_t apiCorrelationId, NVTRACE_LUID processLuid)
        {
            std::vector<uint64_t> flowIds;
            auto tuple = std::make_tuple(apiCorrelationId, processLuid);
            auto it = m_apiCorrelationToFlowSourceEventIdsMap.find(tuple);
            if (it != m_apiCorrelationToFlowSourceEventIdsMap.end())
            {
                for (EventId sourceEventId : it->second)
                {
                    const uint64_t flowId = m_nextAvailableFlowId++;
                    m_writer.AddFlowId(sourceEventId, flowId);
                    flowIds.push_back(flowId);
                }
            }
            return flowIds;
        }

        inline void AddHwCorrelationFlowSource(uint64_t hwCorrelationId, EventId eventId)
        {
            m_hwCorrelationToFlowSourceEventIdsMap[hwCorrelationId].push_back(eventId);
        }

        inline std::vector<uint64_t> CreateHwCorrelationFlowIds(uint64_t hwCorrelationId)
        {
            std::vector<uint64_t> flowIds;
            auto it = m_hwCorrelationToFlowSourceEventIdsMap.find(hwCorrelationId);
            if (it != m_hwCorrelationToFlowSourceEventIdsMap.end())
            {
                for (EventId sourceEventId : it->second)
                {
                    const uint64_t flowId = m_nextAvailableFlowId++;
                    m_writer.AddFlowId(sourceEventId, flowId);
                    flowIds.push_back(flowId);
                }
            }
            return flowIds;
        }

        inline void AddNvtxTracePushPopRangeFlowSource(
            NVTRACE_LUID processLuid,
            const Codec::NvtxTraceData::PushPopRangeData* pPushPopRange,
            EventId eventId)
        {
            m_nvtxTracePushPopRangeToFlowSourceEventIdsMap[std::make_tuple(pPushPopRange, processLuid)].push_back(eventId);
        }

        inline std::vector<uint64_t> CreateNvtxTracePushPopRangeFlowIds(
            NVTRACE_LUID processLuid,
            const Codec::NvtxTraceData::PushPopRangeData* pPushPopRange)
        {
            std::vector<uint64_t> flowIds;
            auto key = std::make_tuple(pPushPopRange, processLuid);
            auto it = m_nvtxTracePushPopRangeToFlowSourceEventIdsMap.find(key);
            if (it != m_nvtxTracePushPopRangeToFlowSourceEventIdsMap.end())
            {
                for (EventId sourceEventId : it->second)
                {
                    const uint64_t flowId = m_nextAvailableFlowId++;
                    m_writer.AddFlowId(sourceEventId, flowId);
                    flowIds.push_back(flowId);
                }
            }
            return flowIds;
        }

        inline uint64_t AddDeviceGraphLaunchFlowId(NVTRACE_LUID processLuid, size_t graphLaunchIdx)
        {
            auto& flowIds = m_deviceGraphLaunchToFlowIdsMap[std::make_tuple(processLuid, graphLaunchIdx)];
            flowIds.push_back(m_nextAvailableFlowId++);
            return flowIds.back();
        }

        inline const std::vector<uint64_t>& GetDeviceGraphLaunchFlowIds(NVTRACE_LUID processLuid, size_t graphLaunchIdx) const
        {
            static const std::vector<uint64_t> emptyFlowIds;
            auto it = m_deviceGraphLaunchToFlowIdsMap.find(std::make_tuple(processLuid, graphLaunchIdx));
            if (it != m_deviceGraphLaunchToFlowIdsMap.end())
            {
                return it->second;
            }

            return emptyFlowIds;
        }

        // Returns 0 for eager commands, which correlate through their own API call instead.
        inline uint64_t GetGraphLaunchCorrelationIdForGpuCommand(const Codec::CudaApiTraceData::GpuCommand& gpuCommand, const Codec::CudaApiTraceData& cudaApiTrace)
        {
            switch (gpuCommand.commandType)
            {
                case Codec::CudaApiTraceData::GpuCommandType::GraphMemcpy:
                {
                    const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin* const pGraphMemcpyBegin = m_dataModel.GetGraphMemcpyBeginPayload(gpuCommand.recordLocation);
                    return pGraphMemcpyBegin ? pGraphMemcpyBegin->enclosingCorrelationId : 0;
                }
                case Codec::CudaApiTraceData::GpuCommandType::GraphMemset:
                {
                    const NvTrace_CUDA_ApiTrace_GraphMemsetBegin* const pGraphMemsetBegin = m_dataModel.GetGraphMemsetBeginPayload(gpuCommand.recordLocation);
                    return pGraphMemsetBegin ? pGraphMemsetBegin->enclosingCorrelationId : 0;
                }
                case Codec::CudaApiTraceData::GpuCommandType::MemOp:
                {
                    if (gpuCommand.graphNodeCorrelationId == UINT64_MAX)
                    {
                        return 0; // eager invocation
                    }
                    const NvTrace_CUDA_ApiTrace_MemOpBegin* const pMemOpBegin = m_dataModel.GetMemOpBeginPayload(gpuCommand.recordLocation, cudaApiTrace);
                    return pMemOpBegin ? pMemOpBegin->enclosingCorrelationId : 0;
                }
                default:
                    return 0;
            }
        }

        inline const std::map<uint64_t, uint64_t>* GetDeviceEventFlowIdMap(const Codec::DeviceEvent& deviceEvent)
        {
            if (!deviceEvent)
            {
                return nullptr;
            }

            const size_t hash = reinterpret_cast<size_t>(deviceEvent.GetEventPointer());
            auto [deviceEventMapItr, isNewMap] = m_deviceEventToTimestampToFlowIdMap.try_emplace(hash, std::map<uint64_t, uint64_t>());
            auto& flowIdMap = deviceEventMapItr->second;

            auto [flowIdItr, inserted] = flowIdMap.try_emplace(deviceEvent.startTimestampGpuNsec, 0);
            if (inserted)
            {
                flowIdItr->second = m_nextAvailableFlowId++;
            }
            return &flowIdMap;
        }

        inline void AddSystemInfo(TrackUuid groupTrack, const Codec::SystemInfo& systemInfo)
        {
            std::string name = "SystemInfo";
            const TrackUuid track = GetSubTrack(groupTrack, "System Info");

            std::vector<Annotation> annotations;
            annotations.push_back(m_writer.MakeString("Hostname", systemInfo.hostname));
            annotations.push_back(m_writer.MakeString("OS Version", systemInfo.osVersion));
            annotations.push_back(m_writer.MakeString("OS Family", EnumNameOsFamily(systemInfo.osFamily)));
            annotations.push_back(m_writer.MakeString("CPU Model", systemInfo.cpuModel));
            annotations.push_back(m_writer.MakeString("CPU Architecture", EnumNameCpuArch(systemInfo.cpuArch)));
            annotations.push_back(m_writer.MakeString("System Memory Total", FormatBytesToHumanReadable(systemInfo.sysMemTotal)));
            annotations.push_back(m_writer.MakeString("NVIDIA RM Kernel Driver Version", SprintfToString("%" PRIu32 ".%" PRIu32 ".%" PRIu32, systemInfo.nvidiaRmKernelDriverVersion.major, systemInfo.nvidiaRmKernelDriverVersion.minor, systemInfo.nvidiaRmKernelDriverVersion.patch)));
            if (systemInfo.nvidiaMrmKernelDriverVersion.major != 0 || systemInfo.nvidiaMrmKernelDriverVersion.minor != 0 || systemInfo.nvidiaMrmKernelDriverVersion.patch != 0)
            {
                annotations.push_back(m_writer.MakeString("NVIDIA MRM Kernel Driver Version", SprintfToString("%" PRIu32 ".%" PRIu32 ".%" PRIu32, systemInfo.nvidiaMrmKernelDriverVersion.major, systemInfo.nvidiaMrmKernelDriverVersion.minor, systemInfo.nvidiaMrmKernelDriverVersion.patch)));
            }

            m_writer.AddSlice(
                track,
                "system_info",
                name,
                ExportTimestampNsec(m_dataModel.m_observationStartUtcNsec),
                m_dataModel.m_observationEndUtcNsec - m_dataModel.m_observationStartUtcNsec,
                std::move(annotations));
        }

        inline void AddTraceInfo(TrackUuid groupTrack)
        {
            std::string name = "TraceInfo";
            const TrackUuid track = GetSubTrack(groupTrack, "System Info");

            std::vector<Annotation> annotations;
            annotations.push_back(m_writer.MakeString("Trace Begin Time (UTC)", FormatUtcNsecToHumanReadable(m_dataModel.m_observationStartUtcNsec)));
            annotations.push_back(m_writer.MakeString("Trace End Time (UTC)", FormatUtcNsecToHumanReadable(m_dataModel.m_observationEndUtcNsec)));
            annotations.push_back(m_writer.MakeString("Trace Begin Timestamp (Unix Epoch)", SprintfToString("%" PRIu64 " ns", m_dataModel.m_observationStartUtcNsec)));
            annotations.push_back(m_writer.MakeString("Trace End Timestamp (Unix Epoch)", SprintfToString("%" PRIu64 " ns", m_dataModel.m_observationEndUtcNsec)));
            annotations.push_back(m_writer.MakeString("Trace Duration", SprintfToString("%" PRIu64 " ns", m_dataModel.m_observationEndUtcNsec - m_dataModel.m_observationStartUtcNsec)));

            m_writer.AddSlice(
                track,
                "system_info",
                name,
                ExportTimestampNsec(m_dataModel.m_observationStartUtcNsec),
                m_dataModel.m_observationEndUtcNsec - m_dataModel.m_observationStartUtcNsec,
                std::move(annotations));
        }

        inline void AddGpuPartitionInfo(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuPartitionData& gpuPartition)
        {
            std::string name = "GPU" + std::to_string(deviceOrdinal);

            const uint64_t creationTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(gpuPartition.creationTimestampTsc);
            const uint64_t destructionTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(gpuPartition.destructionTimestampTsc);

            const uint64_t durationNsec = destructionTimestampUtcNsec - creationTimestampUtcNsec;

            const TrackUuid track = GetSubTrack(groupTrack, "System Info");

            std::vector<Annotation> annotations;
            annotations.push_back(m_writer.MakeString("UUID", ToString(gpuPartition.gpuUuid)));
            annotations.push_back(m_writer.MakeString("Chip Name", gpuPartition.info.chipName));
            annotations.push_back(m_writer.MakeString("Device Name", gpuPartition.info.deviceName));
            annotations.push_back(m_writer.MakeString("Driver Version", SprintfToString("%" PRIu32 ".%" PRIu32 ".%" PRIu32, gpuPartition.info.driverVersion.major, gpuPartition.info.driverVersion.minor, gpuPartition.info.driverVersion.patch)));
            annotations.push_back(m_writer.MakeString("Driver Type", EnumNameDriverType(gpuPartition.info.driverType)));
            annotations.push_back(m_writer.MakeString("Device Memory Size", FormatBytesToHumanReadable(gpuPartition.info.deviceMemorySize)));
            annotations.push_back(m_writer.MakeBool("Is Profiled", gpuPartition.info.isProfiled));

            std::vector<Annotation> supportedProviders;
            supportedProviders.reserve(gpuPartition.info.supportedProviders.size());
            for (const auto& supportedProvider : gpuPartition.info.supportedProviders)
            {
                supportedProviders.push_back(m_writer.MakeStringValue(GetProviderTypeName(GetProviderType(supportedProvider))));
            }
            annotations.push_back(m_writer.MakeArray("Supported Providers", std::move(supportedProviders)));

            std::vector<Annotation> attributes;
            attributes.reserve(gpuPartition.info.attributes.size());
            for (auto& attribute : gpuPartition.info.attributes)
            {
                const char* pName = GetCudaDevicePublicEnumString(static_cast<size_t>(attribute.key));
                if (pName)
                {
                    attributes.push_back(m_writer.MakeInt(pName, attribute.value));
                }
            }
            annotations.push_back(m_writer.MakeDictionary("Attributes", std::move(attributes)));

            m_writer.AddSlice(
                track,
                "system_info",
                name,
                ExportTimestampNsec(creationTimestampUtcNsec),
                durationNsec,
                std::move(annotations));
        }

        inline void AddNvTraceDiagnostic(TrackUuid groupTrack, const char* pDiagnosticsRowName, DiagnosticSeverity severity, const char* pComponentName, const char* pMessage, uint64_t startUtcNsec, uint64_t endUtcNsec, std::vector<Annotation> annotations)
        {
            const uint64_t durationNsec = endUtcNsec - startUtcNsec;
            const char* pSeverityName = GetNvTraceDiagnosticSeverity(severity);

            annotations.push_back(m_writer.MakeString("Severity", pSeverityName));
            annotations.push_back(m_writer.MakeString("Component", pComponentName));
            annotations.push_back(m_writer.MakeString("Message", pMessage));

            m_writer.AddSlice(
                GetSubTrack(groupTrack, pDiagnosticsRowName),
                "nvtrace_diagnostics",
                SprintfToString("%s: %s", pSeverityName, pComponentName),
                ExportTimestampNsec(startUtcNsec),
                durationNsec,
                std::move(annotations));
        }

        inline void AddCpuClockCalibration(TrackUuid groupTrack)
        {
            const auto& cpuClockCalibration = m_dataModel.cpuClockCalibration;

            if (cpuClockCalibration.calibrationPoints.empty())
            {
                NVTRACE_LOG_WRN(99, "No CPU clock calibration points");
                return;
            }

            const auto& calibrationPoint0 = cpuClockCalibration.calibrationPoints[0];

            NVTRACE_LOG_INF(99, "CPU initial calibration point; tscTick: %" PRIu64, calibrationPoint0.tscTick);
            NVTRACE_LOG_INF(99, "CPU initial calibration point; utcNsec: %" PRIu64, calibrationPoint0.utcNsec);

            const TrackUuid track = GetSubTrack(groupTrack, "CPU Clock Calibration");
            // Emit one complete slice per calibration interval (protobuf has no B/E phase pairs).
            m_dataModel.ForEachCpuClockCalibrationInterval([&](const TscRtcCalibrationPoint& calibrationPoint, const Codec::ClockCalibrationInterval& interval) {
                std::vector<Annotation> annotations;
                annotations.push_back(m_writer.MakeUint("tscTick", calibrationPoint.tscTick));
                annotations.push_back(m_writer.MakeUint("utcNsec", calibrationPoint.utcNsec));
                annotations.push_back(m_writer.MakeDouble("tscToUtcFactor", calibrationPoint.tscToUtcFactor));
                annotations.push_back(m_writer.MakeDouble("utcToTscFactor", calibrationPoint.utcToTscFactor));
                annotations.push_back(m_writer.MakeUint("intervalStartUtcNsec", interval.startUtcNsec));
                annotations.push_back(m_writer.MakeUint("intervalEndUtcNsec", interval.endUtcNsec));
                annotations.push_back(m_writer.MakeBool("isExtrapolated", interval.isExtrapolated));

                const char* pName = interval.isExtrapolated ? "Calibration Interval (Extrapolated)" : "Calibration Interval";
                m_writer.AddSlice(track, "cpu_clock", pName, ExportTimestampNsec(interval.clampedStartUtcNsec), interval.clampedEndUtcNsec - interval.clampedStartUtcNsec, std::move(annotations));
            });
        }

        inline void AddGpuClockCalibration(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuClockCalibrationData& gpuClockCalibrationData)
        {
            NVTRACE_LOG_INF(50, "GPU %zu; number of clock calibration points: %zu", deviceOrdinal, gpuClockCalibrationData.calibrationPoints.size());
            if (gpuClockCalibrationData.calibrationPoints.empty())
            {
                NVTRACE_LOG_WRN(90, "GPU %zu; no GPU clock calibration points", deviceOrdinal);
                return;
            }

            const auto& calibrationPoint0 = gpuClockCalibrationData.calibrationPoints.front();

            NVTRACE_LOG_INF(99, "GPU %zu; initial calibration point; tscTick: %" PRIu64, deviceOrdinal, calibrationPoint0.tscTick);
            NVTRACE_LOG_INF(99, "GPU %zu; initial calibration point; gpuNsec: %" PRIu64, deviceOrdinal, calibrationPoint0.gpuNsec);

            const TrackUuid track = GetSubTrack(groupTrack, "GPU Clock Calibration");
            // Emit one complete slice per calibration interval (protobuf has no B/E phase pairs).
            m_dataModel.ForEachGpuClockCalibrationInterval(gpuClockCalibrationData, [&](const TscGpuCalibrationPoint& calibrationPoint, const Codec::ClockCalibrationInterval& interval) {
                std::vector<Annotation> annotations;
                annotations.push_back(m_writer.MakeUint("tscTick", calibrationPoint.tscTick));
                annotations.push_back(m_writer.MakeUint("gpuNsec", calibrationPoint.gpuNsec));
                annotations.push_back(m_writer.MakeDouble("tscToGpuFactor", calibrationPoint.tscToGpuFactor));
                annotations.push_back(m_writer.MakeDouble("gpuToTscFactor", calibrationPoint.gpuToTscFactor));
                annotations.push_back(m_writer.MakeUint("intervalStartUtcNsec", interval.startUtcNsec));
                annotations.push_back(m_writer.MakeUint("intervalEndUtcNsec", interval.endUtcNsec));
                annotations.push_back(m_writer.MakeBool("isExtrapolated", interval.isExtrapolated));

                const char* pName = interval.isExtrapolated ? "Calibration Interval (Extrapolated)" : "Calibration Interval";
                m_writer.AddSlice(track, "gpu_clock", pName, ExportTimestampNsec(interval.clampedStartUtcNsec), interval.clampedEndUtcNsec - interval.clampedStartUtcNsec, std::move(annotations));
            });
        }

        // Collects flattened debug annotations and correlation ids for a CUDA API
        // call's inner events (kernel launches, memcpy/memset, graph launches).
        struct InnerEventStats
        {
            std::vector<Annotation> annotations;
            std::vector<uint64_t> apiCorrelationIds;
            std::vector<uint64_t> hwComputeCorrelationIds;
            uint64_t graphLaunchApiCorrelationId = 0;
        };

        InnerEventStats CollectInnerEventStats(const std::vector<uint64_t>& innerRecordLocations, int processId, const Codec::CudaApiTraceData& cudaApiTrace, const std::string& apiFunctionName)
        {
            InnerEventStats stats;

            for (size_t innerRecordLocationIndex = 0; innerRecordLocationIndex < innerRecordLocations.size(); ++innerRecordLocationIndex)
            {
                const uint64_t innerRecordLocation = innerRecordLocations[innerRecordLocationIndex];
                const RecordMetadata recordMetadata = m_dataModel.GetRecordMetadata(innerRecordLocation);
                const NvTrace_CUDA_ApiTrace_RecordType recordType = static_cast<NvTrace_CUDA_ApiTrace_RecordType>(recordMetadata.recordType);
                const bool isCore =
                    (recordType == NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch && apiFunctionName.find("LaunchKernel") != std::string::npos)
                    || (recordType == NvTrace_CUDA_ApiTrace_RecordType_GraphLaunch && apiFunctionName.find("GraphLaunch") != std::string::npos)
                    || (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin && apiFunctionName.find("Memcpy") != std::string::npos)
                    || (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin && apiFunctionName.find("Memset") != std::string::npos)
                    || (recordType == NvTrace_CUDA_ApiTrace_RecordType_HostCallbackLaunch && apiFunctionName.find("HostFunc") != std::string::npos);
                const std::string prefix = innerRecordLocations.size() > 1
                                             ? SprintfToString("%s[%zu].", isCore ? ".core" : "ancillary", innerRecordLocationIndex)
                                             : SprintfToString("%s.", isCore ? ".core" : "ancillary");

                if (recordType == NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch)
                {
                    const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch* pImmediateKernelLaunchEvent = m_dataModel.GetImmediateKernelLaunchPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pImmediateKernelLaunchEvent, 90, "CUDA API trace, process %d; immediate kernel launch payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));
                    const CuFunctionNames functionNames = GetCuFunctionNames(cudaApiTrace, pImmediateKernelLaunchEvent->cufunctionId);

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pImmediateKernelLaunchEvent->custreamId));
                    stats.annotations.push_back(m_writer.MakeString(prefix + "kernelName", functionNames.pDemangledName));
                    stats.annotations.push_back(m_writer.MakeString(prefix + "mangledName", functionNames.pMangledName));
                    stats.annotations.push_back(m_writer.MakeString(prefix + ".function", "Immediate Kernel Launch"));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "gridDim.x", pImmediateKernelLaunchEvent->gridDim_x));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "gridDim.y", pImmediateKernelLaunchEvent->gridDim_y));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "gridDim.z", pImmediateKernelLaunchEvent->gridDim_z));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "blockDim.x", pImmediateKernelLaunchEvent->blockDim_x));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "blockDim.y", pImmediateKernelLaunchEvent->blockDim_y));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "blockDim.z", pImmediateKernelLaunchEvent->blockDim_z));
                    // For now this will suffice to match HW kernel intervals to their CUDA API launch events
                    //   (ignoring possible aliasing while limited to 30 meaningful bits); in a future iteration
                    //   the data model will have already correlated them in its organization phase.
                    uint64_t apiCorrelationId = recordMetadata.correlationId;
                    // Note that this suggests that logic to convert an API correlation ID to a kernel one will
                    //   need to be visible in the data model; hacking in for now.
                    uint64_t hwComputeCorrelationId = (static_cast<uint32_t>(apiCorrelationId) & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK) | NVTRACE_CUDA_HW_CORRELATION_ID_IMMEDIATE_MSBS;
                    stats.apiCorrelationIds.push_back(apiCorrelationId);
                    stats.hwComputeCorrelationIds.push_back(hwComputeCorrelationId);
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "apiCorrelationId", apiCorrelationId));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "hwComputeCorrelationId", hwComputeCorrelationId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_GraphLaunch)
                {
                    const NvTrace_CUDA_ApiTrace_GraphLaunch* pGraphLaunchEvent = m_dataModel.GetGraphLaunchPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pGraphLaunchEvent, 90, "CUDA API trace, process %d; graph launch payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pGraphLaunchEvent->custreamId));
                    stats.annotations.push_back(m_writer.MakeString(prefix + ".function", "Graph Kernel Launch"));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "graphExecId", pGraphLaunchEvent->cugraphExecId));
                    uint64_t apiCorrelationId = recordMetadata.correlationId;
                    stats.apiCorrelationIds.push_back(apiCorrelationId);
                    if (isCore)
                    {
                        stats.graphLaunchApiCorrelationId = apiCorrelationId;
                    }
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "apiCorrelationId", apiCorrelationId));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "hwGraphLaunchCorrelationId", (static_cast<uint32_t>(apiCorrelationId) & NVTRACE_CUDA_HW_CORRELATION_ID_LSB_MASK) | NVTRACE_CUDA_HW_CORRELATION_ID_GRAPH_LAUNCH_HOST_MSBS));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin)
                {
                    const NvTrace_CUDA_ApiTrace_MemcpyBegin* pMemcpyBeginEvent = m_dataModel.GetMemcpyBeginPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pMemcpyBeginEvent, 90, "CUDA API trace, process %d; memcpy begin payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));
                    const bool isAsync = (pMemcpyBeginEvent->memcpyFlags & NvTrace_CUDA_ApiTrace_MemcpyFlags_IsAsync) != 0;
                    const bool isP2P = (pMemcpyBeginEvent->memcpyFlags & NvTrace_CUDA_ApiTrace_MemcpyFlags_IsP2P) != 0;

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pMemcpyBeginEvent->custreamId));
                    stats.annotations.push_back(m_writer.MakeString(prefix + ".function", "Memory Copy"));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "byteCount", pMemcpyBeginEvent->byteCount));
                    stats.annotations.push_back(m_writer.MakeString(prefix + "srcAddressKind", GetAddressKindName(pMemcpyBeginEvent->srcAddressKind)));
                    stats.annotations.push_back(m_writer.MakeString(prefix + "dstAddressKind", GetAddressKindName(pMemcpyBeginEvent->dstAddressKind)));
                    stats.annotations.push_back(m_writer.MakePointer(prefix + "srcDevPtr", static_cast<uint64_t>(pMemcpyBeginEvent->srcDevPtr)));
                    stats.annotations.push_back(m_writer.MakePointer(prefix + "dstDevPtr", static_cast<uint64_t>(pMemcpyBeginEvent->dstDevPtr)));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "width", pMemcpyBeginEvent->width));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "height", pMemcpyBeginEvent->height));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "depth", pMemcpyBeginEvent->depth));
                    stats.annotations.push_back(m_writer.MakeBool(prefix + "isAsync", isAsync));
                    stats.annotations.push_back(m_writer.MakeBool(prefix + "isP2P", isP2P));
                    // correlation IDs
                    uint64_t apiCorrelationId = recordMetadata.correlationId;
                    stats.apiCorrelationIds.push_back(apiCorrelationId);
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "apiCorrelationId", apiCorrelationId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin)
                {
                    const NvTrace_CUDA_ApiTrace_MemsetBegin* pMemsetBeginEvent = m_dataModel.GetMemsetBeginPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pMemsetBeginEvent, 90, "CUDA API trace, process %d; memset begin payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pMemsetBeginEvent->custreamId));
                    stats.annotations.push_back(m_writer.MakeString(prefix + ".function", "Memory Set"));
                    stats.annotations.push_back(m_writer.MakePointer(prefix + "devPtr", static_cast<uint64_t>(pMemsetBeginEvent->devPtr)));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "byteCount", pMemsetBeginEvent->byteCount));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "value", pMemsetBeginEvent->value));
                    // correlation IDs
                    uint64_t apiCorrelationId = recordMetadata.correlationId;
                    stats.apiCorrelationIds.push_back(apiCorrelationId);
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "apiCorrelationId", apiCorrelationId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_HostCallbackLaunch)
                {
                    const NvTrace_CUDA_ApiTrace_HostCallbackLaunch* pHostCallbackLaunchEvent = m_dataModel.GetHostCallbackLaunchPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pHostCallbackLaunchEvent, 90, "CUDA API trace, process %d; host callback launch payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pHostCallbackLaunchEvent->custreamId));
                    stats.annotations.push_back(m_writer.MakeString(prefix + ".function", "Host Callback Launch"));
                    stats.annotations.push_back(m_writer.MakePointer(prefix + "callback", static_cast<uint64_t>(pHostCallbackLaunchEvent->callback)));
                    stats.annotations.push_back(m_writer.MakePointer(prefix + "userData", static_cast<uint64_t>(pHostCallbackLaunchEvent->userData)));
                    // correlation IDs
                    uint64_t apiCorrelationId = recordMetadata.correlationId;
                    stats.apiCorrelationIds.push_back(apiCorrelationId);
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "apiCorrelationId", apiCorrelationId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackLaunch)
                {
                    const NvTrace_CUDA_ApiTrace_GraphHostCallbackLaunch* pGraphHostCallbackLaunchEvent = m_dataModel.GetGraphHostCallbackLaunchPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pGraphHostCallbackLaunchEvent, 90, "CUDA API trace, process %d; graph host callback launch payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pGraphHostCallbackLaunchEvent->custreamId));
                    stats.annotations.push_back(m_writer.MakeString(prefix + ".function", "Graph Host Callback Launch"));
                    stats.annotations.push_back(m_writer.MakeUint(prefix + "graphNodeCorrelationId", pGraphHostCallbackLaunchEvent->graphNodeCorrelationId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyBegin)
                {
                    const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin* pGraphMemcpyBeginEvent = m_dataModel.GetGraphMemcpyBeginPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pGraphMemcpyBeginEvent, 90, "CUDA API trace, process %d; graph memcpy begin payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pGraphMemcpyBeginEvent->custreamId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetBegin)
                {
                    const NvTrace_CUDA_ApiTrace_GraphMemsetBegin* pGraphMemsetBeginEvent = m_dataModel.GetGraphMemsetBeginPayload(innerRecordLocation);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pGraphMemsetBeginEvent, 90, "CUDA API trace, process %d; graph memset begin payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pGraphMemsetBeginEvent->custreamId));
                }
                else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemOpBegin)
                {
                    const NvTrace_CUDA_ApiTrace_MemOpBegin* pMemOpBeginEvent = m_dataModel.GetMemOpBeginPayload(innerRecordLocation, cudaApiTrace);
                    NVTRACE_LOG_WRN_CONTINUE_IF(!pMemOpBeginEvent, 90, "CUDA API trace, process %d; memop begin payload not found (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu recordType=%u)", processId, innerRecordLocation, innerRecordLocationIndex, static_cast<unsigned>(recordType));

                    // parameters
                    stats.annotations.push_back(m_writer.MakeUint("streamId", pMemOpBeginEvent->custreamId));
                }
                else
                {
                    NVTRACE_LOG_WRN(90, "CUDA API trace, process %d; unknown record type %u (innerRecordLocation=%" PRIu64 " innerRecordLocationIndex=%zu)", processId, static_cast<unsigned>(recordType), innerRecordLocation, innerRecordLocationIndex);
                }
            }

            return stats;
        }

        inline void AddCudaApiTraceEvents(TrackUuid overviewGroup, TrackUuid streamsGroup, int processId, NVTRACE_LUID processLuid, const Codec::CudaApiTraceData& cudaApiTrace, const Codec::NvtxTraceData& nvtxTrace, const std::unordered_set<uint64_t>& cudaThreadIds)
        {
            if (cudaApiTrace.providerLuid == NVTRACE_NIL_LUID)
            {
                NVTRACE_LOG_WRN(90, "CUDA API trace, process %d; no CUDA API Trace provider", processId);
                return;
            }

            // TODO: These two lambdas could potentially be squashed into a single one that takes an ApiCall& and a function name string
            auto addRuntimeApiCallEvent = [&](const Codec::CudaApiTraceData::RuntimeApiCall& runtimeApiCall, TrackUuid groupTrack, const std::string& streamLaneName) {
                if (!runtimeApiCall.startRecordLocation || !runtimeApiCall.endRecordLocation)
                {
                    // TODO: Handle events with missing timestamps.
                    return;
                }

                const uint64_t startTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(runtimeApiCall.startTimestampTsc);
                const uint64_t endTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(runtimeApiCall.endTimestampTsc);

                const uint64_t durNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                const std::string apiFunctionName = GetRuntimeApiFunctionName(runtimeApiCall.runtimeApiFunctionId);
                InnerEventStats stats = CollectInnerEventStats(runtimeApiCall.innerRecordLocations, processId, cudaApiTrace, apiFunctionName);

                m_writer.AddSlice(
                    GetSubTrack(groupTrack, streamLaneName),
                    "cuda_runtime",
                    apiFunctionName,
                    ExportTimestampNsec(startTimestampUtcNsec),
                    durNsec,
                    std::move(stats.annotations));
            };

            auto addDriverApiCallEvent = [&](const Codec::CudaApiTraceData::DriverApiCall& driverApiCall, TrackUuid groupTrack, const std::string& streamLaneName, NVTRACE_LUID processLuid) {
                if (!driverApiCall.startRecordLocation || !driverApiCall.endRecordLocation)
                {
                    // TODO: Handle events with missing timestamps.
                    return;
                }

                const uint64_t startTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(driverApiCall.startTimestampTsc);
                const uint64_t endTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(driverApiCall.endTimestampTsc);

                const uint64_t durNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                const std::string apiFunctionName = GetDriverApiFunctionName(driverApiCall.driverApiFunctionId);
                InnerEventStats stats = CollectInnerEventStats(driverApiCall.innerRecordLocations, processId, cudaApiTrace, apiFunctionName);

                const EventId eventId = m_writer.AddSlice(
                    GetSubTrack(groupTrack, streamLaneName),
                    "cuda_driver",
                    apiFunctionName,
                    ExportTimestampNsec(startTimestampUtcNsec),
                    durNsec,
                    std::move(stats.annotations));

                if (stats.graphLaunchApiCorrelationId)
                {
                    m_writer.SetCorrelationId(eventId, stats.graphLaunchApiCorrelationId);
                }

                // Register every emitted timeline copy as a source. Each destination creates a
                // distinct flow ID so one API call can fan out to multiple GPU children.
                for (uint64_t apiCorrelationId : stats.apiCorrelationIds)
                {
                    AddApiCorrelationFlowSource(apiCorrelationId, processLuid, eventId);
                }
                for (uint64_t hwComputeCorrelationId : stats.hwComputeCorrelationIds)
                {
                    AddHwCorrelationFlowSource(hwComputeCorrelationId, eventId);
                }
            };

            // Group completed host-callback instances by executing thread once for the process.
            std::map<uint64_t, std::vector<const Codec::CudaApiTraceData::HostCallbackInstance*>> hostCallbackInstancesByExecutingThread;
            for (const auto& hostCallbackInstance : cudaApiTrace.hostCallbackInstances)
            {
                if (hostCallbackInstance.startTimestampTsc == Codec::InvalidTimestamp || hostCallbackInstance.endTimestampTsc == Codec::InvalidTimestamp)
                {
                    continue;
                }
                hostCallbackInstancesByExecutingThread[hostCallbackInstance.executingUniqueThreadId].push_back(&hostCallbackInstance);
            }

            // Add per-thread timeline events
            for (const auto& [uniqueThreadId, thread] : cudaApiTrace.threads)
            {
                const bool isCudaThread = cudaThreadIds.count(uniqueThreadId) != 0;
                for (const std::string& rowName : GetCpuTraceRowNames(uniqueThreadId, isCudaThread, nvtxTrace))
                {
                    for (const auto& runtimeApiCall : thread.runtimeApiCalls)
                    {
                        addRuntimeApiCallEvent(runtimeApiCall, overviewGroup, rowName);
                    }

                    for (const auto& driverApiCall : thread.driverApiCalls)
                    {
                        addDriverApiCallEvent(driverApiCall, overviewGroup, rowName, processLuid);
                    }
                }
            }

            // Add per-stream timeline rows
            if (m_options.enableStreamTimelines)
            {
                for (const auto& [custreamId, stream] : cudaApiTrace.streams)
                {
                    std::string const kStreamLaneName = "Stream " + std::to_string(custreamId);

                    for (const Codec::CudaApiTraceData::ApiCallRef& apiCallRef : stream.apiCallReferences)
                    {
                        auto threadItr = cudaApiTrace.threads.find(apiCallRef.uniqueThreadId);
                        if (threadItr == cudaApiTrace.threads.end())
                        {
                            continue;
                        }
                        const auto& thread = threadItr->second;

                        if (apiCallRef.runtimeApiCallIdx < thread.runtimeApiCalls.size())
                        {
                            const Codec::CudaApiTraceData::RuntimeApiCall& runtimeApiCall = thread.runtimeApiCalls[apiCallRef.runtimeApiCallIdx];
                            addRuntimeApiCallEvent(runtimeApiCall, streamsGroup, kStreamLaneName);
                        }

                        if (apiCallRef.driverApiCallIdx < thread.driverApiCalls.size())
                        {
                            const auto& driverApiCall = thread.driverApiCalls[apiCallRef.driverApiCallIdx];

                            addDriverApiCallEvent(driverApiCall, streamsGroup, kStreamLaneName, processLuid);
                        }
                    }
                }
            }

            // Host-callback spans on the executing thread's CPU rows.
            // Emitted after all driver/runtime API spans (overview + stream) so every launching
            // API source event is registered before destination flow IDs are created below.
            for (const auto& threadPair : cudaApiTrace.threads)
            {
                const uint64_t uniqueThreadId = threadPair.first;
                auto hostCallbackInstancesItr = hostCallbackInstancesByExecutingThread.find(uniqueThreadId);
                if (hostCallbackInstancesItr == hostCallbackInstancesByExecutingThread.end())
                {
                    continue;
                }

                const bool isCudaThread = cudaThreadIds.count(uniqueThreadId) != 0;
                for (const std::string& rowName : GetCpuTraceRowNames(uniqueThreadId, isCudaThread, nvtxTrace))
                {
                    for (const Codec::CudaApiTraceData::HostCallbackInstance* pHostCallbackInstance : hostCallbackInstancesItr->second)
                    {
                        const auto& hostCallbackInstance = *pHostCallbackInstance;

                        const uint64_t endTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(hostCallbackInstance.endTimestampTsc);
                        if (endTimestampUtcNsec == Codec::InvalidTimestamp || endTimestampUtcNsec < m_dataModel.m_observationStartUtcNsec)
                        {
                            continue;
                        }
                        const uint64_t startTimestampUtcNsecRaw = m_dataModel.ConvertTscTickToUtcNsec(hostCallbackInstance.startTimestampTsc);
                        if (startTimestampUtcNsecRaw == Codec::InvalidTimestamp)
                        {
                            continue;
                        }
                        const uint64_t startTimestampUtcNsec = std::max(m_dataModel.m_observationStartUtcNsec, startTimestampUtcNsecRaw);
                        const uint64_t durNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                        const bool isEager = (hostCallbackInstance.kind == Codec::CudaApiTraceData::HostCallbackKind::Eager);

                        std::vector<Annotation> annotations;
                        annotations.push_back(m_writer.MakeUint("streamId", hostCallbackInstance.custreamId));
                        annotations.push_back(m_writer.MakeUint("enclosingCorrelationId", hostCallbackInstance.enclosingCorrelationId));
                        if (isEager)
                        {
                            annotations.push_back(m_writer.MakeUint("launchApiCorrelationId", hostCallbackInstance.launchApiCorrelationId));
                            annotations.push_back(m_writer.MakePointer("callback", static_cast<uint64_t>(hostCallbackInstance.callback)));
                            annotations.push_back(m_writer.MakePointer("userData", static_cast<uint64_t>(hostCallbackInstance.userData)));
                        }
                        else
                        {
                            annotations.push_back(m_writer.MakeUint("graphNodeCorrelationId", hostCallbackInstance.graphNodeCorrelationId));
                        }

                        const uint64_t flowCorrelationId = isEager ? hostCallbackInstance.launchApiCorrelationId : hostCallbackInstance.enclosingCorrelationId;
                        std::vector<uint64_t> terminatingFlowIds = CreateApiCorrelationFlowIds(flowCorrelationId, processLuid);

                        const EventId eventId = m_writer.AddSlice(
                            GetSubTrack(overviewGroup, rowName),
                            isEager ? "cuda_host_callback" : "cuda_graph_host_callback",
                            isEager ? "Host Callback" : "Graph Host Callback",
                            ExportTimestampNsec(startTimestampUtcNsec),
                            durNsec,
                            std::move(annotations),
                            {},
                            std::move(terminatingFlowIds));
                        if (!isEager)
                        {
                            m_writer.SetCorrelationId(eventId, hostCallbackInstance.enclosingCorrelationId);
                        }
                    }
                }
            }
        }

        inline void AddGpuEventTimeline(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuPartitionData& gpuPartition, NVTRACE_LUID targetProcessLuid, uint32_t targetNvtxDomainIndex)
        {
            // NvTrace_CUDA_ApiTrace_MemcpyBegin or NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams
            auto formatMemcpyBeginEvent = [&](const auto* pMemcpyBeginEvent, NvTrace_CUDA_ApiTrace_CommandType commandType, std::vector<Annotation>& annotations) -> std::string {
                annotations.push_back(m_writer.MakeBool("isMemcpy", true));
                if (pMemcpyBeginEvent)
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(*pMemcpyBeginEvent)>, NvTrace_CUDA_ApiTrace_MemcpyBegin>)
                    {
                        annotations.push_back(m_writer.MakeUint("streamId", pMemcpyBeginEvent->custreamId));
                    }
                    const char* pSrcAddressKindName = GetAddressKindName(pMemcpyBeginEvent->srcAddressKind);
                    const char* pDstAddressKindName = GetAddressKindName(pMemcpyBeginEvent->dstAddressKind);
                    annotations.push_back(m_writer.MakeString("SrcAddressKind", pSrcAddressKindName));
                    annotations.push_back(m_writer.MakeString("DstAddressKind", pDstAddressKindName));
                    annotations.push_back(m_writer.MakeString("CommandType", CommandTypeToCString(commandType)));
                    const bool isAsync = (pMemcpyBeginEvent->memcpyFlags & NvTrace_CUDA_ApiTrace_MemcpyFlags_IsAsync) != 0;
                    const bool isP2P = (pMemcpyBeginEvent->memcpyFlags & NvTrace_CUDA_ApiTrace_MemcpyFlags_IsP2P) != 0;
                    return SprintfToString("Memcpy%c2%c(byteCount=%zu, width=%zu, height=%zu, depth=%zu, srcDevPtr=0x%016zx, dstDevPtr=0x%016zx, isAsync=%u, isP2P=%u)", pSrcAddressKindName[0], pDstAddressKindName[0], size_t(pMemcpyBeginEvent->byteCount), size_t(pMemcpyBeginEvent->width), size_t(pMemcpyBeginEvent->height), size_t(pMemcpyBeginEvent->depth), size_t(pMemcpyBeginEvent->srcDevPtr), size_t(pMemcpyBeginEvent->dstDevPtr), unsigned(isAsync), unsigned(isP2P));
                }
                NVTRACE_LOG_WRN_IF(!pMemcpyBeginEvent, 90, "CUDA API device trace; memcpy begin payload not found");
                return "Memcpy(?\?)"; // note: ?? is a trigraph
            };
            // NvTrace_CUDA_ApiTrace_MemsetBegin or NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams
            auto formatMemsetBeginEvent = [&](const auto* pMemsetBeginEvent, NvTrace_CUDA_ApiTrace_CommandType commandType, std::vector<Annotation>& annotations) -> std::string {
                annotations.push_back(m_writer.MakeBool("isMemset", true));
                if (pMemsetBeginEvent)
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(*pMemsetBeginEvent)>, NvTrace_CUDA_ApiTrace_MemsetBegin>)
                    {
                        annotations.push_back(m_writer.MakeUint("streamId", pMemsetBeginEvent->custreamId));
                    }
                    annotations.push_back(m_writer.MakeString("CommandType", CommandTypeToCString(commandType)));
                    return SprintfToString("Memset(byteCount=%zu, value=%u, devPtr=0x%016zx)", size_t(pMemsetBeginEvent->byteCount), pMemsetBeginEvent->value, size_t(pMemsetBeginEvent->devPtr));
                }
                NVTRACE_LOG_WRN_IF(!pMemsetBeginEvent, 90, "CUDA API device trace; memset begin payload not found");
                return "Memset(?\?)"; // note: ?? is a trigraph
            };

            auto formatMemOp = [](const std::string& details) {
                return SprintfToString("MemOp(%s)", details.c_str());
            };

            auto formatGpuCommand = [&](const Codec::CudaApiTraceData::GpuCommand& gpuCommand, const Codec::DeviceEventOrigin& deviceEventOrigin, std::vector<Annotation>& annotations) -> std::string {
                const Codec::CudaApiTraceData* pCudaApiTrace = deviceEventOrigin ? deviceEventOrigin.pCudaApiTrace : nullptr;

                switch (gpuCommand.commandType)
                {
                    case Codec::CudaApiTraceData::GpuCommandType::Invalid:
                        break;
                    case Codec::CudaApiTraceData::GpuCommandType::EagerMemcpy:
                    {
                        const NvTrace_CUDA_ApiTrace_MemcpyBegin* pMemcpyBeginEvent = m_dataModel.GetMemcpyBeginPayload(gpuCommand.recordLocation);
                        return formatMemcpyBeginEvent(pMemcpyBeginEvent, pMemcpyBeginEvent ? pMemcpyBeginEvent->commandType : NvTrace_CUDA_ApiTrace_CommandType_Invalid, annotations);
                    }
                    case Codec::CudaApiTraceData::GpuCommandType::EagerMemset:
                    {
                        const NvTrace_CUDA_ApiTrace_MemsetBegin* pMemsetBeginEvent = m_dataModel.GetMemsetBeginPayload(gpuCommand.recordLocation);
                        return formatMemsetBeginEvent(pMemsetBeginEvent, pMemsetBeginEvent ? pMemsetBeginEvent->commandType : NvTrace_CUDA_ApiTrace_CommandType_Invalid, annotations);
                    }
                    case Codec::CudaApiTraceData::GpuCommandType::GraphMemcpy:
                    {
                        const NvTrace_CUDA_ApiTrace_GraphMemcpyBegin* pGraphMemcpyBeginEvent = m_dataModel.GetGraphMemcpyBeginPayload(gpuCommand.recordLocation);
                        const auto* const pGraphNodeMemcpyParams = [&]() -> const NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams* {
                            if (!pGraphMemcpyBeginEvent)
                            {
                                return nullptr;
                            }
                            annotations.push_back(m_writer.MakeUint("streamId", pGraphMemcpyBeginEvent->custreamId));
                            if (!pCudaApiTrace)
                            {
                                return nullptr;
                            }
                            auto* pGraphNodeData = m_dataModel.FindCuGraphNode(*pCudaApiTrace, pGraphMemcpyBeginEvent->graphNodeCorrelationId);
                            if (!pGraphNodeData)
                            {
                                return nullptr;
                            }
                            const auto graphNodeType = m_dataModel.GetCudaApiTraceGraphNodeType(pGraphNodeData->recordLocation, *pCudaApiTrace);
                            if (graphNodeType != NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy)
                            {
                                return nullptr;
                            }
                            const auto* const pGraphNodeParams = m_dataModel.GetCudaApiTraceGraphNodeParams(pGraphNodeData->recordLocation, *pCudaApiTrace, graphNodeType);
                            return pGraphNodeParams ? &pGraphNodeParams->memcpy : nullptr;
                        }();
                        return formatMemcpyBeginEvent(pGraphNodeMemcpyParams, pGraphMemcpyBeginEvent ? pGraphMemcpyBeginEvent->commandType : NvTrace_CUDA_ApiTrace_CommandType_Invalid, annotations);
                    }
                    case Codec::CudaApiTraceData::GpuCommandType::GraphMemset:
                    {
                        const NvTrace_CUDA_ApiTrace_GraphMemsetBegin* pGraphMemsetBeginEvent = m_dataModel.GetGraphMemsetBeginPayload(gpuCommand.recordLocation);
                        const auto* const pGraphNodeMemsetParams = [&]() -> const NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams* {
                            if (!pGraphMemsetBeginEvent)
                            {
                                return nullptr;
                            }
                            annotations.push_back(m_writer.MakeUint("streamId", pGraphMemsetBeginEvent->custreamId));
                            if (!pCudaApiTrace)
                            {
                                return nullptr;
                            }
                            auto* pGraphNodeData = m_dataModel.FindCuGraphNode(*pCudaApiTrace, pGraphMemsetBeginEvent->graphNodeCorrelationId);
                            if (!pGraphNodeData)
                            {
                                return nullptr;
                            }
                            const auto graphNodeType = m_dataModel.GetCudaApiTraceGraphNodeType(pGraphNodeData->recordLocation, *pCudaApiTrace);
                            if (graphNodeType != NvTrace_CUDA_ApiTrace_GraphNodeType_Memset)
                            {
                                return nullptr;
                            }
                            const auto* const pGraphNodeParams = m_dataModel.GetCudaApiTraceGraphNodeParams(pGraphNodeData->recordLocation, *pCudaApiTrace, graphNodeType);
                            return pGraphNodeParams ? &pGraphNodeParams->memset : nullptr;
                        }();
                        return formatMemsetBeginEvent(pGraphNodeMemsetParams, pGraphMemsetBeginEvent ? pGraphMemsetBeginEvent->commandType : NvTrace_CUDA_ApiTrace_CommandType_Invalid, annotations);
                    }
                    case Codec::CudaApiTraceData::GpuCommandType::MemOp:
                    {
                        if (!pCudaApiTrace)
                        {
                            return formatMemOp(kUnknownData);
                        }
                        const NvTrace_CUDA_ApiTrace_MemOpBegin* const pMemOpBegin = m_dataModel.GetMemOpBeginPayload(gpuCommand.recordLocation, *pCudaApiTrace);
                        if (!pMemOpBegin)
                        {
                            return formatMemOp(kUnknownData);
                        }
                        annotations.push_back(m_writer.MakeUint("streamId", pMemOpBegin->custreamId));
                        const auto memOpType = static_cast<NvTrace_CUDA_ApiTrace_MemOpType>(pMemOpBegin->memOpType);
                        const NvTrace_CUDA_ApiTrace_MemOpParams* const pMemOpParams = m_dataModel.GetMemOpParamsPayload(gpuCommand.recordLocation, *pCudaApiTrace, memOpType);
                        switch (memOpType)
                        {
                            case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue32:
                                if (pMemOpParams)
                                {
                                    const auto& waitValue = pMemOpParams->waitValue;
                                    return formatMemOp(SprintfToString("WAIT_VALUE_32, address=%" PRIu64 ", value=%u", waitValue.address, static_cast<uint32_t>(waitValue.value)));
                                }
                                break;
                            case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue32:
                                if (pMemOpParams)
                                {
                                    const auto& writeValue = pMemOpParams->writeValue;
                                    return formatMemOp(SprintfToString("WRITE_VALUE_32, address=%" PRIu64 ", value=%u", writeValue.address, static_cast<uint32_t>(writeValue.value)));
                                }
                                break;
                            case NvTrace_CUDA_ApiTrace_MemOpType_FlushRemoteWrites:
                                return formatMemOp("FLUSH_REMOTE_WRITES");
                            case NvTrace_CUDA_ApiTrace_MemOpType_WaitValue64:
                                if (pMemOpParams)
                                {
                                    const auto& waitValue = pMemOpParams->waitValue;
                                    return formatMemOp(SprintfToString("WAIT_VALUE_64, address=%" PRIu64 ", value=%" PRIu64, waitValue.address, waitValue.value));
                                }
                                break;
                            case NvTrace_CUDA_ApiTrace_MemOpType_WriteValue64:
                                if (pMemOpParams)
                                {
                                    const auto& writeValue = pMemOpParams->writeValue;
                                    return formatMemOp(SprintfToString("WRITE_VALUE_64, address=%" PRIu64 ", value=%" PRIu64, writeValue.address, writeValue.value));
                                }
                                break;
                            case NvTrace_CUDA_ApiTrace_MemOpType_Barrier:
                                return formatMemOp("BARRIER");
                            case NvTrace_CUDA_ApiTrace_MemOpType_AtomicReduction:
                                if (pMemOpParams)
                                {
                                    const auto& atomicReduction = pMemOpParams->atomicReduction;
                                    const char* const pReductionOp = ReductionOpToCString(atomicReduction.reductionOp, atomicReduction.dataType);
                                    return formatMemOp(SprintfToString("ATOMIC_REDUCTION, op=%s, address=%" PRIu64 ", value=%" PRIu64, pReductionOp, atomicReduction.address, atomicReduction.value));
                                }
                                break;
                            case NvTrace_CUDA_ApiTrace_MemOpType_Invalid:
                                break;
                        }
                        return formatMemOp(kUnknownData);
                    }
                    case Codec::CudaApiTraceData::GpuCommandType::HostCallback:
                    {
                        if (!pCudaApiTrace || gpuCommand.hostCallbackInstanceIndex >= pCudaApiTrace->hostCallbackInstances.size())
                        {
                            break;
                        }
                        const Codec::CudaApiTraceData::HostCallbackInstance& hostCallbackInstance = pCudaApiTrace->hostCallbackInstances[gpuCommand.hostCallbackInstanceIndex];
                        const bool isEager = hostCallbackInstance.kind == Codec::CudaApiTraceData::HostCallbackKind::Eager;
                        annotations.push_back(m_writer.MakeUint("streamId", hostCallbackInstance.custreamId));
                        annotations.push_back(m_writer.MakeUint("enclosingCorrelationId", hostCallbackInstance.enclosingCorrelationId));
                        if (isEager)
                        {
                            annotations.push_back(m_writer.MakeUint("launchApiCorrelationId", hostCallbackInstance.launchApiCorrelationId));
                            annotations.push_back(m_writer.MakePointer("callback", static_cast<uint64_t>(hostCallbackInstance.callback)));
                            annotations.push_back(m_writer.MakePointer("userData", static_cast<uint64_t>(hostCallbackInstance.userData)));
                            return "Host Callback";
                        }
                        annotations.push_back(m_writer.MakeUint("graphNodeCorrelationId", hostCallbackInstance.graphNodeCorrelationId));
                        return "Graph Host Callback";
                    }
                }
                return kUnknownData;
            };

            const std::string rowName = SprintfToString("GPU %zu", deviceOrdinal);
            const std::string nodeId = GetNodeId();
            std::vector<TrackUuid> eventTracksByDepth;
            const auto getEventTrack = [&](size_t eventDepth) {
                if (eventTracksByDepth.size() <= eventDepth)
                {
                    eventTracksByDepth.resize(eventDepth + 1);
                }
                TrackUuid& track = eventTracksByDepth[eventDepth];
                if (!track)
                {
                    const std::string trackName = SprintfToString(
                        "%s (PAR %zu) - %s",
                        rowName.c_str(),
                        eventDepth + 1,
                        nodeId.c_str());
                    track = GetSubTrack(groupTrack, trackName);
                }
                return track;
            };

            m_dataModel.GenerateGpuPartitionTimelineWithNvtxProjection(
                gpuPartition,
                targetProcessLuid,
                targetNvtxDomainIndex,
                [&](const Codec::DeviceEventOrigin& deviceEventOrigin, const Codec::NvtxDeviceProjectionEvent& nvtxDeviceProjectionEvent, size_t eventDepth) {
                    if (!deviceEventOrigin)
                    {
                        return;
                    }
                    const auto& processData = *deviceEventOrigin.pProcessData;
                    const auto& pushPopRange = *nvtxDeviceProjectionEvent.pPushPopRange;

                    // TODO: When available, grab this from the GPU Clock Calibration Data
                    uint64_t deviceClockIntervalNsec = 32;
                    // Flow arrows use start timestamp as the binding point, which means start time must be unique. Pull start time back by depth to make it unique.
                    // Limit the maximum offset based on clock interval to avoid overlapping events
                    const uint64_t requiredStartOffsetAmount = static_cast<uint64_t>(nvtxDeviceProjectionEvent.initialStackSize + 1) - nvtxDeviceProjectionEvent.stackDepth;
                    const uint64_t startOffsetAmount = std::min(requiredStartOffsetAmount, static_cast<uint64_t>(deviceClockIntervalNsec - 1));
                    const bool startTimestampIsUnique = startOffsetAmount == requiredStartOffsetAmount;

                    uint64_t startTimestampUtcNsec = std::max(m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, nvtxDeviceProjectionEvent.startTimestampGpuNsec), startOffsetAmount) - startOffsetAmount;
                    uint64_t endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, nvtxDeviceProjectionEvent.endTimestampGpuNsec);
                    uint64_t durationNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                    std::string messageStorage;
                    std::string label = m_dataModel.GetNvtxTracePushPopRangeMessage(processData.nvtxTrace, pushPopRange, messageStorage);
                    std::string domainName = GetNvtxTraceDomainExportName(processData.nvtxTrace, pushPopRange.domainIndex);
                    std::string nvtxProjectionCategory = GetNvtxTraceCategory(processData.nvtxTrace, pushPopRange.domainIndex, "nvtx_gpu_projection");

                    std::vector<Annotation> annotations;
                    annotations.push_back(m_writer.MakeInt("Process ID", processData.namespacePid));
                    annotations.push_back(m_writer.MakeUint("Thread ID", pushPopRange.uniqueThreadId));
                    annotations.push_back(m_writer.MakeString("Domain", domainName));

                    std::vector<uint64_t> terminatingFlowIds;
                    if (startTimestampIsUnique)
                    {
                        terminatingFlowIds = CreateNvtxTracePushPopRangeFlowIds(deviceEventOrigin.processLuid, nvtxDeviceProjectionEvent.pPushPopRange);
                    }

                    m_writer.AddSlice(
                        getEventTrack(eventDepth),
                        nvtxProjectionCategory,
                        label,
                        ExportTimestampNsec(startTimestampUtcNsec),
                        durationNsec,
                        std::move(annotations),
                        {},
                        std::move(terminatingFlowIds));
                },
                [&](const Codec::DeviceEventOrigin& deviceEventOrigin, const Codec::DeviceEvent& deviceEvent, size_t eventDepth) {
                    std::vector<Annotation> annotations;
                    std::vector<uint64_t> terminatingFlowIds;
                    std::vector<uint64_t> flowIds;

                    // Shared by ImmediateKernelLaunch / GraphNodeLaunch slice emission after the switch.
                    uint64_t startTimestampUtcNsec = 0;
                    uint64_t endTimestampUtcNsec = 0;
                    std::string kernelExecutionLabel;
                    const char* pEventCategory = "cuda_gpu_kernel_launch";
                    bool emitSharedSlice = false;
                    uint64_t correlationId = 0;

                    switch (deviceEvent.eventType)
                    {
                        case Codec::DeviceEventType_ImmediateKernelLaunch:
                        {
                            const Codec::CudaApiTraceData::ImmediateKernelLaunch& immediateKernelLaunch = *deviceEvent.pImmediateKernelLaunch;

                            startTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, deviceEvent.startTimestampGpuNsec);
                            endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, deviceEvent.endTimestampGpuNsec);

                            const NvTrace_CUDA_ApiTrace_RecordType recordType = m_dataModel.GetCudaApiTraceRecordType(immediateKernelLaunch.recordLocation);
                            if (deviceEventOrigin)
                            {
                                annotations.push_back(m_writer.MakeInt("Process ID", deviceEventOrigin.pProcessData->namespacePid));
                            }
                            else
                            {
                                annotations.push_back(m_writer.MakeString("Process ID", kUnknownData));
                            }
                            annotations.push_back(m_writer.MakeUint("Thread ID", immediateKernelLaunch.driverApiCallRef.uniqueThreadId));
                            annotations.push_back(m_writer.MakeUint("apiCorrelationId", immediateKernelLaunch.apiCorrelationId));
                            annotations.push_back(m_writer.MakeUint("hwCorrelationId", immediateKernelLaunch.GetHwCorrelationId()));
                            annotations.push_back(m_writer.MakeBool("isImmediateLaunch", true));

                            std::string cudaDeviceOrdinalStr = kUnknownData;
                            if (deviceEventOrigin)
                            {
                                cudaDeviceOrdinalStr = SprintfToString("%u", static_cast<uint32_t>(deviceEventOrigin.pCuDevice->cudaDeviceOrdinal));
                            }

                            if (recordType == NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch)
                            {
                                const NvTrace_CUDA_ApiTrace_ImmediateKernelLaunch* pImmediateKernelLaunchEvent = m_dataModel.GetImmediateKernelLaunchPayload(immediateKernelLaunch.recordLocation);
                                NVTRACE_LOG_WRN_RETURN_IF(!pImmediateKernelLaunchEvent, , 90, "CUDA API device trace, CUdevice[%s]; immediate kernel launch payload not found", cudaDeviceOrdinalStr.c_str());
                                annotations.push_back(m_writer.MakeUint("streamId", pImmediateKernelLaunchEvent->custreamId));
                                const uint64_t cufunctionId = pImmediateKernelLaunchEvent->cufunctionId;

                                CuFunctionNames functionNames{kUnknownData, kUnknownData};
                                if (deviceEventOrigin)
                                {
                                    functionNames = GetCuFunctionNames(*deviceEventOrigin.pCudaApiTrace, cufunctionId);
                                }
                                kernelExecutionLabel += functionNames.pDemangledName;
                                kernelExecutionLabel += SprintfToString("<<<(%u, %u, %u)", pImmediateKernelLaunchEvent->gridDim_x, pImmediateKernelLaunchEvent->gridDim_y, pImmediateKernelLaunchEvent->gridDim_z);
                                kernelExecutionLabel += SprintfToString(", (%u, %u, %u)>>>", pImmediateKernelLaunchEvent->blockDim_x, pImmediateKernelLaunchEvent->blockDim_y, pImmediateKernelLaunchEvent->blockDim_z);
                                annotations.push_back(m_writer.MakeString("kernelName", functionNames.pDemangledName));
                                annotations.push_back(m_writer.MakeString("mangledName", functionNames.pMangledName));
                            }
                            else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin)
                            {
                                const NvTrace_CUDA_ApiTrace_MemcpyBegin* pMemcpyBeginEvent = m_dataModel.GetMemcpyBeginPayload(immediateKernelLaunch.recordLocation);
                                const NvTrace_CUDA_ApiTrace_CommandType commandType = pMemcpyBeginEvent ? pMemcpyBeginEvent->commandType : NvTrace_CUDA_ApiTrace_CommandType_Invalid;
                                kernelExecutionLabel = formatMemcpyBeginEvent(pMemcpyBeginEvent, commandType, annotations);
                            }
                            else if (recordType == NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin)
                            {
                                const NvTrace_CUDA_ApiTrace_MemsetBegin* pMemsetBeginEvent = m_dataModel.GetMemsetBeginPayload(immediateKernelLaunch.recordLocation);
                                const NvTrace_CUDA_ApiTrace_CommandType commandType = pMemsetBeginEvent ? pMemsetBeginEvent->commandType : NvTrace_CUDA_ApiTrace_CommandType_Invalid;
                                kernelExecutionLabel = formatMemsetBeginEvent(pMemsetBeginEvent, commandType, annotations);
                            }
                            else
                            {
                                NVTRACE_LOG_INF(90, "GPU Kernels:Immediate: apiCorrelationId=%" PRIu64 " hwCorrelationID=%u recordLocation=%" PRIu64 "; unexpected record type %d; dropping", immediateKernelLaunch.apiCorrelationId, immediateKernelLaunch.GetHwCorrelationId(), immediateKernelLaunch.recordLocation, static_cast<int>(recordType));
                                return;
                            }

                            if (deviceEventOrigin)
                            {
                                terminatingFlowIds = CreateApiCorrelationFlowIds(immediateKernelLaunch.apiCorrelationId, deviceEventOrigin.processLuid);
                            }

                            emitSharedSlice = true;
                            break;
                        }
                        case Codec::DeviceEventType_GraphNodeLaunch:
                        {
                            const Codec::CudaApiTraceData::GraphNodeLaunch& graphNodeLaunch = *deviceEvent.pGraphNodeLaunch;

                            const uint64_t startTimestampTsc = m_dataModel.ConvertGpuNsecToTscTick(gpuPartition, deviceEvent.startTimestampGpuNsec);
                            startTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(startTimestampTsc);
                            endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, deviceEvent.endTimestampGpuNsec);

                            kernelExecutionLabel = kUnknownData;
                            const Codec::CudaApiTraceData::CuGraphNodeData* pGraphNode = nullptr;
                            const Codec::CudaApiTraceData::GraphLaunch* pGraphLaunch = nullptr;

                            if (deviceEventOrigin)
                            {
                                const Codec::CudaApiTraceData& cudaApiTrace = *deviceEventOrigin.pCudaApiTrace;
                                pGraphLaunch = m_dataModel.FindGraphLaunch(cudaApiTrace, graphNodeLaunch);
                                pGraphNode = m_dataModel.FindCuGraphNode(cudaApiTrace, graphNodeLaunch.graphNodeCorrelationId);
                                if (pGraphNode)
                                {
                                    const auto graphNodeType = m_dataModel.GetCudaApiTraceGraphNodeType(pGraphNode->recordLocation, cudaApiTrace);
                                    const CudaApiTraceGraphNodeParams* const pGraphNodeParams = m_dataModel.GetCudaApiTraceGraphNodeParams(pGraphNode->recordLocation, cudaApiTrace, graphNodeType);

                                    // NvTrace carries nothing more descriptive for node kinds other than these three.
                                    if (graphNodeType == NvTrace_CUDA_ApiTrace_GraphNodeType_Kernel)
                                    {
                                        if (pGraphNodeParams)
                                        {
                                            const NvTrace_CUDA_ApiTrace_GraphNode_KernelParams& kernelParams = pGraphNodeParams->kernel;
                                            const CuFunctionNames functionNames = GetCuFunctionNames(cudaApiTrace, kernelParams.cufunctionId);
                                            kernelExecutionLabel = functionNames.pDemangledName;
                                            kernelExecutionLabel += SprintfToString("<<<(%u, %u, %u)", kernelParams.gridDimX, kernelParams.gridDimY, kernelParams.gridDimZ);
                                            kernelExecutionLabel += SprintfToString(", (%u, %u, %u)>>>", kernelParams.blockDimX, kernelParams.blockDimY, kernelParams.blockDimZ);
                                            annotations.push_back(m_writer.MakeString("kernelName", functionNames.pDemangledName));
                                            annotations.push_back(m_writer.MakeString("mangledName", functionNames.pMangledName));
                                        }
                                    }
                                    else if (graphNodeType == NvTrace_CUDA_ApiTrace_GraphNodeType_Memcpy)
                                    {
                                        kernelExecutionLabel = formatMemcpyBeginEvent(pGraphNodeParams ? &pGraphNodeParams->memcpy : nullptr, NvTrace_CUDA_ApiTrace_CommandType_Kernel, annotations);
                                    }
                                    else if (graphNodeType == NvTrace_CUDA_ApiTrace_GraphNodeType_Memset)
                                    {
                                        kernelExecutionLabel = formatMemsetBeginEvent(pGraphNodeParams ? &pGraphNodeParams->memset : nullptr, NvTrace_CUDA_ApiTrace_CommandType_Kernel, annotations);
                                    }
                                }
                            }

                            annotations.push_back(m_writer.MakeBool("isGraphLaunch", true));
                            if (deviceEventOrigin)
                            {
                                annotations.push_back(m_writer.MakeInt("Process ID", deviceEventOrigin.pProcessData->namespacePid));
                            }
                            else
                            {
                                annotations.push_back(m_writer.MakeString("Process ID", kUnknownData));
                            }
                            annotations.push_back(m_writer.MakeUint("hwCorrelationId", graphNodeLaunch.hwCorrelationId));
                            if (pGraphNode)
                            {
                                annotations.push_back(m_writer.MakeUint("apiCorrelationId", pGraphNode->graphNodeCorrelationId));
                                annotations.push_back(m_writer.MakeUint("graphExecId", pGraphNode->cugraphExecId));
                            }
                            else
                            {
                                annotations.push_back(m_writer.MakeString("apiCorrelationId", kUnknownData));
                                annotations.push_back(m_writer.MakeString("graphExecId", kUnknownData));
                            }

                            if (pGraphLaunch)
                            {
                                annotations.push_back(m_writer.MakeUint("graphLaunchHwCorrelationId", pGraphLaunch->hwCorrelationId));
                                switch (pGraphLaunch->source)
                                {
                                    case Codec::CudaApiTraceData::GraphLaunchSource::Host:
                                        annotations.push_back(m_writer.MakeUint("streamId", pGraphLaunch->payload.host.custreamId));
                                        if (pGraphLaunch->payload.host.driverApiCallRef.uniqueThreadId != 0)
                                        {
                                            annotations.push_back(m_writer.MakeUint("Thread ID", pGraphLaunch->payload.host.driverApiCallRef.uniqueThreadId));
                                        }
                                        else
                                        {
                                            annotations.push_back(m_writer.MakeString("Thread ID", kUnknownData));
                                        }
                                        annotations.push_back(m_writer.MakeString("graphLaunchSource", "Host"));
                                        annotations.push_back(m_writer.MakeUint("graphLaunchApiCorrelationId", pGraphLaunch->payload.host.apiCorrelationId));
                                        break;
                                    case Codec::CudaApiTraceData::GraphLaunchSource::Device:
                                        annotations.push_back(m_writer.MakeString("graphLaunchSource", "Device"));
                                        annotations.push_back(m_writer.MakeString("graphLaunchApiCorrelationId", kNotApplicableData));
                                        annotations.push_back(m_writer.MakeUint("graphLaunchTimestampGpuNsec", pGraphLaunch->payload.device.timestampGpuNsec));
                                        annotations.push_back(m_writer.MakeUint("graphLaunchCudaDeviceOrdinal", pGraphLaunch->payload.device.cudaDeviceOrdinal));
                                        terminatingFlowIds.push_back(AddDeviceGraphLaunchFlowId(deviceEventOrigin.processLuid, graphNodeLaunch.graphLaunchIdx));
                                        break;
                                    case Codec::CudaApiTraceData::GraphLaunchSource::Unknown:
                                    default:
                                        annotations.push_back(m_writer.MakeString("graphLaunchSource", kUnknownData));
                                        annotations.push_back(m_writer.MakeString("graphLaunchApiCorrelationId", kUnknownData));
                                        break;
                                }
                            }
                            else
                            {
                                annotations.push_back(m_writer.MakeString("graphLaunchSource", kUnknownData));
                                annotations.push_back(m_writer.MakeString("graphLaunchApiCorrelationId", kUnknownData));
                                annotations.push_back(m_writer.MakeString("graphLaunchHwCorrelationId", kUnknownData));
                            }

                            if (deviceEventOrigin
                                && pGraphLaunch
                                && pGraphLaunch->source == Codec::CudaApiTraceData::GraphLaunchSource::Host)
                            {
                                terminatingFlowIds = CreateApiCorrelationFlowIds(pGraphLaunch->payload.host.apiCorrelationId, deviceEventOrigin.processLuid);
                                correlationId = pGraphLaunch->payload.host.apiCorrelationId;
                            }

                            emitSharedSlice = true;
                            break;
                        }
                        case Codec::DeviceEventType_GpuCommand:
                        {
                            const Codec::CudaApiTraceData::GpuCommand& gpuCommand = *deviceEvent.pGpuCommand;
                            startTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, deviceEvent.startTimestampGpuNsec);
                            endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, deviceEvent.endTimestampGpuNsec);

                            kernelExecutionLabel = formatGpuCommand(gpuCommand, deviceEventOrigin, annotations);
                            pEventCategory = "cuda_gpu_command";

                            if (deviceEventOrigin)
                            {
                                annotations.push_back(m_writer.MakeInt("Process ID", deviceEventOrigin.pProcessData->namespacePid));
                            }
                            else
                            {
                                annotations.push_back(m_writer.MakeString("Process ID", kUnknownData));
                            }
                            if (gpuCommand.driverApiCallRef.uniqueThreadId != 0)
                            {
                                annotations.push_back(m_writer.MakeUint("Thread ID", gpuCommand.driverApiCallRef.uniqueThreadId));
                            }
                            else
                            {
                                annotations.push_back(m_writer.MakeString("Thread ID", kUnknownData));
                            }

                            if (deviceEventOrigin)
                            {
                                if (gpuCommand.commandType == Codec::CudaApiTraceData::GpuCommandType::HostCallback)
                                {
                                    if (gpuCommand.hostCallbackInstanceIndex < deviceEventOrigin.pCudaApiTrace->hostCallbackInstances.size())
                                    {
                                        const Codec::CudaApiTraceData::HostCallbackInstance& hostCallbackInstance =
                                            deviceEventOrigin.pCudaApiTrace->hostCallbackInstances[gpuCommand.hostCallbackInstanceIndex];
                                        const bool isEager = hostCallbackInstance.kind == Codec::CudaApiTraceData::HostCallbackKind::Eager;
                                        const uint64_t flowCorrelationId = isEager ? hostCallbackInstance.launchApiCorrelationId : hostCallbackInstance.enclosingCorrelationId;
                                        terminatingFlowIds = CreateApiCorrelationFlowIds(flowCorrelationId, deviceEventOrigin.processLuid);
                                        if (!isEager)
                                        {
                                            correlationId = hostCallbackInstance.enclosingCorrelationId;
                                        }
                                    }
                                }
                                else
                                {
                                    uint64_t apiCorrelationId = GetGraphLaunchCorrelationIdForGpuCommand(gpuCommand, *deviceEventOrigin.pCudaApiTrace);
                                    if (apiCorrelationId)
                                    {
                                        correlationId = apiCorrelationId;
                                    }
                                    else
                                    {
                                        apiCorrelationId = m_dataModel.GetRecordMetadata(gpuCommand.recordLocation).correlationId;
                                    }
                                    terminatingFlowIds = CreateApiCorrelationFlowIds(apiCorrelationId, deviceEventOrigin.processLuid);
                                }
                            }

                            emitSharedSlice = true;
                            break;
                        }
                        default:
                            return;
                    }

                    // Kernel splitting flow events
                    const std::map<uint64_t, uint64_t>* pDeviceEventFlowIdMap = GetDeviceEventFlowIdMap(deviceEvent);
                    if (pDeviceEventFlowIdMap)
                    {
                        const auto& deviceEventFlowIdMap = *pDeviceEventFlowIdMap;
                        auto flowIdItr = deviceEventFlowIdMap.begin();
                        while (flowIdItr != deviceEventFlowIdMap.end() && flowIdItr->first < deviceEvent.startTimestampGpuNsec)
                        {
                            terminatingFlowIds.push_back(flowIdItr->second);
                            ++flowIdItr;
                        }
                        if (flowIdItr != deviceEventFlowIdMap.end() && flowIdItr->first == deviceEvent.startTimestampGpuNsec)
                        {
                            flowIds.push_back(flowIdItr->second);
                        }
                    }

                    if (emitSharedSlice)
                    {
                        const EventId eventId = m_writer.AddSlice(
                            getEventTrack(eventDepth),
                            pEventCategory,
                            kernelExecutionLabel,
                            ExportTimestampNsec(startTimestampUtcNsec),
                            endTimestampUtcNsec - startTimestampUtcNsec,
                            std::move(annotations),
                            std::move(flowIds),
                            std::move(terminatingFlowIds));
                        if (correlationId)
                        {
                            m_writer.SetCorrelationId(eventId, correlationId);
                        }
                    }
                });
        }

        inline void AddDeviceGraphLaunchMarkers(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuPartitionData& gpuPartition)
        {
            const std::string rowName = m_options.isolateGpuEventRows
                                          ? SprintfToString("GPU %zu: Device Graph Launches", deviceOrdinal)
                                          : "Device Graph Launches";

            for (const auto& processMapPair : m_dataModel.processes)
            {
                const Codec::ProcessData& processData = processMapPair.second;
                for (size_t graphLaunchIndex = 0; graphLaunchIndex < processData.cudaApiTrace.graphLaunches.size(); ++graphLaunchIndex)
                {
                    const Codec::CudaApiTraceData::GraphLaunch& graphLaunch = processData.cudaApiTrace.graphLaunches[graphLaunchIndex];
                    if (graphLaunch.source != Codec::CudaApiTraceData::GraphLaunchSource::Device
                        || graphLaunch.payload.device.timestampGpuNsec == Codec::InvalidTimestamp)
                    {
                        continue;
                    }

                    const uint32_t cudaDeviceOrdinal = graphLaunch.payload.device.cudaDeviceOrdinal;
                    if (cudaDeviceOrdinal >= processData.cudaApiTrace.devices.size())
                    {
                        continue;
                    }
                    const Codec::CudaApiTraceData::CuDeviceData& cuDevice = processData.cudaApiTrace.devices[cudaDeviceOrdinal];
                    if (cuDevice.gpuUuid != gpuPartition.gpuUuid)
                    {
                        continue;
                    }

                    const uint64_t timestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, graphLaunch.payload.device.timestampGpuNsec);
                    std::vector<Annotation> annotations;
                    annotations.push_back(m_writer.MakeInt("Process ID", processData.namespacePid));
                    annotations.push_back(m_writer.MakeUint("graphExecId", graphLaunch.cugraphExecId));
                    annotations.push_back(m_writer.MakeUint("graphLaunchHwCorrelationId", graphLaunch.hwCorrelationId));
                    annotations.push_back(m_writer.MakeUint("graphLaunchTimestampGpuNsec", graphLaunch.payload.device.timestampGpuNsec));
                    annotations.push_back(m_writer.MakeUint("graphLaunchCudaDeviceOrdinal", cudaDeviceOrdinal));

                    const std::vector<uint64_t>& flowIds = GetDeviceGraphLaunchFlowIds(processMapPair.first, graphLaunchIndex);
                    if (flowIds.empty())
                    {
                        continue;
                    }

                    m_writer.AddInstant(
                        GetSubTrack(groupTrack, rowName),
                        "cuda_gpu_graph_launch",
                        "Device Graph Launch",
                        ExportTimestampNsec(timestampUtcNsec),
                        std::move(annotations),
                        flowIds);
                }
            }
        }

        inline void AddGpuHwTrace(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuPartitionData& gpuPartition, const Codec::GpuHwTraceData& gpuHwTrace)
        {
            const TrackUuid gpuDiagnosticGroup = GetGpuDiagnosticsGroup(groupTrack);

            for (const Codec::GpuHwTraceData::ProviderDiagnostic& providerDiagnostic : gpuHwTrace.providerDiagnostics)
            {
                if (providerDiagnostic.recordType != NvTrace_GPU_HwTrace_RecordType_ProviderDiagnostic_InitializationFailure)
                {
                    continue;
                }

                const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(providerDiagnostic.timestampTsc);
                const uint64_t endUtcNsec = m_dataModel.m_observationEndUtcNsec;

                DiagnosticSeverity severity = DiagnosticSeverity::Error;
                const char* pMessage = "GPU HW Trace provider failed during initialization.";

                switch (providerDiagnostic.initializationFailureReason)
                {
                    case NvTrace_GPU_HwTrace_ProviderFailureReason_Disabled:
                        severity = DiagnosticSeverity::Warning;
                        pMessage = "GPU HW Trace provider was disabled by NSIGHT_PERF_GPU_HW_TRACE_ENABLE.";
                        break;

                    case NvTrace_GPU_HwTrace_ProviderFailureReason_UnsupportedDevice:
                        pMessage = "GPU HW Trace provider is unsupported on this device.";
                        break;

                    case NvTrace_GPU_HwTrace_ProviderFailureReason_ConfigurationInvalid:
                        pMessage = "GPU HW Trace provider failed because its configuration was invalid.";
                        break;

                    case NvTrace_GPU_HwTrace_ProviderFailureReason_InitializationFailed:
                        pMessage = "GPU HW Trace provider failed during initialization.";
                        break;

                    case NvTrace_GPU_HwTrace_ProviderFailureReason_None:
                    case NvTrace_GPU_HwTrace_ProviderFailureReason_FORCE_INT:
                        continue;
                }

                std::vector<Annotation> annotations;
                annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                annotations.push_back(m_writer.MakeUint("Timestamp TSC", providerDiagnostic.timestampTsc));

                AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU HW Trace Diagnostics", severity, "Initialization Failure", pMessage, startUtcNsec, endUtcNsec, annotations);
            }


            for (const Codec::GpuProviderLifecycleEvent& lifecycleEvent : gpuHwTrace.lifecycleEvents)
            {
                const bool isPauseEvent = lifecycleEvent.lifecycleEventType == NvTrace_GPU_Provider_LifecycleEventType_Pause;
                const bool isResumeEvent = lifecycleEvent.lifecycleEventType == NvTrace_GPU_Provider_LifecycleEventType_Resume;
                if (!isPauseEvent && !isResumeEvent)
                {
                    continue;
                }

                const bool isSuccessfulResumeEvent = isResumeEvent && lifecycleEvent.resumePhase == NvTrace_GPU_Provider_ResumePhase_Succeeded && lifecycleEvent.resumeFailureReason == NvTrace_GPU_Provider_ResumeFailureReason_None;

                const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(lifecycleEvent.timestampTsc);
                const uint64_t endUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(lifecycleEvent.endTimestampTsc);

                std::vector<Annotation> annotations;
                annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                annotations.push_back(m_writer.MakeString("Pause Reason", GetGpuProviderPauseReason(lifecycleEvent.pauseReason)));
                annotations.push_back(m_writer.MakeString("Resume Phase", GetGpuProviderResumePhase(lifecycleEvent.resumePhase)));
                annotations.push_back(m_writer.MakeString("Failure Reason", GetGpuProviderResumeFailureReason(lifecycleEvent.resumeFailureReason)));
                annotations.push_back(m_writer.MakeUint("Retry Count", lifecycleEvent.retryCount));
                annotations.push_back(m_writer.MakeUint("Timestamp TSC", lifecycleEvent.timestampTsc));
                annotations.push_back(m_writer.MakeString("End Timestamp TSC", lifecycleEvent.endTimestampTsc == Codec::MaxTimestampTsc ? std::string("Open") : SprintfToString("%" PRIu64, lifecycleEvent.endTimestampTsc)));

                const char* pEventName = "GPU HW Trace Paused";
                if (isResumeEvent)
                {
                    switch (lifecycleEvent.resumePhase)
                    {
                        case NvTrace_GPU_Provider_ResumePhase_Started:
                            pEventName = "GPU HW Trace Resume Started";
                            break;
                        case NvTrace_GPU_Provider_ResumePhase_Failed:
                            pEventName = "GPU HW Trace Paused";
                            break;
                        case NvTrace_GPU_Provider_ResumePhase_Succeeded:
                            pEventName = "GPU HW Trace Resume Succeeded";
                            break;
                        case NvTrace_GPU_Provider_ResumePhase_Invalid:
                        case NvTrace_GPU_Provider_ResumePhase_COUNT:
                        case NvTrace_GPU_Provider_ResumePhase_FORCEINT:
                            pEventName = "GPU HW Trace Resume";
                            break;
                    }
                }

                if (!isSuccessfulResumeEvent)
                {
                    m_writer.AddSlice(GetSubTrack(groupTrack, "GPU HW Trace Lifecycle"), "gpu_hw", pEventName, ExportTimestampNsec(startUtcNsec), endUtcNsec - startUtcNsec, annotations);
                }

                if (isResumeEvent && lifecycleEvent.resumePhase == NvTrace_GPU_Provider_ResumePhase_Failed && lifecycleEvent.resumeFailureReason != NvTrace_GPU_Provider_ResumeFailureReason_None)
                {
                    AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU HW Trace Diagnostics", DiagnosticSeverity::Warning, "Resume Failure", "GPU HW Trace provider failed to resume.", startUtcNsec, endUtcNsec, annotations);
                }
                else if (isSuccessfulResumeEvent)
                {
                    AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU HW Trace Diagnostics", DiagnosticSeverity::Info, "Resume Succeeded", "GPU HW Trace provider resumed successfully.", startUtcNsec, endUtcNsec, annotations);
                }
            }

            // Handle context intervals
            if (gpuHwTrace.contextTrace.empty())
            {
                NVTRACE_LOG_INF(90, "GPU %zu; no GPU HW trace context intervals", deviceOrdinal);
            }
            else
            {
                for (const auto& contextTrace : gpuHwTrace.contextTrace)
                {
                    const uint64_t startTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, contextTrace.loadTimestampGpuNsec);
                    const uint64_t endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, contextTrace.saveTimestampGpuNsec);

                    const uint64_t durNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                    std::vector<Annotation> annotations;
                    annotations.push_back(m_writer.MakeUint("contextId", contextTrace.contextId));

                    m_writer.AddSlice(GetSubTrack(groupTrack, "GPU HW Context Trace"), "gpu_hw", "GPU HW Context Interval", ExportTimestampNsec(startTimestampUtcNsec), durNsec, std::move(annotations));
                }
            }
        }

        inline void AddGpuPmCounters(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuPartitionData& gpuPartition)
        {
            const Codec::GpuPmCountersData& gpuPmCounters = gpuPartition.gpuPmCounters;

            const TrackUuid gpuDiagnosticGroup = GetGpuDiagnosticsGroup(groupTrack);

            for (const Codec::GpuPmCountersData::ProviderDiagnostic& providerDiagnostic : gpuPmCounters.providerDiagnostics)
            {
                switch (providerDiagnostic.recordType)
                {
                    case NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_InitializationFailure:
                    {
                        const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(providerDiagnostic.payload.initializationFailure.timestampTsc);
                        const uint64_t endUtcNsec = m_dataModel.m_observationEndUtcNsec;

                        DiagnosticSeverity severity = DiagnosticSeverity::Error;
                        const char* pMessage = "GPU PM Counters provider failed during initialization.";

                        switch (providerDiagnostic.payload.initializationFailure.failureReason)
                        {
                            case NvTrace_GPU_PmCounters_ProviderFailureReason_Disabled:
                                severity = DiagnosticSeverity::Warning;
                                pMessage = "GPU PM Counters provider was disabled by NSIGHT_PERF_GPU_PM_ENABLE.";
                                break;

                            case NvTrace_GPU_PmCounters_ProviderFailureReason_NoMetricsConfigured:
                                pMessage = "GPU PM Counters provider failed because no metrics were configured for this GPU.";
                                break;

                            case NvTrace_GPU_PmCounters_ProviderFailureReason_PerfWorksInitializationFailed:
                                pMessage = "GPU PM Counters provider failed during PerfWorks initialization.";
                                break;

                            case NvTrace_GPU_PmCounters_ProviderFailureReason_InitializationFailed:
                                pMessage = "GPU PM Counters provider failed during initialization.";
                                break;

                            case NvTrace_GPU_PmCounters_ProviderFailureReason_None:
                            case NvTrace_GPU_PmCounters_ProviderFailureReason_FORCE_INT:
                                continue;
                        }

                        std::vector<Annotation> annotations;
                        annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                        annotations.push_back(m_writer.MakeUint("Timestamp TSC", providerDiagnostic.payload.initializationFailure.timestampTsc));

                        AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU PM Counters Diagnostics", severity, "Initialization Failure", pMessage, startUtcNsec, endUtcNsec, annotations);
                        break;
                    }
                    case NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_UpdateTimeExceeded:
                    {
                        const auto& updateTimeExceeded = providerDiagnostic.payload.updateTimeExceeded;
                        const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(updateTimeExceeded.updateStartTimestampTsc);
                        const uint64_t endUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(updateTimeExceeded.updateEndTimestampTsc);

                        std::vector<Annotation> annotations;
                        annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                        annotations.push_back(m_writer.MakeUint("Configured Sampling Rate (Hz)", gpuPmCounters.pmSampleRateHz));
                        annotations.push_back(m_writer.MakeUint("Thread CPU Time (ns)", updateTimeExceeded.updateThreadCpuTimeNsec));

                        const char* pMessage = "GPU PM Counters provider update time exceeded its CPU-budget; try reducing the sampling rate.";
                        if (updateTimeExceeded.updateThreadCpuTimeNsec != 0 && updateTimeExceeded.updateWallTimeNsec != 0)
                        {
                            double updateCpuUsagePercentage = 100.0 * updateTimeExceeded.updateThreadCpuTimeNsec / updateTimeExceeded.updateWallTimeNsec;
                            annotations.push_back(m_writer.MakeDouble("Thread CPU Usage (%)", updateCpuUsagePercentage));

                            if (updateCpuUsagePercentage < 50)
                            {
                                pMessage = "GPU PM Counters provider update time exceeded its CPU-budget; try reducing the sampling rate. Thread CPU usage is below 50%; system load may be delaying execution.";
                            }
                        }

                        AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU PM Counters Diagnostics", DiagnosticSeverity::Warning, "Update CPU-Budget Exceeded", pMessage, startUtcNsec, endUtcNsec, annotations);
                        break;
                    }
                    case NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_SampleDiscontinuity:
                    {
                        const auto& sampleDiscontinuity = providerDiagnostic.payload.sampleDiscontinuity;
                        const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(sampleDiscontinuity.startTimestampTsc);
                        const uint64_t endUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(sampleDiscontinuity.endTimestampTsc);

                        std::vector<Annotation> annotations;
                        annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                        annotations.push_back(m_writer.MakeUint("Configured Sampling Rate (Hz)", gpuPmCounters.pmSampleRateHz));
                        const char* pMessage = "";
                        switch (sampleDiscontinuity.reason)
                        {
                            case NvTrace_GPU_PmCounters_SampleDiscontinuityReason_Unknown:
                                annotations.push_back(m_writer.MakeString("Sample Discontinuity Reason", "Unknown"));
                                pMessage = "GPU PM Counters sample discontinuity detected; cause unknown.";
                                break;
                            case NvTrace_GPU_PmCounters_SampleDiscontinuityReason_HwBufferOverrun:
                                annotations.push_back(m_writer.MakeString("Sample Discontinuity Reason", "Hardware buffer overrun"));
                                pMessage = "GPU PM Counters sample discontinuity caused by a hardware buffer overrun.";
                                break;
                            case NvTrace_GPU_PmCounters_SampleDiscontinuityReason_FORCE_INT:
                                continue;
                        }

                        AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU PM Counters Sample Diagnostics", DiagnosticSeverity::Warning, "Sample Discontinuity Detected", pMessage, startUtcNsec, endUtcNsec, annotations);
                        break;
                    }
                    default:
                        continue;
                }
            }


            for (const auto& lifecycleEvent : gpuPmCounters.lifecycleEvents)
            {
                const bool isPauseEvent = lifecycleEvent.lifecycleEventType == NvTrace_GPU_Provider_LifecycleEventType_Pause;
                const bool isResumeEvent = lifecycleEvent.lifecycleEventType == NvTrace_GPU_Provider_LifecycleEventType_Resume;
                if (!isPauseEvent && !isResumeEvent)
                {
                    continue;
                }

                const bool isSuccessfulResumeEvent = isResumeEvent && lifecycleEvent.resumePhase == NvTrace_GPU_Provider_ResumePhase_Succeeded && lifecycleEvent.resumeFailureReason == NvTrace_GPU_Provider_ResumeFailureReason_None;

                const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(lifecycleEvent.timestampTsc);
                const uint64_t endUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(lifecycleEvent.endTimestampTsc);

                const uint64_t durationNsec = endUtcNsec - startUtcNsec;

                std::vector<Annotation> annotations;
                annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                annotations.push_back(m_writer.MakeString("Pause Reason", GetGpuProviderPauseReason(lifecycleEvent.pauseReason)));
                annotations.push_back(m_writer.MakeString("Resume Phase", GetGpuProviderResumePhase(lifecycleEvent.resumePhase)));
                annotations.push_back(m_writer.MakeString("Failure Reason", GetGpuProviderResumeFailureReason(lifecycleEvent.resumeFailureReason)));
                annotations.push_back(m_writer.MakeUint("Retry Count", lifecycleEvent.retryCount));
                annotations.push_back(m_writer.MakeUint("Timestamp TSC", lifecycleEvent.timestampTsc));
                annotations.push_back(m_writer.MakeString("End Timestamp TSC", lifecycleEvent.endTimestampTsc == Codec::MaxTimestampTsc ? std::string("Open") : SprintfToString("%" PRIu64, lifecycleEvent.endTimestampTsc)));

                const char* pEventName = "GPU Pm Counters Paused";
                if (isResumeEvent)
                {
                    switch (lifecycleEvent.resumePhase)
                    {
                        case NvTrace_GPU_Provider_ResumePhase_Started:
                            pEventName = "GPU Pm Counters Resume Started";
                            break;
                        case NvTrace_GPU_Provider_ResumePhase_Failed:
                            pEventName = "GPU Pm Counters Paused";
                            break;
                        case NvTrace_GPU_Provider_ResumePhase_Succeeded: // Success is shown as an informational diagnostic below.
                            pEventName = "GPU Pm Counters Resume Succeeded";
                            break;
                        case NvTrace_GPU_Provider_ResumePhase_Invalid:
                        case NvTrace_GPU_Provider_ResumePhase_COUNT:
                        case NvTrace_GPU_Provider_ResumePhase_FORCEINT:
                            pEventName = "GPU Pm Counters Resume";
                            break;
                    }
                }

                if (!isSuccessfulResumeEvent)
                {
                    m_writer.AddSlice(GetSubTrack(groupTrack, "GPU PM Counters Lifecycle"), "gpu_pm", pEventName, ExportTimestampNsec(startUtcNsec), durationNsec, annotations);
                }

                const bool isFailedResumeEvent = isResumeEvent && lifecycleEvent.resumePhase == NvTrace_GPU_Provider_ResumePhase_Failed && lifecycleEvent.resumeFailureReason != NvTrace_GPU_Provider_ResumeFailureReason_None;
                if (isFailedResumeEvent)
                {
                    DiagnosticSeverity severity = DiagnosticSeverity::Warning;
                    const char* pMessage = "GPU PM Counters provider failed to resume.";

                    switch (lifecycleEvent.resumeFailureReason)
                    {
                        case NvTrace_GPU_Provider_ResumeFailureReason_InsufficientPrivilege:
                            severity = DiagnosticSeverity::Error;
                            pMessage = "GPU Profiler Permissions were not granted by the sysadmin. See https://developer.nvidia.com/ERR_NVGPUCTRPERM";
                            break;
                        case NvTrace_GPU_Provider_ResumeFailureReason_ResourceInUse:
                            severity = DiagnosticSeverity::Warning;
                            pMessage = "Another tool has exclusively acquired the GPU counter hardware.";
                            break;
                        case NvTrace_GPU_Provider_ResumeFailureReason_InsufficientDriverVersion:
                            severity = DiagnosticSeverity::Error;
                            pMessage = "The installed CUDA or kernel driver is incompatible with this provider.";
                            break;
                        case NvTrace_GPU_Provider_ResumeFailureReason_Unknown:
                            severity = DiagnosticSeverity::Warning;
                            pMessage = "Generic GPU PM Counters resume failure.";
                            break;
                        case NvTrace_GPU_Provider_ResumeFailureReason_None:
                        case NvTrace_GPU_Provider_ResumeFailureReason_COUNT:
                        case NvTrace_GPU_Provider_ResumeFailureReason_FORCEINT:
                            break;
                    }

                    AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU PM Counters Diagnostics", severity, "Resume Failure", pMessage, startUtcNsec, endUtcNsec, annotations);
                }
                else if (isSuccessfulResumeEvent)
                {
                    AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU PM Counters Diagnostics", DiagnosticSeverity::Info, "Resume Succeeded", "GPU PM Counters provider resumed successfully.", startUtcNsec, endUtcNsec, annotations);
                }
            }

            NVTRACE_LOG_INF(50, "GPU %zu; pmSampleRateHz: %u", deviceOrdinal, gpuPmCounters.pmSampleRateHz);
            NVTRACE_LOG_INF(50, "GPU %zu; updateIntervalNsec: %" PRIu64, deviceOrdinal, gpuPmCounters.updateIntervalNsec);
            NVTRACE_LOG_INF(50, "GPU %zu; readbackHoldoffNsec: %" PRIu64, deviceOrdinal, gpuPmCounters.readbackHoldoffNsec);
            NVTRACE_LOG_INF(50, "GPU %zu; keepLatest: %s", deviceOrdinal, gpuPmCounters.keepLatest ? "true" : "false");

            const auto& metricsToConfigure = gpuPmCounters.metricsToConfigure;
            const auto& sampleRecordLocations = gpuPmCounters.sampleRecordLocations;

            for (size_t metricIdx = 0; metricIdx < metricsToConfigure.size(); ++metricIdx)
            {
                NVTRACE_LOG_INF(90, "GPU %zu; configured metric [%zu]: %s (keepInstances: %s)", deviceOrdinal, metricIdx, metricsToConfigure[metricIdx].metricName.c_str(), metricsToConfigure[metricIdx].keepInstances ? "true" : "false");
            }

            char logContext[64];
            snprintf(logContext, sizeof(logContext), "GPU %zu", deviceOrdinal);
            Codec::GpuPmExportPrepared prepared = Codec::PrepareGpuPmMetricsForExport(
                m_dataModel, gpuPartition, gpuPartition.info.chipName, logContext);
            const auto& exportPlan = prepared.plan;
            const auto& metricNamesForExport = exportPlan.metricNamesForExport;
            Codec::MetricsEvaluationContext& evaluationContext = prepared.evaluationContext;
            std::vector<double> offlineMetricValues(evaluationContext.GetMetricNames().size());
            if (metricNamesForExport.empty())
            {
                NVTRACE_LOG_WRN(90, "GPU %zu; no metrics to export", deviceOrdinal);
                return;
            }
            for (size_t metricIdx = 0; metricIdx < metricNamesForExport.size(); ++metricIdx)
            {
                NVTRACE_LOG_INF(90, "GPU %zu; metric to export [%zu]: %s", deviceOrdinal, metricIdx, metricNamesForExport[metricIdx].c_str());
            }

            if (sampleRecordLocations.empty())
            {
                NVTRACE_LOG_WRN(90, "GPU %zu; no GPU PM counters samples", deviceOrdinal);
                return;
            }
            if (gpuPmCounters.pmSampleRateHz == 0)
            {
                NVTRACE_LOG_WRN(90, "GPU %zu; pmSampleRateHz is 0", deviceOrdinal);
                return;
            }

            // Prepare counter tracks. The metric set states a row hierarchy, and Perfetto tracks nest to
            // any depth, so the rows are walked as authored rather than flattened to a fixed shape.
            // A metric no row names stays on the device's own track under its counter name.
            const auto& rowLabelsForExport = exportPlan.rowLabelsForExport;
            const auto& rowCategoriesForExport = exportPlan.rowCategoriesForExport;
            const auto& timeline = gpuPartition.gpuPmCounters.timeline;

            // Sorts after every row the set declared, so the collector row sits at the bottom.
            constexpr int32_t kUnclaimedMetricRowRank = 1000000;

            // Resolved once, then each counter walks its own row path down from the device track.
            const auto rowPathsByMetricName = timeline.ResolveRowPathsByMetricName();

            // A group states how its row summarizes the metrics under it. That only reads as a quantity when
            // the group also picked one sub-metric: max over a mix of percentages and byte counts is a number
            // with no meaning, so a group without a selection stays a plain row.
            std::map<std::string, const Codec::GpuPmCountersData::TimelineMetricGroup*> aggregatingGroupByName;
            for (const auto& metricGroup : timeline.metricGroups)
            {
                if (!metricGroup.subMetricNames.empty())
                {
                    aggregatingGroupByName.emplace(metricGroup.name, &metricGroup);
                }
            }

            // Rows that carry their own value, in creation order. The row track is the counter track: the
            // value belongs to the row, not to a sibling that happens to sit beside its members.
            struct RowAggregate
            {
                TrackUuid track = 0;
                bool sumAcrossMetrics = false;
            };
            std::vector<RowAggregate> rowAggregates;
            std::unordered_map<std::string, size_t> aggregateIndexByRowKey;

            std::unordered_map<std::string, TrackUuid> trackByRowPath;
            const auto fnTrackForRowPath = [&](const std::vector<std::string>& rowPath) {
                TrackUuid parentTrack = groupTrack;
                std::string key;
                for (const std::string& rowName : rowPath)
                {
                    key += rowName;
                    key += '\x1f';
                    const auto trackIt = trackByRowPath.find(key);
                    if (trackIt != trackByRowPath.end())
                    {
                        parentTrack = trackIt->second;
                        continue;
                    }
                    // Rank by insertion, so rows sit in the order the set declared rather than by name.
                    const std::string rowTrackName = SprintfToString("GPU %zu - %s", deviceOrdinal, rowName.c_str());
                    const auto groupIt = aggregatingGroupByName.find(rowName);
                    if (groupIt != aggregatingGroupByName.end())
                    {
                        parentTrack = GetCounterSubTrack(parentTrack, rowTrackName, "", static_cast<int32_t>(trackByRowPath.size()));
                        aggregateIndexByRowKey.emplace(key, rowAggregates.size());
                        rowAggregates.push_back({parentTrack, groupIt->second->sumAcrossMetrics});
                    }
                    else
                    {
                        parentTrack = GetOrCreateNamedGroup(parentTrack, rowTrackName, static_cast<int32_t>(trackByRowPath.size()));
                    }
                    trackByRowPath.emplace(key, parentTrack);
                }
                return parentTrack;
            };

            // Rows are created before any metric is placed, in the order the view declares them, so sibling
            // rank follows the set rather than whichever metric happened to mention a row first.
            for (const auto& declaredRowPath : timeline.RowPathsInDeclaredOrder())
            {
                fnTrackForRowPath(declaredRowPath);
            }

            // One entry per metric, holding a counter track for each row that owns it. A metric named by two
            // rows is drawn under both, so the same value is emitted once per track.
            std::vector<std::vector<TrackUuid>> metricTracks;
            // Per metric, the rows whose own value folds this metric in. Parallel to metricTracks.
            std::vector<std::vector<size_t>> aggregateIndicesForMetric;
            for (size_t metricIndex = 0; metricIndex < metricNamesForExport.size(); metricIndex++)
            {
                const std::string& metricName = metricNamesForExport[metricIndex];
                const bool hasRowLabel = metricIndex < rowLabelsForExport.size() && !rowLabelsForExport[metricIndex].empty();
                const std::string metricRowName = SprintfToString("%s (GPU Metric)", hasRowLabel ? rowLabelsForExport[metricIndex].c_str() : metricName.c_str());
                const std::string unitName = metricName.find(".pct_of_peak_sustained_elapsed") != std::string::npos ? "%" : "";

                std::vector<TrackUuid> parentTracks;
                std::vector<size_t> aggregateIndices;
                if (!timeline.Empty())
                {
                    // Rows reference the timeline metric, not the per-sub-metric label, so the lookup uses
                    // the coarser name.
                    const auto& rowMetricNamesForExport = exportPlan.rowMetricNamesForExport;
                    const bool hasRowMetric = metricIndex < rowMetricNamesForExport.size() && !rowMetricNamesForExport[metricIndex].empty();
                    const auto pathIt = hasRowMetric ? rowPathsByMetricName.find(rowMetricNamesForExport[metricIndex]) : rowPathsByMetricName.end();
                    const std::string& subMetricName = metricIndex < exportPlan.rowSubMetricNamesForExport.size()
                                                         ? exportPlan.rowSubMetricNamesForExport[metricIndex]
                                                         : std::string();
                    if (pathIt != rowPathsByMetricName.end())
                    {
                        for (const auto& placement : pathIt->second)
                        {
                            // A group names the sub-metrics its row draws, and a metric usually defines more
                            // than one row wants. Skipping the rest is what keeps a row to its own columns.
                            if (placement.DrawsSubMetric(subMetricName))
                            {
                                parentTracks.push_back(fnTrackForRowPath(placement.rowPath));
                                // The row summarizes exactly what it draws, so the same filter decides both.
                                std::string aggregateKey;
                                for (const std::string& rowName : placement.rowPath)
                                {
                                    aggregateKey += rowName;
                                    aggregateKey += '\x1f';
                                }
                                const auto aggregateIt = aggregateIndexByRowKey.find(aggregateKey);
                                if (aggregateIt != aggregateIndexByRowKey.end())
                                {
                                    aggregateIndices.push_back(aggregateIt->second);
                                }
                            }
                        }
                    }
                }
                else
                {
                    // A trace written before the timeline section carries a row name per metric instead.
                    const std::string category = metricIndex < rowCategoriesForExport.size() ? rowCategoriesForExport[metricIndex] : std::string();
                    if (!category.empty())
                    {
                        parentTracks.push_back(GetSubTrack(groupTrack, SprintfToString("GPU %zu - %s", deviceOrdinal, category.c_str())));
                    }
                }
                // No row draws it, either because no group lists it or because the groups that do list it
                // did not pick this sub-metric. It still collected, so it is kept, but under one row of its
                // own rather than beside the set's rows where it would read as a top-level result.
                if (parentTracks.empty())
                {
                    parentTracks.push_back(timeline.Empty() ? groupTrack : GetOrCreateNamedGroup(groupTrack, SprintfToString("GPU %zu - Other", deviceOrdinal), kUnclaimedMetricRowRank));
                }
                std::vector<TrackUuid> tracksForMetric;
                tracksForMetric.reserve(parentTracks.size());
                for (const TrackUuid parentTrack : parentTracks)
                {
                    tracksForMetric.push_back(GetCounterSubTrack(parentTrack, metricRowName, unitName));
                }
                metricTracks.push_back(std::move(tracksForMetric));
                aggregateIndicesForMetric.push_back(std::move(aggregateIndices));
            }

            auto emitMetricSample = [&](size_t metricIndex, const uint64_t timestampUtcNsec, const double metricValue) {
                // Evaluated once, drawn once per owning row, so extra rows cost an emit rather than an evaluation.
                for (const TrackUuid metricTrack : metricTracks[metricIndex])
                {
                    m_writer.AddCounterSample(metricTrack, ExportTimestampNsec(timestampUtcNsec), metricValue);
                }
            };

            // Export-only sentinel: no PM measurement is available during this interval.
            constexpr double missingSampleValue = 0.0;
            auto emitMissingSample = [&](uint64_t timestampUtcNsec) {
                for (size_t metricIndex = 0; metricIndex < metricNamesForExport.size(); ++metricIndex)
                {
                    emitMetricSample(metricIndex, timestampUtcNsec, missingSampleValue);
                }
                for (const auto& aggregate : rowAggregates)
                {
                    m_writer.AddCounterSample(aggregate.track, ExportTimestampNsec(timestampUtcNsec), missingSampleValue);
                }
            };

            const uint64_t samplePeriodNsec = 1'000'000'000 / gpuPmCounters.pmSampleRateHz;
            const uint64_t discontinuityThresholdNsec = samplePeriodNsec + samplePeriodNsec / 2;
            // Reused across samples so a row's own value costs no allocation per sample.
            std::vector<double> rowAggregateValues(rowAggregates.size(), 0.0);
            std::vector<bool> rowAggregateHasValue(rowAggregates.size(), false);

            // For each sample point, for each metric, output one event.
            uint64_t previousSampleEndTimestampUtcNsec = Codec::InvalidTimestamp;
            m_dataModel.ForEachGpuPmCountersSamples(gpuPartition, [&](const Codec::GpuPmCountersSample& gpuPmCountersSample) {
                // Detect discontinuities in the PM counter data amd assign missing sample value across missing samples.
                if (previousSampleEndTimestampUtcNsec != Codec::InvalidTimestamp
                    && previousSampleEndTimestampUtcNsec < gpuPmCountersSample.startTimestampUtcNsec
                    && discontinuityThresholdNsec < gpuPmCountersSample.endTimestampUtcNsec - previousSampleEndTimestampUtcNsec)
                {
                    emitMissingSample(previousSampleEndTimestampUtcNsec);
                }
                previousSampleEndTimestampUtcNsec = gpuPmCountersSample.endTimestampUtcNsec;

                bool offlineMetricValuesAvailable = offlineMetricValues.empty();
                if (!offlineMetricValues.empty())
                {
                    offlineMetricValuesAvailable = [&]() {
                        if (!gpuPmCountersSample.pSampleRangeData)
                        {
                            return false;
                        }
                        if (!evaluationContext.SetRangeData(gpuPmCountersSample.pSampleRangeData, gpuPmCountersSample.sampleRangeDataSize))
                        {
                            return false;
                        }
                        if (!evaluationContext.EvaluateToGpuValues(
                                offlineMetricValues.data(),
                                offlineMetricValues.size()))
                        {
                            return false;
                        }
                        return true;
                    }();
                }
                std::fill(rowAggregateValues.begin(), rowAggregateValues.end(), 0.0);
                rowAggregateHasValue.assign(rowAggregates.size(), false);

                for (size_t metricIndex = 0; metricIndex < metricNamesForExport.size(); metricIndex++)
                {
                    double metricValue = 0.0;
                    if (!Codec::TryResolveGpuPmExportMetricValue(
                            exportPlan,
                            metricIndex,
                            gpuPmCountersSample.pSampleEvaluatedMetrics,
                            gpuPmCountersSample.sampleEvaluatedMetricCount,
                            offlineMetricValuesAvailable ? offlineMetricValues.data() : nullptr,
                            offlineMetricValuesAvailable ? offlineMetricValues.size() : 0,
                            metricValue))
                    {
                        continue;
                    }
                    emitMetricSample(metricIndex, gpuPmCountersSample.startTimestampUtcNsec, metricValue);

                    // Folded as the metric is resolved, so a metric that could not be resolved is left out of
                    // its row's value rather than counted as a zero.
                    for (const size_t aggregateIndex : aggregateIndicesForMetric[metricIndex])
                    {
                        double& aggregateValue = rowAggregateValues[aggregateIndex];
                        if (!rowAggregateHasValue[aggregateIndex])
                        {
                            aggregateValue = metricValue;
                            rowAggregateHasValue[aggregateIndex] = true;
                        }
                        else if (rowAggregates[aggregateIndex].sumAcrossMetrics)
                        {
                            aggregateValue += metricValue;
                        }
                        else
                        {
                            aggregateValue = std::max(aggregateValue, metricValue);
                        }
                    }
                }

                // One value per row, after every metric under it has been folded in.
                for (size_t aggregateIndex = 0; aggregateIndex < rowAggregates.size(); ++aggregateIndex)
                {
                    if (rowAggregateHasValue[aggregateIndex])
                    {
                        m_writer.AddCounterSample(
                            rowAggregates[aggregateIndex].track,
                            ExportTimestampNsec(gpuPmCountersSample.startTimestampUtcNsec),
                            rowAggregateValues[aggregateIndex]);
                    }
                }
            });

            // To normalize the y-axis of a percentage row to 100%, add a phony event to the end of such rows.
            for (size_t metricIndex = 0; metricIndex < metricNamesForExport.size(); metricIndex++)
            {
                const std::string& metricName = metricNamesForExport[metricIndex];

                if (metricName.find(".pct_of_peak_sustained_elapsed") == std::string::npos)
                {
                    continue;
                }

                if (previousSampleEndTimestampUtcNsec == Codec::InvalidTimestamp)
                {
                    continue;
                }

                const double metricValue = 100.0;
                emitMetricSample(metricIndex, previousSampleEndTimestampUtcNsec, metricValue);
            }
        }

        inline void AddGpuGrCtxTrace(TrackUuid groupTrack, size_t deviceOrdinal, const Codec::GpuPartitionData& gpuPartition)
        {
            const auto& gpuGrCtxTrace = gpuPartition.gpuGrCtxTrace;

            const TrackUuid gpuDiagnosticGroup = GetGpuDiagnosticsGroup(groupTrack);

            for (const Codec::GpuGrCtxTraceData::ProviderDiagnostic& providerDiagnostic : gpuGrCtxTrace.providerDiagnostics)
            {
                if (providerDiagnostic.recordType != NvTrace_GPU_GrCtxTrace_RecordType_ProviderDiagnostic_InitializationFailure)
                {
                    continue;
                }

                const uint64_t startUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(providerDiagnostic.timestampTsc);
                const uint64_t endUtcNsec = m_dataModel.m_observationEndUtcNsec;

                DiagnosticSeverity severity = DiagnosticSeverity::Error;
                const char* pMessage = "GPU GR Context Trace provider failed during initialization.";

                switch (providerDiagnostic.initializationFailureReason)
                {
                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_InsufficientDriverVersion:
                        pMessage = "GPU GR Context Trace provider is unavailable because the installed driver does not support CUDA GR context trace.";
                        break;

                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_ConfigurationInvalid:
                        pMessage = "GPU GR Context Trace provider failed because its configuration was invalid.";
                        break;

                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_RmDeviceLookupFailed:
                        pMessage = "GPU GR Context Trace provider failed to find the RM device for this GPU.";
                        break;

                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_RmEventBufferInitializationFailed:
                        pMessage = "GPU GR Context Trace provider failed to initialize the RM event buffer.";
                        break;

                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_InitializationFailed:
                        pMessage = "GPU GR Context Trace provider failed during initialization.";
                        break;

                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_None:
                    case NvTrace_GPU_GrCtxTrace_ProviderFailureReason_FORCE_INT:
                        continue;
                }

                std::vector<Annotation> annotations;
                annotations.push_back(m_writer.MakeUint("GPU", deviceOrdinal));
                annotations.push_back(m_writer.MakeUint("Timestamp TSC", providerDiagnostic.timestampTsc));

                AddNvTraceDiagnostic(gpuDiagnosticGroup, "GPU GR Context Trace Diagnostics", severity, "Initialization Failure", pMessage, startUtcNsec, endUtcNsec, annotations);
            }

            if (gpuGrCtxTrace.residentContexts.empty())
            {
                NVTRACE_LOG_INF(90, "GPU %zu; no GR Contexts", deviceOrdinal);
                return;
            }

            for (const auto& grCtx : gpuGrCtxTrace.residentContexts)
            {
                if (grCtx.pid == Codec::GpuGrCtxTraceData::HiddenPid)
                {
                    NVTRACE_LOG_INF(50, "GPU %zu; GR context pid is hidden. Skipping.", deviceOrdinal);
                    continue;
                }
                if (grCtx.pid == Codec::GpuGrCtxTraceData::UnknownPid)
                {
                    NVTRACE_LOG_INF(50, "GPU %zu; GR context pid is unknown. Skipping.", deviceOrdinal);
                    continue;
                }

                const uint64_t startTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, grCtx.ctxLoadTimestampGpuNsec);
                const uint64_t endTimestampUtcNsec = m_dataModel.ConvertGpuNsecToUtcNsec(gpuPartition, grCtx.ctxSaveTimestampGpuNsec);

                // grCtx.pid is the host PID; the user-facing PID is the correlated process's namespace PID.
                const auto* pProcess = grCtx.isMultipleEventsDropped ? nullptr : m_dataModel.FindProcess(grCtx.processLuid);
                std::string ctxNameLabel;
                if (pProcess)
                {
                    ctxNameLabel += pProcess->path;
                    ctxNameLabel += SprintfToString(" Process %d", pProcess->namespacePid);
                }
                else if (grCtx.isMultipleEventsDropped)
                {
                    // Dropped-events marker row: the GR-context event buffer overflowed and events were
                    // lost, so pid (and all args) are unknown. Label it explicitly so it is not mistaken
                    // for a correlation failure.
                    ctxNameLabel += "[GR context events dropped]";
                }
                else
                {
                    // No registered process matched (rmCid/globalPid/namespacePid all missed -- e.g. the
                    // in-driver provider never registered, or a sibling-container PID can't be correlated),
                    // but grCtx.pid is a real host PID (Hidden/Unknown pids were already skipped above).
                    // Attribute the context to that host PID instead of "??", so GPU activity is never
                    // left fully unattributed.
                    ctxNameLabel += SprintfToString("Host PID %llu", static_cast<unsigned long long>(grCtx.pid));
                }

                std::vector<Annotation> annotations;
                if (grCtx.isMultipleEventsDropped)
                {
                    annotations.push_back(m_writer.MakeString("vmId", kUnknownData));
                    annotations.push_back(m_writer.MakeString("channelId", kUnknownData));
                    annotations.push_back(m_writer.MakeString("rmSeqNum", "Multiple GR context events are dropped"));
                    annotations.push_back(m_writer.MakeString("namespacePid", kUnknownData));
                    annotations.push_back(m_writer.MakeString("globalPid", kUnknownData));
                    annotations.push_back(m_writer.MakeString("pidNamespaceId", kUnknownData));
                    annotations.push_back(m_writer.MakeString("pidNamespaceDeviceId", kUnknownData));
                }
                else
                {
                    annotations.push_back(m_writer.MakeUint("vmId", grCtx.vmid));
                    annotations.push_back(m_writer.MakeUint("channelId", grCtx.rmCid));
                    annotations.push_back(m_writer.MakeUint("rmSeqNum", grCtx.rmSequenceNumber)); // need for debugging
                    annotations.push_back(pProcess ? m_writer.MakeInt("namespacePid", pProcess->namespacePid) : m_writer.MakeString("namespacePid", kUnknownData));
                    annotations.push_back(m_writer.MakeUint("globalPid", grCtx.pid)); // host PID reported by the GR-context trace
                    // PID namespace identity (device, inode) disambiguates equal namespace PIDs across containers (0 => unavailable).
                    annotations.push_back(pProcess ? m_writer.MakeUint("pidNamespaceId", pProcess->pidNamespaceId.inode) : m_writer.MakeString("pidNamespaceId", kUnknownData));
                    annotations.push_back(pProcess ? m_writer.MakeUint("pidNamespaceDeviceId", pProcess->pidNamespaceId.device) : m_writer.MakeString("pidNamespaceDeviceId", kUnknownData));
                }

                const uint64_t durationNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                m_writer.AddSlice(GetSubTrack(groupTrack, "GPU GR Context Trace"), "gpu_grctx", ctxNameLabel, ExportTimestampNsec(startTimestampUtcNsec), durationNsec, std::move(annotations));
            }
        }

        inline void AddNvtxTraceEvents(TrackUuid overviewGroup, const Codec::ProcessData& processData, const std::unordered_set<uint64_t>& cudaThreadIds)
        {
            std::string messageStorage;
            const auto& nvtxTrace = processData.nvtxTrace;

            for (uint32_t domainCliqueIndex = 0; domainCliqueIndex < nvtxTrace.domainCliques.size(); domainCliqueIndex++)
            {
                const auto& domainClique = nvtxTrace.domainCliques[domainCliqueIndex];
                for (const auto& [uniqueThreadId, threadData] : domainClique.threads)
                {
                    const bool isCudaThread = cudaThreadIds.count(uniqueThreadId) != 0;
                    const std::string rowName = GetNameForNvtxTraceCpuTraceRow(uniqueThreadId, domainCliqueIndex, domainClique, nvtxTrace, isCudaThread);

                    for (const auto& mark : threadData.marks)
                    {
                        if (mark.timestampTsc == Codec::InvalidTimestamp)
                        {
                            continue;
                        }
                        const uint64_t timestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(mark.timestampTsc);
                        if (timestampUtcNsec < m_dataModel.m_observationStartUtcNsec)
                        {
                            continue;
                        }

                        const std::string domainName = GetNvtxTraceDomainExportName(nvtxTrace, mark.domainIndex);
                        const std::string nvtxCategory = GetNvtxTraceCategory(nvtxTrace, mark.domainIndex, "nvtx");

                        std::vector<Annotation> annotations;
                        annotations.push_back(m_writer.MakeString("Domain", domainName));

                        m_writer.AddInstant(
                            GetSubTrack(overviewGroup, rowName),
                            nvtxCategory,
                            m_dataModel.GetNvtxTraceMarkMessage(nvtxTrace, mark, messageStorage),
                            ExportTimestampNsec(timestampUtcNsec),
                            std::move(annotations));
                    }

                    threadData.ForEachObservedPushPopRange([&](const auto& pushPopRange) {
                        if (pushPopRange.startTimestampTsc == Codec::InvalidTimestamp || pushPopRange.endTimestampTsc == Codec::InvalidTimestamp)
                        {
                            return;
                        }
                        const uint64_t endTimestampUtcNsec = m_dataModel.ConvertTscTickToUtcNsec(pushPopRange.endTimestampTsc);
                        if (endTimestampUtcNsec < m_dataModel.m_observationStartUtcNsec)
                        {
                            return;
                        }
                        const uint64_t startTimestampUtcNsec = std::max(m_dataModel.m_observationStartUtcNsec, m_dataModel.ConvertTscTickToUtcNsec(pushPopRange.startTimestampTsc));
                        const uint64_t durNsec = endTimestampUtcNsec - startTimestampUtcNsec;

                        const std::string domainName = GetNvtxTraceDomainExportName(nvtxTrace, pushPopRange.domainIndex);
                        const std::string nvtxCategory = GetNvtxTraceCategory(nvtxTrace, pushPopRange.domainIndex, "nvtx");

                        std::vector<Annotation> annotations;
                        annotations.push_back(m_writer.MakeString("Domain", domainName));

                        const EventId eventId = m_writer.AddSlice(
                            GetSubTrack(overviewGroup, rowName),
                            nvtxCategory,
                            m_dataModel.GetNvtxTracePushPopRangeMessage(nvtxTrace, pushPopRange, messageStorage),
                            ExportTimestampNsec(startTimestampUtcNsec),
                            durNsec,
                            std::move(annotations));
                        AddNvtxTracePushPopRangeFlowSource(processData.processLuid, &pushPopRange, eventId);
                    });
                }
            }
        }

    private:
        PerfettoTraceWriter m_writer;

        TrackUuid m_traceRootTrack = 0;
        std::map<std::pair<TrackUuid, std::string>, TrackUuid> m_namedChildTrack;
        std::map<std::pair<TrackUuid, std::string>, TrackUuid> m_laneTrack;
        std::map<std::pair<TrackUuid, std::string>, TrackUuid> m_counterTrack;

        uint64_t m_nextAvailableFlowId = 0;
        std::map<uint64_t, std::vector<EventId>> m_hwCorrelationToFlowSourceEventIdsMap;
        std::map<std::tuple<uint64_t, NVTRACE_LUID>, std::vector<EventId>> m_apiCorrelationToFlowSourceEventIdsMap;
        std::map<std::tuple<const Codec::NvtxTraceData::PushPopRangeData*, NVTRACE_LUID>, std::vector<EventId>> m_nvtxTracePushPopRangeToFlowSourceEventIdsMap;
        std::map<std::tuple<NVTRACE_LUID, size_t>, std::vector<uint64_t>> m_deviceGraphLaunchToFlowIdsMap;
        std::map<size_t, std::map<uint64_t, uint64_t>> m_deviceEventToTimestampToFlowIdMap;
    };

    template <typename TTraceReader>
    inline void PerfettoTraceExporter<TTraceReader>::Export(const std::filesystem::path& archiveFilePath)
    {
        m_writer.SetTimestampOriginUtcNsec(m_dataModel.m_observationStartUtcNsec);

        NVTRACE_LOG_INF(50, "%zu CPU clock calibration samples", m_dataModel.cpuClockCalibration.calibrationPoints.size());
        NVTRACE_LOG_INF(50, "%zu GPU partitions", m_dataModel.gpuPartitions.size());
        NVTRACE_LOG_INF(50, "%zu processes", m_dataModel.processes.size());

        const TrackUuid rootTrack = GetTraceRootTrack();

        // Begin System Info
        const TrackUuid systemGroup = GetOrCreateNamedGroup(rootTrack, "System Info", kGroupOrderSystemInfo);

        AddTraceInfo(systemGroup);
        AddSystemInfo(systemGroup, m_dataModel.systemInfo);

        // GPU Info
        size_t deviceOrdinal = 0;
        for (const auto& [gpuPartitionLuid, gpuPartition] : m_dataModel.gpuPartitions)
        {
            AddGpuPartitionInfo(systemGroup, deviceOrdinal, gpuPartition);
            deviceOrdinal++;
        }
        // End System Info

        bool isFirst = true;
        int32_t processOrdinal = 0;
        // Iterate Processes
        for (const auto& [processLuid, process] : m_dataModel.processes)
        {
            const std::string overviewName = "Process " + std::to_string(process.namespacePid) + ": Overview";
            const std::string streamsName = "Process " + std::to_string(process.namespacePid) + ": Streams";
            const TrackUuid overviewGroup = GetOrCreateNamedGroup(rootTrack, overviewName, kGroupOrderOverview + processOrdinal);
            const TrackUuid streamsGroup = GetOrCreateNamedGroup(rootTrack, streamsName, kGroupOrderStreams + processOrdinal);

            if (isFirst)
            {
                isFirst = false;

                if (m_options.enableDebugEvents)
                {
                    // Add CPU clock calibration events
                    AddCpuClockCalibration(overviewGroup);
                }
            }

            NVTRACE_LOG_INF(70, "Process %d; path: %s", process.namespacePid, process.path.c_str());
            std::unordered_set<uint64_t> cudaThreadIds;
            cudaThreadIds = CollectCudaThreadIds(process.cudaApiTrace);
            AddCudaApiTraceEvents(overviewGroup, streamsGroup, process.namespacePid, processLuid, process.cudaApiTrace, process.nvtxTrace, cudaThreadIds);
            AddNvtxTraceEvents(overviewGroup, process, cudaThreadIds);

            processOrdinal++;
        }

        TrackUuid gpuEventGroup = 0;
        const std::string gpuEventTraceRowName = "GPU Event Trace";
        if (m_options.isolateGpuEventRows)
        {
            gpuEventGroup = GetOrCreateNamedGroup(rootTrack, gpuEventTraceRowName, kGroupOrderEventTrace);
        }

        // Add GPU HW trace events
        deviceOrdinal = 0;
        for (const auto& [gpuPartitionLuid, gpuPartition] : m_dataModel.gpuPartitions)
        {
            auto deviceOrdinalGuard = ScopeExitGuard([&]() { deviceOrdinal++; });

            const std::string gpuRowName = gpuPartition.info.chipName.empty()
                                             ? SprintfToString("GPU %zu - %s", deviceOrdinal, GetNodeId().c_str())
                                             : SprintfToString("GPU %zu: %s - %s", deviceOrdinal, gpuPartition.info.chipName.c_str(), GetNodeId().c_str());
            const TrackUuid deviceGroup = GetOrCreateNamedGroup(rootTrack, gpuRowName, kGroupOrderGpu + static_cast<int32_t>(deviceOrdinal));
            const TrackUuid gpuEventTrack = m_options.isolateGpuEventRows ? gpuEventGroup : deviceGroup;

            AddGpuEventTimeline(gpuEventTrack, deviceOrdinal, gpuPartition, NVTRACE_NIL_LUID, 0);
            AddDeviceGraphLaunchMarkers(gpuEventTrack, deviceOrdinal, gpuPartition);
            if (m_options.enableDebugEvents)
            {
                AddGpuHwTrace(deviceGroup, deviceOrdinal, gpuPartition, gpuPartition.gpuHwTrace);
            }

            if (m_options.enableDebugEvents)
            {
                AddGpuClockCalibration(deviceGroup, deviceOrdinal, gpuPartition.gpuClockCalibration);
            }

            AddGpuPmCounters(deviceGroup, deviceOrdinal, gpuPartition);
            AddGpuGrCtxTrace(deviceGroup, deviceOrdinal, gpuPartition);
        }

        // Write the Perfetto protobuf and its node-local clock metadata as a ZIP archive.
        const bool parentDirectoryReady = EnsureOutputParentDirectoryExists(archiveFilePath);
        if (!parentDirectoryReady)
        {
            return;
        }
        bool writeSucceeded = false;
        if (m_options.enableTraceArchive)
        {
            writeSucceeded = WriteArchive(archiveFilePath);
        }
        else
        {
            writeSucceeded = m_writer.WriteToFile(archiveFilePath);
        }
        if (!writeSucceeded)
        {
            NVTRACE_LOG_ERR(100, "Failed to write Perfetto trace archive: %s", archiveFilePath.string().c_str());
        }
    }

} // namespace nv::trace
