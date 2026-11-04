/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>
#include <NvTrace/Target/NvTraceClientLogger.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <nvperf_common.h>
#include <nvperf_device_host.h>
#include <nvperf_host.h>
#include <nvperf_host_impl.h>

#if defined(__linux__)
#include <climits>
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#endif

namespace nv::trace::Codec {

    /// How exporters resolve GPU PM metric values:
    /// - Export every metricsToConfigure name (collection set).
    /// - Prefer live evaluatedMetrics for names listed in metricsToEvaluate when present.
    /// - Offline-evaluate the remaining configure names (or all configure names if live data is unusable).
    /// After PrepareGpuPmMetricsForExport, offline-only names that failed soft evaluation are dropped.
    struct GpuPmExportMetricPlan
    {
        std::vector<std::string> metricNamesForExport;
        /// Per export index: the row label the metric set asked for, "Metric (SubMetric)". Empty when the
        /// config had nothing to say, leaving the counter name as the only label.
        std::vector<std::string> rowLabelsForExport;
        /// Per export index: the group the metric belongs to. Kept apart from the label so a viewer can put
        /// the metric under a heading rather than spelling the category into every row name. Empty only when
        /// the label is: a labelled metric always has a group, falling back to "Other".
        std::vector<std::string> rowCategoriesForExport;
        /// Per export index: the timeline metric this counter belongs to, which is what rows reference.
        /// One timeline metric spans several counters, so this is coarser than rowLabelsForExport.
        /// Empty for a counter no metric claims, and for traces with no timeline section.
        std::vector<std::string> rowMetricNamesForExport;
        /// Per export index: the sub-metric this counter is, so a row drawing only some of them can filter.
        std::vector<std::string> rowSubMetricNamesForExport;
        std::vector<std::string> metricNamesForOfflineEvaluation;
        /// Per export index: index into metricsToEvaluate / live evaluatedMetrics, or SIZE_MAX.
        std::vector<size_t> liveEvalIndexByExportIndex;
        /// Per export index: index into the context's one-sample offline metric buffer, or SIZE_MAX.
        std::vector<size_t> offlineIndexByExportIndex;
        /// True when metricsToEvaluate is non-empty and every sample has matching evaluatedMetrics.
        bool liveMetricsAvailable = false;
    };

    inline bool TryResolveGpuPmExportMetricValue(
        const GpuPmExportMetricPlan& plan,
        size_t exportMetricIndex,
        const double* pLiveMetricValues,
        size_t numLiveMetricValues,
        const double* pOfflineMetricValues,
        size_t numOfflineMetricValues,
        double& metricValue)
    {
        const size_t liveIndex = plan.liveEvalIndexByExportIndex[exportMetricIndex];
        if (pLiveMetricValues && liveIndex < numLiveMetricValues)
        {
            metricValue = pLiveMetricValues[liveIndex];
            return true;
        }
        const size_t offlineIndex = plan.offlineIndexByExportIndex[exportMetricIndex];
        if (!pOfflineMetricValues || offlineIndex >= numOfflineMetricValues)
        {
            return false;
        }
        metricValue = pOfflineMetricValues[offlineIndex];
        return true;
    }

    namespace {

        const char* NvpwStatusName(NVPA_Status status)
        {
            const char* pStatusStr = nullptr;
            const char* pCommentStr = nullptr;
            NVPW_NVPAStatusToString(status, &pStatusStr, &pCommentStr);
            return pStatusStr ? pStatusStr : "unknown";
        }

#if defined(__linux__)
        constexpr const char* kPerfWorksHostLibFileName = "libnvperf_grfx_host.so";
#else
        constexpr const char* kPerfWorksHostLibFileName = "nvperf_grfx_host";
#endif

        void AppendUniqueSearchPath(std::vector<std::string>& paths, const std::string& path)
        {
            if (path.empty())
            {
                return;
            }
            if (std::find(paths.begin(), paths.end(), path) != paths.end())
            {
                return;
            }
            paths.push_back(path);
        }

        std::string GetCallingModuleDirectoryPath()
        {
            Dl_info moduleInfo{};
            link_map* pModuleLinkMap = nullptr;
            if (dladdr1(reinterpret_cast<void*>(&GetCallingModuleDirectoryPath), &moduleInfo, reinterpret_cast<void**>(&pModuleLinkMap), RTLD_DL_LINKMAP) == 0
                || !pModuleLinkMap)
            {
                return {};
            }

            // The loader names the main executable by argv[0], which has no directory when it was found through PATH.
            std::string modulePath;
            if (pModuleLinkMap->l_name[0] == '\0')
            {
                char executablePath[PATH_MAX] = {};
                const ssize_t executablePathLength = readlink("/proc/self/exe", executablePath, sizeof(executablePath) - 1);
                if (executablePathLength <= 0)
                {
                    return {};
                }
                modulePath.assign(executablePath, static_cast<size_t>(executablePathLength));
            }
            else
            {
                modulePath = pModuleLinkMap->l_name;
            }

            if (const size_t slashIndex = modulePath.rfind('/'); slashIndex != std::string::npos)
            {
                modulePath.resize(slashIndex);
            }
            return modulePath;
        }

        void CollectPerfWorksLibrarySearchPaths(std::vector<std::string>& paths)
        {
            if (const char* pEnvPath = std::getenv("NVTRACE_PERFWORKS_LIBRARY_PATH"); pEnvPath && pEnvPath[0] != '\0')
            {
                AppendUniqueSearchPath(paths, pEnvPath);
            }

            AppendUniqueSearchPath(paths, GetCallingModuleDirectoryPath());
        }

        // Perfetto / continuous-profiling export runs in the client process. The out-of-process
        // service initializes PerfWorks there, but offline evaluation needs host init here too.
        bool EnsurePerfWorksHostInitialized()
        {
            static const std::vector<std::string> s_libraryPathStrings = []() {
                std::vector<std::string> libraryPathStrings;
                CollectPerfWorksLibrarySearchPaths(libraryPathStrings);
                return libraryPathStrings;
            }();

            static const bool s_initialized = [&]() -> bool {
                bool searchPathsApplied = false;
                if (!s_libraryPathStrings.empty())
                {
                    std::vector<const char*> libraryPaths;
                    libraryPaths.reserve(s_libraryPathStrings.size());
                    for (const std::string& libraryPath : s_libraryPathStrings)
                    {
                        libraryPaths.push_back(libraryPath.c_str());
                    }

                    NVPW_SetLibraryLoadPaths_Params setLibraryLoadPathsParams = {NVPW_SetLibraryLoadPaths_Params_STRUCT_SIZE};
                    setLibraryLoadPathsParams.numPaths = static_cast<uint32_t>(libraryPaths.size());
                    setLibraryLoadPathsParams.ppPaths = libraryPaths.data();
                    searchPathsApplied = NVPW_SetLibraryLoadPaths(&setLibraryLoadPathsParams) == NVPA_STATUS_SUCCESS;
                }

                NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
                NVPA_Status status = NVPW_InitializeHost(&initializeHostParams);
                if (status != NVPA_STATUS_SUCCESS && searchPathsApplied)
                {
                    // Explicit search paths suppress the loader's default lookup, so retry
                    // without them to still honor the platform library search path.
                    NVPW_SetLibraryLoadPaths_Params clearLibraryLoadPathsParams = {NVPW_SetLibraryLoadPaths_Params_STRUCT_SIZE};
                    if (NVPW_SetLibraryLoadPaths(&clearLibraryLoadPathsParams) == NVPA_STATUS_SUCCESS)
                    {
                        status = NVPW_InitializeHost(&initializeHostParams);
                    }
                }
                if (status != NVPA_STATUS_SUCCESS)
                {
                    fprintf(stderr, "Warning: NVPW_InitializeHost failed: %s (%u); GPU PM metrics that need offline evaluation will be missing. Set NVTRACE_PERFWORKS_LIBRARY_PATH or place %s beside the module.\n", NvpwStatusName(status), static_cast<uint32_t>(status), kPerfWorksHostLibFileName);
                    return false;
                }
                return true;
            }();
            return s_initialized;
        }

        // After conversion, metricEvalRequests[index] corresponds to metricNames[metricEvalNameIndices[index]].
        // Failed names are skipped (warned) and omitted from both output vectors.
        void ConvertMetricNamesToEvalRequests(
            NVPW_MetricsEvaluator* pMetricsEvaluator,
            const std::vector<std::string>& metricNames,
            std::vector<NVPW_MetricEvalRequest>& metricEvalRequests,
            std::vector<size_t>& metricEvalNameIndices)
        {
            metricEvalRequests.clear();
            metricEvalNameIndices.clear();
            metricEvalRequests.reserve(metricNames.size());
            metricEvalNameIndices.reserve(metricNames.size());
            for (size_t metricNameIndex = 0; metricNameIndex < metricNames.size(); ++metricNameIndex)
            {
                const std::string& metricName = metricNames[metricNameIndex];
                NVPW_MetricEvalRequest metricEvalRequest = {};
                NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertParams = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
                convertParams.pMetricsEvaluator = pMetricsEvaluator;
                convertParams.pMetricName = metricName.c_str();
                convertParams.pMetricEvalRequest = &metricEvalRequest;
                convertParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
                const NVPA_Status status = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertParams);
                if (status != NVPA_STATUS_SUCCESS)
                {
                    NVTRACE_LOG_WRN(50, "Skipping unavailable GPU PM offline metric '%s': %s (%u)", metricName.c_str(), NvpwStatusName(status), static_cast<uint32_t>(status));
                    continue;
                }
                metricEvalRequests.push_back(metricEvalRequest);
                metricEvalNameIndices.push_back(metricNameIndex);
            }
        }

    } // namespace

    // Encapsulates the CounterData image, NvPerfMetricsEvaluator, surviving metric names, and metricEvalRequests.
    // Does not own the per-range output buffer; callers allocate and reuse that storage.
    class MetricsEvaluationContext
    {
    public:
        MetricsEvaluationContext() = default;
        MetricsEvaluationContext(const MetricsEvaluationContext&) = delete;
        MetricsEvaluationContext& operator=(const MetricsEvaluationContext&) = delete;

        // Move must transfer the CounterData image and evaluator without copying; then leave
        // the source fully Reset so stale sizes / availability flags cannot outlive the buffers.
        MetricsEvaluationContext(MetricsEvaluationContext&& other) noexcept
            : m_metricsEvaluator(std::move(other.m_metricsEvaluator))
            , m_counterDataImage(std::move(other.m_counterDataImage))
            , m_metricEvalRequests(std::move(other.m_metricEvalRequests))
            , m_metricNames(std::move(other.m_metricNames))
            , m_counterDataPreambleSize(other.m_counterDataPreambleSize)
            , m_sampleRangeDataSize(other.m_sampleRangeDataSize)
            , m_perfWorksHostAvailable(other.m_perfWorksHostAvailable)
        {
            other.Reset();
        }

        MetricsEvaluationContext& operator=(MetricsEvaluationContext&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }
            Reset();
            m_metricsEvaluator = std::move(other.m_metricsEvaluator);
            m_counterDataImage = std::move(other.m_counterDataImage);
            m_metricEvalRequests = std::move(other.m_metricEvalRequests);
            m_metricNames = std::move(other.m_metricNames);
            m_counterDataPreambleSize = other.m_counterDataPreambleSize;
            m_sampleRangeDataSize = other.m_sampleRangeDataSize;
            m_perfWorksHostAvailable = other.m_perfWorksHostAvailable;
            other.Reset();
            return *this;
        }

        /// Hands the set's definitions to this evaluator, the same Initialize / ExecuteScriptFromString /
        /// Commit sequence the provider uses. Returns false if the library rejected any step.
        bool RegisterCustomMetrics(const std::string& customMetricsScript)
        {
            NVPW_MetricsEvaluator_UserDefinedMetrics_Initialize_Params initializeParams = {NVPW_MetricsEvaluator_UserDefinedMetrics_Initialize_Params_STRUCT_SIZE};
            initializeParams.pMetricsEvaluator = m_metricsEvaluator;
            NVPA_Status status = NVPW_MetricsEvaluator_UserDefinedMetrics_Initialize(&initializeParams);
            if (status != NVPA_STATUS_SUCCESS)
            {
                NVTRACE_LOG_WRN(50, "NVPW_MetricsEvaluator_UserDefinedMetrics_Initialize failed: %s (%u)", NvpwStatusName(status), static_cast<uint32_t>(status));
                return false;
            }

            NVPW_MetricsEvaluator_UserDefinedMetrics_ExecuteScriptFromString_Params executeParams = {NVPW_MetricsEvaluator_UserDefinedMetrics_ExecuteScriptFromString_Params_STRUCT_SIZE};
            executeParams.pMetricsEvaluator = m_metricsEvaluator;
            executeParams.pScriptString = customMetricsScript.c_str();
            status = NVPW_MetricsEvaluator_UserDefinedMetrics_ExecuteScriptFromString(&executeParams);
            if (status != NVPA_STATUS_SUCCESS)
            {
                NVTRACE_LOG_WRN(50, "NVPW_MetricsEvaluator_UserDefinedMetrics_ExecuteScriptFromString failed: %s (%u)", NvpwStatusName(status), static_cast<uint32_t>(status));
                return false;
            }

            NVPW_MetricsEvaluator_UserDefinedMetrics_Commit_Params commitParams = {NVPW_MetricsEvaluator_UserDefinedMetrics_Commit_Params_STRUCT_SIZE};
            commitParams.pMetricsEvaluator = m_metricsEvaluator;
            status = NVPW_MetricsEvaluator_UserDefinedMetrics_Commit(&commitParams);
            if (status != NVPA_STATUS_SUCCESS)
            {
                NVTRACE_LOG_WRN(50, "NVPW_MetricsEvaluator_UserDefinedMetrics_Commit failed: %s (%u)", NvpwStatusName(status), static_cast<uint32_t>(status));
                return false;
            }
            return true;
        }

        template <typename TTraceReader>
        bool Initialize(
            const DataModel<TTraceReader>& dataModel,
            const GpuPmCountersData& gpuPmCountersData,
            const char* pChipName,
            const std::vector<std::string>& metricNames)
        {
            Reset();
            if (metricNames.empty())
            {
                return true;
            }
            if (!pChipName || !pChipName[0]
                || !gpuPmCountersData.counterDataPreambleSize
                || !gpuPmCountersData.sampleRangeDataSize)
            {
                NVTRACE_LOG_ERR(20, "Invalid GPU PM offline evaluator context arguments");
                return false;
            }

            // We only need one temp range to evaluate from.
            constexpr size_t counterDataCapacity = 1;
            if (!dataModel.InitializeGpuPmCounterData(
                    gpuPmCountersData,
                    m_counterDataImage,
                    counterDataCapacity))
            {
                return false;
            }
            m_counterDataPreambleSize = gpuPmCountersData.counterDataPreambleSize;
            m_sampleRangeDataSize = gpuPmCountersData.sampleRangeDataSize;

            if (!EnsurePerfWorksHostInitialized())
            {
                NVTRACE_LOG_WRN(50, "GPU PM offline evaluation unavailable (PerfWorks host not loaded); exporting live metrics only");
                Reset();
                return true;
            }
            m_perfWorksHostAvailable = true;
            if (!m_metricsEvaluator.Initialize(pChipName))
            {
                NVTRACE_LOG_WRN(50, "GPU PM offline evaluation unavailable (evaluator initialization failed); exporting live metrics only");
                Reset();
                return true;
            }

            // A set may define metrics of its own. Once registered they resolve like native ones, but
            // registration is per evaluator and this is not the provider's, so without it every name the
            // definitions introduce is dropped as unresolvable below and its row is exported empty.
            if (!gpuPmCountersData.customMetricsScript.empty() && !RegisterCustomMetrics(gpuPmCountersData.customMetricsScript))
            {
                NVTRACE_LOG_WRN(50, "Custom metric definitions could not be registered; metrics built on them will not be evaluated offline");
            }

            // MetricsEvaluator device attributes can be set from just a CounterDataPreamble
            // (e.g. FloorsweepInfo). There is no need to seed range 0 with real sample bytes first.
            NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttributesParams = {NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
            setDeviceAttributesParams.pMetricsEvaluator = m_metricsEvaluator;
            setDeviceAttributesParams.pCounterDataImage = m_counterDataImage.data();
            setDeviceAttributesParams.counterDataImageSize = m_counterDataImage.size();
            const NVPA_Status setDeviceAttributesStatus = NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttributesParams);
            if (setDeviceAttributesStatus != NVPA_STATUS_SUCCESS)
            {
                NVTRACE_LOG_WRN(50, "NVPW_MetricsEvaluator_SetDeviceAttributes failed: %s (%u)", NvpwStatusName(setDeviceAttributesStatus), static_cast<uint32_t>(setDeviceAttributesStatus));
                Reset();
                return true;
            }

            std::vector<size_t> metricEvalNameIndices;
            ConvertMetricNamesToEvalRequests(
                m_metricsEvaluator,
                metricNames,
                m_metricEvalRequests,
                metricEvalNameIndices);
            // Prepare a compacted vector of surviving metric names.
            m_metricNames.reserve(metricEvalNameIndices.size());
            for (const size_t metricNameIndex : metricEvalNameIndices)
            {
                m_metricNames.push_back(metricNames[metricNameIndex]);
            }
            return true;
        }

        void Reset()
        {
            m_metricsEvaluator.Reset();
            m_counterDataImage.clear();
            m_metricEvalRequests.clear();
            m_metricNames.clear();
            m_counterDataPreambleSize = 0;
            m_sampleRangeDataSize = 0;
            m_perfWorksHostAvailable = false;
        }

        bool IsPerfWorksHostAvailable() const
        {
            return m_perfWorksHostAvailable;
        }

        const std::vector<std::string>& GetMetricNames() const
        {
            return m_metricNames;
        }

        // Copy the current sample's CounterData.RangeData into the zeroth range of the temp CounterDataImage.
        bool SetRangeData(const uint8_t* pRangeData, size_t sampleRangeDataSize)
        {
            if (!pRangeData
                || sampleRangeDataSize != m_sampleRangeDataSize
                || m_counterDataImage.size() < m_counterDataPreambleSize + m_sampleRangeDataSize)
            {
                NVTRACE_LOG_ERR(20, "Invalid GPU PM RangeData");
                return false;
            }
            std::memcpy(
                m_counterDataImage.data() + m_counterDataPreambleSize,
                pRangeData,
                m_sampleRangeDataSize);
            return true;
        }

        // Evaluate from range 0 into the caller-provided buffer (overwritten with the current range's values).
        bool EvaluateToGpuValues(double* pMetricValues, size_t numMetricValues) const
        {
            if (m_metricEvalRequests.empty())
            {
                return numMetricValues == 0;
            }
            if (!pMetricValues || numMetricValues != m_metricEvalRequests.size())
            {
                NVTRACE_LOG_ERR(20, "EvaluateToGpuValues called with invalid output storage");
                return false;
            }

            NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateParams = {NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
            evaluateParams.pMetricsEvaluator = m_metricsEvaluator;
            evaluateParams.pMetricEvalRequests = m_metricEvalRequests.data();
            evaluateParams.numMetricEvalRequests = m_metricEvalRequests.size();
            evaluateParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
            evaluateParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
            evaluateParams.pCounterDataImage = m_counterDataImage.data();
            evaluateParams.counterDataImageSize = m_counterDataImage.size();
            evaluateParams.rangeIndex = 0;
            evaluateParams.pMetricValues = pMetricValues;
            const NVPA_Status status = NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateParams);
            if (status != NVPA_STATUS_SUCCESS)
            {
                NVTRACE_LOG_WRN(50, "NVPW_MetricsEvaluator_EvaluateToGpuValues failed: %s (%u)", NvpwStatusName(status), static_cast<uint32_t>(status));
                return false;
            }
            return true;
        }

    private:
        class NvPerfMetricsEvaluator
        {
        private:
            // Prevent accidental use of "delete" keyword on this class' implicit conversions.
            // Introducing a second 'operator CompileErrorOnOperatorDelete*()' triggers an 'ambiguous conversion to void*'
            // on the 'delete', which catches the usage error at compile time.  c.f. http://stackoverflow.com/a/3312507
            struct CompileErrorOnOperatorDelete;
            operator CompileErrorOnOperatorDelete*() const;

            NvPerfMetricsEvaluator(const NvPerfMetricsEvaluator&) = delete;
            NvPerfMetricsEvaluator& operator=(const NvPerfMetricsEvaluator&) = delete;

        public:
            NvPerfMetricsEvaluator() = default;

            ~NvPerfMetricsEvaluator()
            {
                Reset();
            }

            NvPerfMetricsEvaluator(NvPerfMetricsEvaluator&& other) noexcept
                : m_pMetricsEvaluator(other.m_pMetricsEvaluator)
                , m_scratchBuffer(std::move(other.m_scratchBuffer))
            {
                other.m_pMetricsEvaluator = nullptr;
                other.m_scratchBuffer.clear();
            }

            NvPerfMetricsEvaluator& operator=(NvPerfMetricsEvaluator&& other) noexcept
            {
                if (this == &other)
                {
                    return *this;
                }
                Reset();
                m_pMetricsEvaluator = other.m_pMetricsEvaluator;
                m_scratchBuffer = std::move(other.m_scratchBuffer);
                other.m_pMetricsEvaluator = nullptr;
                other.m_scratchBuffer.clear();
                return *this;
            }

            operator NVPW_MetricsEvaluator*() const
            {
                return m_pMetricsEvaluator;
            }

            void Reset()
            {
                if (m_pMetricsEvaluator)
                {
                    NVPW_MetricsEvaluator_Destroy_Params destroyParams = {NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
                    destroyParams.pMetricsEvaluator = m_pMetricsEvaluator;
                    const NVPA_Status status = NVPW_MetricsEvaluator_Destroy(&destroyParams);
                    if (status != NVPA_STATUS_SUCCESS)
                    {
                        NVTRACE_LOG_ERR(20, "NVPW_MetricsEvaluator_Destroy failed: %s (%u)", NvpwStatusName(status), static_cast<uint32_t>(status));
                    }
                    m_pMetricsEvaluator = nullptr;
                }
                m_scratchBuffer.clear();
            }

            bool Initialize(const char* pChipName)
            {
                Reset();

                NVPW_Device_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParams = {NVPW_Device_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
                calculateScratchBufferSizeParams.pChipName = pChipName;
                NVPA_Status status = NVPW_Device_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParams);
                if (status != NVPA_STATUS_SUCCESS || !calculateScratchBufferSizeParams.scratchBufferSize)
                {
                    NVTRACE_LOG_WRN(50, "NVPW_Device_MetricsEvaluator_CalculateScratchBufferSize failed for chip '%s': %s (%u)", pChipName, NvpwStatusName(status), static_cast<uint32_t>(status));
                    return false;
                }

                m_scratchBuffer.resize(calculateScratchBufferSizeParams.scratchBufferSize);
                NVPW_Device_MetricsEvaluator_Initialize_Params initializeParams = {NVPW_Device_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
                initializeParams.pScratchBuffer = m_scratchBuffer.data();
                initializeParams.scratchBufferSize = m_scratchBuffer.size();
                initializeParams.pChipName = pChipName;
                status = NVPW_Device_MetricsEvaluator_Initialize(&initializeParams);
                if (status != NVPA_STATUS_SUCCESS)
                {
                    m_scratchBuffer.clear();
                    NVTRACE_LOG_WRN(50, "NVPW_Device_MetricsEvaluator_Initialize failed for chip '%s': %s (%u)", pChipName, NvpwStatusName(status), static_cast<uint32_t>(status));
                    return false;
                }

                m_pMetricsEvaluator = initializeParams.pMetricsEvaluator;
                return true;
            }

        private:
            NVPW_MetricsEvaluator* m_pMetricsEvaluator = nullptr;
            std::vector<uint8_t> m_scratchBuffer;
        };

        NvPerfMetricsEvaluator m_metricsEvaluator;
        std::vector<uint8_t> m_counterDataImage;
        std::vector<NVPW_MetricEvalRequest> m_metricEvalRequests;
        std::vector<std::string> m_metricNames;
        size_t m_counterDataPreambleSize = 0;
        size_t m_sampleRangeDataSize = 0;
        bool m_perfWorksHostAvailable = false;
    };

    /// Plan plus a reusable one-range context for one GPU partition export.
    struct GpuPmExportPrepared
    {
        GpuPmExportMetricPlan plan;
        MetricsEvaluationContext evaluationContext;
    };

    /// Builds the live/offline export plan and initializes a reusable one-range context.
    /// Offline-only metrics that cannot be converted are dropped. logContext prefixes warnings
    /// (for example "GPU 0" or "GPU partition 42").
    template <typename TTraceReader>
    GpuPmExportPrepared PrepareGpuPmMetricsForExport(
        const DataModel<TTraceReader>& dataModel,
        const GpuPartitionData& gpuPartitionData,
        const std::string& chipName,
        const char* logContext)
    {
        GpuPmExportPrepared prepared;
        const GpuPmCountersData& gpuPmCountersData = gpuPartitionData.gpuPmCounters;

        prepared.plan.liveMetricsAvailable = !gpuPmCountersData.metricsToEvaluate.empty();
        if (prepared.plan.liveMetricsAvailable)
        {
            for (const uint64_t sampleRecordLocation : gpuPmCountersData.sampleRecordLocations)
            {
                const GpuPmCountersSample sample = dataModel.GetGpuPmCountersSample(gpuPartitionData, sampleRecordLocation);
                if (!sample.pSampleEvaluatedMetrics || sample.sampleEvaluatedMetricCount != gpuPmCountersData.metricsToEvaluate.size())
                {
                    prepared.plan.liveMetricsAvailable = false;
                    break;
                }
            }
        }

        prepared.plan.metricNamesForExport.reserve(gpuPmCountersData.metricsToConfigure.size());
        prepared.plan.rowLabelsForExport.reserve(gpuPmCountersData.metricsToConfigure.size());
        prepared.plan.rowCategoriesForExport.reserve(gpuPmCountersData.metricsToConfigure.size());
        // The timeline section names metrics by baseId plus a sub-metric suffix, which is how the counter
        // name was built in the first place, so it maps back onto what is being exported.
        // counter name -> { row label, timeline metric name, sub-metric name }
        std::unordered_map<std::string, std::tuple<std::string, std::string, std::string>> timelineLabelByCounterName;
        for (const auto& timelineMetric : gpuPmCountersData.timeline.metrics)
        {
            for (const auto& subMetric : timelineMetric.subMetrics)
            {
                // A sub-metric may resolve against a different metric than its parent, so the joined form is
                // only a fallback for traces written before fullId was carried.
                const std::string counterName = !subMetric.fullId.empty() ? subMetric.fullId : timelineMetric.baseId + subMetric.suffix;
                const std::string label = subMetric.name.empty() ? timelineMetric.name : timelineMetric.name + " (" + subMetric.name + ")";
                timelineLabelByCounterName.emplace(counterName, std::make_tuple(label, timelineMetric.name, subMetric.name));
            }
        }

        prepared.plan.rowMetricNamesForExport.reserve(gpuPmCountersData.metricsToConfigure.size());
        prepared.plan.rowSubMetricNamesForExport.reserve(gpuPmCountersData.metricsToConfigure.size());
        for (const auto& metricToConfigure : gpuPmCountersData.metricsToConfigure)
        {
            prepared.plan.metricNamesForExport.push_back(metricToConfigure.metricName);
            const auto timelineIt = timelineLabelByCounterName.find(metricToConfigure.metricName);
            if (timelineIt != timelineLabelByCounterName.end())
            {
                prepared.plan.rowLabelsForExport.push_back(std::get<0>(timelineIt->second));
                prepared.plan.rowMetricNamesForExport.push_back(std::get<1>(timelineIt->second));
                prepared.plan.rowSubMetricNamesForExport.push_back(std::get<2>(timelineIt->second));
                prepared.plan.rowCategoriesForExport.emplace_back();
                continue;
            }
            // No timeline section: a trace from before it was written says the same thing one metric at a
            // time. A named metric falls back to "Other" rather than sitting loose, which would read as a
            // viewer bug. A metric with no name at all (architectures[]) gets neither.
            const std::string& displayName = metricToConfigure.displayName;
            prepared.plan.rowLabelsForExport.push_back(displayName);
            prepared.plan.rowMetricNamesForExport.emplace_back();
            prepared.plan.rowSubMetricNamesForExport.emplace_back();
            prepared.plan.rowCategoriesForExport.push_back(
                displayName.empty() ? std::string() : (metricToConfigure.category.empty() ? std::string("Other") : metricToConfigure.category));
        }
        prepared.plan.liveEvalIndexByExportIndex.assign(prepared.plan.metricNamesForExport.size(), SIZE_MAX);
        prepared.plan.offlineIndexByExportIndex.assign(prepared.plan.metricNamesForExport.size(), SIZE_MAX);

        for (size_t exportIndex = 0; exportIndex < prepared.plan.metricNamesForExport.size(); ++exportIndex)
        {
            const std::string& metricName = prepared.plan.metricNamesForExport[exportIndex];
            if (prepared.plan.liveMetricsAvailable)
            {
                const auto liveIt = std::find(
                    gpuPmCountersData.metricsToEvaluate.begin(),
                    gpuPmCountersData.metricsToEvaluate.end(),
                    metricName);
                if (liveIt != gpuPmCountersData.metricsToEvaluate.end())
                {
                    prepared.plan.liveEvalIndexByExportIndex[exportIndex] = static_cast<size_t>(
                        std::distance(gpuPmCountersData.metricsToEvaluate.begin(), liveIt));
                    continue;
                }
            }

            prepared.plan.offlineIndexByExportIndex[exportIndex] = prepared.plan.metricNamesForOfflineEvaluation.size();
            prepared.plan.metricNamesForOfflineEvaluation.push_back(metricName);
        }

        if (!prepared.plan.metricNamesForOfflineEvaluation.empty())
        {
            if (chipName.empty())
            {
                NVTRACE_LOG_WRN(50, "%s; chip name empty; skipping offline GPU PM metrics", logContext);
            }
            else if (!prepared.evaluationContext.Initialize(
                         dataModel,
                         gpuPmCountersData,
                         chipName.c_str(),
                         prepared.plan.metricNamesForOfflineEvaluation))
            {
                NVTRACE_LOG_WRN(50, "%s; offline GPU PM CounterData reconstruction failed; exporting live metrics only", logContext);
            }
        }

        const std::vector<std::string>& survivingOfflineMetricNames =
            prepared.evaluationContext.GetMetricNames();
        std::vector<std::string> newExportNames;
        // Rebuilt alongside the names: every one of these is parallel to metricNamesForExport by index, so
        // dropping a name without dropping its entry here shifts every later metric onto the wrong row.
        std::vector<std::string> newRowLabels;
        std::vector<std::string> newRowMetricNames;
        std::vector<std::string> newRowSubMetricNames;
        std::vector<std::string> newRowCategories;
        std::vector<size_t> newLiveIdx;
        std::vector<size_t> newOfflineIdx;
        std::vector<std::string> newOfflineNames;
        newExportNames.reserve(prepared.plan.metricNamesForExport.size());

        for (size_t exportIndex = 0; exportIndex < prepared.plan.metricNamesForExport.size(); ++exportIndex)
        {
            const size_t liveIndex = prepared.plan.liveEvalIndexByExportIndex[exportIndex];
            if (liveIndex != SIZE_MAX)
            {
                newExportNames.push_back(prepared.plan.metricNamesForExport[exportIndex]);
                newRowLabels.push_back(prepared.plan.rowLabelsForExport[exportIndex]);
                newRowMetricNames.push_back(prepared.plan.rowMetricNamesForExport[exportIndex]);
                newRowSubMetricNames.push_back(prepared.plan.rowSubMetricNamesForExport[exportIndex]);
                newRowCategories.push_back(prepared.plan.rowCategoriesForExport[exportIndex]);
                newLiveIdx.push_back(liveIndex);
                newOfflineIdx.push_back(SIZE_MAX);
                continue;
            }

            const auto& metricName = prepared.plan.metricNamesForExport[exportIndex];
            const auto it = std::find(
                survivingOfflineMetricNames.begin(),
                survivingOfflineMetricNames.end(),
                metricName);
            if (it == survivingOfflineMetricNames.end())
            {
                continue;
            }
            newExportNames.push_back(metricName);
            newRowLabels.push_back(prepared.plan.rowLabelsForExport[exportIndex]);
            newRowMetricNames.push_back(prepared.plan.rowMetricNamesForExport[exportIndex]);
            newRowSubMetricNames.push_back(prepared.plan.rowSubMetricNamesForExport[exportIndex]);
            newRowCategories.push_back(prepared.plan.rowCategoriesForExport[exportIndex]);
            newLiveIdx.push_back(SIZE_MAX);
            newOfflineIdx.push_back(static_cast<size_t>(std::distance(survivingOfflineMetricNames.begin(), it)));
            newOfflineNames.push_back(metricName);
        }

        prepared.plan.metricNamesForExport = std::move(newExportNames);
        prepared.plan.rowLabelsForExport = std::move(newRowLabels);
        prepared.plan.rowMetricNamesForExport = std::move(newRowMetricNames);
        prepared.plan.rowSubMetricNamesForExport = std::move(newRowSubMetricNames);
        prepared.plan.rowCategoriesForExport = std::move(newRowCategories);
        prepared.plan.liveEvalIndexByExportIndex = std::move(newLiveIdx);
        prepared.plan.offlineIndexByExportIndex = std::move(newOfflineIdx);
        prepared.plan.metricNamesForOfflineEvaluation = std::move(newOfflineNames);
        return prepared;
    }

} // namespace nv::trace::Codec
