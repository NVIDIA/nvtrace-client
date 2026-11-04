/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <stdint.h>
#include <string>
#include <system_error>


#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>

#include <NvTrace/Target/NvTraceClientLogger.h>

#include <NvTrace/Offline/DataModel/NvTraceDataModel.h>
#include <NvTrace/Offline/DataModel/NvTraceDataModelBuilderBase.h>
#include <NvTrace/Offline/Exporter/NvTraceDemanglerConvenience.h>
#include <NvTrace/Offline/NvTracePrintHelpers.h>

namespace nv::trace {

    inline const char* GetAddressKindName(enum NvTrace_CUDA_ApiTrace_AddressKind addressKind)
    {
        switch (addressKind)
        {
            case NvTrace_CUDA_ApiTrace_AddressKind_Invalid:
            {
                return "Invalid";
            }
            case NvTrace_CUDA_ApiTrace_AddressKind_Host:
            {
                return "Host";
            }
            case NvTrace_CUDA_ApiTrace_AddressKind_Device:
            {
                return "Device";
            }
            case NvTrace_CUDA_ApiTrace_AddressKind_Array:
            {
                return "Array";
            }
            case NvTrace_CUDA_ApiTrace_AddressKind_COUNT:
            case NvTrace_CUDA_ApiTrace_AddressKind_FORCE_INT:
            {
                break;
            }
        }
        return "Invalid";
    }

    inline const char* CommandTypeToCString(NvTrace_CUDA_ApiTrace_CommandType commandType)
    {
        switch (commandType)
        {
            case NvTrace_CUDA_ApiTrace_CommandType_Kernel:
                return "Kernel";
            case NvTrace_CUDA_ApiTrace_CommandType_InlineToMemory:
                return "InlineToMemory";
            case NvTrace_CUDA_ApiTrace_CommandType_ComputeDma:
                return "Compute DMA";
            case NvTrace_CUDA_ApiTrace_CommandType_AsyncDma:
                return "Async DMA";
            case NvTrace_CUDA_ApiTrace_CommandType_Invalid:
            case NvTrace_CUDA_ApiTrace_CommandType_COUNT:
            case NvTrace_CUDA_ApiTrace_CommandType_FORCE_INT:
                break;
        }
        return "??";
    }

    inline const char* ReductionOpToCString(uint8_t reductionOp, uint8_t dataType)
    {
        switch (reductionOp)
        {
            case NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType_OR:
                return (dataType == NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType_Unsigned32) ? "OR32" : "OR64";
            case NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType_AND:
                return (dataType == NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType_Unsigned32) ? "AND32" : "AND64";
            case NvTrace_CUDA_ApiTrace_MemOpAtomicReductionOpType_ADD:
                return (dataType == NvTrace_CUDA_ApiTrace_MemOpAtomicReductionDataType_Unsigned32) ? "ADD32" : "ADD64";
        }
        return "??";
    }

    inline std::string GetHostName()
    {
        char name[1024];
        name[0] = '\0';
#if defined(__linux__)
        if (gethostname(name, sizeof(name)) == 0)
        {
            name[sizeof(name) - 1] = '\0';
        }
#endif
        return std::string(name);
    }

    enum class DiagnosticSeverity
    {
        None,
        Info,
        Warning,
        Error,
    };

    inline const char* GetNvTraceDiagnosticSeverity(DiagnosticSeverity severity)
    {
        switch (severity)
        {
            case DiagnosticSeverity::None:
                return "None";
            case DiagnosticSeverity::Info:
                return "Info";
            case DiagnosticSeverity::Warning:
                return "Warning";
            case DiagnosticSeverity::Error:
                return "Error";
        }

        return "Unknown";
    }

    inline const char* GetGpuProviderPauseReason(NvTrace_GPU_Provider_PauseReason pauseReason)
    {
        switch (pauseReason)
        {
            case NvTrace_GPU_Provider_PauseReason_None:
                return "None";
            case NvTrace_GPU_Provider_PauseReason_InternalError:
                return "InternalError";
            case NvTrace_GPU_Provider_PauseReason_Unknown:
                return "Unknown";
            case NvTrace_GPU_Provider_PauseReason_FaultedContext:
                return "FaultedContext";
            case NvTrace_GPU_Provider_PauseReason_ManualReset:
                return "ManualReset";
            case NvTrace_GPU_Provider_PauseReason_YieldedToTools:
                return "YieldedToTools";
            case NvTrace_GPU_Provider_PauseReason_COUNT:
            case NvTrace_GPU_Provider_PauseReason_FORCEINT:
                break;
        }
        return "Invalid";
    }

    inline const char* GetGpuProviderResumePhase(NvTrace_GPU_Provider_ResumePhase resumePhase)
    {
        switch (resumePhase)
        {
            case NvTrace_GPU_Provider_ResumePhase_Invalid:
                return "Invalid";
            case NvTrace_GPU_Provider_ResumePhase_Started:
                return "Started";
            case NvTrace_GPU_Provider_ResumePhase_Failed:
                return "Failed";
            case NvTrace_GPU_Provider_ResumePhase_Succeeded:
                return "Succeeded";
            case NvTrace_GPU_Provider_ResumePhase_COUNT:
            case NvTrace_GPU_Provider_ResumePhase_FORCEINT:
                break;
        }
        return "Invalid";
    }

    inline const char* GetGpuProviderResumeFailureReason(NvTrace_GPU_Provider_ResumeFailureReason resumeFailureReason)
    {
        switch (resumeFailureReason)
        {
            case NvTrace_GPU_Provider_ResumeFailureReason_None:
                return "None";
            case NvTrace_GPU_Provider_ResumeFailureReason_Unknown:
                return "Unknown";
            case NvTrace_GPU_Provider_ResumeFailureReason_InsufficientPrivilege:
                return "InsufficientPrivilege";
            case NvTrace_GPU_Provider_ResumeFailureReason_ResourceInUse:
                return "ResourceInUse";
            case NvTrace_GPU_Provider_ResumeFailureReason_InsufficientDriverVersion:
                return "InsufficientDriverVersion";
            case NvTrace_GPU_Provider_ResumeFailureReason_COUNT:
            case NvTrace_GPU_Provider_ResumeFailureReason_FORCEINT:
                break;
        }
        return "Invalid";
    }

    struct TraceExporterOptions
    {
        bool enableDebugEvents = false;
        bool enableStreamTimelines = false;
    };

    struct CuFunctionNames
    {
        const char* pDemangledName = nullptr;
        const char* pMangledName = nullptr;
    };

    template <typename TTraceReader>
    class TraceExporterBase
    {
    protected:
        ~TraceExporterBase() = default;
        TraceExporterBase() = delete;
        TraceExporterBase(Codec::DataModel<TTraceReader>& dataModel)
            : m_dataModel(dataModel)
        {
            m_ppRuntimeApiFunctionNames = FB::Provider::CudaApiTrace::EnumNamesCudaRuntimeApiFunctionId();
            m_ppDriverApiFunctionNames = FB::Provider::CudaApiTrace::EnumNamesCudaDriverApiFunctionId();

            constexpr size_t heuristicMangledNameLength = 1024;
            constexpr size_t heuristicDemangledNameLength = heuristicMangledNameLength * 4;
            constexpr size_t maxDemanglerScratchSpace = 1024 * 1024;
            m_kernelNameDemangler.Initialize(maxDemanglerScratchSpace, SIZE_MAX, heuristicMangledNameLength, heuristicDemangledNameLength);
        }

        const char* GetDemangledCuFunctionName(const Codec::CudaApiTraceData& cudaApiTraceData, uint64_t cufunctionId)
        {
            return GetCuFunctionNames(cudaApiTraceData, cufunctionId).pDemangledName;
        }

        CuFunctionNames GetCuFunctionNames(
            const Codec::CudaApiTraceData& cudaApiTraceData,
            uint64_t cufunctionId)
        {
            const auto functionIterator = cudaApiTraceData.functions.find(cufunctionId);
            if (functionIterator == cudaApiTraceData.functions.end() || functionIterator->second.mangledName.empty())
            {
                return {kUnknownData, kUnknownData};
            }

            const Codec::CudaApiTraceData::CuFunctionData& functionData = functionIterator->second;
            const std::string& mangledName = functionData.mangledName;
            if (functionData.demangledName.empty())
            {
                const char* const pDemangledName = DemangleKernelName(m_kernelNameDemangler, mangledName.data(), mangledName.length());
                if (pDemangledName != mangledName.data() && pDemangledName[0] != '\0')
                {
                    functionData.demangledName = pDemangledName;
                }
                if (functionData.demangledName.empty())
                {
                    // A one-byte NUL string records a failed demangling attempt without retaining a duplicate copy of the mangled name.
                    functionData.demangledName.assign(1, '\0');
                }
            }
            const char* const pMangledName = mangledName.c_str();
            const char* const pDemangledName = functionData.demangledName[0] == '\0' ? pMangledName : functionData.demangledName.c_str();
            return {pDemangledName, pMangledName};
        }

        inline const char* GetRuntimeApiFunctionName(FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId id) const
        {
            if (id <= FB::Provider::CudaApiTrace::CudaRuntimeApiFunctionId::CudaRuntimeApiFunctionId_MAX)
            {
                return m_ppRuntimeApiFunctionNames[id];
            }
            return "??CUDARuntime";
        }

        inline const char* GetDriverApiFunctionName(FB::Provider::CudaApiTrace::CudaDriverApiFunctionId id) const
        {
            if (id <= FB::Provider::CudaApiTrace::CudaDriverApiFunctionId::CudaDriverApiFunctionId_MAX)
            {
                return m_ppDriverApiFunctionNames[id];
            }
            return "??CUDADriver";
        }

        static bool EnsureOutputParentDirectoryExists(const std::filesystem::path& outputFilePath)
        {
            const std::filesystem::path parentDir = outputFilePath.parent_path();
            if (parentDir.empty())
            {
                return true;
            }

            std::error_code ec;
            std::filesystem::create_directories(parentDir, ec);
            if (ec)
            {
                NVTRACE_LOG_ERR(100, "Failed to create output directory for file: %s", outputFilePath.string().c_str());
                return false;
            }

            return true;
        }

    protected:
        Codec::DataModel<TTraceReader>& m_dataModel;

        const char* const* m_ppRuntimeApiFunctionNames = nullptr;
        const char* const* m_ppDriverApiFunctionNames = nullptr;

        // Export is serial, so every emitted kernel name can reuse this one state.
        KernelNameDemangler m_kernelNameDemangler;

        static constexpr char kUnknownData[] = "??";
        static constexpr char kNotApplicableData[] = "N/A";
    };

} //namespace nv::trace
