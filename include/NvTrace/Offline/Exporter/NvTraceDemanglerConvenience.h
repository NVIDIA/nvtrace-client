/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Offline/DataModel/NvTraceDemanglerCuTilePython.h>
#include <NvTrace/Offline/DataModel/NvTraceDemanglerCuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace nv::trace {

    /// Reusable STL-backed state for the common demangle-to-temporary-string path.
    struct KernelNameDemangler
    {
        CudaDemangler cudaDemangler;
        CuTilePythonDemangler cuTilePythonDemangler;
        std::vector<uint8_t> scratchSpace;
        std::string stagedKernelName;
        /// Hard allocation ceiling; names requiring more scratch fall back unchanged.
        size_t m_maxScratchSpace = 0;
        /// Hard rendered-byte limit; reaching it returns the successfully rendered prefix.
        size_t m_maxOutputLength = 0;

        KernelNameDemangler() = default;
        KernelNameDemangler(const KernelNameDemangler&) = delete;
        KernelNameDemangler& operator=(const KernelNameDemangler&) = delete;
        KernelNameDemangler(KernelNameDemangler&&) = delete;
        KernelNameDemangler& operator=(KernelNameDemangler&&) = delete;

        /// @param maxScratchSpace Hard scratch allocation limit; overflow falls back unchanged.
        /// @param maxOutputLength Hard rendered-byte limit; it may exceed initial capacity.
        /// @param expectedInputLength Heuristic used only to size the initial scratch buffer.
        /// @param initialOutputCapacity Initial string reserve; output may grow to maxOutputLength.
        void Initialize(
            size_t maxScratchSpace,
            size_t maxOutputLength,
            size_t expectedInputLength,
            size_t initialOutputCapacity)
        {
            m_maxScratchSpace = maxScratchSpace;
            m_maxOutputLength = maxOutputLength;
            const size_t conservativeBytesPerInputByte = 512;
            const size_t fixedScratchBytes = 4096;
            const size_t estimatedScratchSize = expectedInputLength <= (SIZE_MAX - fixedScratchBytes) / conservativeBytesPerInputByte
                                                  ? expectedInputLength * conservativeBytesPerInputByte + fixedScratchBytes
                                                  : 0;
            const size_t boundedScratchSize = estimatedScratchSize < m_maxScratchSpace
                                                ? estimatedScratchSize
                                                : m_maxScratchSpace;
            scratchSpace.resize(boundedScratchSize);
            stagedKernelName.reserve(initialOutputCapacity);
            cudaDemangler.BindScratchSpace(scratchSpace.data(), scratchSpace.size());
            cuTilePythonDemangler.BindScratchSpace(scratchSpace.data(), scratchSpace.size());
        }
    };

    class DemanglerConveniencePrivate
    {
    public:
        static bool IsCuTileName(const char* pMangledName, size_t mangledNameLength)
        {
            size_t signatureOffset = SIZE_MAX;
            for (size_t inputIndex = 0; inputIndex + 2 < mangledNameLength; ++inputIndex)
            {
                if (pMangledName[inputIndex] == '_' && pMangledName[inputIndex + 1] == 'K')
                {
                    signatureOffset = inputIndex;
                }
            }
            if (signatureOffset == SIZE_MAX || pMangledName[signatureOffset + 2] != 't')
            {
                return false;
            }
            size_t versionIndex = signatureOffset + 3;
            const size_t versionStart = versionIndex;
            while (versionIndex < mangledNameLength
                   && '0' <= pMangledName[versionIndex] && pMangledName[versionIndex] <= '9')
            {
                ++versionIndex;
            }
            return versionIndex != versionStart
                && (versionIndex == mangledNameLength || pMangledName[versionIndex] == '_');
        }

        static const char* DemangleCuTilePython(
            KernelNameDemangler& kernelNameDemangler,
            const char* pMangledName,
            size_t mangledNameLength)
        {
            kernelNameDemangler.stagedKernelName.clear();
            return DemangleWith(
                kernelNameDemangler,
                kernelNameDemangler.cuTilePythonDemangler,
                pMangledName,
                mangledNameLength);
        }

        static const char* DemangleCudaWithFallback(
            KernelNameDemangler& kernelNameDemangler,
            const char* pMangledName,
            size_t mangledNameLength)
        {
            kernelNameDemangler.stagedKernelName.clear();
            CudaDemangler::Status parseStatus = CudaDemangler::Status::NotParsed;
            const char* const pDemangledName = DemangleWith(
                kernelNameDemangler,
                kernelNameDemangler.cudaDemangler,
                pMangledName,
                mangledNameLength,
                &parseStatus);
            if (parseStatus != CudaDemangler::Status::UnsupportedInput)
            {
                return pDemangledName;
            }

            // fallback; either use __cxa_demangle if available, or return the original mangled name
            const char* const pCxaDemangledName = DemangleWithCxa(kernelNameDemangler, pMangledName, mangledNameLength);
            return pCxaDemangledName;
        }

    private:
        template <typename TDemangler>
        static const char* DemangleWith(
            KernelNameDemangler& kernelNameDemangler,
            TDemangler& demangler,
            const char* pMangledName,
            size_t mangledNameLength,
            typename TDemangler::Status* pParseStatus = nullptr)
        {
            using Status = typename TDemangler::Status;
            demangler.BeginDemangling(pMangledName, mangledNameLength);
            Status status = demangler.ParseInput();
            if (status == Status::NeedMoreScratch)
            {
                const size_t requiredScratchSize = demangler.GetRequiredScratchSpaceSize();
                if (kernelNameDemangler.m_maxScratchSpace < requiredScratchSize)
                {
                    return pMangledName;
                }
                kernelNameDemangler.scratchSpace.resize(requiredScratchSize);
                kernelNameDemangler.cudaDemangler.BindScratchSpace(kernelNameDemangler.scratchSpace.data(), kernelNameDemangler.scratchSpace.size());
                kernelNameDemangler.cuTilePythonDemangler.BindScratchSpace(kernelNameDemangler.scratchSpace.data(), kernelNameDemangler.scratchSpace.size());
                status = demangler.ParseInput();
            }
            if (pParseStatus)
            {
                *pParseStatus = status;
            }
            if (status != Status::Complete)
            {
                return pMangledName;
            }

            status = demangler.RenderDemangledName(kernelNameDemangler.m_maxOutputLength, [&](const char* pCharacters, size_t characterCount) {
                kernelNameDemangler.stagedKernelName.append(pCharacters, characterCount);
            });
            if (status == Status::Complete
                || (status == Status::OutputLimitReached
                    && !kernelNameDemangler.stagedKernelName.empty()))
            {
                return kernelNameDemangler.stagedKernelName.c_str();
            }
            return pMangledName;
        }

#if defined(__linux__) && NVTRACE_ENABLE_CXA_DEMANGLE
        using CxaDemangleFunction = char* (*)(const char* mangled_name, char* output_buffer, size_t* length, int* status);

        static CxaDemangleFunction GetPlatformCxaDemangle()
        {
            static const CxaDemangleFunction cxaDemangle = []() {
                void* const pSymbol = ::dlsym(RTLD_DEFAULT, "__cxa_demangle");
                CxaDemangleFunction resolvedFunction = nullptr;
                std::memcpy(&resolvedFunction, &pSymbol, sizeof(resolvedFunction));
                return resolvedFunction;
            }();
            return cxaDemangle;
        }

        static size_t CopyBoundedCString(
            std::string& output,
            const char* pInput,
            size_t maxOutputLength)
        {
            output.clear();
            size_t copiedByteCount = 0;
            while (copiedByteCount < maxOutputLength && pInput[copiedByteCount] != '\0')
            {
                output.push_back(pInput[copiedByteCount]);
                ++copiedByteCount;
            }
            return copiedByteCount;
        }

        static const char* DemangleWithCxa(
            KernelNameDemangler& kernelNameDemangler,
            const char* pMangledName,
            size_t mangledNameLength)
        {
            CxaDemangleFunction cxaDemangle = GetPlatformCxaDemangle();
            if (!cxaDemangle)
            {
                return pMangledName;
            }

            // __cxa_demangle accepts a NUL-terminated name and owns the returned
            // malloc allocation. The NvTrace entry point is length-based, so stage a
            // terminated copy only on this unsupported-input fallback path.
            const std::string terminatedMangledName(pMangledName, mangledNameLength);
            int demangleStatus = -1;
            char* const pDemangledName = cxaDemangle(terminatedMangledName.c_str(), nullptr, nullptr, &demangleStatus);
            if (!pDemangledName || demangleStatus != 0)
            {
                free(pDemangledName);
                return pMangledName;
            }

            const size_t copiedByteCount = CopyBoundedCString(
                kernelNameDemangler.stagedKernelName,
                pDemangledName,
                kernelNameDemangler.m_maxOutputLength);
            free(pDemangledName);
            if (!copiedByteCount)
            {
                return pMangledName;
            }
            return kernelNameDemangler.stagedKernelName.c_str();
        }
#else // not linux || !NVTRACE_ENABLE_CXA_DEMANGLE
        static const char* DemangleWithCxa(
            KernelNameDemangler& kernelNameDemangler,
            const char* pMangledName,
            size_t mangledNameLength)
        {
            return pMangledName;
        }
#endif
    };

    /// Returns the best useful non-null name. Malformed input falls back to the
    /// mangled bytes; unsupported Itanium input may use a dynamically discovered
    /// Linux ABI demangler. An output limit returns the successfully copied prefix.
    inline const char* DemangleKernelName(
        KernelNameDemangler& kernelNameDemangler,
        const char* pMangledName,
        size_t mangledNameLength)
    {
        if (!pMangledName)
        {
            return "";
        }

        kernelNameDemangler.stagedKernelName.clear();
        const bool isItaniumName = mangledNameLength >= 2
                                && pMangledName[0] == '_' && pMangledName[1] == 'Z';
        if (!isItaniumName && DemanglerConveniencePrivate::IsCuTileName(pMangledName, mangledNameLength))
        {
            return DemanglerConveniencePrivate::DemangleCuTilePython(kernelNameDemangler, pMangledName, mangledNameLength);
        }
        return DemanglerConveniencePrivate::DemangleCudaWithFallback(kernelNameDemangler, pMangledName, mangledNameLength);
    }

    /// Precondition: pMangledName != nullptr
    inline const char* DemangleKernelName(KernelNameDemangler& kernelNameDemangler, const char* pMangledName)
    {
        const char* pDemangledName = DemangleKernelName(kernelNameDemangler, pMangledName, std::strlen(pMangledName));
        return pDemangledName;
    }

} // namespace nv::trace
