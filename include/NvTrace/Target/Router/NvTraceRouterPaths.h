/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Router/NvTraceOsGpuAccess.h>
#include <NvTrace/Target/Router/NvTraceOsSecurityIdentifier.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

#if defined(__linux__)
#include <sys/un.h>
#include <unistd.h>
#endif

namespace nv::trace {

    constexpr size_t ROUTER_MAX_PATH_SIZE = 256;

    struct RouterEndpointDesc
    {
        const char* pName;
        size_t maxMessageSize;
        uint32_t maxMessageObjectIdentifierCount;
    };

    constexpr RouterEndpointDesc ROUTER_OBSERVER_ENDPOINT_DESC = {"observer", 1024, std::max<uint32_t>(4, MAX_ACCESSIBLE_GPU_COUNT)};

#if defined(__linux__)
    // The scope's base directory, which every runtime path of the scope lives under. It is provisioned by others
    // (systemd for /run, logind for $XDG_RUNTIME_DIR), so the Router creates directories below it but never changes it.
    // Read once per process, so a later change to $XDG_RUNTIME_DIR is not seen.
    inline std::string_view GetScopeBasePath(bool isGlobalScope)
    {
        using namespace std::string_view_literals;

        if (isGlobalScope)
        {
            return "/run"sv;
        }
        else
        {
            static const std::string_view s_userLocalScopeBasePath = []() {
                // Root has no user-local scope: its $XDG_RUNTIME_DIR is usually unset, and under `sudo -E` it is the
                // invoking user's. A root Router always serves the global scope, so a root client finds it there.
                NVTRACE_LOG_INF_RETURN_IF(geteuid() == 0, std::string_view(), 50, "Running as root, which has only the global NVIDIA Nsight Perf scope");

                // $XDG_RUNTIME_DIR is logind-provided and 0700-owned-by-user. If unset, there's no fallback that
                // preserves user-local isolation, so fail.
                const char* const pXdgRuntimeDir = std::getenv("XDG_RUNTIME_DIR");
                if (!pXdgRuntimeDir || pXdgRuntimeDir[0] == '\0')
                {
                    NVTRACE_LOG_ERR(50, "XDG_RUNTIME_DIR is not set, so user-local NVIDIA Nsight Perf has no socket directory. Set XDG_RUNTIME_DIR or use global scope (--global).");
                    std::fprintf(stderr, "error: XDG_RUNTIME_DIR is not set, so user-local NVIDIA Nsight Perf has no socket directory. Set XDG_RUNTIME_DIR or use global scope (--global).\n");
                    return std::string_view();
                }

                return std::string_view(pXdgRuntimeDir);
            }();

            return s_userLocalScopeBasePath;
        }
    }

    // The Router runtime directory, which holds the scope's router.* files. Composed once per process.
    inline std::string_view GetRouterRuntimeDirectoryPath(bool isGlobalScope)
    {
        const auto fnComposePath = [](bool isGlobalScope, char out_pPath[ROUTER_MAX_PATH_SIZE]) {
            const std::string_view scopeBasePath = GetScopeBasePath(isGlobalScope);
            if (scopeBasePath.empty())
            {
                return std::string_view();
            }

            const int length = std::snprintf(out_pPath, ROUTER_MAX_PATH_SIZE, "%.*s/nvidia/nsight-perf", static_cast<int>(scopeBasePath.size()), scopeBasePath.data());
            NVTRACE_LOG_ERR_RETURN_IF(length < 0 || ROUTER_MAX_PATH_SIZE <= static_cast<size_t>(length), std::string_view(), 50, "Router runtime directory path too long");
            return std::string_view(out_pPath, static_cast<size_t>(length));
        };

        if (isGlobalScope)
        {
            static char s_globalPathBuffer[ROUTER_MAX_PATH_SIZE];
            static const std::string_view s_globalPath = fnComposePath(true, s_globalPathBuffer);
            return s_globalPath;
        }
        else
        {
            static char s_userLocalPathBuffer[ROUTER_MAX_PATH_SIZE];
            static const std::string_view s_userLocalPath = fnComposePath(false, s_userLocalPathBuffer);
            return s_userLocalPath;
        }
    }

    // Resolves the filesystem socket path to connect to. Each endpoint lives in its own directory: router.<endpointName>/.sock.
    inline std::string_view ComposeRouterEndpointPath(const char* pEndpointName, bool isGlobalScope, char out_pPath[ROUTER_MAX_PATH_SIZE])
    {
        NVTRACE_LOG_ERR_RETURN_IF(!pEndpointName || pEndpointName[0] == '\0', std::string_view(), 50, "endpointName must be a non-empty string");

        const std::string_view routerRuntimeDirectoryPath = GetRouterRuntimeDirectoryPath(isGlobalScope);
        if (routerRuntimeDirectoryPath.empty())
        {
            return std::string_view();
        }

        const int length = std::snprintf(out_pPath, ROUTER_MAX_PATH_SIZE, "%.*s/router.%s/.sock", static_cast<int>(routerRuntimeDirectoryPath.size()), routerRuntimeDirectoryPath.data(), pEndpointName);
        NVTRACE_LOG_ERR_RETURN_IF(length < 0 || ROUTER_MAX_PATH_SIZE <= static_cast<size_t>(length), std::string_view(), 50, "Router endpoint path too long");

        // A filesystem-bound socket path plus its NUL terminator must fit sockaddr_un::sun_path.
        NVTRACE_LOG_ERR_RETURN_IF(sizeof(sockaddr_un::sun_path) <= static_cast<size_t>(length), std::string_view(), 50, "Socket path length %d exceeds sun_path budget", length);
        return std::string_view(out_pPath, static_cast<size_t>(length));
    }

#endif

} // namespace nv::trace
