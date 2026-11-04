/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/Router/NvTraceIpcClient.h>
#include <NvTrace/Target/Router/NvTraceRouterPaths.h>

#include <cstdint>
#include <string_view>

namespace nv::trace {

    class RouterClient : protected IpcClient
    {
    protected:
        ~RouterClient() = default;
        RouterClient() = default;

        void Reset() noexcept
        {
            m_lastObjectLuidInClient = NVTRACE_NIL_LUID;
            IpcClient::Reset();
        }

        // Connects to the router endpoint, trying the global router first and then the user-local router
        bool Initialize(const RouterEndpointDesc& endpointDesc, uint32_t timeoutMsec)
        {
            const RouterConnectResult globalRouterConnectResult = TryConnectToRouter(endpointDesc, true, timeoutMsec);
            if (globalRouterConnectResult == RouterConnectResult::Succeeded)
            {
                return true;
            }
            else if (globalRouterConnectResult == RouterConnectResult::CapacityExhausted)
            {
                LogConnectError(globalRouterConnectResult);
                return false;
            }

            const RouterConnectResult localRouterConnectResult = TryConnectToRouter(endpointDesc, false, timeoutMsec);
            if (localRouterConnectResult == RouterConnectResult::Succeeded)
            {
                return true;
            }

            LogConnectError(localRouterConnectResult != RouterConnectResult::Absent ? localRouterConnectResult : globalRouterConnectResult);
            return false;
        }

    private:
        enum class RouterConnectResult
        {
            Succeeded,
            Absent,
            PermissionDenied,
            CapacityExhausted,
            Spoofed,
            UnexpectedFailure
        };

        RouterConnectResult TryConnectToRouter(const RouterEndpointDesc& endpointDesc, bool isGlobalScope, uint32_t timeoutMsec)
        {
            char addressBuffer[ROUTER_MAX_PATH_SIZE];
#if defined(__linux__)
            std::string_view address = ComposeRouterEndpointPath(endpointDesc.pName, isGlobalScope, addressBuffer);
#endif

            if (address.empty())
            {
                return RouterConnectResult::Absent;
            }

            const IpcClient::ConnectResult connectResult = IpcClient::Initialize(address, endpointDesc.maxMessageSize, endpointDesc.maxMessageObjectIdentifierCount, timeoutMsec);
            switch (connectResult)
            {
                case IpcClient::ConnectResult::Succeeded:
                    break;
                case IpcClient::ConnectResult::NotFound:
                    return RouterConnectResult::Absent;
                case IpcClient::ConnectResult::PermissionDenied:
                    return RouterConnectResult::PermissionDenied;
                case IpcClient::ConnectResult::CapacityExhausted:
                    return RouterConnectResult::CapacityExhausted;
                case IpcClient::ConnectResult::UnexpectedFailure:
                    return RouterConnectResult::UnexpectedFailure;
            }


            return RouterConnectResult::Succeeded;
        }

        static void LogConnectError(const RouterConnectResult connectResult)
        {
            switch (connectResult)
            {
                case RouterConnectResult::Succeeded:
                    break;
                case RouterConnectResult::Absent:
                    NVTRACE_LOG_WRN(50, "Router is absent");
                    break;
                case RouterConnectResult::PermissionDenied:
                    NVTRACE_LOG_WRN(50, "Router connection permission denied");
                    break;
                case RouterConnectResult::CapacityExhausted:
                    NVTRACE_LOG_WRN(50, "Router capacity exhausted");
                    break;
                case RouterConnectResult::Spoofed:
                    NVTRACE_LOG_WRN(50, "Router spoofed");
                    break;
                case RouterConnectResult::UnexpectedFailure:
                    NVTRACE_LOG_WRN(50, "Failed to connect to the router");
                    break;
            }
        }

    protected:
        NVTRACE_LUID m_lastObjectLuidInClient = NVTRACE_NIL_LUID;
    };

} // namespace nv::trace
