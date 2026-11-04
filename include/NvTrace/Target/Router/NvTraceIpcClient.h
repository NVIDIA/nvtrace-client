/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Target/NvTraceDynamicArray.h>
#include <NvTrace/Target/Router/NvTraceIpcMessageBuffer.h>
#include <NvTrace/Target/Router/NvTraceOsSecurityIdentifier.h>

#if defined(__linux__)
#include <NvTrace/Target/NvTraceLinuxUtilities.h>
#endif

#include <string_view>
#include <type_traits>

namespace nv::trace {

    // IPC client class that communicates with an IPC server.
    // Send and receive are blocking calls with a timeout. If the timeout passes, the connection is deemed unrecoverably lost.
    class IpcClient
    {
    public:
        static constexpr size_t MAX_SERVER_ADDRESS_LENGTH = 107; // = sizeof(sockaddr_un::sun_path) - 1

    public:
        ~IpcClient();
        IpcClient();

        IpcClient(const IpcClient&) = delete;
        IpcClient& operator=(const IpcClient&) = delete;

        void Reset() noexcept;

        // Classified outcome of Initialize's connect attempt.
        enum class ConnectResult
        {
            Succeeded,
            NotFound,
            PermissionDenied,
            CapacityExhausted,
            UnexpectedFailure
        };

        // Connects to a router endpoint. Server identity is established by
        // filesystem permissions on the address (Linux)
        ConnectResult Initialize(std::string_view serverAddress, size_t maxMessageSize, uint32_t maxMessageObjectIdentifierCount, uint32_t timeoutMsec);


        void Disconnect() noexcept;

        // Costs a syscall
        bool IsConnected();

        NvTrace_WaitableOsHandle GetWaitForPendingReceiveHandle() const
        {
#if defined(__linux__)
            return m_socketFd;
#endif
        }

        bool SendMessage(PFN_IpcWriteMessageCallback pfnWriteMessageCallback, void* pCallbackUserData);
        template <typename TFN_WriteMessageCallback>
        bool SendMessage(TFN_WriteMessageCallback&& fnWriteMessageCallback)
        {
            using WriteMessageCallback = std::remove_reference_t<TFN_WriteMessageCallback>;
            return SendMessage([](void* pUserData, uint32_t& messageType, flatbuffers::FlatBufferBuilder& messageBuilder, NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t& messageObjectIdentifierCount) { (*reinterpret_cast<WriteMessageCallback*>(pUserData))(messageType, messageBuilder, pMessageObjectIdentifiers, messageObjectIdentifierCount); }, &fnWriteMessageCallback);
        }

        bool ReceiveMessage(PFN_IpcReadMessageCallback pfnReadMessageCallback, void* pCallbackUserData);
        template <typename TFN_ReadMessageCallback>
        bool ReceiveMessage(TFN_ReadMessageCallback&& pfnReadMessageCallback)
        {
            using ReadMessageCallback = std::remove_reference_t<TFN_ReadMessageCallback>;
            return ReceiveMessage([](void* pUserData, uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount) { return (*reinterpret_cast<ReadMessageCallback*>(pUserData))(messageType, pMessage, messageSize, pMessageObjectIdentifiers, messageObjectIdentifierCount); }, &pfnReadMessageCallback);
        }

        bool ReceiveMessageIfPending(PFN_IpcReadMessageCallback pfnReadMessageCallback, void* pCallbackUserData);
        template <typename TFN_ReadMessageCallback>
        bool ReceiveMessageIfPending(TFN_ReadMessageCallback&& pfnReadMessageCallback)
        {
            using ReadMessageCallback = std::remove_reference_t<TFN_ReadMessageCallback>;
            return ReceiveMessageIfPending([](void* pUserData, uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount) { return (*reinterpret_cast<ReadMessageCallback*>(pUserData))(messageType, pMessage, messageSize, pMessageObjectIdentifiers, messageObjectIdentifierCount); }, &pfnReadMessageCallback);
        }

        bool HasPendingMessage() const;

    private:
#if defined(__linux__)
        bool PollSocket(short event, bool& timerSet);
#endif

    private:
#if defined(__linux__)
        struct Timeout
        {
            uint32_t sec = 0;
            uint32_t nsec = 0;
        } m_timeout;
#endif

        DynamicArray<uint8_t> m_sendRecvBufferMemory;

        IpcMessageSendBuffer m_sendBuffer;

        IpcMessageReceiveBuffer m_recvBuffer;

#if defined(__linux__)
        UniqueLinuxFd m_timerFd;
        UniqueLinuxFd m_socketFd;
#endif

        bool m_connected = false;
    };

} //namespace nv::trace

#if defined(__linux__)
#include <NvTrace/Target/Router/NvTraceIpcClientLinux.h>
#endif
