/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/Router/NvTraceIpcClient.h>

#include <cstring>
#include <errno.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/timerfd.h>
#include <sys/un.h>

#define NVTRACE_LOG_WRN_DISCONNECT_RETURN_IF(cond_, val_, logger_, level_, ...) NVTRACE_LOG_DO_IF(cond_, Disconnect(); return val_;, Wrn, level_, __VA_ARGS__)
#define NVTRACE_LOG_ERR_DISCONNECT_RETURN_IF(cond_, val_, logger_, level_, ...) NVTRACE_LOG_DO_IF(cond_, Disconnect(); return val_;, Err, level_, __VA_ARGS__)

namespace nv::trace {

    inline IpcClient::~IpcClient()
    {
        Disconnect();
    }

    inline IpcClient::IpcClient() = default;

    inline void IpcClient::Reset() noexcept
    {
        Disconnect();
        m_socketFd.reset();
        m_timerFd.reset();
        m_recvBuffer.Reset();
        m_sendBuffer.Reset();
        m_sendRecvBufferMemory.Reset();
        m_timeout = {};
    }

    inline IpcClient::ConnectResult IpcClient::Initialize(std::string_view serverAddress, size_t maxMessageSize, uint32_t maxMessageObjectIdentifierCount, uint32_t timeoutMS)
    {
        NVTRACE_LOG_ERR_RETURN_IF(serverAddress.empty(), ConnectResult::UnexpectedFailure, 50, "Server address is empty");

        // Convert the server address to a Unix socket address
        NVTRACE_LOG_ERR_RETURN_IF(MAX_SERVER_ADDRESS_LENGTH <= serverAddress.size(), ConnectResult::UnexpectedFailure, 50, "Server address is too long, max length = %d", (int)MAX_SERVER_ADDRESS_LENGTH);

        sockaddr_un socketAddress;
        socketAddress.sun_family = AF_UNIX;
        socklen_t socketAddressLength;

        const bool isFilesystemSocket = serverAddress.size() > 0 && serverAddress[0] == '/';
        if (isFilesystemSocket)
        {
            std::memcpy(socketAddress.sun_path, serverAddress.data(), serverAddress.size());
            socketAddress.sun_path[serverAddress.size()] = '\0';
            socketAddressLength = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + serverAddress.size() + 1);
        }
        else
        {
            socketAddress.sun_path[0] = '\0';
            std::memcpy(socketAddress.sun_path + 1, serverAddress.data(), serverAddress.size());
            socketAddressLength = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + serverAddress.size());
        }

        // Allocate memory for the receive/send buffers
        size_t messageBufferMemorySize = IpcMessageBuffer::CalculateRequiredBufferMemorySize(maxMessageSize, maxMessageObjectIdentifierCount);
        DynamicArray<uint8_t> sendRecvBufferMemory(2 * messageBufferMemorySize); // Times 2 for both receive and send buffers
        NVTRACE_LOG_ERR_RETURN_IF(!sendRecvBufferMemory, ConnectResult::UnexpectedFailure, 50, "IpcMessageBuffer memory allocation failed");

        // Create the timer object, if timeoutMS is not infinite
        UniqueLinuxFd timerFd;
        if (timeoutMS != UINT32_MAX)
        {
            timerFd.reset(timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC));
            NVTRACE_LOG_ERR_RETURN_IF(!timerFd, ConnectResult::UnexpectedFailure, 50, "timerfd_create failed with error %d", errno);
        }

        // Create and connect the socket
        UniqueLinuxFd socketFd(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        NVTRACE_LOG_ERR_RETURN_IF(!socketFd, ConnectResult::UnexpectedFailure, 50, "socket failed with error %d", errno);

        int result = connect(socketFd, reinterpret_cast<sockaddr*>(&socketAddress), socketAddressLength);
        if (result == -1)
        {
            const int connectErrno = errno;
            switch (connectErrno)
            {
                case ENOENT:
                case ECONNREFUSED:
                    return ConnectResult::NotFound;
                case EACCES:
                case EPERM:
                    return ConnectResult::PermissionDenied;
                default:
                    NVTRACE_LOG_WRN(50, "connect to %.*s failed with error %d", static_cast<int>(serverAddress.size()), serverAddress.data(), connectErrno);
                    return ConnectResult::UnexpectedFailure;
            }
        }

        // Finalize
        if (timeoutMS != UINT32_MAX)
        {
            m_timeout.sec = timeoutMS / 1'000;
            // timerfd_settime disarms the timer when it_value is zero, so a zero timeout must still arm for one nanosecond
            m_timeout.nsec = timeoutMS != 0 ? (timeoutMS % 1'000) * 1'000'000 : 1;
        }
        m_sendRecvBufferMemory = std::move(sendRecvBufferMemory);
        m_sendBuffer.Initialize(m_sendRecvBufferMemory.Data(), maxMessageSize, maxMessageObjectIdentifierCount);
        m_recvBuffer.Initialize(m_sendRecvBufferMemory.Data() + messageBufferMemorySize, maxMessageSize, maxMessageObjectIdentifierCount);
        m_timerFd = std::move(timerFd);
        m_socketFd = std::move(socketFd);
        m_connected = true;

        NVTRACE_LOG_INF(50, "Connected to server");
        return ConnectResult::Succeeded;
    }

    inline void IpcClient::Disconnect() noexcept
    {
        if (!m_connected)
        {
            return;
        }

        m_connected = false;
        int result = shutdown(m_socketFd, SHUT_RDWR);
        NVTRACE_LOG_ERR_IF(result == -1, 50, "shutdown failed with error %d", errno);

        m_recvBuffer.Clear();
        m_sendBuffer.Clear();

        NVTRACE_LOG_INF(50, "Disconnected from server");
    }

    inline bool IpcClient::IsConnected()
    {
        if (!m_connected)
        {
            return false;
        }

        // POLLRDHUP, unlike POLLIN, still reports the server closing its end when messages are queued
        pollfd pollFd = {m_socketFd, POLLRDHUP};
        // A signal interrupts the call before it looks at the socket, so ask again rather than answer blind
        int result;
        do
        {
            constexpr int timeoutMsec = 0;
            result = poll(&pollFd, 1, timeoutMsec);
        } while (result == -1 && errno == EINTR);

        NVTRACE_LOG_ERR_DISCONNECT_RETURN_IF(result == -1, false, ipc_client, 50, "poll failed with error %d", errno);

        constexpr short connectionEndedEvents = POLLRDHUP | POLLHUP | POLLERR | POLLNVAL;
        NVTRACE_LOG_WRN_DISCONNECT_RETURN_IF((pollFd.revents & connectionEndedEvents) != 0, false, ipc_client, 50, "Connection dropped by server");

        return true;
    }

    inline bool IpcClient::SendMessage(PFN_IpcWriteMessageCallback pfnWriteMessageCallback, void* pCallbackUserData)
    {
        if (!m_connected)
        {
            return false;
        }

        if (!m_sendBuffer.WriteMessage(pfnWriteMessageCallback, pCallbackUserData))
        {
            return false;
        }

        bool timerSet = false;
        while (true)
        {
            iovec iov = {const_cast<uint8_t*>(m_sendBuffer.GetUnsentMessageDataPtr()), m_sendBuffer.GetUnsentMessageDataSize()};
            msghdr msg = {};
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            msg.msg_control = const_cast<uint8_t*>(m_sendBuffer.GetUnsentAncillaryDataPtr());
            msg.msg_controllen = m_sendBuffer.GetUnsentAncillaryDataSize();
            ssize_t result = sendmsg(m_socketFd, &msg, 0);
            if (result != -1)
            {
                m_sendBuffer.SetDataSent(static_cast<size_t>(result));
                if (m_sendBuffer.IsSendComplete())
                {
                    break;
                }
            }
            else if (errno == EAGAIN)
            {
                // Wait for the socket to become writable
                if (!PollSocket(POLLOUT, timerSet))
                {
                    Disconnect();
                    return false;
                }
            }
            else
            {
                switch (errno)
                {
                    case ECONNRESET:
                    case EPIPE:
                        NVTRACE_LOG_WRN(50, "Connection dropped by server");
                        break;

                    default:
                        NVTRACE_LOG_ERR(50, "sendmsg failed with error %d", errno);
                        break;
                }

                Disconnect();
                return false;
            }
        }

        return true;
    }

    inline bool IpcClient::ReceiveMessage(PFN_IpcReadMessageCallback pfnReadMessageCallback, void* pCallbackUserData)
    {
        if (!m_connected)
        {
            return false;
        }

        bool timerSet = false;
        while (true)
        {
            iovec iov = {m_recvBuffer.GetReceiveMessageWindowPtr(), m_recvBuffer.GetReceiveMessageWindowSize()};
            msghdr msg = {};
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            msg.msg_control = m_recvBuffer.GetReceiveAncillaryWindowPtr();
            msg.msg_controllen = m_recvBuffer.GetReceiveAncillaryWindowSize();
            ssize_t result = recvmsg(m_socketFd, &msg, 0);
            if (result != -1)
            {
                // If the result is 0, it means that the connection was closed by the remote peer
                NVTRACE_LOG_WRN_DISCONNECT_RETURN_IF(result == 0, false, ipc_client, 50, "Connection dropped by server");

                m_recvBuffer.SetDataReceived(static_cast<size_t>(result), msg.msg_controllen, msg.msg_flags);
                if (m_recvBuffer.IsReceiveComplete())
                {
                    break;
                }
            }
            else if (errno == EAGAIN)
            {
                // Wait for the socket to become readable
                if (!PollSocket(POLLIN, timerSet))
                {
                    Disconnect();
                    return false;
                }
            }
            else
            {
                switch (errno)
                {
                    case ECONNRESET:
                    case EPIPE:
                        NVTRACE_LOG_WRN(50, "Connection dropped by server");
                        break;

                    default:
                        NVTRACE_LOG_ERR(50, "recvmsg failed with error %d", errno);
                        break;
                }

                Disconnect();
                return false;
            }
        }

        NVTRACE_LOG_ERR_DISCONNECT_RETURN_IF(!m_recvBuffer.IsReceiveSuccessful(), false, ipc_client, 50, "Received a corrupted message");

        bool readSuccessful = m_recvBuffer.ReadMessage(pfnReadMessageCallback, pCallbackUserData);
        m_recvBuffer.Clear();

        return readSuccessful;
    }

    inline bool IpcClient::ReceiveMessageIfPending(PFN_IpcReadMessageCallback pfnReadMessageCallback, void* pCallbackUserData)
    {
        if (!m_connected)
        {
            return false;
        }

        while (true)
        {
            iovec iov = {m_recvBuffer.GetReceiveMessageWindowPtr(), m_recvBuffer.GetReceiveMessageWindowSize()};
            msghdr msg = {};
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            msg.msg_control = m_recvBuffer.GetReceiveAncillaryWindowPtr();
            msg.msg_controllen = m_recvBuffer.GetReceiveAncillaryWindowSize();
            ssize_t result = recvmsg(m_socketFd, &msg, 0);
            if (result != -1)
            {
                // If the result is 0, it means that the connection was closed by the remote peer
                NVTRACE_LOG_WRN_DISCONNECT_RETURN_IF(result == 0, false, ipc_client, 50, "Connection dropped by server");

                m_recvBuffer.SetDataReceived(static_cast<size_t>(result), msg.msg_controllen, msg.msg_flags);
                if (m_recvBuffer.IsReceiveComplete())
                {
                    break;
                }
            }
            else if (errno == EAGAIN)
            {
                // There is no pending message, this is not an error
                return true;
            }
            else
            {
                switch (errno)
                {
                    case ECONNRESET:
                    case EPIPE:
                        NVTRACE_LOG_WRN(50, "Connection dropped by server");
                        break;

                    default:
                        NVTRACE_LOG_ERR(50, "recvmsg failed with error %d", errno);
                        break;
                }

                Disconnect();
                return false;
            }
        }

        NVTRACE_LOG_ERR_DISCONNECT_RETURN_IF(!m_recvBuffer.IsReceiveSuccessful(), false, ipc_client, 50, "Received a corrupted message");

        bool readSuccessful = m_recvBuffer.ReadMessage(pfnReadMessageCallback, pCallbackUserData);
        m_recvBuffer.Clear();

        return readSuccessful;
    }

    inline bool IpcClient::HasPendingMessage() const
    {
        pollfd pollFd = {m_socketFd, POLLIN};
        constexpr int timeoutMsec = 0;
        int result = poll(&pollFd, 1, timeoutMsec);
        NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "poll failed with error %d", errno);

        return result == 1;
    }

    inline bool IpcClient::PollSocket(short event, bool& timerSet)
    {
        if (m_timerFd && !timerSet)
        {
            itimerspec tmr = {};
            tmr.it_value.tv_sec = static_cast<decltype(timespec::tv_sec)>(m_timeout.sec);
            tmr.it_value.tv_nsec = static_cast<decltype(timespec::tv_nsec)>(m_timeout.nsec);
            int result = timerfd_settime(m_timerFd, 0, &tmr, nullptr);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "timerfd_settime failed with error %d", errno);

            timerSet = true;
        }

        pollfd pollFds[2] = {
            {m_socketFd,  event},
            { m_timerFd, POLLIN}
        };
        int pollFdCount = m_timerFd ? 2 : 1;

        // Failing with EINTR is not an error, it means that the wait was interrupted by a signal.
        // Therefore we poll in a loop to re-enter the wait after receiving EINTR
        while (true)
        {
            constexpr int timeoutMsec = -1;
            int result = poll(pollFds, pollFdCount, timeoutMsec);
            if (result != -1)
            {
                break;
            }
            else
            {
                NVTRACE_LOG_ERR_RETURN_IF(errno != EINTR, false, 50, "poll failed with error %d", errno);
            }
        }

        if (pollFds[0].revents == 0)
        {
            switch (event)
            {
                case POLLOUT:
                    NVTRACE_LOG_WRN(50, "SendMessage timed out");
                    break;

                case POLLIN:
                    NVTRACE_LOG_WRN(50, "ReceiveMessage timed out");
                    break;
            }

            return false;
        }
        else if (pollFds[0].revents != event)
        {
            NVTRACE_LOG_WRN(50, "Connection dropped by server");
            return false;
        }

        return true;
    }

} // namespace nv::trace

#undef NVTRACE_LOG_WRN_DISCONNECT_RETURN_IF
#undef NVTRACE_LOG_ERR_DISCONNECT_RETURN_IF
