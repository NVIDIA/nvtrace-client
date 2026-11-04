/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Target/NvTraceClientLogger.h>

#include <type_traits>

#if defined(__linux__)

#include <NvTrace/Target/NvTraceLinuxUtilities.h>

#include <errno.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/poll.h>
#include <sys/timerfd.h>
#include <unistd.h>

namespace nv::trace {

    struct WaitableObjectDesc
    {
        NvTrace_WaitableOsHandle handle;
        bool isAutoReset;

        WaitableObjectDesc(NvTrace_WaitableOsHandle handle_)
            : handle(handle_)
            , isAutoReset(false)
        {
        }
        WaitableObjectDesc(NvTrace_WaitableOsHandle handle_, bool isAutoReset_)
            : handle(handle_)
            , isAutoReset(isAutoReset_)
        {
        }
    };

    class ManualResetEvent
    {
    public:
        ~ManualResetEvent() = default;
        ManualResetEvent() = default;

        ManualResetEvent(const ManualResetEvent&) = delete;
        ManualResetEvent(ManualResetEvent&&) = default;

        ManualResetEvent& operator=(const ManualResetEvent&) = delete;
        ManualResetEvent& operator=(ManualResetEvent&&) = default;

        void Reset() noexcept
        {
            m_eventFd.reset();
        }

        bool Initialize(bool initiallySignaled = false)
        {
            NVTRACE_LOG_ERR_RETURN_IF(m_eventFd, false, 50, "ManualResetEvent already initialized, call Reset() before re-initializing");

            m_eventFd.reset(::eventfd(initiallySignaled, EFD_NONBLOCK | EFD_CLOEXEC));
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "eventfd failed with error %d", errno);

            return true;
        }

        WaitableObjectDesc Desc()
        {
            return WaitableObjectDesc(m_eventFd, false);
        }

        bool Signal()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "ManualResetEvent is not initialized");

            const uint64_t value = 1;
            ssize_t result = ::write(m_eventFd, &value, sizeof(uint64_t));
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "write to eventfd failed with error %d", errno);

            return true;
        }

        bool Clear()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "ManualResetEvent is not initialized");

            uint64_t value;
            ssize_t result = ::read(m_eventFd, &value, sizeof(uint64_t));
            NVTRACE_LOG_ERR_RETURN_IF(result == -1 && errno != EAGAIN, false, 50, "read from eventfd failed with error %d", errno);

            return true;
        }

        bool Wait()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "ManualResetEvent is not initialized");

            pollfd pollFd = {m_eventFd, POLLIN};
            int result;
            do
            {
                result = ::poll(&pollFd, 1, INFINITE_TIMEOUT);
            } while (result == -1 && errno == EINTR);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "poll failed with error %d", errno);

            return true;
        }

    private:
        UniqueLinuxFd m_eventFd;
    };

    class AutoResetEvent
    {
    public:
        ~AutoResetEvent() = default;
        AutoResetEvent() = default;

        AutoResetEvent(const AutoResetEvent&) = delete;
        AutoResetEvent(AutoResetEvent&&) = default;

        AutoResetEvent& operator=(const AutoResetEvent&) = delete;
        AutoResetEvent& operator=(AutoResetEvent&&) = default;

        void Reset() noexcept
        {
            m_eventFd.reset();
        }

        bool Initialize(bool initiallySignaled = false)
        {
            NVTRACE_LOG_ERR_RETURN_IF(m_eventFd, false, 50, "AutoResetEvent already initialized, call Reset() before re-initializing");

            m_eventFd.reset(::eventfd(initiallySignaled, EFD_CLOEXEC));
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "eventfd failed with error %d", errno);

            return true;
        }

        WaitableObjectDesc Desc()
        {
            return WaitableObjectDesc(m_eventFd, true);
        }

        bool Signal()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "AutoResetEvent is not initialized");

            const uint64_t value = 1;
            ssize_t result = ::write(m_eventFd, &value, sizeof(uint64_t));
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "write to eventfd failed with error %d", errno);

            return true;
        }

        bool Wait()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_eventFd, false, 50, "AutoResetEvent is not initialized");

            uint64_t value;
            ssize_t result;
            do
            {
                result = ::read(m_eventFd, &value, sizeof(uint64_t));
            } while (result == -1 && errno == EINTR);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "read from eventfd failed with error %d", errno);

            return true;
        }

    private:
        UniqueLinuxFd m_eventFd;
    };

    class DeadlineTimer
    {
    public:
        ~DeadlineTimer() = default;
        DeadlineTimer() = default;

        DeadlineTimer(const DeadlineTimer&) = delete;
        DeadlineTimer(DeadlineTimer&&) = default;

        DeadlineTimer& operator=(const DeadlineTimer&) = delete;
        DeadlineTimer& operator=(DeadlineTimer&&) = default;

        void Reset() noexcept
        {
            m_timerFd.reset();
        }

        bool Initialize()
        {
            NVTRACE_LOG_ERR_RETURN_IF(m_timerFd, false, 50, "DeadlineTimer already initialized, call Reset() before re-initializing");

            m_timerFd.reset(::timerfd_create(CLOCK_REALTIME, TFD_CLOEXEC));
            NVTRACE_LOG_ERR_RETURN_IF(!m_timerFd, false, 50, "timerfd_create failed with error %d", errno);

            return true;
        }

        WaitableObjectDesc Desc()
        {
            return WaitableObjectDesc(m_timerFd, true);
        }

        bool Arm(uint64_t deadlineUtcNsec, bool isRelativeTime)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_timerFd, false, 50, "DeadlineTimer is not initialized");

            // If deadlineUtcNsec is zero, the user expects the timer to be signaled immediately.
            // However, timerfd_settime with a 0 value disarms the timer, so we need to handle this case explicitly.
            // We do so by setting the deadline to an absolute time in the past.
            if (deadlineUtcNsec == 0)
            {
                deadlineUtcNsec = 1;
                isRelativeTime = false;
            }

            itimerspec timerSpec = {};
            timerSpec.it_value.tv_sec = deadlineUtcNsec / 1'000'000'000;
            timerSpec.it_value.tv_nsec = deadlineUtcNsec % 1'000'000'000;
            int result = ::timerfd_settime(m_timerFd, isRelativeTime ? 0 : TFD_TIMER_ABSTIME, &timerSpec, nullptr);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "timerfd_settime failed with error %d", errno);

            return true;
        }

        bool Disarm()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_timerFd, false, 50, "DeadlineTimer is not initialized");

            itimerspec timerSpec = {};
            int result = ::timerfd_settime(m_timerFd, 0, &timerSpec, nullptr);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "timerfd_settime failed with error %d", errno);

            return true;
        }

        bool Wait()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_timerFd, false, 50, "DeadlineTimer is not initialized");

            uint64_t value;
            ssize_t result;
            do
            {
                result = ::read(m_timerFd, &value, sizeof(uint64_t));
            } while (result == -1 && errno == EINTR);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "read from timerfd failed with error %d", errno);

            return true;
        }

    private:
        UniqueLinuxFd m_timerFd;
    };

    class WaitMultiplexer
    {
    public:
        ~WaitMultiplexer() = default;
        WaitMultiplexer() = default;

        WaitMultiplexer(const WaitMultiplexer&) = delete;
        WaitMultiplexer(WaitMultiplexer&& rhs) noexcept
            : m_epollFd(std::move(rhs.m_epollFd))
            , m_maxObjectCount(std::exchange(rhs.m_maxObjectCount, 0))
            , m_objectCount(std::exchange(rhs.m_objectCount, 0))
            , m_lastSignaledObjectInfo(std::exchange(rhs.m_lastSignaledObjectInfo, 0))
            , m_hasLastSignaledObject(std::exchange(rhs.m_hasLastSignaledObject, false))
        {
        }

        WaitMultiplexer& operator=(const WaitMultiplexer&) = delete;
        WaitMultiplexer& operator=(WaitMultiplexer&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_epollFd = std::move(rhs.m_epollFd);
                m_maxObjectCount = std::exchange(rhs.m_maxObjectCount, 0);
                m_objectCount = std::exchange(rhs.m_objectCount, 0);
                m_lastSignaledObjectInfo = std::exchange(rhs.m_lastSignaledObjectInfo, 0);
                m_hasLastSignaledObject = std::exchange(rhs.m_hasLastSignaledObject, false);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_hasLastSignaledObject = false;
            m_lastSignaledObjectInfo = 0;
            m_objectCount = 0;
            m_maxObjectCount = 0;
            m_epollFd.reset();
        }

        bool Initialize(uint32_t maxObjectCount)
        {
            NVTRACE_LOG_ERR_RETURN_IF(m_epollFd, false, 50, "WaitMultiplexer is already initialized, call Reset() before re-initializing");
            NVTRACE_LOG_ERR_RETURN_IF(maxObjectCount == 0, false, 50, "maxObjectCount must be greater than 0");

            UniqueLinuxFd epollFd(::epoll_create1(EPOLL_CLOEXEC));
            NVTRACE_LOG_ERR_RETURN_IF(!epollFd, false, 50, "epoll_create1 failed with error %d", errno);

            // Finalize
            m_epollFd = std::move(epollFd);
            m_maxObjectCount = maxObjectCount;
            return true;
        }

        bool AddObject(uint32_t identifier, WaitableObjectDesc objectDesc)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_epollFd, false, 50, "WaitMultiplexer is not initialized");
            NVTRACE_LOG_ERR_RETURN_IF(m_maxObjectCount <= m_objectCount, false, 50, "WaitMultiplexer is full, cannot add more objects (objectCount=%u maxObjectCount=%u identifier=%u fd=%d)", m_objectCount, m_maxObjectCount, identifier, objectDesc.handle);

            epoll_event epollEvent;
            epollEvent.events = EPOLLIN;
            epollEvent.data.u64 = EncodeObjectInfo(identifier, objectDesc);
            int result = ::epoll_ctl(m_epollFd, EPOLL_CTL_ADD, objectDesc.handle, &epollEvent);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "epoll_ctl(EPOLL_CTL_ADD) failed with error %d (identifier=%u fd=%d)", errno, identifier, objectDesc.handle);

            ++m_objectCount;
            return true;
        }
        template <typename TIdentifier>
        std::enable_if_t<std::is_enum_v<TIdentifier> && sizeof(TIdentifier) == sizeof(uint32_t), bool> AddObject(TIdentifier identifier, WaitableObjectDesc objectDesc)
        {
            return AddObject(static_cast<uint32_t>(identifier), objectDesc);
        }

        bool RemoveLastSignaledObject()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_epollFd, false, 50, "WaitMultiplexer is not initialized");
            NVTRACE_LOG_ERR_RETURN_IF(!m_hasLastSignaledObject, false, 50, "WaitMultiplexer has no object returned by Wait()");

            const uint32_t identifier = DecodeIdentifier(m_lastSignaledObjectInfo);
            const int handle = DecodeObjectFd(m_lastSignaledObjectInfo);
            int result = ::epoll_ctl(m_epollFd, EPOLL_CTL_DEL, handle, nullptr);
            NVTRACE_LOG_ERR_RETURN_IF(result == -1, false, 50, "epoll_ctl(EPOLL_CTL_DEL) failed with error %d (identifier=%u fd=%d)", errno, identifier, handle);

            --m_objectCount;
            m_lastSignaledObjectInfo = 0;
            m_hasLastSignaledObject = false;
            return true;
        }

        bool Wait(uint32_t& out_signaledIdentifier)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_epollFd, false, 50, "WaitMultiplexer is not initialized");
            NVTRACE_LOG_ERR_RETURN_IF(m_objectCount == 0, false, 50, "WaitMultiplexer is empty, add objects to it before calling Wait() (objectCount=%u)", m_objectCount);

            while (true)
            {
                epoll_event epollEvent;
                int epollResult = ::epoll_wait(m_epollFd, &epollEvent, 1, INFINITE_TIMEOUT);
                if (epollResult == -1)
                {
                    NVTRACE_LOG_ERR_RETURN_IF(errno != EINTR, false, 50, "epoll_wait failed with error %d", errno);
                    continue;
                }

                if (DecodeIsAutoReset(epollEvent.data.u64))
                {
                    const int signaledObjectFd = DecodeObjectFd(epollEvent.data.u64);
                    uint64_t value;
                    ssize_t readResult = ::read(signaledObjectFd, &value, sizeof(uint64_t));
                    if (readResult == -1)
                    {
                        NVTRACE_LOG_ERR_RETURN_IF(errno != EAGAIN, false, 50, "read from auto-reset eventfd failed with error %d (fd=%d)", errno, signaledObjectFd);
                        continue;
                    }
                }

                out_signaledIdentifier = DecodeIdentifier(epollEvent.data.u64);
                m_lastSignaledObjectInfo = epollEvent.data.u64;
                m_hasLastSignaledObject = true;
                break;
            }

            return true;
        }
        template <typename TIdentifier>
        std::enable_if_t<std::is_enum_v<TIdentifier> && sizeof(TIdentifier) == sizeof(uint32_t), bool> Wait(TIdentifier& out_signaledIdentifier)
        {
            return Wait(reinterpret_cast<uint32_t&>(out_signaledIdentifier));
        }

    private:
        static constexpr uint64_t kAutoResetFlag = uint64_t{1} << 63;

        static uint64_t EncodeObjectInfo(uint32_t identifier, WaitableObjectDesc objectDesc)
        {
            static_assert(sizeof(int) == sizeof(uint32_t));
            return static_cast<uint64_t>(identifier) | (static_cast<uint64_t>(objectDesc.handle) << 32) | (objectDesc.isAutoReset ? kAutoResetFlag : 0);
        }
        static uint32_t DecodeIdentifier(uint64_t encodedObjectInfo)
        {
            return static_cast<uint32_t>(encodedObjectInfo);
        }
        static int DecodeObjectFd(uint64_t encodedObjectInfo)
        {
            return static_cast<int>((encodedObjectInfo & ~kAutoResetFlag) >> 32);
        }
        static bool DecodeIsAutoReset(uint64_t encodedObjectInfo)
        {
            return (encodedObjectInfo & kAutoResetFlag);
        }

    private:
        UniqueLinuxFd m_epollFd;

        uint32_t m_maxObjectCount = 0;
        uint32_t m_objectCount = 0;
        uint64_t m_lastSignaledObjectInfo = 0;
        bool m_hasLastSignaledObject = false;
    };

} // namespace nv::trace

#endif
