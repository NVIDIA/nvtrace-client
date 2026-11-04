/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cuda_api_trace_v1.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace_v1.h>
#include <NvTrace/DataFormats/nvtrace_thread_event_v1.h>
#include <NvTrace/Target/Consumer/NvTraceProviderTypes.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceCpuCalibrationPointTypes.h>
#include <NvTrace/Target/NvTraceCpuTimestamp.h>
#include <NvTrace/Target/NvTraceGpuPartitionInfo_generated.h>
#include <NvTrace/Target/NvTraceMemory.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceSharedMemory.h>
#include <NvTrace/Target/NvTraceSystemInfo_generated.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/NvTraceWaitableObjects.h>
#include <NvTrace/Target/Observer/NvTraceCpuClockCalibrationObserver.h>
#include <NvTrace/Target/Observer/NvTraceCudaApiTraceObserver.h>
#include <NvTrace/Target/Observer/NvTraceGpuClockCalibrationObserver.h>
#include <NvTrace/Target/Observer/NvTraceGpuGrCtxTraceObserver.h>
#include <NvTrace/Target/Observer/NvTraceGpuHwTraceObserver.h>
#include <NvTrace/Target/Observer/NvTraceGpuPmCounterObserver.h>
#include <NvTrace/Target/Observer/NvTraceNvtxTraceObserver.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventObserver.h>
#include <NvTrace/Target/Router/NvTraceRouterObserverClient.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nv::trace {

    /// Best-effort executable path for the process at PID `pid` (globalPid when known, else the
    /// namespacePid fallback -- same convention as the processKey), read from /proc/<pid>/exe. Call
    /// at ProcessCreated-notification receipt, while the process is registered and alive -- reading
    /// later (e.g. at snapshot export) risks the PID being reused by then. The PID is only
    /// meaningful in the namespace where it was resolved, so before trusting the readlink the
    /// process's PID-namespace identity -- the (st_dev, st_ino) of /proc/<pid>/ns/pid, same
    /// convention as IpcServer::GetClientPidNamespaceId -- must match the identity captured at
    /// registration. That rejects a consumer sitting in a different PID namespace, where `pid`
    /// would name an unrelated process. Returns "" when the process is gone, unreadable,
    /// mismatched, or on Windows.
    inline std::string ReadProcessExecutablePath(NVTRACE_PID pid, const NvTrace_LinuxNamespaceId& pidNamespaceId)
    {
#if defined(__linux__)
        if (pid == 0)
        {
            return {};
        }

        if (pidNamespaceId.device != 0 || pidNamespaceId.inode != 0)
        {
            char nsPath[64];
            std::snprintf(nsPath, sizeof(nsPath), "/proc/%d/ns/pid", static_cast<int>(pid));
            struct stat nsStat = {};
            if (stat(nsPath, &nsStat) != 0 || static_cast<uint64_t>(nsStat.st_dev) != pidNamespaceId.device || static_cast<uint64_t>(nsStat.st_ino) != pidNamespaceId.inode)
            {
                return {};
            }
        }

        char exeLink[64];
        std::snprintf(exeLink, sizeof(exeLink), "/proc/%d/exe", static_cast<int>(pid));
        char exePath[PATH_MAX] = {};
        const ssize_t exePathLength = readlink(exeLink, exePath, sizeof(exePath) - 1);
        if (exePathLength > 0)
        {
            return std::string(exePath, static_cast<size_t>(exePathLength));
        }

        // readlink(exe) is ptrace-gated, and a containerized workload can deny it to every
        // host-side reader (dumpable cleared / LSM) even for the same uid. argv[0] from cmdline
        // is not gated -- it is what `ps` reads across users. Less authoritative (a process can
        // rewrite its argv), but the right display value in the common case.
        char cmdlinePath[64];
        std::snprintf(cmdlinePath, sizeof(cmdlinePath), "/proc/%d/cmdline", static_cast<int>(pid));
        if (FILE* pCmdline = std::fopen(cmdlinePath, "rb"))
        {
            char argv0[PATH_MAX] = {};
            const size_t bytesRead = std::fread(argv0, 1, sizeof(argv0) - 1, pCmdline);
            std::fclose(pCmdline);
            if (bytesRead > 0 && argv0[0] != '\0')
            {
                return std::string(argv0); // cmdline is NUL-delimited; this stops at argv[0].
            }
        }
        return {};
#endif
    }

    struct ConsumerObjectModelBase
    {
        struct ObjectState
        {
            NVTRACE_LUID luid = NVTRACE_NIL_LUID;
        };

        struct ProviderState : ObjectState
        {
            // Not bound to a ThreadEvent provider. A consumer-local sentinel that no record identifier's 12-bit
            // source provider index bitfield can hold.
            static constexpr uint32_t kInvalidThreadEventSourceProviderIndex = UINT32_MAX;

            UniqueSharedMemoryPtr pProviderBufferMemory;
            size_t providerBufferSize = 0;
            NVTRACE_LUID threadEventProviderLuid = NVTRACE_NIL_LUID;
            uint32_t threadEventSourceProviderIndex = kInvalidThreadEventSourceProviderIndex;
            NVTRACE_LUID processLuid = NVTRACE_NIL_LUID;

            void Reset() noexcept
            {
                processLuid = NVTRACE_NIL_LUID;
                threadEventSourceProviderIndex = kInvalidThreadEventSourceProviderIndex;
                threadEventProviderLuid = NVTRACE_NIL_LUID;
                providerBufferSize = 0;
                pProviderBufferMemory.reset();
                luid = NVTRACE_NIL_LUID;
            }

            void Initialize(NVTRACE_LUID luid_, UniqueSharedMemoryPtr&& pProviderBufferMemory_, size_t providerBufferSize_)
            {
                luid = luid_;
                pProviderBufferMemory = std::move(pProviderBufferMemory_);
                providerBufferSize = providerBufferSize_;
            }

            bool IsInitialized() const
            {
                return luid != NVTRACE_NIL_LUID;
            }
        };

        template <typename TObserver, typename TProviderState>
        struct TypedProviderState : TProviderState
        {
            TObserver observer;

            void Reset() noexcept
            {
                observer.Reset();
                TProviderState::Reset();
            }

            bool Initialize(NVTRACE_LUID luid_, NvTrace_OsObjectIdentifier providerBufferMemoryIdentifier, size_t providerBufferSize)
            {
                UniqueSharedMemoryPtr pProviderBufferMemory;
                const void* pMappedProviderBufferMemory;
                if (!SharedMemory_ImportReadOnly(providerBufferMemoryIdentifier, providerBufferSize, pProviderBufferMemory, pMappedProviderBufferMemory))
                {
                    return false;
                }

                if (!observer.Initialize(reinterpret_cast<const uint8_t*>(pMappedProviderBufferMemory)))
                {
                    return false;
                }

                TProviderState::Initialize(luid_, std::move(pProviderBufferMemory), providerBufferSize);
                return true;
            }
        };
        using CpuClockCalibrationProviderState = TypedProviderState<CpuClockCalibrationObserver, ProviderState>;
        using GpuClockCalibrationProviderState = TypedProviderState<GpuClockCalibrationObserver, ProviderState>;
        using GpuPmCountersProviderState = TypedProviderState<GpuPmCounterObserver, ProviderState>;
        using GpuHwTraceProviderState = TypedProviderState<GpuHwTraceObserver, ProviderState>;
        using GpuGrCtxTraceProviderState = TypedProviderState<GpuGrCtxTraceObserver, ProviderState>;
        // Shadows the generic Initialize: the CUDA API Trace observer picks its buffer format from the
        // Router-provided provider type UUID.
        template <typename TProviderState>
        struct TypedCudaApiTraceProviderState : TypedProviderState<CudaApiTraceObserver, TProviderState>
        {
            bool Initialize(NVTRACE_LUID luid_, const NVTRACE_UUID& providerTypeUuid, NvTrace_OsObjectIdentifier providerBufferMemoryIdentifier, size_t providerBufferSize)
            {
                UniqueSharedMemoryPtr pProviderBufferMemory;
                const void* pMappedProviderBufferMemory;
                if (!SharedMemory_ImportReadOnly(providerBufferMemoryIdentifier, providerBufferSize, pProviderBufferMemory, pMappedProviderBufferMemory))
                {
                    return false;
                }

                if (!this->observer.Initialize(providerTypeUuid, reinterpret_cast<const uint8_t*>(pMappedProviderBufferMemory), providerBufferSize))
                {
                    return false;
                }

                TProviderState::Initialize(luid_, std::move(pProviderBufferMemory), providerBufferSize);
                return true;
            }
        };
        using CudaApiTraceProviderState = TypedCudaApiTraceProviderState<ProviderState>;

        // Shadows the generic Initialize: the NVTX Trace observer picks its buffer format from the
        // Router-provided provider type UUID.
        template <typename TProviderState>
        struct TypedNvtxTraceProviderState : TypedProviderState<NvtxTraceObserver, TProviderState>
        {
            bool Initialize(NVTRACE_LUID luid_, const NVTRACE_UUID& providerTypeUuid, NvTrace_OsObjectIdentifier providerBufferMemoryIdentifier, size_t providerBufferSize)
            {
                UniqueSharedMemoryPtr pProviderBufferMemory;
                const void* pMappedProviderBufferMemory;
                if (!SharedMemory_ImportReadOnly(providerBufferMemoryIdentifier, providerBufferSize, pProviderBufferMemory, pMappedProviderBufferMemory))
                {
                    return false;
                }

                if (!this->observer.Initialize(providerTypeUuid, reinterpret_cast<const uint8_t*>(pMappedProviderBufferMemory), providerBufferSize))
                {
                    return false;
                }

                TProviderState::Initialize(luid_, std::move(pProviderBufferMemory), providerBufferSize);
                return true;
            }
        };
        using NvtxTraceProviderState = TypedNvtxTraceProviderState<ProviderState>;

        // One source provider index of a ThreadEvent provider: what its registration named, and the
        // provider its records are delivered to once that provider is created.
        template <typename TProviderState>
        struct ThreadEventSourceProviderSlot
        {
            // NIL until the registration is drained
            NVTRACE_UUID providerTypeUuid = NVTRACE_NIL_UUID;

            // Null until the registration is drained; DropThreadEvent when this consumer does not consume the
            // registered type. Never null once bound.
            ThreadEventConsumeCallback pfnConsumeThreadEvent = nullptr;

            // Null while unbound; points into the owning ProcessState
            TProviderState* pProviderState = nullptr;

            // Whether an unbound index's records wait for its provider: set by the registration of a consumed
            // type; cleared when its provider is destroyed or the wait expires.
            bool isWaitingForProvider = false;

            // Until when the wait lasts: the first held record plus the hold timeout. 0 until a record is held.
            uint64_t holdDeadlineTsc = 0;

            bool isUnregisteredDropLogged = false;
        };

        template <typename TProviderState>
        struct TypedThreadEventProviderState : TypedProviderState<ThreadEventObserver, TProviderState>
        {
            using Super = TypedProviderState<ThreadEventObserver, TProviderState>;
            using SourceProviderSlot = ThreadEventSourceProviderSlot<TProviderState>;

            std::vector<SourceProviderSlot> sourceProviderSlots;
            uint64_t holdTimeoutTsc = 0;
            bool isOutOfRangeDropLogged = false;

            void Reset() noexcept
            {
                isOutOfRangeDropLogged = false;
                holdTimeoutTsc = 0;
                sourceProviderSlots.clear();
                Super::Reset();
            }

            // Shadows the generic overload: the ThreadEvent observer picks its buffer format from the
            // Router-provided provider type UUID.
            bool Initialize(NVTRACE_LUID luid_, const NVTRACE_UUID& providerTypeUuid, NvTrace_OsObjectIdentifier providerBufferMemoryIdentifier, size_t providerBufferSize)
            {
                UniqueSharedMemoryPtr pProviderBufferMemory;
                const void* pMappedProviderBufferMemory;
                if (!SharedMemory_ImportReadOnly(providerBufferMemoryIdentifier, providerBufferSize, pProviderBufferMemory, pMappedProviderBufferMemory))
                {
                    return false;
                }

                if (!this->observer.Initialize(providerTypeUuid, reinterpret_cast<const uint8_t*>(pMappedProviderBufferMemory), providerBufferSize))
                {
                    return false;
                }

                TProviderState::Initialize(luid_, std::move(pProviderBufferMemory), providerBufferSize);
                sourceProviderSlots.assign(this->observer.MaxSourceProviderCount(), SourceProviderSlot{});
                return true;
            }

            SourceProviderSlot* FindSourceProviderSlot(uint32_t sourceProviderIndex)
            {
                return sourceProviderIndex < sourceProviderSlots.size() ? &sourceProviderSlots[sourceProviderIndex] : nullptr;
            }

            const SourceProviderSlot* FindSourceProviderSlot(uint32_t sourceProviderIndex) const
            {
                return sourceProviderIndex < sourceProviderSlots.size() ? &sourceProviderSlots[sourceProviderIndex] : nullptr;
            }
        };
        using ThreadEventProviderState = TypedThreadEventProviderState<ProviderState>;

        struct NullProviderState
        {
        };

        template <typename TGpuClockCalibrationProviderState, typename TGpuPmCountersProviderState, typename TGpuHwTraceProviderState, typename TGpuGrCtxTraceProviderState>
        struct GpuPartitionState : ObjectState
        {
            NVTRACE_UUID gpuUuid = NVTRACE_NIL_UUID;
            uint64_t creationTimestampTsc = 0;

            UniqueSharedMemoryPtr pGpuPartitionInfoMemory;
            FlatBufferView<FB::GpuPartitionInfo> gpuPartitionInfo;

            TGpuClockCalibrationProviderState gpuClockCalibrationProvider;
            TGpuPmCountersProviderState gpuPmCountersProvider;
            TGpuHwTraceProviderState gpuHwTraceProvider;
            TGpuGrCtxTraceProviderState gpuGrCtxTraceProvider;
        };

        template <typename TCudaApiTraceProviderState, typename TNvtxTraceProviderState, typename TThreadEventProviderState>
        struct ProcessState : ObjectState
        {
            NVTRACE_PID namespacePid = 0;                 // Namespace PID (virtualized inside a container).
            NVTRACE_PID globalPid = 0;                    // Host PID from the Router. 0 => fall back to namespacePid.
            NvTrace_LinuxNamespaceId pidNamespaceId = {}; // (device, inode) of /proc/<pid>/ns/pid. Both 0 if unavailable. Display only.
            std::string path;                             // Executable path, captured at registration while the process is alive. "" if unreadable.
            uint64_t creationTimestampTsc = 0;

            TCudaApiTraceProviderState cudaApiTraceProvider;
            TNvtxTraceProviderState nvtxTraceProvider;
            std::map<NVTRACE_LUID, TThreadEventProviderState> threadEventProviders;

            TThreadEventProviderState* FindThreadEventProvider(NVTRACE_LUID providerLuid)
            {
                const auto itrProvider = threadEventProviders.find(providerLuid);
                return itrProvider != threadEventProviders.end() ? &itrProvider->second : nullptr;
            }

            const TThreadEventProviderState* FindThreadEventProvider(NVTRACE_LUID providerLuid) const
            {
                const auto itrProvider = threadEventProviders.find(providerLuid);
                return itrProvider != threadEventProviders.end() ? &itrProvider->second : nullptr;
            }
        };

        template <typename TCpuClockCalibrationProviderState, typename TProviderState, typename TGpuPartitionState, typename TProcessState>
        struct SystemState
        {
            UniqueSharedMemoryPtr pSystemInfoMemory;
            FlatBufferView<FB::SystemInfo> systemInfo;

            TCpuClockCalibrationProviderState cpuClockCalibrationProvider;
            TscRtcCalibrationPoint initialTscRtcCalibrationPoint = {};

            std::map<NVTRACE_LUID, TProviderState*> providerLuidToProviderStateMap;
            std::map<NVTRACE_LUID, TGpuPartitionState> gpuPartitions;
            std::map<NVTRACE_LUID, TProcessState> processes;

            void RegisterProviderState(TProviderState& providerState)
            {
                if (providerState.IsInitialized())
                {
                    providerLuidToProviderStateMap[providerState.luid] = &providerState;
                }
            }

            void UnregisterProviderState(NVTRACE_LUID providerLuid)
            {
                providerLuidToProviderStateMap.erase(providerLuid);
            }

            TProviderState* GetProviderState(NVTRACE_LUID providerLuid)
            {
                const auto itrProviderState = providerLuidToProviderStateMap.find(providerLuid);
                if (itrProviderState == providerLuidToProviderStateMap.end())
                {
                    return nullptr;
                }
                return itrProviderState->second;
            }
        };
    };


    enum class ProviderDestroyReason
    {
        Explicit,
        ParentDestroyed,
    };

    template <typename TDerivedConsumer, typename TConsumerObjectModel = ConsumerObjectModelBase>
    class ConsumerBase
    {
    protected:
        using ProviderState = typename TConsumerObjectModel::ProviderState;
        using CpuClockCalibrationProviderState = typename TConsumerObjectModel::CpuClockCalibrationProviderState;
        using GpuClockCalibrationProviderState = typename TConsumerObjectModel::GpuClockCalibrationProviderState;
        using GpuPmCountersProviderState = typename TConsumerObjectModel::GpuPmCountersProviderState;
        using GpuHwTraceProviderState = typename TConsumerObjectModel::GpuHwTraceProviderState;
        using GpuGrCtxTraceProviderState = typename TConsumerObjectModel::GpuGrCtxTraceProviderState;
        using CudaApiTraceProviderState = typename TConsumerObjectModel::CudaApiTraceProviderState;
        using NvtxTraceProviderState = typename TConsumerObjectModel::NvtxTraceProviderState;
        using ThreadEventProviderState = typename TConsumerObjectModel::ThreadEventProviderState;
        using GpuPartitionState = typename TConsumerObjectModel::template GpuPartitionState<GpuClockCalibrationProviderState, GpuPmCountersProviderState, GpuHwTraceProviderState, GpuGrCtxTraceProviderState>;
        using ProcessState = typename TConsumerObjectModel::template ProcessState<CudaApiTraceProviderState, NvtxTraceProviderState, ThreadEventProviderState>;
        using SystemState = typename TConsumerObjectModel::template SystemState<CpuClockCalibrationProviderState, ProviderState, GpuPartitionState, ProcessState>;

    protected:
        struct Options
        {
            const NvTrace_RouterObserverClient_GpuPartitionFilter* pGpuPartitionFilter = nullptr;
            const NvTrace_RouterObserverClient_ProcessFilter* pProcessFilter = nullptr;
            Codec::ProviderTypeTable<bool> isProviderTypeIgnored = {};
            uint64_t consumeIntervalMsec = 100; // Default update interval in milliseconds.

            // How long a thread's records wait for their provider's ProviderCreated before being
            // dropped. 0 = the default, min(1 s, a quarter of the time one thread event buffer holds).
            uint64_t threadEventHoldTimeoutNsec = 0;
        };

        static constexpr uint64_t ConsumeUpToGracePeriodNsec = 250'000'000;
        static constexpr uint64_t MaxDefaultThreadEventHoldTimeoutNsec = 1'000'000'000;

        // A provider type UUID naming an older buffer format of a provider type whose observer still serves
        // it. Providers advertise the alias; ConsumerBase routes by the provider type it resolves to and hands
        // the alias itself only to that type's observer, which picks the format from it. Nothing past
        // ConsumerBase sees an alias.
        struct ProviderTypeUuidAlias
        {
            NVTRACE_UUID aliasProviderTypeUuid;
            NVTRACE_UUID providerTypeUuid;
        };

        static constexpr ProviderTypeUuidAlias ProviderTypeUuidAliases[] = {
            {NvTrace_CUDA_ApiTrace_ProviderTypeUuid_V1, NvTrace_CUDA_ApiTrace_ProviderTypeUuid},
            {    NvTrace_NvtxTrace_ProviderTypeUuid_V1,     NvTrace_NvtxTrace_ProviderTypeUuid},
            {  NvTrace_ThreadEvent_ProviderTypeUuid_V1,   NvTrace_ThreadEvent_ProviderTypeUuid},
        };
        static constexpr size_t ProviderTypeUuidAliasCount = std::size(ProviderTypeUuidAliases);

        static NVTRACE_UUID ResolveProviderTypeUuidAlias(const NVTRACE_UUID& providerTypeUuid)
        {
            for (const ProviderTypeUuidAlias& alias : ProviderTypeUuidAliases)
            {
                if (alias.aliasProviderTypeUuid == providerTypeUuid)
                {
                    return alias.providerTypeUuid;
                }
            }

            return providerTypeUuid;
        }

        static bool DropThreadEvent(void* /* pConsumer */, void* /* pProcessState */, void* /* pProviderState */, const ThreadEventRecord& /* record */)
        {
            return true;
        }

    protected:
        ~ConsumerBase() = default;
        ConsumerBase() = default;

        ConsumerBase(const ConsumerBase&) = delete;
        ConsumerBase& operator=(const ConsumerBase&) = delete;

        void Reset() noexcept
        {
            m_systemState.providerLuidToProviderStateMap.clear();
            m_systemState.processes.clear();
            m_systemState.gpuPartitions.clear();
            m_systemState.cpuClockCalibrationProvider.Reset();
            m_systemState.initialTscRtcCalibrationPoint = {};
            m_consumeCpuClockCalibrationPoint = {};
            m_droppedUnboundThreadEventCount = 0;
            m_isHoldingUnboundThreadEvents = true;
            m_threadEventHoldTimeoutNsec = 0;
            m_consumeIntervalNsec = 0;
            m_consumeTimer.Reset();
            m_pRouterObserverClient.Reset();
            m_isProviderTypeIgnored.fill(false);
            m_threadEventConsumeCallbacks.fill(nullptr);
        }

        bool Initialize(const Options& options, NvTrace_RouterObserverClient_RouterConfig* out_pRouterConfig = nullptr)
        {
            auto errorGuard = ScopeExitGuard([this]() { Reset(); });

            m_isProviderTypeIgnored = options.isProviderTypeIgnored;
            m_isProviderTypeIgnored[static_cast<size_t>(Codec::ProviderType::Invalid)] = false;

            // Figure out the provider type UUIDs we're interested in based on the non-null provider states
            const auto [filterProviderTypeUuids, filterProviderTypeUuidCount] = [&]() {
                std::array<NVTRACE_UUID, ProviderTypeUuidMappingCount + ProviderTypeUuidAliasCount> uuids = {};
                uint32_t uuidIndex = 0;

                for (const ProviderTypeUuidMapping& mapping : ProviderTypeUuidMappingTable)
                {
                    bool hasProviderState = false;
                    switch (mapping.providerType)
                    {
                        case Codec::ProviderType::CpuClockCalibration:
                            hasProviderState = !std::is_empty_v<CpuClockCalibrationProviderState>;
                            break;
                        case Codec::ProviderType::GpuClockCalibration:
                            hasProviderState = !std::is_empty_v<GpuClockCalibrationProviderState>;
                            break;
                        case Codec::ProviderType::GpuPmCounters:
                            hasProviderState = !std::is_empty_v<GpuPmCountersProviderState>;
                            break;
                        case Codec::ProviderType::GpuHwTrace:
                            hasProviderState = !std::is_empty_v<GpuHwTraceProviderState>;
                            break;
                        case Codec::ProviderType::GpuGrCtxTrace:
                            hasProviderState = !std::is_empty_v<GpuGrCtxTraceProviderState>;
                            break;
                        case Codec::ProviderType::CudaApiTrace:
                            hasProviderState = !std::is_empty_v<CudaApiTraceProviderState>;
                            break;
                        case Codec::ProviderType::NvtxTrace:
                            hasProviderState = !std::is_empty_v<NvtxTraceProviderState>;
                            break;
                        case Codec::ProviderType::ThreadEvent:
                            hasProviderState = !std::is_empty_v<ThreadEventProviderState>;
                            break;
                        case Codec::ProviderType::Invalid:
                        case Codec::ProviderType::Unknown:
                        case Codec::ProviderType::Count:
                            break;
                    }

                    if (!hasProviderState || IsProviderTypeIgnored(mapping.providerType))
                    {
                        continue;
                    }

                    uuids[uuidIndex++] = mapping.providerTypeUuid;
                    for (const ProviderTypeUuidAlias& alias : ProviderTypeUuidAliases)
                    {
                        if (alias.providerTypeUuid == mapping.providerTypeUuid)
                        {
                            uuids[uuidIndex++] = alias.aliasProviderTypeUuid;
                        }
                    }
                }

                return std::pair{uuids, uuidIndex};
            }();

            // Create router observer client
            NvTrace_RouterObserverClient_ProviderTypeFilter providerTypeFilter = {filterProviderTypeUuids.data(), filterProviderTypeUuidCount};
            constexpr uint32_t timeoutMsec = 1000;
            NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint initialTscRtcCalibrationPoint = {};
            m_pRouterObserverClient.Reset(New<RouterObserverClient>());
            NVTRACE_LOG_ERR_RETURN_IF(!m_pRouterObserverClient, false, 50, "Failed to allocate RouterObserverClient");
            bool succeeded = m_pRouterObserverClient->Initialize(&providerTypeFilter, options.pGpuPartitionFilter, options.pProcessFilter, timeoutMsec, &initialTscRtcCalibrationPoint, out_pRouterConfig);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "RouterObserverClient::Initialize failed");
            m_systemState.initialTscRtcCalibrationPoint.tscTick = initialTscRtcCalibrationPoint.tscTick;
            m_systemState.initialTscRtcCalibrationPoint.utcNsec = initialTscRtcCalibrationPoint.utcNsec;
            m_systemState.initialTscRtcCalibrationPoint.tscToUtcFactor = initialTscRtcCalibrationPoint.tscToUtcFactor;
            m_systemState.initialTscRtcCalibrationPoint.utcToTscFactor = initialTscRtcCalibrationPoint.utcToTscFactor;

            // Create consume timer
            succeeded = m_consumeTimer.Initialize();
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "DeadlineTimer::Initialize failed");

            m_consumeIntervalNsec = options.consumeIntervalMsec * 1'000'000;
            m_threadEventHoldTimeoutNsec = options.threadEventHoldTimeoutNsec;
            m_threadEventConsumeCallbacks.fill(nullptr);
            static_cast<TDerivedConsumer*>(this)->RegisterThreadEventConsumeCallbacks();

            // Receive initial state from the router synchronously
            bool initialStateReceived = false;
            while (!initialStateReceived)
            {
                succeeded = m_pRouterObserverClient->ReceiveNotification([this, &initialStateReceived](NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) {
                    if (notificationType == NvTrace_RouterObserverClient_NotificationType_InitialStateSent)
                    {
                        initialStateReceived = true;
                        return true;
                    }
                    else
                    {
                        return HandleRouterNotification(notificationType, pNotification);
                    }
                });
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "RouterObserverClient::ReceiveNotification failed");
            }

            errorGuard.Dismiss();
            return true;
        }

        /// Converts a recent CPU TSC timestamp to UTC using the latest available calibration point.
        /// Accuracy degrades for historical timestamps; use the data model's calibration history to
        /// convert timestamps from previously collected records.
        uint64_t ConvertTscTickToUtcNsec(uint64_t timestampTsc) const
        {
            return GetLatestCpuClockCalibrationPoint().ToUtc(timestampTsc);
        }

        // Waits on the calling thread while the consumer thread continues its normal consume loop.
        void ConsumeUpToUtc(uint64_t targetUtcNsec) const
        {
            const uint64_t deadlineUtcNsec = targetUtcNsec + ConsumeUpToGracePeriodNsec;
            while (true)
            {
                const TscRtcCalibrationPoint calibrationPoint = GetLatestCpuClockCalibrationPoint();
                NVTRACE_LOG_ERR_RETURN_IF(calibrationPoint.tscToUtcFactor == 0, , 50, "Cannot wait until a UTC timestamp without a valid TSC-to-UTC calibration point");

                const uint64_t nowUtcNsec = calibrationPoint.ToUtc(ReadCpuTscTimestamp());
                if (deadlineUtcNsec <= nowUtcNsec)
                {
                    return;
                }

                // Naive approach to avoid burdening the providers, deep packet inspection, and edge cases
                const uint64_t remainingNsec = deadlineUtcNsec - nowUtcNsec;
                std::this_thread::sleep_for(std::chrono::nanoseconds(remainingNsec));
            }
        }

        bool HandleRouterNotification(enum NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification)
        {
            switch (notificationType)
            {
                case NvTrace_RouterObserverClient_NotificationType_Invalid:
                    break;

                case NvTrace_RouterObserverClient_NotificationType_InitialStateSent:
                    return true;

                case NvTrace_RouterObserverClient_NotificationType_SystemCreated:
                {
                    const NvTrace_RouterObserverClient_SystemCreatedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_SystemCreatedNotification*>(pNotification);

                    UniqueSharedMemoryPtr pInfoMemory;
                    const void* pMappedInfoMemory;

                    if (!SharedMemory_ImportReadOnly(notification.infoMemoryIdentifier, notification.infoSize, pInfoMemory, pMappedInfoMemory))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to import shared memory for SystemCreated info");
                        return false;
                    }

                    FlatBufferView<FB::SystemInfo> systemInfo = {reinterpret_cast<const uint8_t*>(pMappedInfoMemory), notification.infoSize};
                    if (!systemInfo.Verify())
                    {
                        NVTRACE_LOG_ERR(50, "Invalid SystemCreated info");
                        return false;
                    }

                    m_systemState.pSystemInfoMemory = std::move(pInfoMemory);
                    m_systemState.systemInfo = systemInfo;

                    static_cast<TDerivedConsumer*>(this)->OnSystemCreated(m_systemState);
                    return true;
                }

                case NvTrace_RouterObserverClient_NotificationType_GpuPartitionCreated:
                {
                    const NvTrace_RouterObserverClient_GpuPartitionCreatedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_GpuPartitionCreatedNotification*>(pNotification);

                    UniqueSharedMemoryPtr pInfoMemory;
                    const void* pMappedInfoMemory;
                    const size_t& infoSize = notification.infoSize;

                    if (!SharedMemory_ImportReadOnly(notification.infoMemoryIdentifier, infoSize, pInfoMemory, pMappedInfoMemory))
                    {
                        NVTRACE_LOG_ERR(50, "Failed to import shared memory for GpuPartitionCreated info");
                        return false;
                    }

                    FlatBufferView<FB::GpuPartitionInfo> gpuPartitionInfo = {reinterpret_cast<const uint8_t*>(pMappedInfoMemory), infoSize};
                    if (!gpuPartitionInfo.Verify())
                    {
                        NVTRACE_LOG_ERR(50, "Invalid GpuPartitionCreated info");
                        return false;
                    }

                    auto [itrGpuPartitionState, inserted] = m_systemState.gpuPartitions.try_emplace(notification.gpuPartitionLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(!inserted, false, 50, "Received GpuPartitionCreated notification for an already existing GpuPartition with LUID %llu", notification.gpuPartitionLuid);

                    GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;
                    gpuPartitionState.luid = notification.gpuPartitionLuid;
                    gpuPartitionState.gpuUuid = notification.gpuUuid;
                    gpuPartitionState.creationTimestampTsc = notification.creationTimestampTsc;
                    gpuPartitionState.pGpuPartitionInfoMemory = std::move(pInfoMemory);
                    gpuPartitionState.gpuPartitionInfo = gpuPartitionInfo;

                    static_cast<TDerivedConsumer*>(this)->OnGpuPartitionCreated(gpuPartitionState);
                    return true;
                }

                case NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed:
                {
                    const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification*>(pNotification);
                    auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.gpuPartitionLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState == m_systemState.gpuPartitions.end(), false, 50, "Received GpuPartitionDestroyed notification for a non-existing GpuPartition with LUID %llu", notification.gpuPartitionLuid);

                    GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;

                    // If the GpuPartition was destroyed, it implicitly means that all Providers within it are also destroyed
                    if constexpr (!std::is_empty_v<GpuClockCalibrationProviderState>)
                    {
                        if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuClockCalibrationProvider, ProviderDestroyReason::ParentDestroyed);
                            m_systemState.UnregisterProviderState(gpuPartitionState.gpuClockCalibrationProvider.luid);
                            gpuPartitionState.gpuClockCalibrationProvider.Reset();
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuPmCountersProviderState>)
                    {
                        if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuPmCountersProvider, ProviderDestroyReason::ParentDestroyed);
                            m_systemState.UnregisterProviderState(gpuPartitionState.gpuPmCountersProvider.luid);
                            gpuPartitionState.gpuPmCountersProvider.Reset();
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuHwTraceProviderState>)
                    {
                        if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuHwTraceProvider, ProviderDestroyReason::ParentDestroyed);
                            m_systemState.UnregisterProviderState(gpuPartitionState.gpuHwTraceProvider.luid);
                            gpuPartitionState.gpuHwTraceProvider.Reset();
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuGrCtxTraceProviderState>)
                    {
                        if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuGrCtxTraceProvider, ProviderDestroyReason::ParentDestroyed);
                            m_systemState.UnregisterProviderState(gpuPartitionState.gpuGrCtxTraceProvider.luid);
                            gpuPartitionState.gpuGrCtxTraceProvider.Reset();
                        }
                    }

                    static_cast<TDerivedConsumer*>(this)->OnGpuPartitionDestroyed(gpuPartitionState, notification.destructionTimestampTsc);
                    m_systemState.gpuPartitions.erase(itrGpuPartitionState);
                    return true;
                }

                case NvTrace_RouterObserverClient_NotificationType_ProcessCreated:
                {
                    const NvTrace_RouterObserverClient_ProcessCreatedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProcessCreatedNotification*>(pNotification);
                    auto [itrProcessState, inserted] = m_systemState.processes.try_emplace(notification.processLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(!inserted, false, 50, "Received ProcessCreated notification for an already existing Process with LUID %llu", notification.processLuid);

                    ProcessState& processState = itrProcessState->second;
                    processState.luid = notification.processLuid;
                    processState.namespacePid = notification.namespacePid;
                    processState.globalPid = notification.globalPid;
                    processState.pidNamespaceId = notification.pidNamespaceId;
                    processState.creationTimestampTsc = notification.creationTimestampTsc;
                    // Capture the executable path now, while the process is alive and its PID cannot
                    // yet have been reused -- snapshot consumers encode ProcessCreated much later.
                    // Same fallback convention as the processKey (RouterServer::OnProcessCreated):
                    // globalPid when known, else namespacePid -- valid when the consumer shares the
                    // workload's PID namespace; the namespace-identity check in
                    // ReadProcessExecutablePath rejects it otherwise.
                    const NVTRACE_PID pathLookupPid = notification.globalPid != 0 ? notification.globalPid : notification.namespacePid;
                    processState.path = ReadProcessExecutablePath(pathLookupPid, notification.pidNamespaceId);

                    static_cast<TDerivedConsumer*>(this)->OnProcessCreated(processState);
                    return true;
                }

                case NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed:
                {
                    const NvTrace_RouterObserverClient_ProcessDestroyedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProcessDestroyedNotification*>(pNotification);
                    auto itrProcessState = m_systemState.processes.find(notification.processLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(itrProcessState == m_systemState.processes.end(), false, 50, "Received ProcessDestroyed notification for a non-existing Process with LUID %llu", notification.processLuid);

                    ProcessState& processState = itrProcessState->second;

                    // If the Process was destroyed, it implicitly means that all Providers within it are also destroyed

                    // Consume and destroy ThreadEvent providers while the source provider states bound to them are still available.
                    if constexpr (!std::is_empty_v<ThreadEventProviderState>)
                    {
                        const auto dropUnboundGuard = ScopeGuardDropUnboundThreadEvents();
                        for (auto& [threadEventProviderLuid, threadEventProviderState] : processState.threadEventProviders)
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_ThreadEvent(threadEventProviderState, processState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(threadEventProviderState, ProviderDestroyReason::ParentDestroyed);
                            m_systemState.UnregisterProviderState(threadEventProviderLuid);
                            threadEventProviderState.Reset();
                        }
                        processState.threadEventProviders.clear();
                    }

                    if constexpr (!std::is_empty_v<CudaApiTraceProviderState>)
                    {
                        if (processState.cudaApiTraceProvider.IsInitialized())
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(processState.cudaApiTraceProvider, ProviderDestroyReason::ParentDestroyed);
                            UnbindThreadEventSourceProvider(processState, processState.cudaApiTraceProvider);
                            m_systemState.UnregisterProviderState(processState.cudaApiTraceProvider.luid);
                            processState.cudaApiTraceProvider.Reset();
                        }
                    }
                    if constexpr (!std::is_empty_v<NvtxTraceProviderState>)
                    {
                        if (processState.nvtxTraceProvider.IsInitialized())
                        {
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_NvtxTrace(processState.nvtxTraceProvider, processState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(processState.nvtxTraceProvider, ProviderDestroyReason::ParentDestroyed);
                            UnbindThreadEventSourceProvider(processState, processState.nvtxTraceProvider);
                            m_systemState.UnregisterProviderState(processState.nvtxTraceProvider.luid);
                            processState.nvtxTraceProvider.Reset();
                        }
                    }
                    static_cast<TDerivedConsumer*>(this)->OnProcessDestroyed(processState, notification.destructionTimestampTsc);
                    m_systemState.processes.erase(itrProcessState);
                    return true;
                }

                case NvTrace_RouterObserverClient_NotificationType_ProviderCreated:
                {
                    const NvTrace_RouterObserverClient_ProviderCreatedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProviderCreatedNotification*>(pNotification);
                    const NVTRACE_UUID providerTypeUuid = ResolveProviderTypeUuidAlias(notification.providerTypeUuid);

                    if constexpr (!std::is_empty_v<CpuClockCalibrationProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid)
                        {
                            NVTRACE_LOG_ERR_RETURN_IF(m_systemState.cpuClockCalibrationProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a CpuClockCalibrationProvider within the same System");

                            bool succeeded = m_systemState.cpuClockCalibrationProvider.Initialize(notification.providerLuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize CpuClockCalibrationObserver for Provider with LUID %llu", notification.providerLuid);

                            m_systemState.RegisterProviderState(m_systemState.cpuClockCalibrationProvider);
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_CpuClockCalibration(m_systemState.cpuClockCalibrationProvider);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuClockCalibrationProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid)
                        {
                            auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState == m_systemState.gpuPartitions.end(), false, 50, "Received ProviderCreated notification for a GpuClockCalibrationProvider with an invalid parent GpuPartition LUID %llu", notification.parentLuid);

                            GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;
                            NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionState.gpuClockCalibrationProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a GpuClockCalibrationProvider within the same GpuPartition");

                            bool succeeded = gpuPartitionState.gpuClockCalibrationProvider.Initialize(notification.providerLuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize GpuClockCalibrationObserver for Provider with LUID %llu", notification.providerLuid);

                            m_systemState.RegisterProviderState(gpuPartitionState.gpuClockCalibrationProvider);
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuPmCountersProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_GPU_PmCounters_ProviderTypeUuid)
                        {
                            auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState == m_systemState.gpuPartitions.end(), false, 50, "Received ProviderCreated notification for a GpuPmCountersProvider with an invalid parent GpuPartition LUID %llu", notification.parentLuid);

                            GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;
                            NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionState.gpuPmCountersProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a GpuPmCountersProvider within the same GpuPartition");

                            bool succeeded = gpuPartitionState.gpuPmCountersProvider.Initialize(notification.providerLuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize GpuPmCountersObserver for Provider with LUID %llu", notification.providerLuid);

                            m_systemState.RegisterProviderState(gpuPartitionState.gpuPmCountersProvider);
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_GpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuHwTraceProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_GPU_HwTrace_ProviderTypeUuid)
                        {
                            auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState == m_systemState.gpuPartitions.end(), false, 50, "Received ProviderCreated notification for a GpuHwTraceProvider with an invalid parent GpuPartition LUID %llu", notification.parentLuid);

                            GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;
                            NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionState.gpuHwTraceProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a GpuHwTraceProvider within the same GpuPartition");

                            bool succeeded = gpuPartitionState.gpuHwTraceProvider.Initialize(notification.providerLuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize GpuHwTraceObserver for Provider with LUID %llu", notification.providerLuid);

                            m_systemState.RegisterProviderState(gpuPartitionState.gpuHwTraceProvider);
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_GpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<GpuGrCtxTraceProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_GPU_GrCtxTrace_ProviderTypeUuid)
                        {
                            auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrGpuPartitionState == m_systemState.gpuPartitions.end(), false, 50, "Received ProviderCreated notification for a GpuGrCtxTraceProvider with an invalid parent GpuPartition LUID %llu", notification.parentLuid);

                            GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;
                            NVTRACE_LOG_ERR_RETURN_IF(gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a GpuGrCtxTraceProvider within the same GpuPartition");

                            bool succeeded = gpuPartitionState.gpuGrCtxTraceProvider.Initialize(notification.providerLuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize GpuGrCtxTraceObserver for Provider with LUID %llu", notification.providerLuid);

                            m_systemState.RegisterProviderState(gpuPartitionState.gpuGrCtxTraceProvider);
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_GpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<CudaApiTraceProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_CUDA_ApiTrace_ProviderTypeUuid)
                        {
                            auto itrProcessState = m_systemState.processes.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrProcessState == m_systemState.processes.end(), false, 50, "Received ProviderCreated notification for a CudaApiTraceProvider with an invalid parent Process LUID %llu", notification.parentLuid);

                            ProcessState& processState = itrProcessState->second;
                            NVTRACE_LOG_ERR_RETURN_IF(processState.cudaApiTraceProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a CudaApiTraceProvider within the same Process");

                            const bool succeeded = processState.cudaApiTraceProvider.Initialize(notification.providerLuid, notification.providerTypeUuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize CudaApiTraceObserver for Provider with LUID %llu", notification.providerLuid);

                            processState.cudaApiTraceProvider.processLuid = processState.luid;
                            m_systemState.RegisterProviderState(processState.cudaApiTraceProvider);
                            BindThreadEventSourceProvider(processState, processState.cudaApiTraceProvider, providerTypeUuid, processState.cudaApiTraceProvider.observer.ThreadEventSourceProviderIndex());
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<NvtxTraceProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_NvtxTrace_ProviderTypeUuid)
                        {
                            auto itrProcessState = m_systemState.processes.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrProcessState == m_systemState.processes.end(), false, 50, "Received ProviderCreated notification for a NvtxTraceProvider with an invalid parent Process LUID %llu", notification.parentLuid);

                            ProcessState& processState = itrProcessState->second;
                            NVTRACE_LOG_ERR_RETURN_IF(processState.nvtxTraceProvider.IsInitialized(), false, 50, "Received more than one ProviderCreated notification for a NvtxTraceProvider within the same Process");

                            const bool succeeded = processState.nvtxTraceProvider.Initialize(notification.providerLuid, notification.providerTypeUuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize NvtxTraceObserver for Provider with LUID %llu", notification.providerLuid);

                            processState.nvtxTraceProvider.processLuid = processState.luid;
                            m_systemState.RegisterProviderState(processState.nvtxTraceProvider);
                            BindThreadEventSourceProvider(processState, processState.nvtxTraceProvider, providerTypeUuid, processState.nvtxTraceProvider.observer.ThreadEventSourceProviderIndex());
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_NvtxTrace(processState.nvtxTraceProvider, processState);
                            return true;
                        }
                    }
                    if constexpr (!std::is_empty_v<ThreadEventProviderState>)
                    {
                        if (providerTypeUuid == NvTrace_ThreadEvent_ProviderTypeUuid)
                        {
                            auto itrProcessState = m_systemState.processes.find(notification.parentLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(itrProcessState == m_systemState.processes.end(), false, 50, "Received ProviderCreated notification for a ThreadEventProvider with an invalid parent Process LUID %" PRIu64, notification.parentLuid);

                            ProcessState& processState = itrProcessState->second;
                            auto [itrProviderState, inserted] = processState.threadEventProviders.try_emplace(notification.providerLuid);
                            NVTRACE_LOG_ERR_RETURN_IF(!inserted, false, 50, "Received more than one ProviderCreated notification for ThreadEventProvider LUID %" PRIu64, notification.providerLuid);

                            ThreadEventProviderState& threadEventProviderState = itrProviderState->second;
                            bool succeeded = threadEventProviderState.Initialize(notification.providerLuid, notification.providerTypeUuid, notification.providerBufferMemoryIdentifier, notification.providerBufferSize);
                            if (!succeeded)
                            {
                                processState.threadEventProviders.erase(itrProviderState);
                            }
                            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "Failed to initialize ThreadEventObserver for Provider with LUID %" PRIu64, notification.providerLuid);

                            threadEventProviderState.processLuid = processState.luid;
                            threadEventProviderState.holdTimeoutTsc = CalculateThreadEventHoldTimeoutTsc(threadEventProviderState.observer);
                            m_systemState.RegisterProviderState(threadEventProviderState);
                            static_cast<TDerivedConsumer*>(this)->OnProviderCreated_ThreadEvent(threadEventProviderState, processState);
                            ConsumeThreadEventSourceProviderRegistrations(processState, threadEventProviderState, ReadCpuTscTimestamp());
                            return true;
                        }
                    }

                    NVTRACE_LOG_ERR(50, "Received ProviderCreated notification for an unknown provider type (providerTypeUuid=%s providerLuid=%llu parentLuid=%llu providerBufferSize=%zu pProviderName=%s)", ToString(notification.providerTypeUuid).c_str(), static_cast<unsigned long long>(notification.providerLuid), static_cast<unsigned long long>(notification.parentLuid), notification.providerBufferSize, notification.pProviderName ? notification.pProviderName : "(null)");
                    return false;
                }

                case NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed:
                {
                    const NvTrace_RouterObserverClient_ProviderDestroyedNotification& notification = *reinterpret_cast<const NvTrace_RouterObserverClient_ProviderDestroyedNotification*>(pNotification);
                    const NVTRACE_UUID providerTypeUuid = ResolveProviderTypeUuidAlias(notification.providerTypeUuid);
                    auto* const pProviderState = m_systemState.GetProviderState(notification.providerLuid);
                    NVTRACE_LOG_ERR_RETURN_IF(!pProviderState, false, 50, "Received ProviderDestroyed notification for an unregistered provider (providerTypeUuid=%s providerLuid=%llu parentLuid=%llu)", ToString(notification.providerTypeUuid).c_str(), static_cast<unsigned long long>(notification.providerLuid), static_cast<unsigned long long>(notification.parentLuid));

                    if (notification.parentLuid == NVTRACE_NIL_LUID)
                    {
                        if constexpr (!std::is_empty_v<CpuClockCalibrationProviderState>)
                        {
                            if (pProviderState == &m_systemState.cpuClockCalibrationProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for CpuClockCalibrationProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_CpuClockCalibration(m_systemState.cpuClockCalibrationProvider);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(m_systemState.cpuClockCalibrationProvider, ProviderDestroyReason::Explicit);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                m_systemState.cpuClockCalibrationProvider.Reset();
                                return true;
                            }
                        }

                        NVTRACE_LOG_ERR(50, "Received ProviderDestroyed notification for a System provider that is not initialized (providerTypeUuid=%s providerLuid=%llu)", ToString(notification.providerTypeUuid).c_str(), static_cast<unsigned long long>(notification.providerLuid));
                        return false;
                    }

                    auto itrGpuPartitionState = m_systemState.gpuPartitions.find(notification.parentLuid);
                    if (itrGpuPartitionState != m_systemState.gpuPartitions.end())
                    {
                        GpuPartitionState& gpuPartitionState = itrGpuPartitionState->second;

                        if constexpr (!std::is_empty_v<GpuClockCalibrationProviderState>)
                        {
                            if (pProviderState == &gpuPartitionState.gpuClockCalibrationProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for GpuClockCalibrationProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuClockCalibrationProvider, ProviderDestroyReason::Explicit);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                gpuPartitionState.gpuClockCalibrationProvider.Reset();
                                return true;
                            }
                        }
                        if constexpr (!std::is_empty_v<GpuPmCountersProviderState>)
                        {
                            if (pProviderState == &gpuPartitionState.gpuPmCountersProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_PmCounters_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for GpuPmCountersProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuPmCountersProvider, ProviderDestroyReason::Explicit);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                gpuPartitionState.gpuPmCountersProvider.Reset();
                                return true;
                            }
                        }
                        if constexpr (!std::is_empty_v<GpuHwTraceProviderState>)
                        {
                            if (pProviderState == &gpuPartitionState.gpuHwTraceProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_HwTrace_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for GpuHwTraceProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuHwTraceProvider, ProviderDestroyReason::Explicit);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                gpuPartitionState.gpuHwTraceProvider.Reset();
                                return true;
                            }
                        }
                        if constexpr (!std::is_empty_v<GpuGrCtxTraceProviderState>)
                        {
                            if (pProviderState == &gpuPartitionState.gpuGrCtxTraceProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_GrCtxTrace_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for GpuGrCtxTraceProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_GpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(gpuPartitionState.gpuGrCtxTraceProvider, ProviderDestroyReason::Explicit);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                gpuPartitionState.gpuGrCtxTraceProvider.Reset();
                                return true;
                            }
                        }

                        NVTRACE_LOG_ERR(50, "Received ProviderDestroyed notification for a GpuPartition provider that is not initialized (providerTypeUuid=%s providerLuid=%llu parentLuid=%llu)", ToString(notification.providerTypeUuid).c_str(), static_cast<unsigned long long>(notification.providerLuid), static_cast<unsigned long long>(notification.parentLuid));
                        return false;
                    }

                    auto itrProcessState = m_systemState.processes.find(notification.parentLuid);
                    if (itrProcessState != m_systemState.processes.end())
                    {
                        ProcessState& processState = itrProcessState->second;

                        if constexpr (!std::is_empty_v<CudaApiTraceProviderState>)
                        {
                            if (pProviderState == &processState.cudaApiTraceProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_CUDA_ApiTrace_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for CudaApiTraceProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_CudaApiTrace(processState.cudaApiTraceProvider, processState);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(processState.cudaApiTraceProvider, ProviderDestroyReason::Explicit);
                                UnbindThreadEventSourceProvider(processState, processState.cudaApiTraceProvider);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                processState.cudaApiTraceProvider.Reset();
                                return true;
                            }
                        }
                        if constexpr (!std::is_empty_v<NvtxTraceProviderState>)
                        {
                            if (pProviderState == &processState.nvtxTraceProvider)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_NvtxTrace_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for NvtxTraceProvider LUID %llu", notification.providerLuid);

                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed_NvtxTrace(processState.nvtxTraceProvider, processState);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(processState.nvtxTraceProvider, ProviderDestroyReason::Explicit);
                                UnbindThreadEventSourceProvider(processState, processState.nvtxTraceProvider);
                                m_systemState.UnregisterProviderState(notification.providerLuid);
                                processState.nvtxTraceProvider.Reset();
                                return true;
                            }
                        }
                        if constexpr (!std::is_empty_v<ThreadEventProviderState>)
                        {
                            const auto itrThreadEventProvider = processState.threadEventProviders.find(notification.providerLuid);
                            if (itrThreadEventProvider != processState.threadEventProviders.end() && pProviderState == &itrThreadEventProvider->second)
                            {
                                NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_ThreadEvent_ProviderTypeUuid, false, 50, "Received ProviderDestroyed notification with mismatched provider type for ThreadEventProvider LUID %" PRIu64, notification.providerLuid);
                                static_cast<TDerivedConsumer*>(this)->OnProviderDestroyed(itrThreadEventProvider->second, ProviderDestroyReason::Explicit);
                                NVTRACE_LOG_WRN(50, "Ignoring ProviderDestroyed notification for ThreadEventProvider. ThreadEventProvider is process-lifetime once initialized.");
                                return true;
                            }
                        }

                        NVTRACE_LOG_ERR(50, "Received ProviderDestroyed notification for a Process provider that is not initialized (providerTypeUuid=%s providerLuid=%llu parentLuid=%llu)", ToString(notification.providerTypeUuid).c_str(), static_cast<unsigned long long>(notification.providerLuid), static_cast<unsigned long long>(notification.parentLuid));
                        return false;
                    }

                    NVTRACE_LOG_ERR(50, "Received ProviderDestroyed notification for an unknown provider type (providerTypeUuid=%s providerLuid=%llu parentLuid=%llu)", ToString(notification.providerTypeUuid).c_str(), static_cast<unsigned long long>(notification.providerLuid), static_cast<unsigned long long>(notification.parentLuid));
                    return false;
                }
            }

            return false;
        }

        void RegisterThreadEventConsumeCallback(const NVTRACE_UUID& providerTypeUuid, ThreadEventConsumeCallback pfnConsumeThreadEvent)
        {
            const size_t providerTypeUuidTableIndex = GetProviderTypeUuidTableIndex(providerTypeUuid);
            if (!pfnConsumeThreadEvent
                || providerTypeUuidTableIndex == InvalidProviderTypeUuidTableIndex
                || IsProviderTypeIgnored(GetProviderType(providerTypeUuid)))
            {
                return;
            }
            m_threadEventConsumeCallbacks[providerTypeUuidTableIndex] = pfnConsumeThreadEvent;
        }

        bool IsProviderTypeIgnored(Codec::ProviderType providerType) const
        {
            const size_t providerTypeIndex = static_cast<size_t>(providerType);
            if (providerTypeIndex >= m_isProviderTypeIgnored.size())
            {
                return false;
            }
            return m_isProviderTypeIgnored[providerTypeIndex];
        }

        void RegisterThreadEventConsumeCallbacks() {}

        void OnThreadEventSourceProviderBound(ThreadEventProviderState&, ProviderState&) {}

        // Drains this ThreadEvent provider's newly published source provider registrations, binding each
        // to its provider if that one is already created. A current-format registration happens-before any
        // record of its index and callers drain registrations before events against the same read endpoint,
        // so such a registration always arrives no later than the first record under its index.
        void ConsumeThreadEventSourceProviderRegistrations(ProcessState& processState, ThreadEventProviderState& threadEventProviderState, uint64_t readEndpointTsc)
        {
            if constexpr (std::is_empty_v<ThreadEventProviderState>)
            {
                return;
            }
            else
            {
                threadEventProviderState.observer.ConsumeSourceProviderRegistrations(
                    readEndpointTsc,
                    [&](uint32_t sourceProviderIndex, const NVTRACE_UUID& providerTypeUuid) {
                        RegisterThreadEventSourceProvider(processState, threadEventProviderState, sourceProviderIndex, providerTypeUuid);
                    });
            }
        }

        // A consumed type's records wait for its provider, up to the hold timeout past the first record held.
        // Records are only read once the Router's initial state is handled, so a provider created before this
        // consumer attached is already bound by then.
        void RegisterThreadEventSourceProvider(ProcessState& processState, ThreadEventProviderState& threadEventProviderState, uint32_t sourceProviderIndex, const NVTRACE_UUID& providerTypeUuid)
        {
            auto* const pSlot = threadEventProviderState.FindSourceProviderSlot(sourceProviderIndex);
            NVTRACE_LOG_ERR_RETURN_IF(!pSlot, , 50, "A source provider registration names index %u, beyond the source provider count %zu of ThreadEvent provider LUID %" PRIu64 ".", sourceProviderIndex, threadEventProviderState.sourceProviderSlots.size(), threadEventProviderState.luid);

            // Registrations are memory the profiled process controls, and a slot's type picks both the
            // callback and the provider state its records reach, so a registered slot keeps its type.
            NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid == NVTRACE_NIL_UUID, , 50, "Ignoring a registration of source provider index %u with no provider type (ThreadEvent provider LUID %" PRIu64 ").", sourceProviderIndex, threadEventProviderState.luid);
            NVTRACE_LOG_ERR_RETURN_IF(pSlot->providerTypeUuid != NVTRACE_NIL_UUID && pSlot->providerTypeUuid != providerTypeUuid, , 50, "Ignoring a registration of source provider index %u for provider type %s, already registered for %s (ThreadEvent provider LUID %" PRIu64 ").", sourceProviderIndex, ToString(providerTypeUuid).c_str(), ToString(pSlot->providerTypeUuid).c_str(), threadEventProviderState.luid);

            const bool isThreadEventType = providerTypeUuid == NvTrace_ThreadEvent_ProviderTypeUuid;
            const bool isThreadEventIndex = sourceProviderIndex == NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent;
            NVTRACE_LOG_ERR_RETURN_IF(isThreadEventType != isThreadEventIndex, , 50, "Ignoring a registration of source provider index %u for provider type %s: index %u belongs to the ThreadEvent provider type alone (ThreadEvent provider LUID %" PRIu64 ").", sourceProviderIndex, ToString(providerTypeUuid).c_str(), static_cast<uint32_t>(NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent), threadEventProviderState.luid);

            if (pSlot->providerTypeUuid == NVTRACE_NIL_UUID)
            {
                pSlot->providerTypeUuid = providerTypeUuid;

                // A type outside the provider type table has no callback.
                const size_t providerTypeUuidTableIndex = GetProviderTypeUuidTableIndex(providerTypeUuid);
                const ThreadEventConsumeCallback pfnConsumeThreadEvent = providerTypeUuidTableIndex == InvalidProviderTypeUuidTableIndex ? nullptr : m_threadEventConsumeCallbacks[providerTypeUuidTableIndex];
                if (!pfnConsumeThreadEvent)
                {
                    pSlot->pfnConsumeThreadEvent = &DropThreadEvent;
                }
                else
                {
                    pSlot->pfnConsumeThreadEvent = pfnConsumeThreadEvent;
                    pSlot->isWaitingForProvider = true;
                }
            }

            if (ProviderState* const pProviderState = FindThreadEventSourceProvider(processState, threadEventProviderState, providerTypeUuid, sourceProviderIndex))
            {
                BindThreadEventSourceProviderSlot(threadEventProviderState, *pSlot, *pProviderState, sourceProviderIndex);
            }
        }

        ProviderState* FindThreadEventSourceProvider(ProcessState& processState, ThreadEventProviderState& threadEventProviderState, const NVTRACE_UUID& providerTypeUuid, uint32_t sourceProviderIndex)
        {
            if (providerTypeUuid == NvTrace_ThreadEvent_ProviderTypeUuid)
            {
                return sourceProviderIndex == NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent ? &threadEventProviderState : nullptr;
            }

            if constexpr (!std::is_empty_v<CudaApiTraceProviderState>)
            {
                CudaApiTraceProviderState& cudaApiTraceProvider = processState.cudaApiTraceProvider;
                if (providerTypeUuid == NvTrace_CUDA_ApiTrace_ProviderTypeUuid && cudaApiTraceProvider.IsInitialized() && cudaApiTraceProvider.observer.ThreadEventSourceProviderIndex() == sourceProviderIndex)
                {
                    return &cudaApiTraceProvider;
                }
            }

            if constexpr (!std::is_empty_v<NvtxTraceProviderState>)
            {
                NvtxTraceProviderState& nvtxTraceProvider = processState.nvtxTraceProvider;
                if (providerTypeUuid == NvTrace_NvtxTrace_ProviderTypeUuid && nvtxTraceProvider.IsInitialized() && nvtxTraceProvider.observer.ThreadEventSourceProviderIndex() == sourceProviderIndex)
                {
                    return &nvtxTraceProvider;
                }
            }

            return nullptr;
        }

        // Binds a newly created provider in the ThreadEvent provider of its process that registered the index
        // its buffer header names. The registration is published before the provider is advertised, so
        // draining the registrations first binds it here even when no consume pass ran since. Runs before the
        // consumer's ProviderCreated hook, so the provider's attach already sees its binding.
        void BindThreadEventSourceProvider(ProcessState& processState, ProviderState& providerState, const NVTRACE_UUID& providerTypeUuid, uint32_t sourceProviderIndex)
        {
            // Index 0 is the transport's own; a source provider naming it predates registration, so it
            // rides no transport this consumer reads.
            if (sourceProviderIndex == NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent)
            {
                return;
            }

            if constexpr (!std::is_empty_v<ThreadEventProviderState>)
            {
                for (auto& [threadEventProviderLuid, threadEventProviderState] : processState.threadEventProviders)
                {
                    ConsumeThreadEventSourceProviderRegistrations(processState, threadEventProviderState, ReadCpuTscTimestamp());

                    auto* const pSlot = threadEventProviderState.FindSourceProviderSlot(sourceProviderIndex);
                    if (!pSlot || pSlot->providerTypeUuid == NVTRACE_NIL_UUID)
                    {
                        continue;
                    }

                    if (pSlot->providerTypeUuid != providerTypeUuid)
                    {
                        NVTRACE_LOG_ERR(50, "Provider LUID %" PRIu64 " names source provider index %u, which ThreadEvent provider LUID %" PRIu64 " registered for provider type %s, not %s.", providerState.luid, sourceProviderIndex, threadEventProviderLuid, ToString(pSlot->providerTypeUuid).c_str(), ToString(providerTypeUuid).c_str());
                        continue;
                    }

                    BindThreadEventSourceProviderSlot(threadEventProviderState, *pSlot, providerState, sourceProviderIndex);
                }
            }
        }

        // Both bind points reach the same slot when the registration drain in
        // BindThreadEventSourceProvider binds the provider itself, so a repeated bind must not notify twice.
        template <typename TSourceProviderSlot>
        void BindThreadEventSourceProviderSlot(ThreadEventProviderState& threadEventProviderState, TSourceProviderSlot& slot, ProviderState& providerState, uint32_t sourceProviderIndex)
        {
            if (slot.pProviderState == &providerState)
            {
                return;
            }

            slot.pProviderState = &providerState;

            providerState.threadEventProviderLuid = threadEventProviderState.luid;
            providerState.threadEventSourceProviderIndex = sourceProviderIndex;
            static_cast<TDerivedConsumer*>(this)->OnThreadEventSourceProviderBound(threadEventProviderState, providerState);
        }

        // A destroyed provider's leftover records are dropped rather than held: none of them can bind
        // again, so holding would stall their threads for the full timeout. Leftovers include records the
        // destroy's final drain did not reach because another index's held record stopped their thread.
        void UnbindThreadEventSourceProvider(ProcessState& processState, ProviderState& providerState)
        {
            if constexpr (!std::is_empty_v<ThreadEventProviderState>)
            {
                for (auto& [_, threadEventProviderState] : processState.threadEventProviders)
                {
                    auto* const pSlot = threadEventProviderState.FindSourceProviderSlot(providerState.threadEventSourceProviderIndex);
                    if (pSlot && pSlot->pProviderState == &providerState)
                    {
                        pSlot->pProviderState = nullptr;
                        pSlot->isWaitingForProvider = false;
                    }
                }
            }

            providerState.threadEventSourceProviderIndex = ProviderState::kInvalidThreadEventSourceProviderIndex;
            providerState.threadEventProviderLuid = NVTRACE_NIL_LUID;
        }

        // Routes one record by its source provider index:
        // - Bound: the record goes to its provider.
        // - Registered, provider not created yet: the record is held, which stops reading its thread until the next pass.
        // - Registered, hold deadline passed: the index's records are dropped until it binds.
        // - Otherwise: the record is dropped. A provider registers before its first record; only a V1 provider may not.
        bool ConsumeThreadEvent(ProcessState& processState, ThreadEventProviderState& threadEventProviderState, uint32_t sourceProviderIndex, const ThreadEventRecord& record, uint64_t readEndpointTsc)
        {
            auto* const pSlot = threadEventProviderState.FindSourceProviderSlot(sourceProviderIndex);
            if (!pSlot)
            {
                const bool isOutOfRangeDropLogged = std::exchange(threadEventProviderState.isOutOfRangeDropLogged, true);
                NVTRACE_LOG_ERR_IF(!isOutOfRangeDropLogged, 50, "Dropping the ThreadEvent records with source provider indices beyond the source provider count %zu of ThreadEvent provider LUID %" PRIu64 ", starting with index %u.", threadEventProviderState.sourceProviderSlots.size(), threadEventProviderState.luid, sourceProviderIndex);
                return true;
            }

            if (pSlot->pProviderState)
            {
                return pSlot->pfnConsumeThreadEvent(static_cast<TDerivedConsumer*>(this), &processState, pSlot->pProviderState, record);
            }

            if (pSlot->pfnConsumeThreadEvent == &DropThreadEvent)
            {
                return true;
            }

            if (pSlot->providerTypeUuid == NVTRACE_NIL_UUID && !std::exchange(pSlot->isUnregisteredDropLogged, true))
            {
                NVTRACE_LOG_WRN(50, "Dropping the records of source provider index %u of ThreadEvent provider LUID %" PRIu64 ": no registration of the index precedes them.", sourceProviderIndex, threadEventProviderState.luid);
            }

            return HoldUnboundThreadEvent(threadEventProviderState, *pSlot, sourceProviderIndex, readEndpointTsc);
        }

        template <typename TSourceProviderSlot>
        bool HoldUnboundThreadEvent(const ThreadEventProviderState& threadEventProviderState, TSourceProviderSlot& slot, uint32_t sourceProviderIndex, uint64_t readEndpointTsc)
        {
            if (slot.isWaitingForProvider && m_isHoldingUnboundThreadEvents)
            {
                if (slot.holdDeadlineTsc == 0)
                {
                    // Saturates: a deadline past UINT64_MAX holds indefinitely rather than wrapping into the past.
                    const uint64_t holdTimeoutTsc = threadEventProviderState.holdTimeoutTsc;
                    slot.holdDeadlineTsc = (UINT64_MAX - readEndpointTsc) < holdTimeoutTsc ? UINT64_MAX : (readEndpointTsc + holdTimeoutTsc);
                }

                if (readEndpointTsc < slot.holdDeadlineTsc)
                {
                    return false;
                }

                // Cleared so the expiry is logged once.
                slot.isWaitingForProvider = false;
                NVTRACE_LOG_WRN(50, "Dropping the records of source provider index %u of ThreadEvent provider LUID %" PRIu64 ": its provider was not created within the hold timeout.", sourceProviderIndex, threadEventProviderState.luid);
            }

            m_droppedUnboundThreadEventCount += 1;
            return true;
        }

        // While the returned guard lives, records of providers not created yet are dropped instead of held. Guards nest.
        // Used by passes that must not stall: the warm-up before the observation start, and the final drains.
        [[nodiscard]]
        auto ScopeGuardDropUnboundThreadEvents()
        {
            const bool wasHoldingUnboundThreadEvents = std::exchange(m_isHoldingUnboundThreadEvents, false);
            return ScopeExitGuard([this, wasHoldingUnboundThreadEvents]() { m_isHoldingUnboundThreadEvents = wasHoldingUnboundThreadEvents; });
        }

        // An unbound record cannot be judged yet and counts as one, so ConsumeThreadEvent decides its fate.
        static bool IsObjectLifetimeThreadEvent(const ThreadEventProviderState& threadEventProviderState, uint32_t sourceProviderIndex, const ThreadEventRecord& record)
        {
            const auto* const pSlot = threadEventProviderState.FindSourceProviderSlot(sourceProviderIndex);
            if (!pSlot)
            {
                return false;
            }

            if (!pSlot->pProviderState)
            {
                return true;
            }

            if constexpr (!std::is_empty_v<CudaApiTraceProviderState>)
            {
                if (pSlot->providerTypeUuid == NvTrace_CUDA_ApiTrace_ProviderTypeUuid)
                {
                    return static_cast<const CudaApiTraceProviderState*>(pSlot->pProviderState)->observer.IsObjectLifetimeThreadEventRecord(record.recordType);
                }
            }

            if constexpr (!std::is_empty_v<NvtxTraceProviderState>)
            {
                if (pSlot->providerTypeUuid == NvTrace_NvtxTrace_ProviderTypeUuid)
                {
                    return static_cast<const NvtxTraceProviderState*>(pSlot->pProviderState)->observer.IsObjectLifetimeThreadEventRecord(record.recordType);
                }
            }

            return false;
        }

        uint64_t DroppedUnboundThreadEventCount() const
        {
            return m_droppedUnboundThreadEventCount;
        }

        uint64_t CalculateThreadEventHoldTimeoutTsc(const ThreadEventObserver& observer) const
        {
            const uint64_t eventBufferDurationNsec = observer.GetEventBufferDurationNsec();
            const uint64_t defaultHoldTimeoutNsec = eventBufferDurationNsec == 0 ? MaxDefaultThreadEventHoldTimeoutNsec : std::min(MaxDefaultThreadEventHoldTimeoutNsec, eventBufferDurationNsec / 4);
            const uint64_t holdTimeoutNsec = m_threadEventHoldTimeoutNsec != 0 ? m_threadEventHoldTimeoutNsec : defaultHoldTimeoutNsec;
            const double holdTimeoutTsc = holdTimeoutNsec * GetLatestCpuClockCalibrationPoint().utcToTscFactor;
            return holdTimeoutTsc < static_cast<double>(UINT64_MAX) ? static_cast<uint64_t>(holdTimeoutTsc) : UINT64_MAX;
        }

        // Derived consumer classes that define their own ConsumeData() and use
        // ConvertTscTickToUtcNsecForConsume() must call UpdateConsumeCpuClockCalibration()
        // once before converting timestamps and before calling this implementation.
        void ConsumeData(uint64_t readEndpointTsc = ReadCpuTscTimestamp())
        {
            // Consume data from System providers
            {
                if constexpr (!std::is_empty_v<CpuClockCalibrationProviderState>)
                {
                    const ReadEndpoints readEndpoints = {readEndpointTsc, 0};
                    if (m_systemState.cpuClockCalibrationProvider.IsInitialized())
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_CpuClockCalibration(m_systemState.cpuClockCalibrationProvider, readEndpoints);
                    }
                }
            }

            // Consume data from GpuPartition providers
            for (auto& [_, gpuPartitionState] : m_systemState.gpuPartitions)
            {
                // Resolve both timestamp domains once per GPU. Generic observers select the endpoint matching each
                // circular-buffer section's published timestamp type.
                [[maybe_unused]]
                const ReadEndpoints readEndpoints = {readEndpointTsc, ResolveGpuPtimerReadEndpoint(gpuPartitionState, readEndpointTsc)};

                if constexpr (!std::is_empty_v<GpuClockCalibrationProviderState>)
                {
                    if (gpuPartitionState.gpuClockCalibrationProvider.IsInitialized())
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_GpuClockCalibration(gpuPartitionState.gpuClockCalibrationProvider, gpuPartitionState, readEndpoints);
                    }
                }
                if constexpr (!std::is_empty_v<GpuPmCountersProviderState>)
                {
                    if (gpuPartitionState.gpuPmCountersProvider.IsInitialized())
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_GpuPmCounters(gpuPartitionState.gpuPmCountersProvider, gpuPartitionState, readEndpoints);
                    }
                }
                if constexpr (!std::is_empty_v<GpuHwTraceProviderState>)
                {
                    if (gpuPartitionState.gpuHwTraceProvider.IsInitialized())
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_GpuHwTrace(gpuPartitionState.gpuHwTraceProvider, gpuPartitionState, readEndpoints);
                    }
                }
                if constexpr (!std::is_empty_v<GpuGrCtxTraceProviderState>)
                {
                    if (gpuPartitionState.gpuGrCtxTraceProvider.IsInitialized())
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_GpuGrCtxTrace(gpuPartitionState.gpuGrCtxTraceProvider, gpuPartitionState, readEndpoints);
                    }
                }
            }

            // Consume data from Process providers
            for (auto& [_, processState] : m_systemState.processes)
            {
                if constexpr (!std::is_empty_v<CudaApiTraceProviderState>)
                {
                    if (processState.cudaApiTraceProvider.IsInitialized())
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_CudaApiTrace(processState.cudaApiTraceProvider, processState, readEndpointTsc);
                    }
                }
                if constexpr (!std::is_empty_v<ThreadEventProviderState>)
                {
                    for (auto& threadEventProviderEntry : processState.threadEventProviders)
                    {
                        static_cast<TDerivedConsumer*>(this)->ConsumeData_ThreadEvent(threadEventProviderEntry.second, processState, readEndpointTsc);
                    }
                }
            }
        }

        bool ScheduleImmediateConsumeIteration()
        {
            constexpr bool isRelativeTime = true;
            bool succeeded = m_consumeTimer.Arm(0, isRelativeTime);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "DeadlineTimer::Arm failed");

            return true;
        }

        bool ScheduleNextConsumeIteration()
        {
            constexpr bool isRelativeTime = true;
            bool succeeded = m_consumeTimer.Arm(m_consumeIntervalNsec, isRelativeTime);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "DeadlineTimer::Arm failed");

            return true;
        }

    protected:
        void UpdateConsumeCpuClockCalibration()
        {
            const TscRtcCalibrationPoint calibrationPoint = GetLatestCpuClockCalibrationPoint();
            if (m_consumeCpuClockCalibrationPoint.tscToUtcFactor == 0 || calibrationPoint.tscTick > m_consumeCpuClockCalibrationPoint.tscTick)
            {
                m_consumeCpuClockCalibrationPoint = calibrationPoint;
            }
        }

        uint64_t ConvertTscTickToUtcNsecForConsume(uint64_t timestampTsc)
        {
            if (m_consumeCpuClockCalibrationPoint.tscToUtcFactor == 0)
            {
                UpdateConsumeCpuClockCalibration();
            }
            return m_consumeCpuClockCalibrationPoint.ToUtc(timestampTsc);
        }

        TscRtcCalibrationPoint GetLatestCpuClockCalibrationPoint() const
        {
            if constexpr (!std::is_empty_v<CpuClockCalibrationProviderState>)
            {
                TscRtcCalibrationPoint calibrationPoint;
                if (m_systemState.cpuClockCalibrationProvider.IsInitialized() && m_systemState.cpuClockCalibrationProvider.observer.GetLatestCalibrationPoint(calibrationPoint))
                {
                    return calibrationPoint;
                }
            }
            return m_systemState.initialTscRtcCalibrationPoint;
        }

        // Maps a CPU-TSC read endpoint into a GPU's ptimer domain from its latest clock calibration, so any ptimer-gated
        // circular buffer of that GPU can gate on it. With no calibration yet we cannot convert "now" into the ptimer
        // domain, so we return UINT64_MAX to drain everything buffered rather than stall the pipeline until the first
        // calibration point lands.
        static uint64_t ResolveGpuPtimerReadEndpoint(const GpuPartitionState& parentState, uint64_t readEndpointTsc)
        {
            // When GPU clock calibration is compiled out its state is an empty struct; skip the lookup.
            if constexpr (!std::is_empty_v<GpuClockCalibrationProviderState>)
            {
                TscGpuCalibrationPoint calibrationPoint;
                if (parentState.gpuClockCalibrationProvider.IsInitialized() && parentState.gpuClockCalibrationProvider.observer.GetLatestCalibrationPoint(calibrationPoint))
                {
                    return calibrationPoint.ToGpu(readEndpointTsc);
                }
            }
            return UINT64_MAX;
        }

        void OnSystemCreated(SystemState& systemState) {}
        void OnGpuPartitionCreated(GpuPartitionState& gpuPartitionState) {}
        void OnProcessCreated(ProcessState& processState) {}
        void OnProviderCreated_CpuClockCalibration(CpuClockCalibrationProviderState& providerState) {}
        void OnProviderCreated_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderCreated_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderCreated_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderCreated_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderCreated_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState) {}
        void OnProviderCreated_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState) {}
        void OnProviderCreated_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState) {}

        void OnGpuPartitionDestroyed(GpuPartitionState& gpuPartitionState, uint64_t destructionTimestampTsc) {}
        void OnProcessDestroyed(ProcessState& processState, uint64_t destructionTimestampTsc) {}
        void OnProviderDestroyed(ProviderState& providerState, ProviderDestroyReason reason) {}
        void OnProviderDestroyed_CpuClockCalibration(CpuClockCalibrationProviderState& providerState) {}
        void OnProviderDestroyed_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderDestroyed_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderDestroyed_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderDestroyed_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState) {}
        void OnProviderDestroyed_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState) {}
        void OnProviderDestroyed_NvtxTrace(NvtxTraceProviderState& providerState, ProcessState& parentState) {}
        void OnProviderDestroyed_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState) {}

        void ConsumeData_CpuClockCalibration(CpuClockCalibrationProviderState& providerState, const ReadEndpoints& readEndpoints) {}
        void ConsumeData_GpuClockCalibration(GpuClockCalibrationProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints) {}
        void ConsumeData_GpuPmCounters(GpuPmCountersProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints) {}
        void ConsumeData_GpuHwTrace(GpuHwTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints) {}
        void ConsumeData_GpuGrCtxTrace(GpuGrCtxTraceProviderState& providerState, GpuPartitionState& parentState, const ReadEndpoints& readEndpoints) {}
        void ConsumeData_CudaApiTrace(CudaApiTraceProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc) {}
        void ConsumeData_ThreadEvent(ThreadEventProviderState& providerState, ProcessState& parentState, uint64_t readEndpointTsc) {}

    protected:
        UniquePtr<RouterObserverClient> m_pRouterObserverClient;

        DeadlineTimer m_consumeTimer;
        uint64_t m_consumeIntervalNsec = 0;
        uint64_t m_threadEventHoldTimeoutNsec = 0;
        bool m_isHoldingUnboundThreadEvents = true;
        uint64_t m_droppedUnboundThreadEventCount = 0;

        SystemState m_systemState;
        // Accessed only by the consume thread and refreshed at the beginning of each consume cycle.
        TscRtcCalibrationPoint m_consumeCpuClockCalibrationPoint = {};
        Codec::ProviderTypeTable<bool> m_isProviderTypeIgnored = {};
        ProviderTypeUuidTable<ThreadEventConsumeCallback> m_threadEventConsumeCallbacks = {};
    };

} // namespace nv::trace
