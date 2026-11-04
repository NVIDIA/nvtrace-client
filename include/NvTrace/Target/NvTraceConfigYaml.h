/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace nv::trace {

    inline const char* GetDefaultNvTraceYamlConfig()
    {
        const char* const pNvTraceYamlConfig = R"NVTRACEYAML(
Router:
    maxProviderClients: 32                    # Maximum number of provider clients at any given time.
    maxObserverClients: 32                    # Maximum number of observer clients at any given time.
    maxControllerClients: 32                  # Maximum number of controller clients at any given time.
    maxObserverClientPendingExclusiveNotifications: 32  # troubleshooting option.
    maxGpuPartitions: 32                      # Maximum number of GPU devices/partitions at any given time.
    maxProcesses: 32                          # Maximum number of processes at any given time.
    maxProviderTypes: 32                      # Maximum number of provider types.
    maxProviders: 64                          # Maximum number of providers at any given time.
    securityPolicy:
        # Default policy is world-open, matching NVIDIA Nsight Perf's historical behavior. Lock the Router down by
        # restricting these lists to specific users/groups. Root/admin is always granted full access implicitly.
        authorizedProviderClients:            # Who may register providers ('/everyone' => open to all).
            - principal: /everyone
        authorizedObserverClients:            # Who may observe, and which process/GPU providers they may see.
            - principal: /everyone
              processes: all
              gpus: all
        authorizedControllerClients:          # Who may acquire GPU profiling locks, and on which GPUs.
            - principal: /everyone
              gpus: all

GpuPmCounters:
    # The metric sets the service collects. architectures[] below covers chips no set here claims.
    metricConfig:
        # Specify full metric names to evaluate live on the provider. 
        # Metrics that are collected but not specified here will be evaluated offline.
        metricsToEvaluateOnline: []
        # Single-quoted: an unquoted leading '{' would start a flow mapping, and a double-quoted scalar would
        # read a Windows path's backslashes as escapes.
        file: '{ServiceBinaryDirectory}/ReducedOverview.config'
    # To collect General Metrics instead, replace metricConfig above with the block below, which covers the
    # chips these documents are published for. A document names no chips of its own, so each entry says
    # which chips it is for. A chip no entry names falls through to architectures[] below.
    # metricConfigs:
    #     -   chipRegex: 'GB10[0-9]'
    #         metricsToEvaluateOnline: []
    #         file: '{ServiceBinaryDirectory}/gb10x.config'
    #     -   chipRegex: 'GB11[0-9]'
    #         metricsToEvaluateOnline: []
    #         file: '{ServiceBinaryDirectory}/gb11x.config'
    #     -   chipRegex: 'GB20[0-9]'
    #         metricsToEvaluateOnline: []
    #         file: '{ServiceBinaryDirectory}/gb20x.config'
    #     -   chipRegex: 'GB10[BC]'
    #         metricsToEvaluateOnline: []
    #         file: '{ServiceBinaryDirectory}/gb10y.config'
    #     -   chipRegex: 'GB20[BCF]'
    #         metricsToEvaluateOnline: []
    #         file: '{ServiceBinaryDirectory}/gb20y.config'
    #     -   chipRegex: 'GR10[0-9]'
    #         metricsToEvaluateOnline: []
    #         file: '{ServiceBinaryDirectory}/gr10x.config'
    pmSampleRateHz: 1000                      # The rate at which GPU counters are sampled, in Hertz.
    updateIntervalNsec: 100000000             # The interval at which the CPU copies out GPU counter values. A longer update interval requires a larger hardware buffer.
    readbackHoldoffNsec: 10000000             # Additional latency before GPU counters are read by the CPU. This value also increases the hardware buffer size.
    keepLatest: true                          # When true, puts GPU side buffers in flight recorder mode. When the buffer is full, it will overwrite the oldest record.  An R580 driver or newer is recommended.
    maxHistoryDurationMsec: 0                 # Max sample history (ms) per-GPU buffer. 0 = built-in default (402 ms). Raise (e.g. 1300) for once-per-second consumers.
    architectures:
        -   chipRegex: GB10x                    # includes GB100, GB102, GB110
            # The list of GPU metrics that the hardware will measure. GPU metric names are defined by the NVIDIA Nsight Perf SDK. Metrics specified here will be available for offline evaluation, independent of whether they are evaluated in real time (via metricsToEvaluate).
            # Each entry is either a metric name string, or a map { metricName, keepInstances } where keepInstances (default false) retains per-instance counter values. For example:
            #   - metricName: sm__inst_executed.sum
            #     keepInstances: true
            metricsToConfigure:
                - gpc__cycles_elapsed.avg.per_second                                            # GPC Clock Frequency (Hz)
                - sys__cycles_elapsed.avg.per_second                                            # SYS Clock Frequency (Hz)
                - gr__cycles_active.avg.pct_of_peak_sustained_elapsed                           # GR Active %
                - sm__cycles_active.avg.pct_of_peak_sustained_elapsed                           # SMs Active
                - sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed                  # SM Issue %
                - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed      # Tensor Active %
                - dram__bytes_read.avg.pct_of_peak_sustained_elapsed                            # DRAM Read Bandwidth %
                - dram__bytes_write.avg.pct_of_peak_sustained_elapsed                           # DRAM Write Bandwidth %
                - TPC.TriageCompute.tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed  # SM Compute Warp Occupancy %
                - PCI.TriageCompute.pcie__read_bytes.avg.pct_of_peak_sustained_elapsed                      # PCIe RX Bandwidth %
                - PCI.TriageCompute.pcie__write_bytes.avg.pct_of_peak_sustained_elapsed                     # PCIe TX Bandwidth %
              # - gpu__power.avg                                                                            # Total GPU Power (mW). Opt-in because power collection costs additional memory and background CPU cycles.
              # - NVLRX.TriageCompute.nvlrx__bytes.avg.pct_of_peak_sustained_elapsed                        # NVLINK RX Bandwidth %. Only works on a multi-GPU GB10x setup.
              # - NVLTX.TriageCompute.nvltx__bytes.avg.pct_of_peak_sustained_elapsed                        # NVLINK TX Bandwidth %. Only works on a multi-GPU GB10x setup.
              # - CTC_B.TriageCompute.ctc__rx_bytes.avg.pct_of_peak_sustained_elapsed           # CTC (CPU/GPU) RX Bandwidth %
              # - CTC_A.TriageCompute.ctc__tx_bytes.avg.pct_of_peak_sustained_elapsed           # CTC (CPU/GPU) TX Bandwidth %
            # The list of metrics to evaluate in real time, providing the convenience of real time perf stats. Specifying a subset of the metricsToConfigure is recommended. The longer this list, the more CPU overhead is incurred in real time.
            # Real-time evaluation is optional. Metrics in metricsToConfigure but not in metricsToEvaluate will be evaluated offline during export.
            metricsToEvaluate:
              # - gpc__cycles_elapsed.avg.per_second                                            # GPC Clock Frequency (Hz)
              # - sys__cycles_elapsed.avg.per_second                                            # SYS Clock Frequency (Hz)
              # - gr__cycles_active.avg.pct_of_peak_sustained_elapsed                           # GR Active %
              # - sm__cycles_active.avg.pct_of_peak_sustained_elapsed                           # SMs Active
              # - sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed                  # SM Issue %
              # - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed      # Tensor Active %
              # - dram__bytes_read.avg.pct_of_peak_sustained_elapsed                            # DRAM Read Bandwidth %
              # - dram__bytes_write.avg.pct_of_peak_sustained_elapsed                           # DRAM Write Bandwidth %
              # - TPC.TriageCompute.tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed  # SM Compute Warp Occupancy %
              # - PCI.TriageCompute.pcie__read_bytes.avg.pct_of_peak_sustained_elapsed                      # PCIe RX Bandwidth %
              # - PCI.TriageCompute.pcie__write_bytes.avg.pct_of_peak_sustained_elapsed                     # PCIe TX Bandwidth %
              # - gpu__power.avg                                                                            # Total GPU Power (mW). Opt-in because power collection costs additional memory and background CPU cycles.
              # - NVLRX.TriageCompute.nvlrx__bytes.avg.pct_of_peak_sustained_elapsed                        # NVLINK RX Bandwidth %. Only works on a multi-GPU GB10x setup.
              # - NVLTX.TriageCompute.nvltx__bytes.avg.pct_of_peak_sustained_elapsed                        # NVLINK TX Bandwidth %. Only works on a multi-GPU GB10x setup.
              # - CTC_B.TriageCompute.ctc__rx_bytes.avg.pct_of_peak_sustained_elapsed           # CTC (CPU/GPU) RX Bandwidth %
              # - CTC_A.TriageCompute.ctc__tx_bytes.avg.pct_of_peak_sustained_elapsed           # CTC (CPU/GPU) TX Bandwidth %
        -   chipRegex: GB20x                    # includes GB202, GB203, GB205, GB206, GB207
            # The list of GPU metrics that the hardware will measure. GPU metric names are defined by the NVIDIA Nsight Perf SDK. Metrics specified here will be available for offline evaluation, independent of whether they are evaluated in real time (via metricsToEvaluate).
            metricsToConfigure:
                - gpc__cycles_elapsed.avg.per_second                                         # GPC Clock Frequency (Hz)
                - sys__cycles_elapsed.avg.per_second                                         # SYS Clock Frequency (Hz)
                - gr__cycles_active.avg.pct_of_peak_sustained_elapsed                        # GR Active %
                - sm__cycles_active.avg.pct_of_peak_sustained_elapsed                        # SMs Active
                - sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed               # SM Issue %
                - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed   # Tensor Active %
                - tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed     # SM Compute Warp Occupancy %
                - dram__bytes_op_read.avg.pct_of_peak_sustained_elapsed                      # DRAM Read Bandwidth %
                - dram__bytes_op_write.avg.pct_of_peak_sustained_elapsed                     # DRAM Write Bandwidth %
                - pcie__read_bytes.avg.pct_of_peak_sustained_elapsed                         # PCIe RX Bandwidth %
                - pcie__write_bytes.avg.pct_of_peak_sustained_elapsed                        # PCIe TX Bandwidth %
            # The list of metrics to evaluate in real time, providing the convenience of real time perf stats. Specifying a subset of the metricsToConfigure is recommended. The longer this list, the more CPU overhead is incurred in real time.
            # Real-time evaluation is optional. Metrics in metricsToConfigure but not in metricsToEvaluate will be evaluated offline during export.
            metricsToEvaluate:
              # - gpc__cycles_elapsed.avg.per_second                                         # GPC Clock Frequency (Hz)
              # - sys__cycles_elapsed.avg.per_second                                         # SYS Clock Frequency (Hz)
              # - gr__cycles_active.avg.pct_of_peak_sustained_elapsed                        # GR Active %
              # - sm__cycles_active.avg.pct_of_peak_sustained_elapsed                        # SMs Active
              # - sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed               # SM Issue %
              # - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed   # Tensor Active %
              # - tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed     # SM Compute Warp Occupancy %
              # - dram__bytes_op_read.avg.pct_of_peak_sustained_elapsed                      # DRAM Read Bandwidth %
              # - dram__bytes_op_write.avg.pct_of_peak_sustained_elapsed                     # DRAM Write Bandwidth %
              # - pcie__read_bytes.avg.pct_of_peak_sustained_elapsed                         # PCIe RX Bandwidth %
              # - pcie__write_bytes.avg.pct_of_peak_sustained_elapsed                        # PCIe TX Bandwidth %
        -   chipRegex: GB20B
            # The list of GPU metrics that the hardware will measure. GPU metric names are defined by the NVIDIA Nsight Perf SDK. Metrics specified here will be available for offline evaluation, independent of whether they are evaluated in real time (via metricsToEvaluate).
            metricsToConfigure:
                - sm__inst_executed.sum
                - sm__inst_executed.avg.pct_of_peak_sustained_elapsed
                - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
            # The list of metrics to evaluate in real time, providing the convenience of real time perf stats. Specifying a subset of the metricsToConfigure is recommended. The longer this list, the more CPU overhead is incurred in real time.
            # Real-time evaluation is optional. Metrics in metricsToConfigure but not in metricsToEvaluate will be evaluated offline during export.
            metricsToEvaluate:
              # - sm__inst_executed.sum
              # - sm__inst_executed.avg.pct_of_peak_sustained_elapsed
              # - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
        -   chipRegex: .*                           # all other GPUs
            # The list of GPU metrics that the hardware will measure. GPU metric names are defined by the NVIDIA Nsight Perf SDK. Metrics specified here will be available for offline evaluation, independent of whether they are evaluated in real time (via metricsToEvaluate).
            metricsToConfigure:
                - sm__inst_executed.sum
                - sm__inst_executed.avg.pct_of_peak_sustained_elapsed
                - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
                - dram__bytes.avg.pct_of_peak_sustained_elapsed
                - dram__bytes.sum
                - dram__bytes.sum.per_second
            # The list of metrics to evaluate in real time, providing the convenience of real time perf stats. Specifying a subset of the metricsToConfigure is recommended. The longer this list, the more CPU overhead is incurred in real time.
            # Real-time evaluation is optional. Metrics in metricsToConfigure but not in metricsToEvaluate will be evaluated offline during export.
            metricsToEvaluate:
              # - sm__inst_executed.sum
              # - sm__inst_executed.avg.pct_of_peak_sustained_elapsed
              # - sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
              # - dram__bytes.avg.pct_of_peak_sustained_elapsed
              # - dram__bytes.sum
              # - dram__bytes.sum.per_second
)NVTRACEYAML"
                                               R"NVTRACEYAML(
GpuClockCalibration:
    circularBufferCellCount: 128
    updateIntervalNsec: 1000000000    # 1 second

GpuHwTrace:
    circularBufferCellCount: 524288       # 512 * 1024
    updateIntervalNsec: 100000000         # 100ms
    readbackHoldoffNsec: 10000000         # 10ms
    showInternalEvents: true
    keepLatest: true                      # When true, the PerfWorks record buffer overwrites the oldest records when full (flight-recorder mode).
    # recordBufferSize: 4194304          # 4 MB; PerfWorks record buffer size in bytes.
    # allowLossyProvider: false          # When true, PerfWorks may drop HW trace records when the buffer is full.

GpuGrCtxTrace:
    circularBufferCellCount: 6144
    updateIntervalNsec: 100000000

CudaApiTrace:
    maxContextCount: 128                      # Maximum number of traceable CUcontext objects at any given time.
    maxStreamCount: 2048                      # Maximum number of traceable CUstream objects at any given time.
    maxModuleCount: 4096                      # Maximum number of traceable CUmodule objects at any given time. If kernel names are missing or "?Unknown?" for kernel launch events, try increasing this value.
    maxFunctionCount: 32768                   # Maximum number of traceable CUfunction objects at any given time. If kernel names are missing or "?Unknown?" for kernel launch events, try increasing this value.
    maxGraphExecCount: 16384                  # Maximum number of traceable CUgraphExec objects at any given time. If kernel launch events are missing from your trace, and you see cudaGraphLaunch or cuGraphLaunch calls on the CPU, try increasing this value.
    maxGraphNodeCount: 1048576                # Maximum number of traceable CUgraphNode objects at any given time. If kernel launch events are missing from your trace, and you see cudaGraphLaunch or cuGraphLaunch calls on the CPU, try increasing this value.
    maxStringTableEntries: 262144             # Maximum number of string table entries at any given time. If kernel names are missing or "?Unknown?" for kernel launch events, try increasing this value.
    sizeofStringTableData: 8388608            # Maximum total bytes of CUfunction name text at any given time. If kernel names are missing or "?Unknown?" for kernel launch events, try increasing this value.
    maxStringTableEntryLength: 32768          # Troubleshooting option.
    traceRuntimeApiCalls: false               # Whether to trace runtime CUDA API calls or not
    driverApiTraceFirewall:                   # Firewall rules for driver API trace (default to trace all functions). Will be ignored if traceRuntimeApiCalls set to true.
        - firewallOp: exclude
          cudaApiCategories: [Getters, Setters]
        # # include APIs with modules "MemoryManagement"
        # - firewallOp: include
        #   cudaApiModules: [MemoryManagement]
        # # accept APIs whose names contain "cuLaunch", "Memset" or "Memcpy"
        # - firewallOp: accept
        #   cudaApiNameContains: [cuLaunch, Memset, Memcpy]
        # # reject APIs (1) with categories "Getters" or "Setters" AND (2) modules "MemoryManagement" AND (3) whose names contain "cuLaunch", "Memset" or "Memcpy"
        # - firewallOp: reject
        #   cudaApiCategories: [Getters, Setters]
        #   cudaApiModules: [MemoryManagement]
        #   cudaApiNameContains: [cuLaunch, Memset, Memcpy]

NvtxTrace:
    registeredStringArenaSize: 1048576        # Byte size of the registered-string arena
    maxRegisteredStringRecordPayloadSize: 512 # Per-record payload byte cap for registered strings
    maxDomainCount: 16                        # Maximum traced NVTX domains
    maxStateChunkPoolCountLog2: 4             # log2(number of state chunk pools)
    maxStateChunkCountPerPool: 2048           # State chunks per pool
    maxStateChunkSize: 256                    # State chunk size in bytes
    maxPushPopRangePoolCountLog2: 4           # log2(number of push-pop range state pools)
    maxPushPopRangeCountPerPool: 16384        # Push-pop range state entries per pool
    minPushPopRangeInlineMessageSize: 128     # Inline message bytes per push-pop range state
    maxStackGroupPoolCountLog2: 4             # log2(number of stack group state pools)
    maxStackGroupCountPerPool: 4096           # Stack group state entries per pool
    minStackGroupInlineStackRefCount: 2       # Inline StackRef entries per stack group state

ThreadEvent:
    maxThreadCount: 1024                      # Maximum traced threads shared by thread-event producers
    circularFifoEntriesPerThread: 1024        # FIFO entries per thread
    chunkCount: 32768                         # Number of chunks for thread-event buffer management
    sizeofChunk: 2048                         # Size of each chunk for thread-event buffer management
    maxSourceProviderCount: 16                # Source provider index slots, the static ThreadEvent, CUDA and NVTX ones included
)NVTRACEYAML";
        return pNvTraceYamlConfig;
    }

} // namespace nv::trace
