/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <variant>
#include <vector>

#include <NvTrace/DataFormats/nvtrace_common_types.h>

// Database-agnostic description of exported tables and rows.
//
// A table sink (e.g. DuckDbTableSink) turns these descriptions into a concrete store. A sink type
// passed to TableExporter must provide:
//   bool BeginSnapshotIngestionTransaction();
//   bool AppendRow(size_t tableIndex, const TableValue* pValues, size_t valueCount);
//   bool QueueUpdate(size_t updateIndex, const TableValue* pValues, size_t valueCount);
//   bool CommitSnapshotIngestionTransaction();
//   void AbortSnapshotIngestionTransaction();
namespace nv::trace {

    enum class TableColumnType : uint8_t
    {
        Bool,
        Int8,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Double,
        String,
        Uuid,
    };

    // How a column of an existing row is updated when a later snapshot writes the same primary key.
    enum class TableMergePolicy : uint8_t
    {
        Overwrite,    // The most recently ingested known value wins
        CoalesceKeep, // The first ingested known value wins
        Keep,         // The first value wins, including NULL
        Least,        // The smallest known value wins. Independent of ingest order
        Greatest,     // The largest known value wins. Independent of ingest order
        LogicalOr,    // True once any snapshot reports true
    };

    enum class TableColumnRole : uint8_t
    {
        Required,   // NOT NULL
        Nullable,   // NULL means unknown
        PrimaryKey, // NOT NULL and part of the primary key
    };

    struct TableColumnDesc
    {
        const char* pName = nullptr;
        TableColumnType type = TableColumnType::UInt64;
        TableColumnRole role = TableColumnRole::Required;
        TableMergePolicy mergePolicy = TableMergePolicy::Overwrite;

        bool IsPrimaryKey() const
        {
            return role == TableColumnRole::PrimaryKey;
        }

        bool IsNullable() const
        {
            return role == TableColumnRole::Nullable;
        }
    };

    struct TableDesc
    {
        const char* pName = nullptr;
        std::vector<TableColumnDesc> columns;
    };

    struct TableViewDesc
    {
        const char* pName = nullptr;
        const char* pSelectSql = nullptr;
    };

    // Sets NULL columns of one row ingested from an earlier snapshot, for an event whose later half can't
    // build the row's primary key.
    struct TableUpdateDesc
    {
        const char* pName = nullptr;
        size_t tableIndex = 0;
        std::vector<const char*> matchColumns;
        std::vector<const char*> updateColumns;
        // When set, a value after the match values bounds this column, and only the matching row with the greatest
        // value up to the bound is updated
        const char* pLatestColumn = nullptr;

        // Values are the match values, then the latest column's bound if any, then the update values
        size_t ConditionValueCount() const
        {
            return matchColumns.size() + (pLatestColumn ? 1 : 0);
        }

        size_t ValueCount() const
        {
            return ConditionValueCount() + updateColumns.size();
        }

        const char* ValueColumn(size_t valueIndex) const
        {
            if (valueIndex < matchColumns.size())
            {
                return matchColumns[valueIndex];
            }
            if (valueIndex < ConditionValueCount())
            {
                return pLatestColumn;
            }
            return updateColumns[valueIndex - ConditionValueCount()];
        }
    };

    inline const TableColumnDesc* FindTableColumnDesc(const TableDesc& tableDesc, std::string_view columnName)
    {
        for (const TableColumnDesc& columnDesc : tableDesc.columns)
        {
            if (columnName == columnDesc.pName)
            {
                return &columnDesc;
            }
        }
        return nullptr;
    }

    // std::monostate is NULL. Other alternatives must match the column's TableColumnType.
    using TableValue = std::variant<std::monostate, bool, int8_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, double, std::string_view, NVTRACE_UUID>;

    inline bool IsTableValueOfType(const TableValue& value, TableColumnType type)
    {
        switch (type)
        {
            case TableColumnType::Bool:
                return std::holds_alternative<bool>(value);
            case TableColumnType::Int8:
                return std::holds_alternative<int8_t>(value);
            case TableColumnType::Int32:
                return std::holds_alternative<int32_t>(value);
            case TableColumnType::Int64:
                return std::holds_alternative<int64_t>(value);
            case TableColumnType::UInt8:
                return std::holds_alternative<uint8_t>(value);
            case TableColumnType::UInt16:
                return std::holds_alternative<uint16_t>(value);
            case TableColumnType::UInt32:
                return std::holds_alternative<uint32_t>(value);
            case TableColumnType::UInt64:
                return std::holds_alternative<uint64_t>(value);
            case TableColumnType::Double:
                return std::holds_alternative<double>(value);
            case TableColumnType::String:
                return std::holds_alternative<std::string_view>(value);
            case TableColumnType::Uuid:
                return std::holds_alternative<NVTRACE_UUID>(value);
        }
        return false;
    }

    // Version of the schema described by GetTableDescs().
    // Changing this requires adding some logic in UpgradeSchema() in NvTraceDuckDbTableSink.h to migrate from older versions.
    constexpr int32_t TableSchemaVersion = 1;

    // Exported tables, indexing GetTableDescs().
    enum TableIndex : size_t
    {
        TableIndex_Traces,
        TableIndex_SnapshotMetadata,
        TableIndex_Processes,
        TableIndex_ThreadEventThreads,
        TableIndex_CudaDevices,
        TableIndex_CudaContexts,
        TableIndex_CudaStreams,
        TableIndex_CudaModules,
        TableIndex_CudaFunctions,
        TableIndex_CudaRuntimeApiCalls,
        TableIndex_CudaDriverApiCalls,
        TableIndex_CudaStreamApiCallRefs,
        TableIndex_CudaGraphExecs,
        TableIndex_CudaGraphNodes,
        TableIndex_CudaGraphLaunches,
        TableIndex_CudaImmediateKernelLaunches,
        TableIndex_CudaGraphNodeLaunches,
        TableIndex_CudaGpuCommands,
        TableIndex_CudaGpuMemcpyDetails,
        TableIndex_CudaGpuMemsetDetails,
        TableIndex_CudaGpuMemOps,
        TableIndex_CudaHostCallbackInstances,
        TableIndex_NvtxDomains,
        TableIndex_NvtxMarks,
        TableIndex_NvtxPushPopRanges,
        TableIndex_GpuPartitions,
        TableIndex_GpuPartitionSupportedProviders,
        TableIndex_GpuPartitionCudaAttributes,
        TableIndex_GpuGrCtxResidentContexts,
        TableIndex_GpuPmSamples,
        TableIndex_GpuMetrics,
        TableIndex_NvtxGpuProjectionSegments,
        TableIndex_Count
    };

    inline const std::vector<TableDesc>& GetTableDescs()
    {
        constexpr TableColumnRole PrimaryKey = TableColumnRole::PrimaryKey;
        constexpr TableColumnRole Required = TableColumnRole::Required;
        constexpr TableColumnRole Nullable = TableColumnRole::Nullable;
        // clang-format off
        static const std::vector<TableDesc> tableDescs = {
            // Contains capture-artifact provenance and the host system configuration recorded with each loaded TSF snapshot.
            {"traces", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"trace_dir",                                    TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"capture_start_utc_ns",                         TableColumnType::UInt64, PrimaryKey},
                {"capture_end_utc_ns",                           TableColumnType::UInt64, Required,   TableMergePolicy::Greatest},
                {"os_version",                                   TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"os_family",                                    TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"cpu_model",                                    TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"cpu_arch",                                     TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"sysmem_bytes_total",                           TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"nvidia_rm_kernel_driver_version_major",        TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"nvidia_rm_kernel_driver_version_minor",        TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"nvidia_rm_kernel_driver_version_patch",        TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"nvidia_mrm_kernel_driver_version_major",       TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"nvidia_mrm_kernel_driver_version_minor",       TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"nvidia_mrm_kernel_driver_version_patch",       TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Records each snapshot's observation window and when it was ingested.
            {"snapshot_metadata", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"capture_start_utc_ns",                         TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"observation_start_utc_ns",                     TableColumnType::UInt64, PrimaryKey},
                {"observation_end_utc_ns",                       TableColumnType::UInt64, PrimaryKey},
                {"snapshot_number",                              TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"ingested_at_utc_ns",                           TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains process identities, namespace information, executable paths, and observed lifetimes.
            {"processes", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_namespace",                         TableColumnType::Int32,  Required,   TableMergePolicy::CoalesceKeep},
                {"process_parent_id_namespace",                  TableColumnType::Int32,  Required,   TableMergePolicy::CoalesceKeep},
                {"process_id_host",                              TableColumnType::Int32,  Required,   TableMergePolicy::CoalesceKeep},
                {"process_namespace_inode",                      TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"process_namespace_device",                     TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"path",                                         TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains process thread identities, native thread IDs, and observed lifetimes.
            {"thread_event_threads", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"thread_id_native",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains process-local CUDA device mappings, device handles, and physical GPU identities.
            {"cuda_devices", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cudevice_handle",                              TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains CUDA context identities, handles, and observed lifetimes.
            {"cuda_contexts", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"cucontext_id",                                 TableColumnType::UInt64, PrimaryKey},
                {"cucontext_handle",                             TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains CUDA stream identities, handles, owning contexts, devices, and observed lifetimes.
            {"cuda_streams", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   Required,   TableMergePolicy::CoalesceKeep},
                {"custream_id",                                  TableColumnType::UInt64, PrimaryKey},
                {"cucontext_id",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"custream_handle",                              TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains loaded CUDA module identities, handles, owning contexts, and observed lifetimes.
            {"cuda_modules", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"cumodule_id",                                  TableColumnType::UInt64, PrimaryKey},
                {"cumodule_handle",                              TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"context_id",                                   TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains CUDA function identities, handles, module ownership, names, and observed lifetimes.
            {"cuda_functions", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"cufunction_id",                                TableColumnType::UInt64, PrimaryKey},
                {"cufunction_handle",                            TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cumodule_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"name_mangled",                                 TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"name_demangled",                               TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"name_unqualified",                             TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains timed CUDA Runtime API invocations attributed to their calling threads.
            {"cuda_runtime_api_calls", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"runtime_api_correlation_id",                   TableColumnType::UInt64, PrimaryKey},
                {"start_utc_ns",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"end_utc_ns",                                   TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
                {"runtime_api_function_id",                      TableColumnType::UInt16, Required,   TableMergePolicy::CoalesceKeep},
                {"runtime_api_function_name",                    TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains timed CUDA Driver API invocations attributed to their calling threads.
            {"cuda_driver_api_calls", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"driver_api_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"start_utc_ns",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"end_utc_ns",                                   TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
                {"driver_api_function_id",                       TableColumnType::UInt16, Required,   TableMergePolicy::CoalesceKeep},
                {"driver_api_function_name",                     TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains references from CUDA streams to the runtime and driver API calls that submitted work to them.
            {"cuda_stream_api_call_refs", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"custream_id",                                  TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"driver_api_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"runtime_api_correlation_id",                   TableColumnType::UInt64, PrimaryKey},
            }},
            // Contains instantiated CUDA graph executable resources and their observed lifetimes.
            {"cuda_graph_execs", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"cugraph_exec_id",                              TableColumnType::UInt64, PrimaryKey},
                {"cugraph_exec_handle",                          TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains CUDA graph node resources, graph-executable ownership, recording threads, and observed lifetimes.
            {"cuda_graph_nodes", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"cugraph_node_id",                              TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cugraph_node_handle",                          TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cugraph_exec_id",                              TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"thread_id_unique",                             TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"lifetime_start_utc_ns",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"lifetime_end_utc_ns",                          TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains host- and device-initiated CUDA graph launch events and their graph, stream, and device associations.
            // A host launch's row is written with its launch record, and its GridNullEnd sets timestamp_gpu_ns, possibly
            // from a later snapshot.
            {"cuda_graph_launches", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"graph_launch_api_correlation_id",              TableColumnType::UInt64, PrimaryKey},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cucontext_id",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"source",                                       TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"hw_correlation_id",                            TableColumnType::UInt32, PrimaryKey},
                {"cugraph_exec_id",                              TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"driver_api_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                // Device launches have no API correlation ID, so their timestamp keys them. 0 for host launches
                {"device_launch_timestamp_gpu_ns",               TableColumnType::UInt64, PrimaryKey},
                {"timestamp_gpu_ns",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"timestamp_utc_ns",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Nullable,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains immediate CUDA kernel executions with function, stream, context, thread, and GPU timing attribution.
            {"cuda_immediate_kernel_launches", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"kernel_launch_correlation_id",                 TableColumnType::UInt64, PrimaryKey},
                {"hw_correlation_id",                            TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"driver_api_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cucontext_id",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"cufunction_id",                                TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_id",                                      TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_dim_x",                                   TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_dim_y",                                   TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_dim_z",                                   TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"block_dim_x",                                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"block_dim_y",                                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"block_dim_z",                                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cluster_dim_x",                                TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cluster_dim_y",                                TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cluster_dim_z",                                TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"shmem_per_block_bytes_total",                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"shmem_per_block_bytes_static",                 TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"gpu_start_utc_ns",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"gpu_end_utc_ns",                               TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains CUDA graph-node kernel executions with function, dimensions, and GPU times. The graph launch that
            // executed one is the latest launch of its node's graph executable on the GPU at or before
            // start_timestamp_gpu_ns, since launches of one executable run one at a time.
            {"cuda_graph_node_launches", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"start_timestamp_gpu_ns",                       TableColumnType::UInt64, PrimaryKey},
                {"cufunction_id",                                TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_dim_x",                                   TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_dim_y",                                   TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"grid_dim_z",                                   TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"block_dim_x",                                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"block_dim_y",                                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"block_dim_z",                                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cluster_dim_x",                                TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cluster_dim_y",                                TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"cluster_dim_z",                                TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"shmem_per_block_bytes_total",                  TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"shmem_per_block_bytes_static",                 TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"gpu_start_utc_ns",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"gpu_end_utc_ns",                               TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
            }},
            // Contains common timing, type, stream, context, and correlation data for non-kernel CUDA GPU commands.
            {"cuda_gpu_commands", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"start_gpu_timestamp_report_ref",               TableColumnType::UInt64, PrimaryKey},
                {"end_gpu_timestamp_report_ref",                 TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"gpu_command_type",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cucontext_id",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"gpu_start_utc_ns",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"gpu_end_utc_ns",                               TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"thread_id_unique",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"driver_api_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"name_display",                                 TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains memcpy-specific payload details for memcpy rows in cuda_gpu_commands.
            {"cuda_gpu_memcpy_details", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"start_gpu_timestamp_report_ref",               TableColumnType::UInt64, PrimaryKey},
                {"end_gpu_timestamp_report_ref",                 TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"enclosing_correlation_id",                     TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"engine_command_t",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"byte_count",                                   TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"src_address_kind",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"dst_address_kind",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"src_dev_ptr",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"dst_dev_ptr",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"width_bytes",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"height",                                       TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"depth",                                        TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"is_async",                                     TableColumnType::Bool,   Required,   TableMergePolicy::CoalesceKeep},
                {"is_p2p",                                       TableColumnType::Bool,   Required,   TableMergePolicy::CoalesceKeep},
                {"copy_direction",                               TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains memset-specific payload details for memset rows in cuda_gpu_commands.
            {"cuda_gpu_memset_details", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"start_gpu_timestamp_report_ref",               TableColumnType::UInt64, PrimaryKey},
                {"end_gpu_timestamp_report_ref",                 TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"enclosing_correlation_id",                     TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"engine_command_t",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"dev_ptr",                                      TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"byte_count",                                   TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"value",                                        TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains decoded operations within CUDA stream-batch memory-operation commands.
            {"cuda_gpu_mem_ops", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"start_gpu_timestamp_report_ref",               TableColumnType::UInt64, PrimaryKey},
                {"end_gpu_timestamp_report_ref",                 TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"enclosing_correlation_id",                     TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"mem_op_type",                                  TableColumnType::UInt16, Required,   TableMergePolicy::CoalesceKeep},
                {"mem_op_index",                                 TableColumnType::UInt16, PrimaryKey},
                {"flags",                                        TableColumnType::Int32,  Nullable,   TableMergePolicy::CoalesceKeep},
                {"address",                                      TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"value",                                        TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"reduction_op",                                 TableColumnType::UInt8,  Nullable,   TableMergePolicy::CoalesceKeep},
                {"data_type",                                    TableColumnType::UInt8,  Nullable,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains CPU execution intervals and CUDA associations for eager and graph host callbacks.
            {"cuda_host_callback_instances", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   Nullable,   TableMergePolicy::CoalesceKeep},
                {"kind",                                         TableColumnType::Int8,   Required,   TableMergePolicy::CoalesceKeep},
                {"start_utc_ns",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"end_utc_ns",                                   TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
                {"custream_id",                                  TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"cucontext_id",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"host_callback_launch_correlation_id",          TableColumnType::UInt64, PrimaryKey},
                {"graph_launch_api_correlation_id",              TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"graph_launch_cugraph_exec_id",                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"enclosing_correlation_id",                     TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"callback",                                     TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"user_data",                                    TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"executing_thread_id_unique",                   TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"thread_id_unique",                             TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains process-scoped NVTX domain identities and resolved domain names.
            {"nvtx_domains", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"domain_index",                                 TableColumnType::UInt32, PrimaryKey},
                {"domain_name",                                  TableColumnType::String, Nullable,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains timestamped NVTX marks with resolved domains, messages, and thread attribution.
            {"nvtx_marks", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"nvtx_mark_correlation_id",                     TableColumnType::UInt64, PrimaryKey},
                {"timestamp_utc_ns",                             TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"domain_index",                                 TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"domain_name",                                  TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"message_encoding",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"message",                                      TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains timed NVTX push/pop ranges with resolved messages, nesting, ancestry, and thread attribution.
            {"nvtx_push_pop_ranges", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"nvtx_range_ref",                               TableColumnType::UInt64, PrimaryKey},
                {"start_utc_ns",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"end_utc_ns",                                   TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"domain_index",                                 TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"domain_name",                                  TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"nesting_depth",                                TableColumnType::UInt16, Required,   TableMergePolicy::CoalesceKeep},
                {"message_encoding",                             TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"message",                                      TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"parent_nvtx_range_ref",                        TableColumnType::UInt64, Nullable,   TableMergePolicy::Keep},
                {"is_record_time_range",                         TableColumnType::Bool,   Required,   TableMergePolicy::LogicalOr},
                {"is_observation_time_range",                    TableColumnType::Bool,   Required,   TableMergePolicy::LogicalOr},
            }},
            // Contains physical GPU identity and hardware properties.
            {"gpu_partitions", {
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"node_id",                                      TableColumnType::String, Required},
                {"chip_name",                                    TableColumnType::String, Required},
                {"device_name",                                  TableColumnType::String, Required},
            }},
            // Contains the provider types advertised as supported by each GPU.
            {"gpu_partition_supported_providers", {
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"provider_uuid",                                TableColumnType::Uuid,   PrimaryKey},
            }},
            // Contains numeric CUDA device attributes reported for each GPU.
            {"gpu_partition_cuda_attributes", {
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"key",                                          TableColumnType::UInt32, PrimaryKey},
                {"value",                                        TableColumnType::Int32,  Required},
            }},
            // Contains GPU graphics-context residency intervals and their resolved process and CUDA-context attribution.
            // A GPU partition has one resident context at a time, so the load identifies the residency. Residencies
            // whose load wasn't observed have no row.
            {"gpu_gr_ctx_resident_contexts", {
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"ctx_load_sequence_number",                     TableColumnType::UInt64, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"process_id_unique",                            TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
                {"ctx_load_utc_ns",                              TableColumnType::UInt64, Nullable,   TableMergePolicy::Least},
                {"ctx_save_utc_ns",                              TableColumnType::UInt64, Nullable,   TableMergePolicy::Greatest},
                {"vmid",                                         TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"pid",                                          TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"is_multiple_events_dropped",                   TableColumnType::Bool,   Required,   TableMergePolicy::Overwrite},
                {"cucontext_id",                                 TableColumnType::UInt64, Nullable,   TableMergePolicy::CoalesceKeep},
            }},
            // Contains timing metadata for decoded GPU performance-monitoring sample periods.
            {"gpu_pm_samples", {
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"gpu_start_utc_ns",                             TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"gpu_end_utc_ns",                               TableColumnType::UInt64, PrimaryKey},
            }},
            // Contains long-form metric names and values evaluated for each GPU performance-monitoring sample.
            {"gpu_metrics", {
                {"gpu_uuid",                                     TableColumnType::Uuid,   Required},
                {"gpu_end_utc_ns",                               TableColumnType::UInt64, Required},
                {"metric_name",                                  TableColumnType::String, Required},
                {"metric_value",                                 TableColumnType::Double, Required},
            }},
            // Contains one row per GPU operation and NVTX range on that operation's projected stack. The operation's GPU times are in cuda_gpu_operations.
            {"nvtx_gpu_projection_segments", {
                {"node_id",                                      TableColumnType::String, PrimaryKey},
                {"service_start_tsc",                            TableColumnType::UInt64, PrimaryKey},
                {"process_id_unique",                            TableColumnType::UInt64, PrimaryKey},
                {"gpu_uuid",                                     TableColumnType::Uuid,   PrimaryKey},
                {"cudevice_ordinal",                             TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"domain_index",                                 TableColumnType::UInt32, Required,   TableMergePolicy::CoalesceKeep},
                {"nvtx_thread_id_unique",                        TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"nvtx_range_ref",                               TableColumnType::UInt64, PrimaryKey},
                {"projection_stack_root_nvtx_range_ref",         TableColumnType::UInt64, Required,   TableMergePolicy::CoalesceKeep},
                {"projected_parent_nvtx_range_ref",              TableColumnType::UInt64, Nullable,   TableMergePolicy::Keep},
                {"domain_name",                                  TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"message",                                      TableColumnType::String, Required,   TableMergePolicy::CoalesceKeep},
                {"operation_source",                             TableColumnType::String, PrimaryKey},
                {"thread_id_unique",                             TableColumnType::UInt64, PrimaryKey},
                {"driver_api_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"kernel_launch_correlation_id",                 TableColumnType::UInt64, PrimaryKey},
                {"graph_launch_api_correlation_id",              TableColumnType::UInt64, PrimaryKey},
                {"graph_launch_cugraph_exec_id",                 TableColumnType::UInt64, PrimaryKey},
                {"graph_launch_hw_correlation_id",               TableColumnType::UInt32, PrimaryKey},
                {"graph_launch_device_timestamp_gpu_ns",         TableColumnType::UInt64, PrimaryKey},
                {"graph_node_correlation_id",                    TableColumnType::UInt64, PrimaryKey},
                {"start_gpu_timestamp_report_ref",               TableColumnType::UInt64, PrimaryKey},
                {"association_sources",                          TableColumnType::UInt8,  Required,   TableMergePolicy::CoalesceKeep},
                {"stack_depth",                                  TableColumnType::UInt16, Required,   TableMergePolicy::CoalesceKeep},
            }},
        };
        // clang-format on
        return tableDescs;
    }

    enum TableUpdateIndex : size_t
    {
        TableUpdateIndex_CudaImmediateKernelLaunchGpuTimes,
        TableUpdateIndex_CudaGpuCommandGpuStart,
        TableUpdateIndex_CudaGpuCommandGpuEnd,
        TableUpdateIndex_CudaGraphLaunchTimestamp,
        TableUpdateIndex_CudaGraphNodeLaunchGpuEnd,
        TableUpdateIndex_GpuGrCtxResidentContextSave,
        TableUpdateIndex_Count
    };

    inline const std::vector<TableUpdateDesc>& GetTableUpdateDescs()
    {
        static const std::vector<TableUpdateDesc> tableUpdateDescs = {
            // clang-format off
            // A kernel's GPU start and end events can land in a later snapshot than its launch record. They carry only
            // the hardware correlation ID, which correlation IDs being unique per process makes enough to match.
            {"cuda_immediate_kernel_launch_gpu_times", TableIndex_CudaImmediateKernelLaunches,
                {"node_id", "service_start_tsc", "process_id_unique", "gpu_uuid", "hw_correlation_id"},
                {"gpu_start_utc_ns", "gpu_end_utc_ns"}},
            // A GPU command's timestamp reports can land in a later snapshot than its begin record. A report carries
            // only its reference, which is unique per process, and doesn't say whether it is a start or an end.
            {"cuda_gpu_command_gpu_start", TableIndex_CudaGpuCommands,
                {"node_id", "service_start_tsc", "process_id_unique", "gpu_uuid", "start_gpu_timestamp_report_ref"},
                {"gpu_start_utc_ns"}},
            {"cuda_gpu_command_gpu_end", TableIndex_CudaGpuCommands,
                {"node_id", "service_start_tsc", "process_id_unique", "gpu_uuid", "end_gpu_timestamp_report_ref"},
                {"gpu_end_utc_ns"}},
            // A host graph launch's GridNullEnd can land in a later snapshot than its launch record. It carries only
            // the hardware correlation ID. Device launches always have their timestamp, so only host rows match.
            {"cuda_graph_launch_timestamp", TableIndex_CudaGraphLaunches,
                {"node_id", "service_start_tsc", "process_id_unique", "gpu_uuid", "hw_correlation_id"},
                {"timestamp_gpu_ns", "timestamp_utc_ns"}},
            // A graph node's GPU end event can land in a later snapshot than its start. Launches of one graph
            // executable run one at a time, so the end belongs to the latest execution that started before it. An
            // earlier execution whose end was dropped also has no end.
            {"cuda_graph_node_launch_gpu_end", TableIndex_CudaGraphNodeLaunches,
                {"node_id", "service_start_tsc", "process_id_unique", "gpu_uuid", "graph_node_correlation_id"},
                {"gpu_end_utc_ns"},
                "start_timestamp_gpu_ns"},
            // A GR context's save event can land in a later snapshot than its load. The provider's buffer numbers its
            // records in sequence, so the save follows its load by one.
            {"gpu_gr_ctx_resident_context_save", TableIndex_GpuGrCtxResidentContexts,
                {"gpu_uuid", "ctx_load_sequence_number"},
                {"ctx_save_utc_ns"}},
            // clang-format on
        };
        return tableUpdateDescs;
    }

    inline const std::vector<TableViewDesc>& GetTableViewDescs()
    {
        static const std::vector<TableViewDesc> tableViewDescs = {
            // Convenience derived table approximating the events row, with a unified query surface for kernel and non-kernel CUDA GPU activity intervals.
            {      "cuda_gpu_operations", R"sql(
                SELECT node_id, service_start_tsc, process_id_unique, gpu_uuid, cudevice_ordinal,
                       'immediate_kernel_launch' AS operation_source,
                       thread_id_unique, kernel_launch_correlation_id,
                       NULL::UBIGINT AS graph_launch_api_correlation_id,
                       NULL::UBIGINT AS graph_launch_cugraph_exec_id,
                       NULL::UINTEGER AS graph_launch_hw_correlation_id,
                       NULL::UBIGINT AS graph_launch_device_timestamp_gpu_ns,
                       NULL::UBIGINT AS graph_node_correlation_id,
                       NULL::UBIGINT AS start_gpu_timestamp_report_ref,
                       'kernel' AS operation_kind, custream_id, cucontext_id,
                       cufunction_id AS function_id,
                       NULL::INTEGER AS command_type,
                       gpu_start_utc_ns, gpu_end_utc_ns,
                       CASE WHEN gpu_start_utc_ns IS NOT NULL AND gpu_end_utc_ns IS NOT NULL
                            THEN gpu_end_utc_ns - gpu_start_utc_ns END AS duration_ns
                FROM cuda_immediate_kernel_launches
                UNION ALL
                SELECT n.node_id, n.service_start_tsc, n.process_id_unique, n.gpu_uuid, n.cudevice_ordinal,
                       'graph_node_launch', gl.thread_id_unique, NULL::UBIGINT,
                       gl.graph_launch_api_correlation_id, gl.cugraph_exec_id,
                       gl.hw_correlation_id,
                       gl.device_launch_timestamp_gpu_ns, n.graph_node_correlation_id,
                       NULL::UBIGINT, 'kernel', gl.custream_id, gl.cucontext_id,
                       n.cufunction_id, NULL::INTEGER, n.gpu_start_utc_ns, n.gpu_end_utc_ns,
                       CASE WHEN n.gpu_start_utc_ns IS NOT NULL AND n.gpu_end_utc_ns IS NOT NULL
                            THEN n.gpu_end_utc_ns - n.gpu_start_utc_ns END
                FROM cuda_graph_node_launches n
                LEFT JOIN cuda_graph_nodes g
                    ON g.node_id = n.node_id
                   AND g.service_start_tsc = n.service_start_tsc
                   AND g.process_id_unique = n.process_id_unique
                   AND g.graph_node_correlation_id = n.graph_node_correlation_id
                ASOF LEFT JOIN cuda_graph_launches gl
                    ON gl.node_id = n.node_id
                   AND gl.service_start_tsc = n.service_start_tsc
                   AND gl.process_id_unique = n.process_id_unique
                   AND gl.gpu_uuid = n.gpu_uuid
                   AND gl.cugraph_exec_id = g.cugraph_exec_id
                   AND n.start_timestamp_gpu_ns >= gl.timestamp_gpu_ns
                UNION ALL
                SELECT node_id, service_start_tsc, process_id_unique, gpu_uuid, cudevice_ordinal,
                       'gpu_command', thread_id_unique, NULL::UBIGINT,
                       NULL::UBIGINT, NULL::UBIGINT, NULL::UINTEGER, NULL::UBIGINT,
                       NULL::UBIGINT, start_gpu_timestamp_report_ref,
                       CASE gpu_command_type
                           WHEN 1 THEN 'memcpy' WHEN 3 THEN 'memcpy'
                           WHEN 2 THEN 'memset' WHEN 4 THEN 'memset'
                           WHEN 5 THEN 'mem_op' ELSE 'other_gpu_command'
                       END,
                       custream_id, cucontext_id, NULL::UBIGINT, gpu_command_type,
                       gpu_start_utc_ns, gpu_end_utc_ns,
                       CASE WHEN gpu_start_utc_ns IS NOT NULL AND gpu_end_utc_ns IS NOT NULL
                            THEN gpu_end_utc_ns - gpu_start_utc_ns END
                FROM cuda_gpu_commands
                WHERE gpu_command_type <> 6
            )sql"},
            // Coalesces nvtx_gpu_projection_segments into one span per run of consecutive GPU operations, in GPU start
            // order per launching thread, that carry the same NVTX range. Record-time-only ranges additionally split
            // at driver API call boundaries so separate launches of the same graph cannot form one span.
            // Computed over the merged snapshots, so a run crossing a snapshot boundary is a single span.
            {"nvtx_gpu_projection_spans", R"sql(
                WITH operations AS (
                    SELECT o.node_id, o.service_start_tsc, o.process_id_unique, o.gpu_uuid, o.operation_source,
                           COALESCE(o.thread_id_unique, 0) AS thread_id_unique,
                           COALESCE(o.kernel_launch_correlation_id, 0) AS kernel_launch_correlation_id,
                           COALESCE(o.graph_launch_api_correlation_id, 0) AS graph_launch_api_correlation_id,
                           COALESCE(o.graph_launch_cugraph_exec_id, 0) AS graph_launch_cugraph_exec_id,
                           COALESCE(o.graph_launch_hw_correlation_id, 0) AS graph_launch_hw_correlation_id,
                           COALESCE(o.graph_launch_device_timestamp_gpu_ns, 0) AS graph_launch_device_timestamp_gpu_ns,
                           COALESCE(o.graph_node_correlation_id, 0) AS graph_node_correlation_id,
                           COALESCE(o.start_gpu_timestamp_report_ref, 0) AS start_gpu_timestamp_report_ref,
                           o.gpu_start_utc_ns, o.gpu_end_utc_ns
                    FROM cuda_gpu_operations o
                ),
                sequenced_operations AS (
                    SELECT *,
                           row_number() OVER (
                               PARTITION BY node_id, service_start_tsc, process_id_unique, gpu_uuid, thread_id_unique
                               ORDER BY gpu_start_utc_ns NULLS LAST, gpu_end_utc_ns NULLS LAST, operation_source,
                                        kernel_launch_correlation_id, graph_launch_api_correlation_id, graph_launch_cugraph_exec_id,
                                        graph_launch_hw_correlation_id, graph_launch_device_timestamp_gpu_ns, graph_node_correlation_id,
                                        start_gpu_timestamp_report_ref
                           ) AS operation_sequence
                    FROM operations
                ),
                grouped_segments AS (
                    SELECT s.node_id, s.service_start_tsc, s.process_id_unique, s.gpu_uuid, s.cudevice_ordinal, s.thread_id_unique,
                           s.domain_index, s.domain_name, s.nvtx_thread_id_unique, s.nvtx_range_ref, s.message,
                           s.projection_stack_root_nvtx_range_ref, s.projected_parent_nvtx_range_ref,
                           s.association_sources, s.stack_depth,
                           CASE WHEN s.association_sources = 2 THEN s.driver_api_correlation_id ELSE 0 END
                               AS span_driver_api_correlation_id,
                           q.gpu_start_utc_ns AS operation_start_utc_ns,
                           q.gpu_end_utc_ns AS operation_end_utc_ns,
                           q.operation_sequence - row_number() OVER (
                               PARTITION BY s.node_id, s.service_start_tsc, s.process_id_unique, s.gpu_uuid, s.thread_id_unique, s.nvtx_range_ref,
                                            CASE WHEN s.association_sources = 2 THEN s.driver_api_correlation_id ELSE 0 END
                               ORDER BY q.operation_sequence
                           ) AS span_group
                    FROM nvtx_gpu_projection_segments s
                    JOIN sequenced_operations q
                        ON q.node_id = s.node_id
                       AND q.service_start_tsc = s.service_start_tsc
                       AND q.process_id_unique = s.process_id_unique
                       AND q.gpu_uuid = s.gpu_uuid
                       AND q.operation_source = s.operation_source
                       AND q.thread_id_unique = s.thread_id_unique
                       AND q.kernel_launch_correlation_id = s.kernel_launch_correlation_id
                       AND q.graph_launch_api_correlation_id = s.graph_launch_api_correlation_id
                       AND q.graph_launch_cugraph_exec_id = s.graph_launch_cugraph_exec_id
                       AND q.graph_launch_hw_correlation_id = s.graph_launch_hw_correlation_id
                       AND q.graph_launch_device_timestamp_gpu_ns = s.graph_launch_device_timestamp_gpu_ns
                       AND q.graph_node_correlation_id = s.graph_node_correlation_id
                       AND q.start_gpu_timestamp_report_ref = s.start_gpu_timestamp_report_ref
                ),
                ranked_segments AS (
                    SELECT *,
                           row_number() OVER (
                               PARTITION BY node_id, service_start_tsc, process_id_unique, gpu_uuid, thread_id_unique, nvtx_range_ref, span_driver_api_correlation_id, span_group
                               ORDER BY stack_depth, operation_start_utc_ns NULLS LAST
                           ) AS depth_rank
                    FROM grouped_segments
                )
                SELECT node_id, service_start_tsc, process_id_unique, gpu_uuid,
                       any_value(cudevice_ordinal) AS cudevice_ordinal,
                       thread_id_unique,
                       any_value(domain_index) AS domain_index,
                       any_value(domain_name) AS domain_name,
                       any_value(nvtx_thread_id_unique) AS nvtx_thread_id_unique,
                       nvtx_range_ref,
                       max(CASE WHEN depth_rank = 1 THEN projection_stack_root_nvtx_range_ref END) AS projection_stack_root_nvtx_range_ref,
                       max(CASE WHEN depth_rank = 1 THEN projected_parent_nvtx_range_ref END) AS projected_parent_nvtx_range_ref,
                       any_value(message) AS message,
                       CAST(max(association_sources % 2) + 2 * max(association_sources // 2 % 2) AS UTINYINT) AS association_sources,
                       min(stack_depth) AS stack_depth,
                       count(*) AS operation_count,
                       min(operation_start_utc_ns) AS gpu_start_utc_ns,
                       max(operation_end_utc_ns) AS gpu_end_utc_ns,
                       max(operation_end_utc_ns) - min(operation_start_utc_ns) AS duration_ns
                FROM ranked_segments
                GROUP BY node_id, service_start_tsc, process_id_unique, gpu_uuid, thread_id_unique, nvtx_range_ref, span_driver_api_correlation_id, span_group
            )sql"},
        };
        return tableViewDescs;
    }

} // namespace nv::trace
