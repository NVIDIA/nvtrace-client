/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <duckdb.hpp>
#include <duckdb/common/types/uuid.hpp>

#include <NvTrace/Offline/Exporter/NvTraceTableSchema.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

namespace nv::trace {

    // Table sink backed by a DuckDB database file. Rows are buffered per table and upserted into the
    // tables with ON CONFLICT whenever a buffer flushes, and updates are buffered per update and applied in
    // batches, all inside a single transaction per snapshot ingestion.
    class DuckDbTableSink
    {
    public:
        static constexpr const char* AppendedRowsTableName = "appended_rows";
        static constexpr size_t DefaultUpsertFlushThresholdBytes = size_t(16) << 20;

        DuckDbTableSink() = default;
        DuckDbTableSink(const DuckDbTableSink&) = delete;
        DuckDbTableSink& operator=(const DuckDbTableSink&) = delete;
        ~DuckDbTableSink()
        {
            Close();
        }

        // Opens databasePath and readies it for ingestion with tableDescs at schemaVersion.
        bool Open(const std::filesystem::path& databasePath, const std::vector<TableDesc>& tableDescs, int32_t schemaVersion, size_t upsertFlushThresholdBytes = DefaultUpsertFlushThresholdBytes, size_t memoryLimitBytes = 0)
        {
            Close();
            m_upsertFlushThresholdBytes = upsertFlushThresholdBytes;
            try
            {
                m_pDatabase = std::make_unique<duckdb::DuckDB>(databasePath.string());
                m_pConnection = std::make_unique<duckdb::Connection>(*m_pDatabase);
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to open DuckDB database %s: %s", databasePath.string().c_str(), e.what());
                Close();
                return false;
            }
            if (memoryLimitBytes && !Execute("SET memory_limit = '" + std::to_string(memoryLimitBytes) + "B'"))
            {
                NVTRACE_LOG_ERR(50, "Failed to set the memory limit of DuckDB database %s", databasePath.string().c_str());
                Close();
                return false;
            }
            if (!PrepareSchema(tableDescs, schemaVersion))
            {
                NVTRACE_LOG_ERR(50, "Failed to prepare schema for DuckDB database %s", databasePath.string().c_str());
                Close();
                return false;
            }
            return true;
        }

        void Close()
        {
            AbortSnapshotIngestionTransaction();
            m_pConnection.reset();
            m_pDatabase.reset();
            m_tableDescs.clear();
        }

        bool BeginSnapshotIngestionTransaction()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_pConnection, false, 50, "DuckDB database is not open");
            NVTRACE_LOG_ERR_RETURN_IF(m_isSnapshotIngestionTransactionActive, false, 50, "A snapshot ingestion transaction is already active");
            NVTRACE_LOG_ERR_RETURN_IF(!Execute("BEGIN TRANSACTION"), false, 50, "Failed to begin snapshot ingestion transaction");
            m_isSnapshotIngestionTransactionActive = true;

            auto abortGuard = ScopeExitGuard([this]() { AbortSnapshotIngestionTransaction(); });
            const duckdb::idx_t flushMemoryThreshold = m_upsertFlushThresholdBytes ? duckdb::idx_t(m_upsertFlushThresholdBytes) : duckdb::DConstants::INVALID_INDEX;
            try
            {
                for (const TableDesc& tableDesc : m_tableDescs)
                {
                    std::vector<duckdb::LogicalType> columnTypes;
                    std::vector<std::string> columnNames;
                    for (const TableColumnDesc& columnDesc : tableDesc.columns)
                    {
                        columnTypes.push_back(GetDuckDbLogicalType(columnDesc.type));
                        columnNames.push_back(columnDesc.pName);
                    }
                    m_upsertAppenders.push_back(std::make_unique<duckdb::QueryAppender>(*m_pConnection, UpsertSql(tableDesc), std::move(columnTypes), std::move(columnNames), AppendedRowsTableName, flushMemoryThreshold));
                }
                for (const TableUpdateDesc& updateDesc : GetTableUpdateDescs())
                {
                    NVTRACE_LOG_ERR_RETURN_IF(updateDesc.tableIndex >= m_tableDescs.size(), false, 50, "Update %s targets an unknown table", updateDesc.pName);
                    const TableDesc& tableDesc = m_tableDescs[updateDesc.tableIndex];
                    std::vector<duckdb::LogicalType> columnTypes;
                    std::vector<std::string> columnNames;
                    for (size_t valueIndex = 0; valueIndex < updateDesc.ValueCount(); ++valueIndex)
                    {
                        const TableColumnDesc* pColumnDesc = FindTableColumnDesc(tableDesc, updateDesc.ValueColumn(valueIndex));
                        NVTRACE_LOG_ERR_RETURN_IF(!pColumnDesc, false, 50, "Update %s names unknown column %s.%s", updateDesc.pName, tableDesc.pName, updateDesc.ValueColumn(valueIndex));
                        columnTypes.push_back(GetDuckDbLogicalType(pColumnDesc->type));
                        columnNames.push_back(pColumnDesc->pName);
                    }
                    m_updateAppenders.push_back(std::make_unique<duckdb::QueryAppender>(*m_pConnection, BatchUpdateSql(tableDesc, updateDesc), std::move(columnTypes), std::move(columnNames), AppendedRowsTableName, flushMemoryThreshold));
                }
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to create DuckDB upsert and update appenders: %s", e.what());
                return false;
            }
            abortGuard.Dismiss();
            return true;
        }

        bool AppendRow(size_t tableIndex, const TableValue* pValues, size_t valueCount)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_isSnapshotIngestionTransactionActive, false, 50, "No snapshot ingestion transaction is active");
            NVTRACE_LOG_ERR_RETURN_IF(tableIndex >= m_tableDescs.size(), false, 50, "Invalid tableIndex=%zu", tableIndex);
            const TableDesc& tableDesc = m_tableDescs[tableIndex];
            NVTRACE_LOG_ERR_RETURN_IF(valueCount != tableDesc.columns.size(), false, 50, "Table %s expects %zu values, got %zu", tableDesc.pName, tableDesc.columns.size(), valueCount);

            for (size_t columnIndex = 0; columnIndex < valueCount; ++columnIndex)
            {
                if (!IsValidColumnValue(tableDesc, tableDesc.columns[columnIndex], pValues[columnIndex]))
                {
                    return false;
                }
            }
            duckdb::QueryAppender& appender = *m_upsertAppenders[tableIndex];
            try
            {
                appender.BeginRow();
                for (size_t columnIndex = 0; columnIndex < valueCount; ++columnIndex)
                {
                    const TableValue& value = pValues[columnIndex];
                    if (std::holds_alternative<std::monostate>(value))
                    {
                        appender.Append(nullptr);
                        continue;
                    }
                    AppendValue(appender, value);
                }
                appender.EndRow();
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to append row to %s: %s", tableDesc.pName, e.what());
                return false;
            }
            return true;
        }

        // Updates are buffered per update and applied in batches, when a buffer flushes or the
        // snapshot commits. When queued updates disagree on a column's value, the one applied is arbitrary.
        bool QueueUpdate(size_t updateIndex, const TableValue* pValues, size_t valueCount)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_isSnapshotIngestionTransactionActive, false, 50, "No snapshot ingestion transaction is active");
            const std::vector<TableUpdateDesc>& updateDescs = GetTableUpdateDescs();
            NVTRACE_LOG_ERR_RETURN_IF(updateIndex >= updateDescs.size(), false, 50, "Invalid updateIndex=%zu", updateIndex);
            const TableUpdateDesc& updateDesc = updateDescs[updateIndex];
            NVTRACE_LOG_ERR_RETURN_IF(updateDesc.tableIndex >= m_tableDescs.size(), false, 50, "Update %s targets an unknown table", updateDesc.pName);
            const TableDesc& tableDesc = m_tableDescs[updateDesc.tableIndex];
            const size_t conditionValueCount = updateDesc.ConditionValueCount();
            NVTRACE_LOG_ERR_RETURN_IF(valueCount != updateDesc.ValueCount(), false, 50, "Update %s expects %zu values, got %zu", updateDesc.pName, updateDesc.ValueCount(), valueCount);

            for (size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex)
            {
                const char* pColumnName = updateDesc.ValueColumn(valueIndex);
                const TableColumnDesc* pColumnDesc = FindTableColumnDesc(tableDesc, pColumnName);
                NVTRACE_LOG_ERR_RETURN_IF(!pColumnDesc, false, 50, "Update %s names unknown column %s.%s", updateDesc.pName, tableDesc.pName, pColumnName);
                const TableValue& value = pValues[valueIndex];
                NVTRACE_LOG_ERR_RETURN_IF(valueIndex < conditionValueCount && std::holds_alternative<std::monostate>(value), false, 50, "NULL in update %s condition column %s", updateDesc.pName, pColumnName);
                if (!IsValidColumnValue(tableDesc, *pColumnDesc, value))
                {
                    return false;
                }
            }

            duckdb::QueryAppender& appender = *m_updateAppenders[updateIndex];
            try
            {
                appender.BeginRow();
                for (size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex)
                {
                    const TableValue& value = pValues[valueIndex];
                    if (std::holds_alternative<std::monostate>(value))
                    {
                        appender.Append(nullptr);
                        continue;
                    }
                    AppendValue(appender, value);
                }
                appender.EndRow();
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to queue update %s: %s", updateDesc.pName, e.what());
                return false;
            }
            return true;
        }

        bool CommitSnapshotIngestionTransaction()
        {
            NVTRACE_LOG_ERR_RETURN_IF(!m_isSnapshotIngestionTransactionActive, false, 50, "No snapshot ingestion transaction is active");

            auto abortGuard = ScopeExitGuard([this]() { AbortSnapshotIngestionTransaction(); });
            try
            {
                for (std::unique_ptr<duckdb::QueryAppender>& pAppender : m_upsertAppenders)
                {
                    pAppender->Close();
                }
                m_upsertAppenders.clear();
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to upsert buffered rows: %s", e.what());
                return false;
            }
            // Applied after the upserts, so a batch can also update rows that this snapshot inserted.
            const std::vector<TableUpdateDesc>& updateDescs = GetTableUpdateDescs();
            for (size_t updateIndex = 0; updateIndex < m_updateAppenders.size(); ++updateIndex)
            {
                try
                {
                    m_updateAppenders[updateIndex]->Close();
                }
                catch (const std::exception& e)
                {
                    NVTRACE_LOG_ERR(50, "Failed to apply queued %s updates: %s", updateDescs[updateIndex].pName, e.what());
                    return false;
                }
            }
            m_updateAppenders.clear();

            NVTRACE_LOG_ERR_RETURN_IF(!Execute("COMMIT"), false, 50, "Failed to commit snapshot ingestion transaction");

            abortGuard.Dismiss();
            m_isSnapshotIngestionTransactionActive = false;
            return true;
        }

        void AbortSnapshotIngestionTransaction()
        {
            if (!m_isSnapshotIngestionTransactionActive)
            {
                return;
            }

            // Destroying a QueryAppender discards its buffered rows without flushing them.
            m_upsertAppenders.clear();
            m_updateAppenders.clear();
            m_isSnapshotIngestionTransactionActive = false;

            if (!Execute("ROLLBACK"))
            {
                NVTRACE_LOG_ERR(50, "Failed to roll back snapshot ingestion transaction");
            }
        }

        static std::string CreateTableSql(const TableDesc& tableDesc)
        {
            std::ostringstream sql;
            sql << "CREATE TABLE IF NOT EXISTS " << tableDesc.pName << " (";
            const char* pSeparator = "";
            for (const TableColumnDesc& columnDesc : tableDesc.columns)
            {
                sql << pSeparator << columnDesc.pName << ' ' << GetDuckDbTypeName(columnDesc.type);
                if (!columnDesc.IsNullable())
                {
                    sql << " NOT NULL";
                }
                pSeparator = ", ";
            }
            const std::string primaryKeyColumns = GetPrimaryKeyColumns(tableDesc);
            if (!primaryKeyColumns.empty())
            {
                sql << ", PRIMARY KEY (" << primaryKeyColumns << ')';
            }
            sql << ')';
            return sql.str();
        }

        // Run by each flush of a table's buffered rows, which the query reads as AppendedRowsTableName.
        // Assumes a snapshot appends each primary key at most once.
        static std::string UpsertSql(const TableDesc& tableDesc)
        {
            std::ostringstream insertColumns;
            std::ostringstream updateColumns;
            const char* pInsertSeparator = "";
            const char* pUpdateSeparator = "";
            for (const TableColumnDesc& columnDesc : tableDesc.columns)
            {
                const std::string_view column = columnDesc.pName;
                insertColumns << pInsertSeparator << column;
                pInsertSeparator = ", ";
                if (!columnDesc.IsPrimaryKey())
                {
                    updateColumns << pUpdateSeparator << column << " = ";
                    WriteConflictUpdateSql(updateColumns, column, columnDesc.mergePolicy);
                    pUpdateSeparator = ", ";
                }
            }

            const std::string insertColumnList = insertColumns.str();
            std::ostringstream sql;
            sql << "INSERT INTO " << tableDesc.pName << " (" << insertColumnList << ") SELECT " << insertColumnList << " FROM " << AppendedRowsTableName;
            const std::string primaryKeyColumns = GetPrimaryKeyColumns(tableDesc);
            if (!primaryKeyColumns.empty())
            {
                const std::string updateColumnList = updateColumns.str();
                sql << " ON CONFLICT (" << primaryKeyColumns << ") ";
                if (updateColumnList.empty())
                {
                    sql << "DO NOTHING";
                }
                else
                {
                    sql << "DO UPDATE SET " << updateColumnList;
                }
            }
            return sql.str();
        }

        static std::string BatchUpdateSql(const TableDesc& tableDesc, const TableUpdateDesc& updateDesc)
        {
            std::ostringstream mergedRows;
            mergedRows << "SELECT target.rowid AS target_rowid";
            for (const char* pColumnName : updateDesc.updateColumns)
            {
                mergedRows << ", any_value(pending." << pColumnName << ") AS " << pColumnName;
            }
            mergedRows << " FROM " << AppendedRowsTableName << " AS pending " << (updateDesc.pLatestColumn ? "ASOF JOIN " : "JOIN ") << tableDesc.pName << " AS target ON ";
            for (size_t matchIndex = 0; matchIndex < updateDesc.matchColumns.size(); ++matchIndex)
            {
                const char* pColumnName = updateDesc.matchColumns[matchIndex];
                mergedRows << (matchIndex ? " AND " : "") << "target." << pColumnName << " = pending." << pColumnName;
            }
            if (updateDesc.pLatestColumn)
            {
                mergedRows << " AND pending." << updateDesc.pLatestColumn << " >= target." << updateDesc.pLatestColumn;
            }
            mergedRows << " GROUP BY target.rowid";

            std::ostringstream setColumns;
            std::ostringstream updatableRow;
            for (size_t updateColumnIndex = 0; updateColumnIndex < updateDesc.updateColumns.size(); ++updateColumnIndex)
            {
                const char* pColumnName = updateDesc.updateColumns[updateColumnIndex];
                setColumns << (updateColumnIndex ? ", " : "") << pColumnName << " = COALESCE(" << tableDesc.pName << '.' << pColumnName << ", merged." << pColumnName << ')';
                updatableRow << (updateColumnIndex ? " OR " : "") << '(' << tableDesc.pName << '.' << pColumnName << " IS NULL AND merged." << pColumnName << " IS NOT NULL)";
            }

            std::ostringstream sql;
            sql << "UPDATE " << tableDesc.pName << " SET " << setColumns.str() << " FROM (" << mergedRows.str() << ") AS merged WHERE " << tableDesc.pName << ".rowid = merged.target_rowid AND (" << updatableRow.str() << ')';
            return sql.str();
        }

        static std::string CreateViewSql(const TableViewDesc& tableViewDesc)
        {
            std::ostringstream sql;
            sql << "CREATE OR REPLACE VIEW " << tableViewDesc.pName << " AS " << tableViewDesc.pSelectSql;
            return sql.str();
        }

    private:
        static bool IsValidColumnValue(const TableDesc& tableDesc, const TableColumnDesc& columnDesc, const TableValue& value)
        {
            if (std::holds_alternative<std::monostate>(value))
            {
                NVTRACE_LOG_ERR_RETURN_IF(!columnDesc.IsNullable(), false, 50, "NULL in non-nullable column %s.%s", tableDesc.pName, columnDesc.pName);
            }
            else
            {
                NVTRACE_LOG_ERR_RETURN_IF(!IsTableValueOfType(value, columnDesc.type), false, 50, "Type mismatch for column %s.%s", tableDesc.pName, columnDesc.pName);
            }
            return true;
        }

        static const char* GetDuckDbTypeName(TableColumnType type)
        {
            switch (type)
            {
                case TableColumnType::Bool:
                    return "BOOLEAN";
                case TableColumnType::Int8:
                    return "TINYINT";
                case TableColumnType::Int32:
                    return "INTEGER";
                case TableColumnType::Int64:
                    return "BIGINT";
                case TableColumnType::UInt8:
                    return "UTINYINT";
                case TableColumnType::UInt16:
                    return "USMALLINT";
                case TableColumnType::UInt32:
                    return "UINTEGER";
                case TableColumnType::UInt64:
                    return "UBIGINT";
                case TableColumnType::Double:
                    return "DOUBLE";
                case TableColumnType::String:
                    return "VARCHAR";
                case TableColumnType::Uuid:
                    return "UUID";
            }
            return "VARCHAR";
        }

        static duckdb::LogicalType GetDuckDbLogicalType(TableColumnType type)
        {
            switch (type)
            {
                case TableColumnType::Bool:
                    return duckdb::LogicalType::BOOLEAN;
                case TableColumnType::Int8:
                    return duckdb::LogicalType::TINYINT;
                case TableColumnType::Int32:
                    return duckdb::LogicalType::INTEGER;
                case TableColumnType::Int64:
                    return duckdb::LogicalType::BIGINT;
                case TableColumnType::UInt8:
                    return duckdb::LogicalType::UTINYINT;
                case TableColumnType::UInt16:
                    return duckdb::LogicalType::USMALLINT;
                case TableColumnType::UInt32:
                    return duckdb::LogicalType::UINTEGER;
                case TableColumnType::UInt64:
                    return duckdb::LogicalType::UBIGINT;
                case TableColumnType::Double:
                    return duckdb::LogicalType::DOUBLE;
                case TableColumnType::String:
                    return duckdb::LogicalType::VARCHAR;
                case TableColumnType::Uuid:
                    return duckdb::LogicalType::UUID;
            }
            return duckdb::LogicalType::VARCHAR;
        }

        static std::string GetPrimaryKeyColumns(const TableDesc& tableDesc)
        {
            std::ostringstream columns;
            const char* pSeparator = "";
            for (const TableColumnDesc& columnDesc : tableDesc.columns)
            {
                if (columnDesc.IsPrimaryKey())
                {
                    columns << pSeparator << columnDesc.pName;
                    pSeparator = ", ";
                }
            }
            return columns.str();
        }

        // Unqualified column names refer to the existing row; excluded.* is the incoming row.
        static void WriteConflictUpdateSql(std::ostream& sql, std::string_view column, TableMergePolicy mergePolicy)
        {
            switch (mergePolicy)
            {
                case TableMergePolicy::Overwrite:
                    sql << "COALESCE(excluded." << column << ", " << column << ')';
                    return;
                case TableMergePolicy::CoalesceKeep:
                    sql << "COALESCE(" << column << ", excluded." << column << ')';
                    return;
                case TableMergePolicy::Keep:
                    sql << column;
                    return;
                case TableMergePolicy::Least:
                    sql << "CASE WHEN " << column << " IS NULL OR excluded." << column << " < " << column << " THEN excluded." << column << " ELSE " << column << " END";
                    return;
                case TableMergePolicy::Greatest:
                    sql << "CASE WHEN " << column << " IS NULL OR excluded." << column << " > " << column << " THEN excluded." << column << " ELSE " << column << " END";
                    return;
                case TableMergePolicy::LogicalOr:
                    sql << "CASE WHEN " << column << " IS NULL THEN excluded." << column << " WHEN excluded." << column << " IS NULL THEN " << column << " ELSE " << column << " OR excluded." << column << " END";
                    return;
            }
            sql << "excluded." << column;
        }

        static void AppendValue(duckdb::BaseAppender& appender, const TableValue& value)
        {
            if (const auto* pValue = std::get_if<bool>(&value))
            {
                appender.Append<bool>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<int8_t>(&value))
            {
                appender.Append<int8_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<int32_t>(&value))
            {
                appender.Append<int32_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<int64_t>(&value))
            {
                appender.Append<int64_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<uint8_t>(&value))
            {
                appender.Append<uint8_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<uint16_t>(&value))
            {
                appender.Append<uint16_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<uint32_t>(&value))
            {
                appender.Append<uint32_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<uint64_t>(&value))
            {
                appender.Append<uint64_t>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<double>(&value))
            {
                appender.Append<double>(*pValue);
                return;
            }
            if (const auto* pValue = std::get_if<std::string_view>(&value))
            {
                appender.Append(pValue->data(), static_cast<uint32_t>(pValue->size()));
                return;
            }
            if (const auto* pValue = std::get_if<NVTRACE_UUID>(&value))
            {
                appender.Append<duckdb::Value>(ToDuckDbUuid(*pValue));
            }
        }

        bool Execute(const std::string& sql)
        {
            if (!m_pConnection)
            {
                return false;
            }
            try
            {
                std::unique_ptr<duckdb::MaterializedQueryResult> pResult = m_pConnection->Query(sql);
                NVTRACE_LOG_ERR_RETURN_IF(pResult->HasError(), false, 50, "DuckDB query failed: %s\n  %s", pResult->GetError().c_str(), sql.c_str());
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "DuckDB query failed: %s\n  %s", e.what(), sql.c_str());
                return false;
            }
            return true;
        }

        // A new database gets the tables and is stamped with schemaVersion, a database at the same version
        // is used as is, one at an older version is upgraded, and one at a newer version is refused. The
        // database is left unchanged unless every step succeeds.
        bool PrepareSchema(const std::vector<TableDesc>& tableDescs, int32_t schemaVersion)
        {
            NVTRACE_LOG_ERR_RETURN_IF(!Execute("BEGIN TRANSACTION"), false, 50, "Failed to begin schema transaction");
            auto rollbackGuard = ScopeExitGuard([this]() { Execute("ROLLBACK"); });

            std::optional<int32_t> storedSchemaVersion;
            if (!Execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)") || !ReadStoredSchemaVersion(storedSchemaVersion))
            {
                return false;
            }

            if (storedSchemaVersion && *storedSchemaVersion != schemaVersion)
            {
                NVTRACE_LOG_ERR_RETURN_IF(*storedSchemaVersion > schemaVersion, false, 50, "Database schema version %d is newer than this exporter's schema version %d", *storedSchemaVersion, schemaVersion);
                if (!UpgradeSchema(*storedSchemaVersion, schemaVersion) || !Execute("UPDATE schema_version SET version = " + std::to_string(schemaVersion)))
                {
                    return false;
                }
            }

            for (const TableDesc& tableDesc : tableDescs)
            {
                if (!Execute(CreateTableSql(tableDesc)))
                {
                    return false;
                }
            }
            for (const TableViewDesc& tableViewDesc : GetTableViewDescs())
            {
                if (!Execute(CreateViewSql(tableViewDesc)))
                {
                    return false;
                }
            }
            if (!storedSchemaVersion && !Execute("INSERT INTO schema_version VALUES (" + std::to_string(schemaVersion) + ")"))
            {
                return false;
            }

            NVTRACE_LOG_ERR_RETURN_IF(!Execute("COMMIT"), false, 50, "Failed to commit schema transaction");
            rollbackGuard.Dismiss();
            m_tableDescs = tableDescs;
            return true;
        }

        // Runs inside the schema transaction.
        bool UpgradeSchema(int32_t fromSchemaVersion, int32_t toSchemaVersion)
        {
            // TODO: Implement once the schema changes after version 1.
            NVTRACE_LOG_ERR(50, "No upgrade path from database schema version %d to %d", fromSchemaVersion, toSchemaVersion);
            return false;
        }

        // Leaves out_schemaVersion empty for a database that has no schema yet.
        bool ReadStoredSchemaVersion(std::optional<int32_t>& out_schemaVersion)
        {
            try
            {
                std::unique_ptr<duckdb::MaterializedQueryResult> pResult = m_pConnection->Query("SELECT version FROM schema_version");
                NVTRACE_LOG_ERR_RETURN_IF(pResult->HasError(), false, 50, "Failed to read schema version: %s", pResult->GetError().c_str());
                const size_t rowCount = static_cast<size_t>(pResult->RowCount());
                NVTRACE_LOG_ERR_RETURN_IF(rowCount > 1, false, 50, "schema_version holds %zu rows, expected at most 1", rowCount);
                out_schemaVersion.reset();
                if (rowCount == 1)
                {
                    out_schemaVersion = pResult->GetValue(0, 0).GetValue<int32_t>();
                }
            }
            catch (const std::exception& e)
            {
                NVTRACE_LOG_ERR(50, "Failed to read schema version: %s", e.what());
                return false;
            }
            return true;
        }

        // NVTRACE_UUID bytes are in canonical RFC 9562 order, which is what FromBlob expects.
        static duckdb::Value ToDuckDbUuid(const NVTRACE_UUID& uuid)
        {
            return duckdb::Value::UUID(duckdb::BaseUUID::FromBlob(uuid.bytes));
        }

        std::unique_ptr<duckdb::DuckDB> m_pDatabase;
        std::unique_ptr<duckdb::Connection> m_pConnection;
        std::vector<TableDesc> m_tableDescs;
        std::vector<std::unique_ptr<duckdb::QueryAppender>> m_upsertAppenders;
        std::vector<std::unique_ptr<duckdb::QueryAppender>> m_updateAppenders;
        size_t m_upsertFlushThresholdBytes = DefaultUpsertFlushThresholdBytes;
        bool m_isSnapshotIngestionTransactionActive = false;
    };

} // namespace nv::trace
