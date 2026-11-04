/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceMathUtilities.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <locale>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv {
namespace trace {

    /// A small, dependency-free DOM codec for JSON and a conservative YAML 1.2 subset.
    /// YAML supports one document, string-key block mappings, block sequences, quoted and
    /// plain scalars, comments, and a complete JSON document as YAML flow syntax. Comments
    /// are discarded. Anchors, aliases, tags, directives, complex keys, nested flow syntax,
    /// and multiline block scalars are rejected.
    class YamlCodec
    {
    public:
        template <typename TValue>
        class Vector
        {
        public:
            /// Constructs an empty collection without allocating storage.
            Vector() = default;

            /// Destroys all live values and releases the contiguous backing allocation.
            ~Vector()
            {
                Clear();
                ::operator delete(m_pValues);
            }

            /// Vector is move-only because codec nodes are move-only.
            Vector(const Vector&) = delete;
            Vector& operator=(const Vector&) = delete;

            /// Transfers the backing allocation without moving individual values.
            Vector(Vector&& rhs) noexcept
                : m_pValues(rhs.m_pValues)
                , m_size(rhs.m_size)
                , m_capacity(rhs.m_capacity)
            {
                rhs.m_pValues = nullptr;
                rhs.m_size = 0;
                rhs.m_capacity = 0;
            }

            /// Releases the current contents, then transfers rhs's backing allocation.
            Vector& operator=(Vector&& rhs) noexcept
            {
                if (this != &rhs)
                {
                    Clear();
                    ::operator delete(m_pValues);
                    m_pValues = rhs.m_pValues;
                    m_size = rhs.m_size;
                    m_capacity = rhs.m_capacity;
                    rhs.m_pValues = nullptr;
                    rhs.m_size = 0;
                    rhs.m_capacity = 0;
                }
                return *this;
            }

            /// Returns the number of live values.
            size_t Size() const
            {
                return m_size;
            }

            /// Returns true when there are no live values.
            bool Empty() const
            {
                return m_size == 0;
            }

            /// Returns the number of values that fit without reallocating.
            size_t Capacity() const
            {
                return m_capacity;
            }

            /// Returns the value at valueIndex. Precondition: valueIndex < Size().
            TValue& operator[](size_t valueIndex)
            {
                return m_pValues[valueIndex];
            }

            /// Returns the value at valueIndex. Precondition: valueIndex < Size().
            const TValue& operator[](size_t valueIndex) const
            {
                return m_pValues[valueIndex];
            }

            /// Returns the last value. Precondition: !Empty().
            TValue& Back()
            {
                return m_pValues[m_size - 1];
            }

            /// Returns the last value. Precondition: !Empty().
            const TValue& Back() const
            {
                return m_pValues[m_size - 1];
            }

            /// Returns the first value, or nullptr when no backing allocation exists.
            TValue* begin()
            {
                return m_pValues;
            }

            /// Returns the first value, or nullptr when no backing allocation exists.
            const TValue* begin() const
            {
                return m_pValues;
            }

            /// Returns the iterator sentinel for the live values.
            TValue* end()
            {
                return m_pValues == nullptr ? nullptr : m_pValues + m_size;
            }

            /// Returns the iterator sentinel for the live values.
            const TValue* end() const
            {
                return m_pValues == nullptr ? nullptr : m_pValues + m_size;
            }

            /// Ensures capacity without changing Size(). Reallocation invalidates all references and pointers.
            void Reserve(size_t capacity)
            {
                if (capacity <= m_capacity)
                {
                    return;
                }

                static_assert(std::is_nothrow_move_constructible<TValue>::value, "YamlCodec::Vector values must have a nonthrowing move constructor");
                TValue* pValues = static_cast<TValue*>(::operator new(sizeof(TValue) * capacity));
                for (size_t valueIndex = 0; valueIndex < m_size; ++valueIndex)
                {
                    new (pValues + valueIndex) TValue(std::move(m_pValues[valueIndex]));
                }
                for (size_t valueIndex = 0; valueIndex < m_size; ++valueIndex)
                {
                    m_pValues[valueIndex].~TValue();
                }
                ::operator delete(m_pValues);
                m_pValues = pValues;
                m_capacity = capacity;
            }

            /// Appends a default-constructed value and returns it. Growth invalidates all references and pointers.
            TValue& PushBackDefault()
            {
                GrowForOne();
                new (m_pValues + m_size) TValue();
                ++m_size;
                return Back();
            }

            /// Move-appends value. Growth invalidates all references and pointers.
            void PushBack(TValue&& value)
            {
                GrowForOne();
                new (m_pValues + m_size) TValue(std::move(value));
                ++m_size;
            }

            /// Destroys all live values while retaining the backing allocation.
            void Clear() noexcept
            {
                while (0 < m_size)
                {
                    --m_size;
                    m_pValues[m_size].~TValue();
                }
            }

        private:
            void GrowForOne()
            {
                if (m_size < m_capacity)
                {
                    return;
                }
                const size_t capacity = m_capacity == 0 ? 4 : m_capacity * 2;
                Reserve(capacity);
            }

        private:
            TValue* m_pValues = nullptr;
            size_t m_size = 0;
            size_t m_capacity = 0;
        };

        /// Discriminates which NodeValue fields contain meaningful data.
        enum class NodeType
        {
            Null,
            Boolean,
            SignedInteger,
            UnsignedInteger,
            FloatingPoint,
            String,
            Sequence,
            Mapping,
        };

        class DataNode;
        struct SequenceEntry;

        /// Directly readable storage for every node kind. DataNode owns all mutation.
        struct NodeValue
        {
            NodeType nodeType = NodeType::Null;
            union
            {
                int64_t s64;
                uint64_t u64;
                double f64;
            };
            std::string str;
            Vector<SequenceEntry> entries;

            /// Constructs a Null value with zeroed scalar storage.
            NodeValue()
                : s64(0)
            {
            }

            /// Node values are move-only together with their recursively owned entries.
            NodeValue(const NodeValue&) = delete;
            NodeValue& operator=(const NodeValue&) = delete;
            NodeValue(NodeValue&&) noexcept = default;
            NodeValue& operator=(NodeValue&&) noexcept = default;

        private:
            std::vector<size_t> m_mappingIndices;
            friend class DataNode;
            friend class YamlCodec;
        };

        class DataNode
        {
        public:
            /// Constructs a Null node.
            DataNode() = default;
            ~DataNode() = default;

            /// Nodes are move-only; moving transfers their complete owned subtree.
            DataNode(const DataNode&) = delete;
            DataNode& operator=(const DataNode&) = delete;
            DataNode(DataNode&&) noexcept = default;
            DataNode& operator=(DataNode&&) noexcept = default;

            /// Destroys all children and resets the node to Null while retaining reusable capacities.
            void Reset()
            {
                m_value.entries.Clear();
                m_value.m_mappingIndices.clear();
                m_value.str.clear();
                m_value.s64 = 0;
                m_value.nodeType = NodeType::Null;
            }

            /// Returns direct, read-only access to the node discriminator and stored value.
            const NodeValue& GetValue() const
            {
                return m_value;
            }

            /// Converts the node to Null and discards its previous value.
            void SetNull()
            {
                Reset();
            }

            /// Converts the node to Boolean and stores a normalized zero or one in NodeValue::s64.
            void SetBoolean(bool value)
            {
                ResetTo(NodeType::Boolean);
                m_value.s64 = value ? 1 : 0;
            }

            /// Converts the node to SignedInteger.
            void SetInt64(int64_t value)
            {
                ResetTo(NodeType::SignedInteger);
                m_value.s64 = value;
            }

            /// Stores values through INT64_MAX as SignedInteger and larger values as UnsignedInteger.
            void SetUInt64(uint64_t value)
            {
                if (value <= static_cast<uint64_t>(INT64_MAX))
                {
                    SetInt64(static_cast<int64_t>(value));
                    return;
                }
                ResetTo(NodeType::UnsignedInteger);
                m_value.u64 = value;
            }

            /// Converts the node to FloatingPoint. Writers reject non-finite values.
            void SetDouble(double value)
            {
                ResetTo(NodeType::FloatingPoint);
                m_value.f64 = value;
            }

            /// Converts the node to String and copies value.
            void SetString(const std::string& value)
            {
                ResetTo(NodeType::String);
                m_value.str = value;
            }

            /// Converts the node to String and moves value.
            void SetString(std::string&& value)
            {
                ResetTo(NodeType::String);
                m_value.str = std::move(value);
            }

            /// Converts a non-sequence node to an empty Sequence, then invokes updateSequence synchronously.
            /// An existing Sequence retains its entries. The callback receives direct mutable access to entries;
            /// keys are inert for Sequence nodes. Appending may invalidate earlier entry references and pointers.
            template <typename TUpdateSequence>
            void UpdateSequence(TUpdateSequence&& updateSequence)
            {
                ResetTo(NodeType::Sequence);
                updateSequence(m_value.entries);
            }

            /// Inserts a Null mapping value by copying key. Returns false for a duplicate key.
            /// On success, GetValue().entries.Back() is the new insertion-order entry.
            bool MapInsert(const std::string& key)
            {
                std::string copiedKey = key;
                return MapInsert(std::move(copiedKey));
            }

            /// Inserts a Null mapping value by moving key. Returns false for a duplicate key.
            /// A non-Mapping node first becomes an empty Mapping. Successful insertion may invalidate entry pointers.
            bool MapInsert(std::string&& key)
            {
                ResetTo(NodeType::Mapping);
                const SequenceEntry* pExistingEntry = MapFind(key);
                if (pExistingEntry != nullptr)
                {
                    return false;
                }

                const size_t entryIndex = m_value.entries.Size();
                m_value.m_mappingIndices.reserve(entryIndex + 1);
                SequenceEntry& entry = m_value.entries.PushBackDefault();
                entry.key = std::move(key);
                m_value.m_mappingIndices.push_back(entryIndex);
                MergeMappingIndexRuns();
                return true;
            }

            /// Inserts value by copying key. Returns false without consuming value for a duplicate key.
            /// On success, GetValue().entries.Back() is the new insertion-order entry.
            bool MapInsert(const std::string& key, DataNode&& value)
            {
                std::string copiedKey = key;
                return MapInsert(std::move(copiedKey), std::move(value));
            }

            /// Inserts value by moving key. Returns false without consuming value for a duplicate key.
            /// A non-Mapping node first becomes an empty Mapping. Successful insertion may invalidate entry pointers.
            bool MapInsert(std::string&& key, DataNode&& value)
            {
                const bool inserted = MapInsert(std::move(key));
                if (inserted)
                {
                    m_value.entries.Back().value = std::move(value);
                }
                return inserted;
            }

            /// Finds a null-terminated mapping key, or returns nullptr. The result is invalidated by mapping insertion.
            const SequenceEntry* MapFind(const char* pKey) const
            {
                if (pKey == nullptr)
                {
                    return nullptr;
                }
                return MapFind(pKey, std::strlen(pKey));
            }

            /// Finds a mapping key, including embedded nulls, or returns nullptr. The result is invalidated by mapping insertion.
            const SequenceEntry* MapFind(const std::string& key) const
            {
                return MapFind(key.data(), key.size());
            }

        private:
            const SequenceEntry* MapFind(const char* pKey, size_t keySize) const
            {
                if (m_value.nodeType != NodeType::Mapping)
                {
                    return nullptr;
                }

                size_t runOffset = 0;
                size_t remainingCount = m_value.m_mappingIndices.size();
                size_t runSize = 0;
                if (remainingCount != 0)
                {
                    const uint32_t runSizeLog2 = CeilLog2(static_cast<uint64_t>(remainingCount));
                    runSize = static_cast<size_t>(1) << runSizeLog2;
                    if (remainingCount < runSize)
                    {
                        runSize /= 2;
                    }
                }
                while (0 < runSize)
                {
                    if (runSize <= remainingCount)
                    {
                        struct KeyView
                        {
                            const char* pData;
                            size_t size;
                        };
                        const KeyView keyView{pKey, keySize};
                        const auto runBegin = m_value.m_mappingIndices.begin() + runOffset;
                        const auto runEnd = runBegin + runSize;
                        auto compareEntryToKey = [&](size_t entryIndex, const KeyView& rhs) {
                            return m_value.entries[entryIndex].key.compare(0, std::string::npos, rhs.pData, rhs.size) < 0;
                        };
                        const auto itrEntryIndex = std::lower_bound(runBegin, runEnd, keyView, compareEntryToKey);
                        if (itrEntryIndex != runEnd)
                        {
                            const size_t entryIndex = *itrEntryIndex;
                            if (m_value.entries[entryIndex].key.compare(0, std::string::npos, pKey, keySize) == 0)
                            {
                                return &m_value.entries[entryIndex];
                            }
                        }
                        runOffset += runSize;
                        remainingCount -= runSize;
                    }
                    runSize /= 2;
                }
                return nullptr;
            }

            void ResetTo(NodeType nodeType)
            {
                if (m_value.nodeType == nodeType)
                {
                    return;
                }
                Reset();
                m_value.nodeType = nodeType;
            }

            void MergeMappingIndexRuns()
            {
                size_t completedEntryCount = m_value.m_mappingIndices.size() - 1;
                size_t runSize = 1;
                while ((completedEntryCount & runSize) != 0)
                {
                    const size_t runBegin = m_value.m_mappingIndices.size() - runSize * 2;
                    const size_t runMiddle = runBegin + runSize;
                    std::inplace_merge(
                        m_value.m_mappingIndices.begin() + runBegin,
                        m_value.m_mappingIndices.begin() + runMiddle,
                        m_value.m_mappingIndices.end(),
                        [&](size_t lhsIndex, size_t rhsIndex) {
                            return m_value.entries[lhsIndex].key < m_value.entries[rhsIndex].key;
                        });
                    runSize *= 2;
                }
            }

        private:
            NodeValue m_value;
            friend class YamlCodec;
        };

        /// One insertion-order collection entry. key is used by Mapping and inert for Sequence.
        struct SequenceEntry
        {
            std::string key;
            DataNode value;

            /// Constructs an entry with an empty key and Null value.
            SequenceEntry() = default;
            /// Entries are move-only together with their value subtrees.
            SequenceEntry(const SequenceEntry&) = delete;
            SequenceEntry& operator=(const SequenceEntry&) = delete;
            SequenceEntry(SequenceEntry&&) noexcept = default;
            SequenceEntry& operator=(SequenceEntry&&) noexcept = default;
        };

        /// Bounds applied while reading one document.
        struct ReadOptions
        {
            /// Approximate retained node, entry, mapping-index, and string bytes. Zero is unlimited.
            size_t maxMemory = 1024 * 1024;
            /// Maximum nested container count. Zero permits scalar documents only.
            size_t maxStackDepth = 64;
        };

        /// Bounds and presentation options applied while writing one document.
        struct WriteOptions
        {
            /// Exact maximum rendered byte count. Zero is unlimited.
            size_t maxOutputLength = 1024 * 1024;
            /// Maximum nested container count. Zero permits scalar documents only.
            size_t maxStackDepth = 64;
            /// YAML spaces per nesting level and pretty JSON indentation width.
            size_t indentSize = 2;
            /// Emits JSON containers with newlines and indentSize spaces per level.
            bool prettyJson = false;
            /// Keeps the first root mapping key adjacent to '{' in pretty JSON for prefix-sensitive formats.
            bool prettyJsonFirstRootKeyInline = false;
        };

        /// Optional operation diagnostic. Reader failures identify a zero-based byte offset and one-based line and column.
        /// Writer failures set message; their line and column remain at the reset defaults because no input location applies.
        struct Error
        {
            size_t offset = 0;
            size_t line = 1;
            size_t column = 1;
            std::string message;

            /// Restores the empty diagnostic at byte zero, line one, column one.
            void Reset()
            {
                offset = 0;
                line = 1;
                column = 1;
                message.clear();
            }
        };

        /// Strictly parses one UTF-8 JSON document. pRoot is reset before parsing and remains Null on failure.
        /// When supplied, pError receives the failure message and original-input byte, line, and column.
        static bool ReadJson(const uint8_t* pData, size_t size, DataNode* pRoot, const ReadOptions& options, Error* pError = nullptr)
        {
            Reader reader(pData, size, options, pError);
            return reader.ReadJson(pRoot);
        }

        /// Strictly parses one UTF-8 JSON document from input; otherwise matches the byte-buffer overload.
        static bool ReadJson(const std::string& input, DataNode* pRoot, const ReadOptions& options, Error* pError = nullptr)
        {
            return ReadJson(reinterpret_cast<const uint8_t*>(input.data()), input.size(), pRoot, options, pError);
        }

        /// Parses one UTF-8 document in the YAML subset described on YamlCodec.
        /// pRoot is reset before parsing and remains Null on failure. pError uses original-input coordinates.
        static bool ReadYaml(const uint8_t* pData, size_t size, DataNode* pRoot, const ReadOptions& options, Error* pError = nullptr)
        {
            Reader reader(pData, size, options, pError);
            return reader.ReadYaml(pRoot);
        }

        /// Parses one UTF-8 YAML-subset document from input; otherwise matches the byte-buffer overload.
        static bool ReadYaml(const std::string& input, DataNode* pRoot, const ReadOptions& options, Error* pError = nullptr)
        {
            return ReadYaml(reinterpret_cast<const uint8_t*>(input.data()), input.size(), pRoot, options, pError);
        }

        /// Renders strict JSON, then synchronously calls sink once as bool(const uint8_t*, size_t).
        /// Returns false for rendering limits, invalid node values, or sink failure.
        template <typename TSink>
        static bool WriteJson(const DataNode& root, const WriteOptions& options, TSink&& sink, Error* pError = nullptr)
        {
            Renderer renderer(options, pError);
            const bool rendered = renderer.RenderJson(root);
            if (!rendered)
            {
                return false;
            }
            const bool submitted = sink(reinterpret_cast<const uint8_t*>(renderer.GetOutput().data()), renderer.GetOutput().size());
            if (!submitted && pError != nullptr)
            {
                pError->message = "output sink failed";
            }
            return submitted;
        }

        /// Renders the supported YAML subset, then synchronously calls sink once as bool(const uint8_t*, size_t).
        /// Returns false for rendering limits, invalid node values, or sink failure.
        template <typename TSink>
        static bool WriteYaml(const DataNode& root, const WriteOptions& options, TSink&& sink, Error* pError = nullptr)
        {
            Renderer renderer(options, pError);
            const bool rendered = renderer.RenderYaml(root);
            if (!rendered)
            {
                return false;
            }
            const bool submitted = sink(reinterpret_cast<const uint8_t*>(renderer.GetOutput().data()), renderer.GetOutput().size());
            if (!submitted && pError != nullptr)
            {
                pError->message = "output sink failed";
            }
            return submitted;
        }

    private:
        class Reader
        {
        public:
            Reader(const uint8_t* pData, size_t size, const ReadOptions& options, Error* pError)
                : m_pData(pData)
                , m_size(size)
                , m_options(options)
                , m_pError(pError)
            {
                if (m_pError != nullptr)
                {
                    m_pError->Reset();
                }
            }

            bool ReadJson(DataNode* pRoot)
            {
                ResetRoot(pRoot);
                if (pRoot == nullptr || (m_pData == nullptr && m_size != 0))
                {
                    return Fail("invalid input or output pointer");
                }
                const bool utf8Valid = ValidateUtf8();
                if (!utf8Valid)
                {
                    return false;
                }
                const bool rootAccounted = Account(sizeof(DataNode));
                if (!rootAccounted)
                {
                    return false;
                }

                SkipJsonWhitespace();
                const bool valueParsed = ParseJsonValue(pRoot);
                if (!valueParsed)
                {
                    pRoot->Reset();
                    return false;
                }

                while (!m_jsonFrames.empty())
                {
                    JsonFrame& frame = m_jsonFrames.back();
                    SkipJsonWhitespace();
                    if (frame.isMapping)
                    {
                        if (frame.expectEntry)
                        {
                            const bool mappingEnded = frame.allowEnd && Consume('}');
                            if (mappingEnded)
                            {
                                m_jsonFrames.pop_back();
                                continue;
                            }
                            std::string key;
                            const bool keyParsed = ParseJsonString(&key);
                            if (!keyParsed)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            const bool mappingEntryAccounted = Account(key.size() + sizeof(SequenceEntry) + sizeof(size_t));
                            if (!mappingEntryAccounted)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            SkipJsonWhitespace();
                            const bool colonConsumed = Consume(':');
                            if (!colonConsumed)
                            {
                                pRoot->Reset();
                                return Fail("expected ':' after mapping key");
                            }
                            const bool inserted = frame.pNode->MapInsert(std::move(key));
                            if (!inserted)
                            {
                                pRoot->Reset();
                                return Fail("duplicate mapping key");
                            }
                            frame.expectEntry = false;
                            DataNode* pValue = &frame.pNode->m_value.entries.Back().value;
                            SkipJsonWhitespace();
                            const bool childParsed = ParseJsonValue(pValue);
                            if (!childParsed)
                            {
                                pRoot->Reset();
                                return false;
                            }
                        }
                        else
                        {
                            const bool mappingEnded = Consume('}');
                            if (mappingEnded)
                            {
                                m_jsonFrames.pop_back();
                                continue;
                            }
                            const bool commaConsumed = Consume(',');
                            if (!commaConsumed)
                            {
                                pRoot->Reset();
                                return Fail("expected ',' or '}'");
                            }
                            frame.expectEntry = true;
                            frame.allowEnd = false;
                        }
                    }
                    else if (frame.expectEntry)
                    {
                        const bool sequenceEnded = frame.allowEnd && Consume(']');
                        if (sequenceEnded)
                        {
                            m_jsonFrames.pop_back();
                            continue;
                        }
                        const bool entryAccounted = Account(sizeof(SequenceEntry));
                        if (!entryAccounted)
                        {
                            pRoot->Reset();
                            return false;
                        }
                        SequenceEntry& entry = frame.pNode->m_value.entries.PushBackDefault();
                        frame.expectEntry = false;
                        const bool childParsed = ParseJsonValue(&entry.value);
                        if (!childParsed)
                        {
                            pRoot->Reset();
                            return false;
                        }
                    }
                    else
                    {
                        const bool sequenceEnded = Consume(']');
                        if (sequenceEnded)
                        {
                            m_jsonFrames.pop_back();
                            continue;
                        }
                        const bool commaConsumed = Consume(',');
                        if (!commaConsumed)
                        {
                            pRoot->Reset();
                            return Fail("expected ',' or ']'");
                        }
                        frame.expectEntry = true;
                        frame.allowEnd = false;
                    }
                }

                SkipJsonWhitespace();
                if (m_offset != m_size)
                {
                    pRoot->Reset();
                    return Fail("unexpected content after document");
                }
                return true;
            }

            bool ReadYaml(DataNode* pRoot)
            {
                ResetRoot(pRoot);
                if (pRoot == nullptr || (m_pData == nullptr && m_size != 0))
                {
                    return Fail("invalid input or output pointer");
                }
                const bool utf8Valid = ValidateUtf8();
                if (!utf8Valid)
                {
                    return false;
                }

                size_t firstOffset = 0;
                while (firstOffset < m_size && std::isspace(m_pData[firstOffset]) != 0)
                {
                    ++firstOffset;
                }
                if (firstOffset < m_size && (m_pData[firstOffset] == '{' || m_pData[firstOffset] == '['))
                {
                    return ReadJson(pRoot);
                }

                const bool rootAccounted = Account(sizeof(DataNode));
                if (!rootAccounted)
                {
                    return false;
                }
                std::vector<YamlLine> lines;
                const bool linesTokenized = TokenizeYamlLines(&lines);
                if (!linesTokenized)
                {
                    return false;
                }
                if (lines.empty())
                {
                    pRoot->SetNull();
                    return true;
                }
                std::vector<YamlFrame> frames;
                for (size_t lineIndex = 0; lineIndex < lines.size(); ++lineIndex)
                {
                    const YamlLine& line = lines[lineIndex];
                    while (!frames.empty() && line.indent <= frames.back().indent)
                    {
                        frames.pop_back();
                    }

                    if (frames.empty())
                    {
                        if (line.indent != 0)
                        {
                            pRoot->Reset();
                            return FailAt(line.offset, "root YAML content must not be indented");
                        }
                    }
                    else if (frames.back().childIndent == InvalidIndent)
                    {
                        frames.back().childIndent = line.indent;
                    }
                    else if (line.indent != frames.back().childIndent)
                    {
                        pRoot->Reset();
                        return FailAt(line.offset, "inconsistent YAML indentation");
                    }

                    DataNode* pParent = frames.empty() ? pRoot : frames.back().pNode;
                    const bool sequenceLine = (line.text.size() == 1 && line.text[0] == '-') || (1 < line.text.size() && line.text[0] == '-' && std::isspace(static_cast<unsigned char>(line.text[1])) != 0);
                    if (sequenceLine)
                    {
                        if (pParent->GetValue().nodeType == NodeType::Null)
                        {
                            const bool rootDepthValid = pParent != pRoot || CheckDepth(1);
                            if (!rootDepthValid)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            pParent->ResetTo(NodeType::Sequence);
                        }
                        if (pParent->GetValue().nodeType != NodeType::Sequence)
                        {
                            pRoot->Reset();
                            return FailAt(line.offset, "sequence item has a non-sequence parent");
                        }
                        const bool entryAccounted = Account(sizeof(SequenceEntry));
                        if (!entryAccounted)
                        {
                            pRoot->Reset();
                            return false;
                        }
                        SequenceEntry& entry = pParent->m_value.entries.PushBackDefault();
                        size_t contentOffset = 1;
                        while (contentOffset < line.text.size() && line.text[contentOffset] == ' ')
                        {
                            ++contentOffset;
                        }
                        const std::string valueText = Trim(line.text.substr(contentOffset));
                        if (valueText.empty())
                        {
                            frames.push_back(YamlFrame{line.indent, InvalidIndent, &entry.value});
                            const bool depthValid = CheckDepth(frames.size() + 1);
                            if (!depthValid)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            continue;
                        }

                        const size_t colonOffset = FindYamlColon(valueText);
                        if (colonOffset != std::string::npos)
                        {
                            entry.value.ResetTo(NodeType::Mapping);
                            const std::string keyText = Trim(valueText.substr(0, colonOffset));
                            std::string key;
                            const bool keyParsed = ParseYamlKey(keyText, &key, line.offset + contentOffset);
                            if (!keyParsed)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            const bool mappingEntryAccounted = Account(key.size() + sizeof(SequenceEntry) + sizeof(size_t));
                            if (!mappingEntryAccounted)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            const bool inserted = entry.value.MapInsert(std::move(key));
                            if (!inserted)
                            {
                                pRoot->Reset();
                                return FailAt(line.offset, "duplicate mapping key");
                            }
                            DataNode* pValue = &entry.value.m_value.entries.Back().value;
                            size_t scalarOffset = colonOffset + 1;
                            while (scalarOffset < valueText.size() && valueText[scalarOffset] == ' ')
                            {
                                ++scalarOffset;
                            }
                            const std::string scalarText = Trim(valueText.substr(scalarOffset));
                            const bool scalarParsed = scalarText.empty() || ParseYamlScalar(scalarText, pValue, line.offset + contentOffset + scalarOffset);
                            if (!scalarParsed)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            frames.push_back(YamlFrame{line.indent, line.indent + contentOffset, &entry.value});
                            const bool mappingDepthValid = CheckDepth(frames.size() + 1);
                            if (!mappingDepthValid)
                            {
                                pRoot->Reset();
                                return false;
                            }
                            if (scalarText.empty())
                            {
                                frames.push_back(YamlFrame{line.indent + contentOffset, InvalidIndent, pValue});
                                const bool valueDepthValid = CheckDepth(frames.size() + 1);
                                if (!valueDepthValid)
                                {
                                    pRoot->Reset();
                                    return false;
                                }
                            }
                            continue;
                        }
                        const bool scalarParsed = ParseYamlScalar(valueText, &entry.value, line.offset + contentOffset);
                        if (!scalarParsed)
                        {
                            pRoot->Reset();
                            return false;
                        }
                        continue;
                    }

                    const size_t colonOffset = FindYamlColon(line.text);
                    if (colonOffset == std::string::npos)
                    {
                        if (lineIndex != 0)
                        {
                            pRoot->Reset();
                            return FailAt(line.offset, "expected a YAML mapping or sequence");
                        }
                        const bool scalarParsed = ParseYamlScalar(line.text, pRoot, line.offset);
                        if (!scalarParsed)
                        {
                            pRoot->Reset();
                            return false;
                        }
                        continue;
                    }
                    if (pParent->GetValue().nodeType == NodeType::Null)
                    {
                        const bool rootDepthValid = pParent != pRoot || CheckDepth(1);
                        if (!rootDepthValid)
                        {
                            pRoot->Reset();
                            return false;
                        }
                        pParent->ResetTo(NodeType::Mapping);
                    }
                    if (pParent->GetValue().nodeType != NodeType::Mapping)
                    {
                        pRoot->Reset();
                        return FailAt(line.offset, "mapping item has a non-mapping parent");
                    }

                    std::string key;
                    const std::string keyText = Trim(line.text.substr(0, colonOffset));
                    const bool keyParsed = ParseYamlKey(keyText, &key, line.offset);
                    if (!keyParsed)
                    {
                        pRoot->Reset();
                        return false;
                    }
                    const bool entryAccounted = Account(key.size() + sizeof(SequenceEntry) + sizeof(size_t));
                    if (!entryAccounted)
                    {
                        pRoot->Reset();
                        return false;
                    }
                    const bool inserted = pParent->MapInsert(std::move(key));
                    if (!inserted)
                    {
                        pRoot->Reset();
                        return FailAt(line.offset, "duplicate mapping key");
                    }
                    DataNode* pValue = &pParent->m_value.entries.Back().value;
                    size_t valueOffset = colonOffset + 1;
                    while (valueOffset < line.text.size() && line.text[valueOffset] == ' ')
                    {
                        ++valueOffset;
                    }
                    const std::string valueText = Trim(line.text.substr(valueOffset));
                    if (valueText.empty())
                    {
                        frames.push_back(YamlFrame{line.indent, InvalidIndent, pValue});
                        const bool depthValid = CheckDepth(frames.size() + 1);
                        if (!depthValid)
                        {
                            pRoot->Reset();
                            return false;
                        }
                    }
                    else
                    {
                        const bool scalarParsed = ParseYamlScalar(valueText, pValue, line.offset + valueOffset);
                        if (!scalarParsed)
                        {
                            pRoot->Reset();
                            return false;
                        }
                    }
                }
                return true;
            }

        private:
            struct JsonFrame
            {
                DataNode* pNode = nullptr;
                bool isMapping = false;
                bool expectEntry = true;
                bool allowEnd = true;
            };

            struct YamlLine
            {
                size_t indent = 0;
                size_t offset = 0;
                std::string text;
            };

            struct YamlFrame
            {
                size_t indent = 0;
                size_t childIndent = InvalidIndent;
                DataNode* pNode = nullptr;
            };

            static constexpr size_t InvalidIndent = static_cast<size_t>(-1);

            static void ResetRoot(DataNode* pRoot)
            {
                if (pRoot != nullptr)
                {
                    pRoot->Reset();
                }
            }

            bool ValidateUtf8()
            {
                size_t charOffset = 0;
                while (charOffset < m_size)
                {
                    const uint8_t firstByte = m_pData[charOffset];
                    if (firstByte <= 0x7F)
                    {
                        ++charOffset;
                        continue;
                    }

                    size_t continuationCount = 0;
                    uint8_t secondByteLowerBound = 0x80;
                    uint8_t secondByteUpperBound = 0xBF;
                    if (0xC2 <= firstByte && firstByte <= 0xDF)
                    {
                        continuationCount = 1;
                    }
                    else if (firstByte == 0xE0)
                    {
                        continuationCount = 2;
                        secondByteLowerBound = 0xA0;
                    }
                    else if ((0xE1 <= firstByte && firstByte <= 0xEC) || (0xEE <= firstByte && firstByte <= 0xEF))
                    {
                        continuationCount = 2;
                    }
                    else if (firstByte == 0xED)
                    {
                        continuationCount = 2;
                        secondByteUpperBound = 0x9F;
                    }
                    else if (firstByte == 0xF0)
                    {
                        continuationCount = 3;
                        secondByteLowerBound = 0x90;
                    }
                    else if (0xF1 <= firstByte && firstByte <= 0xF3)
                    {
                        continuationCount = 3;
                    }
                    else if (firstByte == 0xF4)
                    {
                        continuationCount = 3;
                        secondByteUpperBound = 0x8F;
                    }
                    else
                    {
                        return FailAt(charOffset, "invalid UTF-8 leading byte");
                    }

                    if (m_size - charOffset <= continuationCount)
                    {
                        return FailAt(charOffset, "incomplete UTF-8 sequence");
                    }
                    const uint8_t secondByte = m_pData[charOffset + 1];
                    if (secondByte < secondByteLowerBound || secondByteUpperBound < secondByte)
                    {
                        return FailAt(charOffset + 1, "invalid UTF-8 continuation byte");
                    }
                    for (size_t continuationIndex = 2; continuationIndex <= continuationCount; ++continuationIndex)
                    {
                        const uint8_t continuationByte = m_pData[charOffset + continuationIndex];
                        if (continuationByte < 0x80 || 0xBF < continuationByte)
                        {
                            return FailAt(charOffset + continuationIndex, "invalid UTF-8 continuation byte");
                        }
                    }
                    charOffset += continuationCount + 1;
                }
                return true;
            }

            bool ParseJsonValue(DataNode* pNode)
            {
                if (m_size <= m_offset)
                {
                    return Fail("expected a value");
                }
                const uint8_t value = m_pData[m_offset];
                if (value == '{' || value == '[')
                {
                    ++m_offset;
                    pNode->ResetTo(value == '{' ? NodeType::Mapping : NodeType::Sequence);
                    const bool depthValid = CheckDepth(m_jsonFrames.size() + 1);
                    if (!depthValid)
                    {
                        return false;
                    }
                    m_jsonFrames.push_back(JsonFrame{pNode, value == '{', true, true});
                    return true;
                }
                if (value == '"')
                {
                    std::string str;
                    const bool stringParsed = ParseJsonString(&str);
                    if (!stringParsed)
                    {
                        return false;
                    }
                    const bool stringAccounted = Account(str.size());
                    if (!stringAccounted)
                    {
                        return false;
                    }
                    pNode->SetString(std::move(str));
                    return true;
                }
                const bool trueMatched = Match("true");
                if (trueMatched)
                {
                    pNode->SetBoolean(true);
                    return true;
                }
                const bool falseMatched = Match("false");
                if (falseMatched)
                {
                    pNode->SetBoolean(false);
                    return true;
                }
                const bool nullMatched = Match("null");
                if (nullMatched)
                {
                    pNode->SetNull();
                    return true;
                }
                return ParseNumber(pNode, true);
            }

            bool ParseNumber(DataNode* pNode, bool strictJson)
            {
                const size_t numberOffset = m_offset;
                if (m_offset < m_size && m_pData[m_offset] == '-')
                {
                    ++m_offset;
                }
                const size_t integerOffset = m_offset;
                while (m_offset < m_size && std::isdigit(m_pData[m_offset]) != 0)
                {
                    ++m_offset;
                }
                if (integerOffset == m_offset)
                {
                    m_offset = numberOffset;
                    return Fail("expected a value");
                }
                if (strictJson && m_pData[integerOffset] == '0' && integerOffset + 1 < m_offset)
                {
                    return FailAt(numberOffset, "leading zero in JSON number");
                }

                bool floatingPoint = false;
                if (m_offset < m_size && m_pData[m_offset] == '.')
                {
                    floatingPoint = true;
                    ++m_offset;
                    const size_t fractionOffset = m_offset;
                    while (m_offset < m_size && std::isdigit(m_pData[m_offset]) != 0)
                    {
                        ++m_offset;
                    }
                    if (fractionOffset == m_offset)
                    {
                        return Fail("expected digits after decimal point");
                    }
                }
                if (m_offset < m_size && (m_pData[m_offset] == 'e' || m_pData[m_offset] == 'E'))
                {
                    floatingPoint = true;
                    ++m_offset;
                    if (m_offset < m_size && (m_pData[m_offset] == '+' || m_pData[m_offset] == '-'))
                    {
                        ++m_offset;
                    }
                    const size_t exponentOffset = m_offset;
                    while (m_offset < m_size && std::isdigit(m_pData[m_offset]) != 0)
                    {
                        ++m_offset;
                    }
                    if (exponentOffset == m_offset)
                    {
                        return Fail("expected exponent digits");
                    }
                }

                const std::string number(reinterpret_cast<const char*>(m_pData + numberOffset), m_offset - numberOffset);
                if (floatingPoint)
                {
                    std::istringstream stream(number);
                    stream.imbue(std::locale::classic());
                    double value = 0.0;
                    stream >> value;
                    const bool finite = std::isfinite(value);
                    if (stream.fail() || !stream.eof() || !finite)
                    {
                        return FailAt(numberOffset, "floating-point number is out of range");
                    }
                    pNode->SetDouble(value);
                    return true;
                }

                const bool negative = !number.empty() && number[0] == '-';
                const size_t digitOffset = negative ? 1 : 0;
                const uint64_t limit = negative ? static_cast<uint64_t>(INT64_MAX) + 1 : UINT64_MAX;
                uint64_t magnitude = 0;
                for (size_t digitIndex = digitOffset; digitIndex < number.size(); ++digitIndex)
                {
                    const uint64_t digit = static_cast<uint64_t>(number[digitIndex] - '0');
                    if ((limit - digit) / 10 < magnitude)
                    {
                        return FailAt(numberOffset, negative ? "signed integer is out of range" : "unsigned integer is out of range");
                    }
                    magnitude = magnitude * 10 + digit;
                }
                if (negative)
                {
                    const int64_t value = magnitude == static_cast<uint64_t>(INT64_MAX) + 1 ? INT64_MIN : -static_cast<int64_t>(magnitude);
                    pNode->SetInt64(value);
                    return true;
                }
                pNode->SetUInt64(magnitude);
                return true;
            }

            bool ParseJsonString(std::string* pStr)
            {
                const bool quoteConsumed = Consume('"');
                if (!quoteConsumed)
                {
                    return Fail("expected a quoted string");
                }
                while (m_offset < m_size)
                {
                    const uint8_t value = m_pData[m_offset++];
                    if (value == '"')
                    {
                        return true;
                    }
                    if (value < 0x20)
                    {
                        return Fail("control character in string");
                    }
                    if (value != '\\')
                    {
                        pStr->push_back(static_cast<char>(value));
                        continue;
                    }
                    if (m_size <= m_offset)
                    {
                        return Fail("unterminated escape sequence");
                    }
                    const uint8_t escaped = m_pData[m_offset++];
                    switch (escaped)
                    {
                        case '"': pStr->push_back('"'); break;
                        case '\\': pStr->push_back('\\'); break;
                        case '/': pStr->push_back('/'); break;
                        case 'b': pStr->push_back('\b'); break;
                        case 'f': pStr->push_back('\f'); break;
                        case 'n': pStr->push_back('\n'); break;
                        case 'r': pStr->push_back('\r'); break;
                        case 't': pStr->push_back('\t'); break;
                        case 'u':
                        {
                            uint32_t codePoint = 0;
                            const bool codePointParsed = ParseHexQuad(&codePoint);
                            if (!codePointParsed)
                            {
                                return false;
                            }
                            if (0xD800 <= codePoint && codePoint <= 0xDBFF)
                            {
                                const bool slashConsumed = Consume('\\');
                                const bool unicodePrefixConsumed = slashConsumed && Consume('u');
                                if (!unicodePrefixConsumed)
                                {
                                    return Fail("missing low Unicode surrogate");
                                }
                                uint32_t lowSurrogate = 0;
                                const bool lowSurrogateParsed = ParseHexQuad(&lowSurrogate);
                                if (!lowSurrogateParsed || lowSurrogate < 0xDC00 || 0xDFFF < lowSurrogate)
                                {
                                    return Fail("invalid low Unicode surrogate");
                                }
                                codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + lowSurrogate - 0xDC00;
                            }
                            else if (0xDC00 <= codePoint && codePoint <= 0xDFFF)
                            {
                                return Fail("unexpected low Unicode surrogate");
                            }
                            AppendUtf8(codePoint, pStr);
                            break;
                        }
                        default: return Fail("invalid string escape sequence");
                    }
                }
                return Fail("unterminated string");
            }

            bool ParseHexQuad(uint32_t* pValue)
            {
                if (m_size - m_offset < 4)
                {
                    return Fail("incomplete Unicode escape");
                }
                uint32_t value = 0;
                for (size_t digitIndex = 0; digitIndex < 4; ++digitIndex)
                {
                    const uint8_t digit = m_pData[m_offset++];
                    value *= 16;
                    if ('0' <= digit && digit <= '9')
                    {
                        value += digit - '0';
                    }
                    else if ('a' <= digit && digit <= 'f')
                    {
                        value += digit - 'a' + 10;
                    }
                    else if ('A' <= digit && digit <= 'F')
                    {
                        value += digit - 'A' + 10;
                    }
                    else
                    {
                        return Fail("invalid Unicode escape");
                    }
                }
                *pValue = value;
                return true;
            }

            static void AppendUtf8(uint32_t codePoint, std::string* pStr)
            {
                if (codePoint <= 0x7F)
                {
                    pStr->push_back(static_cast<char>(codePoint));
                }
                else if (codePoint <= 0x7FF)
                {
                    pStr->push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
                    pStr->push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
                }
                else if (codePoint <= 0xFFFF)
                {
                    pStr->push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
                    pStr->push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
                    pStr->push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
                }
                else
                {
                    pStr->push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
                    pStr->push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
                    pStr->push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
                    pStr->push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
                }
            }

            bool TokenizeYamlLines(std::vector<YamlLine>* pLines)
            {
                size_t lineOffset = 0;
                bool documentEnded = false;
                while (lineOffset < m_size)
                {
                    size_t lineEnd = lineOffset;
                    while (lineEnd < m_size && m_pData[lineEnd] != '\n' && m_pData[lineEnd] != '\r')
                    {
                        ++lineEnd;
                    }
                    size_t indent = 0;
                    while (lineOffset + indent < lineEnd && m_pData[lineOffset + indent] == ' ')
                    {
                        ++indent;
                    }
                    if (lineOffset + indent < lineEnd && m_pData[lineOffset + indent] == '\t')
                    {
                        return FailAt(lineOffset + indent, "tabs are not allowed for YAML indentation");
                    }
                    std::string text(reinterpret_cast<const char*>(m_pData + lineOffset + indent), lineEnd - lineOffset - indent);
                    StripYamlComment(&text);
                    text = Trim(text);
                    if (text == "---")
                    {
                        if (!pLines->empty() || documentEnded)
                        {
                            return FailAt(lineOffset + indent, "multiple YAML documents are not supported");
                        }
                    }
                    else if (text == "...")
                    {
                        documentEnded = true;
                    }
                    else if (!text.empty())
                    {
                        if (documentEnded)
                        {
                            return FailAt(lineOffset + indent, "content follows the YAML document end marker");
                        }
                        pLines->push_back(YamlLine{indent, lineOffset + indent, std::move(text)});
                    }
                    lineOffset = lineEnd;
                    if (lineOffset < m_size && m_pData[lineOffset] == '\r')
                    {
                        ++lineOffset;
                    }
                    if (lineOffset < m_size && m_pData[lineOffset] == '\n')
                    {
                        ++lineOffset;
                    }
                }
                return true;
            }

            static void StripYamlComment(std::string* pText)
            {
                bool singleQuoted = false;
                bool doubleQuoted = false;
                bool escaped = false;
                for (size_t charIndex = 0; charIndex < pText->size(); ++charIndex)
                {
                    const char value = (*pText)[charIndex];
                    if (doubleQuoted && value == '\\' && !escaped)
                    {
                        escaped = true;
                        continue;
                    }
                    if (!escaped && !doubleQuoted && value == '\'')
                    {
                        singleQuoted = !singleQuoted;
                    }
                    else if (!escaped && !singleQuoted && value == '"')
                    {
                        doubleQuoted = !doubleQuoted;
                    }
                    else if (!singleQuoted && !doubleQuoted && value == '#' && (charIndex == 0 || std::isspace(static_cast<unsigned char>((*pText)[charIndex - 1])) != 0))
                    {
                        pText->resize(charIndex);
                        return;
                    }
                    escaped = false;
                }
            }

            static size_t FindYamlColon(const std::string& text)
            {
                bool singleQuoted = false;
                bool doubleQuoted = false;
                bool escaped = false;
                for (size_t charIndex = 0; charIndex < text.size(); ++charIndex)
                {
                    const char value = text[charIndex];
                    if (doubleQuoted && value == '\\' && !escaped)
                    {
                        escaped = true;
                        continue;
                    }
                    if (!escaped && !doubleQuoted && value == '\'')
                    {
                        singleQuoted = !singleQuoted;
                    }
                    else if (!escaped && !singleQuoted && value == '"')
                    {
                        doubleQuoted = !doubleQuoted;
                    }
                    else if (!singleQuoted && !doubleQuoted && value == ':' && (charIndex + 1 == text.size() || std::isspace(static_cast<unsigned char>(text[charIndex + 1])) != 0))
                    {
                        return charIndex;
                    }
                    escaped = false;
                }
                return std::string::npos;
            }

            bool ParseYamlKey(const std::string& text, std::string* pKey, size_t offset)
            {
                if (text.empty())
                {
                    return FailAt(offset, "empty YAML mapping key");
                }
                if (text[0] == '"')
                {
                    return ParseYamlDoubleQuoted(text, pKey, offset);
                }
                if (text[0] == '\'')
                {
                    return ParseYamlSingleQuoted(text, pKey, offset);
                }
                *pKey = text;
                return true;
            }

            bool ParseYamlScalar(const std::string& text, DataNode* pNode, size_t offset)
            {
                if (text.empty() || text == "null" || text == "Null" || text == "NULL" || text == "~")
                {
                    pNode->SetNull();
                    return true;
                }
                if (text == "true" || text == "True" || text == "TRUE")
                {
                    pNode->SetBoolean(true);
                    return true;
                }
                if (text == "false" || text == "False" || text == "FALSE")
                {
                    pNode->SetBoolean(false);
                    return true;
                }
                if (text[0] == '"')
                {
                    std::string str;
                    const bool stringParsed = ParseYamlDoubleQuoted(text, &str, offset);
                    if (!stringParsed)
                    {
                        return false;
                    }
                    const bool stringAccounted = Account(str.size());
                    if (!stringAccounted)
                    {
                        return false;
                    }
                    pNode->SetString(std::move(str));
                    return true;
                }
                if (text[0] == '\'')
                {
                    std::string str;
                    const bool stringParsed = ParseYamlSingleQuoted(text, &str, offset);
                    if (!stringParsed)
                    {
                        return false;
                    }
                    const bool stringAccounted = Account(str.size());
                    if (!stringAccounted)
                    {
                        return false;
                    }
                    pNode->SetString(std::move(str));
                    return true;
                }
                if (text[0] == '&' || text[0] == '*' || text[0] == '!' || text[0] == '%' || text[0] == '?' || text[0] == '|' || text[0] == '>' || text[0] == '[' || text[0] == '{')
                {
                    return FailAt(offset, "unsupported YAML feature");
                }
                for (size_t charIndex = 1; charIndex < text.size(); ++charIndex)
                {
                    const bool separated = std::isspace(static_cast<unsigned char>(text[charIndex - 1])) != 0;
                    const bool indicator = text[charIndex] == '&' || text[charIndex] == '*' || text[charIndex] == '!';
                    if (separated && indicator)
                    {
                        return FailAt(offset + charIndex, "unsupported YAML feature");
                    }
                }

                const bool couldBeNumber = text[0] == '-' || std::isdigit(static_cast<unsigned char>(text[0])) != 0;
                if (couldBeNumber)
                {
                    const uint8_t* pSavedData = m_pData;
                    const size_t savedSize = m_size;
                    const size_t savedOffset = m_offset;
                    m_pData = reinterpret_cast<const uint8_t*>(text.data());
                    m_size = text.size();
                    m_offset = 0;
                    const bool parsed = ParseNumber(pNode, false);
                    const bool consumed = m_offset == m_size;
                    const std::string parseError = m_pError == nullptr ? std::string() : m_pError->message;
                    m_pData = pSavedData;
                    m_size = savedSize;
                    m_offset = savedOffset;
                    if (parsed && consumed)
                    {
                        return true;
                    }
                    if (consumed)
                    {
                        const char* pMessage = parseError.empty() ? "invalid numeric YAML scalar" : parseError.c_str();
                        return FailAt(offset, pMessage);
                    }
                    if (m_pError != nullptr)
                    {
                        m_pError->Reset();
                    }
                }

                const bool stringAccounted = Account(text.size());
                if (!stringAccounted)
                {
                    return false;
                }
                pNode->SetString(text);
                return true;
            }

            bool ParseYamlDoubleQuoted(const std::string& text, std::string* pStr, size_t offset)
            {
                const uint8_t* pSavedData = m_pData;
                const size_t savedSize = m_size;
                const size_t savedOffset = m_offset;
                m_pData = reinterpret_cast<const uint8_t*>(text.data());
                m_size = text.size();
                m_offset = 0;
                const bool parsed = ParseJsonString(pStr);
                const bool consumed = parsed && m_offset == m_size;
                m_pData = pSavedData;
                m_size = savedSize;
                m_offset = savedOffset;
                if (!consumed)
                {
                    return FailAt(offset, "invalid double-quoted YAML string");
                }
                return true;
            }

            bool ParseYamlSingleQuoted(const std::string& text, std::string* pStr, size_t offset)
            {
                if (text.size() < 2 || text.back() != '\'')
                {
                    return FailAt(offset, "unterminated single-quoted YAML string");
                }
                for (size_t charIndex = 1; charIndex + 1 < text.size(); ++charIndex)
                {
                    if (text[charIndex] == '\'' && charIndex + 1 < text.size() - 1 && text[charIndex + 1] == '\'')
                    {
                        pStr->push_back('\'');
                        ++charIndex;
                    }
                    else
                    {
                        pStr->push_back(text[charIndex]);
                    }
                }
                return true;
            }

            static std::string Trim(const std::string& text)
            {
                size_t beginOffset = 0;
                while (beginOffset < text.size() && std::isspace(static_cast<unsigned char>(text[beginOffset])) != 0)
                {
                    ++beginOffset;
                }
                size_t endOffset = text.size();
                while (beginOffset < endOffset && std::isspace(static_cast<unsigned char>(text[endOffset - 1])) != 0)
                {
                    --endOffset;
                }
                return text.substr(beginOffset, endOffset - beginOffset);
            }

            void SkipJsonWhitespace()
            {
                while (m_offset < m_size && (m_pData[m_offset] == ' ' || m_pData[m_offset] == '\t' || m_pData[m_offset] == '\r' || m_pData[m_offset] == '\n'))
                {
                    ++m_offset;
                }
            }

            bool Consume(uint8_t value)
            {
                if (m_offset < m_size && m_pData[m_offset] == value)
                {
                    ++m_offset;
                    return true;
                }
                return false;
            }

            bool Match(const char* pText)
            {
                const size_t length = std::strlen(pText);
                const bool matches = m_size - m_offset >= length && std::memcmp(m_pData + m_offset, pText, length) == 0;
                if (!matches)
                {
                    return false;
                }
                m_offset += length;
                return true;
            }

            bool Account(size_t size)
            {
                if (m_options.maxMemory != 0 && m_options.maxMemory < m_memoryUsed + size)
                {
                    return Fail("document exceeds maxMemory");
                }
                m_memoryUsed += size;
                return true;
            }

            bool CheckDepth(size_t depth)
            {
                if (m_options.maxStackDepth < depth)
                {
                    return Fail("document exceeds maxStackDepth");
                }
                return true;
            }

            bool Fail(const char* pMessage)
            {
                return FailAt(m_offset, pMessage);
            }

            bool FailAt(size_t offset, const char* pMessage)
            {
                if (m_pError != nullptr)
                {
                    m_pError->offset = offset;
                    m_pError->line = 1;
                    m_pError->column = 1;
                    for (size_t charIndex = 0; charIndex < offset && charIndex < m_size; ++charIndex)
                    {
                        if (m_pData[charIndex] == '\n')
                        {
                            ++m_pError->line;
                            m_pError->column = 1;
                        }
                        else
                        {
                            ++m_pError->column;
                        }
                    }
                    m_pError->message = pMessage;
                }
                return false;
            }

        private:
            const uint8_t* m_pData = nullptr;
            size_t m_size = 0;
            size_t m_offset = 0;
            ReadOptions m_options;
            Error* m_pError = nullptr;
            size_t m_memoryUsed = 0;
            std::vector<JsonFrame> m_jsonFrames;
        };

        class Renderer
        {
        public:
            Renderer(const WriteOptions& options, Error* pError)
                : m_options(options)
                , m_pError(pError)
            {
                if (m_pError != nullptr)
                {
                    m_pError->Reset();
                }
            }

            bool RenderJson(const DataNode& root)
            {
                return Render(root, m_options.prettyJson);
            }

            bool RenderYaml(const DataNode& root)
            {
                return RenderYamlBlock(root);
            }

            const std::string& GetOutput() const
            {
                return m_output;
            }

        private:
            struct Frame
            {
                const DataNode* pNode = nullptr;
                size_t entryIndex = 0;
                size_t depth = 0;
                bool opened = false;
            };

            bool Render(const DataNode& root, bool prettyJson)
            {
                std::vector<Frame> frames;
                frames.push_back(Frame{&root, 0, 1, false});
                while (!frames.empty())
                {
                    Frame& frame = frames.back();
                    const NodeValue& value = frame.pNode->GetValue();
                    if (value.nodeType != NodeType::Sequence && value.nodeType != NodeType::Mapping)
                    {
                        const bool scalarAppended = AppendScalar(value, false);
                        if (!scalarAppended)
                        {
                            return false;
                        }
                        frames.pop_back();
                        continue;
                    }
                    if (!frame.opened)
                    {
                        if (m_options.maxStackDepth < frame.depth)
                        {
                            return Fail("document exceeds maxStackDepth");
                        }
                        const bool openingAppended = Append(value.nodeType == NodeType::Mapping ? "{" : "[");
                        if (!openingAppended)
                        {
                            return false;
                        }
                        frame.opened = true;
                    }
                    if (value.entries.Size() <= frame.entryIndex)
                    {
                        if (prettyJson && frame.entryIndex != 0)
                        {
                            const bool newlineAppended = Append("\n");
                            if (!newlineAppended)
                            {
                                return false;
                            }
                            const bool indentationAppended = AppendIndent(frame.depth - 1);
                            if (!indentationAppended)
                            {
                                return false;
                            }
                        }
                        const bool closingAppended = Append(value.nodeType == NodeType::Mapping ? "}" : "]");
                        if (!closingAppended)
                        {
                            return false;
                        }
                        frames.pop_back();
                        continue;
                    }
                    const bool separatorAppended = frame.entryIndex == 0 || Append(",");
                    if (!separatorAppended)
                    {
                        return false;
                    }
                    const bool firstRootKeyInline = m_options.prettyJsonFirstRootKeyInline && frame.depth == 1 && value.nodeType == NodeType::Mapping && frame.entryIndex == 0;
                    if (prettyJson && !firstRootKeyInline)
                    {
                        const bool newlineAppended = Append("\n");
                        if (!newlineAppended)
                        {
                            return false;
                        }
                        const bool indentationAppended = AppendIndent(frame.depth);
                        if (!indentationAppended)
                        {
                            return false;
                        }
                    }
                    const SequenceEntry& entry = value.entries[frame.entryIndex++];
                    if (value.nodeType == NodeType::Mapping)
                    {
                        const bool keyAppended = AppendQuoted(entry.key);
                        if (!keyAppended)
                        {
                            return false;
                        }
                        const bool colonAppended = Append(prettyJson ? ": " : ":");
                        if (!colonAppended)
                        {
                            return false;
                        }
                    }
                    frames.push_back(Frame{&entry.value, 0, frame.depth + 1, false});
                }
                return true;
            }

            bool RenderYamlBlock(const DataNode& root)
            {
                const NodeValue& rootValue = root.GetValue();
                if (rootValue.nodeType != NodeType::Sequence && rootValue.nodeType != NodeType::Mapping)
                {
                    const bool scalarAppended = AppendScalar(rootValue, true);
                    if (!scalarAppended)
                    {
                        return false;
                    }
                    return Append("\n");
                }

                std::vector<Frame> frames;
                frames.push_back(Frame{&root, 0, 1, true});
                while (!frames.empty())
                {
                    Frame& frame = frames.back();
                    const NodeValue& value = frame.pNode->GetValue();
                    if (m_options.maxStackDepth < frame.depth)
                    {
                        return Fail("document exceeds maxStackDepth");
                    }
                    if (value.entries.Size() <= frame.entryIndex)
                    {
                        if (value.entries.Empty())
                        {
                            const bool indentAppended = AppendIndent(frame.depth - 1);
                            if (!indentAppended)
                            {
                                return false;
                            }
                            const bool emptyContainerAppended = Append(value.nodeType == NodeType::Mapping ? "{}\n" : "[]\n");
                            if (!emptyContainerAppended)
                            {
                                return false;
                            }
                        }
                        frames.pop_back();
                        continue;
                    }

                    const SequenceEntry& entry = value.entries[frame.entryIndex++];
                    const NodeValue& childValue = entry.value.GetValue();
                    const bool childContainer = childValue.nodeType == NodeType::Sequence || childValue.nodeType == NodeType::Mapping;
                    const bool indentAppended = AppendIndent(frame.depth - 1);
                    if (!indentAppended)
                    {
                        return false;
                    }
                    if (value.nodeType == NodeType::Mapping)
                    {
                        const bool keyAppended = AppendQuoted(entry.key);
                        if (!keyAppended)
                        {
                            return false;
                        }
                        const bool colonAppended = Append(":");
                        if (!colonAppended)
                        {
                            return false;
                        }
                    }
                    else
                    {
                        const bool dashAppended = Append("-");
                        if (!dashAppended)
                        {
                            return false;
                        }
                    }
                    if (childContainer && !childValue.entries.Empty())
                    {
                        const bool newlineAppended = Append("\n");
                        if (!newlineAppended)
                        {
                            return false;
                        }
                        frames.push_back(Frame{&entry.value, 0, frame.depth + 1, true});
                    }
                    else
                    {
                        const bool spaceAppended = Append(" ");
                        if (!spaceAppended)
                        {
                            return false;
                        }
                        const bool scalarAppended = AppendScalar(childValue, true);
                        if (!scalarAppended)
                        {
                            return false;
                        }
                        const bool newlineAppended = Append("\n");
                        if (!newlineAppended)
                        {
                            return false;
                        }
                    }
                }
                return true;
            }

            bool AppendScalar(const NodeValue& value, bool yaml)
            {
                switch (value.nodeType)
                {
                    case NodeType::Null: return Append("null");
                    case NodeType::Boolean: return Append(value.s64 != 0 ? "true" : "false");
                    case NodeType::SignedInteger: return Append(std::to_string(value.s64));
                    case NodeType::UnsignedInteger: return Append(std::to_string(value.u64));
                    case NodeType::FloatingPoint:
                    {
                        const bool finite = std::isfinite(value.f64);
                        if (!finite)
                        {
                            return Fail("cannot render a non-finite floating-point value");
                        }
                        std::ostringstream stream;
                        stream.imbue(std::locale::classic());
                        stream << std::setprecision(17) << value.f64;
                        return Append(stream.str());
                    }
                    case NodeType::String: return AppendQuoted(value.str);
                    case NodeType::Sequence: return Append(yaml ? "[]" : "[]");
                    case NodeType::Mapping: return Append(yaml ? "{}" : "{}");
                }
                return Fail("invalid NodeType");
            }

            bool AppendQuoted(const std::string& value)
            {
                const bool quoteAppended = Append("\"");
                if (!quoteAppended)
                {
                    return false;
                }
                for (unsigned char character : value)
                {
                    switch (character)
                    {
                        case '"':
                        {
                            const bool escapedAppended = Append("\\\"");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        case '\\':
                        {
                            const bool escapedAppended = Append("\\\\");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        case '\b':
                        {
                            const bool escapedAppended = Append("\\b");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        case '\f':
                        {
                            const bool escapedAppended = Append("\\f");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        case '\n':
                        {
                            const bool escapedAppended = Append("\\n");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        case '\r':
                        {
                            const bool escapedAppended = Append("\\r");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        case '\t':
                        {
                            const bool escapedAppended = Append("\\t");
                            if (!escapedAppended)
                            {
                                return false;
                            }
                            break;
                        }
                        default:
                            if (character < 0x20)
                            {
                                const char hex[] = "0123456789abcdef";
                                std::string escaped = "\\u00";
                                escaped.push_back(hex[character >> 4]);
                                escaped.push_back(hex[character & 0x0F]);
                                const bool escapedAppended = Append(escaped);
                                if (!escapedAppended)
                                {
                                    return false;
                                }
                            }
                            else
                            {
                                const char valueChar = static_cast<char>(character);
                                const bool characterAppended = Append(&valueChar, 1);
                                if (!characterAppended)
                                {
                                    return false;
                                }
                            }
                            break;
                    }
                }
                return Append("\"");
            }

            bool AppendIndent(size_t depth)
            {
                const size_t spaceCount = depth * m_options.indentSize;
                if (m_options.maxOutputLength != 0 && m_options.maxOutputLength < m_output.size() + spaceCount)
                {
                    return Fail("output exceeds maxOutputLength");
                }
                m_output.append(spaceCount, ' ');
                return true;
            }

            bool Append(const std::string& value)
            {
                return Append(value.data(), value.size());
            }

            bool Append(const char* pValue)
            {
                return Append(pValue, std::strlen(pValue));
            }

            bool Append(const char* pValue, size_t size)
            {
                if (m_options.maxOutputLength != 0 && m_options.maxOutputLength < m_output.size() + size)
                {
                    return Fail("output exceeds maxOutputLength");
                }
                m_output.append(pValue, size);
                return true;
            }

            bool Fail(const char* pMessage)
            {
                if (m_pError != nullptr)
                {
                    m_pError->message = pMessage;
                }
                return false;
            }

        private:
            WriteOptions m_options;
            Error* m_pError = nullptr;
            std::string m_output;
        };
    };

}
} //namespace nv::trace
