/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv {
namespace trace {

    // Dependency-free streaming ZIP writer. The DEFLATE encoder uses fixed
    // Huffman codes and a bounded 32 KiB LZ77 history. It intentionally favors
    // a small, reviewable implementation over maximum compression ratio.
    class ZipStreamingCompressor
    {
    private:
        template <typename TAcquireBuffer, typename TSubmitBuffer>
        class Output;

    public:
        struct OutputBuffer
        {
            uint8_t* pData = nullptr;
            size_t capacity = 0;
            size_t slot = 0;
        };

        enum class CompressionMethod : uint16_t
        {
            Store = 0,
            Deflate = 8,
        };

        // Selects the fixed-Huffman match finder. Unrecognized values use Fast.
        enum class DeflateStrategy
        {
            // Favors throughput; Perfetto trace data typically compresses by about 4:1.
            Fast,
            // Favors smaller output; Perfetto trace data typically compresses by about 4.5:1.
            Compact,
        };

        struct EncodedEntry
        {
            std::string name;
            CompressionMethod method = CompressionMethod::Store;
            uint16_t flags = 0;
            uint32_t crc32 = 0;
            uint64_t compressedSize = 0;
            uint64_t uncompressedSize = 0;
        };

        static size_t HistorySize()
        {
            return kHistorySize;
        }

        static size_t HashTableSizeBytes()
        {
            return kHashEntryCount * sizeof(int64_t);
        }

        ZipStreamingCompressor() = default;

        ZipStreamingCompressor(const ZipStreamingCompressor&) = delete;
        ZipStreamingCompressor& operator=(const ZipStreamingCompressor&) = delete;

        // Allocates all memory required by the selected strategy. The strategy remains fixed until Reset().
        bool Initialize(size_t entryCapacity = 0, DeflateStrategy deflateStrategy = DeflateStrategy::Fast)
        {
            Reset();
            m_deflateStrategy = deflateStrategy;
            m_hashHeads.assign(kHashEntryCount, -1);
            m_history.assign(kHistorySize, uint8_t{0});
            if (m_deflateStrategy == DeflateStrategy::Compact)
            {
                m_hashPrevious.assign(kHistorySize, -1);
            }
            m_entries.reserve(entryCapacity);
            m_initialized = true;
            ResetDeflater();
            return true;
        }

        void Reset()
        {
            m_archiveOffset = 0;
            m_initialized = false;
            m_entryOpen = false;
            m_finished = false;
            m_failed = false;
            m_current = CentralEntry();
            m_entries.clear();
            m_hashHeads.clear();
            m_hashPrevious.clear();
            m_history.clear();
            m_deflateStrategy = DeflateStrategy::Fast;
            m_streamPosition = 0;
            m_bitBuffer = 0;
            m_bitCount = 0;
        }

        bool IsInitialized() const noexcept
        {
            return m_initialized;
        }

        template <typename TOutput>
        class ArchiveSession;

        template <typename TOutput>
        class EntryWriter
        {
        public:
            EntryWriter() = default;

            EntryWriter(const EntryWriter&) = delete;
            EntryWriter& operator=(const EntryWriter&) = delete;

            EntryWriter(EntryWriter&& rhs) noexcept
                : m_pArchive(rhs.m_pArchive)
                , m_inputBuffer(std::move(rhs.m_inputBuffer))
                , m_inputSize(rhs.m_inputSize)
                , m_active(rhs.m_active)
                , m_succeeded(rhs.m_succeeded)
            {
                rhs.m_pArchive = nullptr;
                rhs.m_inputSize = 0;
                rhs.m_active = false;
            }

            EntryWriter& operator=(EntryWriter&&) = delete;

            ~EntryWriter()
            {
                Close();
            }

            bool Write(const uint8_t* pData, size_t size)
            {
                if (!m_active || !m_succeeded)
                {
                    return false;
                }
                if (m_inputBuffer.empty())
                {
                    m_succeeded = m_pArchive->Write(pData, size);
                    return m_succeeded;
                }
                if (size != 0 && pData == nullptr)
                {
                    m_succeeded = false;
                    return false;
                }

                while (size != 0)
                {
                    const size_t available = m_inputBuffer.size() - m_inputSize;
                    const size_t copySize = std::min(size, available);
                    std::copy_n(pData, copySize, m_inputBuffer.data() + m_inputSize);
                    pData += copySize;
                    size -= copySize;
                    m_inputSize += copySize;
                    if (m_inputSize == m_inputBuffer.size() && !Flush())
                    {
                        return false;
                    }
                }
                return true;
            }

            bool Flush()
            {
                if (!m_active || !m_succeeded)
                {
                    return false;
                }
                if (m_inputSize == 0)
                {
                    return true;
                }
                m_succeeded = m_pArchive->Write(m_inputBuffer.data(), m_inputSize);
                if (m_succeeded)
                {
                    m_inputSize = 0;
                }
                return m_succeeded;
            }

            bool Close()
            {
                if (!m_active)
                {
                    return m_succeeded;
                }
                const bool flushSucceeded = Flush();
                m_active = false;
                m_succeeded = m_pArchive->CloseEntry(flushSucceeded);
                return m_succeeded;
            }

            bool Succeeded() const
            {
                return m_succeeded;
            }

            explicit operator bool() const
            {
                return Succeeded();
            }

        private:
            friend class ArchiveSession<TOutput>;

            EntryWriter(ArchiveSession<TOutput>& archive, bool opened, size_t inputBufferCapacity)
                : m_pArchive(&archive)
                , m_inputBuffer(opened ? inputBufferCapacity : 0)
                , m_active(opened)
                , m_succeeded(opened)
            {
            }

            ArchiveSession<TOutput>* m_pArchive = nullptr;
            std::vector<uint8_t> m_inputBuffer;
            size_t m_inputSize = 0;
            bool m_active = false;
            bool m_succeeded = false;
        };

        template <typename TOutput>
        class ArchiveSession
        {
        public:
            ArchiveSession(ZipStreamingCompressor& compressor, TOutput& output)
                : m_compressor(compressor)
                , m_output(output)
            {
            }

            EntryWriter<TOutput> OpenEntry(
                const std::string& name,
                CompressionMethod method = CompressionMethod::Deflate,
                size_t inputBufferCapacity = 0)
            {
                const bool opened = m_compressor.BeginEntry(m_output, name, method);
                return EntryWriter<TOutput>(*this, opened, inputBufferCapacity);
            }

            /// Writes one stored entry whose complete payload is already available.
            /// Its local header contains final sizes and CRC, so no streaming data descriptor is emitted.
            bool WriteStoredEntry(const std::string& name, const uint8_t* pData, size_t size)
            {
                return m_compressor.WriteStoredEntry(m_output, name, pData, size);
            }

            bool ImportEncodedEntry(
                const EncodedEntry& source,
                const std::string& destinationName,
                std::istream& encodedPayload)
            {
                return m_compressor.ImportEncodedEntry(m_output, source, destinationName, encodedPayload);
            }

            bool Succeeded() const
            {
                return m_compressor.Good();
            }

        private:
            friend class EntryWriter<TOutput>;

            bool Write(const uint8_t* pData, size_t size)
            {
                return m_compressor.Write(m_output, pData, size);
            }

            bool CloseEntry(bool entrySucceeded)
            {
                if (!entrySucceeded)
                {
                    m_compressor.m_entryOpen = false;
                    return m_compressor.Fail();
                }
                return m_compressor.FinishEntry(m_output);
            }

            ZipStreamingCompressor& m_compressor;
            TOutput& m_output;
        };

        template <typename TAcquireBuffer, typename TSubmitBuffer, typename TWriteArchive>
        bool WriteArchive(TAcquireBuffer&& acquireBuffer, TSubmitBuffer&& submitBuffer, TWriteArchive&& writeArchive)
        {
            if (!m_initialized || m_finished || m_entryOpen || m_failed)
            {
                return Fail();
            }

            typedef typename std::remove_reference<TAcquireBuffer>::type AcquireBuffer;
            typedef typename std::remove_reference<TSubmitBuffer>::type SubmitBuffer;
            Output<AcquireBuffer, SubmitBuffer> output(acquireBuffer, submitBuffer);
            ArchiveSession<Output<AcquireBuffer, SubmitBuffer>> archive(*this, output);

            const bool writeArchiveSucceeded = writeArchive(archive);
            if (!writeArchiveSucceeded)
            {
                Fail();
            }

            bool finishSucceeded = false;
            const bool archiveSucceeded = archive.Succeeded();
            if (writeArchiveSucceeded && archiveSucceeded)
            {
                finishSucceeded = Finish(output);
            }
            if (!finishSucceeded)
            {
                output.Release();
                return false;
            }
            return true;
        }

        bool Good() const noexcept
        {
            return !m_failed;
        }

        uint64_t BytesWritten() const noexcept
        {
            return m_archiveOffset;
        }

        static uint32_t ComputeCrc32(const uint8_t* pData, size_t size)
        {
            return UpdateCrc32(0xFFFFFFFFu, pData, size) ^ 0xFFFFFFFFu;
        }

    private:
        enum : size_t
        {
            kHistorySize = 32 * 1024,
            kHistoryMask = kHistorySize - 1,
            kHashEntryCount = 64 * 1024,
            kAggressiveMaximumCandidates = 8,
        };

        static constexpr uint16_t kFlagEncrypted = 1u << 0;
        static constexpr uint16_t kFlagDataDescriptor = 1u << 3;
        static constexpr uint16_t kFlagUtf8 = 1u << 11;
        static constexpr uint16_t kVersionDefault = 20;
        static constexpr uint16_t kVersionZip64 = 45;
        static constexpr uint16_t kZip64ExtraId = 0x0001;
        static constexpr uint16_t kZip16Sentinel = 0xFFFFu;
        static constexpr uint32_t kZip32Sentinel = 0xFFFFFFFFu;
        static constexpr uint32_t kLocalHeaderSignature = 0x04034B50u;
        static constexpr uint32_t kCentralHeaderSignature = 0x02014B50u;
        static constexpr uint32_t kDataDescriptorSignature = 0x08074B50u;
        static constexpr uint32_t kZip64EndSignature = 0x06064B50u;
        static constexpr uint32_t kZip64LocatorSignature = 0x07064B50u;
        static constexpr uint32_t kEndOfCentralDirectorySignature = 0x06054B50u;

        struct CentralEntry : EncodedEntry
        {
            uint64_t localHeaderOffset = 0;
        };

        struct DeflateMatch
        {
            size_t length = 0;
            size_t distance = 0;
        };

        static bool NeedsZip64(uint64_t value)
        {
            return value >= kZip32Sentinel;
        }

        bool ReadyForEntry()
        {
            return m_initialized && !m_failed && !m_finished && !m_entryOpen;
        }

        bool ValidateName(const std::string& name)
        {
            return !name.empty() && name.size() <= std::numeric_limits<uint16_t>::max() && name.find('\0') == std::string::npos;
        }

        bool Fail()
        {
            m_failed = true;
            return false;
        }

        template <typename TAcquireBuffer, typename TSubmitBuffer>
        class Output
        {
        public:
            Output(TAcquireBuffer& acquireBuffer, TSubmitBuffer& submitBuffer)
                : m_acquireBuffer(acquireBuffer)
                , m_submitBuffer(submitBuffer)
            {
            }

            bool Write(const uint8_t* pData, size_t size)
            {
                while (size != 0)
                {
                    if (!m_acquired)
                    {
                        Acquire();
                    }

                    const size_t available = m_buffer.capacity - m_size;
                    const size_t count = std::min(size, available);
                    std::copy_n(pData, count, m_buffer.pData + m_size);
                    pData += count;
                    size -= count;
                    m_size += count;

                    if (m_size == m_buffer.capacity)
                    {
                        const bool submitSucceeded = SubmitCurrent();
                        if (!submitSucceeded)
                        {
                            return false;
                        }
                    }
                }
                return true;
            }

            bool Flush()
            {
                if (!m_acquired)
                {
                    return true;
                }
                return SubmitCurrent();
            }

            void Release()
            {
                if (!m_acquired)
                {
                    return;
                }
                const bool releaseSucceeded = m_submitBuffer(m_buffer, 0);
                static_cast<void>(releaseSucceeded);
                m_buffer = OutputBuffer();
                m_size = 0;
                m_acquired = false;
            }

        private:
            void Acquire()
            {
                m_buffer = m_acquireBuffer();
                assert(m_buffer.pData != nullptr);
                assert(m_buffer.capacity != 0);
                m_size = 0;
                m_acquired = true;
            }

            bool SubmitCurrent()
            {
                const bool submitSucceeded = m_submitBuffer(m_buffer, m_size);
                m_buffer = OutputBuffer();
                m_size = 0;
                m_acquired = false;
                return submitSucceeded;
            }

            TAcquireBuffer& m_acquireBuffer;
            TSubmitBuffer& m_submitBuffer;
            OutputBuffer m_buffer;
            size_t m_size = 0;
            bool m_acquired = false;
        };

        template <typename TOutput>
        bool Emit(TOutput& output, const uint8_t* pData, size_t size)
        {
            if (m_failed)
            {
                return false;
            }
            const bool writeSucceeded = output.Write(pData, size);
            if (!writeSucceeded)
            {
                return Fail();
            }
            m_archiveOffset += size;
            return true;
        }

        template <typename TOutput>
        bool BeginEntry(TOutput& output, const std::string& name, CompressionMethod method)
        {
            const bool readyForEntry = ReadyForEntry();
            const bool nameValid = ValidateName(name);
            if (!readyForEntry || !nameValid)
            {
                return Fail();
            }

            m_current = CentralEntry();
            m_current.name = name;
            m_current.method = method;
            m_current.flags = kFlagDataDescriptor | kFlagUtf8;
            m_current.localHeaderOffset = m_archiveOffset;
            m_current.crc32 = 0xFFFFFFFFu;
            m_entryOpen = true;
            ResetDeflater();

            std::vector<uint8_t> header;
            Put32(header, kLocalHeaderSignature);
            Put16(header, kVersionZip64);
            Put16(header, m_current.flags);
            Put16(header, static_cast<uint16_t>(method));
            Put16(header, 0);
            Put16(header, 0);
            Put32(header, 0);
            Put32(header, 0);
            Put32(header, 0);
            Put16(header, static_cast<uint16_t>(name.size()));
            Put16(header, 0);
            PutString(header, name);
            const bool headerSucceeded = Emit(output, header.data(), header.size());
            if (!headerSucceeded)
            {
                return false;
            }
            if (method == CompressionMethod::Deflate)
            {
                const uint64_t before = m_archiveOffset;
                const bool blockSucceeded = BeginFixedBlock(output, true);
                if (!blockSucceeded)
                {
                    return false;
                }
                m_current.compressedSize += m_archiveOffset - before;
            }
            return true;
        }

        template <typename TOutput>
        bool Write(TOutput& output, const uint8_t* pData, size_t size)
        {
            if (!m_entryOpen || m_failed || (size != 0 && pData == nullptr))
            {
                return Fail();
            }
            m_current.crc32 = UpdateCrc32(m_current.crc32, pData, size);
            if (size > std::numeric_limits<uint64_t>::max() - m_current.uncompressedSize)
            {
                return Fail();
            }
            m_current.uncompressedSize += size;

            if (m_current.method == CompressionMethod::Store)
            {
                const bool emitSucceeded = Emit(output, pData, size);
                if (!emitSucceeded)
                {
                    return false;
                }
                m_current.compressedSize += size;
                return true;
            }
            return Deflate(output, pData, size);
        }

        template <typename TOutput>
        bool FinishEntry(TOutput& output)
        {
            if (!m_entryOpen || m_failed)
            {
                return Fail();
            }
            if (m_current.method == CompressionMethod::Deflate)
            {
                const uint64_t before = m_archiveOffset;
                const bool symbolSucceeded = WriteFixedSymbol(output, 256);
                if (!symbolSucceeded)
                {
                    return false;
                }
                const bool flushSucceeded = FlushBits(output);
                if (!flushSucceeded)
                {
                    return false;
                }
                m_current.compressedSize += m_archiveOffset - before;
            }
            m_current.crc32 ^= 0xFFFFFFFFu;

            std::vector<uint8_t> descriptor;
            Put32(descriptor, kDataDescriptorSignature);
            Put32(descriptor, m_current.crc32);
            const bool zip64 = NeedsZip64(m_current.compressedSize) || NeedsZip64(m_current.uncompressedSize);
            if (zip64)
            {
                Put64(descriptor, m_current.compressedSize);
                Put64(descriptor, m_current.uncompressedSize);
            }
            else
            {
                Put32(descriptor, static_cast<uint32_t>(m_current.compressedSize));
                Put32(descriptor, static_cast<uint32_t>(m_current.uncompressedSize));
            }
            const bool emitSucceeded = Emit(output, descriptor.data(), descriptor.size());
            if (!emitSucceeded)
            {
                return false;
            }
            m_entries.push_back(m_current);
            m_entryOpen = false;
            return true;
        }

        template <typename TOutput>
        bool WriteStoredEntry(
            TOutput& output,
            const std::string& name,
            const uint8_t* pData,
            size_t size)
        {
            const bool readyForEntry = ReadyForEntry();
            const bool nameValid = ValidateName(name);
            const bool payloadValid = size == 0 || pData != nullptr;
            if (!readyForEntry || !nameValid || !payloadValid)
            {
                return Fail();
            }

            CentralEntry entry;
            entry.name = name;
            entry.method = CompressionMethod::Store;
            entry.flags = kFlagUtf8;
            entry.crc32 = ComputeCrc32(pData, size);
            entry.compressedSize = size;
            entry.uncompressedSize = size;
            entry.localHeaderOffset = m_archiveOffset;

            std::vector<uint8_t> extra;
            const bool zip64 = NeedsZip64(size);
            if (zip64)
            {
                Put16(extra, kZip64ExtraId);
                Put16(extra, 16);
                Put64(extra, entry.uncompressedSize);
                Put64(extra, entry.compressedSize);
            }

            std::vector<uint8_t> header;
            Put32(header, kLocalHeaderSignature);
            Put16(header, zip64 ? kVersionZip64 : kVersionDefault);
            Put16(header, entry.flags);
            Put16(header, static_cast<uint16_t>(entry.method));
            Put16(header, 0);
            Put16(header, 0);
            Put32(header, entry.crc32);
            Put32(header, zip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.compressedSize));
            Put32(header, zip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.uncompressedSize));
            Put16(header, static_cast<uint16_t>(entry.name.size()));
            Put16(header, static_cast<uint16_t>(extra.size()));
            PutString(header, entry.name);
            PutBytes(header, extra.data(), extra.size());
            const bool headerSucceeded = Emit(output, header.data(), header.size());
            if (!headerSucceeded)
            {
                return false;
            }
            const bool payloadSucceeded = Emit(output, pData, size);
            if (!payloadSucceeded)
            {
                return false;
            }
            m_entries.push_back(std::move(entry));
            return true;
        }

        template <typename TOutput>
        bool ImportEncodedEntry(
            TOutput& output,
            const EncodedEntry& source,
            const std::string& destinationName,
            std::istream& encodedPayload)
        {
            const bool readyForEntry = ReadyForEntry();
            const bool nameValid = ValidateName(destinationName);
            if (!readyForEntry || !nameValid || (source.method != CompressionMethod::Store && source.method != CompressionMethod::Deflate) || (source.flags & kFlagEncrypted) != 0)
            {
                return Fail();
            }

            CentralEntry entry;
            entry.name = destinationName;
            entry.method = source.method;
            entry.flags = kFlagUtf8;
            entry.crc32 = source.crc32;
            entry.compressedSize = source.compressedSize;
            entry.uncompressedSize = source.uncompressedSize;
            entry.localHeaderOffset = m_archiveOffset;

            std::vector<uint8_t> extra;
            const bool zip64 = NeedsZip64(entry.compressedSize) || NeedsZip64(entry.uncompressedSize);
            if (zip64)
            {
                Put16(extra, kZip64ExtraId);
                Put16(extra, 16);
                Put64(extra, entry.uncompressedSize);
                Put64(extra, entry.compressedSize);
            }

            std::vector<uint8_t> header;
            Put32(header, kLocalHeaderSignature);
            Put16(header, zip64 ? kVersionZip64 : kVersionDefault);
            Put16(header, entry.flags);
            Put16(header, static_cast<uint16_t>(entry.method));
            Put16(header, 0);
            Put16(header, 0);
            Put32(header, entry.crc32);
            Put32(header, zip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.compressedSize));
            Put32(header, zip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.uncompressedSize));
            Put16(header, static_cast<uint16_t>(entry.name.size()));
            Put16(header, static_cast<uint16_t>(extra.size()));
            PutString(header, entry.name);
            PutBytes(header, extra.data(), extra.size());
            const bool headerSucceeded = Emit(output, header.data(), header.size());
            if (!headerSucceeded)
            {
                return false;
            }

            std::vector<uint8_t> copyBuffer(64 * 1024, uint8_t{0});
            uint64_t remaining = entry.compressedSize;
            while (remaining != 0)
            {
                const size_t request = static_cast<size_t>(std::min<uint64_t>(remaining, copyBuffer.size()));
                encodedPayload.read(reinterpret_cast<char*>(copyBuffer.data()), static_cast<std::streamsize>(request));
                if (encodedPayload.gcount() != static_cast<std::streamsize>(request))
                {
                    return Fail();
                }
                const bool emitSucceeded = Emit(output, copyBuffer.data(), request);
                if (!emitSucceeded)
                {
                    return false;
                }
                remaining -= request;
            }
            m_entries.push_back(std::move(entry));
            return true;
        }

        static void Put16(std::vector<uint8_t>& out, uint16_t value)
        {
            out.push_back(static_cast<uint8_t>(value));
            out.push_back(static_cast<uint8_t>(value >> 8));
        }

        static void Put32(std::vector<uint8_t>& out, uint32_t value)
        {
            Put16(out, static_cast<uint16_t>(value));
            Put16(out, static_cast<uint16_t>(value >> 16));
        }

        static void Put64(std::vector<uint8_t>& out, uint64_t value)
        {
            Put32(out, static_cast<uint32_t>(value));
            Put32(out, static_cast<uint32_t>(value >> 32));
        }

        static void PutBytes(std::vector<uint8_t>& out, const uint8_t* pData, size_t size)
        {
            out.insert(out.end(), pData, pData + size);
        }

        static void PutString(std::vector<uint8_t>& out, const std::string& value)
        {
            PutBytes(out, reinterpret_cast<const uint8_t*>(value.data()), value.size());
        }

        static uint32_t UpdateCrc32(uint32_t crc, const uint8_t* pData, size_t size)
        {
            static const std::vector<uint32_t> table = [] {
                std::vector<uint32_t> result(256, 0);
                for (uint32_t tableIndex = 0; tableIndex < result.size(); ++tableIndex)
                {
                    uint32_t value = tableIndex;
                    for (int bit = 0; bit < 8; ++bit)
                    {
                        value = (value >> 1) ^ ((value & 1) ? 0xEDB88320u : 0u);
                    }
                    result[tableIndex] = value;
                }
                return result;
            }();
            for (size_t dataIndex = 0; dataIndex < size; ++dataIndex)
            {
                crc = table[(crc ^ pData[dataIndex]) & 0xFFu] ^ (crc >> 8);
            }
            return crc;
        }

        static uint32_t ReverseBits(uint32_t value, unsigned count)
        {
            // clang-format off
            static constexpr std::array<uint8_t, 256> reversedBytes = {
                0x00u, 0x80u, 0x40u, 0xC0u, 0x20u, 0xA0u, 0x60u, 0xE0u, 0x10u, 0x90u, 0x50u, 0xD0u, 0x30u, 0xB0u, 0x70u, 0xF0u,
                0x08u, 0x88u, 0x48u, 0xC8u, 0x28u, 0xA8u, 0x68u, 0xE8u, 0x18u, 0x98u, 0x58u, 0xD8u, 0x38u, 0xB8u, 0x78u, 0xF8u,
                0x04u, 0x84u, 0x44u, 0xC4u, 0x24u, 0xA4u, 0x64u, 0xE4u, 0x14u, 0x94u, 0x54u, 0xD4u, 0x34u, 0xB4u, 0x74u, 0xF4u,
                0x0Cu, 0x8Cu, 0x4Cu, 0xCCu, 0x2Cu, 0xACu, 0x6Cu, 0xECu, 0x1Cu, 0x9Cu, 0x5Cu, 0xDCu, 0x3Cu, 0xBCu, 0x7Cu, 0xFCu,
                0x02u, 0x82u, 0x42u, 0xC2u, 0x22u, 0xA2u, 0x62u, 0xE2u, 0x12u, 0x92u, 0x52u, 0xD2u, 0x32u, 0xB2u, 0x72u, 0xF2u,
                0x0Au, 0x8Au, 0x4Au, 0xCAu, 0x2Au, 0xAAu, 0x6Au, 0xEAu, 0x1Au, 0x9Au, 0x5Au, 0xDAu, 0x3Au, 0xBAu, 0x7Au, 0xFAu,
                0x06u, 0x86u, 0x46u, 0xC6u, 0x26u, 0xA6u, 0x66u, 0xE6u, 0x16u, 0x96u, 0x56u, 0xD6u, 0x36u, 0xB6u, 0x76u, 0xF6u,
                0x0Eu, 0x8Eu, 0x4Eu, 0xCEu, 0x2Eu, 0xAEu, 0x6Eu, 0xEEu, 0x1Eu, 0x9Eu, 0x5Eu, 0xDEu, 0x3Eu, 0xBEu, 0x7Eu, 0xFEu,
                0x01u, 0x81u, 0x41u, 0xC1u, 0x21u, 0xA1u, 0x61u, 0xE1u, 0x11u, 0x91u, 0x51u, 0xD1u, 0x31u, 0xB1u, 0x71u, 0xF1u,
                0x09u, 0x89u, 0x49u, 0xC9u, 0x29u, 0xA9u, 0x69u, 0xE9u, 0x19u, 0x99u, 0x59u, 0xD9u, 0x39u, 0xB9u, 0x79u, 0xF9u,
                0x05u, 0x85u, 0x45u, 0xC5u, 0x25u, 0xA5u, 0x65u, 0xE5u, 0x15u, 0x95u, 0x55u, 0xD5u, 0x35u, 0xB5u, 0x75u, 0xF5u,
                0x0Du, 0x8Du, 0x4Du, 0xCDu, 0x2Du, 0xADu, 0x6Du, 0xEDu, 0x1Du, 0x9Du, 0x5Du, 0xDDu, 0x3Du, 0xBDu, 0x7Du, 0xFDu,
                0x03u, 0x83u, 0x43u, 0xC3u, 0x23u, 0xA3u, 0x63u, 0xE3u, 0x13u, 0x93u, 0x53u, 0xD3u, 0x33u, 0xB3u, 0x73u, 0xF3u,
                0x0Bu, 0x8Bu, 0x4Bu, 0xCBu, 0x2Bu, 0xABu, 0x6Bu, 0xEBu, 0x1Bu, 0x9Bu, 0x5Bu, 0xDBu, 0x3Bu, 0xBBu, 0x7Bu, 0xFBu,
                0x07u, 0x87u, 0x47u, 0xC7u, 0x27u, 0xA7u, 0x67u, 0xE7u, 0x17u, 0x97u, 0x57u, 0xD7u, 0x37u, 0xB7u, 0x77u, 0xF7u,
                0x0Fu, 0x8Fu, 0x4Fu, 0xCFu, 0x2Fu, 0xAFu, 0x6Fu, 0xEFu, 0x1Fu, 0x9Fu, 0x5Fu, 0xDFu, 0x3Fu, 0xBFu, 0x7Fu, 0xFFu
            };
            // clang-format on
            if (count <= 8)
            {
                return reversedBytes[value & 0xFFu] >> (8 - count);
            }
            const uint32_t reversedLowByte = reversedBytes[value & 0xFFu];
            const uint32_t reversedHighByte = reversedBytes[(value >> 8) & 0xFFu];
            const uint32_t reversedValue = (reversedLowByte << 8) | reversedHighByte;
            return reversedValue >> (16 - count);
        }

        template <typename TOutput>
        bool WriteBits(TOutput& output, uint32_t value, unsigned count)
        {
            m_bitBuffer |= static_cast<uint64_t>(value) << m_bitCount;
            m_bitCount += count;
            while (8 <= m_bitCount)
            {
                const uint8_t byte = static_cast<uint8_t>(m_bitBuffer);
                const bool emitSucceeded = Emit(output, &byte, 1);
                if (!emitSucceeded)
                {
                    return false;
                }
                m_bitBuffer >>= 8;
                m_bitCount -= 8;
            }
            return true;
        }

        template <typename TOutput>
        bool FlushBits(TOutput& output)
        {
            if (m_bitCount != 0)
            {
                const uint8_t byte = static_cast<uint8_t>(m_bitBuffer);
                const bool emitSucceeded = Emit(output, &byte, 1);
                if (!emitSucceeded)
                {
                    return false;
                }
            }
            m_bitBuffer = 0;
            m_bitCount = 0;
            return true;
        }

        template <typename TOutput>
        bool BeginFixedBlock(TOutput& output, bool final)
        {
            const bool finalBitSucceeded = WriteBits(output, final ? 1u : 0u, 1);
            if (!finalBitSucceeded)
            {
                return false;
            }
            return WriteBits(output, 1u, 2);
        }

        template <typename TOutput>
        bool WriteFixedSymbol(TOutput& output, unsigned symbol)
        {
            if (symbol <= 143)
            {
                return WriteBits(output, ReverseBits(0x30u + symbol, 8), 8);
            }
            if (symbol <= 255)
            {
                return WriteBits(output, ReverseBits(0x190u + symbol - 144, 9), 9);
            }
            if (symbol <= 279)
            {
                return WriteBits(output, ReverseBits(symbol - 256, 7), 7);
            }
            return WriteBits(output, ReverseBits(0xC0u + symbol - 280, 8), 8);
        }

        template <typename TOutput>
        bool WriteLengthDistance(TOutput& output, size_t length, size_t distance)
        {
            static constexpr std::array<uint16_t, 29> lengthBase = {
                3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
            static constexpr std::array<uint8_t, 29> lengthExtra = {
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
            static constexpr std::array<uint16_t, 30> distanceBase = {
                1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
            static constexpr std::array<uint8_t, 30> distanceExtra = {
                0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

            size_t lengthCode = 0;
            while (lengthCode + 1 < lengthBase.size() && length >= lengthBase[lengthCode + 1])
            {
                ++lengthCode;
            }
            const bool symbolSucceeded = WriteFixedSymbol(output, static_cast<unsigned>(257 + lengthCode));
            if (!symbolSucceeded)
            {
                return false;
            }
            const unsigned lengthBits = lengthExtra[lengthCode];
            if (lengthBits)
            {
                const bool lengthSucceeded = WriteBits(output, static_cast<uint32_t>(length - lengthBase[lengthCode]), lengthBits);
                if (!lengthSucceeded)
                {
                    return false;
                }
            }

            size_t distanceCode = 0;
            while (distanceCode + 1 < distanceBase.size() && distance >= distanceBase[distanceCode + 1])
            {
                ++distanceCode;
            }
            const bool distanceCodeSucceeded = WriteBits(output, ReverseBits(static_cast<uint32_t>(distanceCode), 5), 5);
            if (!distanceCodeSucceeded)
            {
                return false;
            }
            const unsigned distanceBits = distanceExtra[distanceCode];
            if (!distanceBits)
            {
                return true;
            }
            return WriteBits(output, static_cast<uint32_t>(distance - distanceBase[distanceCode]), distanceBits);
        }

        static uint16_t Hash3(const uint8_t* pData)
        {
            return static_cast<uint16_t>(((static_cast<uint32_t>(pData[0]) * 251u + pData[1]) * 251u + pData[2]) & 0xFFFFu);
        }

        uint8_t HistoryByte(uint64_t position) const
        {
            return m_history[static_cast<size_t>(position) & kHistoryMask];
        }

        void InsertByte(uint64_t position, uint8_t value)
        {
            m_history[static_cast<size_t>(position) & kHistoryMask] = value;
        }

        size_t CalculateMatchLength(
            int64_t candidate,
            uint64_t position,
            const uint8_t* pData,
            size_t inputIndex,
            size_t maximumLength) const
        {
            size_t length = 0;
            while (length < maximumLength)
            {
                const uint64_t sourcePosition = static_cast<uint64_t>(candidate) + length;
                const uint8_t sourceByte = sourcePosition < position
                                             ? HistoryByte(sourcePosition)
                                             : pData[inputIndex + static_cast<size_t>(sourcePosition - position)];
                if (sourceByte != pData[inputIndex + length])
                {
                    break;
                }
                ++length;
            }
            return length;
        }

        DeflateMatch FindFastMatch(const uint8_t* pData, size_t inputIndex, size_t size) const
        {
            DeflateMatch match;
            if (size < inputIndex + 3)
            {
                return match;
            }

            const uint64_t position = m_streamPosition;
            const int64_t candidate = m_hashHeads[Hash3(pData + inputIndex)];
            if (candidate < 0 || position <= static_cast<uint64_t>(candidate))
            {
                return match;
            }
            const uint64_t distance = position - static_cast<uint64_t>(candidate);
            if (kHistorySize < distance)
            {
                return match;
            }

            const size_t maximumLength = std::min<size_t>(258, size - inputIndex);
            const size_t length = CalculateMatchLength(candidate, position, pData, inputIndex, maximumLength);
            if (3 <= length)
            {
                match.length = length;
                match.distance = static_cast<size_t>(distance);
            }
            return match;
        }

        DeflateMatch FindAggressiveMatch(
            const uint8_t* pData,
            size_t inputIndex,
            size_t size,
            uint64_t position) const
        {
            DeflateMatch match;
            if (size < inputIndex + 3)
            {
                return match;
            }

            int64_t candidate = m_hashHeads[Hash3(pData + inputIndex)];
            size_t candidateCount = 0;
            const size_t maximumLength = std::min<size_t>(258, size - inputIndex);
            while (candidateCount < kAggressiveMaximumCandidates && 0 <= candidate)
            {
                if (position <= static_cast<uint64_t>(candidate))
                {
                    break;
                }
                const uint64_t distance = position - static_cast<uint64_t>(candidate);
                if (kHistorySize < distance)
                {
                    break;
                }

                if (match.length != 0)
                {
                    const uint64_t sourcePosition = static_cast<uint64_t>(candidate) + match.length;
                    const uint8_t sourceByte = sourcePosition < position
                                                 ? HistoryByte(sourcePosition)
                                                 : pData[inputIndex + static_cast<size_t>(sourcePosition - position)];
                    if (sourceByte != pData[inputIndex + match.length])
                    {
                        candidate = m_hashPrevious[static_cast<size_t>(candidate) & kHistoryMask];
                        ++candidateCount;
                        continue;
                    }
                }

                const size_t length = CalculateMatchLength(candidate, position, pData, inputIndex, maximumLength);
                if (match.length < length)
                {
                    match.length = length;
                    match.distance = static_cast<size_t>(distance);
                    if (maximumLength == length)
                    {
                        break;
                    }
                }
                candidate = m_hashPrevious[static_cast<size_t>(candidate) & kHistoryMask];
                ++candidateCount;
            }
            if (match.length < 3)
            {
                match = DeflateMatch();
            }
            return match;
        }

        void InsertAggressiveHash(uint64_t position, const uint8_t* pThreeBytes)
        {
            const uint16_t hash = Hash3(pThreeBytes);
            const size_t historyIndex = static_cast<size_t>(position) & kHistoryMask;
            m_hashPrevious[historyIndex] = m_hashHeads[hash];
            m_hashHeads[hash] = static_cast<int64_t>(position);
        }

        void ResolveAggressiveWriteBoundary(const uint8_t* pData, size_t size)
        {
            std::array<uint8_t, 3> hashBytes{};
            if (2 <= m_streamPosition && 1 <= size)
            {
                hashBytes[0] = HistoryByte(m_streamPosition - 2);
                hashBytes[1] = HistoryByte(m_streamPosition - 1);
                hashBytes[2] = pData[0];
                InsertAggressiveHash(m_streamPosition - 2, hashBytes.data());
            }
            if (1 <= m_streamPosition && 2 <= size)
            {
                hashBytes[0] = HistoryByte(m_streamPosition - 1);
                hashBytes[1] = pData[0];
                hashBytes[2] = pData[1];
                InsertAggressiveHash(m_streamPosition - 1, hashBytes.data());
            }
        }

        static bool PreferNextMatch(const DeflateMatch& currentMatch, const DeflateMatch& nextMatch)
        {
            return currentMatch.length + 1 < nextMatch.length;
        }

        template <typename TOutput>
        bool DeflateFast(TOutput& output, const uint8_t* pData, size_t size)
        {
            if (size == 0)
            {
                return true;
            }
            const uint64_t before = m_archiveOffset;
            size_t index = 0;
            while (index < size)
            {
                const DeflateMatch match = FindFastMatch(pData, index, size);
                const size_t consumed = match.length != 0 ? match.length : 1;
                if (match.length != 0)
                {
                    const bool matchSucceeded = WriteLengthDistance(output, match.length, match.distance);
                    if (!matchSucceeded)
                    {
                        return false;
                    }
                }
                else
                {
                    const bool literalSucceeded = WriteFixedSymbol(output, pData[index]);
                    if (!literalSucceeded)
                    {
                        return false;
                    }
                }

                for (size_t consumedIndex = 0; consumedIndex < consumed; ++consumedIndex)
                {
                    const uint64_t insertedPosition = m_streamPosition;
                    InsertByte(insertedPosition, pData[index + consumedIndex]);
                    if (index + consumedIndex + 2 < size)
                    {
                        m_hashHeads[Hash3(pData + index + consumedIndex)] = static_cast<int64_t>(insertedPosition);
                    }
                    ++m_streamPosition;
                }
                index += consumed;
            }
            m_current.compressedSize += m_archiveOffset - before;
            return true;
        }

        template <typename TOutput>
        bool DeflateAggressive(TOutput& output, const uint8_t* pData, size_t size)
        {
            if (size == 0)
            {
                return true;
            }
            const uint64_t before = m_archiveOffset;
            ResolveAggressiveWriteBoundary(pData, size);
            size_t index = 0;
            while (index < size)
            {
                DeflateMatch match = FindAggressiveMatch(pData, index, size, m_streamPosition);
                if (match.length != 0 && index + 1 < size)
                {
                    const DeflateMatch nextMatch = FindAggressiveMatch(pData, index + 1, size, m_streamPosition + 1);
                    const bool nextMatchPreferred = PreferNextMatch(match, nextMatch);
                    if (nextMatchPreferred)
                    {
                        match = DeflateMatch();
                    }
                }

                const size_t consumed = match.length != 0 ? match.length : 1;
                if (match.length != 0)
                {
                    const bool matchSucceeded = WriteLengthDistance(output, match.length, match.distance);
                    if (!matchSucceeded)
                    {
                        return false;
                    }
                }
                else
                {
                    const bool literalSucceeded = WriteFixedSymbol(output, pData[index]);
                    if (!literalSucceeded)
                    {
                        return false;
                    }
                }

                for (size_t consumedIndex = 0; consumedIndex < consumed; ++consumedIndex)
                {
                    const uint64_t insertedPosition = m_streamPosition;
                    InsertByte(insertedPosition, pData[index + consumedIndex]);
                    if (index + consumedIndex + 2 < size)
                    {
                        InsertAggressiveHash(insertedPosition, pData + index + consumedIndex);
                    }
                    ++m_streamPosition;
                }
                index += consumed;
            }
            m_current.compressedSize += m_archiveOffset - before;
            return true;
        }

        template <typename TOutput>
        bool Deflate(TOutput& output, const uint8_t* pData, size_t size)
        {
            if (m_deflateStrategy == DeflateStrategy::Compact)
            {
                return DeflateAggressive(output, pData, size);
            }
            return DeflateFast(output, pData, size);
        }

        void ResetDeflateCommon()
        {
            std::fill(m_hashHeads.begin(), m_hashHeads.end(), -1);
            std::fill(m_history.begin(), m_history.end(), uint8_t{0});
            m_streamPosition = 0;
            m_bitBuffer = 0;
            m_bitCount = 0;
        }

        void ResetDeflateAggressive()
        {
            std::fill(m_hashPrevious.begin(), m_hashPrevious.end(), -1);
        }

        void ResetDeflater()
        {
            ResetDeflateCommon();
            if (m_deflateStrategy == DeflateStrategy::Compact)
            {
                ResetDeflateAggressive();
            }
        }

        template <typename TOutput>
        bool Finish(TOutput& output)
        {
            if (m_finished || m_entryOpen || m_failed)
            {
                return Fail();
            }
            const uint64_t directoryOffset = m_archiveOffset;
            for (const CentralEntry& entry : m_entries)
            {
                const bool entrySucceeded = WriteCentralEntry(output, entry);
                if (!entrySucceeded)
                {
                    return false;
                }
            }
            const uint64_t directorySize = m_archiveOffset - directoryOffset;
            const bool zip64 = m_entries.size() >= kZip16Sentinel || NeedsZip64(directoryOffset) || NeedsZip64(directorySize);
            if (zip64)
            {
                const bool zip64Succeeded = WriteZip64End(output, directoryOffset, directorySize);
                if (!zip64Succeeded)
                {
                    return false;
                }
            }

            std::vector<uint8_t> end;
            Put32(end, kEndOfCentralDirectorySignature);
            Put16(end, 0);
            Put16(end, 0);
            Put16(end, zip64 ? kZip16Sentinel : static_cast<uint16_t>(m_entries.size()));
            Put16(end, zip64 ? kZip16Sentinel : static_cast<uint16_t>(m_entries.size()));
            Put32(end, zip64 ? kZip32Sentinel : static_cast<uint32_t>(directorySize));
            Put32(end, zip64 ? kZip32Sentinel : static_cast<uint32_t>(directoryOffset));
            Put16(end, 0);
            const bool emitSucceeded = Emit(output, end.data(), end.size());
            if (!emitSucceeded)
            {
                return false;
            }
            const bool flushSucceeded = output.Flush();
            if (!flushSucceeded)
            {
                return Fail();
            }
            m_finished = true;
            return true;
        }

        template <typename TOutput>
        bool WriteCentralEntry(TOutput& output, const CentralEntry& entry)
        {
            const bool sizeZip64 = NeedsZip64(entry.compressedSize) || NeedsZip64(entry.uncompressedSize);
            const bool offsetZip64 = NeedsZip64(entry.localHeaderOffset);
            std::vector<uint8_t> extra;
            if (sizeZip64 || offsetZip64)
            {
                std::vector<uint8_t> payload;
                if (sizeZip64)
                {
                    Put64(payload, entry.uncompressedSize);
                    Put64(payload, entry.compressedSize);
                }
                if (offsetZip64)
                {
                    Put64(payload, entry.localHeaderOffset);
                }
                Put16(extra, kZip64ExtraId);
                Put16(extra, static_cast<uint16_t>(payload.size()));
                PutBytes(extra, payload.data(), payload.size());
            }

            std::vector<uint8_t> record;
            Put32(record, kCentralHeaderSignature);
            Put16(record, kVersionZip64);
            Put16(record, (sizeZip64 || offsetZip64) ? kVersionZip64 : kVersionDefault);
            Put16(record, entry.flags);
            Put16(record, static_cast<uint16_t>(entry.method));
            Put16(record, 0);
            Put16(record, 0);
            Put32(record, entry.crc32);
            Put32(record, sizeZip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.compressedSize));
            Put32(record, sizeZip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.uncompressedSize));
            Put16(record, static_cast<uint16_t>(entry.name.size()));
            Put16(record, static_cast<uint16_t>(extra.size()));
            Put16(record, 0);
            Put16(record, 0);
            Put16(record, 0);
            Put32(record, 0);
            Put32(record, offsetZip64 ? kZip32Sentinel : static_cast<uint32_t>(entry.localHeaderOffset));
            PutString(record, entry.name);
            PutBytes(record, extra.data(), extra.size());
            return Emit(output, record.data(), record.size());
        }

        template <typename TOutput>
        bool WriteZip64End(TOutput& output, uint64_t directoryOffset, uint64_t directorySize)
        {
            const uint64_t zip64EndOffset = m_archiveOffset;
            std::vector<uint8_t> record;
            Put32(record, kZip64EndSignature);
            Put64(record, 44);
            Put16(record, kVersionZip64);
            Put16(record, kVersionZip64);
            Put32(record, 0);
            Put32(record, 0);
            Put64(record, m_entries.size());
            Put64(record, m_entries.size());
            Put64(record, directorySize);
            Put64(record, directoryOffset);
            Put32(record, kZip64LocatorSignature);
            Put32(record, 0);
            Put64(record, zip64EndOffset);
            Put32(record, 1);
            return Emit(output, record.data(), record.size());
        }

        uint64_t m_archiveOffset = 0;
        bool m_initialized = false;
        bool m_entryOpen = false;
        bool m_finished = false;
        bool m_failed = false;
        DeflateStrategy m_deflateStrategy = DeflateStrategy::Fast;
        CentralEntry m_current;
        std::vector<CentralEntry> m_entries;

        std::vector<int64_t> m_hashHeads;
        std::vector<int64_t> m_hashPrevious;
        std::vector<uint8_t> m_history;
        uint64_t m_streamPosition = 0;
        uint64_t m_bitBuffer = 0;
        unsigned m_bitCount = 0;
    };

}
} //namespace nv::trace
