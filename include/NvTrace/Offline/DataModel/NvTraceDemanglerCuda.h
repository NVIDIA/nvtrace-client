/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace nv::trace {

    /// Dependency-free demangler for CUDA C++ names using the Itanium C++ ABI.
    /// Specification: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
    ///
    /// The object borrows both input and scratch storage.  Neither may be changed
    /// until parsing and all desired rendering have completed.  BindScratchSpace
    /// may be called again after ParseInput returns NeedMoreScratch; the plan is
    /// rebuilt into the new storage on the next ParseInput call.
    ///
    /// Lifecycle (BeginDemangling and ParseInput restart their respective phases):
    ///
    ///   BindScratchSpace --> BeginDemangling --> ParseInput --> Complete
    ///          ^                                  |
    ///          +---------- NeedMoreScratch -------+
    ///
    ///   Complete --> RenderDemangledName --> OutputLimitReached
    ///                    ^                    |
    ///                    +--------------------+
    ///
    ///   Complete --> FinishCountingRemainingOutputBytes
    ///                    (destructively consumes the remaining render state)
    ///
    /// Rebinding scratch invalidates the plan. Rendering is resumable and an
    /// exact-fit output is Complete; repeating ParseInput rebuilds the same plan.
    /// Grammar nesting and recursive plan traversal are both iterative. Parsing
    /// uses a scratch-backed ParseWorkFrame data stack; rendering reuses that same
    /// storage as a RenderFrame stack. Input depth never consumes C++ call stack.
    class CudaDemangler
    {
    public:
        enum class Status : uint8_t
        {
            Complete,
            NeedMoreScratch,
            OutputLimitReached,
            UnsupportedInput,
            InvalidInput,
            NotParsed,
        };

        CudaDemangler() = default;
        CudaDemangler(const CudaDemangler&) = delete;
        CudaDemangler& operator=(const CudaDemangler&) = delete;
        CudaDemangler(CudaDemangler&&) = delete;
        CudaDemangler& operator=(CudaDemangler&&) = delete;

        void BindScratchSpace(uint8_t* pScratchSpace, size_t scratchSpaceSize)
        {
            m_pScratchSpace = pScratchSpace;
            m_scratchSpaceSize = scratchSpaceSize;
            m_rootNodeIndex = kInvalidNodeIndex;
            m_parseStatus = Status::NotParsed;
            m_parseFailureStatus = Status::InvalidInput;
            ResetPlanStorage();
            ResetRenderState();
        }

        /// Records a new input. Parsing deliberately occurs in ParseInput.
        void BeginDemangling(const char* pMangledName, size_t mangledNameLength)
        {
            m_pInput = pMangledName;
            m_inputLength = mangledNameLength;
            m_inputIndex = 0;
            m_templateArgumentBaseIndex = 0;
            m_rootNodeIndex = kInvalidNodeIndex;
            m_parseStatus = Status::NotParsed;
            m_parseFailureStatus = Status::InvalidInput;
            m_requiredScratchSpaceSize = CalculateRequiredScratchSpaceSize(mangledNameLength);
            ResetPlanStorage();
            ResetRenderState();
        }

        Status ParseInput()
        {
            ResetPlanStorage();
            ResetRenderState();
            m_inputIndex = 0;
            m_templateArgumentBaseIndex = 0;
            m_parseFailureStatus = Status::InvalidInput;

            if (!m_pInput || m_inputLength < 3 || m_inputLength > UINT32_MAX
                || m_pInput[0] != '_' || m_pInput[1] != 'Z')
            {
                m_parseStatus = Status::InvalidInput;
                return m_parseStatus;
            }
            if (!PartitionScratchSpace())
            {
                m_parseStatus = Status::NeedMoreScratch;
                return m_parseStatus;
            }

            m_inputIndex = 2;
            if (IsUnsupportedTopLevelEntity())
            {
                m_parseStatus = Status::UnsupportedInput;
                return m_parseStatus;
            }
            const NodeIndex encodingNodeIndex = RunParser();
            if (encodingNodeIndex == kInvalidNodeIndex || m_inputIndex != m_inputLength)
            {
                m_parseStatus = encodingNodeIndex == kInvalidNodeIndex
                                  ? m_parseFailureStatus
                                  : Status::InvalidInput;
                return m_parseStatus;
            }

            m_rootNodeIndex = encodingNodeIndex;
            m_parseStatus = Status::Complete;
            return m_parseStatus;
        }

        size_t GetRequiredScratchSpaceSize() const
        {
            return m_requiredScratchSpaceSize;
        }

        size_t GetRenderedOutputLength() const
        {
            return m_renderedOutputLength;
        }

        /// Primarily useful when diagnosing malformed or unsupported encodings.
        size_t GetInputBytesConsumed() const
        {
            return m_inputIndex;
        }

        template <typename TAppendOutput>
        Status RenderDemangledName(size_t maxOutputLength, TAppendOutput&& appendOutput)
        {
            if (m_parseStatus != Status::Complete)
            {
                return m_parseStatus;
            }
            if (m_renderMode != RenderMode::Name)
            {
                InitializeRender(RenderMode::Name, static_cast<uint32_t>(-1));
            }
            return RenderWithLimit(maxOutputLength, std::forward<TAppendOutput>(appendOutput));
        }

        /// Emits a stable diagnostic representation. Its exact text is private
        /// implementation detail and may change with the plan representation.
        template <typename TAppendOutput>
        Status RenderPlanTree(size_t maxOutputLength, size_t maxTreeDepth, TAppendOutput&& appendOutput)
        {
            if (m_parseStatus != Status::Complete)
            {
                return m_parseStatus;
            }
            const uint32_t clampedDepth = maxTreeDepth > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(maxTreeDepth);
            if (m_renderMode != RenderMode::Tree || m_treeMaxDepth != clampedDepth)
            {
                InitializeRender(RenderMode::Tree, clampedDepth);
            }
            return RenderWithLimit(maxOutputLength, std::forward<TAppendOutput>(appendOutput));
        }

        /// Counts, consumes, and discards the unrendered portion of name output.
        Status FinishCountingRemainingOutputBytes(size_t& remainingOutputByteCount)
        {
            remainingOutputByteCount = 0;
            if (m_parseStatus != Status::Complete)
            {
                return m_parseStatus;
            }
            if (m_renderMode != RenderMode::Name)
            {
                InitializeRender(RenderMode::Name, static_cast<uint32_t>(-1));
            }
            while (!m_renderComplete)
            {
                const char* pFragment = nullptr;
                size_t fragmentLength = 0;
                if (!GetNextFragment(pFragment, fragmentLength))
                {
                    m_renderComplete = true;
                    break;
                }
                const size_t unconsumedLength = fragmentLength - m_fragmentOffset;
                if (remainingOutputByteCount > SIZE_MAX - unconsumedLength)
                {
                    return Status::InvalidInput;
                }
                remainingOutputByteCount += unconsumedLength;
                m_renderedOutputLength += unconsumedLength;
                m_fragmentOffset = fragmentLength;
            }
            return Status::Complete;
        }

    private:
        // Plan schema
        // -----------
        // Parsing creates an immutable, index-addressed tree in caller-owned
        // scratch space. NodeIndex is used instead of pointers so plan references
        // are compact and relocation after a scratch resize is unambiguous.
        //
        // Every Node has a kind and a forward-only child list:
        //
        //   firstChildIndex --> child --> nextSiblingIndex --> child --> invalid
        //
        // lastChildIndex makes construction O(1); rendering only needs forward
        // traversal. Identifier and literal text remains as an offset/length slice
        // of the borrowed input. The plan never copies source-name bytes.
        //
        // QualifiedName and declarator nodes preserve syntax needed to render C++
        // correctly. This is intentionally not an ownership hierarchy: namespace
        // components are children of QualifiedName so a prefix traversal emits
        // them in lexical order. Alias nodes refer to previously parsed nodes for
        // ABI substitutions without cloning their subtrees.
        //
        // Simple example:
        //
        //   _ZN2ns3fooEi                  ns::foo(int)
        //
        //   Function
        //     QualifiedName
        //       Identifier "ns"
        //       Identifier "foo"
        //     BuiltinType int
        //
        // Declarator example:
        //
        //   _Z1fA4_PFidE                 f(int (*[4])(double))
        //
        //   Function
        //     Identifier "f"
        //     ArrayType "4"
        //       PointerType
        //         FunctionType
        //           BuiltinType int
        //           BuiltinType double
        //
        // ParseWorkFrame holds a grammar production, its symbolic resume stage,
        // production-local values, and a child-result mailbox. Pushing one is the
        // iterative equivalent of calling a child grammar production.
        //
        // RenderFrame is the renderer's work frame. role selects normal,
        // left-declarator, or right-declarator rendering; stage records the next
        // operation when a child frame or output-limit pause returns control.
        // ParseWorkFrame and RenderFrame form scratch-backed data stacks and reuse
        // the same memory because the two phases never overlap.
        // Substitutions and template arguments are separate contiguous NodeIndex
        // tables because both require random access during parsing.

        using NodeIndex = uint32_t;
        static constexpr NodeIndex kInvalidNodeIndex = UINT32_MAX;

        enum class NodeKind : uint8_t
        {
            Identifier,
            BuiltinType,
            QualifiedName,
            TemplateId,
            TemplateArgumentPack,
            Function,
            PointerType,
            LValueReferenceType,
            RValueReferenceType,
            ConstType,
            VolatileType,
            RestrictType,
            ComplexType,
            ImaginaryType,
            VendorQualifierType,
            ArrayType,
            FunctionType,
            MemberPointerType,
            Literal,
            OperatorName,
            Constructor,
            Destructor,
            Lambda,
            UnnamedType,
            Decltype,
            FunctionParameter,
            PrefixExpression,
            ConversionOperator,
            Alias,
        };

        enum NodeFlags : uint8_t
        {
            NodeFlagNone = 0,
            NodeFlagFunctionHasReturnType = 1,
            NodeFlagNegativeLiteral = 2,
            NodeFlagFunctionConst = 4,
            NodeFlagFunctionVolatile = 8,
            NodeFlagFunctionRestrict = 16,
            NodeFlagFunctionLValueReference = 32,
            NodeFlagFunctionRValueReference = 64,
            NodeFlagFunctionExternC = 128,
        };

        static constexpr uint8_t NodeFlagArrayDimensionExpression = 1;

        struct Node
        {
            NodeIndex firstChildIndex;
            NodeIndex lastChildIndex;
            NodeIndex nextSiblingIndex;
            uint32_t inputOffset;
            uint32_t inputLength;
            NodeKind kind;
            uint8_t flags;
            uint16_t value;
        };

        struct RenderFrame
        {
            NodeIndex nodeIndex;
            NodeIndex childIndex;
            uint32_t depth;
            uint32_t childOrdinal;
            uint16_t stage;
            uint8_t role;
        };

        enum class ParseProduction : uint8_t
        {
            Encoding,
            Name,
            NestedName,
            LocalName,
            UnqualifiedName,
            TemplateArguments,
            TemplateArgument,
            Type,
            FunctionType,
            ArrayType,
            MemberPointerType,
            ExtendedType,
            Literal,
            Expression,
            OperatorName,
            LambdaName,
            Decltype,
            StripFunctionQualifiers,
        };

        struct ParseWorkFrame
        {
            NodeIndex nodeIndex;
            NodeIndex secondaryNodeIndex;
            NodeIndex childResult;
            size_t savedValue;
            size_t secondarySavedValue;
            uint16_t childOrdinal;
            ParseProduction production;
            uint8_t stage;
            uint8_t flags;
            uint8_t childFlags;
        };

        enum class RenderRole : uint8_t
        {
            Normal,
            TypeLeft,
            TypeRight,
        };

        enum class RenderMode : uint8_t
        {
            None,
            Name,
            Tree,
        };

        enum TypeLeftRenderStage : uint16_t
        {
            TypeLeftStart,
            TypeLeftFirstStepComplete,
            TypeLeftSecondStepComplete,
            TypeLeftThirdStepComplete,
            TypeLeftFourthStepComplete,
        };

        enum TypeRightRenderStage : uint16_t
        {
            TypeRightStart,
            TypeRightFirstStepComplete,
            TypeRightSecondStepComplete,
            TypeRightThirdStepComplete,
            TypeRightFourthStepComplete,
            TypeRightFifthStepComplete,
        };

        enum NameRenderStage : uint16_t
        {
            NameRenderStart,
            NameRenderPrefixComplete,
            NameRenderValueComplete,
            NameRenderSuffixComplete,
        };

        enum CompositeRenderStage : uint16_t
        {
            CompositeRenderStart = 0,
            CompositeRenderOpenComplete = 1,
            CompositeRenderChildren = 2,
            CompositeRenderSeparator = 3,
            LambdaRenderChildren = 4,
            LambdaRenderSeparator = 5,
            FunctionRenderReturnTypeComplete = 10,
            FunctionRenderReturnSeparatorComplete = 11,
            FunctionRenderConstQualifier = 20,
            FunctionRenderVolatileQualifier = 21,
            FunctionRenderRestrictQualifier = 22,
            FunctionRenderLValueQualifier = 23,
            FunctionRenderRValueQualifier = 24,
            FunctionRenderQualifiersComplete = 25,
        };

        enum TreeRenderStage : uint16_t
        {
            TreeRenderIndentation,
            TreeRenderNodeKind,
            TreeRenderDetailPrefix,
            TreeRenderDetailValue,
            TreeRenderDetailSuffix,
            TreeRenderNewline,
            TreeRenderChildren,
        };

        static constexpr uint16_t kInitialRenderStage = 0;

    private:
        // Scratch-space and plan-construction helpers.

        static size_t AlignUp(size_t value, size_t alignment)
        {
            const size_t alignmentMask = alignment - 1;
            return (value + alignmentMask) & ~alignmentMask;
        }

        static bool TryMultiply(size_t leftValue, size_t rightValue, size_t& result)
        {
            if (leftValue != 0 && rightValue > SIZE_MAX / leftValue)
            {
                result = SIZE_MAX;
                return false;
            }
            result = leftValue * rightValue;
            return true;
        }

        static size_t CalculateRequiredScratchSpaceSize(size_t inputLength)
        {
            size_t nodeBytes = 0;
            size_t indexBytes = 0;
            size_t frameBytes = 0;
            const size_t nodeCapacity = inputLength <= (SIZE_MAX - 64) / 8 ? inputLength * 8 + 64 : SIZE_MAX;
            const size_t indexCapacity = inputLength <= SIZE_MAX - 16 ? inputLength + 16 : SIZE_MAX;
            const size_t frameCapacity = inputLength <= (SIZE_MAX - 64) / 2 ? inputLength * 2 + 64 : SIZE_MAX;
            const size_t workFrameSize = sizeof(ParseWorkFrame) > sizeof(RenderFrame) ? sizeof(ParseWorkFrame) : sizeof(RenderFrame);
            size_t combinedIndexCapacity = 0;
            if (!TryMultiply(nodeCapacity, sizeof(Node), nodeBytes)
                || !TryMultiply(indexCapacity, 2, combinedIndexCapacity)
                || !TryMultiply(combinedIndexCapacity, sizeof(NodeIndex), indexBytes)
                || !TryMultiply(frameCapacity, workFrameSize, frameBytes))
            {
                return SIZE_MAX;
            }
            const size_t workFrameAlignment = alignof(ParseWorkFrame) > alignof(RenderFrame) ? alignof(ParseWorkFrame) : alignof(RenderFrame);
            const size_t alignmentSlop = alignof(Node) + alignof(NodeIndex) * 2 + workFrameAlignment;
            if (nodeBytes > SIZE_MAX - indexBytes || nodeBytes + indexBytes > SIZE_MAX - frameBytes
                || nodeBytes + indexBytes + frameBytes > SIZE_MAX - alignmentSlop)
            {
                return SIZE_MAX;
            }
            return nodeBytes + indexBytes + frameBytes + alignmentSlop;
        }

        bool PartitionScratchSpace()
        {
            if (!m_pScratchSpace || m_requiredScratchSpaceSize == SIZE_MAX || m_scratchSpaceSize < m_requiredScratchSpaceSize)
            {
                return false;
            }

            const size_t nodeCapacity = m_inputLength * 8 + 64;
            const size_t indexCapacity = m_inputLength + 16;
            const size_t frameCapacity = m_inputLength * 2 + 64;
            uintptr_t scratchAddress = reinterpret_cast<uintptr_t>(m_pScratchSpace);
            uintptr_t alignedAddress = AlignUp(scratchAddress, alignof(Node));
            m_pNodes = reinterpret_cast<Node*>(alignedAddress);
            m_nodeCapacity = nodeCapacity;
            alignedAddress += nodeCapacity * sizeof(Node);
            alignedAddress = AlignUp(alignedAddress, alignof(NodeIndex));
            m_pSubstitutions = reinterpret_cast<NodeIndex*>(alignedAddress);
            m_substitutionCapacity = indexCapacity;
            alignedAddress += indexCapacity * sizeof(NodeIndex);
            alignedAddress = AlignUp(alignedAddress, alignof(NodeIndex));
            m_pTemplateArguments = reinterpret_cast<NodeIndex*>(alignedAddress);
            m_templateArgumentCapacity = indexCapacity;
            alignedAddress += indexCapacity * sizeof(NodeIndex);
            const size_t workFrameAlignment = alignof(ParseWorkFrame) > alignof(RenderFrame) ? alignof(ParseWorkFrame) : alignof(RenderFrame);
            alignedAddress = AlignUp(alignedAddress, workFrameAlignment);
            m_pParseWorkFrames = reinterpret_cast<ParseWorkFrame*>(alignedAddress);
            m_parseWorkFrameCapacity = frameCapacity;
            m_pRenderFrames = reinterpret_cast<RenderFrame*>(alignedAddress);
            m_renderFrameCapacity = frameCapacity;
            return true;
        }

        void ResetPlanStorage()
        {
            m_pNodes = nullptr;
            m_nodeCapacity = 0;
            m_nodeCount = 0;
            m_pSubstitutions = nullptr;
            m_substitutionCapacity = 0;
            m_substitutionCount = 0;
            m_pTemplateArguments = nullptr;
            m_templateArgumentCapacity = 0;
            m_templateArgumentCount = 0;
            m_pParseWorkFrames = nullptr;
            m_parseWorkFrameCapacity = 0;
            m_parseWorkFrameCount = 0;
            m_pRenderFrames = nullptr;
            m_renderFrameCapacity = 0;
        }

        void ResetRenderState()
        {
            m_renderMode = RenderMode::None;
            m_renderFrameCount = 0;
            m_renderedOutputLength = 0;
            m_fragmentOffset = 0;
            m_pCurrentFragment = nullptr;
            m_currentFragmentLength = 0;
            m_renderComplete = false;
            m_treeMaxDepth = UINT32_MAX;
        }

        bool HasInput(size_t byteCount = 1) const
        {
            return byteCount <= m_inputLength - (m_inputIndex <= m_inputLength ? m_inputIndex : m_inputLength);
        }

        char PeekInput(size_t lookahead = 0) const
        {
            const size_t peekIndex = m_inputIndex + lookahead;
            return peekIndex < m_inputLength ? m_pInput[peekIndex] : '\0';
        }

        bool IsUnsupportedTopLevelEntity() const
        {
            if (!HasInput(2))
            {
                return false;
            }
            const char firstCode = PeekInput();
            const char secondCode = PeekInput(1);
            return (firstCode == 'T'
                    && (secondCode == 'V' || secondCode == 'T' || secondCode == 'I'
                        || secondCode == 'S' || secondCode == 'h' || secondCode == 'v'
                        || secondCode == 'c'))
                || (firstCode == 'G' && (secondCode == 'V' || secondCode == 'R'));
        }

        void MarkUnsupportedInput()
        {
            m_parseFailureStatus = Status::UnsupportedInput;
        }

        bool ConsumeInput(char expectedByte)
        {
            if (PeekInput() != expectedByte)
            {
                return false;
            }
            ++m_inputIndex;
            return true;
        }

        NodeIndex AddNode(NodeKind kind, uint32_t inputOffset = 0, uint32_t inputLength = 0, uint16_t value = 0, uint8_t flags = 0)
        {
            if (m_nodeCount >= m_nodeCapacity)
            {
                return kInvalidNodeIndex;
            }
            const NodeIndex nodeIndex = static_cast<NodeIndex>(m_nodeCount++);
            Node& node = m_pNodes[nodeIndex];
            node.firstChildIndex = kInvalidNodeIndex;
            node.lastChildIndex = kInvalidNodeIndex;
            node.nextSiblingIndex = kInvalidNodeIndex;
            node.inputOffset = inputOffset;
            node.inputLength = inputLength;
            node.kind = kind;
            node.flags = flags;
            node.value = value;
            return nodeIndex;
        }

        bool AddChild(NodeIndex parentNodeIndex, NodeIndex childNodeIndex)
        {
            if (parentNodeIndex == kInvalidNodeIndex || childNodeIndex == kInvalidNodeIndex)
            {
                return false;
            }
            Node& parentNode = m_pNodes[parentNodeIndex];
            if (parentNode.firstChildIndex == kInvalidNodeIndex)
            {
                parentNode.firstChildIndex = childNodeIndex;
            }
            else
            {
                m_pNodes[parentNode.lastChildIndex].nextSiblingIndex = childNodeIndex;
            }
            parentNode.lastChildIndex = childNodeIndex;
            return true;
        }

        bool AddSubstitution(NodeIndex nodeIndex)
        {
            if (nodeIndex == kInvalidNodeIndex || m_substitutionCount >= m_substitutionCapacity)
            {
                return false;
            }
            m_pSubstitutions[m_substitutionCount++] = nodeIndex;
            return true;
        }

        bool AddTemplateArgument(NodeIndex nodeIndex)
        {
            if (nodeIndex == kInvalidNodeIndex || m_templateArgumentCount >= m_templateArgumentCapacity)
            {
                return false;
            }
            m_pTemplateArguments[m_templateArgumentCount++] = nodeIndex;
            return true;
        }

        NodeIndex AddUnaryNode(NodeKind kind, NodeIndex childNodeIndex)
        {
            const NodeIndex nodeIndex = AddNode(kind);
            if (!AddChild(nodeIndex, childNodeIndex))
            {
                return kInvalidNodeIndex;
            }
            return nodeIndex;
        }

        NodeIndex AddQualifiedNode(NodeIndex leftNodeIndex, NodeIndex rightNodeIndex)
        {
            const NodeIndex nodeIndex = AddNode(NodeKind::QualifiedName);
            if (!AddChild(nodeIndex, leftNodeIndex) || !AddChild(nodeIndex, rightNodeIndex))
            {
                return kInvalidNodeIndex;
            }
            return nodeIndex;
        }

    private:
        // Parsing is a grammar-directed trampoline. A production handler records
        // its symbolic resume stage, pushes a ParseWorkFrame for the child, and
        // returns to the dispatch loop; handlers never recursively call each other.
        // Successful productions append immutable nodes and advance m_inputIndex.
        // ParseInput resets the arena view and restarts from byte zero, making a
        // retry after NeedMoreScratch deterministic.

        bool ParseUnsignedNumber(size_t& numberValue)
        {
            if (PeekInput() < '0' || PeekInput() > '9')
            {
                return false;
            }
            size_t parsedValue = 0;
            while ('0' <= PeekInput() && PeekInput() <= '9')
            {
                const size_t digitValue = static_cast<size_t>(PeekInput() - '0');
                if (parsedValue > (SIZE_MAX - digitValue) / 10)
                {
                    return false;
                }
                parsedValue = parsedValue * 10 + digitValue;
                ++m_inputIndex;
            }
            numberValue = parsedValue;
            return true;
        }

        bool ParseBase36Sequence(size_t& sequenceValue)
        {
            size_t parsedValue = 0;
            bool consumedDigit = false;
            while (true)
            {
                const char inputByte = PeekInput();
                size_t digitValue = 0;
                if ('0' <= inputByte && inputByte <= '9')
                {
                    digitValue = static_cast<size_t>(inputByte - '0');
                }
                else if ('A' <= inputByte && inputByte <= 'Z')
                {
                    digitValue = static_cast<size_t>(inputByte - 'A' + 10);
                }
                else
                {
                    break;
                }
                if (parsedValue > (SIZE_MAX - digitValue) / 36)
                {
                    return false;
                }
                parsedValue = parsedValue * 36 + digitValue;
                consumedDigit = true;
                ++m_inputIndex;
            }
            sequenceValue = parsedValue;
            return consumedDigit;
        }


        NodeIndex ParseSourceName()
        {
            size_t sourceLength = 0;
            if (!ParseUnsignedNumber(sourceLength) || sourceLength == 0 || !HasInput(sourceLength)
                || sourceLength > UINT32_MAX || m_inputIndex > UINT32_MAX)
            {
                return kInvalidNodeIndex;
            }
            const uint32_t inputOffset = static_cast<uint32_t>(m_inputIndex);
            if (!IsValidUtf8(m_pInput + m_inputIndex, sourceLength))
            {
                return kInvalidNodeIndex;
            }
            m_inputIndex += sourceLength;
            return AddNode(NodeKind::Identifier, inputOffset, static_cast<uint32_t>(sourceLength));
        }

        NodeIndex ParseUnnamedTypeName()
        {
            if (!ConsumeInput('U') || !ConsumeInput('t'))
            {
                return kInvalidNodeIndex;
            }
            size_t ordinal = 1;
            if ('0' <= PeekInput() && PeekInput() <= '9')
            {
                size_t sequenceNumber = 0;
                if (!ParseUnsignedNumber(sequenceNumber) || sequenceNumber > UINT32_MAX - 2)
                {
                    return kInvalidNodeIndex;
                }
                ordinal = sequenceNumber + 2;
            }
            if (!ConsumeInput('_'))
            {
                return kInvalidNodeIndex;
            }
            return AddNode(NodeKind::UnnamedType, 0, static_cast<uint32_t>(ordinal));
        }

        NodeIndex ParseFunctionParameter()
        {
            if (!ConsumeInput('f'))
            {
                return kInvalidNodeIndex;
            }
            size_t scopeLevel = 0;
            if (ConsumeInput('p'))
            {
                if (ConsumeInput('T'))
                {
                    return AddNode(NodeKind::FunctionParameter, 0, 0, kFunctionParameterThis);
                }
            }
            else
            {
                if (!ConsumeInput('L'))
                {
                    return kInvalidNodeIndex;
                }
                size_t encodedScopeLevel = 0;
                if (!ParseUnsignedNumber(encodedScopeLevel) || encodedScopeLevel >= UINT32_MAX
                    || !ConsumeInput('p'))
                {
                    return kInvalidNodeIndex;
                }
                scopeLevel = encodedScopeLevel + 1;
            }

            uint8_t qualifierFlags = NodeFlagNone;
            while (PeekInput() == 'r' || PeekInput() == 'V' || PeekInput() == 'K')
            {
                qualifierFlags |= PeekInput() == 'r' ? NodeFlagFunctionRestrict
                                : PeekInput() == 'V' ? NodeFlagFunctionVolatile
                                                     : NodeFlagFunctionConst;
                ++m_inputIndex;
            }

            size_t parameterOrdinal = 1;
            if ('0' <= PeekInput() && PeekInput() <= '9')
            {
                size_t parameterNumber = 0;
                if (!ParseUnsignedNumber(parameterNumber) || parameterNumber > UINT32_MAX - 2)
                {
                    return kInvalidNodeIndex;
                }
                parameterOrdinal = parameterNumber + 2;
            }
            if (!ConsumeInput('_'))
            {
                return kInvalidNodeIndex;
            }
            return AddNode(
                NodeKind::FunctionParameter,
                static_cast<uint32_t>(scopeLevel),
                static_cast<uint32_t>(parameterOrdinal),
                0,
                qualifierFlags);
        }


        NodeIndex ParseTemplateParameter()
        {
            if (!ConsumeInput('T'))
            {
                return kInvalidNodeIndex;
            }
            size_t parameterIndex = 0;
            if (ConsumeInput('_'))
            {
                parameterIndex = 0;
            }
            else
            {
                size_t sequenceValue = 0;
                if (!ParseUnsignedNumber(sequenceValue) || !ConsumeInput('_') || sequenceValue == SIZE_MAX)
                {
                    return kInvalidNodeIndex;
                }
                parameterIndex = sequenceValue + 1;
            }
            const size_t absoluteParameterIndex = m_templateArgumentBaseIndex + parameterIndex;
            if (absoluteParameterIndex < m_templateArgumentBaseIndex || absoluteParameterIndex >= m_templateArgumentCount)
            {
                return kInvalidNodeIndex;
            }
            return AddUnaryNode(NodeKind::Alias, m_pTemplateArguments[absoluteParameterIndex]);
        }

        NodeIndex ParseSubstitution()
        {
            if (!ConsumeInput('S'))
            {
                return kInvalidNodeIndex;
            }
            const char standardCode = PeekInput();
            if ('a' <= standardCode && standardCode <= 't')
            {
                ++m_inputIndex;
                const char* pStandardName = nullptr;
                switch (standardCode)
                {
                    case 't': pStandardName = "std"; break;
                    case 'a': pStandardName = "std::allocator"; break;
                    case 'b': pStandardName = "std::basic_string"; break;
                    case 's': pStandardName = "std::string"; break;
                    case 'i': pStandardName = "std::istream"; break;
                    case 'o': pStandardName = "std::ostream"; break;
                    case 'd': pStandardName = "std::iostream"; break;
                    default: return kInvalidNodeIndex;
                }
                return AddNode(NodeKind::BuiltinType, 0, 0, StandardNameValue(pStandardName));
            }
            size_t substitutionIndex = 0;
            if (ConsumeInput('_'))
            {
                substitutionIndex = 0;
            }
            else
            {
                size_t sequenceValue = 0;
                if (!ParseBase36Sequence(sequenceValue) || !ConsumeInput('_') || sequenceValue == SIZE_MAX)
                {
                    return kInvalidNodeIndex;
                }
                substitutionIndex = sequenceValue + 1;
            }
            if (substitutionIndex >= m_substitutionCount)
            {
                return kInvalidNodeIndex;
            }
            return AddUnaryNode(NodeKind::Alias, m_pSubstitutions[substitutionIndex]);
        }


        enum ParseFrameFlags : uint8_t
        {
            ParseFlagNone = 0,
            ParseFlagPrimary = 1,
            ParseFlagSecondary = 2,
            ParseFlagTertiary = 4,
            ParseFlagFourth = 8,
            ParseFlagFifth = 16,
            ParseFlagHadPrefix = 32,
            ParseFlagFromSubstitution = 64,
            ParseFlagRetainTemplateArguments = 128,
        };

        static constexpr uint8_t kNestedNameTemplateArgumentsCompleteStage = 2;
        static constexpr uint8_t kTypeTemplateArgumentsCompleteStage = 8;
        static constexpr uint8_t kExtendedTypeChildCompleteStage = 1;
        static constexpr uint8_t kOperatorNameTypeCompleteStage = 1;

        bool PushParseWorkFrame(ParseProduction production, NodeIndex nodeIndex = kInvalidNodeIndex, uint8_t flags = ParseFlagNone)
        {
            if (m_parseWorkFrameCount >= m_parseWorkFrameCapacity)
            {
                return false;
            }
            ParseWorkFrame& workFrame = m_pParseWorkFrames[m_parseWorkFrameCount++];
            workFrame.nodeIndex = nodeIndex;
            workFrame.secondaryNodeIndex = kInvalidNodeIndex;
            workFrame.childResult = kInvalidNodeIndex;
            workFrame.savedValue = 0;
            workFrame.secondarySavedValue = 0;
            workFrame.childOrdinal = 0;
            workFrame.production = production;
            workFrame.stage = 0;
            workFrame.flags = flags;
            workFrame.childFlags = ParseFlagNone;
            return true;
        }

        void CompleteParseWorkFrame(NodeIndex result, uint8_t resultFlags = ParseFlagNone)
        {
            --m_parseWorkFrameCount;
            if (m_parseWorkFrameCount == 0)
            {
                m_parseResult = result;
                return;
            }
            ParseWorkFrame& parentFrame = m_pParseWorkFrames[m_parseWorkFrameCount - 1];
            parentFrame.childResult = result;
            parentFrame.childFlags = resultFlags;
        }

        void PushParseChildOrFail(ParseWorkFrame& workFrame, uint8_t resumeStage, ParseProduction production, NodeIndex nodeIndex = kInvalidNodeIndex, uint8_t flags = ParseFlagNone)
        {
            workFrame.stage = resumeStage;
            if (!PushParseWorkFrame(production, nodeIndex, flags))
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
            }
        }

        NodeIndex RunParser()
        {
            m_parseWorkFrameCount = 0;
            m_parseResult = kInvalidNodeIndex;
            if (!PushParseWorkFrame(ParseProduction::Encoding))
            {
                return kInvalidNodeIndex;
            }
            while (m_parseWorkFrameCount != 0)
            {
                StepParseWorkFrame(m_pParseWorkFrames[m_parseWorkFrameCount - 1]);
            }
            return m_parseResult;
        }

        void StepParseWorkFrame(ParseWorkFrame& workFrame)
        {
            switch (workFrame.production)
            {
                case ParseProduction::Encoding: StepParseEncoding(workFrame); break;
                case ParseProduction::Name: StepParseName(workFrame); break;
                case ParseProduction::NestedName: StepParseNestedName(workFrame); break;
                case ParseProduction::LocalName: StepParseLocalName(workFrame); break;
                case ParseProduction::UnqualifiedName: StepParseUnqualifiedName(workFrame); break;
                case ParseProduction::TemplateArguments: StepParseTemplateArguments(workFrame); break;
                case ParseProduction::TemplateArgument: StepParseTemplateArgument(workFrame); break;
                case ParseProduction::Type: StepParseType(workFrame); break;
                case ParseProduction::FunctionType: StepParseFunctionType(workFrame); break;
                case ParseProduction::ArrayType: StepParseArrayType(workFrame); break;
                case ParseProduction::MemberPointerType: StepParseMemberPointerType(workFrame); break;
                case ParseProduction::ExtendedType: StepParseExtendedType(workFrame); break;
                case ParseProduction::Literal: StepParseLiteral(workFrame); break;
                case ParseProduction::Expression: StepParseExpression(workFrame); break;
                case ParseProduction::OperatorName: StepParseOperatorName(workFrame); break;
                case ParseProduction::LambdaName: StepParseLambdaName(workFrame); break;
                case ParseProduction::Decltype: StepParseDecltype(workFrame); break;
                case ParseProduction::StripFunctionQualifiers: StepStripFunctionQualifiers(workFrame); break;
            }
        }

        void StepParseEncoding(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                NameComplete,
                QualifiersComplete,
                ParseTypes,
                TypeComplete,
            };
            if (workFrame.stage == Start)
            {
                workFrame.savedValue = m_templateArgumentCount;
                PushParseChildOrFail(workFrame, NameComplete, ParseProduction::Name, kInvalidNodeIndex, ParseFlagRetainTemplateArguments);
                return;
            }
            if (workFrame.stage == NameComplete)
            {
                if (workFrame.childResult == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, QualifiersComplete, ParseProduction::StripFunctionQualifiers, workFrame.childResult);
                return;
            }
            if (workFrame.stage == QualifiersComplete)
            {
                const NodeIndex nameNodeIndex = workFrame.childResult;
                if (nameNodeIndex == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (!HasInput() || PeekInput() == 'E')
                {
                    CompleteParseWorkFrame(nameNodeIndex);
                    return;
                }
                const NodeIndex functionNodeIndex = AddNode(NodeKind::Function);
                if (!AddChild(functionNodeIndex, nameNodeIndex))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (m_templateArgumentCount != workFrame.savedValue && NameEndsInTemplateId(nameNodeIndex))
                {
                    m_pNodes[functionNodeIndex].flags |= NodeFlagFunctionHasReturnType;
                }
                m_pNodes[functionNodeIndex].flags |= workFrame.childFlags;
                workFrame.nodeIndex = functionNodeIndex;
                workFrame.flags = ParseFlagNone;
                workFrame.stage = ParseTypes;
                return;
            }
            if (workFrame.stage == ParseTypes)
            {
                if (!HasInput() || PeekInput() == 'E')
                {
                    CompleteParseWorkFrame((workFrame.flags & ParseFlagPrimary) != 0 ? workFrame.nodeIndex : kInvalidNodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, TypeComplete, ParseProduction::Type);
                return;
            }
            if (workFrame.childResult == kInvalidNodeIndex || !AddChild(workFrame.nodeIndex, workFrame.childResult))
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            workFrame.flags |= ParseFlagPrimary;
            workFrame.stage = ParseTypes;
        }

        void StepParseName(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                DirectChildComplete,
                UnqualifiedNameComplete,
                StandardNameComplete,
                TemplateArgumentsComplete,
            };
            if (workFrame.stage == Start)
            {
                if (PeekInput() == 'N')
                {
                    PushParseChildOrFail(workFrame, DirectChildComplete, ParseProduction::NestedName, kInvalidNodeIndex, workFrame.flags);
                }
                else if (PeekInput() == 'Z')
                {
                    PushParseChildOrFail(workFrame, DirectChildComplete, ParseProduction::LocalName);
                }
                else if (PeekInput() == 'S' && PeekInput(1) == 't')
                {
                    workFrame.nodeIndex = ParseSubstitution();
                    PushParseChildOrFail(workFrame, StandardNameComplete, ParseProduction::UnqualifiedName);
                }
                else
                {
                    PushParseChildOrFail(workFrame, UnqualifiedNameComplete, ParseProduction::UnqualifiedName);
                }
                return;
            }
            if (workFrame.stage == DirectChildComplete || workFrame.stage == TemplateArgumentsComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            if (workFrame.stage == StandardNameComplete)
            {
                workFrame.childResult = AddQualifiedNode(workFrame.nodeIndex, workFrame.childResult);
            }
            if (workFrame.childResult == kInvalidNodeIndex)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (PeekInput() != 'I')
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            AddSubstitution(workFrame.childResult);
            PushParseChildOrFail(workFrame, TemplateArgumentsComplete, ParseProduction::TemplateArguments, workFrame.childResult, workFrame.flags);
        }

        void StepStripFunctionQualifiers(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                QualifierChildComplete,
                QualifiedChildComplete,
            };
            if (workFrame.stage == Start)
            {
                if (workFrame.nodeIndex == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                const Node& node = m_pNodes[workFrame.nodeIndex];
                uint8_t qualifierFlag = NodeFlagNone;
                if (node.kind == NodeKind::ConstType)
                {
                    qualifierFlag = NodeFlagFunctionConst;
                }
                else if (node.kind == NodeKind::VolatileType)
                {
                    qualifierFlag = NodeFlagFunctionVolatile;
                }
                else if (node.kind == NodeKind::RestrictType)
                {
                    qualifierFlag = NodeFlagFunctionRestrict;
                }
                else if (node.kind == NodeKind::LValueReferenceType)
                {
                    qualifierFlag = NodeFlagFunctionLValueReference;
                }
                else if (node.kind == NodeKind::RValueReferenceType)
                {
                    qualifierFlag = NodeFlagFunctionRValueReference;
                }
                if (qualifierFlag != NodeFlagNone)
                {
                    workFrame.flags = qualifierFlag;
                    PushParseChildOrFail(workFrame, QualifierChildComplete, ParseProduction::StripFunctionQualifiers, node.firstChildIndex);
                    return;
                }
                if (node.kind != NodeKind::QualifiedName || node.lastChildIndex == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(workFrame.nodeIndex);
                    return;
                }
                workFrame.secondaryNodeIndex = node.lastChildIndex;
                PushParseChildOrFail(workFrame, QualifiedChildComplete, ParseProduction::StripFunctionQualifiers, node.lastChildIndex);
                return;
            }
            if (workFrame.stage == QualifierChildComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult, static_cast<uint8_t>(workFrame.flags | workFrame.childFlags));
                return;
            }
            if (workFrame.childFlags == NodeFlagNone)
            {
                CompleteParseWorkFrame(workFrame.nodeIndex);
                return;
            }
            Node& node = m_pNodes[workFrame.nodeIndex];
            const NodeIndex oldLastChildIndex = workFrame.secondaryNodeIndex;
            const NodeIndex newLastChildIndex = workFrame.childResult;
            if (node.firstChildIndex == oldLastChildIndex)
            {
                node.firstChildIndex = newLastChildIndex;
            }
            else
            {
                NodeIndex previousChildIndex = node.firstChildIndex;
                while (previousChildIndex != kInvalidNodeIndex && m_pNodes[previousChildIndex].nextSiblingIndex != oldLastChildIndex)
                {
                    previousChildIndex = m_pNodes[previousChildIndex].nextSiblingIndex;
                }
                if (previousChildIndex == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                m_pNodes[previousChildIndex].nextSiblingIndex = newLastChildIndex;
            }
            m_pNodes[newLastChildIndex].nextSiblingIndex = kInvalidNodeIndex;
            node.lastChildIndex = newLastChildIndex;
            CompleteParseWorkFrame(workFrame.nodeIndex, workFrame.childFlags);
        }

        void CompleteNestedComponent(ParseWorkFrame& workFrame)
        {
            const NodeIndex componentNodeIndex = workFrame.childResult;
            if (componentNodeIndex == kInvalidNodeIndex)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            workFrame.secondaryNodeIndex = componentNodeIndex;
            if (PeekInput() == 'I')
            {
                const NodeIndex templatePrefixNodeIndex = workFrame.nodeIndex == kInvalidNodeIndex
                                                            ? componentNodeIndex
                                                            : AddQualifiedNode(workFrame.nodeIndex, componentNodeIndex);
                if (templatePrefixNodeIndex == kInvalidNodeIndex || !AddSubstitution(templatePrefixNodeIndex))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, kNestedNameTemplateArgumentsCompleteStage, ParseProduction::TemplateArguments, templatePrefixNodeIndex, workFrame.flags & ParseFlagRetainTemplateArguments);
                return;
            }
            workFrame.nodeIndex = workFrame.nodeIndex == kInvalidNodeIndex
                                    ? componentNodeIndex
                                    : AddQualifiedNode(workFrame.nodeIndex, componentNodeIndex);
            if (workFrame.nodeIndex == kInvalidNodeIndex)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            const bool reusedSubstitutionPrefix = (workFrame.flags & ParseFlagHadPrefix) == 0
                                               && (workFrame.flags & ParseFlagFromSubstitution) != 0;
            if (PeekInput() != 'E' && !reusedSubstitutionPrefix && !AddSubstitution(workFrame.nodeIndex))
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            workFrame.stage = 1;
        }

        void StepParseNestedName(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ParseComponent,
                TemplateArgumentsComplete,
                ComponentComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('N'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                while (PeekInput() == 'K' || PeekInput() == 'V' || PeekInput() == 'r' || PeekInput() == 'R' || PeekInput() == 'O')
                {
                    workFrame.flags |= PeekInput() == 'K' ? ParseFlagPrimary
                                     : PeekInput() == 'V' ? ParseFlagSecondary
                                     : PeekInput() == 'r' ? ParseFlagTertiary
                                     : PeekInput() == 'R' ? ParseFlagFourth
                                                          : ParseFlagFifth;
                    ++m_inputIndex;
                }
                workFrame.nodeIndex = kInvalidNodeIndex;
                workFrame.secondaryNodeIndex = kInvalidNodeIndex;
                workFrame.stage = ParseComponent;
                return;
            }
            if (workFrame.stage == TemplateArgumentsComplete)
            {
                if (workFrame.childResult == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.nodeIndex = workFrame.childResult;
                if (PeekInput() != 'E' && !AddSubstitution(workFrame.nodeIndex))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.stage = ParseComponent;
                return;
            }
            if (workFrame.stage == ComponentComplete)
            {
                CompleteNestedComponent(workFrame);
                return;
            }
            if (!HasInput())
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (PeekInput() == 'E')
            {
                ++m_inputIndex;
                NodeIndex accumulatedNodeIndex = workFrame.nodeIndex;
                if ((workFrame.flags & ParseFlagPrimary) != 0)
                {
                    accumulatedNodeIndex = AddUnaryNode(NodeKind::ConstType, accumulatedNodeIndex);
                }
                if ((workFrame.flags & ParseFlagSecondary) != 0)
                {
                    accumulatedNodeIndex = AddUnaryNode(NodeKind::VolatileType, accumulatedNodeIndex);
                }
                if ((workFrame.flags & ParseFlagTertiary) != 0)
                {
                    accumulatedNodeIndex = AddUnaryNode(NodeKind::RestrictType, accumulatedNodeIndex);
                }
                if ((workFrame.flags & ParseFlagFourth) != 0)
                {
                    accumulatedNodeIndex = AddUnaryNode(NodeKind::LValueReferenceType, accumulatedNodeIndex);
                }
                if ((workFrame.flags & ParseFlagFifth) != 0)
                {
                    accumulatedNodeIndex = AddUnaryNode(NodeKind::RValueReferenceType, accumulatedNodeIndex);
                }
                CompleteParseWorkFrame(accumulatedNodeIndex);
                return;
            }
            workFrame.flags &= static_cast<uint8_t>(ParseFlagPrimary | ParseFlagSecondary | ParseFlagTertiary | ParseFlagFourth | ParseFlagFifth | ParseFlagRetainTemplateArguments);
            if (workFrame.nodeIndex != kInvalidNodeIndex)
            {
                workFrame.flags |= ParseFlagHadPrefix;
            }
            if (PeekInput() == 'S')
            {
                workFrame.flags |= ParseFlagFromSubstitution;
                workFrame.childResult = ParseSubstitution();
                CompleteNestedComponent(workFrame);
                return;
            }
            if (PeekInput() == 'T')
            {
                workFrame.childResult = ParseTemplateParameter();
                CompleteNestedComponent(workFrame);
                return;
            }
            PushParseChildOrFail(workFrame, ComponentComplete, ParseProduction::UnqualifiedName, workFrame.secondaryNodeIndex);
        }

        void StepParseLocalName(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                EncodingComplete,
                EntityComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('Z'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.savedValue = m_templateArgumentCount;
                workFrame.secondarySavedValue = m_templateArgumentBaseIndex;
                m_templateArgumentBaseIndex = workFrame.savedValue;
                PushParseChildOrFail(workFrame, EncodingComplete, ParseProduction::Encoding);
                return;
            }
            if (workFrame.stage == EncodingComplete)
            {
                if (workFrame.childResult == kInvalidNodeIndex || !ConsumeInput('E'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.nodeIndex = workFrame.childResult;
                PushParseChildOrFail(workFrame, EntityComplete, ParseProduction::Name);
                return;
            }
            if (workFrame.childResult == kInvalidNodeIndex)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (PeekInput() == '_')
            {
                ++m_inputIndex;
                while ('0' <= PeekInput() && PeekInput() <= '9')
                {
                    ++m_inputIndex;
                }
            }
            const NodeIndex localNameNodeIndex = AddQualifiedNode(workFrame.nodeIndex, workFrame.childResult);
            AddSubstitution(localNameNodeIndex);
            m_templateArgumentCount = workFrame.savedValue;
            m_templateArgumentBaseIndex = workFrame.secondarySavedValue;
            CompleteParseWorkFrame(localNameNodeIndex);
        }

        void StepParseUnqualifiedName(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ChildComplete,
            };
            if (workFrame.stage == ChildComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            if ('0' <= PeekInput() && PeekInput() <= '9')
            {
                CompleteParseWorkFrame(ParseSourceName());
                return;
            }
            if ((PeekInput() == 'C' || PeekInput() == 'D') && '0' <= PeekInput(1) && PeekInput(1) <= '5')
            {
                const bool isConstructor = PeekInput() == 'C';
                m_inputIndex += 2;
                CompleteParseWorkFrame(workFrame.nodeIndex == kInvalidNodeIndex ? kInvalidNodeIndex : AddUnaryNode(isConstructor ? NodeKind::Constructor : NodeKind::Destructor, workFrame.nodeIndex));
                return;
            }
            if (PeekInput() == 'U' && PeekInput(1) == 'l')
            {
                PushParseChildOrFail(workFrame, ChildComplete, ParseProduction::LambdaName);
                return;
            }
            if (PeekInput() == 'U' && PeekInput(1) == 't')
            {
                CompleteParseWorkFrame(ParseUnnamedTypeName());
                return;
            }
            PushParseChildOrFail(workFrame, ChildComplete, ParseProduction::OperatorName);
        }

        void StepParseTemplateArguments(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ParseArguments,
                ArgumentComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('I'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                const NodeIndex templateNodeIndex = AddNode(NodeKind::TemplateId);
                if (!AddChild(templateNodeIndex, workFrame.nodeIndex))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.nodeIndex = templateNodeIndex;
                workFrame.savedValue = m_templateArgumentCount;
                workFrame.flags &= ParseFlagRetainTemplateArguments;
                workFrame.stage = ParseArguments;
                return;
            }
            if (workFrame.stage == ArgumentComplete)
            {
                if (workFrame.childResult == kInvalidNodeIndex || !AddChild(workFrame.nodeIndex, workFrame.childResult)
                    || !AddTemplateArgument(workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.flags |= ParseFlagPrimary;
                workFrame.stage = ParseArguments;
                return;
            }
            if (!HasInput())
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (PeekInput() == 'E')
            {
                if ((workFrame.flags & ParseFlagPrimary) == 0 || !ConsumeInput('E'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if ((workFrame.flags & ParseFlagRetainTemplateArguments) == 0)
                {
                    m_templateArgumentCount = workFrame.savedValue;
                }
                CompleteParseWorkFrame(workFrame.nodeIndex);
                return;
            }
            PushParseChildOrFail(workFrame, ArgumentComplete, ParseProduction::TemplateArgument);
        }

        void StepParseTemplateArgument(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                DirectChildComplete,
                ExpressionComplete,
                ParsePackArguments,
                PackArgumentComplete,
            };
            if (workFrame.stage == DirectChildComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            if (workFrame.stage == ExpressionComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult != kInvalidNodeIndex && ConsumeInput('E') ? workFrame.childResult : kInvalidNodeIndex);
                return;
            }
            if (workFrame.stage == PackArgumentComplete)
            {
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.stage = ParsePackArguments;
                return;
            }
            if (workFrame.stage == ParsePackArguments)
            {
                if (!HasInput())
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (PeekInput() == 'E')
                {
                    ++m_inputIndex;
                    CompleteParseWorkFrame(workFrame.nodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, PackArgumentComplete, ParseProduction::TemplateArgument);
                return;
            }
            if (PeekInput() == 'L')
            {
                PushParseChildOrFail(workFrame, DirectChildComplete, ParseProduction::Literal);
                return;
            }
            if (PeekInput() == 'X')
            {
                ++m_inputIndex;
                PushParseChildOrFail(workFrame, ExpressionComplete, ParseProduction::Expression);
                return;
            }
            if (PeekInput() == 'J')
            {
                ++m_inputIndex;
                workFrame.nodeIndex = AddNode(NodeKind::TemplateArgumentPack);
                workFrame.stage = ParsePackArguments;
                return;
            }
            PushParseChildOrFail(workFrame, DirectChildComplete, ParseProduction::Type);
        }

        void FinishParsedType(ParseWorkFrame& workFrame, NodeIndex typeNodeIndex)
        {
            if (typeNodeIndex == kInvalidNodeIndex)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            workFrame.nodeIndex = typeNodeIndex;
            if (PeekInput() == 'I')
            {
                PushParseChildOrFail(workFrame, kTypeTemplateArgumentsCompleteStage, ParseProduction::TemplateArguments, typeNodeIndex);
                return;
            }
            const char originalTypeCode = static_cast<char>(workFrame.savedValue);
            if (originalTypeCode != 'S' && originalTypeCode != 'Z')
            {
                AddSubstitution(typeNodeIndex);
            }
            CompleteParseWorkFrame(typeNodeIndex);
        }

        void StepParseType(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ExtendedTypeComplete,
                DecltypeComplete,
                VendorQualifiedTypeComplete,
                WrapperTypeComplete,
                SubstitutionSuffixComplete,
                GeneralTypeComplete,
                Reserved,
                TemplateArgumentsComplete = kTypeTemplateArgumentsCompleteStage,
            };
            if (workFrame.stage == ExtendedTypeComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            if (workFrame.stage == DecltypeComplete)
            {
                FinishParsedType(workFrame, workFrame.childResult);
                return;
            }
            if (workFrame.stage == VendorQualifiedTypeComplete)
            {
                const NodeIndex vendorQualifierNodeIndex = AddNode(NodeKind::VendorQualifierType);
                if (!AddChild(vendorQualifierNodeIndex, workFrame.nodeIndex) || !AddChild(vendorQualifierNodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                AddSubstitution(vendorQualifierNodeIndex);
                CompleteParseWorkFrame(vendorQualifierNodeIndex);
                return;
            }
            if (workFrame.stage == WrapperTypeComplete)
            {
                const NodeIndex wrapperNodeIndex = AddUnaryNode(static_cast<NodeKind>(workFrame.secondarySavedValue), workFrame.childResult);
                AddSubstitution(wrapperNodeIndex);
                CompleteParseWorkFrame(wrapperNodeIndex);
                return;
            }
            if (workFrame.stage == SubstitutionSuffixComplete)
            {
                FinishParsedType(workFrame, AddQualifiedNode(workFrame.nodeIndex, workFrame.childResult));
                return;
            }
            if (workFrame.stage == GeneralTypeComplete)
            {
                FinishParsedType(workFrame, workFrame.childResult);
                return;
            }
            if (workFrame.stage == TemplateArgumentsComplete)
            {
                const NodeIndex typeNodeIndex = workFrame.childResult;
                const char originalTypeCode = static_cast<char>(workFrame.savedValue);
                if (typeNodeIndex != kInvalidNodeIndex && originalTypeCode != 'S' && originalTypeCode != 'Z')
                {
                    AddSubstitution(typeNodeIndex);
                }
                CompleteParseWorkFrame(typeNodeIndex);
                return;
            }

            const char typeCode = PeekInput();
            workFrame.savedValue = static_cast<unsigned char>(typeCode);
            if (typeCode == 'D')
            {
                if (PeekInput(1) == 'o')
                {
                    m_inputIndex += 2;
                    PushParseChildOrFail(
                        workFrame,
                        GeneralTypeComplete,
                        ParseProduction::FunctionType,
                        kInvalidNodeIndex,
                        ParseFlagSecondary);
                    return;
                }
                if (PeekInput(1) == 't' || PeekInput(1) == 'T')
                {
                    PushParseChildOrFail(workFrame, DecltypeComplete, ParseProduction::Decltype);
                    return;
                }
                PushParseChildOrFail(workFrame, ExtendedTypeComplete, ParseProduction::ExtendedType);
                return;
            }
            if (typeCode == 'U' && PeekInput(1) == 't')
            {
                PushParseChildOrFail(workFrame, GeneralTypeComplete, ParseProduction::Name);
                return;
            }
            if (typeCode == 'u')
            {
                ++m_inputIndex;
                const NodeIndex vendorTypeNodeIndex = ParseSourceName();
                AddSubstitution(vendorTypeNodeIndex);
                CompleteParseWorkFrame(vendorTypeNodeIndex);
                return;
            }
            if (typeCode == 'U')
            {
                ++m_inputIndex;
                workFrame.nodeIndex = ParseSourceName();
                if (workFrame.nodeIndex == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, VendorQualifiedTypeComplete, ParseProduction::Type);
                return;
            }
            const uint16_t builtinValue = BuiltinTypeValue(typeCode);
            if (builtinValue != 0)
            {
                ++m_inputIndex;
                CompleteParseWorkFrame(AddNode(NodeKind::BuiltinType, 0, 0, builtinValue));
                return;
            }

            NodeKind wrapperKind = NodeKind::Alias;
            if (typeCode == 'P')
            {
                wrapperKind = NodeKind::PointerType;
            }
            else if (typeCode == 'R')
            {
                wrapperKind = NodeKind::LValueReferenceType;
            }
            else if (typeCode == 'O')
            {
                wrapperKind = NodeKind::RValueReferenceType;
            }
            else if (typeCode == 'K')
            {
                wrapperKind = NodeKind::ConstType;
            }
            else if (typeCode == 'V')
            {
                wrapperKind = NodeKind::VolatileType;
            }
            else if (typeCode == 'r')
            {
                wrapperKind = NodeKind::RestrictType;
            }
            else if (typeCode == 'C')
            {
                wrapperKind = NodeKind::ComplexType;
            }
            else if (typeCode == 'G')
            {
                wrapperKind = NodeKind::ImaginaryType;
            }
            if (wrapperKind != NodeKind::Alias)
            {
                ++m_inputIndex;
                workFrame.secondarySavedValue = static_cast<size_t>(wrapperKind);
                PushParseChildOrFail(workFrame, WrapperTypeComplete, ParseProduction::Type);
                return;
            }

            if (typeCode == 'T')
            {
                FinishParsedType(workFrame, ParseTemplateParameter());
                return;
            }
            if (typeCode == 'S')
            {
                const bool isStdNamespace = PeekInput(1) == 't';
                workFrame.nodeIndex = ParseSubstitution();
                if (workFrame.nodeIndex != kInvalidNodeIndex && isStdNamespace
                    && '0' <= PeekInput() && PeekInput() <= '9')
                {
                    PushParseChildOrFail(workFrame, SubstitutionSuffixComplete, ParseProduction::UnqualifiedName);
                    return;
                }
                FinishParsedType(workFrame, workFrame.nodeIndex);
                return;
            }
            if (typeCode == 'F')
            {
                PushParseChildOrFail(workFrame, GeneralTypeComplete, ParseProduction::FunctionType);
                return;
            }
            if (typeCode == 'A')
            {
                PushParseChildOrFail(workFrame, GeneralTypeComplete, ParseProduction::ArrayType);
                return;
            }
            if (typeCode == 'M')
            {
                PushParseChildOrFail(workFrame, GeneralTypeComplete, ParseProduction::MemberPointerType);
                return;
            }
            if (typeCode == 'N' || typeCode == 'Z' || ('0' <= typeCode && typeCode <= '9'))
            {
                PushParseChildOrFail(workFrame, GeneralTypeComplete, ParseProduction::Name);
                return;
            }
            CompleteParseWorkFrame(kInvalidNodeIndex);
        }

        void StepParseFunctionType(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ParseTypes,
                TypeComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('F'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                const bool hasExternCQualifier = ConsumeInput('Y');
                workFrame.nodeIndex = AddNode(NodeKind::FunctionType);
                if (workFrame.nodeIndex != kInvalidNodeIndex && (workFrame.flags & ParseFlagSecondary) != 0)
                {
                    m_pNodes[workFrame.nodeIndex].value |= kFunctionTypeNoexcept;
                }
                if (workFrame.nodeIndex != kInvalidNodeIndex && hasExternCQualifier)
                {
                    m_pNodes[workFrame.nodeIndex].flags |= NodeFlagFunctionExternC;
                }
                workFrame.stage = ParseTypes;
                return;
            }
            if (workFrame.stage == TypeComplete)
            {
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.flags |= ParseFlagPrimary;
                workFrame.stage = ParseTypes;
                return;
            }
            if (!HasInput())
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (PeekInput() == 'E')
            {
                CompleteParseWorkFrame((workFrame.flags & ParseFlagPrimary) != 0 && ConsumeInput('E') ? workFrame.nodeIndex : kInvalidNodeIndex);
                return;
            }
            PushParseChildOrFail(workFrame, TypeComplete, ParseProduction::Type);
        }

        void StepParseArrayType(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                DimensionExpressionComplete,
                ElementTypeComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('A'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.savedValue = m_inputIndex;
                workFrame.secondarySavedValue = 0;
                workFrame.secondaryNodeIndex = kInvalidNodeIndex;
                if ('0' <= PeekInput() && PeekInput() <= '9')
                {
                    while ('0' <= PeekInput() && PeekInput() <= '9')
                    {
                        ++m_inputIndex;
                        ++workFrame.secondarySavedValue;
                    }
                }
                else if (PeekInput() != '_')
                {
                    workFrame.flags |= ParseFlagPrimary;
                    PushParseChildOrFail(workFrame, DimensionExpressionComplete, ParseProduction::Expression);
                    return;
                }
                workFrame.stage = DimensionExpressionComplete;
            }
            if (workFrame.stage == DimensionExpressionComplete)
            {
                if ((workFrame.flags & ParseFlagPrimary) != 0 && workFrame.childResult == kInvalidNodeIndex)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (workFrame.childResult != kInvalidNodeIndex)
                {
                    workFrame.secondaryNodeIndex = workFrame.childResult;
                }
                if (!ConsumeInput('_') || workFrame.savedValue > UINT32_MAX || workFrame.secondarySavedValue > UINT32_MAX)
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.nodeIndex = AddNode(NodeKind::ArrayType, static_cast<uint32_t>(workFrame.savedValue), static_cast<uint32_t>(workFrame.secondarySavedValue));
                if (workFrame.secondaryNodeIndex != kInvalidNodeIndex)
                {
                    m_pNodes[workFrame.nodeIndex].flags |= NodeFlagArrayDimensionExpression;
                    if (!AddChild(workFrame.nodeIndex, workFrame.secondaryNodeIndex))
                    {
                        CompleteParseWorkFrame(kInvalidNodeIndex);
                        return;
                    }
                }
                PushParseChildOrFail(workFrame, ElementTypeComplete, ParseProduction::Type);
                return;
            }
            CompleteParseWorkFrame(AddChild(workFrame.nodeIndex, workFrame.childResult) ? workFrame.nodeIndex : kInvalidNodeIndex);
        }

        void StepParseMemberPointerType(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ClassTypeComplete,
                MemberTypeComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('M'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.nodeIndex = AddNode(NodeKind::MemberPointerType);
                PushParseChildOrFail(workFrame, ClassTypeComplete, ParseProduction::Type);
                return;
            }
            if (workFrame.stage == ClassTypeComplete)
            {
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, MemberTypeComplete, ParseProduction::Type);
                return;
            }
            CompleteParseWorkFrame(AddChild(workFrame.nodeIndex, workFrame.childResult) ? workFrame.nodeIndex : kInvalidNodeIndex);
        }

        void StepParseExtendedType(ParseWorkFrame& workFrame)
        {
            if (workFrame.stage != 0)
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            if (PeekInput() == 'D' && PeekInput(1) == 'h')
            {
                m_inputIndex += 2;
                CompleteParseWorkFrame(AddNode(NodeKind::BuiltinType, 0, 0, kBuiltinHalf));
                return;
            }
            if (PeekInput() == 'D' && PeekInput(1) == 'u')
            {
                m_inputIndex += 2;
                CompleteParseWorkFrame(ParseSourceName());
                return;
            }
            if (PeekInput() == 'D' && PeekInput(1) == 'p')
            {
                m_inputIndex += 2;
                PushParseChildOrFail(workFrame, kExtendedTypeChildCompleteStage, ParseProduction::Type);
                return;
            }
            if (PeekInput() == 'D' && HasInput(2))
            {
                // A well-formed extended-type introducer which is not in the
                // CUDA-focused subset is eligible for the platform fallback.
                MarkUnsupportedInput();
            }
            CompleteParseWorkFrame(kInvalidNodeIndex);
        }

        void StepParseDecltype(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ExpressionComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('D') || (PeekInput() != 't' && PeekInput() != 'T'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                const char decltypeCode = PeekInput();
                ++m_inputIndex;
                workFrame.nodeIndex = AddNode(NodeKind::Decltype, 0, 0, static_cast<uint16_t>(decltypeCode));
                PushParseChildOrFail(workFrame, ExpressionComplete, ParseProduction::Expression);
                return;
            }
            CompleteParseWorkFrame(
                AddChild(workFrame.nodeIndex, workFrame.childResult) && ConsumeInput('E')
                    ? workFrame.nodeIndex
                    : kInvalidNodeIndex);
        }

        void StepParseLiteral(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                TypeComplete,
                ExternalNameComplete,
            };
            if (workFrame.stage == Start)
            {
                if (!ConsumeInput('L'))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (PeekInput() == '_' && PeekInput(1) == 'Z')
                {
                    m_inputIndex += 2;
                    PushParseChildOrFail(workFrame, ExternalNameComplete, ParseProduction::Encoding);
                    return;
                }
                workFrame.nodeIndex = AddNode(NodeKind::Literal);
                PushParseChildOrFail(workFrame, TypeComplete, ParseProduction::Type);
                return;
            }
            if (workFrame.stage == ExternalNameComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult != kInvalidNodeIndex && ConsumeInput('E') ? workFrame.childResult : kInvalidNodeIndex);
                return;
            }
            if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (ConsumeInput('n'))
            {
                m_pNodes[workFrame.nodeIndex].flags |= NodeFlagNegativeLiteral;
            }
            const size_t valueOffset = m_inputIndex;
            while (HasInput() && PeekInput() != 'E')
            {
                ++m_inputIndex;
            }
            const size_t valueLength = m_inputIndex - valueOffset;
            if (!ConsumeInput('E') || valueOffset > UINT32_MAX || valueLength > UINT32_MAX)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            m_pNodes[workFrame.nodeIndex].inputOffset = static_cast<uint32_t>(valueOffset);
            m_pNodes[workFrame.nodeIndex].inputLength = static_cast<uint32_t>(valueLength);
            CompleteParseWorkFrame(workFrame.nodeIndex);
        }

        void StepParseExpression(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                DirectChildComplete,
                ConversionTypeComplete,
                OperatorComplete,
                ConversionArgumentComplete,
                ParseConversionArguments,
                ConversionPackArgumentComplete,
                FallbackTypeComplete,
                UnaryArgumentComplete,
                ParseOperatorArguments,
                OperatorArgumentComplete,
            };
            if (workFrame.stage == DirectChildComplete || workFrame.stage == FallbackTypeComplete)
            {
                CompleteParseWorkFrame(workFrame.childResult);
                return;
            }
            if (workFrame.stage == ConversionTypeComplete)
            {
                const NodeIndex conversionOperatorNodeIndex = AddUnaryNode(NodeKind::ConversionOperator, workFrame.childResult);
                workFrame.nodeIndex = AddNode(NodeKind::Function);
                if (!AddChild(workFrame.nodeIndex, conversionOperatorNodeIndex))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (ConsumeInput('_'))
                {
                    workFrame.flags = ParseFlagNone;
                    workFrame.stage = ParseConversionArguments;
                    return;
                }
                PushParseChildOrFail(workFrame, ConversionArgumentComplete, ParseProduction::Expression);
                return;
            }
            if (workFrame.stage == ConversionArgumentComplete)
            {
                CompleteParseWorkFrame(AddChild(workFrame.nodeIndex, workFrame.childResult) ? workFrame.nodeIndex : kInvalidNodeIndex);
                return;
            }
            if (workFrame.stage == ConversionPackArgumentComplete)
            {
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.flags |= ParseFlagPrimary;
                workFrame.stage = ParseConversionArguments;
                return;
            }
            if (workFrame.stage == ParseConversionArguments)
            {
                if (!HasInput())
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                if (PeekInput() == 'E')
                {
                    CompleteParseWorkFrame((workFrame.flags & ParseFlagPrimary) != 0 && ConsumeInput('E') ? workFrame.nodeIndex : kInvalidNodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, ConversionPackArgumentComplete, ParseProduction::Expression);
                return;
            }
            if (workFrame.stage == OperatorComplete)
            {
                if (workFrame.childResult == kInvalidNodeIndex)
                {
                    PushParseChildOrFail(workFrame, FallbackTypeComplete, ParseProduction::Type);
                    return;
                }
                const uint16_t operatorValue = m_pNodes[workFrame.childResult].value;
                const uint16_t arity = operatorValue >> 8;
                if (arity == 1)
                {
                    workFrame.nodeIndex = AddNode(NodeKind::PrefixExpression, 0, 0, operatorValue);
                    PushParseChildOrFail(workFrame, UnaryArgumentComplete, ParseProduction::Expression);
                    return;
                }
                workFrame.nodeIndex = AddNode(NodeKind::Function);
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.childOrdinal = 0;
                workFrame.savedValue = arity;
                workFrame.stage = ParseOperatorArguments;
                return;
            }
            if (workFrame.stage == UnaryArgumentComplete)
            {
                CompleteParseWorkFrame(AddChild(workFrame.nodeIndex, workFrame.childResult) ? workFrame.nodeIndex : kInvalidNodeIndex);
                return;
            }
            if (workFrame.stage == OperatorArgumentComplete)
            {
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                ++workFrame.childOrdinal;
                workFrame.stage = ParseOperatorArguments;
                return;
            }
            if (workFrame.stage == ParseOperatorArguments)
            {
                if (workFrame.childOrdinal >= workFrame.savedValue)
                {
                    CompleteParseWorkFrame(workFrame.nodeIndex);
                    return;
                }
                PushParseChildOrFail(workFrame, OperatorArgumentComplete, ParseProduction::Expression);
                return;
            }

            if (PeekInput() == 'L')
            {
                PushParseChildOrFail(workFrame, DirectChildComplete, ParseProduction::Literal);
                return;
            }
            if (PeekInput() == 'T')
            {
                CompleteParseWorkFrame(ParseTemplateParameter());
                return;
            }
            if (PeekInput() == 'S')
            {
                CompleteParseWorkFrame(ParseSubstitution());
                return;
            }
            if (PeekInput() == 'f'
                && (PeekInput(1) == 'p'
                    || (PeekInput(1) == 'L' && '0' <= PeekInput(2) && PeekInput(2) <= '9')))
            {
                CompleteParseWorkFrame(ParseFunctionParameter());
                return;
            }
            if (PeekInput() == 'c' && PeekInput(1) == 'v')
            {
                m_inputIndex += 2;
                PushParseChildOrFail(workFrame, ConversionTypeComplete, ParseProduction::Type);
                return;
            }
            const char firstCode = PeekInput();
            const char secondCode = PeekInput(1);
            if ((firstCode == 'f' && secondCode == 'L')
                || ((firstCode == 'd' || firstCode == 'p') && secondCode == 't')
                || (firstCode == 'c' && secondCode == 'l')
                || (firstCode == 'i' && secondCode == 'x')
                || (firstCode == 's' && (secondCode == 'r' || secondCode == 'Z')))
            {
                MarkUnsupportedInput();
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            PushParseChildOrFail(workFrame, OperatorComplete, ParseProduction::OperatorName);
        }

        void StepParseOperatorName(ParseWorkFrame& workFrame)
        {
            if (workFrame.stage != 0)
            {
                CompleteParseWorkFrame(AddUnaryNode(NodeKind::ConversionOperator, workFrame.childResult));
                return;
            }
            if (!HasInput(2))
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            const char firstCode = PeekInput();
            const char secondCode = PeekInput(1);
            if (firstCode == 'c' && secondCode == 'v')
            {
                m_inputIndex += 2;
                PushParseChildOrFail(workFrame, kOperatorNameTypeCompleteStage, ParseProduction::Type);
                return;
            }
            const uint16_t operatorValue = OperatorValue(firstCode, secondCode);
            if (operatorValue == 0)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            m_inputIndex += 2;
            CompleteParseWorkFrame(AddNode(NodeKind::OperatorName, 0, 0, operatorValue));
        }

        void StepParseLambdaName(ParseWorkFrame& workFrame)
        {
            enum Stage : uint8_t
            {
                Start,
                ParseTypes,
                TypeComplete,
            };
            if (workFrame.stage == Start)
            {
                m_inputIndex += 2;
                workFrame.nodeIndex = AddNode(NodeKind::Lambda);
                workFrame.stage = ParseTypes;
                return;
            }
            if (workFrame.stage == TypeComplete)
            {
                if (!AddChild(workFrame.nodeIndex, workFrame.childResult))
                {
                    CompleteParseWorkFrame(kInvalidNodeIndex);
                    return;
                }
                workFrame.stage = ParseTypes;
                return;
            }
            if (!HasInput())
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            if (PeekInput() != 'E')
            {
                PushParseChildOrFail(workFrame, TypeComplete, ParseProduction::Type);
                return;
            }
            ++m_inputIndex;
            const size_t discriminatorOffset = m_inputIndex;
            while ('0' <= PeekInput() && PeekInput() <= '9')
            {
                ++m_inputIndex;
            }
            const size_t discriminatorLength = m_inputIndex - discriminatorOffset;
            if (discriminatorOffset > UINT32_MAX || discriminatorLength > UINT32_MAX)
            {
                CompleteParseWorkFrame(kInvalidNodeIndex);
                return;
            }
            m_pNodes[workFrame.nodeIndex].inputOffset = static_cast<uint32_t>(discriminatorOffset);
            m_pNodes[workFrame.nodeIndex].inputLength = static_cast<uint32_t>(discriminatorLength);
            CompleteParseWorkFrame(ConsumeInput('_') ? workFrame.nodeIndex : kInvalidNodeIndex);
        }

        bool NameEndsInTemplateId(NodeIndex nodeIndex) const
        {
            while (nodeIndex != kInvalidNodeIndex)
            {
                const Node& node = m_pNodes[nodeIndex];
                if (node.kind == NodeKind::TemplateId)
                {
                    return true;
                }
                if (node.kind == NodeKind::QualifiedName)
                {
                    nodeIndex = node.lastChildIndex;
                }
                else if (node.kind == NodeKind::Alias)
                {
                    nodeIndex = node.firstChildIndex;
                }
                else
                {
                    return false;
                }
            }
            return false;
        }

        static constexpr uint16_t kBuiltinHalf = 24;
        static constexpr uint16_t kStandardNameBase = 64;
        static constexpr uint16_t kFunctionParameterThis = 1;
        static constexpr uint16_t kFunctionTypeNoexcept = 1;

        static uint16_t BuiltinTypeValue(char typeCode)
        {
            switch (typeCode)
            {
                case 'v': return 1;
                case 'w': return 2;
                case 'b': return 3;
                case 'c': return 4;
                case 'a': return 5;
                case 'h': return 6;
                case 's': return 7;
                case 't': return 8;
                case 'i': return 9;
                case 'j': return 10;
                case 'l': return 11;
                case 'm': return 12;
                case 'x': return 13;
                case 'y': return 14;
                case 'n': return 15;
                case 'o': return 16;
                case 'f': return 17;
                case 'd': return 18;
                case 'e': return 19;
                case 'g': return 20;
                case 'z': return 21;
                default: return 0;
            }
        }

        static uint16_t StandardNameValue(const char* pStandardName)
        {
            if (std::strcmp(pStandardName, "std") == 0)
            {
                return kStandardNameBase;
            }
            if (std::strcmp(pStandardName, "std::allocator") == 0)
            {
                return kStandardNameBase + 1;
            }
            if (std::strcmp(pStandardName, "std::basic_string") == 0)
            {
                return kStandardNameBase + 2;
            }
            if (std::strcmp(pStandardName, "std::string") == 0)
            {
                return kStandardNameBase + 3;
            }
            if (std::strcmp(pStandardName, "std::istream") == 0)
            {
                return kStandardNameBase + 4;
            }
            if (std::strcmp(pStandardName, "std::ostream") == 0)
            {
                return kStandardNameBase + 5;
            }
            return kStandardNameBase + 6;
        }

        static bool IsValidUtf8(const char* pBytes, size_t byteCount)
        {
            size_t byteIndex = 0;
            while (byteIndex < byteCount)
            {
                const uint8_t leadingByte = static_cast<uint8_t>(pBytes[byteIndex]);
                if (leadingByte < 0x80)
                {
                    if (leadingByte == 0)
                    {
                        return false;
                    }
                    ++byteIndex;
                    continue;
                }

                size_t sequenceLength = 0;
                uint32_t codePoint = 0;
                uint32_t minimumCodePoint = 0;
                if (0xc2 <= leadingByte && leadingByte <= 0xdf)
                {
                    sequenceLength = 2;
                    codePoint = leadingByte & 0x1f;
                    minimumCodePoint = 0x80;
                }
                else if (0xe0 <= leadingByte && leadingByte <= 0xef)
                {
                    sequenceLength = 3;
                    codePoint = leadingByte & 0x0f;
                    minimumCodePoint = 0x800;
                }
                else if (0xf0 <= leadingByte && leadingByte <= 0xf4)
                {
                    sequenceLength = 4;
                    codePoint = leadingByte & 0x07;
                    minimumCodePoint = 0x10000;
                }
                else
                {
                    return false;
                }
                if (sequenceLength > byteCount - byteIndex)
                {
                    return false;
                }
                for (size_t continuationIndex = 1; continuationIndex < sequenceLength; ++continuationIndex)
                {
                    const uint8_t continuationByte = static_cast<uint8_t>(pBytes[byteIndex + continuationIndex]);
                    if ((continuationByte & 0xc0) != 0x80)
                    {
                        return false;
                    }
                    codePoint = (codePoint << 6) | (continuationByte & 0x3f);
                }
                if (codePoint < minimumCodePoint || codePoint > 0x10ffff
                    || (0xd800 <= codePoint && codePoint <= 0xdfff))
                {
                    return false;
                }
                byteIndex += sequenceLength;
            }
            return true;
        }

        static uint16_t OperatorValue(char firstCode, char secondCode)
        {
            struct OperatorEntry
            {
                char firstCode;
                char secondCode;
                uint8_t arity;
                uint8_t textIndex;
            };
            static constexpr OperatorEntry operators[] = {
                {'n', 'w', 1,  1},
                {'n', 'a', 1,  2},
                {'d', 'l', 1,  3},
                {'d', 'a', 1,  4},
                {'p', 's', 1,  5},
                {'n', 'g', 1,  6},
                {'a', 'd', 1,  7},
                {'d', 'e', 1,  8},
                {'c', 'o', 1,  9},
                {'p', 'l', 2, 10},
                {'m', 'i', 2, 11},
                {'m', 'l', 2, 12},
                {'d', 'v', 2, 13},
                {'r', 'm', 2, 14},
                {'a', 'n', 2, 15},
                {'o', 'r', 2, 16},
                {'e', 'o', 2, 17},
                {'a', 'S', 2, 18},
                {'p', 'L', 2, 19},
                {'m', 'I', 2, 20},
                {'m', 'L', 2, 21},
                {'d', 'V', 2, 22},
                {'r', 'M', 2, 23},
                {'a', 'N', 2, 24},
                {'o', 'R', 2, 25},
                {'e', 'O', 2, 26},
                {'l', 's', 2, 27},
                {'r', 's', 2, 28},
                {'l', 'S', 2, 49},
                {'r', 'S', 2, 50},
                {'e', 'q', 2, 29},
                {'n', 'e', 2, 30},
                {'l', 't', 2, 31},
                {'g', 't', 2, 32},
                {'l', 'e', 2, 33},
                {'g', 'e', 2, 34},
                {'n', 't', 1, 35},
                {'a', 'a', 2, 36},
                {'o', 'o', 2, 37},
                {'p', 'p', 1, 38},
                {'m', 'm', 1, 39},
                {'c', 'm', 2, 40},
                {'p', 'm', 2, 41},
                {'p', 't', 2, 42},
                {'c', 'l', 1, 43},
                {'i', 'x', 2, 44},
                {'q', 'u', 3, 45},
                {'s', 't', 1, 46},
                {'s', 'z', 1, 47},
                {'a', 't', 1, 48},
                {'a', 'z', 1, 48},
            };
            for (size_t operatorIndex = 0; operatorIndex < sizeof(operators) / sizeof(operators[0]); ++operatorIndex)
            {
                if (operators[operatorIndex].firstCode == firstCode && operators[operatorIndex].secondCode == secondCode)
                {
                    return static_cast<uint16_t>((operators[operatorIndex].arity << 8) | operators[operatorIndex].textIndex);
                }
            }
            return 0;
        }

        static const char* BuiltinText(uint16_t builtinValue, size_t& textLength)
        {
            static constexpr const char* texts[] = {
                "",
                "void",
                "wchar_t",
                "bool",
                "char",
                "signed char",
                "unsigned char",
                "short",
                "unsigned short",
                "int",
                "unsigned int",
                "long",
                "unsigned long",
                "long long",
                "unsigned long long",
                "__int128",
                "unsigned __int128",
                "float",
                "double",
                "long double",
                "__float128",
                "...",
                "vendor type",
                "decimal",
                "__half",
            };
            static constexpr const char* standardTexts[] = {
                "std",
                "std::allocator",
                "std::basic_string",
                "std::string",
                "std::istream",
                "std::ostream",
                "std::iostream",
            };
            const char* pText = "?";
            if (builtinValue < sizeof(texts) / sizeof(texts[0]))
            {
                pText = texts[builtinValue];
            }
            if (builtinValue >= kStandardNameBase)
            {
                const size_t standardNameIndex = static_cast<size_t>(builtinValue - kStandardNameBase);
                if (standardNameIndex < sizeof(standardTexts) / sizeof(standardTexts[0]))
                {
                    pText = standardTexts[standardNameIndex];
                }
            }
            textLength = std::strlen(pText);
            return pText;
        }

        static const char* OperatorText(uint16_t operatorValue, size_t& textLength)
        {
            static constexpr const char* texts[] = {
                "",
                "operator new",
                "operator new[]",
                "operator delete",
                "operator delete[]",
                "operator+",
                "operator-",
                "operator&",
                "operator*",
                "operator~",
                "operator+",
                "operator-",
                "operator*",
                "operator/",
                "operator%",
                "operator&",
                "operator|",
                "operator^",
                "operator=",
                "operator+=",
                "operator-=",
                "operator*=",
                "operator/=",
                "operator%=",
                "operator&=",
                "operator|=",
                "operator^=",
                "operator<<",
                "operator>>",
                "operator==",
                "operator!=",
                "operator<",
                "operator>",
                "operator<=",
                "operator>=",
                "operator!",
                "operator&&",
                "operator||",
                "operator++",
                "operator--",
                "operator,",
                "operator->*",
                "operator->",
                "operator()",
                "operator[]",
                "operator?",
                "sizeof",
                "sizeof",
                "alignof",
                "operator<<=",
                "operator>>=",
            };
            const size_t textIndex = operatorValue & 0xff;
            const char* pText = textIndex < sizeof(texts) / sizeof(texts[0]) ? texts[textIndex] : "operator?";
            textLength = std::strlen(pText);
            return pText;
        }

    private:
        // Rendering walks the immutable plan iteratively. Recursive tree traversal
        // is represented by a scratch-backed data stack of RenderFrame work frames,
        // never by recursive C++ calls. Each frame's role and symbolic stage permit
        // declarators to emit left and right halves. Fragments reference static text
        // or input slices and are appended once; an output limit preserves the exact
        // frame and UTF-8 offset needed for the next call.

        void InitializeRender(RenderMode renderMode, uint32_t maxTreeDepth)
        {
            m_renderMode = renderMode;
            m_treeMaxDepth = maxTreeDepth;
            m_renderFrameCount = 0;
            m_renderedOutputLength = 0;
            m_fragmentOffset = 0;
            m_pCurrentFragment = nullptr;
            m_currentFragmentLength = 0;
            m_renderComplete = false;
            if (m_rootNodeIndex != kInvalidNodeIndex && m_renderFrameCapacity != 0)
            {
                PushRenderFrame(m_rootNodeIndex, 0);
            }
        }

        bool PushRenderFrame(NodeIndex nodeIndex, uint32_t depth, RenderRole role = RenderRole::Normal)
        {
            if (m_renderFrameCount >= m_renderFrameCapacity)
            {
                return false;
            }
            RenderFrame& renderFrame = m_pRenderFrames[m_renderFrameCount++];
            renderFrame.nodeIndex = nodeIndex;
            renderFrame.childIndex = nodeIndex != kInvalidNodeIndex ? m_pNodes[nodeIndex].firstChildIndex : kInvalidNodeIndex;
            renderFrame.depth = depth;
            renderFrame.stage = kInitialRenderStage;
            renderFrame.childOrdinal = 0;
            renderFrame.role = static_cast<uint8_t>(role);
            return true;
        }

        template <typename TAppendOutput>
        Status RenderWithLimit(size_t maxOutputLength, TAppendOutput&& appendOutput)
        {
            if (maxOutputLength < m_renderedOutputLength)
            {
                return Status::OutputLimitReached;
            }
            while (!m_renderComplete && m_renderedOutputLength < maxOutputLength)
            {
                const char* pFragment = nullptr;
                size_t fragmentLength = 0;
                if (!GetNextFragment(pFragment, fragmentLength))
                {
                    m_renderComplete = true;
                    break;
                }
                const size_t unconsumedLength = fragmentLength - m_fragmentOffset;
                const size_t availableLength = maxOutputLength - m_renderedOutputLength;
                size_t appendLength = unconsumedLength < availableLength ? unconsumedLength : availableLength;
                if (appendLength < unconsumedLength)
                {
                    appendLength = Utf8SafePrefixLength(pFragment + m_fragmentOffset, appendLength);
                }
                if (appendLength == 0)
                {
                    break;
                }
                appendOutput(pFragment + m_fragmentOffset, appendLength);
                m_fragmentOffset += appendLength;
                m_renderedOutputLength += appendLength;
            }
            if (!m_renderComplete && m_renderedOutputLength == maxOutputLength)
            {
                const char* pNextFragment = nullptr;
                size_t nextFragmentLength = 0;
                if (!GetNextFragment(pNextFragment, nextFragmentLength))
                {
                    m_renderComplete = true;
                }
            }
            return m_renderComplete ? Status::Complete : Status::OutputLimitReached;
        }

        static size_t Utf8SafePrefixLength(const char* pBytes, size_t proposedLength)
        {
            if (proposedLength == 0)
            {
                return 0;
            }
            size_t sequenceStart = proposedLength;
            while (sequenceStart > 0 && (static_cast<uint8_t>(pBytes[sequenceStart - 1]) & 0xc0) == 0x80)
            {
                --sequenceStart;
            }
            if (sequenceStart == proposedLength)
            {
                const uint8_t lastByte = static_cast<uint8_t>(pBytes[proposedLength - 1]);
                if (lastByte < 0x80)
                {
                    return proposedLength;
                }
                sequenceStart = proposedLength - 1;
            }
            else if (sequenceStart != 0)
            {
                --sequenceStart;
            }
            const uint8_t leadingByte = static_cast<uint8_t>(pBytes[sequenceStart]);
            const size_t requiredLength = leadingByte < 0xc2 ? 1 : leadingByte < 0xe0 ? 2
                                                             : leadingByte < 0xf0     ? 3
                                                                                      : 4;
            return proposedLength - sequenceStart < requiredLength ? sequenceStart : proposedLength;
        }

        bool GetNextFragment(const char*& pFragment, size_t& fragmentLength)
        {
            if (m_pCurrentFragment && m_fragmentOffset < m_currentFragmentLength)
            {
                pFragment = m_pCurrentFragment;
                fragmentLength = m_currentFragmentLength;
                return true;
            }
            m_fragmentOffset = 0;
            const bool foundFragment = m_renderMode == RenderMode::Tree
                                         ? GetNextTreeFragment(pFragment, fragmentLength)
                                         : GetNextNameFragment(pFragment, fragmentLength);
            m_pCurrentFragment = foundFragment ? pFragment : nullptr;
            m_currentFragmentLength = foundFragment ? fragmentLength : 0;
            return foundFragment;
        }

        bool ReturnStaticFragment(const char* pText, const char*& pFragment, size_t& fragmentLength)
        {
            pFragment = pText;
            fragmentLength = std::strlen(pText);
            return fragmentLength != 0;
        }

        bool ReturnUnsignedFragment(uint32_t value, const char*& pFragment, size_t& fragmentLength)
        {
            const int characterCount = std::snprintf(m_numericFragment, sizeof(m_numericFragment), "%u", value);
            if (characterCount <= 0 || static_cast<size_t>(characterCount) >= sizeof(m_numericFragment))
            {
                return false;
            }
            pFragment = m_numericFragment;
            fragmentLength = static_cast<size_t>(characterCount);
            return true;
        }

        static bool IsDeclaratorTypeKind(NodeKind kind)
        {
            return kind == NodeKind::PointerType || kind == NodeKind::LValueReferenceType
                || kind == NodeKind::RValueReferenceType || kind == NodeKind::ConstType
                || kind == NodeKind::VolatileType || kind == NodeKind::RestrictType
                || kind == NodeKind::ComplexType || kind == NodeKind::ImaginaryType
                || kind == NodeKind::VendorQualifierType || kind == NodeKind::ArrayType
                || kind == NodeKind::FunctionType || kind == NodeKind::MemberPointerType;
        }

        bool IsPostfixDeclarator(NodeIndex nodeIndex) const
        {
            while (nodeIndex != kInvalidNodeIndex)
            {
                const Node& node = m_pNodes[nodeIndex];
                if (node.kind == NodeKind::FunctionType || node.kind == NodeKind::ArrayType)
                {
                    return true;
                }
                if (node.kind != NodeKind::Alias && node.kind != NodeKind::ConstType
                    && node.kind != NodeKind::VolatileType && node.kind != NodeKind::RestrictType
                    && node.kind != NodeKind::ComplexType && node.kind != NodeKind::ImaginaryType)
                {
                    return false;
                }
                nodeIndex = node.firstChildIndex;
            }
            return false;
        }

        bool IsFunctionDeclarator(NodeIndex nodeIndex) const
        {
            while (nodeIndex != kInvalidNodeIndex)
            {
                const Node& node = m_pNodes[nodeIndex];
                if (node.kind == NodeKind::FunctionType)
                {
                    return true;
                }
                if (node.kind != NodeKind::Alias && node.kind != NodeKind::ConstType
                    && node.kind != NodeKind::VolatileType && node.kind != NodeKind::RestrictType)
                {
                    return false;
                }
                nodeIndex = node.firstChildIndex;
            }
            return false;
        }

        NodeIndex GetArrayElementNodeIndex(const Node& node) const
        {
            return (node.flags & NodeFlagArrayDimensionExpression) != 0
                     ? m_pNodes[node.firstChildIndex].nextSiblingIndex
                     : node.firstChildIndex;
        }

        bool GetNextTypeLeftFragment(
            RenderFrame& renderFrame,
            const Node& node,
            const char*& pFragment,
            size_t& fragmentLength)
        {
            const bool isPointerLike = node.kind == NodeKind::PointerType
                                    || node.kind == NodeKind::LValueReferenceType
                                    || node.kind == NodeKind::RValueReferenceType;
            if (isPointerLike)
            {
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                if (renderFrame.stage == TypeLeftFirstStepComplete)
                {
                    renderFrame.stage = TypeLeftSecondStepComplete;
                    const bool needsParentheses = IsPostfixDeclarator(node.firstChildIndex);
                    const char* pText = node.kind == NodeKind::PointerType
                                          ? (needsParentheses ? " (*" : "*")
                                      : node.kind == NodeKind::LValueReferenceType
                                          ? (needsParentheses ? " (&" : "&")
                                          : (needsParentheses ? " (&&" : "&&");
                    return ReturnStaticFragment(pText, pFragment, fragmentLength);
                }
                --m_renderFrameCount;
                return false;
            }

            const bool isCvQualifier = node.kind == NodeKind::ConstType || node.kind == NodeKind::VolatileType
                                    || node.kind == NodeKind::RestrictType;
            if (isCvQualifier)
            {
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                if (renderFrame.stage == TypeLeftFirstStepComplete && !IsFunctionDeclarator(node.firstChildIndex))
                {
                    renderFrame.stage = TypeLeftSecondStepComplete;
                    const char* pText = node.kind == NodeKind::ConstType    ? " const"
                                      : node.kind == NodeKind::VolatileType ? " volatile"
                                                                            : " restrict";
                    return ReturnStaticFragment(pText, pFragment, fragmentLength);
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::ComplexType || node.kind == NodeKind::ImaginaryType)
            {
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    return ReturnStaticFragment(node.kind == NodeKind::ComplexType ? "_Complex " : "_Imaginary ", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeLeftFirstStepComplete)
                {
                    renderFrame.stage = TypeLeftSecondStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::VendorQualifierType)
            {
                const NodeIndex qualifiedTypeNodeIndex = m_pNodes[node.firstChildIndex].nextSiblingIndex;
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    PushRenderFrame(qualifiedTypeNodeIndex, renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                if (renderFrame.stage == TypeLeftFirstStepComplete)
                {
                    renderFrame.stage = TypeLeftSecondStepComplete;
                    return ReturnStaticFragment(" ", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeLeftSecondStepComplete)
                {
                    renderFrame.stage = TypeLeftThirdStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::ArrayType)
            {
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    PushRenderFrame(GetArrayElementNodeIndex(node), renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::FunctionType)
            {
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                if (renderFrame.stage == TypeLeftFirstStepComplete)
                {
                    renderFrame.stage = TypeLeftSecondStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::MemberPointerType)
            {
                const NodeIndex memberTypeNodeIndex = m_pNodes[node.firstChildIndex].nextSiblingIndex;
                const bool needsParentheses = IsPostfixDeclarator(memberTypeNodeIndex);
                if (renderFrame.stage == TypeLeftStart)
                {
                    renderFrame.stage = TypeLeftFirstStepComplete;
                    PushRenderFrame(memberTypeNodeIndex, renderFrame.depth + 1, RenderRole::TypeLeft);
                    return false;
                }
                if (renderFrame.stage == TypeLeftFirstStepComplete)
                {
                    renderFrame.stage = TypeLeftSecondStepComplete;
                    return ReturnStaticFragment(needsParentheses ? " (" : " ", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeLeftSecondStepComplete)
                {
                    renderFrame.stage = TypeLeftThirdStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                    return false;
                }
                if (renderFrame.stage == TypeLeftThirdStepComplete)
                {
                    renderFrame.stage = TypeLeftFourthStepComplete;
                    return ReturnStaticFragment("::*", pFragment, fragmentLength);
                }
                --m_renderFrameCount;
                return false;
            }

            --m_renderFrameCount;
            return false;
        }

        bool GetNextTypeRightFragment(
            RenderFrame& renderFrame,
            const Node& node,
            const char*& pFragment,
            size_t& fragmentLength)
        {
            const bool isPointerLike = node.kind == NodeKind::PointerType
                                    || node.kind == NodeKind::LValueReferenceType
                                    || node.kind == NodeKind::RValueReferenceType;
            if (isPointerLike)
            {
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    if (IsPostfixDeclarator(node.firstChildIndex))
                    {
                        return ReturnStaticFragment(")", pFragment, fragmentLength);
                    }
                }
                if (renderFrame.stage == TypeRightFirstStepComplete)
                {
                    renderFrame.stage = TypeRightSecondStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            const bool isCvQualifier = node.kind == NodeKind::ConstType || node.kind == NodeKind::VolatileType
                                    || node.kind == NodeKind::RestrictType;
            if (isCvQualifier)
            {
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                if (renderFrame.stage == TypeRightFirstStepComplete && IsFunctionDeclarator(node.firstChildIndex))
                {
                    renderFrame.stage = TypeRightSecondStepComplete;
                    const char* pText = node.kind == NodeKind::ConstType    ? " const"
                                      : node.kind == NodeKind::VolatileType ? " volatile"
                                                                            : " restrict";
                    return ReturnStaticFragment(pText, pFragment, fragmentLength);
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::ComplexType || node.kind == NodeKind::ImaginaryType)
            {
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::VendorQualifierType)
            {
                const NodeIndex qualifiedTypeNodeIndex = m_pNodes[node.firstChildIndex].nextSiblingIndex;
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    PushRenderFrame(qualifiedTypeNodeIndex, renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::ArrayType)
            {
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    return ReturnStaticFragment("[", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeRightFirstStepComplete)
                {
                    renderFrame.stage = TypeRightSecondStepComplete;
                    if ((node.flags & NodeFlagArrayDimensionExpression) != 0)
                    {
                        PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                        return false;
                    }
                    pFragment = m_pInput + node.inputOffset;
                    fragmentLength = node.inputLength;
                    if (fragmentLength != 0)
                    {
                        return true;
                    }
                }
                if (renderFrame.stage == TypeRightSecondStepComplete)
                {
                    renderFrame.stage = TypeRightThirdStepComplete;
                    return ReturnStaticFragment("]", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeRightThirdStepComplete)
                {
                    renderFrame.stage = TypeRightFourthStepComplete;
                    PushRenderFrame(GetArrayElementNodeIndex(node), renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::FunctionType)
            {
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    renderFrame.childIndex = m_pNodes[node.firstChildIndex].nextSiblingIndex;
                    renderFrame.childOrdinal = 0;
                    if (renderFrame.childIndex != kInvalidNodeIndex
                        && m_pNodes[renderFrame.childIndex].nextSiblingIndex == kInvalidNodeIndex
                        && m_pNodes[renderFrame.childIndex].kind == NodeKind::BuiltinType
                        && m_pNodes[renderFrame.childIndex].value == BuiltinTypeValue('v'))
                    {
                        renderFrame.childIndex = kInvalidNodeIndex;
                    }
                    return ReturnStaticFragment("(", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeRightSecondStepComplete)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    const NodeIndex childNodeIndex = renderFrame.childIndex;
                    renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                    PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                    return false;
                }
                if (renderFrame.stage == TypeRightFirstStepComplete && renderFrame.childIndex != kInvalidNodeIndex)
                {
                    const NodeIndex childNodeIndex = renderFrame.childIndex;
                    if (renderFrame.childOrdinal++ != 0)
                    {
                        renderFrame.stage = TypeRightSecondStepComplete;
                        return ReturnStaticFragment(", ", pFragment, fragmentLength);
                    }
                    renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                    PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                    return false;
                }
                if (renderFrame.stage <= TypeRightSecondStepComplete)
                {
                    renderFrame.stage = TypeRightThirdStepComplete;
                    return ReturnStaticFragment(")", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TypeRightThirdStepComplete)
                {
                    renderFrame.stage = TypeRightFourthStepComplete;
                    if ((node.value & kFunctionTypeNoexcept) != 0)
                    {
                        return ReturnStaticFragment(" noexcept", pFragment, fragmentLength);
                    }
                }
                if (renderFrame.stage == TypeRightFourthStepComplete)
                {
                    renderFrame.stage = TypeRightFifthStepComplete;
                    if ((node.flags & NodeFlagFunctionExternC) != 0)
                    {
                        return ReturnStaticFragment(" extern \"C\"", pFragment, fragmentLength);
                    }
                }
                --m_renderFrameCount;
                return false;
            }

            if (node.kind == NodeKind::MemberPointerType)
            {
                const NodeIndex memberTypeNodeIndex = m_pNodes[node.firstChildIndex].nextSiblingIndex;
                if (renderFrame.stage == TypeRightStart)
                {
                    renderFrame.stage = TypeRightFirstStepComplete;
                    if (IsPostfixDeclarator(memberTypeNodeIndex))
                    {
                        return ReturnStaticFragment(")", pFragment, fragmentLength);
                    }
                }
                if (renderFrame.stage == TypeRightFirstStepComplete)
                {
                    renderFrame.stage = TypeRightSecondStepComplete;
                    PushRenderFrame(memberTypeNodeIndex, renderFrame.depth + 1, RenderRole::TypeRight);
                    return false;
                }
                --m_renderFrameCount;
                return false;
            }

            --m_renderFrameCount;
            return false;
        }

        bool GetNextNameFragment(const char*& pFragment, size_t& fragmentLength)
        {
            while (m_renderFrameCount != 0)
            {
                RenderFrame& renderFrame = m_pRenderFrames[m_renderFrameCount - 1];
                const Node& node = m_pNodes[renderFrame.nodeIndex];
                const RenderRole renderRole = static_cast<RenderRole>(renderFrame.role);
                if (renderRole == RenderRole::TypeLeft)
                {
                    if (!IsDeclaratorTypeKind(node.kind))
                    {
                        renderFrame.role = static_cast<uint8_t>(RenderRole::Normal);
                        continue;
                    }
                    if (GetNextTypeLeftFragment(renderFrame, node, pFragment, fragmentLength))
                    {
                        return true;
                    }
                    continue;
                }
                if (renderRole == RenderRole::TypeRight)
                {
                    if (!IsDeclaratorTypeKind(node.kind))
                    {
                        --m_renderFrameCount;
                        continue;
                    }
                    if (GetNextTypeRightFragment(renderFrame, node, pFragment, fragmentLength))
                    {
                        return true;
                    }
                    continue;
                }
                if (IsDeclaratorTypeKind(node.kind))
                {
                    renderFrame.role = static_cast<uint8_t>(RenderRole::TypeRight);
                    renderFrame.stage = kInitialRenderStage;
                    PushRenderFrame(renderFrame.nodeIndex, renderFrame.depth, RenderRole::TypeLeft);
                    continue;
                }
                if (node.kind == NodeKind::Identifier)
                {
                    --m_renderFrameCount;
                    static constexpr char anonymousNamespacePrefix[] = "_GLOBAL__N_";
                    if (node.inputLength >= sizeof(anonymousNamespacePrefix) - 1
                        && std::memcmp(m_pInput + node.inputOffset, anonymousNamespacePrefix, sizeof(anonymousNamespacePrefix) - 1)
                               == 0)
                    {
                        return ReturnStaticFragment("(anonymous namespace)", pFragment, fragmentLength);
                    }
                    pFragment = m_pInput + node.inputOffset;
                    fragmentLength = node.inputLength;
                    return true;
                }
                if (node.kind == NodeKind::BuiltinType)
                {
                    --m_renderFrameCount;
                    pFragment = BuiltinText(node.value, fragmentLength);
                    return true;
                }
                if (node.kind == NodeKind::OperatorName)
                {
                    --m_renderFrameCount;
                    pFragment = OperatorText(node.value, fragmentLength);
                    return true;
                }
                if (node.kind == NodeKind::ConversionOperator)
                {
                    if (renderFrame.stage == NameRenderStart)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        return ReturnStaticFragment("operator ", pFragment, fragmentLength);
                    }
                    if (renderFrame.stage == NameRenderPrefixComplete)
                    {
                        renderFrame.stage = NameRenderValueComplete;
                        PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                        continue;
                    }
                    --m_renderFrameCount;
                    continue;
                }
                if (node.kind == NodeKind::Alias)
                {
                    if (renderFrame.stage == NameRenderStart)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                    }
                    else
                    {
                        --m_renderFrameCount;
                    }
                    continue;
                }
                if (node.kind == NodeKind::Constructor || node.kind == NodeKind::Destructor)
                {
                    if (renderFrame.stage == NameRenderStart)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        if (node.kind == NodeKind::Destructor)
                        {
                            return ReturnStaticFragment("~", pFragment, fragmentLength);
                        }
                    }
                    if (renderFrame.stage == NameRenderPrefixComplete)
                    {
                        renderFrame.stage = NameRenderValueComplete;
                        PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                    }
                    else
                    {
                        --m_renderFrameCount;
                    }
                    continue;
                }
                if (node.kind == NodeKind::UnnamedType || node.kind == NodeKind::FunctionParameter)
                {
                    if (node.kind == NodeKind::FunctionParameter && node.value == kFunctionParameterThis)
                    {
                        --m_renderFrameCount;
                        return ReturnStaticFragment("this", pFragment, fragmentLength);
                    }
                    if (renderFrame.stage == NameRenderStart)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        return ReturnStaticFragment(
                            node.kind == NodeKind::UnnamedType ? "'unnamed type#" : "{parm#",
                            pFragment,
                            fragmentLength);
                    }
                    if (renderFrame.stage == NameRenderPrefixComplete)
                    {
                        renderFrame.stage = NameRenderValueComplete;
                        return ReturnUnsignedFragment(node.inputLength, pFragment, fragmentLength);
                    }
                    if (renderFrame.stage == NameRenderValueComplete)
                    {
                        renderFrame.stage = NameRenderSuffixComplete;
                        return ReturnStaticFragment(
                            node.kind == NodeKind::UnnamedType ? "'" : "}",
                            pFragment,
                            fragmentLength);
                    }
                    --m_renderFrameCount;
                    continue;
                }
                if (node.kind == NodeKind::Decltype)
                {
                    if (renderFrame.stage == NameRenderStart)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        return ReturnStaticFragment("decltype(", pFragment, fragmentLength);
                    }
                    if (renderFrame.stage == NameRenderPrefixComplete)
                    {
                        renderFrame.stage = NameRenderValueComplete;
                        PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                        continue;
                    }
                    if (renderFrame.stage == NameRenderValueComplete)
                    {
                        renderFrame.stage = NameRenderSuffixComplete;
                        return ReturnStaticFragment(")", pFragment, fragmentLength);
                    }
                    --m_renderFrameCount;
                    continue;
                }
                if (node.kind == NodeKind::Literal)
                {
                    if (renderFrame.stage == NameRenderStart && (node.flags & NodeFlagNegativeLiteral) != 0)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        return ReturnStaticFragment("-", pFragment, fragmentLength);
                    }
                    if (node.inputLength == 1 && m_pInput[node.inputOffset] == '0')
                    {
                        const Node& literalType = m_pNodes[node.firstChildIndex];
                        if (literalType.kind == NodeKind::BuiltinType && literalType.value == BuiltinTypeValue('b'))
                        {
                            --m_renderFrameCount;
                            return ReturnStaticFragment("false", pFragment, fragmentLength);
                        }
                    }
                    if (node.inputLength == 1 && m_pInput[node.inputOffset] == '1')
                    {
                        const Node& literalType = m_pNodes[node.firstChildIndex];
                        if (literalType.kind == NodeKind::BuiltinType && literalType.value == BuiltinTypeValue('b'))
                        {
                            --m_renderFrameCount;
                            return ReturnStaticFragment("true", pFragment, fragmentLength);
                        }
                    }
                    if (renderFrame.stage <= NameRenderPrefixComplete)
                    {
                        renderFrame.stage = NameRenderValueComplete;
                        pFragment = m_pInput + node.inputOffset;
                        fragmentLength = node.inputLength;
                        return true;
                    }
                    --m_renderFrameCount;
                    const Node& literalType = m_pNodes[node.firstChildIndex];
                    if (literalType.kind == NodeKind::BuiltinType)
                    {
                        const char* pSuffix = literalType.value == BuiltinTypeValue('j') ? "u"
                                            : literalType.value == BuiltinTypeValue('l') ? "l"
                                            : literalType.value == BuiltinTypeValue('m') ? "ul"
                                            : literalType.value == BuiltinTypeValue('x') ? "ll"
                                            : literalType.value == BuiltinTypeValue('y') ? "ull"
                                                                                         : "";
                        if (pSuffix[0] != '\0')
                        {
                            return ReturnStaticFragment(pSuffix, pFragment, fragmentLength);
                        }
                    }
                    continue;
                }
                if (node.kind == NodeKind::PrefixExpression)
                {
                    if (renderFrame.stage == NameRenderStart)
                    {
                        renderFrame.stage = NameRenderPrefixComplete;
                        const size_t operatorTextIndex = node.value & 0xff;
                        const char* pOperatorText = operatorTextIndex == 35 ? "!" : operatorTextIndex == 5                         ? "+"
                                                                              : operatorTextIndex == 6                             ? "-"
                                                                              : operatorTextIndex == 7                             ? "&"
                                                                              : operatorTextIndex == 8                             ? "*"
                                                                              : operatorTextIndex == 9                             ? "~"
                                                                              : operatorTextIndex == 38                            ? "++"
                                                                              : operatorTextIndex == 39                            ? "--"
                                                                              : operatorTextIndex == 46 || operatorTextIndex == 47 ? "sizeof("
                                                                              : operatorTextIndex == 48                            ? "alignof("
                                                                                                                                   : "";
                        return ReturnStaticFragment(pOperatorText, pFragment, fragmentLength);
                    }
                    if (renderFrame.stage == NameRenderPrefixComplete)
                    {
                        renderFrame.stage = NameRenderValueComplete;
                        PushRenderFrame(node.firstChildIndex, renderFrame.depth + 1);
                    }
                    else if (renderFrame.stage == NameRenderValueComplete && ((node.value & 0xff) == 46 || (node.value & 0xff) == 47 || (node.value & 0xff) == 48))
                    {
                        renderFrame.stage = NameRenderSuffixComplete;
                        return ReturnStaticFragment(")", pFragment, fragmentLength);
                    }
                    else
                    {
                        --m_renderFrameCount;
                    }
                    continue;
                }

                if (GetNextCompositeNameFragment(renderFrame, node, pFragment, fragmentLength))
                {
                    return true;
                }
            }
            return false;
        }

        bool GetNextCompositeNameFragment(
            RenderFrame& renderFrame,
            const Node& node,
            const char*& pFragment,
            size_t& fragmentLength)
        {
            if (node.kind == NodeKind::Lambda)
            {
                if (renderFrame.stage == CompositeRenderStart)
                {
                    renderFrame.stage = CompositeRenderOpenComplete;
                    return ReturnStaticFragment("'lambda", pFragment, fragmentLength);
                }
                if (renderFrame.stage == CompositeRenderOpenComplete)
                {
                    renderFrame.stage = CompositeRenderChildren;
                    if (node.inputLength != 0)
                    {
                        pFragment = m_pInput + node.inputOffset;
                        fragmentLength = node.inputLength;
                        return true;
                    }
                }
                if (renderFrame.stage == CompositeRenderChildren)
                {
                    renderFrame.stage = CompositeRenderSeparator;
                    return ReturnStaticFragment("'(", pFragment, fragmentLength);
                }
                if (renderFrame.stage == CompositeRenderSeparator)
                {
                    renderFrame.stage = LambdaRenderChildren;
                    renderFrame.childIndex = node.firstChildIndex;
                    renderFrame.childOrdinal = 0;
                    if (renderFrame.childIndex != kInvalidNodeIndex
                        && m_pNodes[renderFrame.childIndex].nextSiblingIndex == kInvalidNodeIndex
                        && m_pNodes[renderFrame.childIndex].kind == NodeKind::BuiltinType
                        && m_pNodes[renderFrame.childIndex].value == BuiltinTypeValue('v'))
                    {
                        renderFrame.childIndex = kInvalidNodeIndex;
                    }
                }
                if (renderFrame.stage == LambdaRenderSeparator)
                {
                    renderFrame.stage = LambdaRenderChildren;
                    const NodeIndex childNodeIndex = renderFrame.childIndex;
                    renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                    PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                    return false;
                }
                if (renderFrame.childIndex != kInvalidNodeIndex)
                {
                    const NodeIndex childNodeIndex = renderFrame.childIndex;
                    if (renderFrame.childOrdinal++ != 0)
                    {
                        renderFrame.stage = LambdaRenderSeparator;
                        return ReturnStaticFragment(", ", pFragment, fragmentLength);
                    }
                    renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                    PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                    return false;
                }
                --m_renderFrameCount;
                return ReturnStaticFragment(")", pFragment, fragmentLength);
            }

            if (node.kind == NodeKind::Function && renderFrame.stage >= FunctionRenderConstQualifier)
            {
                while (renderFrame.stage < FunctionRenderQualifiersComplete)
                {
                    const uint8_t qualifierFlag = renderFrame.stage == FunctionRenderConstQualifier    ? NodeFlagFunctionConst
                                                : renderFrame.stage == FunctionRenderVolatileQualifier ? NodeFlagFunctionVolatile
                                                : renderFrame.stage == FunctionRenderRestrictQualifier ? NodeFlagFunctionRestrict
                                                : renderFrame.stage == FunctionRenderLValueQualifier   ? NodeFlagFunctionLValueReference
                                                                                                       : NodeFlagFunctionRValueReference;
                    const char* pQualifierText = renderFrame.stage == FunctionRenderConstQualifier    ? " const"
                                               : renderFrame.stage == FunctionRenderVolatileQualifier ? " volatile"
                                               : renderFrame.stage == FunctionRenderRestrictQualifier ? " restrict"
                                               : renderFrame.stage == FunctionRenderLValueQualifier   ? " &"
                                                                                                      : " &&";
                    renderFrame.stage = static_cast<uint16_t>(renderFrame.stage + 1);
                    if ((node.flags & qualifierFlag) != 0)
                    {
                        return ReturnStaticFragment(pQualifierText, pFragment, fragmentLength);
                    }
                }
                --m_renderFrameCount;
                return false;
            }
            const char* pOpenText = nullptr;
            const char* pSeparatorText = nullptr;
            const char* pCloseText = nullptr;
            if (node.kind == NodeKind::QualifiedName)
            {
                pOpenText = "";
                pSeparatorText = "::";
                pCloseText = "";
            }
            else if (node.kind == NodeKind::TemplateId)
            {
                pOpenText = "<";
                pSeparatorText = ", ";
                pCloseText = ">";
            }
            else if (node.kind == NodeKind::TemplateArgumentPack)
            {
                pOpenText = "";
                pSeparatorText = ", ";
                pCloseText = "";
            }
            else if (node.kind == NodeKind::Function || node.kind == NodeKind::FunctionType)
            {
                pOpenText = "(";
                pSeparatorText = ", ";
                pCloseText = ")";
            }
            else
            {
                --m_renderFrameCount;
                return ReturnStaticFragment("?", pFragment, fragmentLength);
            }

            if (node.kind == NodeKind::Function && (node.flags & NodeFlagFunctionHasReturnType) != 0 && renderFrame.stage == CompositeRenderStart)
            {
                renderFrame.stage = FunctionRenderReturnTypeComplete;
                PushRenderFrame(m_pNodes[node.firstChildIndex].nextSiblingIndex, renderFrame.depth + 1);
                return false;
            }
            if (node.kind == NodeKind::Function && renderFrame.stage == FunctionRenderReturnTypeComplete)
            {
                renderFrame.stage = FunctionRenderReturnSeparatorComplete;
                return ReturnStaticFragment(" ", pFragment, fragmentLength);
            }
            if (node.kind == NodeKind::Function && renderFrame.stage == FunctionRenderReturnSeparatorComplete)
            {
                renderFrame.stage = kInitialRenderStage;
            }

            if (renderFrame.stage == CompositeRenderStart)
            {
                renderFrame.stage = CompositeRenderOpenComplete;
                renderFrame.childIndex = node.firstChildIndex;
                renderFrame.childOrdinal = 0;
                if (node.kind == NodeKind::TemplateId || node.kind == NodeKind::Function || node.kind == NodeKind::FunctionType)
                {
                    PushRenderFrame(renderFrame.childIndex, renderFrame.depth + 1);
                    renderFrame.childIndex = m_pNodes[renderFrame.childIndex].nextSiblingIndex;
                    renderFrame.childOrdinal = 0;
                    if (node.kind == NodeKind::Function && (node.flags & NodeFlagFunctionHasReturnType) != 0
                        && renderFrame.childIndex != kInvalidNodeIndex)
                    {
                        renderFrame.childIndex = m_pNodes[renderFrame.childIndex].nextSiblingIndex;
                    }
                    if (node.kind == NodeKind::Function
                        && renderFrame.childIndex != kInvalidNodeIndex
                        && m_pNodes[renderFrame.childIndex].nextSiblingIndex == kInvalidNodeIndex
                        && m_pNodes[renderFrame.childIndex].kind == NodeKind::BuiltinType
                        && m_pNodes[renderFrame.childIndex].value == BuiltinTypeValue('v'))
                    {
                        renderFrame.childIndex = kInvalidNodeIndex;
                    }
                    return false;
                }
            }
            if (renderFrame.stage == CompositeRenderOpenComplete)
            {
                renderFrame.stage = CompositeRenderChildren;
                return ReturnStaticFragment(pOpenText, pFragment, fragmentLength);
            }
            if (renderFrame.stage == CompositeRenderSeparator)
            {
                renderFrame.stage = CompositeRenderChildren;
                const NodeIndex childNodeIndex = renderFrame.childIndex;
                renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                return false;
            }
            if (renderFrame.childIndex != kInvalidNodeIndex)
            {
                const NodeIndex childNodeIndex = renderFrame.childIndex;
                renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                const uint32_t childOrdinal = renderFrame.childOrdinal++;
                if (childOrdinal != 0)
                {
                    renderFrame.stage = CompositeRenderSeparator;
                    renderFrame.childIndex = childNodeIndex;
                    return ReturnStaticFragment(pSeparatorText, pFragment, fragmentLength);
                }
                PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                return false;
            }
            if (node.kind == NodeKind::Function
                && (node.flags & (NodeFlagFunctionConst | NodeFlagFunctionVolatile | NodeFlagFunctionRestrict | NodeFlagFunctionLValueReference | NodeFlagFunctionRValueReference))
                       != 0)
            {
                renderFrame.stage = FunctionRenderConstQualifier;
                return ReturnStaticFragment(pCloseText, pFragment, fragmentLength);
            }
            --m_renderFrameCount;
            return ReturnStaticFragment(pCloseText, pFragment, fragmentLength);
        }

        static const char* NodeKindText(NodeKind kind)
        {
            switch (kind)
            {
                case NodeKind::Identifier: return "Identifier";
                case NodeKind::BuiltinType: return "BuiltinType";
                case NodeKind::QualifiedName: return "QualifiedName";
                case NodeKind::TemplateId: return "TemplateId";
                case NodeKind::TemplateArgumentPack: return "TemplateArgumentPack";
                case NodeKind::Function: return "Function";
                case NodeKind::PointerType: return "PointerType";
                case NodeKind::LValueReferenceType: return "LValueReferenceType";
                case NodeKind::RValueReferenceType: return "RValueReferenceType";
                case NodeKind::ConstType: return "ConstType";
                case NodeKind::VolatileType: return "VolatileType";
                case NodeKind::RestrictType: return "RestrictType";
                case NodeKind::ComplexType: return "ComplexType";
                case NodeKind::ImaginaryType: return "ImaginaryType";
                case NodeKind::VendorQualifierType: return "VendorQualifierType";
                case NodeKind::ArrayType: return "ArrayType";
                case NodeKind::FunctionType: return "FunctionType";
                case NodeKind::MemberPointerType: return "MemberPointerType";
                case NodeKind::Literal: return "Literal";
                case NodeKind::OperatorName: return "OperatorName";
                case NodeKind::Constructor: return "Constructor";
                case NodeKind::Destructor: return "Destructor";
                case NodeKind::Lambda: return "Lambda";
                case NodeKind::UnnamedType: return "UnnamedType";
                case NodeKind::Decltype: return "Decltype";
                case NodeKind::FunctionParameter: return "FunctionParameter";
                case NodeKind::PrefixExpression: return "PrefixExpression";
                case NodeKind::ConversionOperator: return "ConversionOperator";
                case NodeKind::Alias: return "Reference";
            }
            return "Unknown";
        }

        bool GetNextTreeFragment(const char*& pFragment, size_t& fragmentLength)
        {
            static constexpr const char spaces[] =
                "                                                                ";
            while (m_renderFrameCount != 0)
            {
                RenderFrame& renderFrame = m_pRenderFrames[m_renderFrameCount - 1];
                const Node& node = m_pNodes[renderFrame.nodeIndex];
                if (renderFrame.stage == TreeRenderIndentation)
                {
                    const size_t indentationLength = renderFrame.depth * 2;
                    const size_t emittedIndentationLength = renderFrame.childOrdinal;
                    if (emittedIndentationLength < indentationLength)
                    {
                        const size_t remainingIndentationLength = indentationLength - emittedIndentationLength;
                        pFragment = spaces;
                        fragmentLength = remainingIndentationLength < sizeof(spaces) - 1
                                           ? remainingIndentationLength
                                           : sizeof(spaces) - 1;
                        renderFrame.childOrdinal = static_cast<uint32_t>(emittedIndentationLength + fragmentLength);
                        return true;
                    }
                    renderFrame.stage = TreeRenderNodeKind;
                }
                if (renderFrame.stage == TreeRenderNodeKind)
                {
                    renderFrame.stage = TreeRenderDetailPrefix;
                    pFragment = NodeKindText(node.kind);
                    fragmentLength = std::strlen(pFragment);
                    return true;
                }
                if (renderFrame.stage == TreeRenderDetailPrefix && (node.kind == NodeKind::Identifier || node.kind == NodeKind::Literal))
                {
                    renderFrame.stage = TreeRenderDetailValue;
                    return ReturnStaticFragment(" \"", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TreeRenderDetailValue && (node.kind == NodeKind::Identifier || node.kind == NodeKind::Literal))
                {
                    renderFrame.stage = TreeRenderDetailSuffix;
                    pFragment = m_pInput + node.inputOffset;
                    fragmentLength = node.inputLength;
                    return true;
                }
                if (renderFrame.stage == TreeRenderDetailSuffix && (node.kind == NodeKind::Identifier || node.kind == NodeKind::Literal))
                {
                    renderFrame.stage = TreeRenderNewline;
                    return ReturnStaticFragment("\"", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TreeRenderDetailPrefix && node.kind == NodeKind::BuiltinType)
                {
                    renderFrame.stage = TreeRenderDetailValue;
                    return ReturnStaticFragment(" ", pFragment, fragmentLength);
                }
                if (renderFrame.stage == TreeRenderDetailValue && node.kind == NodeKind::BuiltinType)
                {
                    renderFrame.stage = TreeRenderDetailSuffix;
                    pFragment = BuiltinText(node.value, fragmentLength);
                    return true;
                }
                if (renderFrame.stage <= TreeRenderNewline)
                {
                    renderFrame.stage = TreeRenderChildren;
                    return ReturnStaticFragment("\n", pFragment, fragmentLength);
                }
                if (renderFrame.depth >= m_treeMaxDepth)
                {
                    --m_renderFrameCount;
                    continue;
                }
                if (renderFrame.childIndex != kInvalidNodeIndex)
                {
                    const NodeIndex childNodeIndex = renderFrame.childIndex;
                    renderFrame.childIndex = m_pNodes[childNodeIndex].nextSiblingIndex;
                    PushRenderFrame(childNodeIndex, renderFrame.depth + 1);
                    continue;
                }
                --m_renderFrameCount;
            }
            return false;
        }

    private:
        const char* m_pInput = nullptr;
        size_t m_inputLength = 0;
        size_t m_inputIndex = 0;
        uint8_t* m_pScratchSpace = nullptr;
        size_t m_scratchSpaceSize = 0;
        size_t m_requiredScratchSpaceSize = 0;

        Node* m_pNodes = nullptr;
        size_t m_nodeCapacity = 0;
        size_t m_nodeCount = 0;
        NodeIndex* m_pSubstitutions = nullptr;
        size_t m_substitutionCapacity = 0;
        size_t m_substitutionCount = 0;
        NodeIndex* m_pTemplateArguments = nullptr;
        size_t m_templateArgumentCapacity = 0;
        size_t m_templateArgumentCount = 0;
        size_t m_templateArgumentBaseIndex = 0;
        ParseWorkFrame* m_pParseWorkFrames = nullptr;
        size_t m_parseWorkFrameCapacity = 0;
        size_t m_parseWorkFrameCount = 0;
        NodeIndex m_parseResult = kInvalidNodeIndex;
        RenderFrame* m_pRenderFrames = nullptr;
        size_t m_renderFrameCapacity = 0;
        size_t m_renderFrameCount = 0;

        NodeIndex m_rootNodeIndex = kInvalidNodeIndex;
        Status m_parseStatus = Status::NotParsed;
        Status m_parseFailureStatus = Status::InvalidInput;
        RenderMode m_renderMode = RenderMode::None;
        uint32_t m_treeMaxDepth = UINT32_MAX;
        size_t m_renderedOutputLength = 0;
        size_t m_fragmentOffset = 0;
        const char* m_pCurrentFragment = nullptr;
        size_t m_currentFragmentLength = 0;
        char m_numericFragment[16] = {};
        bool m_renderComplete = false;
    };

} // namespace nv::trace
