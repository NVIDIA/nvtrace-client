/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace nv::trace {

    // Compile-time reflection of a callable's signature: its return type, its arity, and the type of
    // each parameter by index via Arg<Index>. Handles free functions, function pointers,
    // (const-qualified) member function pointers, and functors/lambdas (reflected through their
    // operator()). Arg<Index> with Index out of range is a compile error. Does not handle noexcept- or ref-qualified
    // callables (unused in NvTrace) -- add the corresponding specializations if that changes.
    //
    // Primary template: delegate any functor/lambda to its operator().
    template <typename TCallable>
    struct CallableTraits : CallableTraits<decltype(&std::remove_reference_t<TCallable>::operator())>
    {
    };

    template <typename TReturnType, typename... TArgs>
    struct CallableTraits<TReturnType(TArgs...)>
    {
        using ReturnType = TReturnType;
        using ArgTypes = std::tuple<TArgs...>;

        static constexpr size_t Arity = sizeof...(TArgs);

        template <size_t TIndex>
        using ArgType = std::tuple_element_t<TIndex, ArgTypes>;
    };

    template <typename TReturnType, typename... TArgs>
    struct CallableTraits<TReturnType (*)(TArgs...)> : CallableTraits<TReturnType(TArgs...)>
    {
    };

    // Member function pointers: the implicit object parameter is not counted in Arity. A
    // non-mutable lambda's operator() is const, so the const overload covers the lambda case.
    template <typename TClass, typename TReturnType, typename... TArgs>
    struct CallableTraits<TReturnType (TClass::*)(TArgs...)> : CallableTraits<TReturnType(TArgs...)>
    {
    };

    template <typename TClass, typename TReturnType, typename... TArgs>
    struct CallableTraits<TReturnType (TClass::*)(TArgs...) const> : CallableTraits<TReturnType(TArgs...)>
    {
    };

} // namespace nv::trace
