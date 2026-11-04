/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>

namespace nv::trace {

    /// Stores the lambda to be executed at scope-exit by-value.
    template <class TFn>
    class ScopeExitGuardObject
    {
        TFn m_fn;
        bool m_active;

    private:
        ScopeExitGuardObject(const ScopeExitGuardObject<TFn>&);
        ScopeExitGuardObject<TFn>& operator=(const ScopeExitGuardObject<TFn>&);

    public:
        ~ScopeExitGuardObject()
        {
            if (m_active)
            {
                m_fn();
            }
        }
        ScopeExitGuardObject(TFn&& fn)
            : m_fn(std::move(fn))
            , m_active(true)
        {
        }
        ScopeExitGuardObject(ScopeExitGuardObject<TFn>&& rhs)
            : m_fn(std::move(rhs.m_fn))
            , m_active(rhs.m_active)
        {
            rhs.m_active = false;
        }
        void Dismiss()
        {
            m_active = false;
        }
    };

    template <class TFn>
    ScopeExitGuardObject<TFn> ScopeExitGuard(TFn&& fn)
    {
        return ScopeExitGuardObject<TFn>(std::move(fn));
    }

} // namespace nv::trace
