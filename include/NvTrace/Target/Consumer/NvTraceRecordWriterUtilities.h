/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace nv::trace {

    template <typename TWriter>
    bool CommitOrRollback(TWriter& writer, bool succeeded)
    {
        if (succeeded)
        {
            writer.Commit();
        }
        else
        {
            writer.Rollback();
        }
        return succeeded;
    }

} // namespace nv::trace
