// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <shared_mutex>
#include <utility>
#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_map.hpp>
#include "common/types.h"

namespace Core {

/// Guest stack ranges with a small per-thread cache for stack-free gaps
class StackRangeSet {
public:
    using Pieces = boost::container::small_vector<std::pair<VAddr, u64>, 4>;

    StackRangeSet();

    void Add(VAddr start, VAddr end);
    void Remove(VAddr start, VAddr end);

    /// Read this thread's merged range copy with use_copy
    /// Rebuild it when the generation changes and keep the same
    /// answers as the locked map
    bool Overlaps(VAddr virtual_addr, u64 size, bool use_copy = false) const;
    /// Returns the parts of [virtual_addr, virtual_addr + size) that are not stack memory
    Pieces Subtract(VAddr virtual_addr, u64 size, bool use_copy = false) const;
    /// Returns the parts of [virtual_addr, virtual_addr + size) that are stack memory
    Pieces GetIn(VAddr virtual_addr, u64 size, bool use_copy = false) const;

    /// Changes whenever a range is added or removed
    u64 Generation() const {
        return generation.load(std::memory_order_acquire);
    }

    bool HasRanges() const {
        return has_ranges.load(std::memory_order_acquire);
    }

private:
    using RangeMap = boost::icl::interval_map<VAddr, u32>;

    bool InFreeGap(VAddr virtual_addr, u64 size) const;
    void RememberFreeGap(RangeMap::const_iterator next) const;
    /// Call func with the current sorted and merged [start, end) ranges
    /// Return false without calling it if a nested query is already using the copy
    template <typename Func>
    bool WithCopy(Func&& func) const;

    RangeMap ranges;
    mutable std::shared_mutex mutex;
    // Skip locking until the first range is added
    std::atomic<bool> has_ranges{false};
    std::atomic<u64> generation{0};
    // Stable key for the per-thread gap cache
    const u64 id;
};

} // namespace Core
