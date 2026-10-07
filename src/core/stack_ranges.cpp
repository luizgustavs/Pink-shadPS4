// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <iterator>
#include <utility>
#include <vector>
#include "core/stack_ranges.h"

namespace Core {

namespace {

std::atomic<u64> next_set_id{1};

// Gaps with no stack seen by this thread. A query hits an entry only while the generation it was
// taken under is the current one, so a stale entry can only cost a miss
struct FreeGap {
    u64 set_id;
    u64 generation;
    VAddr lo;
    VAddr hi;
};

struct FreeGapCache {
    std::array<FreeGap, 4> entries;
    u32 next;
};

thread_local FreeGapCache free_gaps{};

using Segments = std::vector<std::pair<VAddr, VAddr>>;

// Keep a sorted, merged stack range copy for this thread and generation
// While it is busy, nested signal queries use the locked path to avoid rebuilding it
struct RangeCopy {
    u64 set_id;
    u64 generation;
    Segments segments;
    bool busy;
};

thread_local RangeCopy range_copy{};

/// First segment that ends after `addr`
Segments::const_iterator FirstSegmentAfter(const Segments& segments, VAddr addr) {
    return std::ranges::upper_bound(segments, addr, {}, &std::pair<VAddr, VAddr>::second);
}

} // Anonymous namespace

StackRangeSet::StackRangeSet() : id{next_set_id.fetch_add(1, std::memory_order_relaxed)} {}

void StackRangeSet::Add(VAddr start, VAddr end) {
    std::unique_lock lk{mutex};
    ranges += std::make_pair(boost::icl::interval<VAddr>::right_open(start, end), 1u);
    has_ranges.store(true, std::memory_order_release);
    // Bumped under the unique lock, after the change: a reader that sees the old generation outside
    // the lock answers as if its query ran before this one
    generation.fetch_add(1, std::memory_order_acq_rel);
}

void StackRangeSet::Remove(VAddr start, VAddr end) {
    if (!HasRanges()) {
        return;
    }
    std::unique_lock lk{mutex};
    ranges -= std::make_pair(boost::icl::interval<VAddr>::right_open(start, end), 1u);
    generation.fetch_add(1, std::memory_order_acq_rel);
}

bool StackRangeSet::InFreeGap(VAddr virtual_addr, u64 size) const {
    // Read before any lookup, so a change that races with this query can only make it miss
    const u64 current = generation.load(std::memory_order_acquire);
    for (const FreeGap& gap : free_gaps.entries) {
        // Without `virtual_addr + size`, a range that wraps around misses and takes the locked lookup
        if (gap.set_id == id && gap.generation == current && virtual_addr >= gap.lo &&
            virtual_addr < gap.hi && size <= gap.hi - virtual_addr) {
            return true;
        }
    }
    return false;
}

void StackRangeSet::RememberFreeGap(RangeMap::const_iterator next) const {
    // Called under the shared lock after an empty query, with `next` the first segment after it.
    // Writers bump the generation under the unique lock, so the one read here matches the ranges
    // seen
    // The entry is disowned while it is written: a guest signal handler that runs on this thread in
    // the middle (and faults into a stack query) must not see new bounds with an old key or the
    // reverse
    auto& cache = free_gaps;
    FreeGap& gap = cache.entries[cache.next];
    gap.set_id = 0;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    gap.generation = generation.load(std::memory_order_relaxed);
    gap.lo = next == ranges.begin() ? VAddr{0} : std::prev(next)->first.upper();
    gap.hi = next == ranges.end() ? ~VAddr{0} : next->first.lower();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    gap.set_id = id;
    cache.next = (cache.next + 1) % cache.entries.size();
}

template <typename Func>
bool StackRangeSet::WithCopy(Func&& func) const {
    RangeCopy& copy = range_copy;
    if (copy.busy) {
        return false;
    }
    copy.busy = true;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    // Update the generation after changing the ranges under the write lock
    // A copy with the current generation still gives a valid answer
    if (copy.set_id != id || copy.generation != Generation()) {
        std::shared_lock lk{mutex};
        copy.set_id = id;
        copy.generation = generation.load(std::memory_order_relaxed);
        copy.segments.clear();
        for (const auto& [range, count] : ranges) {
            const VAddr lo = range.lower();
            const VAddr hi = range.upper();
            if (!copy.segments.empty() && copy.segments.back().second == lo) {
                copy.segments.back().second = hi;
            } else {
                copy.segments.emplace_back(lo, hi);
            }
        }
    }
    func(std::as_const(copy.segments));
    std::atomic_signal_fence(std::memory_order_seq_cst);
    copy.busy = false;
    return true;
}

bool StackRangeSet::Overlaps(VAddr virtual_addr, u64 size, bool use_copy) const {
    if (size == 0 || !HasRanges()) {
        return false;
    }
    bool overlaps = false;
    if (use_copy && WithCopy([&](const Segments& segments) {
            const auto it = FirstSegmentAfter(segments, virtual_addr);
            overlaps = it != segments.end() && it->first < virtual_addr + size;
        })) {
        return overlaps;
    }
    if (InFreeGap(virtual_addr, size)) {
        return false;
    }
    std::shared_lock lk{mutex};
    const auto [first, last] = ranges.equal_range(
        boost::icl::interval<VAddr>::right_open(virtual_addr, virtual_addr + size));
    if (first == last) {
        RememberFreeGap(first);
        return false;
    }
    return true;
}

StackRangeSet::Pieces StackRangeSet::Subtract(VAddr virtual_addr, u64 size, bool use_copy) const {
    Pieces result;
    if (size == 0) {
        return result;
    }
    if (!HasRanges()) {
        result.emplace_back(virtual_addr, size);
        return result;
    }
    // The gaps between the (disjoint, sorted) stack segments that overlap the range
    const VAddr end = virtual_addr + size;
    VAddr cursor = virtual_addr;
    if (use_copy && WithCopy([&](const Segments& segments) {
            for (auto it = FirstSegmentAfter(segments, virtual_addr);
                 it != segments.end() && it->first < end; ++it) {
                const VAddr lo = std::max(it->first, virtual_addr);
                if (lo > cursor) {
                    result.emplace_back(cursor, lo - cursor);
                }
                cursor = std::max(cursor, std::min(it->second, end));
            }
        })) {
        if (cursor < end) {
            result.emplace_back(cursor, end - cursor);
        }
        return result;
    }
    if (InFreeGap(virtual_addr, size)) {
        result.emplace_back(virtual_addr, size);
        return result;
    }
    std::shared_lock lk{mutex};
    const auto [first, last] =
        ranges.equal_range(boost::icl::interval<VAddr>::right_open(virtual_addr, end));
    if (first == last) {
        RememberFreeGap(first);
    }
    for (auto it = first; it != last; ++it) {
        const VAddr lo = std::max(it->first.lower(), virtual_addr);
        if (lo > cursor) {
            result.emplace_back(cursor, lo - cursor);
        }
        cursor = std::max(cursor, std::min(it->first.upper(), end));
    }
    if (cursor < end) {
        result.emplace_back(cursor, end - cursor);
    }
    return result;
}

StackRangeSet::Pieces StackRangeSet::GetIn(VAddr virtual_addr, u64 size, bool use_copy) const {
    Pieces result;
    if (size == 0 || !HasRanges()) {
        return result;
    }
    const VAddr end = virtual_addr + size;
    // The copy's segments are already merged
    if (use_copy && WithCopy([&](const Segments& segments) {
            for (auto it = FirstSegmentAfter(segments, virtual_addr);
                 it != segments.end() && it->first < end; ++it) {
                const VAddr lo = std::max(it->first, virtual_addr);
                const VAddr hi = std::min(it->second, end);
                result.emplace_back(lo, hi - lo);
            }
        })) {
        return result;
    }
    if (InFreeGap(virtual_addr, size)) {
        return result;
    }
    std::shared_lock lk{mutex};
    const auto [first, last] =
        ranges.equal_range(boost::icl::interval<VAddr>::right_open(virtual_addr, end));
    if (first == last) {
        RememberFreeGap(first);
    }
    for (auto it = first; it != last; ++it) {
        const VAddr lo = std::max(it->first.lower(), virtual_addr);
        const VAddr hi = std::min(it->first.upper(), end);
        if (hi > lo) {
            if (!result.empty() && result.back().first + result.back().second == lo) {
                // Adjacent ranges with different registration counts are one stack range here
                result.back().second += hi - lo;
            } else {
                result.emplace_back(lo, hi - lo);
            }
        }
    }
    return result;
}

} // namespace Core
