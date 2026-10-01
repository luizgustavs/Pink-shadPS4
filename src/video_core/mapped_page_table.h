// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <vector>
#include "common/types.h"

namespace VideoCore {

/// Track two bits for each 16 KB page in the 40-bit GPU address space, one for a full mapping and
/// one for a partial mapping
/// The rasterizer updates these bits while holding the lock used to change mapped_ranges
/// Readers use a sequence counter to see a consistent table without taking that lock
/// If a range ends inside a partial page, crosses the address limit or overlaps an update, return
/// Unknown and let the interval set answer
/// That keeps uncertain byte ranges on the existing exact path rather than guessing from a partial
/// page
class MappedPageTable {
public:
    enum class Answer : u8 { Unmapped, Mapped, Unknown };

    static constexpr u32 PageBits = 14;
    static constexpr u64 PageSize = 1ULL << PageBits;
    static constexpr u64 AddressLimit = 1ULL << 40;

    MappedPageTable();

    /// Check whether every byte in [addr, end) is mapped, with addr smaller than end, and return
    /// Unknown when the table cannot decide
    Answer Query(VAddr addr, VAddr end) const;

    /// Update the page table after changing the mirrored interval set while still holding the lock
    /// that protects those changes
    /// The covered callback tells us how many bytes remain mapped in each page so partial mappings
    /// are preserved
    /// Callers must finish their interval set update before asking this table to mirror it
    template <typename Covered>
    void Update(VAddr addr, VAddr end, bool mapped, Covered&& covered) {
        end = std::min(end, AddressLimit);
        if (addr >= end) {
            return;
        }
        BeginWrite();
        u64 lo = addr >> PageBits;
        u64 hi = ((end - 1) >> PageBits) + 1;
        // For a page covered only partly by this range, check the remaining mappings too so another
        // range in the same page is not lost
        const auto set_exact = [&](u64 page) {
            const VAddr start = page << PageBits;
            const u64 bytes = covered(start, start + PageSize);
            SetPages(page, page + 1,
                     bytes == PageSize ? State::Full : (bytes != 0 ? State::Partial : State::Empty));
        };
        if (addr % PageSize != 0) {
            set_exact(lo++);
        }
        if (lo < hi && end % PageSize != 0) {
            set_exact(--hi);
        }
        SetPages(lo, hi, mapped ? State::Full : State::Empty);
        EndWrite();
    }

private:
    enum class State : u8 { Empty, Partial, Full };

    static constexpr u64 NumPages = AddressLimit >> PageBits;
    static constexpr u64 NumWords = NumPages / 64;

    void SetPages(u64 lo, u64 hi, State state);
    void BeginWrite();
    void EndWrite();

    /// Store full-page bits at 2 * w and partial-page bits at 2 * w + 1 for the 64 pages belonging
    /// to word w
    /// These words are mutable because reader loads use std::atomic_ref, which needs a non-const
    /// reference
    mutable std::vector<u64> words;
    std::atomic<u64> sequence{};
};

} // namespace VideoCore
