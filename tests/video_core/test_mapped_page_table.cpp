// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <random>

#include <boost/icl/interval_set.hpp>
#include <gtest/gtest.h>

#include "video_core/mapped_page_table.h"

namespace {

using VideoCore::MappedPageTable;
using Set = boost::icl::interval_set<VAddr>;

void Update(MappedPageTable& table, Set& set, VAddr addr, u64 size, bool map) {
    const auto interval = Set::interval_type::right_open(addr, addr + size);
    if (map) {
        set += interval;
    } else {
        set -= interval;
    }
    table.Update(addr, addr + size, map, [&](VAddr start, VAddr end) {
        u64 bytes = 0;
        for (const auto& range : set & Set::interval_type::right_open(start, end)) {
            bytes += boost::icl::length(range);
        }
        return bytes;
    });
}

TEST(MappedPageTable, PartialPagesAreUnknownOrExact) {
    MappedPageTable table;
    Set set;
    constexpr VAddr Base = 0x200000000ULL;
    Update(table, set, Base, 4 * MappedPageTable::PageSize, true);
    EXPECT_EQ(table.Query(Base, Base + 4 * MappedPageTable::PageSize),
              MappedPageTable::Answer::Mapped);
    EXPECT_EQ(table.Query(Base, Base + 5 * MappedPageTable::PageSize),
              MappedPageTable::Answer::Unmapped);

    // Removing the middle of this page leaves a partial mapping, so a range ending inside it must
    // ask the interval set for the exact answer
    Update(table, set, Base + MappedPageTable::PageSize + 0x100, 0x100, false);
    EXPECT_EQ(table.Query(Base, Base + MappedPageTable::PageSize + 0x80),
              MappedPageTable::Answer::Unknown);
    EXPECT_EQ(table.Query(Base, Base + 3 * MappedPageTable::PageSize),
              MappedPageTable::Answer::Unmapped);
    EXPECT_EQ(table.Query(MappedPageTable::AddressLimit - 0x10, MappedPageTable::AddressLimit + 0x10),
              MappedPageTable::Answer::Unknown);
}

// Compare randomized page table queries with boost::icl::interval_set, which provides the exact
// answer used when the table returns Unknown
TEST(MappedPageTable, MatchesIntervalSet) {
    MappedPageTable table;
    Set set;
    std::mt19937_64 rng{12345};
    const auto rand = [&](u64 lo, u64 hi) { return std::uniform_int_distribution<u64>{lo, hi}(rng); };

    // Use a 64 MB window near the top of the 40-bit address space to exercise ranges crossing the
    // limit, or a window near 0x200000000
    const auto pick_addr = [&](bool aligned) {
        const VAddr base = rand(0, 3) == 0 ? MappedPageTable::AddressLimit - 32_MB : 0x200000000ULL;
        VAddr addr = base + rand(0, 64_MB);
        if (aligned) {
            addr &= ~(MappedPageTable::PageSize - 1);
        }
        return addr;
    };

    u64 queries = 0;
    u64 unknown = 0;
    for (int op = 0; op < 4000; ++op) {
        const bool aligned = rand(0, 9) != 0;
        const VAddr addr = pick_addr(aligned);
        u64 size = rand(0, 3) == 0 ? rand(1, 8_MB) : rand(1, 256_KB);
        if (aligned) {
            size = (size + MappedPageTable::PageSize - 1) & ~(MappedPageTable::PageSize - 1);
        }
        Update(table, set, addr, size, rand(0, 2) != 0);

        for (int q = 0; q < 50; ++q) {
            const VAddr qa = pick_addr(rand(0, 1) != 0) + rand(0, 3);
            const u64 qs = rand(0, 2) == 0 ? rand(1, 16_MB) : rand(1, 64_KB);
            const bool exact = boost::icl::contains(set, Set::interval_type::right_open(qa, qa + qs));
            const auto answer = table.Query(qa, qa + qs);
            ++queries;
            if (answer == MappedPageTable::Answer::Unknown) {
                ++unknown;
                continue;
            }
            ASSERT_EQ(answer == MappedPageTable::Answer::Mapped, exact)
                << std::hex << qa << "+" << qs << " op " << std::dec << op;
        }
    }
    // Partial pages may need the Unknown fallback, but most of these queries should still get their
    // answer directly from the table
    EXPECT_LT(unknown * 4, queries);
}

} // namespace
