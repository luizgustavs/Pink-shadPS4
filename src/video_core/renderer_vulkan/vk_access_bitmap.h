// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <vector>

#include "common/interval_set.h"
#include "common/types.h"

namespace Vulkan {

/// Track buffer ranges since the last barrier with bitmaps and exact ranges
/// Use 256 B granules and 64 KiB summary chunks for ranges up to 64 KiB
/// Check larger ranges separately and confirm bitmap hits against exact ranges
class AccessBitmap {
public:
    explicit AccessBitmap(u64 size_bytes);

    [[nodiscard]] u64 SizeBytes() const noexcept {
        return size_bytes;
    }

    void Add(u64 start, u64 end);

    /// Check whether [start, end) overlaps an added range
    /// bitmap_hit reports a bitmap or large-range hit before the exact check
    [[nodiscard]] bool Overlaps(u64 start, u64 end, bool& bitmap_hit);

    /// Removes every range, touching only the chunks that were set
    void Clear();

private:
    static constexpr u64 GranuleShift = 8;
    static constexpr u64 ChunkShift = 16;
    static constexpr u64 WordsPerChunk = (u64{1} << (ChunkShift - GranuleShift)) / 64;
    static constexpr u64 SmallLimit = u64{1} << ChunkShift;
    static constexpr std::size_t RecentSize = 256;
    static constexpr std::size_t UnsortedLimit = 32;

    /// Whether a granule of [start, end) is set; the range is inside the bitmap
    [[nodiscard]] bool AnyGranule(u64 start, u64 end) const;
    [[nodiscard]] bool ExactSmall(u64 start, u64 end);
    /// Sorts and merges the small ranges, so the whole list can be binary searched
    void SortSmall();

    u64 size_bytes;
    u64 coverage;
    std::vector<u64> granules;
    std::vector<u64> chunks;
    std::vector<u32> dirty_chunk_words;
    // Ranges of up to 64 KiB; the first 'sorted' ones are sorted, disjoint and merged
    std::vector<Interval> small;
    std::size_t sorted{};
    std::vector<Interval> big;
    // Ranges added since the last Clear, by hash: the same
    // bindings come back on every draw
    struct Recent {
        u64 start;
        u64 end;
        u32 epoch;
    };
    std::array<Recent, RecentSize> recent{};
    u32 epoch{1};
};

} // namespace Vulkan
