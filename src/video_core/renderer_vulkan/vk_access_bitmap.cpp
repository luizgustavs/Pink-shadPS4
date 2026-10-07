// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>

#include "common/div_ceil.h"
#include "video_core/renderer_vulkan/vk_access_bitmap.h"

namespace Vulkan {

namespace {

/// Bits first..last (inclusive) of a word
constexpr u64 BitRange(u64 first, u64 last) {
    return (~u64{0} >> (63 - last)) & (~u64{0} << first);
}

void SetBits(std::vector<u64>& words, u64 first, u64 last) {
    const u64 w0 = first / 64;
    const u64 w1 = last / 64;
    if (w0 == w1) {
        words[w0] |= BitRange(first % 64, last % 64);
        return;
    }
    words[w0] |= ~u64{0} << (first % 64);
    for (u64 w = w0 + 1; w < w1; ++w) {
        words[w] = ~u64{0};
    }
    words[w1] |= ~u64{0} >> (63 - last % 64);
}

bool AnyBits(const std::vector<u64>& words, u64 first, u64 last) {
    const u64 w0 = first / 64;
    const u64 w1 = last / 64;
    if (w0 == w1) {
        return (words[w0] & BitRange(first % 64, last % 64)) != 0;
    }
    if ((words[w0] & (~u64{0} << (first % 64))) != 0) {
        return true;
    }
    for (u64 w = w0 + 1; w < w1; ++w) {
        if (words[w] != 0) {
            return true;
        }
    }
    return (words[w1] & (~u64{0} >> (63 - last % 64))) != 0;
}

constexpr bool Intersects(const Interval& range, u64 start, u64 end) {
    return range.start < end && start < range.end;
}

} // Anonymous namespace

AccessBitmap::AccessBitmap(u64 size_bytes_) : size_bytes{size_bytes_} {
    const u64 num_chunks = Common::DivCeil(size_bytes, SmallLimit);
    coverage = num_chunks << ChunkShift;
    granules.resize(num_chunks * WordsPerChunk);
    chunks.resize(Common::DivCeil(num_chunks, u64{64}));
}

void AccessBitmap::Add(u64 start, u64 end) {
    if (start >= end) [[unlikely]] {
        return;
    }
    static_assert(RecentSize == 256);
    Recent& last = recent[((start ^ (end << 7)) * 0x9E3779B97F4A7C15ULL) >> 56];
    if (last.epoch == epoch && last.start == start && last.end == end) {
        return;
    }
    last = {start, end, epoch};
    if (end - start > SmallLimit || end > coverage) {
        const bool contained = std::ranges::any_of(
            big, [&](const Interval& range) { return range.start <= start && end <= range.end; });
        if (!contained) {
            big.push_back({start, end});
        }
        return;
    }
    small.push_back({start, end});

    SetBits(granules, start >> GranuleShift, (end - 1) >> GranuleShift);
    for (u64 chunk = start >> ChunkShift; chunk <= (end - 1) >> ChunkShift; ++chunk) {
        u64& word = chunks[chunk / 64];
        if (word == 0) {
            dirty_chunk_words.push_back(static_cast<u32>(chunk / 64));
        }
        word |= u64{1} << (chunk % 64);
    }
}

bool AccessBitmap::Overlaps(u64 start, u64 end, bool& bitmap_hit) {
    bitmap_hit = false;
    if (start >= end) [[unlikely]] {
        return false;
    }
    for (const Interval& range : big) {
        if (Intersects(range, start, end)) {
            bitmap_hit = true;
            return true;
        }
    }
    if (small.empty() || start >= coverage) {
        return false;
    }
    if (!AnyGranule(start, std::min(end, coverage))) {
        return false;
    }
    bitmap_hit = true;
    return ExactSmall(start, end);
}

bool AccessBitmap::AnyGranule(u64 start, u64 end) const {
    const u64 c0 = start >> ChunkShift;
    const u64 c1 = (end - 1) >> ChunkShift;
    // A set chunk bit means a granule in the chunk is set: a chunk inside
    // the query matches outright
    const auto check_word = [&](u64 w) {
        u64 bits = chunks[w];
        if (w == c0 / 64) {
            bits &= ~u64{0} << (c0 % 64);
        }
        if (w == c1 / 64) {
            bits &= ~u64{0} >> (63 - c1 % 64);
        }
        while (bits != 0) {
            const u64 chunk = w * 64 + std::countr_zero(bits);
            bits &= bits - 1;
            if (chunk != c0 && chunk != c1) {
                return true;
            }
            const u64 lo = std::max(start, chunk << ChunkShift);
            const u64 hi = std::min(end, (chunk + 1) << ChunkShift);
            if (AnyBits(granules, lo >> GranuleShift, (hi - 1) >> GranuleShift)) {
                return true;
            }
        }
        return false;
    };
    const u64 w0 = c0 / 64;
    const u64 w1 = c1 / 64;
    if (w1 - w0 < std::max<u64>(dirty_chunk_words.size(), 2)) {
        for (u64 w = w0; w <= w1; ++w) {
            if (check_word(w)) {
                return true;
            }
        }
        return false;
    }
    // A query wider than the chunks in use (an unbounded V#) walks
    // only the set chunk words
    for (const u32 w : dirty_chunk_words) {
        if (w >= w0 && w <= w1 && check_word(w)) {
            return true;
        }
    }
    return false;
}

bool AccessBitmap::ExactSmall(u64 start, u64 end) {
    if (small.size() - sorted > UnsortedLimit) {
        SortSmall();
    }
    const auto sorted_end = small.begin() + sorted;
    const auto it =
        std::upper_bound(small.begin(), sorted_end, start,
                         [](u64 value, const Interval& range) { return value < range.end; });
    if (it != sorted_end && it->start < end) {
        return true;
    }
    return std::any_of(sorted_end, small.end(),
                       [&](const Interval& range) { return Intersects(range, start, end); });
}

void AccessBitmap::SortSmall() {
    std::ranges::sort(small, {}, &Interval::start);
    std::size_t w = 0;
    for (std::size_t r = 1; r < small.size(); ++r) {
        if (small[r].start <= small[w].end) {
            small[w].end = std::max(small[w].end, small[r].end);
        } else {
            small[++w] = small[r];
        }
    }
    small.resize(w + 1);
    sorted = small.size();
}

void AccessBitmap::Clear() {
    for (const u32 w : dirty_chunk_words) {
        u64 bits = chunks[w];
        while (bits != 0) {
            const u64 chunk = u64{w} * 64 + std::countr_zero(bits);
            bits &= bits - 1;
            std::fill_n(granules.begin() + chunk * WordsPerChunk, WordsPerChunk, u64{0});
        }
        chunks[w] = 0;
    }
    dirty_chunk_words.clear();
    small.clear();
    sorted = 0;
    big.clear();
    if (++epoch == 0) {
        recent.fill({});
        epoch = 1;
    }
}

} // namespace Vulkan
