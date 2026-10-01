// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/mapped_page_table.h"

namespace VideoCore {

namespace {

/// Build the mask for the bits in word w whose page numbers fall between first and last, including
/// both ends
u64 WordMask(u64 w, u64 first, u64 last) {
    const u64 lo_bit = w == first / 64 ? first % 64 : 0;
    const u64 hi_bit = w == last / 64 ? last % 64 : 63;
    return (~0ULL >> (63 - hi_bit)) & (~0ULL << lo_bit);
}

u64 Load(u64& word) {
    return std::atomic_ref<u64>(word).load(std::memory_order_relaxed);
}

void Store(u64& word, u64 value) {
    std::atomic_ref<u64>(word).store(value, std::memory_order_relaxed);
}

} // Anonymous namespace

MappedPageTable::MappedPageTable() : words(2 * NumWords) {}

MappedPageTable::Answer MappedPageTable::Query(VAddr addr, VAddr end) const {
    if (end > AddressLimit) {
        return Answer::Unknown;
    }
    const u64 seq = sequence.load(std::memory_order_acquire);
    if (seq % 2 != 0) {
        return Answer::Unknown;
    }
    const u64 first = addr >> PageBits;
    const u64 last = (end - 1) >> PageBits;
    Answer answer = Answer::Mapped;
    for (u64 w = first / 64; w <= last / 64; ++w) {
        const u64 mask = WordMask(w, first, last);
        const u64 missing = mask & ~Load(words[2 * w]);
        if (missing == 0) {
            continue;
        }
        // A range cannot be fully mapped if it contains an empty page or a partial page in the
        // middle
        // Only the first and last pages may be partial, and those still need the interval set to
        // check the exact bytes
        const u64 ends = (w == first / 64 ? 1ULL << (first % 64) : 0) |
                         (w == last / 64 ? 1ULL << (last % 64) : 0);
        if ((missing & ~Load(words[2 * w + 1])) != 0 || (missing & ~ends) != 0) {
            answer = Answer::Unmapped;
            break;
        }
        answer = Answer::Unknown;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (sequence.load(std::memory_order_relaxed) != seq) {
        return Answer::Unknown;
    }
    return answer;
}

void MappedPageTable::SetPages(u64 lo, u64 hi, State state) {
    if (lo >= hi) {
        return;
    }
    const u64 last = hi - 1;
    for (u64 w = lo / 64; w <= last / 64; ++w) {
        const u64 mask = WordMask(w, lo, last);
        const u64 full = Load(words[2 * w]);
        const u64 partial = Load(words[2 * w + 1]);
        Store(words[2 * w], state == State::Full ? full | mask : full & ~mask);
        Store(words[2 * w + 1], state == State::Partial ? partial | mask : partial & ~mask);
    }
}

void MappedPageTable::BeginWrite() {
    sequence.store(sequence.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
}

void MappedPageTable::EndWrite() {
    sequence.store(sequence.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

} // namespace VideoCore
