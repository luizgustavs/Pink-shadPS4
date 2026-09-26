// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/guest_write_journal.h"

namespace Common::GuestWriteJournal {

namespace {

// Layout shared with tools/harness/journal_query.py
struct Entry {
    u64 seq; // index + 1 once the entry is complete, 0 while being written
    u64 time_ns;
    u64 addr;
    u64 size;
    u32 source;
    u32 data_len;
    u64 aux;
    u8 data[16];
};
static_assert(sizeof(Entry) == 64);

struct Header {
    char magic[8]; // "SHADGWJ1"
    u32 entry_size;
    u32 capacity;
    u64 next_index;
    u64 crash_time_ns;
    u64 fault_rip;
    u64 fault_addr;
};

constexpr u64 Capacity = 1ULL << 21; // 128 MiB of demand-zero memory

const char* OutputPath() {
    static const char* path = std::getenv("SHADPS4_GUEST_WRITE_JOURNAL");
    return path;
}

const auto StartTime = std::chrono::steady_clock::now();

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - StartTime)
                                .count());
}

Entry* Ring() {
    static Entry* ring = new Entry[Capacity];
    return ring;
}

std::atomic<u64> next_index{0};
std::atomic_flag dumped{};

} // namespace

bool IsEnabled() {
    static const bool enabled = OutputPath() != nullptr && *OutputPath() != '\0';
    return enabled;
}

void RecordSlow(Source source, u64 addr, u64 size, const void* data, u64 aux) {
    Entry* ring = Ring();
    const u64 index = next_index.fetch_add(1, std::memory_order_relaxed);
    Entry& entry = ring[index & (Capacity - 1)];
    std::atomic_ref<u64>(entry.seq).store(0, std::memory_order_relaxed);
    entry.time_ns = NowNs();
    entry.addr = addr;
    entry.size = size;
    entry.source = static_cast<u32>(source);
    entry.aux = aux;
    entry.data_len =
        data != nullptr ? static_cast<u32>(std::min<u64>(size, sizeof(entry.data))) : 0;
    if (entry.data_len != 0) {
        std::memcpy(entry.data, data, entry.data_len);
    }
    std::atomic_ref<u64>(entry.seq).store(index + 1, std::memory_order_release);
}

void DumpOnCrash(u64 fault_rip, u64 fault_addr) noexcept {
    if (!IsEnabled() || dumped.test_and_set()) {
        return;
    }
    std::FILE* file = std::fopen(OutputPath(), "wb");
    if (file == nullptr) {
        return;
    }
    Header header{};
    std::memcpy(header.magic, "SHADGWJ1", 8);
    header.entry_size = sizeof(Entry);
    header.capacity = static_cast<u32>(Capacity);
    header.next_index = next_index.load(std::memory_order_acquire);
    header.crash_time_ns = NowNs();
    header.fault_rip = fault_rip;
    header.fault_addr = fault_addr;
    std::fwrite(&header, sizeof(header), 1, file);
    const u64 used = std::min<u64>(header.next_index, Capacity);
    std::fwrite(Ring(), sizeof(Entry), used, file);
    std::fclose(file);
}

} // namespace Common::GuestWriteJournal
