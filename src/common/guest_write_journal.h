// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// Track writes into guest memory and the stack, mapping and fault events around them in a ring buffer
// Set SHADPS4_GUEST_WRITE_JOURNAL=<output path> to dump it on a crash and trace writes to a corrupt address
// Use tools/harness/journal_query.py to inspect the dump
namespace Common::GuestWriteJournal {

// The values are part of the file format; never renumber them
enum class Source : u32 {
    BufferReadback = 1,
    BufferReadbackAsync = 2, // unused: readbacks are synchronous in this buffer cache
    ImageDownload = 3,
    Label = 4,
    WriteData = 5,
    DumpConstRam = 6,
    FillCpu = 7,
    FillGpu = 8,
    CopyCpu = 9,
    CopyGpu = 10,
    StackRegister = 11,
    StackUnregister = 12,
    CpuAuthoritativeRelease = 13,
    WrittenBinding = 14,
    ReadbackSkippedCpuModified = 15, // unused: CPU and GPU page states are exclusive here
    FaultWrite = 16,
    FaultRead = 17,
    FaultIgnoredNotGpuMapped = 18,
    GpuMap = 19,
    GpuUnmap = 20,
    Upload = 21,
    StaleGpuRangeDropped = 22, // unused: obsolete with the arena buffer cache
    GuestProtect = 23,
    HeapGuardKept = 24,
};

bool IsEnabled();

void RecordSlow(Source source, u64 addr, u64 size, const void* data, u64 aux);

/// Records one event. `data` (optional) is the source of the write; its first 16 bytes are kept. `aux` is
/// source specific (shader hash for bindings, readback window start, fill value)
inline void Record(Source source, u64 addr, u64 size, const void* data = nullptr, u64 aux = 0) {
    if (IsEnabled()) {
        RecordSlow(source, addr, size, data, aux);
    }
}

/// Writes the ring to the output path, once. Called from the crash handler; never throws
void DumpOnCrash(u64 fault_rip, u64 fault_addr) noexcept;

} // namespace Common::GuestWriteJournal
