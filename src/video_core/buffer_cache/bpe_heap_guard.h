// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <vector>

#include "common/types.h"

namespace Core {
class MemoryManager;
}

namespace VideoCore {

/**
 * Per-game guard for Shadow of the Colossus (GPU.bpe_heap_guard_address, 0 = off)
 *
 * The BPE heap stores a 16-byte header before each block and tree links inside free blocks
 * Stale compute descriptors can write to freed blocks, and readbacks can then overwrite that metadata
 * This caused guest access violations at eboot+0x48ac5/0x48f3d/0x490a0/0x49476
 *
 * Readbacks preserve the CPU's allocator metadata, which is never valid GPU output
 * The guard reads it through physical backing to bypass tracking protection
 */
class BpeHeapGuard {
public:
    BpeHeapGuard(Core::MemoryManager* memory, VAddr heap);

    bool IsValid() const {
        return valid;
    }

    /// Writes `size` bytes of `data` to guest `addr`, skipping allocator metadata. Returns the number of
    /// skipped bytes whose GPU value differed from the CPU value (0 when harmless)
    u64 Write(VAddr addr, const u8* data, u64 size);

private:
    using Span = std::pair<VAddr, VAddr>;

    /// The spans of allocator metadata overlapping [addr, end), sorted
    std::vector<Span> CollectSpans(VAddr addr, VAddr end);

    bool Read(VAddr addr, void* out, u64 size) const;
    bool IsHeapPointer(VAddr addr) const;
    void AddFreeBlock(VAddr block);
    void CollectFreeTree();
    void CollectHeaders(VAddr lo, VAddr hi, std::vector<Span>& spans) const;
    bool NextBlock(VAddr block, VAddr& next) const;

    Core::MemoryManager* memory;
    VAddr heap;
    bool valid = false;
    std::vector<VAddr> free_blocks; // sorted block starts
    std::vector<Span> free_spans;   // sorted, header + tree links of every free block
    // Every backing run resolved during this guard's walk, sorted by start. The free-tree walk reads
    // thousands of nodes per readback; resolving each one through the VMA map was ~25 ms per SotC frame, and
    // the tree spreads over many physical areas
    struct Run {
        VAddr start;
        u64 size;
        const u8* ptr;
    };
    mutable std::vector<Run> runs;
};

} // namespace VideoCore
