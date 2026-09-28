// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SPIR-V test helpers with optional module dumps for spirv-val

#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp>

#include "common/types.h"

namespace SpirvTest {

struct SpirvInst {
    spv::Op op;
    std::span<const u32> words; // including the opcode word
};

inline std::vector<SpirvInst> ParseSpirv(const std::vector<u32>& spirv) {
    std::vector<SpirvInst> insts;
    for (size_t offset = 5; offset < spirv.size();) {
        const u16 word_count = spirv[offset] >> 16;
        if (word_count == 0 || offset + word_count > spirv.size()) {
            ADD_FAILURE() << "malformed SPIR-V at word " << offset;
            break;
        }
        insts.push_back({static_cast<spv::Op>(spirv[offset] & 0xffff),
                         std::span{spirv}.subspan(offset, word_count)});
        offset += word_count;
    }
    return insts;
}

inline void DumpSpirv(const std::vector<u32>& spirv, const std::string& name) {
    const char* dir = std::getenv("SHADPS4_TEST_SPIRV_DIR");
    if (!dir || !*dir) {
        return;
    }
    std::ofstream out(std::filesystem::path{dir} / (name + ".spv"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(u32));
}

} // namespace SpirvTest
