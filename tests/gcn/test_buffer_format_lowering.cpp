// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Unsupported buffer formats lower as null buffers

#include <algorithm>
#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

namespace {

using namespace Shader;
using AmdGpu::DataFormat;
using AmdGpu::NumberFormat;

AmdGpu::Buffer MakeVsharp(u32 data_format, u32 num_format) {
    AmdGpu::Buffer vsharp{};
    vsharp.base_address = 0x10000;
    vsharp.stride = 16;
    vsharp.num_records = 64;
    vsharp.data_format = data_format;
    vsharp.num_format = num_format;
    return vsharp;
}

/// One formatted load and store using an immediate buffer descriptor
struct FormatProgram {
    Pools pools{};
    Info info{};
    IR::Program program{info};
    IR::Block* block;

    FormatProgram(const AmdGpu::Buffer& vsharp, IR::BufferInstInfo flags) {
        info.hw_stage = HwStage::Compute;
        info.sw_stage = SwStage::Compute;
        BufferResource buffer{};
        std::memcpy(buffer.sharp_fetch.immediates.data(), &vsharp, sizeof(vsharp));
        buffer.sharp_fetch.offsets.fill(0);
        buffer.sharp_fetch.load_mask = 0;
        buffer.is_formatted = true;
        info.buffers.push_back(buffer);

        block = pools.block_pool.Create(pools.inst_pool);
        program.blocks = {block};
        IR::IREmitter ir{*block};
        const IR::Value value = ir.LoadBufferFormat(ir.Imm32(0u), ir.Imm32(0u), flags);
        ir.StoreBufferFormat(ir.Imm32(0u), ir.Imm32(16u), value, flags);
    }

    u32 Count(std::initializer_list<IR::Opcode> opcodes) const {
        return static_cast<u32>(std::ranges::count_if(block->Instructions(), [&](const IR::Inst& inst) {
            return std::ranges::find(opcodes, inst.GetOpcode()) != opcodes.end();
        }));
    }

    u32 FormatOps() const {
        return Count({IR::Opcode::LoadBufferFormatF32, IR::Opcode::StoreBufferFormatF32});
    }

    u32 RawOps() const {
        using enum IR::Opcode;
        return Count({LoadBufferU8, LoadBufferU16, LoadBufferU32, LoadBufferU32x2, LoadBufferU32x3,
                      LoadBufferU32x4, LoadBufferU64, LoadBufferF32, LoadBufferF32x2, LoadBufferF32x3,
                      LoadBufferF32x4, StoreBufferU8, StoreBufferU16, StoreBufferU32, StoreBufferU32x2,
                      StoreBufferU32x3, StoreBufferU32x4, StoreBufferU64, StoreBufferF32,
                      StoreBufferF32x2, StoreBufferF32x3, StoreBufferF32x4});
    }
};

struct FormatCase {
    u32 data_format;
    NumberFormat num_format;
};

// Garbage V# pairs that reached an UNREACHABLE or ASSERT: data format 15 (SotC cs 0x41d379bc), 8-bit and
// 2_10_10_10 with float, 32-bit with snorm. They lower like a null buffer: no memory access left
TEST(BufferFormatLowering, UnsupportedVsharpFormatsLowerAsNullBuffer) {
    const FormatCase cases[] = {
        {15, NumberFormat::Float},
        {15, NumberFormat::Unorm},
        {u32(DataFormat::Format8_8), NumberFormat::Float},
        {u32(DataFormat::Format8_8_8_8), NumberFormat::Float},
        {u32(DataFormat::Format2_10_10_10), NumberFormat::Float},
        {u32(DataFormat::Format10_10_10_2), NumberFormat::Float}, // remapped to 2_10_10_10
        {u32(DataFormat::Format32), NumberFormat::Snorm},
        {u32(DataFormat::Format32_32_32_32), NumberFormat::Snorm},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(testing::Message() << "data_format " << c.data_format << " num_format "
                                        << u32(c.num_format));
        FormatProgram p{MakeVsharp(c.data_format, u32(c.num_format)), {}};
        Optimization::LowerBufferFormatToRaw(p.program);
        EXPECT_EQ(p.FormatOps(), 0u);
        EXPECT_EQ(p.RawOps(), 0u);
    }
}

// The typed-instruction path (MTBUF dfmt) goes through the same check
TEST(BufferFormatLowering, UnsupportedInstructionFormatLowersAsNullBuffer) {
    IR::BufferInstInfo flags{};
    flags.typed.Assign(1);
    flags.inst_data_fmt.Assign(DataFormat(15));
    flags.inst_num_fmt.Assign(NumberFormat::Float);
    FormatProgram p{MakeVsharp(u32(DataFormat::Format32), u32(NumberFormat::Float)), flags};
    Optimization::LowerBufferFormatToRaw(p.program);
    EXPECT_EQ(p.FormatOps(), 0u);
    EXPECT_EQ(p.RawOps(), 0u);
}

// Supported pairs keep lowering to raw accesses
TEST(BufferFormatLowering, SupportedFormatsStillLowerToRawAccesses) {
    const FormatCase cases[] = {
        {u32(DataFormat::Format8_8_8_8), NumberFormat::Unorm},
        {u32(DataFormat::Format16_16), NumberFormat::Float},
        {u32(DataFormat::Format2_10_10_10), NumberFormat::Snorm},
        {u32(DataFormat::Format10_11_11), NumberFormat::Float},
        {u32(DataFormat::Format32_32_32_32), NumberFormat::Float},
        {u32(DataFormat::Format32), NumberFormat::Unorm}, // remapped to uint
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(testing::Message() << "data_format " << c.data_format << " num_format "
                                        << u32(c.num_format));
        FormatProgram p{MakeVsharp(c.data_format, u32(c.num_format)), {}};
        Optimization::LowerBufferFormatToRaw(p.program);
        EXPECT_EQ(p.FormatOps(), 0u);
        EXPECT_EQ(p.RawOps(), 2u);
    }
}

} // Anonymous namespace
