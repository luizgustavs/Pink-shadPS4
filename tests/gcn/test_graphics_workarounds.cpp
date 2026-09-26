// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Code generation of the per-game graphics workarounds (FIXES_IMPLEMENTATIONS_TRACKING.md, §8 of the SotC
// port guide), checked on hand-built IR without a GPU. With SHADPS4_TEST_SPIRV_DIR set, the emitted modules
// are written there so spirv-val can check them

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp>

#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace {

using namespace Shader;

struct SpirvInst {
    spv::Op op;
    std::span<const u32> words; // including the opcode word
};

std::vector<SpirvInst> ParseSpirv(const std::vector<u32>& spirv) {
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

void DumpSpirv(const std::vector<u32>& spirv, const std::string& name) {
    const char* dir = std::getenv("SHADPS4_TEST_SPIRV_DIR");
    if (!dir || !*dir) {
        return;
    }
    std::ofstream out(std::filesystem::path{dir} / (name + ".spv"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(u32));
}

Profile MakeProfile() {
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    profile.needs_lds_barriers = true;
    return profile;
}

/// [entry] if (ReadLane(LocalInvocationId.x, #0) == 0) { [body] } [merge] return. The branch condition is a
/// constant-lane ReadLane, which the wave64 lowering makes workgroup-uniform
struct UniformBranchProgram {
    Pools pools{};
    IR::Program program;
    IR::Block* entry;
    IR::Block* body;
    IR::Block* merge;

    explicit UniformBranchProgram(Info& info) : program{info} {
        entry = pools.block_pool.Create(pools.inst_pool);
        body = pools.block_pool.Create(pools.inst_pool);
        merge = pools.block_pool.Create(pools.inst_pool);
        program.blocks = {entry, body, merge};

        IR::IREmitter ir{*entry};
        const IR::U32 x = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
        const IR::U1 cond = ir.IEqual(ir.ReadLane(x, ir.Imm32(0u)), ir.Imm32(0u));

        auto& list = program.syntax_list;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
        list.back().data.block = entry;
        auto& if_node = list.emplace_back();
        if_node.type = IR::AbstractSyntaxNode::Type::If;
        if_node.data.if_node.cond = cond;
        if_node.data.if_node.body = body;
        if_node.data.if_node.merge = merge;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
        list.back().data.block = body;
        auto& end_if = list.emplace_back();
        end_if.type = IR::AbstractSyntaxNode::Type::EndIf;
        end_if.data.end_if.merge = merge;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
        list.back().data.block = merge;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
        program.post_order_blocks = IR::PostOrder(entry);
    }
};

Info MakeComputeInfo() {
    Info info{};
    info.hw_stage = HwStage::Compute;
    info.sw_stage = SwStage::Compute;
    return info;
}

RuntimeInfo MakeComputeRuntime() {
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Compute, SwStage::Compute);
    runtime_info.hw.cs.workgroup_size = {64, 1, 1};
    runtime_info.hw.cs.shared_memory_size = 1024;
    return runtime_info;
}

u32 CountOpcode(const IR::Block& block, IR::Opcode opcode) {
    return static_cast<u32>(std::ranges::count_if(
        block.Instructions(), [opcode](const IR::Inst& inst) { return inst.GetOpcode() == opcode; }));
}

// lds_barrier_uniform_readlane: shared memory accesses inside a branch on a constant-lane ReadLane keep their
// barriers instead of being treated as divergent code (SotC cs 0x57bd9f27)
TEST(LdsBarrierUniformReadlane, BarriersStayInsideReadLaneUniformBranch) {
    for (const bool enabled : {false, true}) {
        EmulatorSettings.SetLdsBarrierUniformReadlane(enabled);
        Info info = MakeComputeInfo();
        UniformBranchProgram p{info};
        {
            IR::IREmitter ir{*p.body};
            const IR::U32 offset = ir.Imm32(16u);
            ir.WriteShared(32, ir.Imm32(1u), offset);
            (void)ir.LoadShared(32, false, offset);
        }
        Optimization::SharedMemoryBarrierPass(p.program, MakeComputeRuntime(), MakeProfile());
        if (enabled) {
            EXPECT_GT(CountOpcode(*p.body, IR::Opcode::Barrier), 0u);
            EXPECT_EQ(CountOpcode(*p.merge, IR::Opcode::Barrier), 0u);
        } else {
            // Divergent branch: no barrier inside, one at the merge block instead
            EXPECT_EQ(CountOpcode(*p.body, IR::Opcode::Barrier), 0u);
            EXPECT_EQ(CountOpcode(*p.merge, IR::Opcode::Barrier), 1u);
        }
    }
    EmulatorSettings.SetLdsBarrierUniformReadlane(false);
}

bool HasReadLaneOfLane(const IR::Block& block, u32 lane) {
    return std::ranges::any_of(block.Instructions(), [lane](const IR::Inst& inst) {
        return inst.GetOpcode() == IR::Opcode::ReadLane && inst.Arg(1).IsImmediate() &&
               inst.Arg(1).U32() == lane;
    });
}

// wave64_uniform_branches: a ReadLane of the upper half of the wave inside a branch on an already lowered
// value is lowered too (an LDS exchange), instead of reading lane 32 of a 32-wide subgroup
TEST(Wave64UniformBranches, LowersReadLaneInsideUniformBranch) {
    for (const bool enabled : {false, true}) {
        EmulatorSettings.SetWave64UniformBranches(enabled);
        Info info = MakeComputeInfo();
        UniformBranchProgram p{info};
        {
            IR::IREmitter ir{*p.body};
            const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
            (void)ir.ReadLane(y, ir.Imm32(32u));
        }
        Optimization::LowerWave64BallotPass(p.program, MakeComputeRuntime(), MakeProfile());
        EXPECT_EQ(HasReadLaneOfLane(*p.body, 32), !enabled);
        EXPECT_EQ(CountOpcode(*p.body, IR::Opcode::LoadSharedU32) > 0, enabled);
        // The branch condition itself is always lowered
        EXPECT_FALSE(HasReadLaneOfLane(*p.entry, 32));
        EXPECT_GT(CountOpcode(*p.entry, IR::Opcode::LoadSharedU32), 0u);
    }
    EmulatorSettings.SetWave64UniformBranches(false);
}

// With more than one wave, a lowered value is only uniform within its wave: the branch may differ between
// waves, so the ReadLane inside it keeps its default handling (no barriers in divergent control flow)
TEST(Wave64UniformBranches, KeepsDefaultWithSeveralWaves) {
    EmulatorSettings.SetWave64UniformBranches(true);
    Info info = MakeComputeInfo();
    UniformBranchProgram p{info};
    {
        IR::IREmitter ir{*p.body};
        const IR::U32 y = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
        (void)ir.ReadLane(y, ir.Imm32(32u));
    }
    RuntimeInfo runtime_info = MakeComputeRuntime();
    runtime_info.hw.cs.workgroup_size = {128, 1, 1};
    Optimization::LowerWave64BallotPass(p.program, runtime_info, MakeProfile());
    EXPECT_TRUE(HasReadLaneOfLane(*p.body, 32));
    EXPECT_EQ(CountOpcode(*p.body, IR::Opcode::Barrier), 0u);
    EmulatorSettings.SetWave64UniformBranches(false);
}

std::vector<u32> EmitEmptyFragment(bool early_tests, bool has_storage, bool has_discard) {
    Info info{};
    info.hw_stage = HwStage::Fragment;
    info.sw_stage = SwStage::Fragment;
    info.has_storage_images = has_storage;
    info.has_discard = has_discard;
    Pools pools{};
    IR::Program program{info};
    auto* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(block);
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Fragment, SwStage::Fragment);
    runtime_info.hw.fs.early_fragment_tests = early_tests;
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(MakeProfile(), runtime_info, program, bindings);
}

bool HasExecutionMode(const std::vector<u32>& spirv, spv::ExecutionMode mode) {
    return std::ranges::any_of(ParseSpirv(spirv), [mode](const SpirvInst& inst) {
        return inst.op == spv::Op::OpExecutionMode && inst.words.size() >= 3 &&
               inst.words[2] == static_cast<u32>(mode);
    });
}

// early_fragment_tests_from_z_order: EarlyFragmentTests only for pixel shaders whose storage writes depend on
// it and that neither discard nor export depth
TEST(EarlyFragmentTestsFromZOrder, OnlyForStorageWritesWithoutDiscard) {
    const auto with_mode = EmitEmptyFragment(true, true, false);
    DumpSpirv(with_mode, "early_fragment_tests");
    EXPECT_TRUE(HasExecutionMode(with_mode, spv::ExecutionModeEarlyFragmentTests));
    EXPECT_FALSE(
        HasExecutionMode(EmitEmptyFragment(false, true, false), spv::ExecutionModeEarlyFragmentTests));
    EXPECT_FALSE(
        HasExecutionMode(EmitEmptyFragment(true, false, false), spv::ExecutionModeEarlyFragmentTests));
    EXPECT_FALSE(
        HasExecutionMode(EmitEmptyFragment(true, true, true), spv::ExecutionModeEarlyFragmentTests));
}

// dynamic_tsharp_array_size: a T# read from a table becomes an array of array_size + 1 sampled images, and
// the sample instruction loads its element through an access chain
TEST(DynamicTsharpArray, SamplesThroughDescriptorArrayElement) {
    constexpr u32 ArraySize = 16;
    constexpr u32 StrideDw = 8;
    constexpr u32 TableOffset = NUM_USER_DATA_REGS;
    Info info = MakeComputeInfo();
    info.flattened_ud_buf.assign(TableOffset + (ArraySize + 1) * StrideDw, 0);
    ImageResource image{};
    for (u32 i = 0; i < 8; ++i) {
        image.sharp_fetch.offsets[i] = static_cast<SharpLocation>(TableOffset + i);
    }
    image.sharp_fetch.load_mask = 0xFF;
    image.array_size = ArraySize;
    image.array_stride_dw = StrideDw;
    info.images.push_back(image);
    info.samplers.push_back(SamplerResource{});
    ASSERT_EQ(info.images[0].NumBindings(info), ArraySize + 1);

    Pools pools{};
    IR::Program program{info};
    auto* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    {
        IR::IREmitter ir{*block};
        // Clamped like the resource patching pass does; out-of-range slots hit the null binding at ArraySize
        const IR::U32 element = ir.UMin(ir.Imm32(5u), ir.Imm32(ArraySize));
        const IR::Value handle = ir.ImageArrayHandle(ir.Imm32(0u), element);
        const IR::Value coords = ir.CompositeConstruct(ir.Imm32(0.5f), ir.Imm32(0.5f));
        IR::TextureInstInfo tex_info{};
        tex_info.has_lod.Assign(1);
        (void)ir.ImageSampleExplicitLod(handle, coords, ir.Imm32(0.0f), IR::Value{}, tex_info);
    }
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(block);

    Backend::Bindings bindings{};
    const auto spirv =
        Backend::SPIRV::EmitSPIRV(MakeProfile(), MakeComputeRuntime(), program, bindings);
    DumpSpirv(spirv, "dynamic_tsharp_array");

    const auto insts = ParseSpirv(spirv);
    std::unordered_map<u32, u32> constants;
    std::unordered_set<u32> image_arrays;
    std::unordered_set<u32> access_chains;
    bool loads_element = false;
    for (const auto& inst : insts) {
        if (inst.op == spv::Op::OpConstant && inst.words.size() == 4) {
            constants.emplace(inst.words[2], inst.words[3]);
        } else if (inst.op == spv::Op::OpTypeArray) {
            const auto it = constants.find(inst.words[3]);
            if (it != constants.end() && it->second == ArraySize + 1) {
                image_arrays.insert(inst.words[1]);
            }
        } else if (inst.op == spv::Op::OpAccessChain) {
            access_chains.insert(inst.words[2]);
        } else if (inst.op == spv::Op::OpLoad && access_chains.contains(inst.words[3])) {
            loads_element = true;
        }
    }
    EXPECT_EQ(image_arrays.size(), 1u);
    EXPECT_TRUE(loads_element);
    // Bindings: the array takes ArraySize + 1 slots, then the sampler
    EXPECT_EQ(bindings.unified, ArraySize + 1 + 1);
}

} // namespace
