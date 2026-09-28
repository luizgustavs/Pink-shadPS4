// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Vulkan SPIR-V checks built from small IR programs without a GPU

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "spirv_test_util.h"

namespace {

using namespace Shader;
using SpirvTest::DumpSpirv;
using SpirvTest::ParseSpirv;
using SpirvTest::SpirvInst;

Profile MakeProfile() {
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;
    return profile;
}

RuntimeInfo MakeRuntime(SwStage stage) {
    RuntimeInfo runtime_info{};
    switch (stage) {
    case SwStage::Vertex:
        runtime_info.Initialize(HwStage::Vertex, SwStage::Vertex);
        break;
    case SwStage::Fragment:
        runtime_info.Initialize(HwStage::Fragment, SwStage::Fragment);
        break;
    default:
        runtime_info.Initialize(HwStage::Compute, SwStage::Compute);
        runtime_info.hw.cs.workgroup_size = {64, 1, 1};
        break;
    }
    return runtime_info;
}

/// One-block program of the given stage whose body is built by `build`; the shader info is collected by the
/// same pass the recompiler runs, so the capabilities come from the opcodes and not from the test
std::vector<u32> EmitProgram(SwStage stage, const std::function<void(IR::IREmitter&)>& build) {
    Info info{};
    info.sw_stage = stage;
    info.hw_stage = stage == SwStage::Vertex     ? HwStage::Vertex
                    : stage == SwStage::Fragment ? HwStage::Fragment
                                                 : HwStage::Compute;
    Pools pools{};
    IR::Program program{info};
    auto* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    {
        IR::IREmitter ir{*block};
        build(ir);
    }
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(block);

    const Profile profile = MakeProfile();
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, MakeRuntime(stage), program, bindings);
}

bool HasCapability(const std::vector<u32>& spirv, spv::Capability capability) {
    return std::ranges::any_of(ParseSpirv(spirv), [capability](const SpirvInst& inst) {
        return inst.op == spv::Op::OpCapability && inst.words[1] == static_cast<u32>(capability);
    });
}

/// Ids decorated BuiltIn `builtin`, and whether any of them is also decorated Flat
struct BuiltinDecorations {
    bool found{};
    bool flat{};
};

BuiltinDecorations FindBuiltin(const std::vector<u32>& spirv, spv::BuiltIn builtin) {
    const auto insts = ParseSpirv(spirv);
    std::unordered_set<u32> ids;
    for (const auto& inst : insts) {
        if (inst.op == spv::Op::OpDecorate && inst.words.size() >= 4 &&
            inst.words[2] == static_cast<u32>(spv::DecorationBuiltIn) &&
            inst.words[3] == static_cast<u32>(builtin)) {
            ids.insert(inst.words[1]);
        }
    }
    BuiltinDecorations result{.found = !ids.empty()};
    for (const auto& inst : insts) {
        if (inst.op == spv::Op::OpDecorate && inst.words.size() >= 3 &&
            inst.words[2] == static_cast<u32>(spv::DecorationFlat) && ids.contains(inst.words[1])) {
            result.flat = true;
        }
    }
    return result;
}

// ShuffleXor (ds_swizzle in xor mode) emits OpGroupNonUniformShuffleXor, which needs the
// GroupNonUniformShuffle capability
TEST(SpirvConformance, ShuffleXorDeclaresShuffleCapability) {
    const auto spirv = EmitProgram(SwStage::Compute, [](IR::IREmitter& ir) {
        (void)ir.ShuffleXor(ir.LaneId(), ir.Imm32(1u));
    });
    DumpSpirv(spirv, "conformance_shuffle_xor");
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniformShuffle));
    EXPECT_TRUE(HasCapability(spirv, spv::CapabilityGroupNonUniform));
}

// SubgroupLocalInvocationId and SubgroupLtMask are integer inputs: Flat is required in a fragment shader
// (VUID-StandaloneSpirv-Flat-04744) and invalid in a vertex shader (VUID-StandaloneSpirv-Flat-06202)
TEST(SpirvConformance, SubgroupInputsFlatOnlyInFragment) {
    const std::pair<SwStage, const char*> stages[] = {
        {SwStage::Vertex, "vertex"},
        {SwStage::Fragment, "fragment"},
        {SwStage::Compute, "compute"},
    };
    for (const auto& [stage, name] : stages) {
        const auto spirv = EmitProgram(stage, [](IR::IREmitter& ir) {
            (void)ir.LaneId();
            (void)ir.GetAttributeU32(IR::Attribute::SubgroupLtMask, 0);
        });
        DumpSpirv(spirv, std::string{"conformance_subgroup_inputs_"} + name);
        const bool fragment = stage == SwStage::Fragment;
        const auto lane_id = FindBuiltin(spirv, spv::BuiltInSubgroupLocalInvocationId);
        const auto lt_mask = FindBuiltin(spirv, spv::BuiltInSubgroupLtMask);
        ASSERT_TRUE(lane_id.found) << name;
        ASSERT_TRUE(lt_mask.found) << name;
        EXPECT_EQ(lane_id.flat, fragment) << name;
        EXPECT_EQ(lt_mask.flat, fragment) << name;
    }
}

} // namespace
