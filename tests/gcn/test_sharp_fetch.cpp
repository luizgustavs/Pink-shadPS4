// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Sharp fetch coverage for contiguous, immediate and constant VGPR cases

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace {

using namespace Shader;

using Summary = SharpFetch<AmdGpu::Buffer>::Summary;

constexpr u32 ImmediateDword = 0xCAFE0000u;

std::array<u32, 16> MakeFlatbuf() {
    std::array<u32, 16> flatbuf;
    for (u32 i = 0; i < flatbuf.size(); ++i) {
        flatbuf[i] = 0x1000u + i;
    }
    return flatbuf;
}

/// One block with a raw load through a V# built from the given dwords (an immediate or a user data sgpr)
struct VsharpProgram {
    Pools pools{};
    Info info{};
    IR::Program program{info};

    explicit VsharpProgram(const std::array<int, 4>& sgprs) {
        info.hw_stage = HwStage::Compute;
        info.sw_stage = SwStage::Compute;
        const auto flatbuf = MakeFlatbuf();
        info.flattened_ud_buf.assign(flatbuf.begin(), flatbuf.end()); // read by the patch pass
        IR::Block* block = pools.block_pool.Create(pools.inst_pool);
        program.blocks = {block};
        IR::IREmitter ir{*block};
        std::array<IR::Value, 4> dwords;
        for (u32 i = 0; i < 4; ++i) {
            dwords[i] = sgprs[i] < 0 ? IR::Value{ImmediateDword + i}
                                     : IR::Value{ir.GetUserData(IR::ScalarReg(sgprs[i]))};
        }
        const IR::Value handle = ir.CompositeConstruct(dwords[0], dwords[1], dwords[2], dwords[3]);
        // Address operand as the translator builds it: (index, voffset, soffset)
        const IR::Value address = ir.CompositeConstruct(ir.Imm32(0u), ir.Imm32(0u), ir.Imm32(0u));
        const IR::Value value = ir.LoadBufferU32(1, handle, address, {});
        ir.StoreBufferU32(1, handle, address, value, {});

        Profile profile{};
        profile.supported_spirv = 0x00010600;
        profile.subgroup_size = 32;
        const auto resources = Optimization::ResourceDiscoverPass(program, profile);
        Optimization::ResourcePatchingPass(info, resources, profile);
    }

    std::array<u32, 4> Fetch(const std::array<u32, 16>& flatbuf) const {
        std::array<u32, 4> out{};
        EXPECT_EQ(info.buffers.size(), 1u);
        EXPECT_TRUE(info.buffers[0].sharp_fetch.Fetch(flatbuf.data(),
                                                      reinterpret_cast<AmdGpu::Buffer*>(out.data())));
        return out;
    }
};

TEST(SharpFetch, ContiguousUserDataIsSingleLoad) {
    const VsharpProgram p{{4, 5, 6, 7}};
    EXPECT_EQ(p.info.buffers[0].sharp_fetch.summary, Summary::SingleLoad);
    EXPECT_EQ(p.Fetch(MakeFlatbuf()), (std::array<u32, 4>{0x1004, 0x1005, 0x1006, 0x1007}));
}

TEST(SharpFetch, ScatteredUserDataIsMultiLoad) {
    const VsharpProgram p{{4, 9, 6, 7}};
    EXPECT_EQ(p.info.buffers[0].sharp_fetch.summary, Summary::MultiLoad);
    EXPECT_EQ(p.Fetch(MakeFlatbuf()), (std::array<u32, 4>{0x1004, 0x1009, 0x1006, 0x1007}));
}

// dword 0 immediate and dwords 1..3 in s1..s3: the unset offsets[0] is 0, so the offsets look contiguous
TEST(SharpFetch, ImmediateDwordKeepsMultiLoad) {
    const VsharpProgram p{{-1, 1, 2, 3}};
    EXPECT_EQ(p.info.buffers[0].sharp_fetch.summary, Summary::MultiLoad);
    EXPECT_EQ(p.Fetch(MakeFlatbuf()), (std::array<u32, 4>{ImmediateDword, 0x1001, 0x1002, 0x1003}));
}

// s_load_dwordx4 s[8:11], s[2:3], 0; s_waitcnt 0; buffer_load_format_xyz v[4:6], v0, s[8:11], 0 idxen;
// v_mov_b32 v7, 1.0; v_add_i32 v0, s5, v0; s_setpc_b64 s[0:1]
TEST(FetchShader, KeepsConstantOneAndVertexOffset) {
    const std::array<u32, 7> code = {
        0xC0000000u | (2u << 22) | (8u << 15) | (1u << 9) | (1u << 8), // s_load_dwordx4
        0xBF800000u | (12u << 16),                                     // s_waitcnt
        0xE0000000u | (2u << 18) | (1u << 13),                         // buffer_load_format_xyz
        (4u << 8) | (2u << 16) | (128u << 24),                         // v[4:6], s[8:11], soffset 0
        0x7E000000u | (7u << 17) | (1u << 9) | 242u,                   // v_mov_b32 v7, 1.0
        (37u << 25) | 5u,                                              // v_add_i32 v0, s5, v0
        0xBE800000u | (32u << 8),                                      // s_setpc_b64 s[0:1]
    };
    std::array<u32, 16> user_data{};
    const u32* code_ptr = code.data();
    std::memcpy(&user_data[0], &code_ptr, sizeof(code_ptr));

    Info info{};
    info.user_data = user_data;
    info.has_fetch_shader = true;
    info.fetch_shader_sgpr_base = 0;
    Gcn::FetchShaderData data{};
    ASSERT_TRUE(Gcn::ParseFetchShader(info, data));
    EXPECT_EQ(data.size, code.size() * sizeof(u32));
    ASSERT_EQ(data.attributes.size(), 1u);
    EXPECT_EQ(data.attributes[0].dest_vgpr, 4u);
    EXPECT_EQ(data.attributes[0].num_elements, 3u);
    EXPECT_EQ(data.attributes[0].sgpr_base, 2u);
    ASSERT_EQ(data.one_vgprs.size(), 1u);
    EXPECT_EQ(data.one_vgprs[0], 7u);
    EXPECT_EQ(data.vertex_offset_sgpr, 5);
    EXPECT_EQ(data.instance_offset_sgpr, -1);
}

} // namespace
