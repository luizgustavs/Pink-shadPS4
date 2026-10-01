// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "shader_recompiler/ir/reinterpret.h"
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

namespace {

using AmdGpu::CompMapping;
using AmdGpu::CompSwizzle;

TEST(ComponentSwizzle, AllDescriptorSelectorsAgreeBetweenCpuAndVulkan) {
    const std::array<float, 4> texel{0.25f, 0.5f, 0.75f, 0.875f};
    const std::array<float, 8> expected{0.f, 1.f, 0.f, 0.f, 0.25f, 0.5f, 0.75f, 0.875f};
    const std::array<vk::ComponentSwizzle, 8> expected_vk{
        vk::ComponentSwizzle::eZero, vk::ComponentSwizzle::eOne, vk::ComponentSwizzle::eZero,
        vk::ComponentSwizzle::eZero, vk::ComponentSwizzle::eR,   vk::ComponentSwizzle::eG,
        vk::ComponentSwizzle::eB,    vk::ComponentSwizzle::eA,
    };
    for (u32 selector = 0; selector < 8; ++selector) {
        SCOPED_TRACE(selector);
        const auto swizzle = CompSwizzle(selector);
        const CompMapping mapping{swizzle, swizzle, swizzle, swizzle};
        EXPECT_EQ(mapping.Apply(texel),
                  (std::array<float, 4>{expected[selector], expected[selector], expected[selector],
                                        expected[selector]}));
        EXPECT_EQ(Vulkan::LiverpoolToVK::ComponentSwizzle(swizzle), expected_vk[selector]);
    }
}

TEST(ComponentSwizzle, ReservedSelectorsShareTheZeroMappingCacheKey) {
    const CompMapping reserved{CompSwizzle(2), CompSwizzle::Green, CompSwizzle(3),
                               CompSwizzle::One};
    const CompMapping zero{CompSwizzle::Zero, CompSwizzle::Green, CompSwizzle::Zero,
                           CompSwizzle::One};
    EXPECT_EQ(reserved.Normalized(), zero);
    EXPECT_EQ(reserved.Normalized().Normalized(), zero);
    EXPECT_EQ(Vulkan::LiverpoolToVK::ComponentMapping(reserved),
              Vulkan::LiverpoolToVK::ComponentMapping(zero));
}

TEST(ComponentSwizzle, DescriptorNormalizationPreservesFormatRemapping) {
    AmdGpu::Image image{};
    AmdGpu::Buffer buffer{};
    image.data_format = buffer.data_format = u32(AmdGpu::DataFormat::Format10_10_10_2);
    image.dst_sel_x = buffer.dst_sel_x = 2;
    image.dst_sel_y = buffer.dst_sel_y = u32(CompSwizzle::Green);
    image.dst_sel_z = buffer.dst_sel_z = 3;
    image.dst_sel_w = buffer.dst_sel_w = u32(CompSwizzle::One);
    const CompMapping expected{CompSwizzle::One, CompSwizzle::Zero, CompSwizzle::Green,
                               CompSwizzle::Zero};
    EXPECT_EQ(image.DstSelect(), expected);
    EXPECT_EQ(buffer.DstSelect(), expected);
}

TEST(ComponentSwizzle, SingleChannelAlphaRemappingStillReadsRed) {
    AmdGpu::Image image{};
    image.data_format = u32(AmdGpu::DataFormat::Format8);
    image.dst_sel_x = 2;
    image.dst_sel_y = 3;
    image.dst_sel_z = u32(CompSwizzle::Alpha);
    image.dst_sel_w = u32(CompSwizzle::One);
    const CompMapping expected{CompSwizzle::Zero, CompSwizzle::Zero, CompSwizzle::Red,
                               CompSwizzle::One};
    EXPECT_EQ(image.DstSelect(), expected);
}

TEST(ComponentSwizzle, ShaderShuffleUsesCanonicalSelectors) {
    Shader::Pools pools{};
    auto* block = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter ir{*block};
    const auto texel =
        ir.CompositeConstruct(ir.Imm32(0.25f), ir.Imm32(0.5f), ir.Imm32(0.75f), ir.Imm32(0.875f));
    const CompMapping reserved{CompSwizzle(2), CompSwizzle::Green, CompSwizzle(3),
                               CompSwizzle::One};
    const auto result = Shader::IR::ApplySwizzle(ir, texel, reserved);
    const auto* shuffle = result.Inst();
    ASSERT_EQ(shuffle->GetOpcode(), Shader::IR::Opcode::CompositeShuffleF32x4);
    EXPECT_EQ(shuffle->Arg(2).U32(), 0u);
    EXPECT_EQ(shuffle->Arg(3).U32(), 5u);
    EXPECT_EQ(shuffle->Arg(4).U32(), 0u);
    EXPECT_EQ(shuffle->Arg(5).U32(), 1u);
}

TEST(ComponentSwizzle, ValuesOutsideDescriptorRangeRemainInvalid) {
    // Values outside the selector range should still reach the warning path, even though the
    // reserved selectors are accepted as zero
    EXPECT_EQ(AmdGpu::NormalizeCompSwizzle(CompSwizzle(8)), CompSwizzle(8));
    EXPECT_EQ(AmdGpu::NormalizeCompSwizzle(CompSwizzle(255)), CompSwizzle(255));
}

} // namespace
