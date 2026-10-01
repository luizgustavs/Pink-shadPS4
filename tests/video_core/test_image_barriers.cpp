// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/texture_cache/image_barriers.h"

namespace {

using namespace VideoCore;
using Stage = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;
using Layout = vk::ImageLayout;

class ImageBarriersTest : public testing::Test {
protected:
    void Transit(Layout layout, vk::AccessFlags2 access, vk::PipelineStageFlags2 stage,
                 std::optional<SubresourceRange> range = {}) {
        GetImageBarriers({}, vk::ImageAspectFlagBits::eColor, resources, state, subresources,
                         barriers, layout, access, stage, range);
    }

    SubresourceExtent resources{2, 2};
    ImageState state{};
    std::vector<ImageState> subresources;
    ImageBarriers barriers;
    const SubresourceRange tile{{1, 1}, {1, 1}};
};

TEST_F(ImageBarriersTest, ConsecutiveStorageDispatchesNeedMemoryDependency) {
    const auto access = Access::eShaderRead | Access::eShaderWrite;
    Transit(Layout::eGeneral, access, Stage::eComputeShader);
    barriers.clear();
    Transit(Layout::eGeneral, access, Stage::eComputeShader);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcAccessMask, access);
    EXPECT_EQ(barriers[0].dstAccessMask, access);
    EXPECT_EQ(barriers[0].oldLayout, Layout::eGeneral);
    EXPECT_EQ(barriers[0].newLayout, Layout::eGeneral);
    EXPECT_EQ(barriers[0].srcStageMask, Stage::eComputeShader);
    EXPECT_EQ(barriers[0].dstStageMask, Stage::eComputeShader);
}

TEST_F(ImageBarriersTest, ClearThenCopyOrdersTransferWritesAcrossStages) {
    Transit(Layout::eTransferDstOptimal, Access::eTransferWrite, Stage::eClear);
    barriers.clear();
    Transit(Layout::eTransferDstOptimal, Access::eTransferWrite, Stage::eCopy);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcStageMask, Stage::eClear);
    EXPECT_EQ(barriers[0].dstStageMask, Stage::eCopy);
    EXPECT_EQ(barriers[0].srcAccessMask, Access::eTransferWrite);
}

TEST_F(ImageBarriersTest, StorageWriteUsesSynchronization2AccessMask) {
    Transit(Layout::eGeneral, Access::eShaderStorageWrite, Stage::eComputeShader);
    barriers.clear();
    Transit(Layout::eGeneral, Access::eShaderStorageWrite, Stage::eComputeShader);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcAccessMask, Access::eShaderStorageWrite);
}

TEST_F(ImageBarriersTest, PartialStorageWritesOnlyBarrierTheAffectedMipAndLayer) {
    const auto access = Access::eShaderRead | Access::eShaderWrite;
    Transit(Layout::eGeneral, access, Stage::eComputeShader, tile);
    barriers.clear();
    Transit(Layout::eGeneral, access, Stage::eComputeShader, tile);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].subresourceRange.baseMipLevel, 1u);
    EXPECT_EQ(barriers[0].subresourceRange.baseArrayLayer, 1u);
    EXPECT_EQ(barriers[0].subresourceRange.levelCount, 1u);
    EXPECT_EQ(barriers[0].subresourceRange.layerCount, 1u);
    EXPECT_EQ(subresources[0].layout, Layout::eUndefined);
    EXPECT_EQ(subresources[1].layout, Layout::eUndefined);
    EXPECT_EQ(subresources[2].layout, Layout::eUndefined);
}

TEST_F(ImageBarriersTest, ReadOnlyReuseSkipsBarrierButKeepsEveryReaderStage) {
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eFragmentShader);
    barriers.clear();
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eComputeShader);
    EXPECT_TRUE(barriers.empty());
    Transit(Layout::eTransferDstOptimal, Access::eTransferWrite, Stage::eCopy);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcStageMask, Stage::eFragmentShader | Stage::eComputeShader);
}

TEST_F(ImageBarriersTest, PartialReadReuseKeepsEveryReaderStage) {
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eFragmentShader, tile);
    barriers.clear();
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eComputeShader, tile);
    EXPECT_TRUE(barriers.empty());
    Transit(Layout::eGeneral, Access::eShaderWrite, Stage::eComputeShader, tile);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcStageMask, Stage::eFragmentShader | Stage::eComputeShader);
}

TEST_F(ImageBarriersTest, WholeImageConsolidationRetainsPartialReaderScopes) {
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eFragmentShader);
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eComputeShader, tile);
    barriers.clear();
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eVertexShader);
    EXPECT_TRUE(barriers.empty());
    EXPECT_TRUE(subresources.empty());
    Transit(Layout::eTransferDstOptimal, Access::eTransferWrite, Stage::eCopy);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcStageMask,
              Stage::eFragmentShader | Stage::eComputeShader | Stage::eVertexShader);
    EXPECT_EQ(barriers[0].subresourceRange.levelCount, VK_REMAINING_MIP_LEVELS);
    EXPECT_EQ(barriers[0].subresourceRange.layerCount, VK_REMAINING_ARRAY_LAYERS);
}

TEST_F(ImageBarriersTest, WholeImageWriteSynchronizesEveryPartialState) {
    Transit(Layout::eTransferDstOptimal, Access::eTransferWrite, Stage::eClear);
    Transit(Layout::eGeneral, Access::eShaderWrite, Stage::eComputeShader, tile);
    barriers.clear();
    Transit(Layout::eTransferDstOptimal, Access::eTransferWrite, Stage::eCopy);
    ASSERT_EQ(barriers.size(), 4u);
    EXPECT_EQ(barriers[0].srcStageMask, Stage::eClear);
    EXPECT_EQ(barriers[3].srcStageMask, Stage::eComputeShader);
    EXPECT_TRUE(subresources.empty());
    EXPECT_EQ(state.pl_stage, Stage::eCopy);
}

TEST_F(ImageBarriersTest, RepeatedAttachmentsKeepExistingRasterizationOrdering) {
    const auto access = Access::eColorAttachmentRead | Access::eColorAttachmentWrite;
    Transit(Layout::eColorAttachmentOptimal, access, Stage::eColorAttachmentOutput);
    barriers.clear();
    Transit(Layout::eColorAttachmentOptimal, access, Stage::eColorAttachmentOutput);
    EXPECT_TRUE(barriers.empty());
    const auto depth_access =
        Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite;
    Transit(Layout::eDepthStencilAttachmentOptimal, depth_access, Stage::eEarlyFragmentTests);
    barriers.clear();
    Transit(Layout::eDepthStencilAttachmentOptimal, depth_access, Stage::eLateFragmentTests);
    EXPECT_TRUE(barriers.empty());
    EXPECT_EQ(state.pl_stage, Stage::eEarlyFragmentTests | Stage::eLateFragmentTests);
}

TEST_F(ImageBarriersTest, LayoutChangeBetweenReadersStillNeedsTransition) {
    Transit(Layout::eGeneral, Access::eShaderRead, Stage::eComputeShader);
    barriers.clear();
    Transit(Layout::eShaderReadOnlyOptimal, Access::eShaderRead, Stage::eFragmentShader);
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].oldLayout, Layout::eGeneral);
    EXPECT_EQ(barriers[0].newLayout, Layout::eShaderReadOnlyOptimal);
}

} // namespace
