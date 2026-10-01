// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "video_core/texture_cache/image_barriers.h"

namespace VideoCore {

void GetImageBarriers(vk::Image image, vk::ImageAspectFlags aspect_mask,
                      SubresourceExtent resources, ImageState& last_state,
                      std::vector<ImageState>& subresource_states, ImageBarriers& barriers,
                      vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                      vk::PipelineStageFlags2 dst_stage,
                      std::optional<SubresourceRange> subres_range) {
    // Keep the attachment ordering we already use here
    // Shader and transfer writes still need a memory barrier when the next command keeps the same
    // layout and access mask
    constexpr vk::AccessFlags2 explicit_writes =
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite |
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eMemoryWrite;

    const auto transition = [&](ImageState& state, vk::ImageSubresourceRange range) {
        if (state.layout == dst_layout && state.access_mask == dst_mask &&
            !(state.access_mask & explicit_writes)) {
            // Wait for every stage that read this subresource before another write or layout change
            // Keeping only the first or last reader would leave some reads running when the write
            // starts
            state.pl_stage |= dst_stage;
            return;
        }
        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = state.pl_stage,
            .srcAccessMask = state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        });
        state = {dst_stage, dst_mask, dst_layout};
    };

    const bool needs_partial_transition =
        subres_range &&
        (subres_range->base != SubresourceBase{} || subres_range->extent != resources);
    if (needs_partial_transition || !subresource_states.empty()) {
        if (subresource_states.empty()) {
            subresource_states.assign(resources.levels * resources.layers, last_state);
        }
        const SubresourceRange range =
            needs_partial_transition ? *subres_range : SubresourceRange{{}, resources};
        vk::PipelineStageFlags2 stages{};
        for (u32 mip = range.base.level; mip < range.base.level + range.extent.levels; ++mip) {
            for (u32 layer = range.base.layer; layer < range.base.layer + range.extent.layers;
                 ++layer) {
                const auto subres_idx = mip * resources.layers + layer;
                ASSERT(subres_idx < subresource_states.size());
                auto& state = subresource_states[subres_idx];
                transition(state, {
                                      .aspectMask = aspect_mask,
                                      .baseMipLevel = mip,
                                      .levelCount = 1,
                                      .baseArrayLayer = layer,
                                      .layerCount = 1,
                                  });
                stages |= state.pl_stage;
            }
        }
        last_state = {stages, dst_mask, dst_layout};
        if (!needs_partial_transition) {
            // All subresources now use the same layout and access mask, so we can track the image
            // as a whole again
            // Keep the combined reader stages so the next write still waits for every earlier read
            subresource_states.clear();
        }
    } else {
        transition(last_state, {
                                   .aspectMask = aspect_mask,
                                   .baseMipLevel = 0,
                                   .levelCount = VK_REMAINING_MIP_LEVELS,
                                   .baseArrayLayer = 0,
                                   .layerCount = VK_REMAINING_ARRAY_LAYERS,
                               });
    }
}

} // namespace VideoCore
