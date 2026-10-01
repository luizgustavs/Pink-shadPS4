// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <vector>
#include <boost/container/small_vector.hpp>
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {

struct ImageState {
    vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
    vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

using ImageBarriers = boost::container::small_vector<vk::ImageMemoryBarrier2, 32>;

// Plan image transitions without needing a Vulkan device and keep track of every access that still
// needs synchronization
void GetImageBarriers(vk::Image image, vk::ImageAspectFlags aspect_mask,
                      SubresourceExtent resources, ImageState& last_state,
                      std::vector<ImageState>& subresource_states, ImageBarriers& barriers,
                      vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                      vk::PipelineStageFlags2 dst_stage,
                      std::optional<SubresourceRange> subres_range = {});

} // namespace VideoCore
