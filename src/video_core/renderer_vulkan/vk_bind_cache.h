// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {
class Buffer;
}

namespace Vulkan {

/// Remember how the previous call of this pipeline resolved each V# and T# slot so unchanged
/// bindings can avoid repeated lookups
/// A read-only V# from an arena can be reused while GPU mappings, arenas and the sync batch stay
/// unchanged
/// That skips IsMapped, ClampRangeSize, GetArena, EnsureResident and adding the same range to the
/// batch again
/// The batch only grows until a flush, so its ranges stay present and CPU writes are still uploaded
/// at the flush
/// A T# can be reused while GPU mappings and the registered images stay unchanged because FindImage
/// only searches that registered set
/// Read the generations before saving a slot so a change during resolution makes that saved result
/// expire
/// The earlier P3 probe found repeated sharps in 83 percent of slots, with 58 percent of calls
/// repeating at least 80 percent of them
/// Keeping the cache per pipeline makes that common case useful without changing the order or
/// validity checks of a full binding
struct BindCache {
    struct Generations {
        u64 sync_flushes;
        u64 image_sets;
        u64 maps;
        u64 arenas;
    };

    struct BufferSlot {
        AmdGpu::Buffer sharp;
        /// Leave this null for bindings that cannot be reused, including written buffers, stream
        /// copies and null descriptors
        const VideoCore::Buffer* arena{};
        u64 offset{};
        u32 size{};
        u64 sync_flushes{};
        u64 maps{};
        u64 arenas{};

        bool Matches(const AmdGpu::Buffer& vsharp, const Generations& now) const {
            return arena && sync_flushes == now.sync_flushes && maps == now.maps &&
                   arenas == now.arenas && std::memcmp(&sharp, &vsharp, sizeof(sharp)) == 0;
        }
    };

    struct ImageSlot {
        using Binding = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
        AmdGpu::Image sharp;
        bool valid{};
        /// Remember that this T# failed validation so a reused result still binds null descriptors
        /// instead of looking up an image
        bool rejected{};
        u64 image_sets{};
        u64 maps{};
        /// Keep the image and adjusted view returned by FindImage before applying any depth
        /// redirect during the actual binding
        boost::container::small_vector<Binding, 1> bindings;

        bool Matches(const AmdGpu::Image& tsharp, const Generations& now) const {
            return valid && image_sets == now.image_sets && maps == now.maps &&
                   std::memcmp(&sharp, &tsharp, sizeof(sharp)) == 0;
        }
    };

    BufferSlot& Buffer(u32 index) {
        if (index >= buffers.size()) {
            buffers.resize(index + 1);
        }
        return buffers[index];
    }

    ImageSlot& Image(u32 index) {
        if (index >= images.size()) {
            images.resize(index + 1);
        }
        return images[index];
    }

    std::vector<BufferSlot> buffers;
    std::vector<ImageSlot> images;
};

} // namespace Vulkan
