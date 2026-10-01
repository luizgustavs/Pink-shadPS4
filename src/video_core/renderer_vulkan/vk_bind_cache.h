// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <xxhash.h>

#include "common/types.h"
#include "shader_recompiler/resource.h"
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
/// Read generations before saving each slot so changes during resolution make that saved result
/// expire
/// Check the image generation before each T# because an earlier FindImage call may register or
/// remove images
/// The earlier P3 probe found repeated sharps in 83 percent of slots, with 58 percent of calls
/// repeating at least 80 percent of them
/// Checking each texture separately keeps a lookup earlier in the same draw from hiding a
/// registration change that invalidates the next slot
/// Revalidate those dependencies before deciding whether the saved slot can skip its lookup
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

    /// Keep either a rejected T# result or the images and adjusted views returned by FindImage
    /// before the depth redirect
    /// image_sets and maps record the generations before the lookup, while address and size
    /// describe the range searched
    /// Those values let the next binding check whether the saved lookup still applies to the same
    /// resource pages
    struct ImageResolution {
        using Binding = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
        bool rejected{};
        u64 image_sets{};
        u64 maps{};
        VAddr address{};
        u64 size{};
        boost::container::small_vector<Binding, 1> bindings;

        /// With incremental_bind alone, reuse the result only while no images were registered or
        /// removed and no GPU mappings changed
        bool SameGenerations(const Generations& now) const {
            return image_sets == now.image_sets && maps == now.maps;
        }

        /// Rejected T# results stay valid because their checks only use the descriptor and image
        /// resource
        /// For image results, reuse them while their pages have no new registrations or removals,
        /// including removals caused by a GPU unmap
        /// Changes on another page do not affect the registered images FindImage can choose for
        /// this range
        bool PagesUnchanged(const VideoCore::TextureCache& texture_cache) const {
            return rejected || texture_cache.PagesUnchangedSince(address, size, image_sets);
        }
    };

    struct ImageSlot {
        AmdGpu::Image sharp;
        bool valid{};
        ImageResolution resolution;

        bool SameSharp(const AmdGpu::Image& tsharp) const {
            return valid && std::memcmp(&sharp, &tsharp, sizeof(sharp)) == 0;
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

/// Share T# resolutions between pipelines using the descriptor and the image resource fields that
/// affect lookup or binding
/// This also helps when the previous call of a pipeline used another T# or an unrelated page
/// changed
/// Check the pages used by the result instead of invalidating it whenever any image changes
/// elsewhere
/// ImageResolution::PagesUnchanged checks whether the saved page generations still allow us to
/// reuse those images and adjusted views
struct TsharpCache {
    struct Key {
        std::array<u64, 4> sharp;
        u64 resource;

        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept {
            return XXH3_64bits(&key, sizeof(key));
        }
    };
    /// Limit this cache to about 5 MB and clear it only when adding a new key would exceed that
    /// limit
    static constexpr size_t MaxEntries = 16384;

    static Key MakeKey(const AmdGpu::Image& tsharp, const Shader::ImageResource& image_desc,
                       u32 num_bindings) {
        Key key{};
        static_assert(sizeof(AmdGpu::Image) == sizeof(key.sharp));
        std::memcpy(key.sharp.data(), &tsharp, sizeof(key.sharp));
        key.resource = u64(image_desc.is_depth) | u64(image_desc.is_array) << 1 |
                       u64(image_desc.is_written) << 2 | u64(image_desc.mip_fallback_mode) << 8 |
                       u64(image_desc.constant_mip_index) << 24 | u64(num_bindings) << 32;
        return key;
    }

    const BindCache::ImageResolution* Find(const Key& key) const {
        const auto it = entries.find(key);
        return it != entries.end() ? &it->second : nullptr;
    }

    void Store(const Key& key, const BindCache::ImageResolution& resolution) {
        // Replacing an existing key does not use another cache slot, so only a new key needs the
        // capacity check and possible clear
        if (const auto it = entries.find(key); it != entries.end()) {
            it->second = resolution;
            return;
        }
        if (entries.size() >= MaxEntries) {
            entries.clear();
        }
        entries.emplace(key, resolution);
    }

    std::unordered_map<Key, BindCache::ImageResolution, KeyHash> entries;
};

} // namespace Vulkan
