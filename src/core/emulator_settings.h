// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream> // Windows static guest red-zone protection
#include <sstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/types.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection

// D5: a plain pointer read; GetInstance() locks a mutex and copies a shared_ptr on every access, which cost
// ~3 ms per SotC frame on the command processor thread alone
#define EmulatorSettings (EmulatorSettingsImpl::Current())

enum HideCursorState : int {
    Never,
    Idle,
    Always,
};

enum UsbBackendType : int {
    Real,
    SkylandersPortal,
    InfinityBase,
    DimensionsToypad,
};

enum GpuReadbacksMode : int {
    Disabled,
    Relaxed,
    Precise,
};

// Windows static guest red-zone protection
NLOHMANN_JSON_SERIALIZE_ENUM(WindowsGuestRedZoneProtectionMode,
                             {{WindowsGuestRedZoneProtectionMode::Disabled, "Disabled"},
                              {WindowsGuestRedZoneProtectionMode::StaticPatching,
                               "StaticPatching"}})

inline std::ostream& operator<<(std::ostream& output, WindowsGuestRedZoneProtectionMode mode) {
    return output << nlohmann::json(mode).get<std::string>();
}

enum class ConfigMode {
    Default,
    Global,
    Clean,
};

enum AudioBackend : int {
    SDL,
    OpenAL,
    // Add more backends as needed
};

enum OpenALHrtfMode : int {
    HrtfAuto, // Let OpenAL Soft decide (on for headphone-like stereo outputs)
    HrtfOn,   // Force HRTF binaural rendering
    HrtfOff,  // Never use HRTF
};

enum OpenALOutputMode : int {
    OutputAuto,       // Let OpenAL Soft negotiate with the device
    OutputStereo,     // Force stereo output
    OutputQuad,       // Force quadraphonic output
    OutputSurround51, // Force 5.1 surround output
    OutputSurround71, // Force 7.1 surround output
};

template <typename T>
struct Setting {
    T default_value{};
    T value{};
    std::optional<T> game_specific_value{};

    Setting() = default;
    // Single-argument ctor: initialises both default_value and value so
    // that CleanMode can always recover the intended factory default.
    /*implicit*/ Setting(T init) : default_value(std::move(init)), value(default_value) {}

    /// Return the active value under the given mode.
    T get(ConfigMode mode = ConfigMode::Default) const {
        switch (mode) {
        case ConfigMode::Default:
            return game_specific_value.value_or(value);
        case ConfigMode::Global:
            return value;
        case ConfigMode::Clean:
            return default_value;
        }
        return value;
    }

    /// Write v to the base layer.
    /// Set proper value as base or game_specific
    void set(const T& v, bool game_specific = false) {
        if (game_specific) {
            game_specific_value = v;
        } else {
            value = v;
        }
    }

    /// Discard the game-specific override; subsequent get(Default) will
    /// fall back to the base value.
    void reset_game_specific() {
        game_specific_value = std::nullopt;
    }
};

template <typename T>
void to_json(nlohmann::json& j, const Setting<T>& s) {
    j = s.value;
}

template <typename T>
void from_json(const nlohmann::json& j, Setting<T>& s) {
    s.value = j.get<T>();
}

struct OverrideItem {
    const char* key;
    std::function<void(void* group_ptr, const nlohmann::json& entry,
                       std::vector<std::string>& changed)>
        apply;
    /// Return the value that should be written to the per-game config file.
    /// Falls back to base value if no game-specific override is set.
    std::function<nlohmann::json(const void* group_ptr)> get_for_save;

    /// Clear game_specific_value for this field.
    std::function<void(void* group_ptr)> reset_game_specific;
};

template <typename Struct, typename T>
inline OverrideItem make_override(const char* key, Setting<T> Struct::* member) {
    return OverrideItem{
        key,
        [member, key](void* base, const nlohmann::json& entry, std::vector<std::string>& changed) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            Setting<T>& dst = obj->*member;
            try {
                T newValue = entry.get<T>();
                if (dst.value != newValue) {
                    std::ostringstream oss;
                    oss << key << " ( " << dst.value << " -> " << newValue << " )";
                    changed.push_back(oss.str());
                }
                dst.game_specific_value = newValue;
            } catch (const std::exception& e) {
                LOG_ERROR(Config, "[make_override] error parsing {}: {}", key, e.what());
                LOG_ERROR(Config, "[make_override] Entry was: {}", entry.dump());
                LOG_ERROR(Config, "[make_override] Type name: {}", entry.type_name());
            }
        },

        // --- get_for_save -------------------------------------------
        // Returns game_specific_value when present, otherwise base value.
        // This means a freshly-opened game-specific dialog still shows
        // useful (current-global) values rather than empty entries.
        [member](const void* base) -> nlohmann::json {
            const Struct* obj = reinterpret_cast<const Struct*>(base);
            const Setting<T>& src = obj->*member;
            return nlohmann::json(src.game_specific_value.value_or(src.value));
        },

        // --- reset_game_specific ------------------------------------
        [member](void* base) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            (obj->*member).reset_game_specific();
        }};
}

// -------------------------------
// Support types
// -------------------------------
struct GameInstallDir {
    std::filesystem::path path;
    bool enabled;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GameInstallDir, path, enabled)

// -------------------------------
// General settings
// -------------------------------
struct GeneralSettings {
    Setting<std::vector<GameInstallDir>> install_dirs;
    Setting<std::filesystem::path> addon_install_dir;
    Setting<std::filesystem::path> home_dir;
    Setting<std::filesystem::path> sys_modules_dir;
    Setting<std::filesystem::path> font_dir;

    Setting<int> volume_slider{100};
    Setting<bool> neo_mode{false};
    Setting<bool> dev_kit_mode{false};
    Setting<int> extra_dmem_in_mbytes{0};
    Setting<int> extra_fmem_in_mbytes{0};
    Setting<bool> shad_net_enabled{false};
    Setting<bool> trophy_popup_disabled{false};
    Setting<double> trophy_notification_duration{6.0};
    Setting<std::string> trophy_notification_side{"right"};
    Setting<bool> show_splash{false};
    Setting<bool> connected_to_network{false};
    Setting<bool> discord_rpc_enabled{false};
    Setting<bool> show_fps_counter{false};
    Setting<int> console_language{1};
    Setting<int> big_picture_scale{1000};
    Setting<std::string> shadnet_server{"srv.shadps4.net:31313"};
    Setting<std::string> shadnet_webapi_server{"http://srv.shadps4.net:31315"};
    Setting<std::string> signaling_info{};
    Setting<bool> enable_upnp{true};
    // Redirect /app0/logs to writable per-game storage
    Setting<bool> redirect_app0_logs{false};
    // Process affinity mask, or 0 to leave it unchanged
    Setting<u64> cpu_affinity_mask{0};
    // Poll the keyboard slot and connected pads from one 125 Hz timer
    Setting<bool> poll_connected_pads_only{false};
    // Enable Pro mode with 3840x2160 rendering and output for Shadow of the Colossus
    // Reserve at least 1536 MB of extra memory and patch every dynamic
    // resolution level to 2160p
    Setting<bool> sotc_pro_4k{false};

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GeneralSettings>("volume_slider", &GeneralSettings::volume_slider),
            make_override<GeneralSettings>("neo_mode", &GeneralSettings::neo_mode),
            make_override<GeneralSettings>("dev_kit_mode", &GeneralSettings::dev_kit_mode),
            make_override<GeneralSettings>("extra_dmem_in_mbytes",
                                           &GeneralSettings::extra_dmem_in_mbytes),
            make_override<GeneralSettings>("extra_fmem_in_mbytes",
                                           &GeneralSettings::extra_fmem_in_mbytes),
            make_override<GeneralSettings>("shad_net_enabled", &GeneralSettings::shad_net_enabled),
            make_override<GeneralSettings>("trophy_popup_disabled",
                                           &GeneralSettings::trophy_popup_disabled),
            make_override<GeneralSettings>("trophy_notification_duration",
                                           &GeneralSettings::trophy_notification_duration),
            make_override<GeneralSettings>("show_splash", &GeneralSettings::show_splash),
            make_override<GeneralSettings>("trophy_notification_side",
                                           &GeneralSettings::trophy_notification_side),
            make_override<GeneralSettings>("connected_to_network",
                                           &GeneralSettings::connected_to_network),
            make_override<GeneralSettings>("console_language", &GeneralSettings::console_language),
            make_override<GeneralSettings>("shadnet_server", &GeneralSettings::shadnet_server),
            make_override<GeneralSettings>("shadnet_webapi_server",
                                           &GeneralSettings::shadnet_webapi_server),
            make_override<GeneralSettings>("signaling_info", &GeneralSettings::signaling_info),
            make_override<GeneralSettings>("enable_upnp", &GeneralSettings::enable_upnp),
            make_override<GeneralSettings>("redirect_app0_logs",
                                           &GeneralSettings::redirect_app0_logs),
            make_override<GeneralSettings>("cpu_affinity_mask",
                                           &GeneralSettings::cpu_affinity_mask),
            make_override<GeneralSettings>("poll_connected_pads_only",
                                           &GeneralSettings::poll_connected_pads_only),
            make_override<GeneralSettings>("sotc_pro_4k", &GeneralSettings::sotc_pro_4k)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneralSettings, install_dirs, addon_install_dir, home_dir,
                                   sys_modules_dir, font_dir, volume_slider, neo_mode, dev_kit_mode,
                                   extra_dmem_in_mbytes, extra_fmem_in_mbytes, shad_net_enabled,
                                   trophy_popup_disabled, trophy_notification_duration, show_splash,
                                   trophy_notification_side, connected_to_network,
                                   discord_rpc_enabled, show_fps_counter, console_language,
                                   big_picture_scale, shadnet_server, shadnet_webapi_server,
                                   signaling_info, enable_upnp, redirect_app0_logs,
                                   cpu_affinity_mask, poll_connected_pads_only, sotc_pro_4k)

// -------------------------------
// Log settings
// -------------------------------
struct LogSettings {
    Setting<bool> append{false}; // specific
    Setting<bool> enable{true};  // specific
    Setting<std::string> filter{""};
    Setting<std::string> flush_level{""};
    Setting<u32> max_skip_duration{5'000};
    Setting<bool> separate{false}; // specific
    Setting<unsigned long long> size_limit{100_MB};
    Setting<bool> skip_duplicate{true};
    Setting<bool> sync{true};
    // Default to "events" for crash, shader and render problems, Critical messages and exit summaries
    // Keep all messages in the log file and use "all" to mirror them in the console
    // With "all" and a synced log, each warning also writes to the launcher's pipe on its logging thread
    Setting<std::string> console_mode{"events"};
#ifdef _WIN32
    Setting<std::string> type{"wincolor"};
#endif

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<LogSettings>("append", &LogSettings::append),
            make_override<LogSettings>("enable", &LogSettings::enable),
            make_override<LogSettings>("filter", &LogSettings::filter),
            make_override<LogSettings>("flush_level", &LogSettings::flush_level),
            make_override<LogSettings>("max_skip_duration", &LogSettings::max_skip_duration),
            make_override<LogSettings>("separate", &LogSettings::separate),
            make_override<LogSettings>("size_limit", &LogSettings::size_limit),
            make_override<LogSettings>("skip_duplicate", &LogSettings::skip_duplicate),
            make_override<LogSettings>("sync", &LogSettings::sync),
            make_override<LogSettings>("console_mode", &LogSettings::console_mode),
#ifdef _WIN32
            make_override<LogSettings>("type", &LogSettings::type),
#endif
        };
    }
};
#ifdef _WIN32
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync,
                                   console_mode, type)
#else
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync,
                                   console_mode)
#endif

// -------------------------------
// Debug settings
// -------------------------------
struct DebugSettings {
    Setting<bool> debug_dump{false};         // specific
    Setting<bool> shader_collect{false};     // specific
    Setting<std::string> config_version{""}; // specific

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<DebugSettings>("debug_dump", &DebugSettings::debug_dump),
            make_override<DebugSettings>("shader_collect", &DebugSettings::shader_collect)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugSettings, debug_dump, shader_collect, config_version)

// -------------------------------
// Input settings
// -------------------------------

struct InputSettings {
    Setting<int> cursor_state{HideCursorState::Idle};      // specific
    Setting<int> cursor_hide_timeout{5};                   // specific
    Setting<int> usb_device_backend{UsbBackendType::Real}; // specific
    Setting<bool> use_special_pad{false};
    Setting<int> special_pad_class{1};
    Setting<bool> motion_controls_enabled{true}; // specific
    Setting<bool> use_unified_input_config{true};
    Setting<std::string> default_controller_id{""};
    Setting<bool> background_controller_input{false}; // specific
    Setting<bool> ime_accessibility_enabled{false};   // specific
    Setting<bool> ime_url_mail_short_panel{false};    // specific
    Setting<bool> is_circle_enter{false};             // specific
    Setting<s32> camera_id{-1};
    Setting<bool> use_mice_as_mice{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<InputSettings>("cursor_state", &InputSettings::cursor_state),
            make_override<InputSettings>("cursor_hide_timeout",
                                         &InputSettings::cursor_hide_timeout),
            make_override<InputSettings>("usb_device_backend", &InputSettings::usb_device_backend),
            make_override<InputSettings>("motion_controls_enabled",
                                         &InputSettings::motion_controls_enabled),
            make_override<InputSettings>("background_controller_input",
                                         &InputSettings::background_controller_input),
            make_override<InputSettings>("ime_accessibility_enabled",
                                         &InputSettings::ime_accessibility_enabled),
            make_override<InputSettings>("ime_url_mail_short_panel",
                                         &InputSettings::ime_url_mail_short_panel),
            make_override<InputSettings>("is_circle_enter", &InputSettings::is_circle_enter),
            make_override<InputSettings>("camera_id", &InputSettings::camera_id),
            make_override<InputSettings>("use_mice_as_mice", &InputSettings::use_mice_as_mice)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputSettings, cursor_state, cursor_hide_timeout,
                                   usb_device_backend, use_special_pad, special_pad_class,
                                   motion_controls_enabled, use_unified_input_config,
                                   default_controller_id, background_controller_input,
                                   ime_accessibility_enabled, ime_url_mail_short_panel, camera_id,
                                   is_circle_enter, use_mice_as_mice)
// -------------------------------
// Audio settings
// -------------------------------
struct AudioSettings {
    Setting<u32> audio_backend{AudioBackend::SDL};
    Setting<std::string> sdl_mic_device{"Default Device"};
    Setting<std::string> sdl_main_output_device{"Default Device"};
    Setting<std::string> sdl_padSpk_output_device{"Default Device"};
    Setting<std::string> openal_mic_device{"Default Device"};
    Setting<std::string> openal_main_output_device{"Default Device"};
    Setting<std::string> openal_padSpk_output_device{"Default Device"};
    Setting<u32> openal_hrtf{OpenALHrtfMode::HrtfAuto};
    Setting<u32> openal_output_mode{OpenALOutputMode::OutputAuto};
    // Match audio speed and pitch to the measured guest frame rate
    Setting<bool> audio_follow_game_speed{false};
    // Lowest playback speed in percent, clamped to 5-100
    Setting<u32> audio_min_game_speed{10};
    // Target frame rate, or 0 to use sceVideoOutSetFlipRate
    Setting<u32> audio_game_target_fps{0};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<AudioSettings>("audio_backend", &AudioSettings::audio_backend),
            make_override<AudioSettings>("sdl_mic_device", &AudioSettings::sdl_mic_device),
            make_override<AudioSettings>("sdl_main_output_device",
                                         &AudioSettings::sdl_main_output_device),
            make_override<AudioSettings>("sdl_padSpk_output_device",
                                         &AudioSettings::sdl_padSpk_output_device),
            make_override<AudioSettings>("openal_mic_device", &AudioSettings::openal_mic_device),
            make_override<AudioSettings>("openal_main_output_device",
                                         &AudioSettings::openal_main_output_device),
            make_override<AudioSettings>("openal_padSpk_output_device",
                                         &AudioSettings::openal_padSpk_output_device),
            make_override<AudioSettings>("openal_hrtf", &AudioSettings::openal_hrtf),
            make_override<AudioSettings>("openal_output_mode", &AudioSettings::openal_output_mode),
            make_override<AudioSettings>("audio_follow_game_speed",
                                         &AudioSettings::audio_follow_game_speed),
            make_override<AudioSettings>("audio_min_game_speed",
                                         &AudioSettings::audio_min_game_speed),
            make_override<AudioSettings>("audio_game_target_fps",
                                         &AudioSettings::audio_game_target_fps)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AudioSettings, audio_backend, sdl_mic_device,
                                   sdl_main_output_device, sdl_padSpk_output_device,
                                   openal_mic_device, openal_main_output_device,
                                   openal_padSpk_output_device, openal_hrtf, openal_output_mode,
                                   audio_follow_game_speed, audio_min_game_speed,
                                   audio_game_target_fps)

// Windows static guest red-zone protection
struct WindowsGuestRedZoneProtectionSettings {
    Setting<WindowsGuestRedZoneProtectionMode> windows_guest_red_zone_protection_mode{
        WindowsGuestRedZoneProtectionMode::Disabled};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{make_override<WindowsGuestRedZoneProtectionSettings>(
            "windows_guest_red_zone_protection_mode",
            &WindowsGuestRedZoneProtectionSettings::windows_guest_red_zone_protection_mode)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WindowsGuestRedZoneProtectionSettings,
                                   windows_guest_red_zone_protection_mode)

// -------------------------------
// GPU settings
// -------------------------------
struct GPUSettings {
    Setting<u32> window_width{1280};
    Setting<u32> window_height{720};
    Setting<u32> internal_screen_width{1280};
    Setting<u32> internal_screen_height{720};
    Setting<bool> null_gpu{false};
    Setting<bool> copy_gpu_buffers{false};
    Setting<u32> readbacks_mode{GpuReadbacksMode::Disabled};
    Setting<bool> readback_linear_images_enabled{false};
    Setting<bool> direct_memory_access_enabled{false};
    Setting<bool> dump_shaders{false};
    Setting<bool> patch_shaders{false};
    Setting<u32> vblank_frequency{60};
    Setting<bool> full_screen{false};
    Setting<std::string> full_screen_mode{"Windowed"};
    Setting<std::string> present_mode{"Mailbox"};
    Setting<bool> hdr_allowed{false};
    Setting<bool> fsr_enabled{false};
    Setting<bool> rcas_enabled{true};
    Setting<int> rcas_attenuation{250};
    // Run DLSS Neural Rendering over the final frame without motion vectors or depth
    // Require Windows, an NVIDIA RTX GPU and nvngx_dlssnr.dll next to shadps4.exe
    // Leave it off by default and use hotkey_toggle_dlss_nr to switch it in game
    Setting<bool> dlss_nr_enabled{false};
    // Select the model style: 0 Default, 1 Natural, 2 Cinematic
    Setting<u32> dlss_nr_style{0};
    // Set detail strength from 0 to 2
    Setting<double> dlss_nr_intensity{1.0};
    // Preserve the game's HUD
    Setting<bool> dlss_nr_ui_correction{true};
    Setting<bool> userfaultfd{false};
    // Per-game loop limit (0 = off): stop runaway compute shaders after this many back edges
    // Shadow of the Colossus uses 1048576 to avoid a GPU timeout
    Setting<u32> compute_loop_cap{0};
    // On Windows, restore guest and GPU tracking protections after splitting a mapped placeholder
    Setting<bool> preserve_split_protection{false};
    // Keep guest thread and fiber stacks writable and CPU-owned; a protected stack page can crash the host
    Setting<bool> cpu_authoritative_stacks{false};
    // SotC 01.01 BPE heap address (0 = off); protect its allocator metadata from GPU readbacks
    Setting<u64> bpe_heap_guard_address{0};
    // Treat constant-lane ReadLane as uniform so guarded LDS barriers stay in place
    Setting<bool> lds_barrier_uniform_readlane{false};
    // Add EarlyFragmentTests to storage-writing pixel shaders when the guest runs depth/stencil first
    Setting<bool> early_fragment_tests_from_z_order{false};
    // Report bound T# counter banks as sampled at LOD 0 so games can request finer mips
    Setting<bool> lod_stats_from_bindings{false};
    // Turn dynamically indexed compute T# reads into descriptor arrays of this size (0 = off, max 64)
    Setting<u32> dynamic_tsharp_array_size{0};
    // Treat branches on LDS-exchanged values as uniform when lowering ReadLane and Ballot
    Setting<bool> wave64_uniform_branches{false};
    // Use GCN reduction identities for lanes missing from small host workgroups
    Setting<bool> wave64_missing_lane_identity{true};
    // Add LDS barriers after divergent branches in compute groups larger than one GCN wave so
    // narrower host subgroups see completed shared writes
    Setting<bool> lds_barriers_large_groups{true};
    // Start instance ID VGPRs at zero on every draw
    // Shaders that add the start instance themselves would otherwise count it twice
    Setting<bool> relative_instance_id{false};
    // Match GCN NaN handling for negated compares, min/max and clamp
    // SotC uses NaN to mark occluded fog froxels
    Setting<bool> gcn_nan_semantics{false};
    // Use the legacy multiply rule for MAC and MAD: zero times Inf or NaN is zero
    Setting<bool> legacy_mad_semantics{false};
    // Record GPU commands and submits to identify the first unfinished submit after device loss
    // Add NVIDIA checkpoints when available; recording adds a small cost per command
    Setting<bool> gpu_checkpoints{false};
    // Read clean SRT data from guest memory without draining the GPU
    Setting<bool> srt_walker_clean_reads{false};
    // Compare clean shader code through guest backing memory
    Setting<bool> shader_code_clean_reads{false};
    // Submit safe readbacks ahead of the command buffer being recorded
    Setting<bool> readback_ahead{false};
    // Submit after this many guest commands without waiting, or keep it off with 0
    Setting<u32> periodic_flush_commands{0};
    // Cache mapped runs used by IsMapped while dma_sync_once_per_batch handles skipping repeated
    // DMA range sweeps separately
    Setting<bool> cp_recording_cuts{false};
    // Detile host textures from VRAM and skip duplicate pipeline binds
    Setting<bool> gpu_overhead_cuts{false};
    // Leave registered guest stacks out of bulk DMA sweeps
    Setting<bool> dma_sweep_skip_stacks{false};
    // Poll before blocking GPU waits for this many microseconds, or keep it off with 0
    Setting<u32> wait_spin_us{0};
    // Run ahead readbacks on a transfer-only queue when one is available
    Setting<bool> readback_ahead_transfer_queue{false};
    // Answer IsMapped from a table of mapped 16 KB pages so common buffer checks can avoid taking
    // the shared lock
    Setting<bool> mapped_page_table{false};
    // Add resident DMA ranges once per sync batch because later DMA syncs in the same batch would
    // add the same ranges again
    Setting<bool> dma_sync_once_per_batch{false};
    // Reuse unchanged read-only V# and T# bindings from the previous call of this pipeline while
    // the resources they depend on stay valid
    Setting<bool> incremental_bind{false};
    // Reuse the last T# resolution from any pipeline while the pages it uses still contain the same
    // registered images
    Setting<bool> tsharp_cache{false};
    // Check a GPU-written tick in host memory before querying the driver in mode 1, and also allow
    // readbacks to finish on that marker in mode 2
    Setting<u32> wait_marker{0};
    // Sub-allocate images up to 16 MB from pools with 32 MB blocks
    // to avoid individual device
    // memory allocations for small images
    Setting<bool> image_memory_pool{false};
    // Delay gfx EOP labels by this many microseconds as a temporary timing workaround, with 0
    // disabling it and no GPU completion guarantee
    Setting<u32> eop_label_delay_us{0};
    // Move Vulkan command recording and submissions from the command processor to a separate
    // recording thread when this setting is enabled
    Setting<bool> cp_record_thread{false};
    // Minimum anisotropy for linear mipmapped samplers: 0, 2, 4, 8 or 16
    // Keep stronger guest ratios, point filtering and depth compares unchanged
    // Cap the result at the device limit
    Setting<u32> force_anisotropy{0};
    // Download other recent GPU-written windows during the same drain
    // Their next CPU read can use the data without another GPU wait
    Setting<bool> readback_hot_regions{false};
    // Share layout and access between a sampled read-only depth
    // target and its attachment
    // This avoids barriers that would end the render pass between draws
    Setting<bool> depth_target_sampled_layout{false};
    // Skip memory barriers for same-ring dispatches with no guest sync between them
    // Require every tracked access to come from guest dispatches without DMA
    Setting<bool> gcn_unordered_dispatches{false};
    // Track buffer accesses in 256 B granules with a summary bit per 64 KiB
    // Confirm bitmap hits against exact ranges to keep barrier decisions unchanged
    Setting<bool> access_bitmap_tracking{false};
    // Use one-byte locks for tracked 4 KiB pages, spinning briefly before yielding
    // Read only at boot because changing lock tables during use is unsafe
    Setting<bool> page_spin_locks{false};
    // Skip repeated binding, sync and residency work when the exact answer is known
    // Merge DMA ranges in one pass and use a per-thread stack copy
    Setting<bool> range_fast_paths{false};
    // Cap read-only V# descriptors with num_records 0xffffffff, in MiB
    // Use 0 to disable the cap
    // This avoids keeping gigabytes resident after a bind made during
    // the initial large mapping
    Setting<u32> unbounded_vsharp_cap_mb{0};
    // Count only freed images against the GC budget so it reaches idle CPU textures
    // Below memory pressure, wait 300 unused ticks instead of 16 to avoid
    // reloading active textures
    Setting<bool> texture_gc_full_scan{false};
    // Device memory use (MB, as the driver reports it, sparse buffer arena
    // included) above which the image GC starts freeing idle images; 0 = automatic
    // (half of the budget beyond 8 GB)
    Setting<u32> texture_gc_trigger_mb{0};
    // Give sampled textures above this KiB threshold their own allocation
    // Use 0 to disable the threshold and require image_memory_pool
    // Keep render targets, storage images and smaller textures in the pool
    Setting<u32> image_pool_texture_max_kb{0};
    // Release idle 2 MiB arena pieces after this many seconds,
    // with 0 disabling eviction
    // Keep GPU-written data and allocations made by DMA faults
    // Upload the bytes again when an evicted range is reused
    Setting<u32> arena_evict_idle_s{0};
    Setting<bool> inline_fetch_shader{false};
    // TODO add overrides
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GPUSettings>("null_gpu", &GPUSettings::null_gpu),
            make_override<GPUSettings>("copy_gpu_buffers", &GPUSettings::copy_gpu_buffers),
            make_override<GPUSettings>("full_screen", &GPUSettings::full_screen),
            make_override<GPUSettings>("full_screen_mode", &GPUSettings::full_screen_mode),
            make_override<GPUSettings>("present_mode", &GPUSettings::present_mode),
            make_override<GPUSettings>("window_height", &GPUSettings::window_height),
            make_override<GPUSettings>("window_width", &GPUSettings::window_width),
            make_override<GPUSettings>("hdr_allowed", &GPUSettings::hdr_allowed),
            make_override<GPUSettings>("fsr_enabled", &GPUSettings::fsr_enabled),
            make_override<GPUSettings>("rcas_enabled", &GPUSettings::rcas_enabled),
            make_override<GPUSettings>("rcas_attenuation", &GPUSettings::rcas_attenuation),
            make_override<GPUSettings>("dlss_nr_enabled", &GPUSettings::dlss_nr_enabled),
            make_override<GPUSettings>("dlss_nr_style", &GPUSettings::dlss_nr_style),
            make_override<GPUSettings>("dlss_nr_intensity", &GPUSettings::dlss_nr_intensity),
            make_override<GPUSettings>("dlss_nr_ui_correction",
                                       &GPUSettings::dlss_nr_ui_correction),
            make_override<GPUSettings>("dump_shaders", &GPUSettings::dump_shaders),
            make_override<GPUSettings>("patch_shaders", &GPUSettings::patch_shaders),
            make_override<GPUSettings>("readbacks_mode", &GPUSettings::readbacks_mode),
            make_override<GPUSettings>("readback_linear_images_enabled",
                                       &GPUSettings::readback_linear_images_enabled),
            make_override<GPUSettings>("direct_memory_access_enabled",
                                       &GPUSettings::direct_memory_access_enabled),
            make_override<GPUSettings>("vblank_frequency", &GPUSettings::vblank_frequency),
            make_override<GPUSettings>("compute_loop_cap", &GPUSettings::compute_loop_cap),
            make_override<GPUSettings>("preserve_split_protection",
                                       &GPUSettings::preserve_split_protection),
            make_override<GPUSettings>("cpu_authoritative_stacks",
                                       &GPUSettings::cpu_authoritative_stacks),
            make_override<GPUSettings>("bpe_heap_guard_address",
                                       &GPUSettings::bpe_heap_guard_address),
            make_override<GPUSettings>("lds_barrier_uniform_readlane",
                                       &GPUSettings::lds_barrier_uniform_readlane),
            make_override<GPUSettings>("early_fragment_tests_from_z_order",
                                       &GPUSettings::early_fragment_tests_from_z_order),
            make_override<GPUSettings>("lod_stats_from_bindings",
                                       &GPUSettings::lod_stats_from_bindings),
            make_override<GPUSettings>("dynamic_tsharp_array_size",
                                       &GPUSettings::dynamic_tsharp_array_size),
            make_override<GPUSettings>("wave64_uniform_branches",
                                       &GPUSettings::wave64_uniform_branches),
            make_override<GPUSettings>("wave64_missing_lane_identity",
                                       &GPUSettings::wave64_missing_lane_identity),
            make_override<GPUSettings>("lds_barriers_large_groups",
                                       &GPUSettings::lds_barriers_large_groups),
            make_override<GPUSettings>("relative_instance_id",
                                       &GPUSettings::relative_instance_id),
            make_override<GPUSettings>("gcn_nan_semantics", &GPUSettings::gcn_nan_semantics),
            make_override<GPUSettings>("legacy_mad_semantics",
                                       &GPUSettings::legacy_mad_semantics),
            make_override<GPUSettings>("gpu_checkpoints", &GPUSettings::gpu_checkpoints),
            make_override<GPUSettings>("srt_walker_clean_reads",
                                       &GPUSettings::srt_walker_clean_reads),
            make_override<GPUSettings>("shader_code_clean_reads",
                                       &GPUSettings::shader_code_clean_reads),
            make_override<GPUSettings>("readback_ahead", &GPUSettings::readback_ahead),
            make_override<GPUSettings>("periodic_flush_commands",
                                       &GPUSettings::periodic_flush_commands),
            make_override<GPUSettings>("cp_recording_cuts", &GPUSettings::cp_recording_cuts),
            make_override<GPUSettings>("gpu_overhead_cuts", &GPUSettings::gpu_overhead_cuts),
            make_override<GPUSettings>("dma_sweep_skip_stacks",
                                       &GPUSettings::dma_sweep_skip_stacks),
            make_override<GPUSettings>("wait_spin_us", &GPUSettings::wait_spin_us),
            make_override<GPUSettings>("readback_ahead_transfer_queue",
                                       &GPUSettings::readback_ahead_transfer_queue),
            make_override<GPUSettings>("mapped_page_table", &GPUSettings::mapped_page_table),
            make_override<GPUSettings>("dma_sync_once_per_batch",
                                       &GPUSettings::dma_sync_once_per_batch),
            make_override<GPUSettings>("incremental_bind", &GPUSettings::incremental_bind),
            make_override<GPUSettings>("tsharp_cache", &GPUSettings::tsharp_cache),
            make_override<GPUSettings>("wait_marker", &GPUSettings::wait_marker),
            make_override<GPUSettings>("image_memory_pool", &GPUSettings::image_memory_pool),
            make_override<GPUSettings>("eop_label_delay_us", &GPUSettings::eop_label_delay_us),
            make_override<GPUSettings>("cp_record_thread", &GPUSettings::cp_record_thread),
            make_override<GPUSettings>("force_anisotropy", &GPUSettings::force_anisotropy),
            make_override<GPUSettings>("readback_hot_regions",
                                       &GPUSettings::readback_hot_regions),
            make_override<GPUSettings>("depth_target_sampled_layout",
                                       &GPUSettings::depth_target_sampled_layout),
            make_override<GPUSettings>("gcn_unordered_dispatches",
                                       &GPUSettings::gcn_unordered_dispatches),
            make_override<GPUSettings>("access_bitmap_tracking",
                                       &GPUSettings::access_bitmap_tracking),
            make_override<GPUSettings>("page_spin_locks", &GPUSettings::page_spin_locks),
            make_override<GPUSettings>("range_fast_paths", &GPUSettings::range_fast_paths),
            make_override<GPUSettings>("unbounded_vsharp_cap_mb",
                                       &GPUSettings::unbounded_vsharp_cap_mb),
            make_override<GPUSettings>("texture_gc_full_scan",
                                       &GPUSettings::texture_gc_full_scan),
            make_override<GPUSettings>("texture_gc_trigger_mb",
                                       &GPUSettings::texture_gc_trigger_mb),
            make_override<GPUSettings>("image_pool_texture_max_kb",
                                       &GPUSettings::image_pool_texture_max_kb),
            make_override<GPUSettings>("arena_evict_idle_s", &GPUSettings::arena_evict_idle_s),
            make_override<GPUSettings>("inline_fetch_shader", &GPUSettings::inline_fetch_shader),
        };
    }
};
// Split the GPU JSON members into two lists to stay below the 64-member macro limit
#define SHADPS4_GPU_SETTINGS_JSON(op)                                                               \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(                                                       \
        op, window_width, window_height, internal_screen_width, internal_screen_height, null_gpu,   \
        copy_gpu_buffers, readbacks_mode, readback_linear_images_enabled,                           \
        direct_memory_access_enabled, dump_shaders, patch_shaders, vblank_frequency, full_screen,   \
        full_screen_mode, present_mode, hdr_allowed, fsr_enabled, rcas_enabled, rcas_attenuation,   \
        compute_loop_cap, preserve_split_protection, cpu_authoritative_stacks,                      \
        bpe_heap_guard_address, lds_barrier_uniform_readlane, early_fragment_tests_from_z_order,    \
        lod_stats_from_bindings, dynamic_tsharp_array_size, wave64_uniform_branches,                \
        wave64_missing_lane_identity, lds_barriers_large_groups, relative_instance_id))             \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(                                                       \
        op, gcn_nan_semantics, legacy_mad_semantics, gpu_checkpoints, srt_walker_clean_reads,       \
        shader_code_clean_reads, readback_ahead, periodic_flush_commands, cp_recording_cuts,        \
        gpu_overhead_cuts, dma_sweep_skip_stacks, wait_spin_us, readback_ahead_transfer_queue,      \
        mapped_page_table, dma_sync_once_per_batch, incremental_bind, tsharp_cache, wait_marker,    \
        image_memory_pool, eop_label_delay_us, cp_record_thread, force_anisotropy,                  \
        readback_hot_regions, depth_target_sampled_layout, gcn_unordered_dispatches,                \
        access_bitmap_tracking, page_spin_locks, range_fast_paths, unbounded_vsharp_cap_mb,         \
        texture_gc_full_scan, texture_gc_trigger_mb, image_pool_texture_max_kb,                     \
        arena_evict_idle_s, dlss_nr_enabled, dlss_nr_style, dlss_nr_intensity,                      \
        dlss_nr_ui_correction))
template <typename BasicJsonType,
          nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void to_json(BasicJsonType& nlohmann_json_j, const GPUSettings& nlohmann_json_t) {
    SHADPS4_GPU_SETTINGS_JSON(NLOHMANN_JSON_TO)
}
template <typename BasicJsonType,
          nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void from_json(const BasicJsonType& nlohmann_json_j, GPUSettings& nlohmann_json_t) {
    SHADPS4_GPU_SETTINGS_JSON(NLOHMANN_JSON_FROM)
}
#undef SHADPS4_GPU_SETTINGS_JSON
// -------------------------------
// Vulkan settings
// -------------------------------
struct VulkanSettings {
    Setting<s32> gpu_id{-1};
    Setting<bool> renderdoc_enabled{false};
    Setting<bool> vkvalidation_enabled{false};
    Setting<bool> vkvalidation_core_enabled{true};
    Setting<bool> vkvalidation_sync_enabled{false};
    Setting<bool> vkvalidation_gpu_enabled{false};
    Setting<bool> vkcrash_diagnostic_enabled{false};
    Setting<bool> vkhost_markers{false};
    Setting<bool> vkguest_markers{false};
    Setting<bool> pipeline_cache_enabled{false};
    Setting<bool> pipeline_cache_archived{false};
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<VulkanSettings>("gpu_id", &VulkanSettings::gpu_id),
            make_override<VulkanSettings>("renderdoc_enabled", &VulkanSettings::renderdoc_enabled),
            make_override<VulkanSettings>("vkvalidation_enabled",
                                          &VulkanSettings::vkvalidation_enabled),
            make_override<VulkanSettings>("vkvalidation_core_enabled",
                                          &VulkanSettings::vkvalidation_core_enabled),
            make_override<VulkanSettings>("vkvalidation_sync_enabled",
                                          &VulkanSettings::vkvalidation_sync_enabled),
            make_override<VulkanSettings>("vkvalidation_gpu_enabled",
                                          &VulkanSettings::vkvalidation_gpu_enabled),
            make_override<VulkanSettings>("vkcrash_diagnostic_enabled",
                                          &VulkanSettings::vkcrash_diagnostic_enabled),
            make_override<VulkanSettings>("vkhost_markers", &VulkanSettings::vkhost_markers),
            make_override<VulkanSettings>("vkguest_markers", &VulkanSettings::vkguest_markers),
            make_override<VulkanSettings>("pipeline_cache_enabled",
                                          &VulkanSettings::pipeline_cache_enabled),
            make_override<VulkanSettings>("pipeline_cache_archived",
                                          &VulkanSettings::pipeline_cache_archived),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VulkanSettings, gpu_id, renderdoc_enabled, vkvalidation_enabled,
                                   vkvalidation_core_enabled, vkvalidation_sync_enabled,
                                   vkvalidation_gpu_enabled, vkcrash_diagnostic_enabled,
                                   vkhost_markers, vkguest_markers, pipeline_cache_enabled,
                                   pipeline_cache_archived)

// -------------------------------
// Main manager
// -------------------------------
class EmulatorSettingsImpl {
public:
    EmulatorSettingsImpl();
    ~EmulatorSettingsImpl();

    static std::shared_ptr<EmulatorSettingsImpl> GetInstance();
    static void SetInstance(std::shared_ptr<EmulatorSettingsImpl> instance);
    /// The current instance without locking or reference counting. Instances replaced by SetInstance are kept
    /// alive, so a reference taken before the swap stays valid
    static EmulatorSettingsImpl& Current();

    bool Save(const std::string& serial = "");
    bool Load(const std::string& serial = "");
    void SetDefaultValues();
    bool TransferSettings();

    // Config mode
    ConfigMode GetConfigMode() const {
        return m_configMode;
    }
    void SetConfigMode(ConfigMode mode) {
        m_configMode = mode;
    }

    //
    // Game-specific override management
    /// Clears all per-game overrides.  Call this when a game exits so
    /// the emulator reverts to global settings.
    void ClearGameSpecificOverrides();

    /// Reset a single field's game-specific override by its JSON ke
    void ResetGameSpecificValue(const std::string& key);

    // general accessors
    bool AddGameInstallDir(const std::filesystem::path& dir, bool enabled = true);
    std::vector<std::filesystem::path> GetGameInstallDirs() const;
    void SetAllGameInstallDirs(const std::vector<GameInstallDir>& dirs);
    void RemoveGameInstallDir(const std::filesystem::path& dir);
    void SetGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled);
    void SetGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config);
    const std::vector<bool> GetGameInstallDirsEnabled();
    const std::vector<GameInstallDir>& GetAllGameInstallDirs() const;

    std::filesystem::path GetHomeDir();
    void SetHomeDir(const std::filesystem::path& dir);
    std::filesystem::path GetSysModulesDir();
    void SetSysModulesDir(const std::filesystem::path& dir);
    std::filesystem::path GetFontsDir();
    void SetFontsDir(const std::filesystem::path& dir);
    std::filesystem::path GetAddonInstallDir();
    void SetAddonInstallDir(const std::filesystem::path& dir);

private:
    GeneralSettings m_general{};
    LogSettings m_log{};
    DebugSettings m_debug{};
    InputSettings m_input{};
    AudioSettings m_audio{};
    // Windows static guest red-zone protection
    WindowsGuestRedZoneProtectionSettings m_windows_guest_red_zone_protection{};
    GPUSettings m_gpu{};
    VulkanSettings m_vulkan{};
    ConfigMode m_configMode{ConfigMode::Default};

    // Runtime-only override: when true, IsShadNetEnabled() reports false for the
    // rest of this run regardless of the persisted setting
    std::atomic<bool> m_shadnet_session_disabled{false};

    bool m_loaded{false};

    static std::shared_ptr<EmulatorSettingsImpl> s_instance;
    static std::mutex s_mutex;
    static std::atomic<EmulatorSettingsImpl*> s_current;
    static std::vector<std::shared_ptr<EmulatorSettingsImpl>> s_retired;

    /// Apply overrideable fields from groupJson into group.game_specific_value.
    template <typename Group>
    void ApplyGroupOverrides(Group& group, const nlohmann::json& groupJson,
                             std::vector<std::string>& changed) {
        for (auto& item : group.GetOverrideableFields()) {
            if (!groupJson.contains(item.key))
                continue;
            item.apply(&group, groupJson.at(item.key), changed);
        }
    }

    // Write all overrideable fields from group into out (for game-specific save).
    template <typename Group>
    static void SaveGroupGameSpecific(const Group& group, nlohmann::json& out) {
        for (auto& item : group.GetOverrideableFields())
            out[item.key] = item.get_for_save(&group);
    }

    // Discard every game-specific override in group.
    template <typename Group>
    static void ClearGroupOverrides(Group& group) {
        for (auto& item : group.GetOverrideableFields())
            item.reset_game_specific(&group);
    }

    static void PrintChangedSummary(const std::vector<std::string>& changed);
    void ApplyGameOverrides(const nlohmann::json& gj, std::vector<std::string>& changed);

public:
    // Add these getters to access overrideable fields
    std::vector<OverrideItem> GetGeneralOverrideableFields() const {
        return m_general.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetDebugOverrideableFields() const {
        return m_debug.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetInputOverrideableFields() const {
        return m_input.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetAudioOverrideableFields() const {
        return m_audio.GetOverrideableFields();
    }
    // Windows static guest red-zone protection
    std::vector<OverrideItem> GetWindowsGuestRedZoneProtectionOverrideableFields() const {
        return m_windows_guest_red_zone_protection.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetGPUOverrideableFields() const {
        return m_gpu.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetVulkanOverrideableFields() const {
        return m_vulkan.GetOverrideableFields();
    }
    std::vector<std::string> GetAllOverrideableKeys() const;

#define SETTING_FORWARD(group, Name, field)                                                        \
    auto Get##Name() const {                                                                       \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(const decltype((group).field.value)& v, bool specific = false) {                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL(group, Name, field)                                                   \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(bool v, bool specific = false) {                                                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL_READONLY(group, Name, field)                                          \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }

    // General settings
    SETTING_FORWARD(m_general, VolumeSlider, volume_slider)
    SETTING_FORWARD_BOOL(m_general, Neo, neo_mode)
    SETTING_FORWARD_BOOL(m_general, DevKit, dev_kit_mode)
    SETTING_FORWARD(m_general, ExtraDmemInMBytes, extra_dmem_in_mbytes)
    SETTING_FORWARD(m_general, ExtraFmemInMBytes, extra_fmem_in_mbytes)
    bool IsShadNetEnabled() const {
        return m_general.shad_net_enabled.get(m_configMode) &&
               !m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    void SetShadNetEnabled(bool v, bool specific = false) {
        m_general.shad_net_enabled.set(v, specific);
    }
    bool IsShadNetEnabledSetting() const {
        return m_general.shad_net_enabled.get(m_configMode);
    }
    void SetShadNetSessionDisabled(bool v) {
        m_shadnet_session_disabled.store(v, std::memory_order_relaxed);
    }
    bool IsShadNetSessionDisabled() const {
        return m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    SETTING_FORWARD_BOOL(m_general, TrophyPopupDisabled, trophy_popup_disabled)
    SETTING_FORWARD(m_general, TrophyNotificationDuration, trophy_notification_duration)
    SETTING_FORWARD(m_general, TrophyNotificationSide, trophy_notification_side)
    SETTING_FORWARD_BOOL(m_general, ShowSplash, show_splash)
    SETTING_FORWARD_BOOL(m_general, ConnectedToNetwork, connected_to_network)
    SETTING_FORWARD_BOOL(m_general, DiscordRPCEnabled, discord_rpc_enabled)
    SETTING_FORWARD_BOOL(m_general, ShowFpsCounter, show_fps_counter)
    SETTING_FORWARD(m_general, ConsoleLanguage, console_language)
    SETTING_FORWARD(m_general, BigPictureScale, big_picture_scale)
    SETTING_FORWARD(m_general, ShadNetServer, shadnet_server)
    SETTING_FORWARD(m_general, ShadNetWebApiServer, shadnet_webapi_server)
    SETTING_FORWARD(m_general, SignalingInfo, signaling_info)
    SETTING_FORWARD_BOOL(m_general, UPnPEnabled, enable_upnp)
    SETTING_FORWARD_BOOL(m_general, RedirectApp0Logs, redirect_app0_logs)
    SETTING_FORWARD(m_general, CpuAffinityMask, cpu_affinity_mask)
    SETTING_FORWARD_BOOL(m_general, PollConnectedPadsOnly, poll_connected_pads_only)
    SETTING_FORWARD_BOOL(m_general, SotcPro4k, sotc_pro_4k)

    // Log settings
    SETTING_FORWARD_BOOL(m_log, LogAppend, append)
    SETTING_FORWARD_BOOL(m_log, LogEnable, enable)
    SETTING_FORWARD(m_log, LogFilter, filter)
    SETTING_FORWARD(m_log, LogFlushLevel, flush_level)
    SETTING_FORWARD(m_log, LogMaxSkipDuration, max_skip_duration)
    SETTING_FORWARD_BOOL(m_log, LogSeparate, separate)
    SETTING_FORWARD(m_log, LogSizeLimit, size_limit)
    SETTING_FORWARD_BOOL(m_log, LogSkipDuplicate, skip_duplicate)
    SETTING_FORWARD_BOOL(m_log, LogSync, sync)
    SETTING_FORWARD(m_log, LogConsoleMode, console_mode)
#ifdef _WIN32
    SETTING_FORWARD(m_log, LogType, type)
#endif

    // Audio settings
    SETTING_FORWARD(m_audio, AudioBackend, audio_backend)
    SETTING_FORWARD(m_audio, SDLMicDevice, sdl_mic_device)
    SETTING_FORWARD(m_audio, SDLMainOutputDevice, sdl_main_output_device)
    SETTING_FORWARD(m_audio, SDLPadSpkOutputDevice, sdl_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALMicDevice, openal_mic_device)
    SETTING_FORWARD(m_audio, OpenALMainOutputDevice, openal_main_output_device)
    SETTING_FORWARD(m_audio, OpenALPadSpkOutputDevice, openal_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALHrtf, openal_hrtf)
    SETTING_FORWARD(m_audio, OpenALOutputMode, openal_output_mode)
    SETTING_FORWARD_BOOL(m_audio, AudioFollowGameSpeed, audio_follow_game_speed)
    SETTING_FORWARD(m_audio, AudioMinGameSpeed, audio_min_game_speed)
    SETTING_FORWARD(m_audio, AudioGameTargetFps, audio_game_target_fps)

    // Windows static guest red-zone protection
    SETTING_FORWARD(m_windows_guest_red_zone_protection, WindowsGuestRedZoneProtectionMode,
                    windows_guest_red_zone_protection_mode)

    // Debug settings
    SETTING_FORWARD_BOOL(m_debug, DebugDump, debug_dump)
    SETTING_FORWARD_BOOL(m_debug, ShaderCollect, shader_collect)
    SETTING_FORWARD(m_debug, ConfigVersion, config_version)

    // GPU Settings
    SETTING_FORWARD_BOOL(m_gpu, NullGPU, null_gpu)
    SETTING_FORWARD_BOOL(m_gpu, DumpShaders, dump_shaders)
    SETTING_FORWARD_BOOL(m_gpu, CopyGpuBuffers, copy_gpu_buffers)
    SETTING_FORWARD_BOOL(m_gpu, FullScreen, full_screen)
    SETTING_FORWARD(m_gpu, FullScreenMode, full_screen_mode)
    SETTING_FORWARD(m_gpu, PresentMode, present_mode)
    SETTING_FORWARD(m_gpu, WindowHeight, window_height)
    SETTING_FORWARD(m_gpu, WindowWidth, window_width)
    SETTING_FORWARD(m_gpu, InternalScreenHeight, internal_screen_height)
    SETTING_FORWARD(m_gpu, InternalScreenWidth, internal_screen_width)
    SETTING_FORWARD_BOOL(m_gpu, HdrAllowed, hdr_allowed)
    SETTING_FORWARD_BOOL(m_gpu, FsrEnabled, fsr_enabled)
    SETTING_FORWARD_BOOL(m_gpu, RcasEnabled, rcas_enabled)
    SETTING_FORWARD(m_gpu, RcasAttenuation, rcas_attenuation)
    SETTING_FORWARD_BOOL(m_gpu, DlssNrEnabled, dlss_nr_enabled)
    SETTING_FORWARD(m_gpu, DlssNrStyle, dlss_nr_style)
    SETTING_FORWARD(m_gpu, DlssNrIntensity, dlss_nr_intensity)
    SETTING_FORWARD_BOOL(m_gpu, DlssNrUiCorrection, dlss_nr_ui_correction)
    SETTING_FORWARD(m_gpu, ReadbacksMode, readbacks_mode)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesEnabled, readback_linear_images_enabled)
    SETTING_FORWARD_BOOL(m_gpu, DirectMemoryAccessEnabled, direct_memory_access_enabled)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, PatchShaders, patch_shaders)
    SETTING_FORWARD_BOOL(m_gpu, UserfaultfdTracking, userfaultfd)
    SETTING_FORWARD(m_gpu, ComputeLoopCap, compute_loop_cap)
    SETTING_FORWARD_BOOL(m_gpu, PreserveSplitProtection, preserve_split_protection)
    SETTING_FORWARD_BOOL(m_gpu, CpuAuthoritativeStacks, cpu_authoritative_stacks)
    SETTING_FORWARD(m_gpu, BpeHeapGuardAddress, bpe_heap_guard_address)
    SETTING_FORWARD_BOOL(m_gpu, LdsBarrierUniformReadlane, lds_barrier_uniform_readlane)
    SETTING_FORWARD_BOOL(m_gpu, EarlyFragmentTestsFromZOrder, early_fragment_tests_from_z_order)
    SETTING_FORWARD_BOOL(m_gpu, LodStatsFromBindings, lod_stats_from_bindings)
    SETTING_FORWARD(m_gpu, DynamicTsharpArraySize, dynamic_tsharp_array_size)
    SETTING_FORWARD_BOOL(m_gpu, Wave64UniformBranches, wave64_uniform_branches)
    SETTING_FORWARD_BOOL(m_gpu, Wave64MissingLaneIdentity, wave64_missing_lane_identity)
    SETTING_FORWARD_BOOL(m_gpu, LdsBarriersLargeGroups, lds_barriers_large_groups)
    SETTING_FORWARD_BOOL(m_gpu, RelativeInstanceId, relative_instance_id)
    SETTING_FORWARD_BOOL(m_gpu, GcnNanSemantics, gcn_nan_semantics)
    SETTING_FORWARD_BOOL(m_gpu, LegacyMadSemantics, legacy_mad_semantics)
    SETTING_FORWARD_BOOL(m_gpu, GpuCheckpoints, gpu_checkpoints)
    SETTING_FORWARD_BOOL(m_gpu, SrtWalkerCleanReads, srt_walker_clean_reads)
    SETTING_FORWARD_BOOL(m_gpu, ShaderCodeCleanReads, shader_code_clean_reads)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackAhead, readback_ahead)
    SETTING_FORWARD(m_gpu, PeriodicFlushCommands, periodic_flush_commands)
    SETTING_FORWARD_BOOL(m_gpu, CpRecordingCuts, cp_recording_cuts)
    SETTING_FORWARD_BOOL(m_gpu, GpuOverheadCuts, gpu_overhead_cuts)
    SETTING_FORWARD_BOOL(m_gpu, DmaSweepSkipStacks, dma_sweep_skip_stacks)
    SETTING_FORWARD(m_gpu, WaitSpinUs, wait_spin_us)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackAheadTransferQueue, readback_ahead_transfer_queue)
    SETTING_FORWARD_BOOL(m_gpu, MappedPageTable, mapped_page_table)
    SETTING_FORWARD_BOOL(m_gpu, DmaSyncOncePerBatch, dma_sync_once_per_batch)
    SETTING_FORWARD_BOOL(m_gpu, IncrementalBind, incremental_bind)
    SETTING_FORWARD_BOOL(m_gpu, TsharpCache, tsharp_cache)
    SETTING_FORWARD(m_gpu, WaitMarker, wait_marker)
    SETTING_FORWARD_BOOL(m_gpu, ImageMemoryPool, image_memory_pool)
    SETTING_FORWARD(m_gpu, EopLabelDelayUs, eop_label_delay_us)
    SETTING_FORWARD_BOOL(m_gpu, CpRecordThread, cp_record_thread)
    SETTING_FORWARD(m_gpu, ForceAnisotropy, force_anisotropy)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackHotRegions, readback_hot_regions)
    SETTING_FORWARD_BOOL(m_gpu, DepthTargetSampledLayout, depth_target_sampled_layout)
    SETTING_FORWARD_BOOL(m_gpu, GcnUnorderedDispatches, gcn_unordered_dispatches)
    SETTING_FORWARD_BOOL(m_gpu, AccessBitmapTracking, access_bitmap_tracking)
    SETTING_FORWARD_BOOL(m_gpu, PageSpinLocks, page_spin_locks)
    SETTING_FORWARD_BOOL(m_gpu, RangeFastPaths, range_fast_paths)
    SETTING_FORWARD(m_gpu, UnboundedVsharpCapMb, unbounded_vsharp_cap_mb)
    SETTING_FORWARD_BOOL(m_gpu, TextureGcFullScan, texture_gc_full_scan)
    SETTING_FORWARD(m_gpu, TextureGcTriggerMb, texture_gc_trigger_mb)
    SETTING_FORWARD(m_gpu, ImagePoolTextureMaxKb, image_pool_texture_max_kb)
    SETTING_FORWARD(m_gpu, ArenaEvictIdleS, arena_evict_idle_s)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, InlineFetchShader, inline_fetch_shader)

    u32 GetVblankFrequency() {
        if (m_gpu.vblank_frequency.value < 30) {
            return 30;
        }
        return m_gpu.vblank_frequency.get();
    }
    void SetVblankFrequency(const u32& v, bool is_specific = false) {
        u32 val = v < 30 ? 30 : v;
        if (is_specific) {
            m_gpu.vblank_frequency.game_specific_value = val;
        } else {
            m_gpu.vblank_frequency.value = val;
        }
    }

    // Input Settings
    SETTING_FORWARD(m_input, CursorState, cursor_state)
    SETTING_FORWARD(m_input, CursorHideTimeout, cursor_hide_timeout)
    SETTING_FORWARD(m_input, UsbDeviceBackend, usb_device_backend)
    SETTING_FORWARD_BOOL(m_input, MotionControlsEnabled, motion_controls_enabled)
    SETTING_FORWARD_BOOL(m_input, BackgroundControllerInput, background_controller_input)
    SETTING_FORWARD_BOOL(m_input, ImeAccessibilityEnabled, ime_accessibility_enabled)
    SETTING_FORWARD_BOOL(m_input, ImeUrlMailShortPanel, ime_url_mail_short_panel)
    SETTING_FORWARD(m_input, DefaultControllerId, default_controller_id)
    SETTING_FORWARD_BOOL(m_input, UsingSpecialPad, use_special_pad)
    SETTING_FORWARD(m_input, SpecialPadClass, special_pad_class)
    SETTING_FORWARD_BOOL(m_input, UseUnifiedInputConfig, use_unified_input_config)
    SETTING_FORWARD(m_input, CameraId, camera_id)
    SETTING_FORWARD_BOOL(m_input, CircleEnter, is_circle_enter)
    SETTING_FORWARD_BOOL(m_input, MiceUsedAsMice, use_mice_as_mice)

    // Vulkan settings
    SETTING_FORWARD(m_vulkan, GpuId, gpu_id)
    SETTING_FORWARD_BOOL(m_vulkan, RenderdocEnabled, renderdoc_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationEnabled, vkvalidation_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationCoreEnabled, vkvalidation_core_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationSyncEnabled, vkvalidation_sync_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationGpuEnabled, vkvalidation_gpu_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkCrashDiagnosticEnabled, vkcrash_diagnostic_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkHostMarkersEnabled, vkhost_markers)
    SETTING_FORWARD_BOOL(m_vulkan, VkGuestMarkersEnabled, vkguest_markers)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheEnabled, pipeline_cache_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheArchived, pipeline_cache_archived)

#undef SETTING_FORWARD
#undef SETTING_FORWARD_BOOL
#undef SETTING_FORWARD_BOOL_READONLY
};
