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
    // Per-game workaround: mount user/game_logs/<serial> writable at /app0/logs so games that
    // write their own engine log next to the executable can do so.
    Setting<bool> redirect_app0_logs{false};
    // Processor affinity mask applied to the whole process at startup (0 = leave it alone). On a two-CCD
    // Ryzen, the CCD with the faster cores (0xFFFF on a 9950X) is ~10 % faster than landing on the other one
    Setting<u64> cpu_affinity_mask{0};
    // One 125 Hz input timer instead of four 250 Hz ones (one per pad slot), pushing the state of the first slot
    // (keyboard and main pad) and of slots with a pad connected. The SDL timer thread's cost follows its wake rate
    Setting<bool> poll_connected_pads_only{false};

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
                                           &GeneralSettings::poll_connected_pads_only)};
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
                                   cpu_affinity_mask, poll_connected_pads_only)

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
    // "all": the console mirrors the log file. "events": the console only shows crash/broken
    // shader/render problem events, Critical messages and a session summary on exit (the log file
    // keeps everything and also gets the events)
    Setting<std::string> console_mode{"all"};
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
    // Plays audio at the game's speed: measured guest flips/s over the target frame rate, so a
    // 30 fps game running at 7.5 fps plays audio at 0.25x (lower pitch). Off by default.
    Setting<bool> audio_follow_game_speed{false};
    // Lowest playback speed in percent (clamped to 5-100).
    Setting<u32> audio_min_game_speed{10};
    // Target frame rate; 0 = derive it from sceVideoOutSetFlipRate (60 / (rate + 1)).
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
    // Record GPU commands and submits to identify the first unfinished submit after device loss
    // Add NVIDIA checkpoints when available; recording adds a small cost per command
    Setting<bool> gpu_checkpoints{false};
    // Serve SRT walker reads of bytes the GPU never wrote from guest memory, keeping the page protected,
    // instead of reading back a page shared with a GPU-written buffer and draining the GPU
    Setting<bool> srt_walker_clean_reads{false};
    // Compare cached shader code bytes the GPU never wrote through the backing, keeping their page protected,
    // instead of reading back a page shared with a GPU-written buffer and draining the GPU
    Setting<bool> shader_code_clean_reads{false};
    // Copy a readback in a command buffer submitted ahead of the one being recorded when none of its
    // bytes has a writer there, instead of submitting and draining the whole batch recorded so far
    Setting<bool> readback_ahead{false};
    // Submit without waiting after this many guest draws/dispatches (0 = off), so the GPU runs the frame
    // while it is recorded instead of at the readback drains. Use it only with readback_ahead: the fork lost
    // the device once (WriteInvalid) with 128 and readback_ahead off; with readback_ahead and 64 it held
    Setting<u32> periodic_flush_commands{0};
    // Command processor recording cuts (Idea E): lock-free cache of GPU-mapped runs for IsMapped, and
    // skip re-adding every resident range to the sync batch when a DMA sync since the last flush covered it
    Setting<bool> cp_recording_cuts{false};
    // GPU overhead cuts (Idea G): detile textures uploaded from guest memory out of VRAM instead of reading
    // the host staging buffer across PCIe with the detiler's scattered loads, and drop pipeline binds of the
    // pipeline the command buffer already has bound
    Setting<bool> gpu_overhead_cuts{false};
    // Leave guest stacks out of the DMA sync sweep: stack pages are never write-watched, so every sweep
    // re-uploaded all resident stack bytes. Explicit bindings that cover a stack still upload it. Only makes
    // sense with cpu_authoritative_stacks: without it no stack is registered and the sweep is the same
    Setting<bool> dma_sweep_skip_stacks{false};
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
            make_override<GPUSettings>("inline_fetch_shader", &GPUSettings::inline_fetch_shader),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GPUSettings, window_width, window_height, internal_screen_width,
                                   internal_screen_height, null_gpu, copy_gpu_buffers,
                                   readbacks_mode, readback_linear_images_enabled,
                                   direct_memory_access_enabled, dump_shaders, patch_shaders,
                                   vblank_frequency, full_screen, full_screen_mode, present_mode,
                                   hdr_allowed, fsr_enabled, rcas_enabled, rcas_attenuation,
                                   compute_loop_cap, preserve_split_protection,
                                   cpu_authoritative_stacks, bpe_heap_guard_address,
                                   lds_barrier_uniform_readlane, early_fragment_tests_from_z_order,
                                   lod_stats_from_bindings, dynamic_tsharp_array_size,
                                   wave64_uniform_branches, gpu_checkpoints,
                                   srt_walker_clean_reads, shader_code_clean_reads,
                                   readback_ahead, periodic_flush_commands,
                                   cp_recording_cuts, gpu_overhead_cuts, dma_sweep_skip_stacks)
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
    SETTING_FORWARD_BOOL(m_gpu, GpuCheckpoints, gpu_checkpoints)
    SETTING_FORWARD_BOOL(m_gpu, SrtWalkerCleanReads, srt_walker_clean_reads)
    SETTING_FORWARD_BOOL(m_gpu, ShaderCodeCleanReads, shader_code_clean_reads)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackAhead, readback_ahead)
    SETTING_FORWARD(m_gpu, PeriodicFlushCommands, periodic_flush_commands)
    SETTING_FORWARD_BOOL(m_gpu, CpRecordingCuts, cp_recording_cuts)
    SETTING_FORWARD_BOOL(m_gpu, GpuOverheadCuts, gpu_overhead_cuts)
    SETTING_FORWARD_BOOL(m_gpu, DmaSweepSkipStacks, dma_sweep_skip_stacks)
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
