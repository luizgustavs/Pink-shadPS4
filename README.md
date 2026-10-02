# Pink-shadPS4

A fork of shadPS4 focused on fixes specific to Shadow of the Colossus, including experimental fixes and improvements. Some changes may help other games or cause regressions.
The focus is Windows with NVIDIA graphics cards. I do not have the means to test other systems or AMD hardware, and some changes are specific to NVIDIA.

# DOWNLOAD

[![Download latest release](https://img.shields.io/badge/Download-Latest%20Release-pink?style=for-the-badge)](https://github.com/luizgustavs/Pink-shadPS4/releases/download/0.2.0/Pink-shardPS4.0.2.0.zip)

[Download Pink-shardPS4 0.2.0 ZIP](https://github.com/luizgustavs/Pink-shadPS4/releases/download/0.2.0/Pink-shardPS4.0.2.0.zip)

The ZIP already includes the launcher and emulator with specific settings for the game. Use the launcher included in this version; other launchers won't enable the required workarounds.

## Quick start

1. Download the ZIP and extract it into a folder.
2. Run the launcher once to create the `games` folder, then put your game folder (`CUSAXXXXX`) inside it.
3. Play! The launcher should apply every setting for and and automatically import the correct shadPS4.exe for you 

### Recommended settings

#### NVIDIA and AMD

- **Workarounds > Command recording cuts:** turn this on for some extra FPS. It may cause occasional hiccups or rare graphical glitches, so turn it off for a more stable experience.
- **General > Additional DMMem allocation:** `512 MB` may help. If your system is memory-constrained, leave it at `0`.

#### Additional AMD settings

AMD users may get a better experience with the following workaround settings:

- **Periodic flush (commands):** set to `0`.
- **Readback ahead on the copy engine:** uncheck.
- **GPU wait spin:** set to `0`.
- **GPU overhead cuts:** uncheck.

## Known issues

1. Many issues remain; this fork is still experimental.
2. VRAM usage is very high. It is unclear whether playing with less than 12 GB of VRAM is viable.
3. Some textures occasionally fail to load. Restarting the game usually fixes this.
4. Stability on AMD graphics cards is currently uncertain.
5. Some level-of-detail (LOD) issues remain.

## Technical stuff (Workaround keys)

Workarounds can improve specific games but may reduce performance or cause regressions elsewhere. Configure them in the launcher's **Workarounds** tab or in a per-game preset.

### General

| Key | Description |
| --- | --- |
| `extra_fmem_in_mbytes` | Adds flexible memory for games that run out of it. |
| `redirect_app0_logs` | Redirects `/app0/logs` to a writable per-game log folder. |
| `cpu_affinity_mask` | On Windows, pins the emulator to the selected logical CPUs; `0` disables it. |
| `poll_connected_pads_only` | Polls the keyboard and connected controllers at 125 Hz to reduce input-thread CPU use. |

### GPU

| Key | Description |
| --- | --- |
| `compute_loop_cap` | Stops runaway compute shader loops after this many iterations; `0` disables the limit. |
| `preserve_split_protection` | On Windows, restores GPU memory protections lost when a mapped placeholder is split. |
| `cpu_authoritative_stacks` | Keeps guest thread and fiber stacks writable and owned by the CPU. |
| `bpe_heap_guard_address` | Protects SotC's BPE heap metadata from GPU readbacks; `0` disables the guard. |
| `lds_barrier_uniform_readlane` | Preserves shared-memory barriers in uniform `ReadLane` branches to fix lighting stripes. |
| `early_fragment_tests_from_z_order` | Runs depth and stencil tests before eligible storage-writing pixel shaders. |
| `lod_stats_from_bindings` | Derives texture LOD statistics from bound textures so higher-resolution mipmaps can load. |
| `dynamic_tsharp_array_size` | Sets the descriptor-array size for dynamically indexed texture tables; `0` disables it. |
| `wave64_uniform_branches` | Correctly lowers wave64 lane operations inside workgroup-uniform branches. |
| `wave64_missing_lane_identity` | Supplies PS4 reduction identity values for wave64 lanes missing on the host GPU. |
| `gpu_checkpoints` | Records GPU commands and submits to help diagnose device loss, at a performance cost. |
| `srt_walker_clean_reads` | Reads clean shader-resource-table data from guest memory without draining the GPU. |
| `shader_code_clean_reads` | Checks clean shader code through guest memory to avoid unnecessary GPU readbacks. |
| `readback_ahead` | Submits safe GPU readbacks ahead of the current command buffer to reduce waiting. |
| `periodic_flush_commands` | Submits after this many draws or dispatches without waiting; `0` disables it. |
| `readback_ahead_transfer_queue` | Runs readback-ahead copies on a transfer-only queue when available. |
| `wait_spin_us` | Polls GPU waits for this many microseconds before sleeping; `0` disables spinning. |
| `dma_sweep_skip_stacks` | Excludes guest stacks from bulk DMA synchronization; requires `cpu_authoritative_stacks`. |
| `gpu_overhead_cuts` | Detiles uploaded textures in VRAM and skips redundant pipeline bindings. |
| `cp_recording_cuts` | Caches memory lookups, but may cause visual glitches. The DMA sync skip moved to `dma_sync_once_per_batch`. |

#### GPU keys added in 0.3.0

These defaults are from the emulator; per-game presets may override them.

| Key | Default | Description |
| --- | --- | --- |
| `mapped_page_table` | `false` | Answers GPU memory lookups from a table of mapped 16 KB pages to avoid locked searches. |
| `dma_sync_once_per_batch` | `false` | Adds resident buffer ranges to a DMA sync batch only once instead of on every DMA dispatch. This replaces the DMA sync skip previously included in `cp_recording_cuts`. |
| `incremental_bind` | `false` | Reuses unchanged read-only buffer and texture bindings from the previous call of a pipeline while their resources remain valid. |
| `tsharp_cache` | `false` | Reuses texture descriptor lookups across pipelines while the registered images on their pages remain unchanged; changes what the GPU samples, so check visually. |
| `wait_marker` | `0` | `0` disables it; `1` polls a GPU-written marker before querying the driver; `2` also lets readbacks complete on that marker. Requires `wait_spin_us` greater than `0`. |
| `image_memory_pool` | `false` | Sub-allocates images up to 16 MB from pools with 64 MB blocks instead of allocating memory separately for each image. |
| `cp_record_thread` | `false` | Moves Vulkan command recording and submissions to a separate worker thread; pairs with `readback_ahead_transfer_queue`. |
| `lds_barriers_large_groups` | `true` | Adds shared-memory barriers to compute workgroups larger than 64 threads to fix the blown-out sun on NVIDIA. |
| `eop_label_delay_us` | `0` | Temporary workaround for SotC's flickering rocks: delays graphics EOP labels by a fixed number of microseconds (`2000` was used in testing); `0` disables it. This does not guarantee GPU completion, and enabling `readback_ahead` can bring the flicker back. |

### Other settings used by the SotC presets

These are regular emulator settings stored alongside the workarounds in the shipped presets.

| Key | Description |
| --- | --- |
| `extra_dmem_in_mbytes` | Adds direct memory for games that need a larger allocation pool. |
| `filter` | Selects which log categories and severity levels are recorded. |
| `direct_memory_access_enabled` | Allows direct access to guest memory from the GPU path. |
| `readbacks_mode` | Selects the GPU readback accuracy mode; the SotC preset uses precise mode (`2`). |
| `pipeline_cache_enabled` | Reuses cached Vulkan pipelines to reduce shader compilation stutter. |
| `windows_guest_red_zone_protection_mode` | Selects the Windows guest red-zone protection strategy. |

### Shader cache

If a game shows a burst of rejected textures, thousands of new shader permutations or a device loss right
after boot, the shader cache may have been corrupted by an older build (a new permutation could overwrite
another one on disk). Delete the whole `user/cache/<serial>` folder (or `<serial>.zip`) and let it rebuild.

### Shadow of the Colossus engine settings

Not an emulator setting: adding both `+doIndexBufferCulling=0` and `+doSunShadowIndexBufferCulling=0` to
`games/<serial>/CommandLineArgs.txt` raised the sanctuary from ~22 to ~28 fps in testing, with the same image.
Always use the two together; one without the other gives wrong geometry or a device loss.

### Audio

| Key | Description |
| --- | --- |
| `audio_follow_game_speed` | Slows audio with the game to avoid crackling below the target frame rate. |
| `audio_min_game_speed` | Sets the minimum audio playback speed as a percentage. |
| `audio_game_target_fps` | Sets the frame rate considered full speed; `0` uses the game's flip rate. |

## Original repositories

- [shadPS4](https://github.com/shadps4-emu/shadPS4)
- [shadPS4 QtLauncher](https://github.com/shadps4-emu/shadps4-qtlauncher)

## License

GPL-2.0, as both upstream projects.
