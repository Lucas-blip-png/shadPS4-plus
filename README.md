<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->
# shadPS4-plus

A PS4 emulator for Windows, forked from [GR2fork](https://github.com/junminlee2004/GR2fork) (itself a performance fork of
[shadPS4](https://github.com/shadps4-emu/shadPS4)), with fixes and options gathered from other forks so that two demanding
titles run well on a mid-range PC:

| Game | Title ID | State on shadPS4-plus |
|---|---|---|
| inFAMOUS Second Son | CUSA00223 | Near full speed (see results), smoke and all effects intact |
| Shadow of the Colossus (2018) | CUSA08034 (US 1.00) | Playable at about 30 fps, 25/25 clean boots; **no sound yet** (see Known issues) |

Everything is off by default. A game only changes behaviour when its per-game config turns an option on, so other
titles keep running exactly as they do on GR2fork.

## Quick start

1. Download the release zip and extract `shadPS4.exe` next to a `user` folder (portable mode), or use it from a launcher.
2. Copy the preset for your game from [`configs/`](configs) to `user/custom_configs/<TITLE ID>.json`.
3. For Shadow of the Colossus also read the notes below (`extra_dmem_in_mbytes`, `bpe_heap_guard_address`).
4. Start the game with `shadPS4.exe <path to eboot.bin>`.

The first launch of a game compiles its shaders and is slow; the pipeline cache makes later launches fast. Changing the
executable invalidates part of that cache once.

Building from source: see [documents/building-windows.md](documents/building-windows.md) (Release, clang-cl + Ninja).

## What is in here

Base: GR2fork `GR2fork-EXODUS` (upstream shadPS4 merged with GR2fork's CPU-side optimisations).

Added on top, by the maintainer of this fork:

- **`readback_transfer_queue`**: guest readbacks whose window the open command batch did not write are copied on a
  transfer-only queue and wait only for the in-flight batch that last wrote that window, instead of submitting the open
  batch. Falls back automatically when the GPU has no transfer-only queue family.
- Pipeline cache rebuild, real occlusion queries (`real_occlusion_queries`), threaded command recording
  (`threaded_cmd_recording`), compute pipelines compiled without optimisations first and optimised on a background thread
  (`compute_fast_compile`), a JIT patch race fix and several input fixes.
- A fix for `ResourcePool` captured GPU ticks that never refreshed (a slow memory leak).
- SRT walker faults skip only the faulting load instead of patching shared code; a V# outside the guest address space
  binds a null buffer.
- NGS2 diagnostics (`Lib.Ngs2:Info`, `Lib.AudioOut:Info` log levels) that report what a game feeds the audio engine.

Ported from [Pink-shadPS4](https://github.com/luizgustavs/Pink-shadPS4) (luizgustavs), re-applied by hand to the GR2fork
code, each behind its own setting:

| Setting | Default | Purpose |
|---|---|---|
| `wave64_uniform_branches`, `wave64_missing_lane_identity` | off | Wave64 shaders on narrower host subgroups |
| `lds_barrier_uniform_readlane`, `lds_barriers_large_groups` | off | Extra LDS barriers for subgroup-size mismatches |
| `compute_loop_cap` | 0 | Stops runaway compute loops after N back edges (avoids GPU timeouts) |
| `dynamic_tsharp_array_size` | 0 | Runtime-indexed sampled T# tables as descriptor arrays |
| `early_fragment_tests_from_z_order` | off | Early depth/stencil test for storage-writing pixel shaders |
| `lod_stats_from_bindings` | off | Answers `IT_GET_LOD_STATS` from bound T# counter banks |
| `preserve_split_protection` | off | Keeps page protections when a mapped region is split (Windows) |
| `cpu_authoritative_stacks` | off | GPU readbacks never overwrite guest stacks |
| `bpe_heap_guard_address` | 0 (off) | Protects a game allocator heap from GPU readbacks |
| `arena_reserve_mb` | 0 (off) | Reserves VRAM for the sparse buffer arena at boot (8 GB cards) |

and unconditional fixes: handling of host C++ exceptions in the signal handler, a smaller fault-handler stack frame,
`TryWriteBacking` copying the right source slice across physical pieces, texture reader-stage barriers, reserved swizzle
selectors treated as zero, and an LRU touch of freed nodes.

## Results

Measured on a Ryzen 7 5700X, RTX 5060 Ti 8 GB, Windows 11, 1080p.

| Test | Result |
|---|---|
| inFAMOUS, standing still on a rooftop | 56-58 fps (capped at 60) |
| inFAMOUS, 7-minute city route (same input script) | 48 fps average, no leak, no black frames |
| Shadow of the Colossus, intro | 47-50 fps in the first seconds, 28-30 fps in the heavy scene |
| Shadow of the Colossus, 25 boots of ~100 s | 25 clean, 0 crashes |

## Shadow of the Colossus notes

- `stream_copy_workers` must be **2** for this game. The default (1) is the fast, unsafe stream copy lane meant for titles
  that never unmap memory during play; this game does, and it drops to 2-5 fps.
- `extra_dmem_in_mbytes` 1024: the game aborts with "Out of Memory GPUMemory" with the default.
- `bpe_heap_guard_address` depends on `extra_dmem_in_mbytes` for US 1.00: `0x33160fd20 + extra_dmem` (1024 MB gives
  `14787083552`). A wrong address only disables the guard (a warning is logged once).
- `arena_reserve_mb` 2560 keeps the buffer arena from running out of VRAM on 8 GB cards.

## Known issues

- Shadow of the Colossus has **no sound**: the game renders its audio through NGS2, whose HLE in shadPS4 is a set of
  stubs. A software mixer is being developed (setting `ngs2_software_mixer`); until it lands the game is silent.
- `IT_SET_PREDICATION` is not implemented (the log warns about it, as upstream does).

## Credits and license

- [shadPS4](https://github.com/shadps4-emu/shadPS4) and its contributors.
- [GR2fork](https://github.com/junminlee2004/GR2fork) by junminlee2004: base of this fork. The original README is kept in
  [documents/README-GR2fork.md](documents/README-GR2fork.md).
- [Pink-shadPS4](https://github.com/luizgustavs/Pink-shadPS4) by luizgustavs: the Shadow of the Colossus work ported here.
- Ideas for inFAMOUS from the Infamous-FixBuild fork by rayhlancaner.

Licensed under GPL-2.0-or-later, like shadPS4. See [LICENSE](LICENSE).
