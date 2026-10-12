// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <deque>
#include <memory>

#include "common/types.h"
#include "core/libraries/ngs2/ngs2.h"

namespace Libraries::Ngs2 {

constexpr u32 NGS2_WAVEFORM_TYPE_PCM_I16 = 0x12;
constexpr u32 NGS2_WAVEFORM_TYPE_PCM_F32 = 0x18;
constexpr u32 NGS2_WAVEFORM_TYPE_ATRAC9 = 0x40;

constexpr u32 NGS2_WAVEFORM_BLOCKS_FLAG_CONTINUE = 1;
constexpr u32 NGS2_WAVEFORM_BLOCKS_FLAG_APPEND = 2;
constexpr u32 NGS2_WAVEFORM_BLOCKS_FLAG_RESET = 4;

constexpr u32 NGS2_WAVEFORM_QUEUE_MAX = 64;
constexpr u32 NGS2_REPEAT_FOREVER = 0xFFFFFFFFu;

enum class WaveformCodec : u8 {
    None,
    PcmI16,
    PcmF32,
    Atrac9,
    Unsupported,
};

struct WaveformBlockEntry {
    const u8* data;
    u32 dataSize;
    u32 numRepeats;
    u32 numSkipSamples;
    u32 numSamples;
    u64 userData;
    bool append;
};

// Turns a queue of waveform blocks into a sequential stream of interleaved float frames at the
// waveform's own sample rate. Not thread safe: the owner serializes access.
class WaveformStream {
public:
    WaveformStream();
    ~WaveformStream();

    void Configure(const OrbisNgs2WaveformFormat& format);
    void Clear();

    u32 AddBlocks(const WaveformBlockEntry* blocks, u32 count, u32 flags);
    u32 Pull(float* dst, u32 max_frames);

    size_t QueueSize() const {
        return blocks.size();
    }
    u32 NumChannels() const {
        return channels;
    }
    WaveformCodec Codec() const {
        return codec;
    }
    bool Configured() const {
        return codec != WaveformCodec::None;
    }
    bool HasData() const {
        return !blocks.empty();
    }
    bool Ended() const {
        return ever_added && blocks.empty() && !more_expected;
    }
    bool Starved() const {
        return blocks.empty() && (more_expected || !ever_added);
    }
    void ExitLoop() {
        cursor.repeats_left = 0;
    }
    u64 DecodedSamples() const {
        return decoded_samples;
    }
    u64 DecodedBytes() const {
        return decoded_bytes;
    }
    u64 CurrentUserData() const {
        return blocks.empty() ? last_user_data : blocks.front().userData;
    }
    const void* CurrentData() const {
        return blocks.empty() ? last_data_end
                              : blocks.front().data + (cursor.open ? cursor.byte_pos : 0);
    }
    u64 TakeDecodeErrors() {
        const u64 errors = decode_errors;
        decode_errors = 0;
        return errors;
    }

private:
    struct Cursor {
        u32 repeats_left{};
        u32 skip_left{};
        u32 frames_left{};
        u32 byte_pos{};
        u32 pass_frames{};
        bool unlimited{};
        bool open{};
    };
    struct Atrac9Decoder;

    void StartPass(bool first_pass);
    void FinishPass();
    u32 PullPcm(float* dst, u32 max_frames, bool& pass_done);
    u32 PullAtrac9(float* dst, u32 max_frames, bool& pass_done);
    u32 PullSilence(float* dst, u32 max_frames, bool& pass_done);

    std::deque<WaveformBlockEntry> blocks;
    Cursor cursor;
    WaveformCodec codec{WaveformCodec::None};
    u32 channels{1};
    bool more_expected{};
    bool ever_added{};
    std::unique_ptr<Atrac9Decoder> at9;

    u64 decoded_samples{};
    u64 decoded_bytes{};
    u64 decode_errors{};
    u64 last_user_data{};
    const u8* last_data_end{};
};

} // namespace Libraries::Ngs2
