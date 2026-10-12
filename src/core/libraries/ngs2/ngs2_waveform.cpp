// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

#include "core/libraries/ngs2/ngs2_waveform.h"

#include "libatrac9.h"

namespace Libraries::Ngs2 {

namespace {

constexpr u8 ATRAC9_CONFIG_HEADER = 0xFE;
constexpr int ATRAC9_MAX_FRAME_SAMPLES = 4096;

// LibAtrac9 fills shared lookup tables while initializing a decoder
std::mutex atrac9_init_mutex;

bool ReadAtrac9Config(u32 config_data, u8 (&out)[4]) {
    if ((config_data >> 24) == ATRAC9_CONFIG_HEADER) {
        out[0] = static_cast<u8>(config_data >> 24);
        out[1] = static_cast<u8>(config_data >> 16);
        out[2] = static_cast<u8>(config_data >> 8);
        out[3] = static_cast<u8>(config_data);
        return true;
    }
    if ((config_data & 0xFF) == ATRAC9_CONFIG_HEADER) {
        std::memcpy(out, &config_data, sizeof(out));
        return true;
    }
    return false;
}

} // namespace

struct WaveformStream::Atrac9Decoder {
    ~Atrac9Decoder() {
        if (handle) {
            Atrac9ReleaseHandle(handle);
        }
    }

    bool Init(u32 config_data) {
        if (!ReadAtrac9Config(config_data, config)) {
            return false;
        }
        fresh = false;
        return Reset();
    }

    // A decoder keeps overlap state between frames, so a fresh stream needs a fresh handle
    bool Reset() {
        if (!fresh) {
            std::scoped_lock lock{atrac9_init_mutex};
            if (handle) {
                Atrac9ReleaseHandle(handle);
            }
            handle = Atrac9GetHandle();
            if (!handle) {
                return false;
            }
            if (Atrac9InitDecoder(handle, config) != 0 || Atrac9GetCodecInfo(handle, &info) != 0 ||
                info.channels < 1 || info.channels > ORBIS_NGS2_MAX_VOICE_CHANNELS ||
                info.frameSamples < 1 || info.frameSamples > ATRAC9_MAX_FRAME_SAMPLES ||
                info.superframeSize < 1 || info.framesInSuperframe < 1) {
                Atrac9ReleaseHandle(handle);
                handle = nullptr;
                return false;
            }
            frame.assign(static_cast<size_t>(info.frameSamples) * info.channels, 0.0f);
            fresh = true;
        }
        frame_pos = 0;
        frame_len = 0;
        superframe_remain = static_cast<u32>(info.superframeSize);
        frames_in_superframe = 0;
        return true;
    }

    enum class Result { Frame, End, Error };

    Result NextFrame(const u8* data, u32 size, u32& byte_pos) {
        if (!handle) {
            return Result::Error;
        }
        if (byte_pos >= size) {
            return Result::End;
        }
        int used = 0;
        if (Atrac9DecodeF32(handle, data + byte_pos, static_cast<int>(size - byte_pos),
                            frame.data(), &used, 0) != 0) {
            return Result::Error;
        }
        fresh = false;
        const u32 used_bytes = std::min<u32>(static_cast<u32>(std::max(used, 0)), size - byte_pos);
        byte_pos += used_bytes;
        superframe_remain = superframe_remain > used_bytes ? superframe_remain - used_bytes : 0;
        if (++frames_in_superframe >= static_cast<u32>(info.framesInSuperframe)) {
            byte_pos = std::min(size, byte_pos + superframe_remain);
            superframe_remain = static_cast<u32>(info.superframeSize);
            frames_in_superframe = 0;
        }
        frame_pos = 0;
        frame_len = static_cast<u32>(info.frameSamples);
        return Result::Frame;
    }

    void* handle{};
    Atrac9CodecInfo info{};
    u8 config[4]{};
    std::vector<float> frame;
    u32 frame_pos{};
    u32 frame_len{};
    u32 superframe_remain{};
    u32 frames_in_superframe{};
    bool fresh{};
};

WaveformStream::WaveformStream() = default;
WaveformStream::~WaveformStream() = default;

void WaveformStream::Clear() {
    blocks.clear();
    cursor = {};
    more_expected = false;
    ever_added = false;
    decoded_samples = 0;
    decoded_bytes = 0;
    last_user_data = 0;
    last_data_end = nullptr;
}

void WaveformStream::Configure(const OrbisNgs2WaveformFormat& format) {
    Clear();
    at9.reset();
    channels = std::clamp<u32>(format.numChannels, 1, ORBIS_NGS2_MAX_VOICE_CHANNELS);
    if (format.waveformType == 0 && format.numChannels == 0) {
        codec = WaveformCodec::None;
        return;
    }
    switch (format.waveformType) {
    case NGS2_WAVEFORM_TYPE_PCM_I16:
        codec = WaveformCodec::PcmI16;
        return;
    case NGS2_WAVEFORM_TYPE_PCM_F32:
        codec = WaveformCodec::PcmF32;
        return;
    default:
        break;
    }
    u8 config[4];
    if (format.waveformType == NGS2_WAVEFORM_TYPE_ATRAC9 ||
        ReadAtrac9Config(format.configData, config)) {
        at9 = std::make_unique<Atrac9Decoder>();
        if (at9->Init(format.configData)) {
            codec = WaveformCodec::Atrac9;
            channels = static_cast<u32>(at9->info.channels);
            return;
        }
        at9.reset();
        decode_errors++;
    }
    codec = WaveformCodec::Unsupported;
}

u32 WaveformStream::AddBlocks(const WaveformBlockEntry* new_blocks, u32 count, u32 flags) {
    if (flags & NGS2_WAVEFORM_BLOCKS_FLAG_RESET) {
        blocks.clear();
        cursor = {};
    }
    u32 added = 0;
    for (; added < count && blocks.size() < NGS2_WAVEFORM_QUEUE_MAX; added++) {
        WaveformBlockEntry entry = new_blocks[added];
        entry.append = (flags & NGS2_WAVEFORM_BLOCKS_FLAG_APPEND) != 0;
        blocks.push_back(entry);
    }
    more_expected = (flags & NGS2_WAVEFORM_BLOCKS_FLAG_CONTINUE) != 0;
    ever_added = true;
    return added;
}

void WaveformStream::StartPass(bool first_pass) {
    const WaveformBlockEntry& block = blocks.front();
    cursor.open = true;
    cursor.skip_left = block.numSkipSamples;
    cursor.unlimited = block.numSamples == 0;
    cursor.frames_left = block.numSamples;
    cursor.byte_pos = 0;
    cursor.pass_frames = 0;
    if (first_pass) {
        cursor.repeats_left = block.numRepeats;
    }
    if (!block.data) {
        cursor.skip_left = 0;
        cursor.unlimited = false;
        return;
    }
    switch (codec) {
    case WaveformCodec::PcmI16:
    case WaveformCodec::PcmF32: {
        const u32 bytes_per_frame = channels * (codec == WaveformCodec::PcmF32 ? 4 : 2);
        const u32 total_frames = block.dataSize / bytes_per_frame;
        const u32 skipped = std::min(cursor.skip_left, total_frames);
        cursor.skip_left = 0;
        cursor.byte_pos = skipped * bytes_per_frame;
        const u32 available = total_frames - skipped;
        if (cursor.unlimited || cursor.frames_left > available) {
            cursor.frames_left = available;
        }
        cursor.unlimited = false;
        break;
    }
    case WaveformCodec::Atrac9:
        if (!(first_pass && block.append) && !at9->Reset()) {
            decode_errors++;
        }
        break;
    default:
        cursor.skip_left = 0;
        cursor.unlimited = false;
        break;
    }
}

void WaveformStream::FinishPass() {
    if (cursor.repeats_left != 0 && cursor.pass_frames != 0) {
        if (cursor.repeats_left != NGS2_REPEAT_FOREVER) {
            cursor.repeats_left--;
        }
        StartPass(false);
        return;
    }
    const WaveformBlockEntry& block = blocks.front();
    last_user_data = block.userData;
    last_data_end = block.data + block.dataSize;
    blocks.pop_front();
    cursor.open = false;
}

u32 WaveformStream::PullPcm(float* dst, u32 max_frames, bool& pass_done) {
    const bool is_f32 = codec == WaveformCodec::PcmF32;
    const u32 bytes_per_frame = channels * (is_f32 ? 4 : 2);
    const u32 frames = std::min(max_frames, cursor.frames_left);
    const u8* src = blocks.front().data + cursor.byte_pos;
    const size_t samples = static_cast<size_t>(frames) * channels;
    if (is_f32) {
        std::memcpy(dst, src, samples * sizeof(float));
    } else {
        for (size_t i = 0; i < samples; i++) {
            s16 value;
            std::memcpy(&value, src + i * sizeof(s16), sizeof(s16));
            dst[i] = static_cast<float>(value) * (1.0f / 32768.0f);
        }
    }
    cursor.byte_pos += frames * bytes_per_frame;
    cursor.frames_left -= frames;
    decoded_bytes += static_cast<u64>(frames) * bytes_per_frame;
    pass_done = cursor.frames_left == 0;
    return frames;
}

u32 WaveformStream::PullAtrac9(float* dst, u32 max_frames, bool& pass_done) {
    const WaveformBlockEntry& block = blocks.front();
    Atrac9Decoder& decoder = *at9;
    u32 produced = 0;
    while (produced < max_frames) {
        if (!cursor.unlimited && cursor.frames_left == 0) {
            pass_done = true;
            break;
        }
        if (decoder.frame_pos >= decoder.frame_len) {
            const u32 before = cursor.byte_pos;
            const auto result = decoder.NextFrame(block.data, block.dataSize, cursor.byte_pos);
            decoded_bytes += cursor.byte_pos - before;
            if (result != Atrac9Decoder::Result::Frame) {
                if (result == Atrac9Decoder::Result::Error) {
                    decode_errors++;
                }
                pass_done = true;
                break;
            }
        }
        const u32 available = decoder.frame_len - decoder.frame_pos;
        if (cursor.skip_left != 0) {
            const u32 skipped = std::min(available, cursor.skip_left);
            decoder.frame_pos += skipped;
            cursor.skip_left -= skipped;
            continue;
        }
        u32 frames = std::min(available, max_frames - produced);
        if (!cursor.unlimited) {
            frames = std::min(frames, cursor.frames_left);
            cursor.frames_left -= frames;
        }
        std::memcpy(dst + static_cast<size_t>(produced) * channels,
                    decoder.frame.data() + static_cast<size_t>(decoder.frame_pos) * channels,
                    static_cast<size_t>(frames) * channels * sizeof(float));
        decoder.frame_pos += frames;
        produced += frames;
    }
    return produced;
}

u32 WaveformStream::PullSilence(float* dst, u32 max_frames, bool& pass_done) {
    const u32 frames = std::min(max_frames, cursor.frames_left);
    std::memset(dst, 0, static_cast<size_t>(frames) * channels * sizeof(float));
    cursor.frames_left -= frames;
    pass_done = cursor.frames_left == 0;
    return frames;
}

u32 WaveformStream::Pull(float* dst, u32 max_frames) {
    u32 produced = 0;
    while (produced < max_frames && !blocks.empty()) {
        if (!cursor.open) {
            StartPass(true);
        }
        bool pass_done = false;
        float* out = dst + static_cast<size_t>(produced) * channels;
        const u32 want = max_frames - produced;
        u32 frames = 0;
        switch (blocks.front().data ? codec : WaveformCodec::Unsupported) {
        case WaveformCodec::PcmI16:
        case WaveformCodec::PcmF32:
            frames = PullPcm(out, want, pass_done);
            break;
        case WaveformCodec::Atrac9:
            frames = PullAtrac9(out, want, pass_done);
            break;
        default:
            frames = PullSilence(out, want, pass_done);
            break;
        }
        produced += frames;
        cursor.pass_frames += frames;
        decoded_samples += frames;
        if (pass_done) {
            FinishPass();
        } else if (frames == 0) {
            break;
        }
    }
    return produced;
}

} // namespace Libraries::Ngs2
