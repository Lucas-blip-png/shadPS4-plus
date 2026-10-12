// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/ngs2/ngs2_custom.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "core/libraries/ngs2/ngs2_mastering.h"
#include "core/libraries/ngs2/ngs2_mixer.h"
#include "core/libraries/ngs2/ngs2_reverb.h"
#include "core/libraries/ngs2/ngs2_sampler.h"
#include "core/libraries/ngs2/ngs2_submixer.h"
#include "core/libraries/ngs2/ngs2_waveform.h"

namespace Libraries::Ngs2::Mixer {

namespace {

constexpr u32 MAX_CHANNELS = ORBIS_NGS2_MAX_VOICE_CHANNELS;
constexpr u32 MAX_MATRIX_LEVELS = ORBIS_NGS2_MAX_MATRIX_LEVELS;
constexpr u32 MAX_PORTS = 16;
constexpr u32 MAX_MATRICES = 8;
constexpr u32 MAX_VOICES_PER_RACK = 1024;
constexpr u32 MAX_PARAMS_PER_CALL = 64;
constexpr u32 MAX_PARAM_STRIDE = 4096;
constexpr u32 MAX_ENVELOPE_POINTS = 64;
constexpr u32 MAX_RENDER_BUFFERS = 16;
constexpr u32 MAX_STATE_SIZE = 256;
constexpr u32 MAX_USER_COUNTERS = 64;
constexpr double MAX_RESAMPLE_STEP = 8.0;
constexpr float MAX_PITCH_RATIO = 4.0f;
constexpr float MAX_ENVELOPE_HEIGHT = 8.0f;
constexpr u64 SUMMARY_PERIOD_US = 5'000'000;

constexpr u32 STATE_FLAG_INUSE = 0x01;
constexpr u32 STATE_FLAG_PLAYING = 0x02;
constexpr u32 STATE_FLAG_PAUSED = 0x04;
constexpr u32 STATE_FLAG_STOPPED = 0x08;
constexpr u32 STATE_FLAG_EMPTY = 0x20;

constexpr u32 RACK_ID_SAMPLER = 0x1000;
constexpr u32 RACK_ID_SUBMIXER = 0x2000;
constexpr u32 RACK_ID_REVERB = 0x2001;
constexpr u32 RACK_ID_MASTERING = 0x3000;
constexpr u32 RACK_ID_CUSTOM_SAMPLER = 0x4001;
constexpr u32 RACK_ID_CUSTOM_SUBMIXER = 0x4002;
constexpr u32 CUSTOM_MODULE_CLASS = 0x4000;
constexpr u32 CUSTOM_MODULE_DELAY = 0x1d;

constexpr u32 WAVEFORM_OUTPUT_PCM_I16 = NGS2_WAVEFORM_TYPE_PCM_I16;
constexpr u32 WAVEFORM_OUTPUT_PCM_F32 = NGS2_WAVEFORM_TYPE_PCM_F32;

enum class RackKind : u8 { Sampler, Submixer, Reverb, Mastering, Custom, Unknown };
enum class VoiceState : u8 { Free, Idle, Playing, Paused, Stopping };

enum VoiceEvent : u32 {
    EVENT_PLAY = 0,
    EVENT_STOP = 1,
    EVENT_STOP_IMMEDIATE = 2,
    EVENT_KILL = 3,
    EVENT_PAUSE = 4,
    EVENT_RESUME = 5,
};

enum CommonParamId : u32 {
    PARAM_MATRIX_LEVELS = 1,
    PARAM_PORT_VOLUME = 2,
    PARAM_PORT_MATRIX = 3,
    PARAM_PORT_DELAY = 4,
    PARAM_PATCH = 5,
    PARAM_EVENT = 6,
    PARAM_CALLBACK = 7,
};

enum SamplerParamIndex : u32 {
    SAMPLER_SETUP = 0,
    SAMPLER_WAVEFORM_BLOCKS = 1,
    SAMPLER_EXIT_LOOP = 4,
    SAMPLER_PITCH = 5,
    SAMPLER_ENVELOPE = 6,
};

enum SubmixerParamIndex : u32 { SUBMIXER_SETUP = 0 };
enum ReverbParamIndex : u32 { REVERB_SETUP = 0, REVERB_I3DL2 = 1 };
enum MasteringParamIndex : u32 {
    MASTERING_SETUP = 0,
    MASTERING_LIMITER = 3,
    MASTERING_GAIN = 4,
    MASTERING_OUTPUT = 5,
};
enum CustomParamIndex : u32 { CUSTOM_SETUP = 0 };

struct Voice;
struct Rack;
struct System;

struct Matrix {
    u32 num_levels{};
    float levels[ORBIS_NGS2_MAX_MATRIX_LEVELS]{};
};

struct Port {
    Voice* dest{};
    u32 dest_input{};
    float volume{1.0f};
    s32 matrix_id{-1};
};

struct SamplerData {
    WaveformStream stream;
    float pitch{1.0f};
    u32 source_rate{};
    std::vector<OrbisNgs2EnvelopePoint> forward;
    std::vector<OrbisNgs2EnvelopePoint> release;
    float env_level{1.0f};
    float env_start{1.0f};
    double env_progress{};
    size_t env_point{};
    bool releasing{};
    std::vector<float> fifo;
    u32 fifo_frames{};
    double phase{};
};

struct BusData {
    std::vector<float> acc;
    u32 channels{MAX_CHANNELS};
    bool has_signal{};
    float dry_level{1.0f};
    u32 output_id{};
    float fbw_level{1.0f};
    float lfe_level{1.0f};
};

struct Voice {
    Rack* rack{};
    u32 index{};
    VoiceState state{VoiceState::Free};
    u32 num_ports{};
    std::array<Port, MAX_PORTS> ports{};
    std::vector<Matrix> matrices;
    std::unique_ptr<SamplerData> sampler;
    std::unique_ptr<BusData> bus;
};

struct Rack {
    u32 serial{};
    u32 rack_id{};
    RackKind kind{RackKind::Unknown};
    System* system{};
    char name[ORBIS_NGS2_RACK_NAME_LENGTH]{};
    u32 max_voices{};
    u32 max_matrices{};
    u32 max_ports{};
    uintptr_t user_data{};
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2BufferFreeHandler free_handler{};
    std::vector<Voice> voices;
};

struct Stats {
    u64 grains{};
    u64 grains_total{};
    u64 decode_errors{};
    u64 decode_errors_total{};
    u64 invalid_params{};
    u64 invalid_params_total{};
    u64 last_summary_us{};
    std::map<u32, u64> unhandled_ids;
    std::map<u32, u64> events;
    std::map<std::tuple<u32, u32, u32, u32>, u64> formats;
    std::map<std::pair<u32, u32>, u64> state_sizes;
    std::array<float, MAX_RENDER_BUFFERS> out_peak{};
};

struct System {
    std::mutex mutex;
    u32 serial{};
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH]{};
    u32 sample_rate{48000};
    u32 num_grain_samples{256};
    u32 max_grain_samples{512};
    uintptr_t user_data{};
    OrbisNgs2ContextBufferInfo buffer_info{};
    OrbisNgs2BufferFreeHandler free_handler{};
    bool destroyed{};
    std::map<u32, std::unique_ptr<Rack>> racks;
    std::vector<Voice*> bus_order;
    bool order_dirty{true};
    std::vector<float> sampler_out;
    std::array<std::vector<float>, MAX_RENDER_BUFFERS> output_scratch;
    std::vector<s16> pcm16_scratch;
    Stats stats;
};

struct Registry {
    std::mutex mutex;
    std::unordered_map<u32, std::shared_ptr<System>> systems;
    std::unordered_map<u32, std::shared_ptr<System>> rack_owner;
    u32 next_system_serial{1};
    u32 next_rack_serial{1};
};

Registry& GetRegistry() {
    static Registry registry;
    return registry;
}

// Handles are opaque to the game. System: tag | serial << 4. Rack: serial << 24.
// Voice: rack | (index + 1) << 4. Low nibble stays clear like a real aligned pointer.
constexpr OrbisNgs2Handle SYSTEM_HANDLE_TAG = OrbisNgs2Handle{1} << 48;
constexpr u32 SERIAL_MASK = 0xFFFFFF;
constexpr u32 VOICE_FIELD_MASK = 0xFFFFF;

OrbisNgs2Handle MakeSystemHandle(u32 serial) {
    return SYSTEM_HANDLE_TAG | (static_cast<OrbisNgs2Handle>(serial) << 4);
}

OrbisNgs2Handle MakeRackHandle(u32 serial) {
    return static_cast<OrbisNgs2Handle>(serial) << 24;
}

OrbisNgs2Handle MakeVoiceHandle(u32 serial, u32 index) {
    return MakeRackHandle(serial) | (static_cast<OrbisNgs2Handle>(index + 1) << 4);
}

enum class HandleKind : u8 { Invalid, System, Rack, Voice };

struct HandleParts {
    HandleKind kind{HandleKind::Invalid};
    u32 serial{};
    u32 voice_index{};
};

HandleParts DecodeHandle(OrbisNgs2Handle handle) {
    if (handle == 0 || (handle & 0xF) != 0) {
        return {};
    }
    if (handle & SYSTEM_HANDLE_TAG) {
        const OrbisNgs2Handle rest = handle & ~SYSTEM_HANDLE_TAG;
        if (rest >> 28 != 0) {
            return {};
        }
        return {HandleKind::System, static_cast<u32>((rest >> 4) & SERIAL_MASK), 0};
    }
    if (handle >> 48 != 0) {
        return {};
    }
    const u32 serial = static_cast<u32>((handle >> 24) & SERIAL_MASK);
    const u32 voice_field = static_cast<u32>((handle >> 4) & VOICE_FIELD_MASK);
    if (serial == 0) {
        return {};
    }
    if (voice_field == 0) {
        return {HandleKind::Rack, serial, 0};
    }
    return {HandleKind::Voice, serial, voice_field - 1};
}

u64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool IsGuestRangeValid(const void* ptr, u64 size) {
    constexpr u64 min_address = 0x10000;
    constexpr u64 max_address = 0x800000000000ull;
    const u64 address = reinterpret_cast<u64>(ptr);
    return address >= min_address && address < max_address && size <= max_address - address;
}

RackKind KindFromRackId(u32 rack_id) {
    switch (rack_id) {
    case RACK_ID_SAMPLER:
    case RACK_ID_CUSTOM_SAMPLER:
        return RackKind::Sampler;
    case RACK_ID_SUBMIXER:
        return RackKind::Submixer;
    case RACK_ID_REVERB:
        return RackKind::Reverb;
    case RACK_ID_MASTERING:
        return RackKind::Mastering;
    case RACK_ID_CUSTOM_SUBMIXER:
        return RackKind::Custom;
    default:
        return RackKind::Unknown;
    }
}

bool HasBus(RackKind kind) {
    return kind == RackKind::Submixer || kind == RackKind::Reverb || kind == RackKind::Mastering ||
           kind == RackKind::Custom;
}

const char* KindName(RackKind kind) {
    switch (kind) {
    case RackKind::Sampler:
        return "sampler";
    case RackKind::Submixer:
        return "submixer";
    case RackKind::Reverb:
        return "reverb";
    case RackKind::Mastering:
        return "mastering";
    case RackKind::Custom:
        return "custom";
    default:
        return "unknown";
    }
}

bool IsSupportedSampleRate(u32 rate) {
    return rate == 11025 || rate == 12000 || rate == 22050 || rate == 24000 || rate == 44100 ||
           rate == 48000 || rate == 88200 || rate == 96000 || rate == 176400 || rate == 192000;
}

bool IsGrainSamplesValid(u32 samples) {
    return samples >= 64 && samples <= 1024 && (samples & 63) == 0;
}

std::shared_ptr<System> FindSystem(OrbisNgs2Handle handle) {
    const HandleParts parts = DecodeHandle(handle);
    if (parts.kind != HandleKind::System) {
        return nullptr;
    }
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    const auto it = registry.systems.find(parts.serial);
    return it == registry.systems.end() ? nullptr : it->second;
}

std::shared_ptr<System> FindSystemOfRack(u32 rack_serial) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    const auto it = registry.rack_owner.find(rack_serial);
    return it == registry.rack_owner.end() ? nullptr : it->second;
}

Rack* FindRack(System& sys, u32 serial) {
    const auto it = sys.racks.find(serial);
    return it == sys.racks.end() ? nullptr : it->second.get();
}

// A rack handle used as a patch destination stands for the first voice of that rack
Voice* FindVoice(System& sys, OrbisNgs2Handle handle, bool accept_rack) {
    const HandleParts parts = DecodeHandle(handle);
    if (parts.kind != HandleKind::Voice && !(accept_rack && parts.kind == HandleKind::Rack)) {
        return nullptr;
    }
    Rack* rack = FindRack(sys, parts.serial);
    if (!rack || parts.voice_index >= rack->voices.size()) {
        return nullptr;
    }
    return &rack->voices[parts.voice_index];
}

void NoteInvalid(System& sys) {
    sys.stats.invalid_params++;
}

void NoteUnhandled(System& sys, u32 id) {
    auto& ids = sys.stats.unhandled_ids;
    if (ids.size() < MAX_USER_COUNTERS || ids.count(id)) {
        ids[id]++;
    }
}

float SanitizeFloat(float value, float low, float high, float fallback) {
    if (!std::isfinite(value)) {
        return fallback;
    }
    return std::clamp(value, low, high);
}

void ResetRouting(System& sys, Voice& voice) {
    voice.ports.fill(Port{});
    for (Matrix& matrix : voice.matrices) {
        matrix.num_levels = 0;
    }
    sys.order_dirty = true;
}

// Envelope heights ramp linearly between points; durations are milliseconds
void AdvanceEnvelope(SamplerData& s, u32 sample_rate, double samples) {
    const auto& points = s.releasing ? s.release : s.forward;
    const double samples_per_ms = static_cast<double>(sample_rate) / 1000.0;
    while (s.env_point < points.size()) {
        const OrbisNgs2EnvelopePoint& point = points[s.env_point];
        const double duration = static_cast<double>(point.duration) * samples_per_ms;
        const double remaining = duration - s.env_progress;
        if (remaining <= samples) {
            samples -= remaining;
            s.env_level = point.height;
            s.env_start = point.height;
            s.env_progress = 0.0;
            s.env_point++;
        } else {
            s.env_progress += samples;
            s.env_level = s.env_start + (point.height - s.env_start) *
                                            static_cast<float>(s.env_progress / duration);
            break;
        }
    }
}

void StartEnvelope(SamplerData& s, u32 sample_rate) {
    s.releasing = false;
    s.env_level = s.forward.empty() ? 1.0f : 0.0f;
    s.env_start = s.env_level;
    s.env_progress = 0.0;
    s.env_point = 0;
    AdvanceEnvelope(s, sample_rate, 0.0);
}

void StartRelease(SamplerData& s) {
    s.releasing = true;
    s.env_start = s.env_level;
    s.env_progress = 0.0;
    s.env_point = 0;
}

void ResetSamplerPlayback(SamplerData& s) {
    s.stream.Clear();
    s.fifo_frames = 0;
    s.phase = 0.0;
    s.releasing = false;
    s.env_level = 1.0f;
    s.env_start = 1.0f;
    s.env_progress = 0.0;
    s.env_point = 0;
}

void ReleaseVoice(Voice& voice) {
    if (voice.sampler) {
        ResetSamplerPlayback(*voice.sampler);
    }
    voice.state = VoiceState::Free;
}

// A voice that ran out of audio keeps its stream so the last block stays visible in its state
void FinishVoice(Voice& voice) {
    SamplerData& s = *voice.sampler;
    s.fifo_frames = 0;
    s.phase = 0.0;
    s.releasing = false;
    s.env_level = 1.0f;
    voice.state = VoiceState::Free;
}

void HandleEvent(System& sys, Voice& voice, u32 event_id) {
    auto& events = sys.stats.events;
    if (events.size() < MAX_USER_COUNTERS || events.count(event_id)) {
        events[event_id]++;
    }
    SamplerData* sampler = voice.sampler.get();
    const bool is_bus = voice.bus != nullptr;
    switch (event_id) {
    case EVENT_PLAY:
        if (sampler && (voice.state == VoiceState::Idle || voice.state == VoiceState::Free) &&
            sampler->stream.Configured()) {
            StartEnvelope(*sampler, sys.sample_rate);
            voice.state = VoiceState::Playing;
        } else if (is_bus && voice.state == VoiceState::Idle) {
            voice.state = VoiceState::Playing;
        }
        break;
    case EVENT_STOP:
        if (sampler && (voice.state == VoiceState::Playing || voice.state == VoiceState::Paused)) {
            StartRelease(*sampler);
            voice.state = VoiceState::Stopping;
        } else if (is_bus && voice.state != VoiceState::Free) {
            voice.state = VoiceState::Idle;
        }
        break;
    case EVENT_STOP_IMMEDIATE:
    case EVENT_KILL:
        if (sampler) {
            ReleaseVoice(voice);
        } else if (is_bus && voice.state != VoiceState::Free) {
            voice.state = event_id == EVENT_KILL ? VoiceState::Free : VoiceState::Idle;
        }
        break;
    case EVENT_PAUSE:
        if (voice.state == VoiceState::Playing) {
            voice.state = VoiceState::Paused;
        }
        break;
    case EVENT_RESUME:
        if (voice.state == VoiceState::Paused) {
            voice.state = VoiceState::Playing;
        }
        break;
    default:
        NoteInvalid(sys);
        break;
    }
}

template <class T>
const T* ParamAs(System& sys, const OrbisNgs2VoiceParamHeader& header) {
    if (header.size < sizeof(T)) {
        NoteInvalid(sys);
        return nullptr;
    }
    return reinterpret_cast<const T*>(&header);
}

void ApplyMatrixLevels(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    const auto* param = ParamAs<OrbisNgs2VoiceMatrixLevelsParam>(sys, header);
    if (!param) {
        return;
    }
    if (param->matrixId >= voice.matrices.size()) {
        NoteInvalid(sys);
        return;
    }
    Matrix& matrix = voice.matrices[param->matrixId];
    const u32 count = std::min(param->numLevels, MAX_MATRIX_LEVELS);
    if (count == 0) {
        matrix.num_levels = 0;
        return;
    }
    if (!IsGuestRangeValid(param->aLevel, static_cast<u64>(count) * sizeof(float))) {
        NoteInvalid(sys);
        return;
    }
    std::memcpy(matrix.levels, param->aLevel, count * sizeof(float));
    for (u32 i = 0; i < count; i++) {
        matrix.levels[i] = SanitizeFloat(matrix.levels[i], -16.0f, 16.0f, 0.0f);
    }
    matrix.num_levels = count;
}

void ApplyPatch(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    const auto* param = ParamAs<OrbisNgs2VoicePatchParam>(sys, header);
    if (!param) {
        return;
    }
    if (param->port >= voice.num_ports) {
        NoteInvalid(sys);
        return;
    }
    Voice* dest = nullptr;
    if (param->destHandle != 0) {
        dest = FindVoice(sys, param->destHandle, true);
        if (!dest || !dest->bus || dest == &voice) {
            NoteInvalid(sys);
            return;
        }
    }
    Port& port = voice.ports[param->port];
    port.dest = dest;
    port.dest_input = param->destInputId;
    sys.order_dirty = true;
}

void ApplyCommon(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    switch (header.id) {
    case PARAM_MATRIX_LEVELS:
        ApplyMatrixLevels(sys, voice, header);
        break;
    case PARAM_PORT_VOLUME:
        if (const auto* param = ParamAs<OrbisNgs2VoicePortVolumeParam>(sys, header)) {
            if (param->port < voice.num_ports) {
                voice.ports[param->port].volume = SanitizeFloat(param->level, 0.0f, 16.0f, 0.0f);
            } else {
                NoteInvalid(sys);
            }
        }
        break;
    case PARAM_PORT_MATRIX:
        if (const auto* param = ParamAs<OrbisNgs2VoicePortMatrixParam>(sys, header)) {
            if (param->port < voice.num_ports &&
                param->matrixId < static_cast<s32>(voice.matrices.size())) {
                voice.ports[param->port].matrix_id = param->matrixId < 0 ? -1 : param->matrixId;
            } else {
                NoteInvalid(sys);
            }
        }
        break;
    case PARAM_PATCH:
        ApplyPatch(sys, voice, header);
        break;
    case PARAM_EVENT:
        if (const auto* param = ParamAs<OrbisNgs2VoiceEventParam>(sys, header)) {
            HandleEvent(sys, voice, param->eventId);
        }
        break;
    default:
        NoteUnhandled(sys, header.id);
        break;
    }
}

void ApplySamplerEnvelope(System& sys, SamplerData& sampler,
                          const OrbisNgs2VoiceParamHeader& header) {
    const auto* param = ParamAs<OrbisNgs2SamplerVoiceEnvelopeParam>(sys, header);
    if (!param) {
        return;
    }
    const u64 total = static_cast<u64>(param->numForwardPoints) + param->numReleasePoints;
    if (total > MAX_ENVELOPE_POINTS ||
        (total != 0 && !IsGuestRangeValid(param->aPoint, total * sizeof(OrbisNgs2EnvelopePoint)))) {
        NoteInvalid(sys);
        return;
    }
    std::vector<OrbisNgs2EnvelopePoint> points(total);
    if (total != 0) {
        std::memcpy(points.data(), param->aPoint, total * sizeof(OrbisNgs2EnvelopePoint));
    }
    for (OrbisNgs2EnvelopePoint& point : points) {
        point.height = SanitizeFloat(point.height, 0.0f, MAX_ENVELOPE_HEIGHT, 1.0f);
    }
    sampler.forward.assign(points.begin(), points.begin() + param->numForwardPoints);
    sampler.release.assign(points.begin() + param->numForwardPoints, points.end());
}

void ApplyWaveformBlocks(System& sys, SamplerData& sampler,
                         const OrbisNgs2VoiceParamHeader& header) {
    const auto* param = ParamAs<OrbisNgs2SamplerVoiceWaveformBlocksParam>(sys, header);
    if (!param) {
        return;
    }
    const u32 count = std::min(param->numBlocks, NGS2_WAVEFORM_QUEUE_MAX);
    if (param->numBlocks > count) {
        NoteInvalid(sys);
    }
    const u64 blocks_size = static_cast<u64>(count) * sizeof(OrbisNgs2WaveformBlock);
    if (count != 0 && !IsGuestRangeValid(param->aBlock, blocks_size)) {
        NoteInvalid(sys);
        return;
    }
    std::vector<WaveformBlockEntry> entries;
    entries.reserve(count);
    for (u32 i = 0; i < count; i++) {
        const OrbisNgs2WaveformBlock& block = param->aBlock[i];
        const u8* base = static_cast<const u8*>(param->data);
        if (!base || block.dataSize == 0 ||
            !IsGuestRangeValid(base + block.dataOffset, block.dataSize)) {
            NoteInvalid(sys);
            continue;
        }
        entries.push_back({base + block.dataOffset, block.dataSize, block.numRepeats,
                           block.numSkipSamples, block.numSamples, block.userData, false});
    }
    sampler.stream.AddBlocks(entries.data(), static_cast<u32>(entries.size()), param->flags);
}

void ApplySamplerSetup(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    constexpr u32 min_size = sizeof(OrbisNgs2VoiceParamHeader) + sizeof(OrbisNgs2WaveformFormat);
    if (header.size < min_size) {
        NoteInvalid(sys);
        return;
    }
    SamplerData& sampler = *voice.sampler;
    OrbisNgs2WaveformFormat format;
    std::memcpy(&format, &header + 1, sizeof(format));
    ResetSamplerPlayback(sampler);
    sampler.stream.Configure(format);
    sampler.source_rate = format.sampleRate;
    sampler.pitch = 1.0f;
    sampler.forward.clear();
    sampler.release.clear();
    sys.stats.decode_errors += sampler.stream.TakeDecodeErrors();
    ResetRouting(sys, voice);
    voice.state = VoiceState::Idle;

    auto& formats = sys.stats.formats;
    const auto key = std::make_tuple(format.waveformType, format.numChannels, format.sampleRate,
                                     format.configData >> 24);
    if (formats.size() < MAX_USER_COUNTERS || formats.count(key)) {
        formats[key]++;
    }
}

void ApplySampler(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    SamplerData& sampler = *voice.sampler;
    switch (header.id & 0xFFFF) {
    case SAMPLER_SETUP:
        ApplySamplerSetup(sys, voice, header);
        break;
    case SAMPLER_WAVEFORM_BLOCKS:
        ApplyWaveformBlocks(sys, sampler, header);
        break;
    case SAMPLER_EXIT_LOOP:
        sampler.stream.ExitLoop();
        break;
    case SAMPLER_PITCH:
        if (const auto* param = ParamAs<OrbisNgs2SamplerVoicePitchParam>(sys, header)) {
            sampler.pitch = SanitizeFloat(param->ratio, 0.0f, MAX_PITCH_RATIO, 1.0f);
        }
        break;
    case SAMPLER_ENVELOPE:
        ApplySamplerEnvelope(sys, sampler, header);
        break;
    default:
        NoteUnhandled(sys, header.id);
        break;
    }
}

void SetupBus(System& sys, Voice& voice, u32 channels) {
    voice.bus->channels = std::clamp<u32>(channels, 1, MAX_CHANNELS);
    voice.bus->has_signal = false;
    voice.bus->dry_level = 1.0f;
    voice.bus->output_id = 0;
    voice.bus->fbw_level = 1.0f;
    voice.bus->lfe_level = 1.0f;
    std::fill(voice.bus->acc.begin(), voice.bus->acc.end(), 0.0f);
    ResetRouting(sys, voice);
    voice.state = VoiceState::Playing;
}

void ApplySubmixer(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    if ((header.id & 0xFFFF) == SUBMIXER_SETUP) {
        if (const auto* param = ParamAs<OrbisNgs2SubmixerVoiceSetupParam>(sys, header)) {
            SetupBus(sys, voice, param->numIoChannels);
        }
        return;
    }
    NoteUnhandled(sys, header.id);
}

void ApplyReverb(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    switch (header.id & 0xFFFF) {
    case REVERB_SETUP:
        if (const auto* param = ParamAs<OrbisNgs2ReverbVoiceSetupParam>(sys, header)) {
            SetupBus(sys, voice, param->numInputChannels);
        }
        break;
    case REVERB_I3DL2:
        if (const auto* param = ParamAs<OrbisNgs2ReverbVoiceI3DL2Param>(sys, header)) {
            voice.bus->dry_level = SanitizeFloat(param->i3dl2.dry, 0.0f, 4.0f, 1.0f);
        }
        break;
    default:
        NoteUnhandled(sys, header.id);
        break;
    }
}

void ApplyMastering(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    BusData& bus = *voice.bus;
    switch (header.id & 0xFFFF) {
    case MASTERING_SETUP:
        if (const auto* param = ParamAs<OrbisNgs2MasteringVoiceSetupParam>(sys, header)) {
            SetupBus(sys, voice, param->numInputChannels);
        }
        break;
    case MASTERING_GAIN:
        if (const auto* param = ParamAs<OrbisNgs2MasteringVoiceGainParam>(sys, header)) {
            bus.fbw_level = SanitizeFloat(param->fbwLevel, 0.0f, 16.0f, 1.0f);
            bus.lfe_level = SanitizeFloat(param->lfeLevel, 0.0f, 16.0f, 1.0f);
        }
        break;
    case MASTERING_OUTPUT:
        if (const auto* param = ParamAs<OrbisNgs2MasteringVoiceOutputParam>(sys, header)) {
            bus.output_id = param->outputId;
        }
        break;
    case MASTERING_LIMITER:
        // The output stage always clips to full scale
        break;
    default:
        NoteUnhandled(sys, header.id);
        break;
    }
}

void ApplyCustom(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    const u32 klass = header.id >> 16;
    if (klass == CUSTOM_MODULE_CLASS) {
        const u32 module_id = (header.id >> 8) & 0xFF;
        if (module_id == CUSTOM_MODULE_DELAY) {
            if (const auto* param = ParamAs<OrbisNgs2CustomVoiceDelayParam>(sys, header)) {
                voice.bus->dry_level = SanitizeFloat(param->dryLevel, 0.0f, 4.0f, 1.0f);
            }
            return;
        }
        NoteUnhandled(sys, header.id);
        return;
    }
    if ((header.id & 0xFFFF) == CUSTOM_SETUP) {
        if (const auto* param = ParamAs<OrbisNgs2CustomSubmixerVoiceSetupParam>(sys, header)) {
            SetupBus(sys, voice, param->numInputChannels);
        }
        return;
    }
    NoteUnhandled(sys, header.id);
}

void ApplyParam(System& sys, Voice& voice, const OrbisNgs2VoiceParamHeader& header) {
    if (header.id < 0x10000) {
        ApplyCommon(sys, voice, header);
        return;
    }
    const u32 klass = header.id >> 16;
    const Rack& rack = *voice.rack;
    const bool class_matches =
        klass == rack.rack_id || (rack.kind == RackKind::Custom && klass == CUSTOM_MODULE_CLASS);
    if (!class_matches) {
        NoteUnhandled(sys, header.id);
        return;
    }
    switch (rack.kind) {
    case RackKind::Sampler:
        ApplySampler(sys, voice, header);
        break;
    case RackKind::Submixer:
        ApplySubmixer(sys, voice, header);
        break;
    case RackKind::Reverb:
        ApplyReverb(sys, voice, header);
        break;
    case RackKind::Mastering:
        ApplyMastering(sys, voice, header);
        break;
    case RackKind::Custom:
        ApplyCustom(sys, voice, header);
        break;
    default:
        NoteUnhandled(sys, header.id);
        break;
    }
}

// Default channel routing used when a port has no matrix (or an empty one)
float DefaultLevel(u32 src_channels, u32 dst_channels, u32 src, u32 dst) {
    if (src_channels == dst_channels) {
        return src == dst ? 1.0f : 0.0f;
    }
    if (dst_channels == 1) {
        if (src_channels == 2) {
            return 0.707f;
        }
        constexpr float downmix[MAX_CHANNELS] = {0.707f, 0.707f, 1.0f, 0.0f,
                                                 0.5f,   0.5f,   0.5f, 0.5f};
        return downmix[src];
    }
    if (src_channels == 1) {
        if (dst_channels == 2) {
            return 0.707f;
        }
        return dst == 2 ? 1.0f : 0.0f;
    }
    return src == dst ? 1.0f : 0.0f;
}

using Coefficients = std::array<std::array<float, MAX_CHANNELS>, MAX_CHANNELS>;

void BuildCoefficients(const Voice& source, const Port& port, u32 src_channels, u32 dst_channels,
                       Coefficients& coef) {
    const Matrix* matrix = nullptr;
    if (port.matrix_id >= 0 && static_cast<size_t>(port.matrix_id) < source.matrices.size() &&
        source.matrices[port.matrix_id].num_levels != 0) {
        matrix = &source.matrices[port.matrix_id];
    }
    for (u32 d = 0; d < dst_channels; d++) {
        for (u32 s = 0; s < src_channels; s++) {
            if (matrix) {
                const u32 index = d * src_channels + s;
                coef[d][s] = index < matrix->num_levels ? matrix->levels[index] : 0.0f;
            } else {
                coef[d][s] = DefaultLevel(src_channels, dst_channels, s, d);
            }
        }
    }
}

void MixToDestination(const System& sys, const Voice& source, const Port& port,
                      const float* src_data, u32 src_channels, u32 grain) {
    BusData& dest = *port.dest->bus;
    if (port.dest->state != VoiceState::Playing || port.volume == 0.0f) {
        return;
    }
    const u32 dst_channels = dest.channels;
    Coefficients coef;
    BuildCoefficients(source, port, std::min(src_channels, MAX_CHANNELS), dst_channels, coef);
    const size_t stride = sys.max_grain_samples;
    for (u32 d = 0; d < dst_channels; d++) {
        float* out = dest.acc.data() + d * stride;
        for (u32 s = 0; s < src_channels && s < MAX_CHANNELS; s++) {
            const float gain = coef[d][s] * port.volume;
            if (gain == 0.0f) {
                continue;
            }
            const float* in = src_data + static_cast<size_t>(s) * grain;
            for (u32 i = 0; i < grain; i++) {
                out[i] += in[i] * gain;
            }
        }
    }
    dest.has_signal = true;
}

void RebuildBusOrder(System& sys) {
    sys.bus_order.clear();
    std::vector<Voice*> nodes;
    std::unordered_map<const Voice*, u32> in_degree;
    for (auto& [serial, rack] : sys.racks) {
        for (Voice& voice : rack->voices) {
            if (voice.bus) {
                nodes.push_back(&voice);
                in_degree[&voice] = 0;
            }
        }
    }
    for (Voice* node : nodes) {
        for (u32 p = 0; p < node->num_ports; p++) {
            if (node->ports[p].dest) {
                in_degree[node->ports[p].dest]++;
            }
        }
    }
    std::vector<Voice*> ready;
    for (Voice* node : nodes) {
        if (in_degree[node] == 0) {
            ready.push_back(node);
        }
    }
    std::unordered_map<const Voice*, bool> placed;
    for (size_t head = 0; head < ready.size(); head++) {
        Voice* node = ready[head];
        sys.bus_order.push_back(node);
        placed[node] = true;
        for (u32 p = 0; p < node->num_ports; p++) {
            Voice* dest = node->ports[p].dest;
            if (dest && --in_degree[dest] == 0) {
                ready.push_back(dest);
            }
        }
    }
    // Voices on a patch cycle never reach in-degree zero; keep them rendering anyway
    for (Voice* node : nodes) {
        if (!placed[node]) {
            sys.bus_order.push_back(node);
        }
    }
    sys.order_dirty = false;
}

// Returns false once the voice has run out of audio and must be released
bool RenderSamplerVoice(System& sys, Voice& voice, u32& out_channels) {
    SamplerData& s = *voice.sampler;
    const u32 grain = sys.num_grain_samples;
    out_channels = 0;
    if (voice.state == VoiceState::Paused || !s.stream.Configured()) {
        return true;
    }
    if (voice.state == VoiceState::Stopping && s.release.empty()) {
        return false;
    }

    const u32 channels = s.stream.NumChannels();
    const double source_rate = s.source_rate ? s.source_rate : sys.sample_rate;
    const double step = std::clamp(source_rate * s.pitch / sys.sample_rate, 0.0, MAX_RESAMPLE_STEP);

    const u32 needed = static_cast<u32>(s.phase + step * grain) + 2;
    if (s.fifo_frames < needed && !s.stream.Ended()) {
        const size_t needed_samples = static_cast<size_t>(needed) * channels;
        if (s.fifo.size() < needed_samples) {
            s.fifo.resize(needed_samples);
        }
        s.fifo_frames += s.stream.Pull(
            s.fifo.data() + static_cast<size_t>(s.fifo_frames) * channels, needed - s.fifo_frames);
    }
    const bool exhausted = s.stream.Ended();

    float* out = sys.sampler_out.data();
    std::fill_n(out, static_cast<size_t>(channels) * grain, 0.0f);
    double position = s.phase;
    for (u32 i = 0; i < grain; i++) {
        const u32 index = static_cast<u32>(position);
        if (index >= s.fifo_frames) {
            break;
        }
        const bool has_next = index + 1 < s.fifo_frames;
        if (!has_next && !exhausted) {
            break;
        }
        const float fraction = static_cast<float>(position - index);
        const float* current = s.fifo.data() + static_cast<size_t>(index) * channels;
        const float* next = current + channels;
        for (u32 c = 0; c < channels; c++) {
            const float a = current[c];
            const float b = has_next ? next[c] : 0.0f;
            out[static_cast<size_t>(c) * grain + i] = a + (b - a) * fraction;
        }
        position += step;
    }

    const u32 consumed = std::min(static_cast<u32>(position), s.fifo_frames);
    const bool drained = exhausted && consumed >= s.fifo_frames;
    if (consumed != 0) {
        const u32 remaining = s.fifo_frames - consumed;
        std::memmove(s.fifo.data(), s.fifo.data() + static_cast<size_t>(consumed) * channels,
                     static_cast<size_t>(remaining) * channels * sizeof(float));
        s.fifo_frames = remaining;
        position -= consumed;
    }
    s.phase = position;

    const float start_gain = s.env_level;
    AdvanceEnvelope(s, sys.sample_rate, grain);
    const float end_gain = s.env_level;
    if (start_gain != 1.0f || end_gain != 1.0f) {
        const float delta = (end_gain - start_gain) / static_cast<float>(grain);
        for (u32 c = 0; c < channels; c++) {
            float* row = out + static_cast<size_t>(c) * grain;
            for (u32 i = 0; i < grain; i++) {
                row[i] *= start_gain + delta * static_cast<float>(i);
            }
        }
    }

    out_channels = channels;
    const bool release_done = s.releasing && s.env_point >= s.release.size();
    return !(drained || release_done);
}

struct OutputTarget {
    bool valid{};
    u32 channels{};
    u32 bytes_per_sample{};
};

void ProcessBusVoice(System& sys, Voice& voice,
                     const std::array<OutputTarget, MAX_RENDER_BUFFERS>& targets, u32 num_targets) {
    BusData& bus = *voice.bus;
    if (!bus.has_signal) {
        return;
    }
    const u32 grain = sys.num_grain_samples;
    const size_t stride = sys.max_grain_samples;
    bus.has_signal = false;
    if (voice.state == VoiceState::Playing) {
        const Rack& rack = *voice.rack;
        if (rack.kind == RackKind::Mastering) {
            if (bus.output_id < num_targets && targets[bus.output_id].valid) {
                const OutputTarget& target = targets[bus.output_id];
                std::vector<float>& scratch = sys.output_scratch[bus.output_id];
                for (u32 d = 0; d < target.channels; d++) {
                    for (u32 s = 0; s < bus.channels; s++) {
                        float gain = DefaultLevel(bus.channels, target.channels, s, d);
                        gain *= (s == 3 && bus.channels > 3) ? bus.lfe_level : bus.fbw_level;
                        if (gain == 0.0f) {
                            continue;
                        }
                        const float* in = bus.acc.data() + s * stride;
                        float* out = scratch.data() + d;
                        for (u32 i = 0; i < grain; i++) {
                            out[static_cast<size_t>(i) * target.channels] += in[i] * gain;
                        }
                    }
                }
            }
        } else {
            if (rack.kind == RackKind::Reverb || rack.kind == RackKind::Custom) {
                for (u32 c = 0; c < bus.channels; c++) {
                    float* row = bus.acc.data() + c * stride;
                    for (u32 i = 0; i < grain; i++) {
                        row[i] *= bus.dry_level;
                    }
                }
            }
            // Mix from a packed copy because channel rows are strided by the system capacity
            std::vector<float>& packed = sys.sampler_out;
            for (u32 c = 0; c < bus.channels; c++) {
                std::memcpy(packed.data() + static_cast<size_t>(c) * grain,
                            bus.acc.data() + c * stride, grain * sizeof(float));
            }
            for (u32 p = 0; p < voice.num_ports; p++) {
                const Port& port = voice.ports[p];
                if (port.dest) {
                    MixToDestination(sys, voice, port, packed.data(), bus.channels, grain);
                }
            }
        }
    }
    for (u32 c = 0; c < bus.channels; c++) {
        std::fill_n(bus.acc.data() + c * stride, grain, 0.0f);
    }
}

void FinalizeOutput(System& sys, u32 index, const OrbisNgs2RenderBufferInfo& info,
                    const OutputTarget& target) {
    const u32 grain = sys.num_grain_samples;
    std::vector<float>& scratch = sys.output_scratch[index];
    const size_t frame_bytes = static_cast<size_t>(target.channels) * target.bytes_per_sample;
    const u32 frames = static_cast<u32>(std::min<size_t>(grain, info.bufferSize / frame_bytes));
    const size_t samples = static_cast<size_t>(frames) * target.channels;

    float peak = 0.0f;
    for (size_t i = 0; i < samples; i++) {
        float value = scratch[i];
        if (!std::isfinite(value)) {
            value = 0.0f;
        }
        value = std::clamp(value, -1.0f, 1.0f);
        scratch[i] = value;
        peak = std::max(peak, std::fabs(value));
    }
    sys.stats.out_peak[index] = std::max(sys.stats.out_peak[index], peak);

    u8* dst = static_cast<u8*>(info.buffer);
    const size_t written_bytes = samples * target.bytes_per_sample;
    if (target.bytes_per_sample == sizeof(float)) {
        std::memcpy(dst, scratch.data(), written_bytes);
    } else {
        sys.pcm16_scratch.resize(samples);
        for (size_t i = 0; i < samples; i++) {
            sys.pcm16_scratch[i] = static_cast<s16>(std::lrintf(scratch[i] * 32767.0f));
        }
        std::memcpy(dst, sys.pcm16_scratch.data(), written_bytes);
    }
    if (info.bufferSize > written_bytes) {
        std::memset(dst + written_bytes, 0, info.bufferSize - written_bytes);
    }
}

std::string FormatCounts(const std::map<u32, u64>& counts) {
    std::string text;
    for (const auto& [id, count] : counts) {
        text += fmt::format(" {:#x}:{}", id, count);
    }
    return text;
}

void MaybeLogSummary(System& sys) {
    if (!Common::Log::ShouldLog(Common::Log::Class::Lib_Ngs2, Common::Log::Level::Info)) {
        return;
    }
    const u64 now = NowUs();
    if (sys.stats.last_summary_us == 0) {
        sys.stats.last_summary_us = now;
        return;
    }
    if (now - sys.stats.last_summary_us < SUMMARY_PERIOD_US) {
        return;
    }
    sys.stats.last_summary_us = now;

    std::array<u32, 6> active{};
    std::array<u32, 6> total{};
    for (const auto& [serial, rack] : sys.racks) {
        const size_t kind = static_cast<size_t>(rack->kind);
        for (const Voice& voice : rack->voices) {
            if (voice.state != VoiceState::Free) {
                total[kind]++;
            }
            if (voice.state == VoiceState::Playing || voice.state == VoiceState::Stopping) {
                active[kind]++;
            }
        }
    }
    std::string voices;
    for (size_t k = 0; k < active.size(); k++) {
        voices += fmt::format(" {}={}/{}", KindName(static_cast<RackKind>(k)), active[k], total[k]);
    }
    std::string peaks;
    for (size_t i = 0; i < 4; i++) {
        peaks += fmt::format(" {:.4f}", sys.stats.out_peak[i]);
    }
    std::string formats;
    for (const auto& [key, count] : sys.stats.formats) {
        formats += fmt::format(" type={:#x}/ch={}/rate={}/cfg={:#x}:{}", std::get<0>(key),
                               std::get<1>(key), std::get<2>(key), std::get<3>(key), count);
    }
    std::string state_sizes;
    for (const auto& [key, count] : sys.stats.state_sizes) {
        state_sizes += fmt::format(" rack={:#x}/size={}:{}", key.first, key.second, count);
    }

    sys.stats.decode_errors_total += sys.stats.decode_errors;
    sys.stats.invalid_params_total += sys.stats.invalid_params;
    LOG_INFO(Lib_Ngs2,
             "mixer: system#{} grains=+{} ({}) activeVoices{{{} }} outputPeaks[{} ] "
             "decodeErrors=+{} ({}) invalidParams=+{} ({}) unhandledParamIds{{{} }} "
             "events{{{} }} samplerFormats{{{} }} stateSizes{{{} }}",
             sys.serial, sys.stats.grains, sys.stats.grains_total, voices, peaks,
             sys.stats.decode_errors, sys.stats.decode_errors_total, sys.stats.invalid_params,
             sys.stats.invalid_params_total, FormatCounts(sys.stats.unhandled_ids),
             FormatCounts(sys.stats.events), formats, state_sizes);
    sys.stats.grains = 0;
    sys.stats.decode_errors = 0;
    sys.stats.invalid_params = 0;
    sys.stats.out_peak.fill(0.0f);
}

void RenderGrain(System& sys, const OrbisNgs2RenderBufferInfo* buffers, u32 num_buffers) {
    const u32 grain = sys.num_grain_samples;
    std::array<OutputTarget, MAX_RENDER_BUFFERS> targets{};
    for (u32 i = 0; i < num_buffers; i++) {
        const OrbisNgs2RenderBufferInfo& info = buffers[i];
        if (!info.buffer || info.bufferSize == 0 ||
            !IsGuestRangeValid(info.buffer, info.bufferSize)) {
            continue;
        }
        const u32 bytes = info.waveformType == WAVEFORM_OUTPUT_PCM_F32   ? 4
                          : info.waveformType == WAVEFORM_OUTPUT_PCM_I16 ? 2
                                                                         : 0;
        if (bytes == 0 || info.numChannels == 0 || info.numChannels > MAX_CHANNELS) {
            std::memset(info.buffer, 0, info.bufferSize);
            continue;
        }
        targets[i] = {true, info.numChannels, bytes};
        sys.output_scratch[i].assign(static_cast<size_t>(grain) * info.numChannels, 0.0f);
    }

    if (sys.order_dirty) {
        RebuildBusOrder(sys);
    }

    for (auto& [serial, rack] : sys.racks) {
        if (rack->kind != RackKind::Sampler) {
            continue;
        }
        for (Voice& voice : rack->voices) {
            if (voice.state != VoiceState::Playing && voice.state != VoiceState::Stopping) {
                continue;
            }
            u32 channels = 0;
            const bool alive = RenderSamplerVoice(sys, voice, channels);
            sys.stats.decode_errors += voice.sampler->stream.TakeDecodeErrors();
            if (channels != 0) {
                for (u32 p = 0; p < voice.num_ports; p++) {
                    const Port& port = voice.ports[p];
                    if (port.dest) {
                        MixToDestination(sys, voice, port, sys.sampler_out.data(), channels, grain);
                    }
                }
            }
            if (!alive) {
                FinishVoice(voice);
            }
        }
    }

    for (Voice* voice : sys.bus_order) {
        ProcessBusVoice(sys, *voice, targets, num_buffers);
    }

    for (u32 i = 0; i < num_buffers; i++) {
        if (targets[i].valid) {
            FinalizeOutput(sys, i, buffers[i], targets[i]);
        }
    }

    sys.stats.grains++;
    sys.stats.grains_total++;
    MaybeLogSummary(sys);
}

struct RackParams {
    u32 rack_id{};
    u32 max_voices{1};
    u32 max_matrices{1};
    u32 max_ports{1};
    char name[ORBIS_NGS2_RACK_NAME_LENGTH]{};
};

s32 ParseRackOption(u32 rack_id, const OrbisNgs2RackOption* option, RackParams& params) {
    params.rack_id = rack_id;
    if (!option) {
        return ORBIS_OK;
    }
    if (!IsGuestRangeValid(option, sizeof(OrbisNgs2RackOption))) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_ADDRESS;
    }
    if (option->size < sizeof(OrbisNgs2RackOption)) {
        return ORBIS_NGS2_ERROR_INVALID_OPTION_SIZE;
    }
    if (option->maxVoices == 0 || option->maxVoices > MAX_VOICES_PER_RACK) {
        return ORBIS_NGS2_ERROR_INVALID_MAX_VOICES;
    }
    params.max_voices = option->maxVoices;
    params.max_matrices = std::min(option->maxMatrices, MAX_MATRICES);
    params.max_ports = std::min(option->maxPorts, MAX_PORTS);
    std::memcpy(params.name, option->name, sizeof(params.name));
    params.name[sizeof(params.name) - 1] = '\0';
    return ORBIS_OK;
}

size_t RackBufferSize(const RackParams& params) {
    return 256 + static_cast<size_t>(params.max_voices) * 64;
}

std::shared_ptr<System> NewSystem(const OrbisNgs2SystemOption* option,
                                  const OrbisNgs2ContextBufferInfo& buffer_info,
                                  OrbisNgs2BufferFreeHandler free_handler) {
    auto sys = std::make_shared<System>();
    if (option) {
        sys->sample_rate = option->sampleRate;
        sys->max_grain_samples = option->maxGrainSamples;
        sys->num_grain_samples = std::min(option->numGrainSamples, option->maxGrainSamples);
        std::memcpy(sys->name, option->name, sizeof(sys->name));
        sys->name[sizeof(sys->name) - 1] = '\0';
    }
    sys->buffer_info = buffer_info;
    sys->free_handler = free_handler;
    sys->sampler_out.assign(static_cast<size_t>(MAX_CHANNELS) * sys->max_grain_samples, 0.0f);

    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    sys->serial = registry.next_system_serial++;
    registry.systems[sys->serial] = sys;
    return sys;
}

s32 CreateRackObject(const std::shared_ptr<System>& sys, const RackParams& params,
                     const OrbisNgs2ContextBufferInfo& buffer_info,
                     OrbisNgs2BufferFreeHandler free_handler, OrbisNgs2Handle* out_handle) {
    auto rack = std::make_unique<Rack>();
    rack->rack_id = params.rack_id;
    rack->kind = KindFromRackId(params.rack_id);
    rack->system = sys.get();
    std::memcpy(rack->name, params.name, sizeof(rack->name));
    rack->max_voices = params.max_voices;
    rack->max_matrices = params.max_matrices;
    rack->max_ports = params.max_ports;
    rack->buffer_info = buffer_info;
    rack->free_handler = free_handler;
    rack->voices.resize(params.max_voices);
    const size_t acc_size = static_cast<size_t>(MAX_CHANNELS) * sys->max_grain_samples;
    for (u32 i = 0; i < params.max_voices; i++) {
        Voice& voice = rack->voices[i];
        voice.rack = rack.get();
        voice.index = i;
        voice.num_ports = params.max_ports;
        voice.matrices.resize(params.max_matrices);
        if (rack->kind == RackKind::Sampler) {
            voice.sampler = std::make_unique<SamplerData>();
        } else if (HasBus(rack->kind)) {
            voice.bus = std::make_unique<BusData>();
            voice.bus->acc.assign(acc_size, 0.0f);
        }
    }

    Registry& registry = GetRegistry();
    u32 serial;
    {
        std::scoped_lock lock{registry.mutex};
        if (registry.next_rack_serial > SERIAL_MASK) {
            return ORBIS_NGS2_ERROR_FAIL;
        }
        serial = registry.next_rack_serial++;
        registry.rack_owner[serial] = sys;
    }
    rack->serial = serial;
    bool inserted = false;
    {
        std::scoped_lock lock{sys->mutex};
        if (!sys->destroyed) {
            sys->racks[serial] = std::move(rack);
            sys->order_dirty = true;
            inserted = true;
        }
    }
    if (!inserted) {
        std::scoped_lock lock{registry.mutex};
        registry.rack_owner.erase(serial);
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    *out_handle = MakeRackHandle(serial);
    return ORBIS_OK;
}

void DetachPatchesTo(System& sys, const Rack& dead) {
    for (auto& [serial, rack] : sys.racks) {
        for (Voice& voice : rack->voices) {
            for (Port& port : voice.ports) {
                if (port.dest && port.dest->rack == &dead) {
                    port.dest = nullptr;
                }
            }
        }
    }
    sys.order_dirty = true;
}

u32 ComputeStateFlags(const Voice& voice) {
    switch (voice.state) {
    case VoiceState::Free:
        return 0;
    case VoiceState::Idle:
        return STATE_FLAG_INUSE;
    case VoiceState::Playing: {
        u32 flags = STATE_FLAG_INUSE | STATE_FLAG_PLAYING;
        if (voice.sampler && voice.sampler->stream.Starved() && voice.sampler->fifo_frames == 0) {
            flags |= STATE_FLAG_EMPTY;
        }
        return flags;
    }
    case VoiceState::Paused:
        return STATE_FLAG_INUSE | STATE_FLAG_PAUSED;
    case VoiceState::Stopping:
        return STATE_FLAG_INUSE | STATE_FLAG_PLAYING | STATE_FLAG_STOPPED;
    }
    return 0;
}

template <class T>
void PutState(std::array<u8, MAX_STATE_SIZE>& buffer, size_t capacity, size_t offset, T value) {
    if (offset + sizeof(T) <= capacity) {
        std::memcpy(buffer.data() + offset, &value, sizeof(T));
    }
}

void FillState(const Voice& voice, std::array<u8, MAX_STATE_SIZE>& buffer, size_t size) {
    PutState<u32>(buffer, size, 0, ComputeStateFlags(voice));
    const Rack& rack = *voice.rack;
    if (rack.kind == RackKind::Sampler && voice.sampler) {
        // The wide layout carries an error code after the flags
        const size_t header = size >= 56 ? 8 : 4;
        const size_t numeric_base = header == 8 ? 24 : 16;
        const SamplerData& s = *voice.sampler;
        PutState<float>(buffer, size, header, s.env_level);
        PutState<u64>(buffer, size, numeric_base, s.stream.DecodedSamples());
        PutState<u64>(buffer, size, numeric_base + 8, s.stream.DecodedBytes());
        PutState<u64>(buffer, size, numeric_base + 16, s.stream.CurrentUserData());
        PutState<u64>(buffer, size, numeric_base + 24,
                      reinterpret_cast<u64>(s.stream.CurrentData()));
    } else if (rack.kind == RackKind::Submixer && voice.state != VoiceState::Free) {
        PutState<float>(buffer, size, size >= 20 ? 8 : 4, 1.0f);
    }
}

} // namespace

bool IsEnabled() {
    static const bool enabled = EmulatorSettings.IsNgs2SoftwareMixer();
    return enabled;
}

std::string DescribeVoice(OrbisNgs2Handle voice_handle) {
    const HandleParts parts = DecodeHandle(voice_handle);
    if (parts.kind != HandleKind::Voice) {
        return "unknown";
    }
    const auto sys = FindSystemOfRack(parts.serial);
    if (!sys) {
        return "unknown";
    }
    std::scoped_lock lock{sys->mutex};
    const Rack* rack = FindRack(*sys, parts.serial);
    if (!rack) {
        return "unknown";
    }
    return fmt::format("rack#{} rackId={:#x} voice={}", rack->serial, rack->rack_id,
                       parts.voice_index);
}

u32 RackCount() {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    return static_cast<u32>(registry.rack_owner.size());
}

s32 SystemCreate(const OrbisNgs2SystemOption* option, const OrbisNgs2ContextBufferInfo* bufferInfo,
                 OrbisNgs2Handle* outHandle) {
    if (!bufferInfo) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    OrbisNgs2ContextBufferInfo required{};
    const s32 result = SystemSetup(option, &required, nullptr, nullptr);
    if (result < 0) {
        return result;
    }
    const auto sys = NewSystem(option, *bufferInfo, nullptr);
    *outHandle = MakeSystemHandle(sys->serial);
    LOG_INFO(Lib_Ngs2, "mixer: system#{} created sampleRate={} grain={}/{}", sys->serial,
             sys->sample_rate, sys->num_grain_samples, sys->max_grain_samples);
    return ORBIS_OK;
}

s32 SystemCreateWithAllocator(const OrbisNgs2SystemOption* option,
                              const OrbisNgs2BufferAllocator* allocator,
                              OrbisNgs2Handle* outHandle) {
    if (!allocator || !allocator->allocHandler) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    OrbisNgs2ContextBufferInfo buffer_info{};
    s32 result = SystemSetup(option, &buffer_info, nullptr, nullptr);
    if (result < 0) {
        return result;
    }
    buffer_info.userData = allocator->userData;
    result = allocator->allocHandler(&buffer_info);
    if (result < 0) {
        return result;
    }
    const auto sys = NewSystem(option, buffer_info, allocator->freeHandler);
    *outHandle = MakeSystemHandle(sys->serial);
    LOG_INFO(Lib_Ngs2, "mixer: system#{} created sampleRate={} grain={}/{}", sys->serial,
             sys->sample_rate, sys->num_grain_samples, sys->max_grain_samples);
    return ORBIS_OK;
}

s32 SystemDestroy(OrbisNgs2Handle systemHandle, OrbisNgs2ContextBufferInfo* outBufferInfo) {
    const HandleParts parts = DecodeHandle(systemHandle);
    if (parts.kind != HandleKind::System) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    std::shared_ptr<System> sys;
    {
        Registry& registry = GetRegistry();
        std::scoped_lock lock{registry.mutex};
        const auto it = registry.systems.find(parts.serial);
        if (it == registry.systems.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        sys = it->second;
        registry.systems.erase(it);
        std::erase_if(registry.rack_owner, [&](const auto& entry) { return entry.second == sys; });
    }
    std::map<u32, std::unique_ptr<Rack>> racks;
    u64 grains_total;
    {
        std::scoped_lock lock{sys->mutex};
        sys->destroyed = true;
        racks = std::move(sys->racks);
        sys->racks.clear();
        sys->bus_order.clear();
        grains_total = sys->stats.grains_total;
    }
    racks.clear();

    OrbisNgs2ContextBufferInfo buffer_info = sys->buffer_info;
    if (sys->free_handler) {
        sys->free_handler(&buffer_info);
        if (outBufferInfo) {
            *outBufferInfo = {};
        }
    } else if (outBufferInfo) {
        *outBufferInfo = buffer_info;
    }
    LOG_INFO(Lib_Ngs2, "mixer: system#{} destroyed after {} grains", sys->serial, grains_total);
    return ORBIS_OK;
}

s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* aBufferInfo,
                 u32 numBufferInfo) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (numBufferInfo != 0 &&
        !IsGuestRangeValid(aBufferInfo,
                           static_cast<u64>(std::min(numBufferInfo, MAX_RENDER_BUFFERS)) *
                               sizeof(OrbisNgs2RenderBufferInfo))) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
    }
    std::scoped_lock lock{sys->mutex};
    if (sys->destroyed) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    RenderGrain(*sys, aBufferInfo, std::min(numBufferInfo, MAX_RENDER_BUFFERS));
    return ORBIS_OK;
}

s32 SystemLock(OrbisNgs2Handle systemHandle) {
    return FindSystem(systemHandle) ? ORBIS_OK : ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
}

s32 SystemUnlock(OrbisNgs2Handle systemHandle) {
    return FindSystem(systemHandle) ? ORBIS_OK : ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
}

s32 SystemGetUserData(OrbisNgs2Handle systemHandle, uintptr_t* outUserData) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (!IsGuestRangeValid(outUserData, sizeof(uintptr_t))) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    std::scoped_lock lock{sys->mutex};
    *outUserData = sys->user_data;
    return ORBIS_OK;
}

s32 SystemSetUserData(OrbisNgs2Handle systemHandle, uintptr_t userData) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    std::scoped_lock lock{sys->mutex};
    sys->user_data = userData;
    return ORBIS_OK;
}

s32 SystemSetGrainSamples(OrbisNgs2Handle systemHandle, u32 numSamples) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    std::scoped_lock lock{sys->mutex};
    if (!IsGrainSamplesValid(numSamples) || numSamples > sys->max_grain_samples) {
        return ORBIS_NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES;
    }
    sys->num_grain_samples = numSamples;
    return ORBIS_OK;
}

s32 SystemSetSampleRate(OrbisNgs2Handle systemHandle, u32 sampleRate) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (!IsSupportedSampleRate(sampleRate)) {
        return ORBIS_NGS2_ERROR_INVALID_SAMPLE_RATE;
    }
    std::scoped_lock lock{sys->mutex};
    sys->sample_rate = sampleRate;
    return ORBIS_OK;
}

s32 RackQueryBufferSize(u32 rackId, const OrbisNgs2RackOption* option,
                        OrbisNgs2ContextBufferInfo* outBufferInfo) {
    if (!outBufferInfo) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ADDRESS;
    }
    RackParams params;
    const s32 result = ParseRackOption(rackId, option, params);
    if (result < 0) {
        return result;
    }
    outBufferInfo->hostBuffer = nullptr;
    outBufferInfo->hostBufferSize = RackBufferSize(params);
    std::memset(outBufferInfo->reserved, 0, sizeof(outBufferInfo->reserved));
    return ORBIS_OK;
}

s32 RackCreate(OrbisNgs2Handle systemHandle, u32 rackId, const OrbisNgs2RackOption* option,
               const OrbisNgs2ContextBufferInfo* bufferInfo, OrbisNgs2Handle* outHandle) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    RackParams params;
    const s32 result = ParseRackOption(rackId, option, params);
    if (result < 0) {
        return result;
    }
    const OrbisNgs2ContextBufferInfo buffer_info =
        bufferInfo && IsGuestRangeValid(bufferInfo, sizeof(OrbisNgs2ContextBufferInfo))
            ? *bufferInfo
            : OrbisNgs2ContextBufferInfo{};
    return CreateRackObject(sys, params, buffer_info, nullptr, outHandle);
}

s32 RackCreateWithAllocator(OrbisNgs2Handle systemHandle, u32 rackId,
                            const OrbisNgs2RackOption* option,
                            const OrbisNgs2BufferAllocator* allocator, OrbisNgs2Handle* outHandle) {
    const auto sys = FindSystem(systemHandle);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (!allocator || !allocator->allocHandler) {
        return ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    RackParams params;
    s32 result = ParseRackOption(rackId, option, params);
    if (result < 0) {
        return result;
    }
    OrbisNgs2ContextBufferInfo buffer_info{};
    buffer_info.hostBufferSize = RackBufferSize(params);
    buffer_info.userData = allocator->userData;
    result = allocator->allocHandler(&buffer_info);
    if (result < 0) {
        return result;
    }
    result = CreateRackObject(sys, params, buffer_info, allocator->freeHandler, outHandle);
    if (result < 0 && allocator->freeHandler) {
        allocator->freeHandler(&buffer_info);
    }
    return result;
}

s32 RackDestroy(OrbisNgs2Handle rackHandle, OrbisNgs2ContextBufferInfo* outBufferInfo) {
    const HandleParts parts = DecodeHandle(rackHandle);
    if (parts.kind != HandleKind::Rack) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    std::shared_ptr<System> sys;
    {
        Registry& registry = GetRegistry();
        std::scoped_lock lock{registry.mutex};
        const auto it = registry.rack_owner.find(parts.serial);
        if (it == registry.rack_owner.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        sys = it->second;
        registry.rack_owner.erase(it);
    }
    std::unique_ptr<Rack> rack;
    {
        std::scoped_lock lock{sys->mutex};
        const auto it = sys->racks.find(parts.serial);
        if (it == sys->racks.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        rack = std::move(it->second);
        sys->racks.erase(it);
        DetachPatchesTo(*sys, *rack);
    }
    OrbisNgs2ContextBufferInfo buffer_info = rack->buffer_info;
    const OrbisNgs2BufferFreeHandler free_handler = rack->free_handler;
    rack.reset();
    if (free_handler) {
        free_handler(&buffer_info);
        if (outBufferInfo) {
            *outBufferInfo = {};
        }
    } else if (outBufferInfo) {
        *outBufferInfo = buffer_info;
    }
    return ORBIS_OK;
}

s32 RackGetVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex, OrbisNgs2Handle* outHandle) {
    const HandleParts parts = DecodeHandle(rackHandle);
    if (parts.kind != HandleKind::Rack) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    if (!outHandle) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    const auto sys = FindSystemOfRack(parts.serial);
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    std::scoped_lock lock{sys->mutex};
    const Rack* rack = FindRack(*sys, parts.serial);
    if (!rack) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    if (voiceIndex >= rack->voices.size()) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_INDEX;
    }
    *outHandle = MakeVoiceHandle(parts.serial, voiceIndex);
    return ORBIS_OK;
}

s32 RackGetUserData(OrbisNgs2Handle rackHandle, uintptr_t* outUserData) {
    const HandleParts parts = DecodeHandle(rackHandle);
    const auto sys = parts.kind == HandleKind::Rack ? FindSystemOfRack(parts.serial) : nullptr;
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    if (!IsGuestRangeValid(outUserData, sizeof(uintptr_t))) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    std::scoped_lock lock{sys->mutex};
    const Rack* rack = FindRack(*sys, parts.serial);
    if (!rack) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    *outUserData = rack->user_data;
    return ORBIS_OK;
}

s32 RackSetUserData(OrbisNgs2Handle rackHandle, uintptr_t userData) {
    const HandleParts parts = DecodeHandle(rackHandle);
    const auto sys = parts.kind == HandleKind::Rack ? FindSystemOfRack(parts.serial) : nullptr;
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    std::scoped_lock lock{sys->mutex};
    Rack* rack = FindRack(*sys, parts.serial);
    if (!rack) {
        return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
    }
    rack->user_data = userData;
    return ORBIS_OK;
}

s32 RackLock(OrbisNgs2Handle rackHandle) {
    const HandleParts parts = DecodeHandle(rackHandle);
    return parts.kind == HandleKind::Rack && FindSystemOfRack(parts.serial)
               ? ORBIS_OK
               : ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
}

s32 RackUnlock(OrbisNgs2Handle rackHandle) {
    return RackLock(rackHandle);
}

s32 VoiceControl(OrbisNgs2Handle voiceHandle, const OrbisNgs2VoiceParamHeader* paramList) {
    const HandleParts parts = DecodeHandle(voiceHandle);
    const auto sys = parts.kind == HandleKind::Voice ? FindSystemOfRack(parts.serial) : nullptr;
    if (!sys) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    }
    if (!paramList) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS;
    }
    std::scoped_lock lock{sys->mutex};
    Voice* voice = sys->destroyed ? nullptr : FindVoice(*sys, voiceHandle, false);
    if (!voice) {
        return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    }
    const OrbisNgs2VoiceParamHeader* header = paramList;
    for (u32 n = 0; n < MAX_PARAMS_PER_CALL; n++) {
        if (!IsGuestRangeValid(header, sizeof(OrbisNgs2VoiceParamHeader)) ||
            header->size < sizeof(OrbisNgs2VoiceParamHeader) ||
            !IsGuestRangeValid(header, header->size)) {
            NoteInvalid(*sys);
            break;
        }
        ApplyParam(*sys, *voice, *header);
        if (header->next <= 0 || header->next < header->size ||
            static_cast<u32>(header->next) > MAX_PARAM_STRIDE) {
            break;
        }
        header = reinterpret_cast<const OrbisNgs2VoiceParamHeader*>(
            reinterpret_cast<const u8*>(header) + header->next);
    }
    return ORBIS_OK;
}

s32 VoiceGetState(OrbisNgs2Handle voiceHandle, OrbisNgs2VoiceState* outState, size_t stateSize) {
    if (!outState || stateSize == 0) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    const size_t size = std::min<size_t>(stateSize, MAX_STATE_SIZE);
    if (!IsGuestRangeValid(outState, size)) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    std::array<u8, MAX_STATE_SIZE> buffer{};
    s32 result = ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    const HandleParts parts = DecodeHandle(voiceHandle);
    if (const auto sys =
            parts.kind == HandleKind::Voice ? FindSystemOfRack(parts.serial) : nullptr) {
        std::scoped_lock lock{sys->mutex};
        const Voice* voice = sys->destroyed ? nullptr : FindVoice(*sys, voiceHandle, false);
        if (voice) {
            FillState(*voice, buffer, size);
            auto& sizes = sys->stats.state_sizes;
            const auto key = std::make_pair(voice->rack->rack_id, static_cast<u32>(stateSize));
            if (sizes.size() < MAX_USER_COUNTERS || sizes.count(key)) {
                sizes[key]++;
            }
            result = ORBIS_OK;
        }
    }
    std::memcpy(outState, buffer.data(), size);
    return result;
}

s32 VoiceGetStateFlags(OrbisNgs2Handle voiceHandle, u32* outStateFlags) {
    if (!IsGuestRangeValid(outStateFlags, sizeof(u32))) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    u32 flags = 0;
    s32 result = ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
    const HandleParts parts = DecodeHandle(voiceHandle);
    if (const auto sys =
            parts.kind == HandleKind::Voice ? FindSystemOfRack(parts.serial) : nullptr) {
        std::scoped_lock lock{sys->mutex};
        const Voice* voice = sys->destroyed ? nullptr : FindVoice(*sys, voiceHandle, false);
        if (voice) {
            flags = ComputeStateFlags(*voice);
            result = ORBIS_OK;
        }
    }
    *outStateFlags = flags;
    return result;
}

} // namespace Libraries::Ngs2::Mixer
