// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/ngs2/ngs2_custom.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "core/libraries/ngs2/ngs2_geom.h"
#include "core/libraries/ngs2/ngs2_impl.h"
#include "core/libraries/ngs2/ngs2_mixer.h"
#include "core/libraries/ngs2/ngs2_pan.h"
#include "core/libraries/ngs2/ngs2_report.h"
#include "core/libraries/ngs2/ngs2_sampler.h"

namespace Libraries::Ngs2 {

// Rack and voice handles are opaque to the game. Until NGS2 is really emulated these are fake,
// unique, non-null values: rack n -> n << 24, voice i of that rack -> rack | (i + 1) << 4.
// The records below only exist so the diagnostics can say which rack a voice belongs to.
namespace {

struct RackRecord {
    u32 rackId;
    u32 maxVoices;
};

constexpr u32 DIAG_MAX_DETAILED_CALLS = 200;
constexpr u32 DIAG_MAX_DETAILED_RENDERS = 50;
constexpr u32 DIAG_MAX_PARAMS_PER_CALL = 64;
constexpr u64 DIAG_SUMMARY_PERIOD_US = 5'000'000;

std::mutex diag_mutex;
// ponytail: never shrinks, racks are created a handful of times per run; free on RackDestroy
// once the real implementation tracks them.
std::vector<RackRecord> rack_records;
std::map<u32, u64> param_id_counts; // guarded by diag_mutex

std::atomic<u64> voice_control_calls{0};
std::atomic<u64> voice_control_params{0};
std::atomic<u64> voice_get_state_calls{0};
std::atomic<u64> render_calls{0};
std::atomic<u64> last_summary_us{0};
u64 prev_voice_control_calls = 0;
u64 prev_voice_get_state_calls = 0;
u64 prev_render_calls = 0;

bool DiagEnabled() {
    return Common::Log::ShouldLog(Common::Log::Class::Lib_Ngs2, Common::Log::Level::Info);
}

u64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string_view FixedName(const char* name, size_t max) {
    size_t len = 0;
    while (len < max && name[len] != '\0') {
        len++;
    }
    return {name, len};
}

OrbisNgs2Handle NewFakeRackHandle(u32 rackId, u32 maxVoices) {
    std::scoped_lock lock{diag_mutex};
    rack_records.push_back({rackId, maxVoices});
    return static_cast<OrbisNgs2Handle>(rack_records.size()) << 24;
}

OrbisNgs2Handle FakeVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex) {
    return rackHandle | ((static_cast<OrbisNgs2Handle>(voiceIndex % 0xFFFFF) + 1) << 4);
}

std::string DescribeVoice(OrbisNgs2Handle voice) {
    if (Mixer::IsEnabled()) {
        return Mixer::DescribeVoice(voice);
    }
    const u64 rack_index = voice >> 24;
    const u64 voice_index = ((voice >> 4) & 0xFFFFF);
    std::scoped_lock lock{diag_mutex};
    if (rack_index == 0 || rack_index > rack_records.size() || voice_index == 0) {
        return "unknown";
    }
    return fmt::format("rack#{} rackId={:#x} voice={}", rack_index,
                       rack_records[rack_index - 1].rackId, voice_index - 1);
}

void LogSystemOption(const char* api, const OrbisNgs2SystemOption* option) {
    if (!option) {
        LOG_INFO(Lib_Ngs2, "{}: option=null (defaults 48000 Hz, grain 256/512)", api);
        return;
    }
    LOG_INFO(Lib_Ngs2,
             "{}: option size={} name='{}' flags={:#x} sampleRate={} numGrainSamples={} "
             "maxGrainSamples={}",
             api, option->size, FixedName(option->name, sizeof(option->name)), option->flags,
             option->sampleRate, option->numGrainSamples, option->maxGrainSamples);
}

void LogRackOption(const char* api, u32 rackId, const OrbisNgs2RackOption* option) {
    if (!option) {
        LOG_INFO(Lib_Ngs2, "{}: rackId={} ({:#x}) option=null", api, rackId, rackId);
        return;
    }
    LOG_INFO(Lib_Ngs2,
             "{}: rackId={} ({:#x}) option size={} name='{}' flags={:#x} maxGrainSamples={} "
             "maxVoices={} maxInputDelayBlocks={} maxMatrices={} maxPorts={}",
             api, rackId, rackId, option->size, FixedName(option->name, sizeof(option->name)),
             option->flags, option->maxGrainSamples, option->maxVoices,
             option->maxInputDelayBlocks, option->maxMatrices, option->maxPorts);
    // Sampler rack options extend the base option by nine u32 fields
    if ((rackId & 0xF000) == 0x1000 &&
        option->size >= sizeof(OrbisNgs2RackOption) + 9 * sizeof(u32)) {
        const auto* s = reinterpret_cast<const OrbisNgs2SamplerRackOption*>(option);
        LOG_INFO(Lib_Ngs2,
                 "{}:   sampler option maxChannelWorks={} maxCodecCaches={} maxWaveformBlocks={} "
                 "maxEnvelopePoints={} maxFilters={} maxAtrac9Decoders={} maxAtrac9ChannelWorks={} "
                 "maxAjmAtrac9Decoders={} numPeakMeterBlocks={}",
                 api, s->maxChannelWorks, s->maxCodecCaches, s->maxWaveformBlocks,
                 s->maxEnvelopePoints, s->maxFilters, s->maxAtrac9Decoders,
                 s->maxAtrac9ChannelWorks, s->maxAjmAtrac9Decoders, s->numPeakMeterBlocks);
    }
}

// One line every DIAG_SUMMARY_PERIOD_US with call counts and a histogram of VoiceControl param ids
void MaybeLogSummary() {
    if (!DiagEnabled()) {
        return;
    }
    const u64 now = NowUs();
    u64 last = last_summary_us.load(std::memory_order_relaxed);
    if (now - last < DIAG_SUMMARY_PERIOD_US ||
        !last_summary_us.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const u64 vc = voice_control_calls.load(std::memory_order_relaxed);
    const u64 gs = voice_get_state_calls.load(std::memory_order_relaxed);
    const u64 rc = render_calls.load(std::memory_order_relaxed);
    std::string ids;
    {
        std::scoped_lock lock{diag_mutex};
        for (const auto& [id, count] : param_id_counts) {
            ids += fmt::format(" {:#x}:{}", id, count);
        }
        LOG_INFO(Lib_Ngs2,
                 "summary: VoiceControl calls={} (+{}) params={} GetState calls={} (+{}) "
                 "SystemRender calls={} (+{}) racks={} paramIds={{{} }}",
                 vc, vc - prev_voice_control_calls,
                 voice_control_params.load(std::memory_order_relaxed), gs,
                 gs - prev_voice_get_state_calls, rc, rc - prev_render_calls,
                 Mixer::IsEnabled() ? Mixer::RackCount() : rack_records.size(), ids);
        prev_voice_control_calls = vc;
        prev_voice_get_state_calls = gs;
        prev_render_calls = rc;
    }
}

void LogVoiceParam(u32 index, const OrbisNgs2VoiceParamHeader* p) {
    std::string words;
    if (p->size > sizeof(OrbisNgs2VoiceParamHeader)) {
        const u32 num_words = std::min<u32>((p->size - sizeof(OrbisNgs2VoiceParamHeader)) / 4, 10);
        const u32* payload = reinterpret_cast<const u32*>(p + 1);
        for (u32 i = 0; i < num_words; i++) {
            words += fmt::format(" {:08x}", payload[i]);
        }
    }
    LOG_INFO(Lib_Ngs2, "  param[{}] id={:#010x} size={} next={} payload:{}", index, p->id, p->size,
             p->next, words);
    // The id alone does not say which struct this is; decode by size as a hint only.
    if (p->size == sizeof(OrbisNgs2SamplerVoiceSetupParam)) {
        const auto* sp = reinterpret_cast<const OrbisNgs2SamplerVoiceSetupParam*>(p);
        LOG_INFO(Lib_Ngs2,
                 "    if SamplerVoiceSetup: waveformType={:#x} numChannels={} sampleRate={} "
                 "configData={:#x} frameOffset={} frameMargin={} flags={:#x}",
                 sp->format.waveformType, sp->format.numChannels, sp->format.sampleRate,
                 sp->format.configData, sp->format.frameOffset, sp->format.frameMargin, sp->flags);
    } else if (p->size == sizeof(OrbisNgs2SamplerVoiceWaveformBlocksParam)) {
        const auto* bp = reinterpret_cast<const OrbisNgs2SamplerVoiceWaveformBlocksParam*>(p);
        LOG_INFO(Lib_Ngs2, "    if SamplerVoiceWaveformBlocks: data={} flags={:#x} numBlocks={}",
                 bp->data, bp->flags, bp->numBlocks);
        if (bp->aBlock && bp->numBlocks >= 1 && bp->numBlocks <= 64) {
            for (u32 b = 0; b < std::min<u32>(bp->numBlocks, 4); b++) {
                const auto& blk = bp->aBlock[b];
                LOG_INFO(Lib_Ngs2,
                         "      block[{}] dataOffset={} dataSize={} numRepeats={} "
                         "numSkipSamples={} numSamples={}",
                         b, blk.dataOffset, blk.dataSize, blk.numRepeats, blk.numSkipSamples,
                         blk.numSamples);
            }
            // First bytes of the waveform data help identify the codec (RIFF, VAGp, ...)
            const auto& first = bp->aBlock[0];
            if (bp->data && first.dataSize >= 16) {
                const u8* bytes = static_cast<const u8*>(bp->data) + first.dataOffset;
                std::string hex;
                for (u32 i = 0; i < 16; i++) {
                    hex += fmt::format("{:02x}", bytes[i]);
                }
                LOG_INFO(Lib_Ngs2, "      block[0] first 16 bytes: {}", hex);
            }
        }
    }
}

void DiagVoiceControl(OrbisNgs2Handle voice, const OrbisNgs2VoiceParamHeader* list) {
    const u64 call = voice_control_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool detailed = call <= DIAG_MAX_DETAILED_CALLS;
    if (detailed) {
        LOG_INFO(Lib_Ngs2, "VoiceControl #{} voice={:#x} ({}) list={}", call, voice,
                 DescribeVoice(voice), static_cast<const void*>(list));
    }
    const OrbisNgs2VoiceParamHeader* p = list;
    for (u32 i = 0; p && i < DIAG_MAX_PARAMS_PER_CALL; i++) {
        voice_control_params.fetch_add(1, std::memory_order_relaxed);
        {
            std::scoped_lock lock{diag_mutex};
            // Bounded: a corrupt list must not grow the histogram forever
            if (param_id_counts.size() < 256 || param_id_counts.count(p->id)) {
                param_id_counts[p->id]++;
            }
        }
        if (detailed) {
            LogVoiceParam(i, p);
        }
        // next is a byte offset to the following header; 0 ends the chain. Refuse implausible
        // offsets instead of reading arbitrary guest memory.
        if (p->next <= 0 || p->next < p->size || p->next > 4096) {
            break;
        }
        p = reinterpret_cast<const OrbisNgs2VoiceParamHeader*>(reinterpret_cast<const u8*>(p) +
                                                                p->next);
    }
    MaybeLogSummary();
}

void DiagRender(OrbisNgs2Handle system, const OrbisNgs2RenderBufferInfo* buffers, u32 num) {
    const u64 call = render_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call <= DIAG_MAX_DETAILED_RENDERS) {
        LOG_INFO(Lib_Ngs2, "SystemRender #{} system={:#x} numBufferInfo={}", call, system, num);
        for (u32 i = 0; buffers && i < std::min<u32>(num, 16); i++) {
            LOG_INFO(Lib_Ngs2,
                     "  buf[{}] ptr={} size={} waveformType={:#x} numChannels={}", i,
                     buffers[i].buffer, buffers[i].bufferSize, buffers[i].waveformType,
                     buffers[i].numChannels);
        }
    }
    MaybeLogSummary();
}

} // namespace

// Ngs2

s32 PS4_SYSV_ABI sceNgs2CalcWaveformBlock(const OrbisNgs2WaveformFormat* format, u32 samplePos,
                                          u32 numSamples, OrbisNgs2WaveformBlock* outBlock) {
    LOG_ERROR(Lib_Ngs2, "samplePos = {}, numSamples = {}", samplePos, numSamples);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GetWaveformFrameInfo(const OrbisNgs2WaveformFormat* format,
                                             u32* outFrameSize, u32* outNumFrameSamples,
                                             u32* outUnitsPerFrame, u32* outNumDelaySamples) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ParseWaveformData(const void* data, size_t dataSize,
                                          OrbisNgs2WaveformInfo* outInfo) {
    LOG_ERROR(Lib_Ngs2, "dataSize = {}", dataSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ParseWaveformFile(const char* path, u64 offset,
                                          OrbisNgs2WaveformInfo* outInfo) {
    LOG_ERROR(Lib_Ngs2, "path = {}, offset = {}", path, offset);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ParseWaveformUser(OrbisNgs2ParseReadHandler handler, uintptr_t userData,
                                          OrbisNgs2WaveformInfo* outInfo) {
    LOG_ERROR(Lib_Ngs2, "userData = {}", userData);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackCreate(OrbisNgs2Handle systemHandle, u32 rackId,
                                   const OrbisNgs2RackOption* option,
                                   const OrbisNgs2ContextBufferInfo* bufferInfo,
                                   OrbisNgs2Handle* outHandle) {
    LogRackOption("sceNgs2RackCreate", rackId, option);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        const s32 result = Mixer::RackCreate(systemHandle, rackId, option, bufferInfo, outHandle);
        LOG_INFO(Lib_Ngs2, "sceNgs2RackCreate: rackId={:#x} -> result {:#x} rack handle {:#x}",
                 rackId, static_cast<u32>(result), result >= 0 && outHandle ? *outHandle : 0);
        return result;
    }
    if (outHandle) {
        *outHandle = NewFakeRackHandle(rackId, option ? option->maxVoices : 0);
        LOG_INFO(Lib_Ngs2, "sceNgs2RackCreate: rackId={:#x} -> fake rack handle {:#x}", rackId,
                 *outHandle);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackCreateWithAllocator(OrbisNgs2Handle systemHandle, u32 rackId,
                                                const OrbisNgs2RackOption* option,
                                                const OrbisNgs2BufferAllocator* allocator,
                                                OrbisNgs2Handle* outHandle) {
    LogRackOption("sceNgs2RackCreateWithAllocator", rackId, option);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        const s32 result =
            Mixer::RackCreateWithAllocator(systemHandle, rackId, option, allocator, outHandle);
        LOG_INFO(Lib_Ngs2,
                 "sceNgs2RackCreateWithAllocator: rackId={:#x} -> result {:#x} rack handle {:#x}",
                 rackId, static_cast<u32>(result), result >= 0 && outHandle ? *outHandle : 0);
        return result;
    }
    if (outHandle) {
        *outHandle = NewFakeRackHandle(rackId, option ? option->maxVoices : 0);
        LOG_INFO(Lib_Ngs2, "sceNgs2RackCreateWithAllocator: rackId={:#x} -> fake rack handle {:#x}",
                 rackId, *outHandle);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackDestroy(OrbisNgs2Handle rackHandle,
                                    OrbisNgs2ContextBufferInfo* outBufferInfo) {
    if (Mixer::IsEnabled()) {
        return Mixer::RackDestroy(rackHandle, outBufferInfo);
    }
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackGetInfo(OrbisNgs2Handle rackHandle, OrbisNgs2RackInfo* outInfo,
                                    size_t infoSize) {
    LOG_ERROR(Lib_Ngs2, "infoSize = {}", infoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackGetUserData(OrbisNgs2Handle rackHandle, uintptr_t* outUserData) {
    if (Mixer::IsEnabled()) {
        return Mixer::RackGetUserData(rackHandle, outUserData);
    }
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackGetVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex,
                                           OrbisNgs2Handle* outHandle) {
    if (Mixer::IsEnabled()) {
        const s32 result = Mixer::RackGetVoiceHandle(rackHandle, voiceIndex, outHandle);
        static std::atomic<u32> mixer_logged{0};
        if (mixer_logged.load(std::memory_order_relaxed) < 64 && DiagEnabled()) {
            mixer_logged.fetch_add(1, std::memory_order_relaxed);
            LOG_INFO(Lib_Ngs2,
                     "sceNgs2RackGetVoiceHandle: rack={:#x} voiceIndex={} -> result {:#x} {:#x}",
                     rackHandle, voiceIndex, static_cast<u32>(result),
                     result >= 0 && outHandle ? *outHandle : 0);
        }
        return result;
    }
    if (outHandle) {
        *outHandle = FakeVoiceHandle(rackHandle, voiceIndex);
    }
    static std::atomic<u32> logged{0};
    if (logged.load(std::memory_order_relaxed) < 64 && DiagEnabled()) {
        logged.fetch_add(1, std::memory_order_relaxed);
        LOG_INFO(Lib_Ngs2, "sceNgs2RackGetVoiceHandle: rack={:#x} voiceIndex={} -> {:#x}",
                 rackHandle, voiceIndex, FakeVoiceHandle(rackHandle, voiceIndex));
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackLock(OrbisNgs2Handle rackHandle) {
    if (Mixer::IsEnabled()) {
        return Mixer::RackLock(rackHandle);
    }
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackQueryBufferSize(u32 rackId, const OrbisNgs2RackOption* option,
                                            OrbisNgs2ContextBufferInfo* outBufferInfo) {
    LogRackOption("sceNgs2RackQueryBufferSize", rackId, option);
    if (Mixer::IsEnabled()) {
        return Mixer::RackQueryBufferSize(rackId, option, outBufferInfo);
    }
    return RackQueryBufferSize(option, outBufferInfo);
}

s32 PS4_SYSV_ABI sceNgs2RackSetUserData(OrbisNgs2Handle rackHandle, uintptr_t userData) {
    if (Mixer::IsEnabled()) {
        return Mixer::RackSetUserData(rackHandle, userData);
    }
    LOG_ERROR(Lib_Ngs2, "userData = {}", userData);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackUnlock(OrbisNgs2Handle rackHandle) {
    if (Mixer::IsEnabled()) {
        return Mixer::RackUnlock(rackHandle);
    }
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemCreate(const OrbisNgs2SystemOption* option,
                                     const OrbisNgs2ContextBufferInfo* bufferInfo,
                                     OrbisNgs2Handle* outHandle) {
    s32 result;
    OrbisNgs2ContextBufferInfo localInfo;
    LogSystemOption("sceNgs2SystemCreate", option);
    if (Mixer::IsEnabled()) {
        return Mixer::SystemCreate(option, bufferInfo, outHandle);
    }
    if (!bufferInfo || !outHandle) {
        if (!bufferInfo) {
            result = ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
            LOG_ERROR(Lib_Ngs2, "Invalid system buffer info {}", (void*)bufferInfo);
        } else {
            result = ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
            LOG_ERROR(Lib_Ngs2, "Invalid system handle address {}", (void*)outHandle);
        }

        // TODO: Report errors?
    } else {
        // Make bufferInfo copy
        localInfo.hostBuffer = bufferInfo->hostBuffer;
        localInfo.hostBufferSize = bufferInfo->hostBufferSize;
        for (int i = 0; i < 5; i++) {
            localInfo.reserved[i] = bufferInfo->reserved[i];
        }
        localInfo.userData = bufferInfo->userData;

        result = SystemSetup(option, &localInfo, 0, outHandle);
    }

    // TODO: API reporting?

    LOG_INFO(Lib_Ngs2, "called");
    return result;
}

s32 PS4_SYSV_ABI sceNgs2SystemCreateWithAllocator(const OrbisNgs2SystemOption* option,
                                                  const OrbisNgs2BufferAllocator* allocator,
                                                  OrbisNgs2Handle* outHandle) {
    s32 result;
    LogSystemOption("sceNgs2SystemCreateWithAllocator", option);
    if (Mixer::IsEnabled()) {
        return Mixer::SystemCreateWithAllocator(option, allocator, outHandle);
    }
    if (allocator && allocator->allocHandler != 0) {
        OrbisNgs2BufferAllocHandler hostAlloc = allocator->allocHandler;
        if (outHandle) {
            OrbisNgs2BufferFreeHandler hostFree = allocator->freeHandler;
            OrbisNgs2ContextBufferInfo bufferInfo;
            result = SystemSetup(option, &bufferInfo, 0, 0);
            if (result >= 0) {
                uintptr_t sysUserData = allocator->userData;
                result = hostAlloc(&bufferInfo);
                if (result >= 0) {
                    OrbisNgs2Handle* handleCopy = outHandle;
                    result = SystemSetup(option, &bufferInfo, hostFree, handleCopy);
                    if (result < 0) {
                        if (hostFree) {
                            hostFree(&bufferInfo);
                        }
                    }
                }
            }
        } else {
            result = ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
            LOG_ERROR(Lib_Ngs2, "Invalid system handle address {}", (void*)outHandle);
        }
    } else {
        result = ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
        LOG_ERROR(Lib_Ngs2, "Invalid system buffer allocator {}", (void*)allocator);
    }
    LOG_INFO(Lib_Ngs2, "called");
    return result;
}

s32 PS4_SYSV_ABI sceNgs2SystemDestroy(OrbisNgs2Handle systemHandle,
                                      OrbisNgs2ContextBufferInfo* outBufferInfo) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemDestroy(systemHandle, outBufferInfo);
    }
    LOG_INFO(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemEnumHandles(OrbisNgs2Handle* aOutHandle, u32 maxHandles) {
    LOG_ERROR(Lib_Ngs2, "maxHandles = {}", maxHandles);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemEnumRackHandles(OrbisNgs2Handle systemHandle,
                                              OrbisNgs2Handle* aOutHandle, u32 maxHandles) {
    LOG_ERROR(Lib_Ngs2, "maxHandles = {}", maxHandles);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemGetInfo(OrbisNgs2Handle rackHandle, OrbisNgs2SystemInfo* outInfo,
                                      size_t infoSize) {
    LOG_ERROR(Lib_Ngs2, "infoSize = {}", infoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemGetUserData(OrbisNgs2Handle systemHandle, uintptr_t* outUserData) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemGetUserData(systemHandle, outUserData);
    }
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemLock(OrbisNgs2Handle systemHandle) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemLock(systemHandle);
    }
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemQueryBufferSize(const OrbisNgs2SystemOption* option,
                                              OrbisNgs2ContextBufferInfo* outBufferInfo) {
    s32 result;
    if (outBufferInfo) {
        result = SystemSetup(option, outBufferInfo, 0, 0);
        LOG_INFO(Lib_Ngs2, "called");
    } else {
        result = ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        LOG_ERROR(Lib_Ngs2, "Invalid system buffer info {}", (void*)outBufferInfo);
    }

    return result;
}

s32 PS4_SYSV_ABI sceNgs2SystemRender(OrbisNgs2Handle systemHandle,
                                     const OrbisNgs2RenderBufferInfo* aBufferInfo,
                                     u32 numBufferInfo) {
    LOG_DEBUG(Lib_Ngs2, "(STUBBED) numBufferInfo = {}", numBufferInfo);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (DiagEnabled()) {
        DiagRender(systemHandle, aBufferInfo, numBufferInfo);
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemRender(systemHandle, aBufferInfo, numBufferInfo);
    }
    // Nothing is rendered: the guest buffers are left untouched
    return ORBIS_OK;
}

static s32 PS4_SYSV_ABI sceNgs2SystemResetOption(OrbisNgs2SystemOption* outOption) {
    static const OrbisNgs2SystemOption option = {
        sizeof(OrbisNgs2SystemOption), "", 0, 512, 256, 48000, {0}};

    if (!outOption) {
        LOG_ERROR(Lib_Ngs2, "Invalid system option address {}", (void*)outOption);
        return ORBIS_NGS2_ERROR_INVALID_OPTION_ADDRESS;
    }
    *outOption = option;

    LOG_INFO(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemSetGrainSamples(OrbisNgs2Handle systemHandle, u32 numSamples) {
    LOG_INFO(Lib_Ngs2, "sceNgs2SystemSetGrainSamples: numSamples = {}", numSamples);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemSetGrainSamples(systemHandle, numSamples);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemSetSampleRate(OrbisNgs2Handle systemHandle, u32 sampleRate) {
    LOG_INFO(Lib_Ngs2, "sceNgs2SystemSetSampleRate: sampleRate = {}", sampleRate);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemSetSampleRate(systemHandle, sampleRate);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemSetUserData(OrbisNgs2Handle systemHandle, uintptr_t userData) {
    LOG_ERROR(Lib_Ngs2, "userData = {}", userData);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemSetUserData(systemHandle, userData);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemUnlock(OrbisNgs2Handle systemHandle) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    if (Mixer::IsEnabled()) {
        return Mixer::SystemUnlock(systemHandle);
    }
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceControl(OrbisNgs2Handle voiceHandle,
                                     const OrbisNgs2VoiceParamHeader* paramList) {
    // Hot path (per audio grain). Logged at Info for diagnosis; applied only by the software mixer.
    if (DiagEnabled()) {
        DiagVoiceControl(voiceHandle, paramList);
    }
    if (Mixer::IsEnabled()) {
        return Mixer::VoiceControl(voiceHandle, paramList);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetMatrixInfo(OrbisNgs2Handle voiceHandle, u32 matrixId,
                                           OrbisNgs2VoiceMatrixInfo* outInfo, size_t outInfoSize) {
    LOG_DEBUG(Lib_Ngs2, "matrixId = {}, outInfoSize = {}", matrixId, outInfoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetOwner(OrbisNgs2Handle voiceHandle, OrbisNgs2Handle* outRackHandle,
                                      u32* outVoiceId) {
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetPortInfo(OrbisNgs2Handle voiceHandle, u32 port,
                                         OrbisNgs2VoicePortInfo* outInfo, size_t outInfoSize) {
    LOG_DEBUG(Lib_Ngs2, "port = {}, outInfoSize = {}", port, outInfoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetState(OrbisNgs2Handle voiceHandle, OrbisNgs2VoiceState* outState,
                                      size_t stateSize) {
    // Polled every audio grain by SotC; stays quiet. Only the software mixer writes the state
    LOG_DEBUG(Lib_Ngs2, "stateSize = {}", stateSize);
    voice_get_state_calls.fetch_add(1, std::memory_order_relaxed);
    MaybeLogSummary();
    if (Mixer::IsEnabled()) {
        return Mixer::VoiceGetState(voiceHandle, outState, stateSize);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetStateFlags(OrbisNgs2Handle voiceHandle, u32* outStateFlags) {
    if (Mixer::IsEnabled()) {
        return Mixer::VoiceGetStateFlags(voiceHandle, outStateFlags);
    }
    // Report an idle voice instead of leaving the caller's variable uninitialized
    if (outStateFlags) {
        *outStateFlags = 0;
    }
    LOG_DEBUG(Lib_Ngs2, "called");
    return ORBIS_OK;
}

// Ngs2Custom

s32 PS4_SYSV_ABI sceNgs2CustomRackGetModuleInfo(OrbisNgs2Handle rackHandle, u32 moduleIndex,
                                                OrbisNgs2CustomModuleInfo* outInfo,
                                                size_t infoSize) {
    LOG_ERROR(Lib_Ngs2, "moduleIndex = {}, infoSize = {}", moduleIndex, infoSize);
    return ORBIS_OK;
}

// Ngs2Geom

s32 PS4_SYSV_ABI sceNgs2GeomResetListenerParam(OrbisNgs2GeomListenerParam* outListenerParam) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GeomResetSourceParam(OrbisNgs2GeomSourceParam* outSourceParam) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GeomCalcListener(const OrbisNgs2GeomListenerParam* param,
                                         OrbisNgs2GeomListenerWork* outWork, u32 flags) {
    LOG_ERROR(Lib_Ngs2, "flags = {}", flags);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GeomApply(const OrbisNgs2GeomListenerWork* listener,
                                  const OrbisNgs2GeomSourceParam* source,
                                  OrbisNgs2GeomAttribute* outAttrib, u32 flags) {
    LOG_ERROR(Lib_Ngs2, "flags = {}", flags);
    return ORBIS_OK;
}

// Ngs2Pan

s32 PS4_SYSV_ABI sceNgs2PanInit(OrbisNgs2PanWork* work, const float* aSpeakerAngle, float unitAngle,
                                u32 numSpeakers) {
    LOG_ERROR(Lib_Ngs2, "unitAngle = {}, numSpeakers = {}", unitAngle, numSpeakers);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2PanGetVolumeMatrix(OrbisNgs2PanWork* work, const OrbisNgs2PanParam* aParam,
                                           u32 numParams, u32 matrixFormat,
                                           float* outVolumeMatrix) {
    LOG_ERROR(Lib_Ngs2, "numParams = {}, matrixFormat = {}", numParams, matrixFormat);
    return ORBIS_OK;
}

// Ngs2Report

s32 PS4_SYSV_ABI sceNgs2ReportRegisterHandler(u32 reportType, OrbisNgs2ReportHandler handler,
                                              uintptr_t userData, OrbisNgs2Handle* outHandle) {
    LOG_INFO(Lib_Ngs2, "reportType = {}, userData = {}", reportType, userData);
    if (!handler) {
        LOG_ERROR(Lib_Ngs2, "handler is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_REPORT_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ReportUnregisterHandler(OrbisNgs2Handle reportHandle) {
    if (!reportHandle) {
        LOG_ERROR(Lib_Ngs2, "reportHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_REPORT_HANDLE;
    }
    LOG_INFO(Lib_Ngs2, "called");
    return ORBIS_OK;
}

// Unknown

int PS4_SYSV_ABI sceNgs2FftInit() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2FftProcess() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2FftQuerySize() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2JobSchedulerResetOption() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2ModuleArrayEnumItems() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2ModuleEnumConfigs() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2ModuleQueueEnumItems() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2RackQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2RackRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2SystemQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2SystemRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2SystemSetLoudThreshold() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamCreate() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamCreateWithAllocator() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamDestroy() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamQueryBufferSize() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamResetOption() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2VoiceQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2VoiceRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("3pCNbVM11UA", "libSceNgs2", 1, "libSceNgs2", sceNgs2CalcWaveformBlock);
    LIB_FUNCTION("6qN1zaEZuN0", "libSceNgs2", 1, "libSceNgs2", sceNgs2CustomRackGetModuleInfo);
    LIB_FUNCTION("Kg1MA5j7KFk", "libSceNgs2", 1, "libSceNgs2", sceNgs2FftInit);
    LIB_FUNCTION("D8eCqBxSojA", "libSceNgs2", 1, "libSceNgs2", sceNgs2FftProcess);
    LIB_FUNCTION("-YNfTO6KOMY", "libSceNgs2", 1, "libSceNgs2", sceNgs2FftQuerySize);
    LIB_FUNCTION("eF8yRCC6W64", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomApply);
    LIB_FUNCTION("1WsleK-MTkE", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomCalcListener);
    LIB_FUNCTION("7Lcfo8SmpsU", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomResetListenerParam);
    LIB_FUNCTION("0lbbayqDNoE", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomResetSourceParam);
    LIB_FUNCTION("ekGJmmoc8j4", "libSceNgs2", 1, "libSceNgs2", sceNgs2GetWaveformFrameInfo);
    LIB_FUNCTION("BcoPfWfpvVI", "libSceNgs2", 1, "libSceNgs2", sceNgs2JobSchedulerResetOption);
    LIB_FUNCTION("EEemGEQCjO8", "libSceNgs2", 1, "libSceNgs2", sceNgs2ModuleArrayEnumItems);
    LIB_FUNCTION("TaoNtmMKkXQ", "libSceNgs2", 1, "libSceNgs2", sceNgs2ModuleEnumConfigs);
    LIB_FUNCTION("ve6bZi+1sYQ", "libSceNgs2", 1, "libSceNgs2", sceNgs2ModuleQueueEnumItems);
    LIB_FUNCTION("gbMKV+8Enuo", "libSceNgs2", 1, "libSceNgs2", sceNgs2PanGetVolumeMatrix);
    LIB_FUNCTION("xa8oL9dmXkM", "libSceNgs2", 1, "libSceNgs2", sceNgs2PanInit);
    LIB_FUNCTION("hyVLT2VlOYk", "libSceNgs2", 1, "libSceNgs2", sceNgs2ParseWaveformData);
    LIB_FUNCTION("iprCTXPVWMI", "libSceNgs2", 1, "libSceNgs2", sceNgs2ParseWaveformFile);
    LIB_FUNCTION("t9T0QM17Kvo", "libSceNgs2", 1, "libSceNgs2", sceNgs2ParseWaveformUser);
    LIB_FUNCTION("cLV4aiT9JpA", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackCreate);
    LIB_FUNCTION("U546k6orxQo", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackCreateWithAllocator);
    LIB_FUNCTION("lCqD7oycmIM", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackDestroy);
    LIB_FUNCTION("M4LYATRhRUE", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackGetInfo);
    LIB_FUNCTION("Mn4XNDg03XY", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackGetUserData);
    LIB_FUNCTION("MwmHz8pAdAo", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackGetVoiceHandle);
    LIB_FUNCTION("MzTa7VLjogY", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackLock);
    LIB_FUNCTION("0eFLVCfWVds", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackQueryBufferSize);
    LIB_FUNCTION("TZqb8E-j3dY", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackQueryInfo);
    LIB_FUNCTION("MI2VmBx2RbM", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackRunCommands);
    LIB_FUNCTION("JNTMIaBIbV4", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackSetUserData);
    LIB_FUNCTION("++YZ7P9e87U", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackUnlock);
    LIB_FUNCTION("uBIN24Tv2MI", "libSceNgs2", 1, "libSceNgs2", sceNgs2ReportRegisterHandler);
    LIB_FUNCTION("nPzb7Ly-VjE", "libSceNgs2", 1, "libSceNgs2", sceNgs2ReportUnregisterHandler);
    LIB_FUNCTION("koBbCMvOKWw", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemCreate);
    LIB_FUNCTION("mPYgU4oYpuY", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemCreateWithAllocator);
    LIB_FUNCTION("u-WrYDaJA3k", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemDestroy);
    LIB_FUNCTION("vubFP0T6MP0", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemEnumHandles);
    LIB_FUNCTION("U-+7HsswcIs", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemEnumRackHandles);
    LIB_FUNCTION("vU7TQ62pItw", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemGetInfo);
    LIB_FUNCTION("4lFaRxd-aLs", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemGetUserData);
    LIB_FUNCTION("gThZqM5PYlQ", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemLock);
    LIB_FUNCTION("pgFAiLR5qT4", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemQueryBufferSize);
    LIB_FUNCTION("3oIK7y7O4k0", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemQueryInfo);
    LIB_FUNCTION("i0VnXM-C9fc", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemRender);
    LIB_FUNCTION("AQkj7C0f3PY", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemResetOption);
    LIB_FUNCTION("gXiormHoZZ4", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemRunCommands);
    LIB_FUNCTION("l4Q2dWEH6UM", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetGrainSamples);
    LIB_FUNCTION("Wdlx0ZFTV9s", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetLoudThreshold);
    LIB_FUNCTION("-tbc2SxQD60", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetSampleRate);
    LIB_FUNCTION("GZB2v0XnG0k", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetUserData);
    LIB_FUNCTION("JXRC5n0RQls", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemUnlock);
    LIB_FUNCTION("sU2St3agdjg", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamCreate);
    LIB_FUNCTION("I+RLwaauggA", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamCreateWithAllocator);
    LIB_FUNCTION("bfoMXnTRtwE", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamDestroy);
    LIB_FUNCTION("dxulc33msHM", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamQueryBufferSize);
    LIB_FUNCTION("rfw6ufRsmow", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamQueryInfo);
    LIB_FUNCTION("q+2W8YdK0F8", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamResetOption);
    LIB_FUNCTION("qQHCi9pjDps", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamRunCommands);
    LIB_FUNCTION("uu94irFOGpA", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceControl);
    LIB_FUNCTION("jjBVvPN9964", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetMatrixInfo);
    LIB_FUNCTION("W-Z8wWMBnhk", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetOwner);
    LIB_FUNCTION("WCayTgob7-o", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetPortInfo);
    LIB_FUNCTION("-TOuuAQ-buE", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetState);
    LIB_FUNCTION("rEh728kXk3w", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetStateFlags);
    LIB_FUNCTION("9eic4AmjGVI", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceQueryInfo);
    LIB_FUNCTION("AbYvTOZ8Pts", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceRunCommands);
};

} // namespace Libraries::Ngs2
