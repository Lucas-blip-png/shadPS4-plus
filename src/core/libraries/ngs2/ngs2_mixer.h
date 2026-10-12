// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "core/libraries/ngs2/ngs2.h"

namespace Libraries::Ngs2::Mixer {

bool IsEnabled();
std::string DescribeVoice(OrbisNgs2Handle voice);
u32 RackCount();

s32 SystemCreate(const OrbisNgs2SystemOption* option, const OrbisNgs2ContextBufferInfo* bufferInfo,
                 OrbisNgs2Handle* outHandle);
s32 SystemCreateWithAllocator(const OrbisNgs2SystemOption* option,
                              const OrbisNgs2BufferAllocator* allocator,
                              OrbisNgs2Handle* outHandle);
s32 SystemDestroy(OrbisNgs2Handle systemHandle, OrbisNgs2ContextBufferInfo* outBufferInfo);
s32 SystemRender(OrbisNgs2Handle systemHandle, const OrbisNgs2RenderBufferInfo* aBufferInfo,
                 u32 numBufferInfo);
s32 SystemLock(OrbisNgs2Handle systemHandle);
s32 SystemUnlock(OrbisNgs2Handle systemHandle);
s32 SystemGetUserData(OrbisNgs2Handle systemHandle, uintptr_t* outUserData);
s32 SystemSetUserData(OrbisNgs2Handle systemHandle, uintptr_t userData);
s32 SystemSetGrainSamples(OrbisNgs2Handle systemHandle, u32 numSamples);
s32 SystemSetSampleRate(OrbisNgs2Handle systemHandle, u32 sampleRate);

s32 RackQueryBufferSize(u32 rackId, const OrbisNgs2RackOption* option,
                        OrbisNgs2ContextBufferInfo* outBufferInfo);
s32 RackCreate(OrbisNgs2Handle systemHandle, u32 rackId, const OrbisNgs2RackOption* option,
               const OrbisNgs2ContextBufferInfo* bufferInfo, OrbisNgs2Handle* outHandle);
s32 RackCreateWithAllocator(OrbisNgs2Handle systemHandle, u32 rackId,
                            const OrbisNgs2RackOption* option,
                            const OrbisNgs2BufferAllocator* allocator, OrbisNgs2Handle* outHandle);
s32 RackDestroy(OrbisNgs2Handle rackHandle, OrbisNgs2ContextBufferInfo* outBufferInfo);
s32 RackGetVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex, OrbisNgs2Handle* outHandle);
s32 RackGetUserData(OrbisNgs2Handle rackHandle, uintptr_t* outUserData);
s32 RackSetUserData(OrbisNgs2Handle rackHandle, uintptr_t userData);
s32 RackLock(OrbisNgs2Handle rackHandle);
s32 RackUnlock(OrbisNgs2Handle rackHandle);

s32 VoiceControl(OrbisNgs2Handle voiceHandle, const OrbisNgs2VoiceParamHeader* paramList);
s32 VoiceGetState(OrbisNgs2Handle voiceHandle, OrbisNgs2VoiceState* outState, size_t stateSize);
s32 VoiceGetStateFlags(OrbisNgs2Handle voiceHandle, u32* outStateFlags);

} // namespace Libraries::Ngs2::Mixer
