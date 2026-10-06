// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include <chrono>

#include "common/logging/log.h"
#include "shader_recompiler/info.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, PipelineLayoutCache* layouts,
                                 const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
                                 ComputePipelineKey compute_key_, const Shader::Info& info_,
                                 vk::ShaderModule module, SerializationSupport& sdata,
                                 bool preloading /*=false*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, layouts, true},
      compute_key{compute_key_} {
    auto& info = stages[int(Shader::SwStage::Compute)];
    info = &info_;
    const auto debug_str = GetDebugString();

    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    u32 binding{};
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& buffer : info->buffers) {
        // During deserialization, we don't have access to the UD to fetch sharp data. To address
        // this properly we need to track shaprs or portion of them in `sdata`, but since we're
        // interested only in "is storage" flag (which is not even effective atm), we can take a
        // shortcut there.
        const auto sharp = preloading ? AmdGpu::Buffer{} : buffer.GetSharp(*info);
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    for (const auto& image : info->images) {
        const u32 num_bindings = image.NumBindings(*info);
        bindings.push_back({
            .binding = binding,
            .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                               : vk::DescriptorType::eSampledImage,
            .descriptorCount = num_bindings,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
        has_desc_arrays |= num_bindings > 1;
        binding += num_bindings;
    }
    for (const auto& sampler : info->samplers) {
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    uses_push_descriptors = FitsPushDescriptors(binding);
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    AssignLayouts(bindings, flags, push_constants, debug_str);
    const auto device = instance.GetDevice();

    // Also runs on the optimize thread, so it only uses what lives as long as this pipeline.
    // Returns no pipeline when one may only come from the driver's cache and isn't there.
    const auto create = [device, cache = pipeline_cache, layout = pipeline_layout, module,
                         subgroup64 = instance.IsSubgroupSize64Supported(),
                         debug_str](vk::PipelineCreateFlags flags) -> vk::UniquePipeline {
        const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
            .requiredSubgroupSize = 64,
        };
        const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
            .flags = flags,
            .stage{
                .pNext = subgroup64 ? &subgroup_size_ci : nullptr,
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = module,
                .pName = "main",
            },
            .layout = layout,
        };
        const auto start = std::chrono::steady_clock::now();
        auto [pipeline_result, pipe] = device.createComputePipelineUnique(cache, compute_pipeline_ci);
        if (pipeline_result == vk::Result::ePipelineCompileRequired) {
            return {};
        }
        ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
                   vk::to_string(pipeline_result));
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (ms >= 20 || (flags & vk::PipelineCreateFlagBits::eDisableOptimization)) {
            LOG_INFO(Render_Vulkan, "{}compute pipeline {} built in {} ms",
                     flags & vk::PipelineCreateFlagBits::eDisableOptimization ? "Unoptimized "
                                                                              : "",
                     debug_str, ms);
        }
        SetObjectName(device, *pipe, "Compute Pipeline {}", debug_str);
        return std::move(pipe);
    };

    // A dispatch waits on a pipeline met in game, and compute work can't be skipped. One the
    // driver built before comes from its disk cache at once. A new one takes about 100 ms, and a
    // new effect brings a few dozen, which froze the game for seconds: one built without
    // optimizations, several times quicker, is used now and the optimized one replaces it.
    if (!preloading && instance.IsPipelineCreationCacheControlSupported() &&
        EmulatorSettings.IsComputeFastCompile()) {
        if (!EmulatorSettings.IsComputeFastCompileForce()) {
            pipeline = create(vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
        }
        if (!pipeline) {
            optimize_job = std::make_shared<OptimizeJob>();
            SubmitOptimize(optimize_job, [create] { return create({}); });
            pipeline = create(vk::PipelineCreateFlagBits::eDisableOptimization);
        }
    } else {
        pipeline = create({});
    }
}

ComputePipeline::~ComputePipeline() {
    CancelOptimize();
}

} // namespace Vulkan
