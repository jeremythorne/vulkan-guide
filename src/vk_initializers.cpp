#include <vk_initializers.h>

VkCommandPoolCreateInfo vkinit::command_pool_create_info(
    uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .queueFamilyIndex = queueFamilyIndex,
    };
}


VkCommandBufferAllocateInfo vkinit::command_buffer_allocate_info(
    VkCommandPool pool, uint32_t count, VkCommandBufferLevel level) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = level,
        .commandBufferCount = count,
    };
}

VkPipelineShaderStageCreateInfo vkinit::pipeline_shader_stage_create_info(
        VkShaderStageFlagBits stage, VkShaderModule shaderModule) {

    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = stage,
        .module = shaderModule,
        .pName = "main"
    };
}

VkPipelineVertexInputStateCreateInfo vkinit::vertex_input_state_create_info() {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
}

VkPipelineInputAssemblyStateCreateInfo
vkinit::input_assembly_state_create_info(VkPrimitiveTopology topology) {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .topology = topology,
        .primitiveRestartEnable = VK_FALSE
    };
}

VkPipelineRasterizationStateCreateInfo
vkinit::rasterization_state_create_info(VkPolygonMode polygonMode) {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = polygonMode,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };
}

VkPipelineMultisampleStateCreateInfo
vkinit::multisampling_state_create_info() {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE
    };
}

VkPipelineColorBlendAttachmentState vkinit::color_blend_attachment_state() {
    VkPipelineColorBlendAttachmentState color_blend_state{};

    color_blend_state
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | 
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    color_blend_state
        .blendEnable = VK_FALSE;
    return color_blend_state;
}


VkPipelineLayoutCreateInfo vkinit::pipeline_layout_create_info() {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    };
}

VkFenceCreateInfo vkinit::fence_create_info(VkFenceCreateFlags flags) {
    return {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags
    };
}

VkSemaphoreCreateInfo vkinit::semaphore_create_info(
    VkSemaphoreCreateFlags flags) {
    return {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags
    };
}

VkImageCreateInfo vkinit::image_create_info(VkFormat format,
        VkImageUsageFlags usageFlags, VkExtent3D extent) {

    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usageFlags,
    };
}

VkImageViewCreateInfo vkinit::imageview_create_info(VkFormat format,
        VkImage image, VkImageAspectFlags aspectFlags) {

    return {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
}

VkPipelineDepthStencilStateCreateInfo vkinit::depth_stencil_create_info(
        bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp) {

    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .depthTestEnable = (VkBool32)(bDepthTest ? VK_TRUE : VK_FALSE),
        .depthWriteEnable = (VkBool32)(bDepthWrite ? VK_TRUE : VK_FALSE),
        .depthCompareOp = bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .minDepthBounds = 0.f,
        .maxDepthBounds = 1.f,
    };
}

VkRenderPassBeginInfo vkinit::renderpass_begin_info(VkRenderPass renderPass,
        VkFramebuffer framebuffer, VkExtent2D windowExtent) {
    return {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = renderPass,
        .framebuffer = framebuffer,
        .renderArea = {
            .offset = {.x  = 0, .y = 0},
            .extent = windowExtent
        },
        .clearValueCount = 0,
        .pClearValues = nullptr
    };
}

VkDescriptorSetLayoutBinding vkinit::descriptorset_layout_binding(
    VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding) {
    return {
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = 1,
        .stageFlags = stageFlags,
    };
}

VkWriteDescriptorSet vkinit::write_descriptor_buffer(
        VkDescriptorType type, VkDescriptorSet dstSet, 
        VkDescriptorBufferInfo *bufferInfo, uint32_t binding) {
    return {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = dstSet,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = bufferInfo,
    };
}

VkCommandBufferBeginInfo vkinit::command_buffer_begin_info(
    VkCommandBufferUsageFlags flags) {
    return {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = flags,
        .pInheritanceInfo = nullptr,
    };
}

VkSubmitInfo vkinit::submit_info(VkCommandBuffer* cmd) {
    return {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = cmd,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
   };
}
