
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include <glm/gtx/transform.hpp>

#include "vk_types.h"
#include "vk_initializers.h"

#include <fstream>
#include <iostream>

#define VK_CHECK(x) \
    do \
    { \
        VkResult err = x; \
        if (err) { \
            std::cout << "detected Vulkan error: " << err << std::endl; \
            abort(); \
        } \
    } while (0)

void VulkanEngine::init()
{
	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
	
	_window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_windowExtent.width,
		_windowExtent.height,
		window_flags
	);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_default_renderpass();
    init_framebuffers();	
    init_sync_structures();
    init_pipelines();
    load_meshes();

	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::cleanup()
{	
	if (_isInitialized) {
        vkWaitForFences(_device, 1, &_renderFence, true, 1000000000);

        _mainDeletionQueue.flush();
         
        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
		SDL_DestroyWindow(_window);
	}
}

void VulkanEngine::draw()
{
    // wait on CPU for rendering to finish
    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &_renderFence));

    // acquire next image to render into, and semaphore to pass to rendering
    // to know when that image is safe to use
    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000,
        _presentSemaphore, nullptr, &swapchainImageIndex));
    
    VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));
    auto cmd = _mainCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    VkClearValue clearValue;
    float flash = abs(sin(_frameNumber / 120.f));
    clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };

    VkClearValue depthClear;
    depthClear.depthStencil.depth = 1.f;

    VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(
        _renderPass, _framebuffers[swapchainImageIndex], _windowExtent);

    VkClearValue clearValues[2] = { clearValue, depthClear };

    rpInfo.clearValueCount = 2,
    rpInfo.pClearValues = &clearValues[0];
        
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
//        _selectedShader ? _trianglePipeline : _redTrianglePipeline);
        _meshPipeline);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &_monkeyMesh._vertexBuffer._buffer,
        &offset);


    glm::vec3 camPos = { 0.f, 0.f, -2.f};
    glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
    glm::mat4 projection = glm::perspective(glm::radians(70.f),
        17.f / 9.f, 0.1f, 200.f);
    projection[1][1] *= -1;
    glm::mat4 model = glm::rotate(glm::mat4(1.f),
        glm::radians(_frameNumber * 0.4f), glm::vec3(0.f, 1.f, 0.f));

    glm::mat4 mesh_matrix = projection * view * model;
    MeshPushConstants constants;
    constants.render_matrix = mesh_matrix;
    vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
        0, sizeof(MeshPushConstants), &constants);

    vkCmdDraw(cmd, _monkeyMesh._vertices.size(), 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &_presentSemaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &_renderSemaphore,
   };

    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &_renderSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &_swapchain,
        .pImageIndices = &swapchainImageIndex,
    };

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    _frameNumber++;
}

void VulkanEngine::run()
{
	SDL_Event e;
	bool bQuit = false;

	//main loop
	while (!bQuit)
	{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//close the window when user alt-f4s or clicks the X button			
			if (e.type == SDL_QUIT) {
                bQuit = true;
            } else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_SPACE) {
                    _selectedShader = (_selectedShader + 1) % 2;
                }
            }
		}

		draw();
	}
}
void VulkanEngine::init_vulkan() {
    vkb::InstanceBuilder builder;
    auto inst_ret = builder.set_app_name("Example Vulkan App")
            .request_validation_layers(true)
            .require_api_version(1, 1, 0)
            .use_default_debug_messenger()
            .build();

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 0)
        .set_surface(_surface)
        .select()
        .value();

    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice
        .get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo{
        .physicalDevice = _chosenGPU,
        .device = _device,
        .instance = _instance
    };
    vmaCreateAllocator(&allocatorInfo, &_allocator);
}

void VulkanEngine::init_swapchain() {
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface };
    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .use_default_format_selection()
    .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
    .set_desired_extent(_windowExtent.width, _windowExtent.height)
    .build()
    .value();

    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
    _swapchainImageFormat = vkbSwapchain.image_format;

    defer_delete([=]() {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    });

    VkExtent3D depthImageExtent{
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    _depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(
        _depthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        depthImageExtent
    );

    VmaAllocationCreateInfo dimg_alloc_info{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = VkMemoryPropertyFlags(
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    };

    vmaCreateImage(_allocator, &dimg_info, &dimg_alloc_info,
        &_depthImage._image, &_depthImage._allocation, nullptr);

    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
        _depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT
    );

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr,
        &_depthImageView));

    defer_delete([=]() {
        vkDestroyImageView(_device, _depthImageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage._image,
            _depthImage._allocation);
    });
}

void VulkanEngine::init_commands() {
    VkCommandPoolCreateInfo commandPoolInfo = 
        vkinit::command_pool_create_info(
            _graphicsQueueFamily,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
        &_commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(
            _commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
        &_mainCommandBuffer));

    defer_delete([=]() {
        vkDestroyCommandPool(_device, _commandPool, nullptr);
    });
}

void VulkanEngine::init_default_renderpass() {
    VkAttachmentDescription color_attachment{
        .format = _swapchainImageFormat,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    VkAttachmentReference color_attachment_ref{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentDescription depth_attachment{
        .format = _depthFormat,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };

    VkAttachmentReference depth_attachment_ref{
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = &depth_attachment_ref,
    };
   
    VkSubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
    };

    VkSubpassDependency depth_dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
    }; 

    VkAttachmentDescription attachments[2] = {
        color_attachment, depth_attachment };

    VkSubpassDependency dependencies[2] = {
        dependency, depth_dependency };

    VkRenderPassCreateInfo render_pass_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 2,
        .pAttachments = &attachments[0],
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 2,
        .pDependencies = &dependencies[0],
    };

    VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr,
        &_renderPass));

    defer_delete([=]() {
        vkDestroyRenderPass(_device, _renderPass, nullptr);
    });
}

void VulkanEngine::init_framebuffers() {
    VkFramebufferCreateInfo fb_info{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .renderPass = _renderPass,
        .attachmentCount = 1,
        .width = _windowExtent.width,
        .height = _windowExtent.height,
        .layers = 1
    };

    const auto swapchain_imagecount = _swapchainImages.size();
    _framebuffers.resize(swapchain_imagecount);

    for(auto i = 0; i < swapchain_imagecount; i++) {
        VkImageView attachments[2] = {
            _swapchainImageViews[i], _depthImageView };
        fb_info.pAttachments = &attachments[0];
        fb_info.attachmentCount = 2;

        VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, 
            &_framebuffers[i]));

        defer_delete([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
    } 
}
	
void VulkanEngine::init_sync_structures() {
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(
        VK_FENCE_CREATE_SIGNALED_BIT);

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

    VkSemaphoreCreateInfo semaphoreCreateInfo =
        vkinit::semaphore_create_info(0);
    
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
        &_presentSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
        &_renderSemaphore));

    defer_delete([=]() {
        vkDestroyFence(_device, _renderFence, nullptr);
        vkDestroySemaphore(_device, _presentSemaphore, nullptr);
        vkDestroySemaphore(_device, _renderSemaphore, nullptr); 
    });
}

bool VulkanEngine::load_shader_module(const char* filePath,
        VkShaderModule * outShaderModule) {
   std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = buffer.size() * sizeof(uint32_t),
        .pCode = buffer.data()
    };

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule)
        != VK_SUCCESS) {
        return false;
    }
    *outShaderModule = shaderModule;
    return true;
}

void VulkanEngine::init_pipelines() {
    VkShaderModule triangleFragShader;
    if (!load_shader_module("shaders/colored_triangle.frag.spv",
        &triangleFragShader)) {
        std::cout << "error building triangle frag shader" << std::endl;
    }
    VkShaderModule triangleVertShader;
    if (!load_shader_module("shaders/colored_triangle.vert.spv",
        &triangleVertShader)) {
        std::cout << "error building triangle vert shader" << std::endl;
    }
    
    VkShaderModule redTriangleFragShader;
    if (!load_shader_module("shaders/triangle.frag.spv",
        &redTriangleFragShader)) {
        std::cout << "error building red triangle frag shader" << std::endl;
    }

    VkShaderModule redTriangleVertShader;
    if (!load_shader_module("shaders/triangle.vert.spv",
        &redTriangleVertShader)) {
        std::cout << "error building red triangle vert shader" << std::endl;
    }


    VkPipelineLayoutCreateInfo pipeline_layout_info =
        vkinit::pipeline_layout_create_info();

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr,
        &_trianglePipelineLayout));

    PipelineBuilder pipelineBuilder{
        ._shaderStages = {
            vkinit::pipeline_shader_stage_create_info(
                VK_SHADER_STAGE_VERTEX_BIT, triangleVertShader),
            vkinit::pipeline_shader_stage_create_info(
                VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader)
        },
        ._vertexInputInfo = vkinit::vertex_input_state_create_info(),
        ._inputAssembly = vkinit::input_assembly_state_create_info(
                            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
        ._viewport = {
                        .x = 0.0f,
                        .y = 0.0f, 
                        .width = (float)_windowExtent.width,
                        .height = (float)_windowExtent.height,
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f,
        },
        ._scissor = { 
                        .offset = {0, 0},
                        .extent = _windowExtent
        },
        ._rasterizer = vkinit::rasterization_state_create_info(
                        VK_POLYGON_MODE_FILL),
        ._colorBlendAttachment = vkinit::color_blend_attachment_state(),
        ._multisampling = vkinit::multisampling_state_create_info(),
        ._pipelineLayout = _trianglePipelineLayout,
        ._depthStencil = vkinit::depth_stencil_create_info(true, true,
            VK_COMPARE_OP_LESS_OR_EQUAL),
    };
    
    _trianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

     pipelineBuilder
        ._shaderStages = {
            vkinit::pipeline_shader_stage_create_info(
                VK_SHADER_STAGE_VERTEX_BIT, redTriangleVertShader),
            vkinit::pipeline_shader_stage_create_info(
                VK_SHADER_STAGE_FRAGMENT_BIT, redTriangleFragShader)
        };

    _redTrianglePipeline = 
        pipelineBuilder.build_pipeline(_device, _renderPass);

    // build the mesh pipeline

    VertexInputDescription vertexDescription =
        Vertex::get_vertex_description();

    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions =
        vertexDescription.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount =
        vertexDescription.attributes.size();
    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions =
        vertexDescription.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount =
        vertexDescription.bindings.size();

    VkShaderModule meshVertShader;
    if (!load_shader_module("shaders/tri_mesh.vert.spv",
        &meshVertShader)) {
        std::cout << "error building triangle mesh vertex shader" << std::endl;
    }

    pipelineBuilder
        ._shaderStages = {
            vkinit::pipeline_shader_stage_create_info(
                VK_SHADER_STAGE_VERTEX_BIT, meshVertShader),
            vkinit::pipeline_shader_stage_create_info(
                VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader)
        };


    VkPipelineLayoutCreateInfo mesh_pipeline_layout_info =
        vkinit::pipeline_layout_create_info();

    VkPushConstantRange push_constant{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(MeshPushConstants),
    };
    mesh_pipeline_layout_info.pushConstantRangeCount = 1;
    mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
    VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info,
        nullptr, &_meshPipelineLayout));
    
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    _meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);

    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleVertShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);

    defer_delete([=]() {
        vkDestroyPipeline(_device, _redTrianglePipeline, nullptr);
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);

        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    });

 }


VkPipeline
PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass) {
    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .viewportCount = 1,
        .pViewports = &_viewport,
        .scissorCount = 1,
        .pScissors = &_scissor
    };

    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &_colorBlendAttachment
    };

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stageCount = (uint32_t)_shaderStages.size(),
        .pStages = _shaderStages.data(),
        .pVertexInputState = &_vertexInputInfo,
        .pInputAssemblyState = &_inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &_rasterizer,
        .pMultisampleState = &_multisampling,
        .pDepthStencilState = &_depthStencil,
        .pColorBlendState = &colorBlending,
        .layout = _pipelineLayout,
        .renderPass = pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE
    };

    VkPipeline newPipeline;
    if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
            nullptr, &newPipeline) != VK_SUCCESS) {
        std::cout << "failed to create pipeline" << std::endl;
        return VK_NULL_HANDLE;
    }
    return newPipeline;
}

void VulkanEngine::load_meshes() {
    _triangleMesh._vertices.resize(3);

    glm::vec3 positions[] = {
        { 1.f,  1.f, 0.f},
        {-1.f,  1.f, 0.f},
        { 0.f, -1.f, 0.f},
    };

    glm::vec3 green = { 0.f, 1.f, 0.f };

    for(int i = 0; i < 3; i++) {
        _triangleMesh._vertices[i].position = positions[i];
        _triangleMesh._vertices[i].color = green;
        
    }

    _monkeyMesh.load_from_obj("assets/monkey_smooth.obj");

    upload_mesh(_triangleMesh);
    upload_mesh(_monkeyMesh);
}

void VulkanEngine::upload_mesh(Mesh& mesh) {
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = mesh._vertices.size() * sizeof(Vertex),
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    };

    VmaAllocationCreateInfo vmaAllocInfo{
        .usage = VMA_MEMORY_USAGE_CPU_TO_GPU
    };

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
        &mesh._vertexBuffer._buffer,
        &mesh._vertexBuffer._allocation,
        nullptr));

    defer_delete([=]() {
        vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer,
            mesh._vertexBuffer._allocation);
    });

    void *data;
    vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);
    memcpy(data, mesh._vertices.data(),
        mesh._vertices.size() * sizeof(Vertex));

    vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}
