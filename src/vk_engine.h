// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <glm/glm.hpp>
#include "vk_types.h"
#include "vk_mesh.h"
#include <functional>
#include <vector>

struct DeletionQueue {
    std::vector<std::function<void()>> deletors;

    void push(std::function<void()>&& function) {
        deletors.push_back(function);
    }

    void flush() {
        for(auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();
        }
        deletors.clear();
    }
};

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
};

class VulkanEngine {
public:
	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

private:
	bool _isInitialized{ false };
	int _frameNumber {0};
    int _selectedShader = 0;
    
    DeletionQueue _mainDeletionQueue;

	VkExtent2D _windowExtent{ 850 , 450 };

	struct SDL_Window* _window{ nullptr };

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;
    VkSurfaceKHR _surface;

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkRenderPass _renderPass;
    std::vector<VkFramebuffer> _framebuffers;

    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkPipelineLayout _trianglePipelineLayout;
    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _trianglePipeline;
    VkPipeline _redTrianglePipeline;
    VkPipeline _meshPipeline;
    Mesh _triangleMesh;
    Mesh _monkeyMesh;

    VmaAllocator _allocator;

    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_default_renderpass();
    void init_framebuffers();
    void init_sync_structures();
    bool load_shader_module(const char* filePath,
        VkShaderModule * outShaderModule);
    void init_pipelines();
    void load_meshes();
    void upload_mesh(Mesh& mesh);

    void defer_delete(std::function<void()>&& function) {
        _mainDeletionQueue.push(
            std::forward<std::function<void()>>(function));
    }
};

class PipelineBuilder {
public:
    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkViewport _viewport;
    VkRect2D _scissor;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
    VkPipelineLayout _pipelineLayout;
    
    VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};


