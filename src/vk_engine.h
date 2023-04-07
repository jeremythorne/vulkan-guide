// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <glm/glm.hpp>
#include "vk_types.h"
#include "vk_mesh.h"
#include <functional>
#include <string>
#include <unordered_map>
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

struct Material {
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct RenderObject {
    Mesh *mesh;
    Material *material;
    glm::mat4 transformMatrix;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct FrameData {
    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
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

    FrameData _frames[FRAME_OVERLAP];

    VkRenderPass _renderPass;
    std::vector<VkFramebuffer> _framebuffers;

    std::vector<RenderObject> _renderables;
    std::unordered_map<std::string,Material> _materials;
    std::unordered_map<std::string,Mesh> _meshes;

    VkImageView _depthImageView;
    AllocatedImage _depthImage;
    VkFormat _depthFormat;

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
    void init_scene();

    Material* create_material(VkPipeline pipeline, VkPipelineLayout layout,
        const std::string& name);

    Material* get_material(const std::string& name);

    Mesh* get_mesh(const std::string& name);

    void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

    FrameData& get_current_frame();

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
    VkPipelineDepthStencilStateCreateInfo _depthStencil;
    
    VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};


