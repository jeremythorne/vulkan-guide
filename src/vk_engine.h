// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <glm/glm.hpp>
#include "vk_types.h"
#include "vk_mesh.h"
#include "vk_textures.h"
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

struct Texture {
    AllocatedImage _image;
    VkImageView _imageView;
};

struct MeshPushConstants {
    uint32_t index;
};

struct Material {
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct UploadContext {
    VkFence _uploadFence;
    VkCommandPool _commandPool;
    VkCommandBuffer _commandBuffer;
};

struct RenderObject {
    Mesh *mesh;
    Material *material;
    glm::mat4 transformMatrix;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct GPUCameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
};

struct GPUSceneData {
    glm::vec4 fogColor;
    glm::vec4 fogDistances;
    glm::vec4 ambientColor;
    glm::vec4 sunlighDirection;
    glm::vec4 sunlightColor;
};

struct FrameData {
    VkSemaphore _presentSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    AllocatedBuffer _cameraBuffer;
    VkDescriptorSet _globalDescriptor;

    AllocatedBuffer _objectBuffer;
    VkDescriptorSet _objectDescriptor;
};

struct GPUObjectData {
    glm::mat4 modelMatrix;
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

    //create, upload, delete utils
    AllocatedBuffer create_buffer(size_t allocSize,
        VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    void with_buffer(AllocatedBuffer &buffer,
        std::function<void(void *)> function);

    void immediate_submit(
        std::function<void(VkCommandBuffer cmd)>&& function);

    void defer_delete(std::function<void()>&& function) {
        _mainDeletionQueue.push(
            std::forward<std::function<void()>>(function));
    }

    VmaAllocator _allocator;
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

    UploadContext _uploadContext;

    VkRenderPass _renderPass;
    std::vector<VkFramebuffer> _framebuffers;

    std::vector<RenderObject> _renderables;
    std::unordered_map<std::string, Material> _materials;
    std::unordered_map<std::string, Mesh> _meshes;
    std::unordered_map<std::string, Texture> _loadedTextures;

    VkImageView _depthImageView;
    AllocatedImage _depthImage;
    VkFormat _depthFormat;

    VkDescriptorSetLayout _globalSetLayout;
    VkDescriptorSetLayout _objectSetLayout;
    VkDescriptorPool _descriptorPool;

    VkPhysicalDeviceProperties _gpuProperties;

    GPUSceneData _sceneParameters;
    AllocatedBuffer _sceneParameterBuffer;


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
    void load_images();
    void init_scene();
    void init_descriptors();

    Material* create_material(VkPipeline pipeline, VkPipelineLayout layout,
        const std::string& name);

    Material* get_material(const std::string& name);

    Mesh* get_mesh(const std::string& name);

    void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

    FrameData& get_current_frame();

    size_t pad_uniform_buffer_size(size_t originalSize);
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


