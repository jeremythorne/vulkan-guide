#include <vk_textures.h>
#include <iostream>

#include <vk_engine.h>
#include <vk_initializers.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

bool vkutil::load_image_from_file(VulkanEngine& engine, const char* file,
        AllocatedImage& outImage) {

    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(file, &texWidth, &texHeight, &texChannels,
        STBI_rgb_alpha);

    if (!pixels) {
        std::cout << "failed to load texture file " << file << std::endl;
        return false;
    }

    void* pixel_ptr = pixels;
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    VkFormat imageFormat = VK_FORMAT_R8G8B8A8_SRGB;
    AllocatedBuffer stagingBuffer = engine.create_buffer(imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    engine.with_buffer(stagingBuffer, [&](void* data) {
        memcpy(data, pixel_ptr, static_cast<size_t>(imageSize));
    });

    stbi_image_free(pixels);

    VkExtent3D imageExtent = {
        .width = static_cast<uint32_t>(texWidth),
        .height = static_cast<uint32_t>(texHeight),
        .depth = 1,
    };

    VkImageCreateInfo dimg_info = vkinit::image_create_info(imageFormat,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        imageExtent);

    AllocatedImage newImage;
    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(engine._allocator, &dimg_info, &dimg_allocinfo,
        &newImage._image, &newImage._allocation, nullptr);

    engine.immediate_submit([&](VkCommandBuffer cmd) {
        VkImageSubresourceRange range = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        };

        VkImageMemoryBarrier imageBarrier_toTransfer = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = newImage._image,
            .subresourceRange = range,
        };

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
            1, &imageBarrier_toTransfer);

        VkBufferImageCopy copyRegion = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageExtent = imageExtent,
        };

        vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer,
            newImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);

        VkImageMemoryBarrier imageBarrier_toReadable = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image = newImage._image,
            .subresourceRange = range,
        };

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
            1, &imageBarrier_toReadable);

    });

    engine.defer_delete([=](){
        vmaDestroyImage(engine._allocator, newImage._image,
            newImage._allocation);
    });

    outImage = newImage;

    return true;
}

