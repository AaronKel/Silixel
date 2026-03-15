#include <LibSL/LibSL.h>
#include "simul_vulkan.h"
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#define CYCLE_BUFFER_LEN 1024

using namespace std;

// Globals for Vulkan
struct VulkanState {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkFence fence;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkPipelineLayout pipelineLayout;
    VkPipeline pipelineSimul;
    VkPipeline pipelinePosEdge;
    VkPipeline pipelineInit;
    VkPipeline pipelineOutPorts;
    VkCommandBuffer commandBufferSimul;

    // Buffers
    VkBuffer bufferCfg;
    VkDeviceMemory memoryCfg;
    VkBuffer bufferAddrs;
    VkDeviceMemory memoryAddrs;
    VkBuffer bufferOutputs;
    VkDeviceMemory memoryOutputs;
    VkBuffer bufferOutPortsLocs;
    VkDeviceMemory memoryOutPortsLocs;
    VkBuffer bufferOutPortsVals;
    VkDeviceMemory memoryOutPortsVals;
    VkBuffer bufferOutInits;
    VkDeviceMemory memoryOutInits;
    VkBuffer bufferChangeCount;
    VkDeviceMemory memoryChangeCount;

    // Async Readback
    VkBuffer bufferStaging;
    VkDeviceMemory memoryStaging;
    bool stagingReady = false;
} v;

extern map<string, v2i> g_OutPorts;
extern Array<int>       g_OutPortsValues;
extern int              g_Cycle;

// Helper to find memory type
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(v.physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw runtime_error("failed to find suitable memory type!");
}

// Helper to create buffer
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(v.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(v.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(v.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(v.device, buffer, bufferMemory, 0);
}

// Helper to load shader module
VkShaderModule createShaderModule(const string& filename) {
    ifstream file(filename, ios::ate | ios::binary);
    if (!file.is_open()) {
        throw runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(v.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

void simulInit_vulkan(const vector<t_lut>& luts, const vector<int>& ones) {
    // 1. Create Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Silixel Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &v.instance) != VK_SUCCESS) {
        throw runtime_error("failed to create vulkan instance!");
    }

    // 2. Pick Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(v.instance, &deviceCount, nullptr);
    vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(v.instance, &deviceCount, devices.data());
    v.physicalDevice = devices[0]; // Just pick first one

    // 3. Find Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(v.physicalDevice, &queueFamilyCount, nullptr);
    vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(v.physicalDevice, &queueFamilyCount, queueFamilies.data());
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            v.queueFamilyIndex = i;
            break;
        }
    }

    // 4. Create Logical Device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = v.queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (vkCreateDevice(v.physicalDevice, &deviceCreateInfo, nullptr, &v.device) != VK_SUCCESS) {
        throw runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(v.device, v.queueFamilyIndex, 0, &v.computeQueue);

    // 5. Create Buffers
    size_t n_luts = luts.size();
    createBuffer(n_luts * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, v.bufferCfg, v.memoryCfg);
    createBuffer(n_luts * 4 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, v.bufferAddrs, v.memoryAddrs);
    createBuffer(n_luts * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, v.bufferOutputs, v.memoryOutputs);
    createBuffer(g_OutPorts.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, v.bufferOutPortsLocs, v.memoryOutPortsLocs);
    createBuffer(g_OutPorts.size() * sizeof(uint32_t) * CYCLE_BUFFER_LEN, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, v.bufferOutPortsVals, v.memoryOutPortsVals);
    createBuffer(ones.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, v.bufferOutInits, v.memoryOutInits);
    createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, v.bufferChangeCount, v.memoryChangeCount);
    
    // Staging buffer for readback
    createBuffer(g_OutPorts.size() * sizeof(uint32_t) * CYCLE_BUFFER_LEN, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, v.bufferStaging, v.memoryStaging);

    // Populate static buffers
    void* data;
    vkMapMemory(v.device, v.memoryCfg, 0, n_luts * sizeof(uint32_t), 0, &data);
    for (size_t i = 0; i < n_luts; i++) ((uint32_t*)data)[i] = luts[i].cfg;
    vkUnmapMemory(v.device, v.memoryCfg);

    vkMapMemory(v.device, v.memoryAddrs, 0, n_luts * 4 * sizeof(uint32_t), 0, &data);
    for (size_t i = 0; i < n_luts; i++) {
        for (int j = 0; j < 4; j++) ((uint32_t*)data)[i*4+j] = max(0, luts[i].inputs[j]);
    }
    vkUnmapMemory(v.device, v.memoryAddrs);

    vkMapMemory(v.device, v.memoryOutPortsLocs, 0, g_OutPorts.size() * sizeof(uint32_t), 0, &data);
    for (auto op : g_OutPorts) ((uint32_t*)data)[op.second[0]] = op.second[1];
    vkUnmapMemory(v.device, v.memoryOutPortsLocs);

    vkMapMemory(v.device, v.memoryOutInits, 0, ones.size() * sizeof(uint32_t), 0, &data);
    for (size_t i = 0; i < ones.size(); i++) ((uint32_t*)data)[i] = ones[i];
    vkUnmapMemory(v.device, v.memoryOutInits);

    // 6. Descriptor Set Layout
    vector<VkDescriptorSetLayoutBinding> bindings(7);
    for (int i = 0; i < 7; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(v.device, &layoutInfo, nullptr, &v.descriptorSetLayout) != VK_SUCCESS) {
        throw runtime_error("failed to create descriptor set layout!");
    }

    // 7. Descriptor Pool & Set
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 7;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(v.device, &poolInfo, nullptr, &v.descriptorPool) != VK_SUCCESS) {
        throw runtime_error("failed to create descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = v.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &v.descriptorSetLayout;

    if (vkAllocateDescriptorSets(v.device, &allocInfo, &v.descriptorSet) != VK_SUCCESS) {
        throw runtime_error("failed to allocate descriptor sets!");
    }

    vector<VkWriteDescriptorSet> descriptorWrites(7);
    VkDescriptorBufferInfo bufferInfos[7];
    VkBuffer buffers[] = { v.bufferCfg, v.bufferAddrs, v.bufferOutputs, v.bufferOutPortsLocs, v.bufferOutPortsVals, v.bufferOutInits, v.bufferChangeCount };
    size_t sizes[] = { n_luts * sizeof(uint32_t), n_luts * 4 * sizeof(uint32_t), n_luts * sizeof(uint32_t), g_OutPorts.size() * sizeof(uint32_t), g_OutPorts.size() * sizeof(uint32_t) * CYCLE_BUFFER_LEN, ones.size() * sizeof(uint32_t), sizeof(uint32_t) };

    for (int i = 0; i < 7; i++) {
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = sizes[i];

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = v.descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(v.device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

    // 8. Pipeline Layout
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(uint32_t) * 2;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &v.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(v.device, &pipelineLayoutInfo, nullptr, &v.pipelineLayout) != VK_SUCCESS) {
        throw runtime_error("failed to create pipeline layout!");
    }

    // 9. Pipelines
    auto createComputePipeline = [](const string& spvFile) {
        VkShaderModule shaderModule = createShaderModule(spvFile);
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = v.pipelineLayout;

        VkPipeline pipeline;
        if (vkCreateComputePipelines(v.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw runtime_error("failed to create compute pipeline!");
        }
        vkDestroyShaderModule(v.device, shaderModule, nullptr);
        return pipeline;
    };

    v.pipelineSimul = createComputePipeline("src/vulkan/sh_simul.spv");
    v.pipelinePosEdge = createComputePipeline("src/vulkan/sh_posedge.spv");
    v.pipelineInit = createComputePipeline("src/vulkan/sh_init.spv");
    v.pipelineOutPorts = createComputePipeline("src/vulkan/sh_outports.spv");

    // 10. Command Pool & Buffer
    VkCommandPoolCreateInfo poolInfoCmd{};
    poolInfoCmd.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfoCmd.queueFamilyIndex = v.queueFamilyIndex;
    poolInfoCmd.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(v.device, &poolInfoCmd, nullptr, &v.commandPool) != VK_SUCCESS) {
        throw runtime_error("failed to create command pool!");
    }

    VkCommandBufferAllocateInfo allocInfoCmd{};
    allocInfoCmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfoCmd.commandPool = v.commandPool;
    allocInfoCmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfoCmd.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(v.device, &allocInfoCmd, &v.commandBuffer) != VK_SUCCESS) {
        throw runtime_error("failed to allocate command buffers!");
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(v.device, &fenceInfo, nullptr, &v.fence);

    VkCommandBufferAllocateInfo allocInfoSimul{};
    allocInfoSimul.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfoSimul.commandPool = v.commandPool;
    allocInfoSimul.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfoSimul.commandBufferCount = 1;
    vkAllocateCommandBuffers(v.device, &allocInfoSimul, &v.commandBufferSimul);
}

void simulBegin_vulkan(const vector<t_lut>& luts, const vector<int>& step_starts, const vector<int>& step_ends, const vector<int>& ones) {
    vkWaitForFences(v.device, 1, &v.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(v.device, 1, &v.fence);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(v.commandBuffer, &beginInfo);

    vkCmdBindDescriptorSets(v.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineLayout, 0, 1, &v.descriptorSet, 0, nullptr);

    // Initial clear of outputs (not strictly needed if we use sh_init)
    vkCmdFillBuffer(v.commandBuffer, v.bufferOutputs, 0, luts.size() * sizeof(uint32_t), 0);
    
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(v.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Init cells
    vkCmdBindPipeline(v.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineInit);
    uint32_t n_ones = (uint32_t)ones.size();
    vkCmdPushConstants(v.commandBuffer, v.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n_ones);
    vkCmdDispatch(v.commandBuffer, (n_ones + 127) / 128, 1, 1);

    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(v.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Resolve constant cells (simplified, just run simul and posedge once)
    vkCmdBindPipeline(v.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineSimul);
    uint32_t n_depth0 = step_ends[0] - step_starts[0] + 1;
    uint32_t push[2] = { (uint32_t)step_starts[0], n_depth0 };
    vkCmdPushConstants(v.commandBuffer, v.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2, push);
    vkCmdDispatch(v.commandBuffer, (n_depth0 + 127) / 128, 1, 1);

    vkCmdPipelineBarrier(v.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(v.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelinePosEdge);
    uint32_t n_luts = (uint32_t)luts.size();
    vkCmdPushConstants(v.commandBuffer, v.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n_luts);
    vkCmdDispatch(v.commandBuffer, (n_luts + 127) / 128, 1, 1);

    vkEndCommandBuffer(v.commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &v.commandBuffer;

    vkQueueSubmit(v.computeQueue, 1, &submitInfo, v.fence);
}

void simulCycle_vulkan(const vector<t_lut>& luts, const vector<int>& step_starts, const vector<int>& step_ends) {
    static bool recorded = false;
    if (!recorded) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(v.commandBufferSimul, &beginInfo);
        vkCmdBindDescriptorSets(v.commandBufferSimul, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineLayout, 0, 1, &v.descriptorSet, 0, nullptr);

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        // Simulation levels
        vkCmdBindPipeline(v.commandBufferSimul, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineSimul);
        for (size_t depth = 1; depth < step_starts.size(); depth++) {
            uint32_t n = step_ends[depth] - step_starts[depth] + 1;
            uint32_t push[2] = { (uint32_t)step_starts[depth], n };
            vkCmdPushConstants(v.commandBufferSimul, v.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2, push);
            vkCmdDispatch(v.commandBufferSimul, (n + 127) / 128, 1, 1);
            vkCmdPipelineBarrier(v.commandBufferSimul, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // PosEdge
        vkCmdBindPipeline(v.commandBufferSimul, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelinePosEdge);
        uint32_t n_luts = (uint32_t)luts.size();
        vkCmdPushConstants(v.commandBufferSimul, v.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n_luts);
        vkCmdDispatch(v.commandBufferSimul, (n_luts + 127) / 128, 1, 1);
        vkCmdPipelineBarrier(v.commandBufferSimul, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        vkEndCommandBuffer(v.commandBufferSimul);
        recorded = true;
    }

    // Command buffer for OutPorts (this one has dynamic offset so we record it)
    vkWaitForFences(v.device, 1, &v.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(v.device, 1, &v.fence);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(v.commandBuffer, &beginInfo);
    vkCmdBindDescriptorSets(v.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineLayout, 0, 1, &v.descriptorSet, 0, nullptr);

    // OutPorts
    extern uint32_t g_RBCycle;
    vkCmdBindPipeline(v.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipelineOutPorts);
    uint32_t n_ports = (uint32_t)g_OutPorts.size();
    uint32_t push_out[2] = { n_ports, n_ports * g_RBCycle };
    vkCmdPushConstants(v.commandBuffer, v.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2, push_out);
    vkCmdDispatch(v.commandBuffer, (n_ports + 31) / 32, 1, 1);
    vkEndCommandBuffer(v.commandBuffer);

    VkSubmitInfo submitInfo[2]{};
    submitInfo[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo[0].commandBufferCount = 1;
    submitInfo[0].pCommandBuffers = &v.commandBufferSimul;

    submitInfo[1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo[1].commandBufferCount = 1;
    submitInfo[1].pCommandBuffers = &v.commandBuffer;

    vkQueueSubmit(v.computeQueue, 2, submitInfo, v.fence);
    
    g_RBCycle++;
    g_Cycle++;
}

bool simulReadback_vulkan() {
    extern uint32_t g_RBCycle;
    if (g_RBCycle == CYCLE_BUFFER_LEN) {
        vkWaitForFences(v.device, 1, &v.fence, VK_TRUE, UINT64_MAX);
        
        // Copy to staging
        VkCommandBufferAllocateInfo allocInfoCmd{};
        allocInfoCmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfoCmd.commandPool = v.commandPool;
        allocInfoCmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfoCmd.commandBufferCount = 1;
        VkCommandBuffer tempCmd;
        vkAllocateCommandBuffers(v.device, &allocInfoCmd, &tempCmd);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(tempCmd, &beginInfo);
        
        uint32_t size = g_OutPorts.size() * sizeof(uint32_t) * CYCLE_BUFFER_LEN;
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(tempCmd, v.bufferOutPortsVals, v.bufferStaging, 1, &copyRegion);
        vkEndCommandBuffer(tempCmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &tempCmd;
        vkQueueSubmit(v.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(v.computeQueue);
        vkFreeCommandBuffers(v.device, v.commandPool, 1, &tempCmd);

        void* data;
        vkMapMemory(v.device, v.memoryStaging, 0, size, 0, &data);
        memcpy(g_OutPortsValues.raw(), data, size);
        vkUnmapMemory(v.device, v.memoryStaging);

        g_RBCycle = 0;
        return true;
    }
    return false;
}

void simulTerminate_vulkan() {
    vkDeviceWaitIdle(v.device);
    vkDestroyPipeline(v.device, v.pipelineSimul, nullptr);
    vkDestroyPipeline(v.device, v.pipelinePosEdge, nullptr);
    vkDestroyPipeline(v.device, v.pipelineInit, nullptr);
    vkDestroyPipeline(v.device, v.pipelineOutPorts, nullptr);
    vkDestroyPipelineLayout(v.device, v.pipelineLayout, nullptr);
    vkDestroyDescriptorPool(v.device, v.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(v.device, v.descriptorSetLayout, nullptr);
    
    auto destroyBuffer = [](VkBuffer& buffer, VkDeviceMemory& memory) {
        vkDestroyBuffer(v.device, buffer, nullptr);
        vkFreeMemory(v.device, memory, nullptr);
    };
    destroyBuffer(v.bufferCfg, v.memoryCfg);
    destroyBuffer(v.bufferAddrs, v.memoryAddrs);
    destroyBuffer(v.bufferOutputs, v.memoryOutputs);
    destroyBuffer(v.bufferOutPortsLocs, v.memoryOutPortsLocs);
    destroyBuffer(v.bufferOutPortsVals, v.memoryOutPortsVals);
    destroyBuffer(v.bufferOutInits, v.memoryOutInits);
    destroyBuffer(v.bufferChangeCount, v.memoryChangeCount);
    destroyBuffer(v.bufferStaging, v.memoryStaging);

    vkDestroyFence(v.device, v.fence, nullptr);
    vkDestroyCommandPool(v.device, v.commandPool, nullptr);
    vkDestroyDevice(v.device, nullptr);
    vkDestroyInstance(v.instance, nullptr);
}
