#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <cmath>

#include <dlfcn.h>

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>

class ComputeBuffer {
public:
    VkBuffer buffer;
    VkDeviceMemory memory;
    ComputeBuffer(VkBuffer buffer, VkDeviceMemory memory): buffer(buffer), memory(memory) {};
    ~ComputeBuffer() {};
};

struct ComputeShader {
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    VkQueue queue;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    std::vector<ComputeBuffer> buffers;

    VkQueryPool queryPool;
    float timestampPeriod;

    VkDebugReportCallbackEXT reportcallback;
    VkDebugUtilsMessengerEXT msgcallback;
};

struct VulkanLib {
    void *lib;
    std::unique_ptr<std::map<std::string, void *>> symbols;
};

struct VulkanLib vklib;

#define OP_GET_FUNC(name) \
    auto name = reinterpret_cast<PFN_##name>(vklib.symbols->at(#name)); \
    if (!name) \
        std::cout << "fail to get function " << #name << std::endl

void checkInstanceExtension()
{
    uint32_t pPropertyCount;
    OP_GET_FUNC(vkEnumerateInstanceExtensionProperties);
    vkEnumerateInstanceExtensionProperties(nullptr, &pPropertyCount, nullptr);
    std::cout << "instance extension count: " << pPropertyCount << std::endl;
    std::vector<VkExtensionProperties> pProperties(pPropertyCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &pPropertyCount, pProperties.data());
    for (uint32_t i = 0; i < pPropertyCount; i++) {
        std::cout << "instance " << pProperties[i].extensionName << std::endl;
    }

}

void checkDeviceExtension(struct ComputeShader &shader)
{
    uint32_t _extensionCount = 0;
    OP_GET_FUNC(vkEnumerateDeviceExtensionProperties);
    vkEnumerateDeviceExtensionProperties( shader.physicalDevice, NULL, &_extensionCount, NULL);

    std::vector<VkExtensionProperties> extProps(_extensionCount);

    vkEnumerateDeviceExtensionProperties( shader.physicalDevice, NULL, &_extensionCount, extProps.data());
    for (uint32_t i = 0; i < _extensionCount; i++) {
        std::cout << extProps[i].extensionName << std::endl;
    }
}

VkInstance OpCreateInstance(std::vector<const char *> &enabledLayerNames) {
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Vulkan Compute Shader Benchmark";
    // vkGetPhysicalDeviceProperties2 requires 1.1.0
    // SPV_KHR_vulkan_memory_model, use_vulkan_memory_model in spirv requires 1.2.0
    applicationInfo.apiVersion = VK_MAKE_VERSION(1, 2, 0);
    applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.pEngineName = "Vulkan benchmark";
    applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;

    //enable debug and validation layers
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayerNames.size());
    instanceCreateInfo.ppEnabledLayerNames = enabledLayerNames.data();
#if VK_EXT_debug_utils
    std::vector<const char *> enabledExtensionNames{
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
        };
#else
    std::vector<const char *> enabledExtensionNames{
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME
        };
#endif
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNames.size());
    instanceCreateInfo.ppEnabledExtensionNames = enabledExtensionNames.data();
    // std::cout << "Enabling debug extension " << instanceCreateInfo.enabledExtensionCount << std::endl;


    VkInstance instance = VK_NULL_HANDLE;
    OP_GET_FUNC(vkCreateInstance);
    VkResult error = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    if (error != VK_SUCCESS) {
        std::cout << "fail to create instance " << error << std::endl;
        return nullptr;
    }

    return instance;
}

void OpCreateDevice(struct ComputeShader &shader, VkInstance instance, std::vector<const char *> enabledLayerNames)
{
    VkDevice device = VK_NULL_HANDLE;
    VkResult error;
    uint32_t count;

    OP_GET_FUNC(vkEnumeratePhysicalDevices);
    error = vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (error != VK_SUCCESS) {
        return;
    }
    std::vector<VkPhysicalDevice> physicalDevices(count);
    error = vkEnumeratePhysicalDevices(instance, &count, physicalDevices.data());
    if (error != VK_SUCCESS) {
        return;
    }

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    OP_GET_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
    uint32_t queueFamilyIndex = 0;
    for (auto device : physicalDevices) {
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProperties(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count,
                                                queueFamilyProperties.data());
        uint32_t index = 0;
        for (auto &properties : queueFamilyProperties) {
            if (properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice = device;
                queueFamilyIndex = index;
                break;
            }
            index++;
        }
        if (physicalDevice) {
            break;
        }
    }

    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
    subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_properties.pNext = nullptr;
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroup_properties;

    OP_GET_FUNC(vkGetPhysicalDeviceProperties2);
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
    shader.timestampPeriod = properties2.properties.limits.timestampPeriod;
    std::cout << "GPU " << properties2.properties.deviceName << std::endl;

    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    float queuePriority = 1.0f;  // specifies if this queue gets preference
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledLayerCount = enabledLayerNames.size();
    deviceCreateInfo.ppEnabledLayerNames = enabledLayerNames.data();

    OP_GET_FUNC(vkCreateDevice);
    error = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    if (error != VK_SUCCESS) {
        return;
    }

    VkQueue queue;
    OP_GET_FUNC(vkGetDeviceQueue);
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    shader.physicalDevice = physicalDevice;
    shader.queueFamilyIndex = queueFamilyIndex;
    shader.device = device;
    shader.queue = queue;
}


VkPipelineLayout OpCreatePipelineLayout(struct ComputeShader &shader, std::vector<VkDescriptorSetLayoutBinding> &layoutBindings)
{
    VkDevice device = shader.device;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkResult error;
 

    // use the descriptor bindings to define a layout to tell the driver where
    // descriptors are expected to live this is descriptor set 0 and refers to
    // set=0 in the shader
    VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo = {};
    setLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutCreateInfo.bindingCount = layoutBindings.size();
    setLayoutCreateInfo.pBindings = layoutBindings.data();

    OP_GET_FUNC(vkCreateDescriptorSetLayout);
    error = vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr,
                                        &setLayout);
    if (error != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    // pipeline layouts can consist of multiple descritor set layouts
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;  // but we only need one
    pipelineLayoutCreateInfo.pSetLayouts = &setLayout;

    OP_GET_FUNC(vkCreatePipelineLayout);
    error = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr,
                                    &pipelineLayout);
    if (error != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    shader.descriptorSetLayout = setLayout;
    return pipelineLayout;
}

// load a SPIR-V binary from file
std::vector<char> OpLoadShaderCode(const char *filename) {
  std::vector<char> shaderCode;
  if (FILE *fp = fopen(filename, "rb")) {
    char buf[1024];
    while (size_t len = fread(buf, 1, sizeof(buf), fp)) {
      shaderCode.insert(shaderCode.end(), buf, buf + len);
    }
    fclose(fp);
  }
  return shaderCode;
}

VkPipeline OpCreatePipeline(struct ComputeShader &shader, std::vector<VkDescriptorSetLayoutBinding> &layoutBindings, uint32_t loop_count)
{
    VkDevice device = shader.device;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult error;

    VkPipelineLayout pipelineLayout = OpCreatePipelineLayout(shader, layoutBindings);
    if (!pipelineLayout) {
        std::cout << "failed to create pipeline layout!" << std::endl;
        return VK_NULL_HANDLE;
    }
    shader.layout = pipelineLayout;

    // load vector_add.spv from file so we can create a pipeline
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    auto shaderCode = OpLoadShaderCode( "benchmark.spv");
    shaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t *>(shaderCode.data());
    shaderModuleCreateInfo.codeSize = shaderCode.size();

    OP_GET_FUNC(vkCreateShaderModule);
    error = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr,
                                 &shaderModule);
    if (error != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    VkSpecializationInfo spec_constant_info = {};
    VkSpecializationMapEntry spec_constant_entry = {};
    spec_constant_entry.constantID = 0;
    spec_constant_entry.offset = 0;
    spec_constant_entry.size = sizeof(uint32_t);

    spec_constant_info.mapEntryCount = 1;
    spec_constant_info.pMapEntries = &spec_constant_entry;
    spec_constant_info.dataSize = sizeof(uint32_t);
    spec_constant_info.pData = &loop_count;

    VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
    shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_info.pNext = nullptr;
    shader_stage_create_info.flags = 0;
    shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_create_info.module = shaderModule;
    shader_stage_create_info.pName = "main";
    shader_stage_create_info.pSpecializationInfo = &spec_constant_info;

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = nullptr;
    pipelineCreateInfo.flags = 0;
    pipelineCreateInfo.stage = shader_stage_create_info;
    pipelineCreateInfo.layout = pipelineLayout;

    OP_GET_FUNC(vkCreateComputePipelines);
    error = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                    &pipelineCreateInfo, nullptr, &pipeline);
    if (error) {
        return VK_NULL_HANDLE;
    }

    // a shader module can be destroyed after being consumed by the pipeline
    OP_GET_FUNC(vkDestroyShaderModule);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    return pipeline;
}


std::vector<VkDescriptorSetLayoutBinding> OpDescriptorSetLayoutBinding(void)
{
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    // describe the first SSBO input used in the vector_add shader
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = 0;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings.emplace_back(layoutBinding);

    layoutBinding.binding = 1;
    layoutBindings.emplace_back(layoutBinding);

    layoutBinding.binding = 2;
    layoutBindings.emplace_back(layoutBinding);

    return layoutBindings;
}


int32_t findMemoryTypeFromProperties(
    uint32_t memoryTypeBits,
    VkPhysicalDeviceMemoryProperties properties,
    VkMemoryPropertyFlags requiredProperties)
{
    if ((memoryTypeBits & (requiredProperties)) != requiredProperties) {
        return -1;
    }

    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if (((properties.memoryTypes[index].propertyFlags & requiredProperties) ==
            requiredProperties)) {
            return (int32_t)index;
        }
    }
    return -1;
}

VkResult __OpCreateBuffer(struct ComputeShader &shader, int bufferflags, int memoryflags, int num_element)
{
    VkDevice device = shader.device;
    uint32_t queueFamilyIndex = shader.queueFamilyIndex;
    VkResult error;
    // create the buffers which will hold the data to be consumed by shader
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = sizeof(uint32_t) * num_element;
    bufferCreateInfo.usage = bufferflags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 1;
    bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

    VkBuffer buffer;
    OP_GET_FUNC(vkCreateBuffer);
    error = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer);
    if (error) {
        std::cout << "failed to create buffer!" << std::endl;
        return error;
    }

    VkMemoryRequirements memoryRequirements;
    OP_GET_FUNC(vkGetBufferMemoryRequirements);
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkPhysicalDeviceMemoryProperties memoryProperties;
    OP_GET_FUNC(vkGetPhysicalDeviceMemoryProperties);
    vkGetPhysicalDeviceMemoryProperties(shader.physicalDevice, &memoryProperties);
    // std::cout << "memoryRequireBits: " << std::bitset<sizeof(memoryRequirements.memoryTypeBits)*8>(memoryRequirements.memoryTypeBits) << std::endl;
    // std::cout << "memoryTypeCount: " << memoryProperties.memoryTypeCount << std::endl;
    // for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    //     std::cout << "memoryType[" << i << "].propertyFlags: " << std::bitset<sizeof(memoryProperties.memoryTypes[i].propertyFlags)*8>(memoryProperties.memoryTypes[i].propertyFlags) << std::endl;
    // }
    // std::cout << "memoryflags" << std::bitset<sizeof(memoryflags)*8>(memoryflags) << std::endl;

    auto memoryTypeIndex = findMemoryTypeFromProperties(
        memoryRequirements.memoryTypeBits, memoryProperties,
        memoryflags);
    if (0 > memoryTypeIndex) {
        std::cout << "failed to find compatible memory type" << std::endl;
        return VK_ERROR_UNKNOWN;
    }

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    OP_GET_FUNC(vkAllocateMemory);
    error = vkAllocateMemory(device, &allocateInfo, nullptr, &memory);
    if (error) {
        std::cout << "failed to allocate memory!" << std::endl;
        return error;
    }

    OP_GET_FUNC(vkBindBufferMemory);
    error = vkBindBufferMemory(device, buffer, memory, 0);
    if (error) {
        return error;
    }

    ComputeBuffer combuffer(buffer, memory);
    shader.buffers.push_back(combuffer);

    return VK_SUCCESS;
}

VkResult OpCreateDescriptorPool(struct ComputeShader &shader,std::vector<VkDescriptorSetLayoutBinding> layoutBindings)
{
    VkDevice device = shader.device;
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // we only need one set in this example
    descriptorPoolCreateInfo.maxSets = 1;
    // and we only need one type of descriptor, when an application uses more
    // descriptor types a new pool is required for each descriptor type
    descriptorPoolCreateInfo.poolSizeCount = 1;
    VkDescriptorPoolSize poolSize = {};
    // we must provide the type of descriptor the pool will allocate
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // and the number of descriptors
    poolSize.descriptorCount = layoutBindings.size();
    descriptorPoolCreateInfo.pPoolSizes = &poolSize;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    OP_GET_FUNC(vkCreateDescriptorPool);
    vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr,
                            &descriptorPool);
    shader.descriptorPool = descriptorPool;

    return VK_SUCCESS;
}

VkResult OpAllocateDescriptorSets(struct ComputeShader &shader)
{
    VkDevice device = shader.device;
    VkResult error;

    // now we have our pool we can allocate a descriptor set
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = shader.descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    // this is the same layout we used to describe to the pipeline which
    // descriptors will be used
    descriptorSetAllocateInfo.pSetLayouts = &shader.descriptorSetLayout;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    OP_GET_FUNC(vkAllocateDescriptorSets);
    error = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
                                    &descriptorSet);
    if (error) {
        return error;
    }
    shader.descriptorSet = descriptorSet;

    return VK_SUCCESS;
}

VkResult OpWriteDescriptorSet(struct ComputeShader &shader, VkBuffer buffer, int index)
{
    
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = shader.descriptorSet;
    writeDescriptorSet.dstBinding = index;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet.pBufferInfo = &bufferInfo;

    OP_GET_FUNC(vkUpdateDescriptorSets);
    vkUpdateDescriptorSets(shader.device,1, &writeDescriptorSet, 0, nullptr);
    return VK_SUCCESS;
}

VkResult OpCreateBuffers(struct ComputeShader &shader, std::vector<VkDescriptorSetLayoutBinding> layoutBindings, int num_element)
{
    VkDevice device = shader.device;
    VkResult error;

    error = __OpCreateBuffer(shader, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, num_element);
    if (error) {
        std::cout << "failed to create buffer1 !" << std::endl;
        return error;
    }

    error = __OpCreateBuffer(shader, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, num_element);
    if (error) {
        std::cout << "failed to create buffer2 !" << std::endl;
        return error;
    }

    error = __OpCreateBuffer(shader, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, num_element);
    if (error) {
        std::cout << "failed to create buffer3 !" << std::endl;
        return error;
    }

    OpCreateDescriptorPool(shader, layoutBindings);

    OpAllocateDescriptorSets(shader);

    OpWriteDescriptorSet(shader, shader.buffers[0].buffer, 0);
    OpWriteDescriptorSet(shader, shader.buffers[1].buffer, 1);
    OpWriteDescriptorSet(shader, shader.buffers[2].buffer, 2);

    // as with descriptor sets command buffers are allocated from a pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // our command buffer will only be used once so we set the transient bit
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT|VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = shader.queueFamilyIndex;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    OP_GET_FUNC(vkCreateCommandPool);
    error = vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr,
                                &commandPool);
    if (error) {
        std::cout << "failed to create command pool!" << std::endl;
        return error;
    }
    shader.commandPool = commandPool;

    return VK_SUCCESS;
}

VkResult OpDispatchCommand(struct ComputeShader &shader, const int num_element)
{
    VkResult error;
    VkDevice device = shader.device;
    VkCommandPool commandPool = shader.commandPool;
    VkQueryPool queryPool = shader.queryPool;
    VkPipeline pipeline = shader.pipeline;
    VkPipelineLayout pipelineLayout = shader.layout;
    VkDescriptorSet descriptorSet = shader.descriptorSet;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    OP_GET_FUNC(vkAllocateCommandBuffers);
    error = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,
                                    &commandBuffer);
    if (error) {
        return error;
    }
    shader.commandBuffer = commandBuffer;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    OP_GET_FUNC(vkBeginCommandBuffer);
    OP_GET_FUNC(vkCmdBindPipeline);
    OP_GET_FUNC(vkCmdBindDescriptorSets);
    OP_GET_FUNC(vkCmdResetQueryPool);
    OP_GET_FUNC(vkCmdWriteTimestamp);
    OP_GET_FUNC(vkCmdDispatch);
    OP_GET_FUNC(vkEndCommandBuffer);

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);    

    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    vkCmdDispatch(commandBuffer, num_element/(4*16), 1, 1);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);

    error = vkEndCommandBuffer(commandBuffer);
    if (error) {
        return error;
    }
    return VK_SUCCESS;
}

void OpSubmitWork(struct ComputeShader &shader, const int num_element)
{
    VkDevice device = shader.device;

    VkDeviceSize size = num_element * sizeof(uint32_t);

    void *aptr = nullptr, *bptr = nullptr;
    OP_GET_FUNC(vkMapMemory);
    OP_GET_FUNC(vkUnmapMemory);
    OP_GET_FUNC(vkQueueSubmit);
    OP_GET_FUNC(vkQueueWaitIdle);

    vkMapMemory(device, shader.buffers[0].memory, 0, size, 0, &aptr);
    vkMapMemory(device, shader.buffers[1].memory, 0, size, 0, &bptr);
    if (!aptr || !bptr) {
        std::cout << "failed to map memory!" << std::endl;
        return;
    }

    float *aData = static_cast<float *>(aptr);
    float *bData = static_cast<float *>(bptr);

    for (auto index = 0; index < num_element; index++) {
        aData[index] = float((index % 9) + 1) * 0.1f;
        bData[index] = float((index % 5) + 1) * 1.f;
    }
    vkUnmapMemory(device, shader.buffers[0].memory);
    vkUnmapMemory(device, shader.buffers[1].memory);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &shader.commandBuffer;

#if 0
    VkFenceCreateInfo fence_create_info = {};
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_create_info.pNext = nullptr;
    fence_create_info.flags = 0;
    VkFence fence = VK_NULL_HANDLE;
    OP_GET_FUNC(vkCreateFence);
    vkCreateFence(shader.device, &fence_create_info, nullptr, &fence);

    vkQueueSubmit(shader.queue, 1, &submitInfo, fence);
#else
    vkQueueSubmit(shader.queue, 1, &submitInfo, nullptr);
#endif
    vkQueueWaitIdle(shader.queue);
#if 0
    std::cout << "waiting for fence" << std::endl;
    OP_GET_FUNC(vkWaitForFences);
    OP_GET_FUNC(vkDestroyFence);
    vkWaitForFences(shader.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(shader.device, fence, nullptr);
    std::cout << "fence done" << std::endl;
#endif
}

void OpVerifyWork(struct ComputeShader &shader, const int num_element)
{
    VkDevice device = shader.device;

    VkDeviceSize size = num_element * sizeof(uint32_t);

    void *cptr = nullptr;
    OP_GET_FUNC(vkMapMemory);
    OP_GET_FUNC(vkUnmapMemory);

    vkMapMemory(device, shader.buffers[2].memory, 0, size, 0, &cptr);

    // float *aData = static_cast<float *>(aptr);
    // float *bData = static_cast<float *>(bptr);
    float *rData = static_cast<float *>(cptr);
    for (auto i  = 0; i < num_element; i++) {
        if (std::fabs(rData[i] - float((i % 5) + 1) * 1.f *(1.f / (1.f - float((i % 9) + 1) * 0.1f))) > 0.01f) {
            std::cout << "Verification failed at index " << i << std::endl;
            std::cout << "Expected: " << float((i % 5) + 1) * 1.f *(1.f / (1.f - float((i % 9) + 1) * 0.1f)) << std::endl;
            std::cout << "Got: " << rData[i] << std::endl;
            // std::cout << "a: " << aData[i] << std::endl;
            // std::cout << "b: " << bData[i] << std::endl;
            break;
        }
    }

    vkUnmapMemory(device, shader.buffers[2].memory);
}

void OpDestroyHandles(struct ComputeShader &shader, VkInstance instance)
{
    VkDevice device = shader.device;
    VkCommandPool commandPool = shader.commandPool;

    OP_GET_FUNC(vkDestroyCommandPool);
    OP_GET_FUNC(vkFreeMemory);
    OP_GET_FUNC(vkDestroyBuffer);
    OP_GET_FUNC(vkDestroyQueryPool);
    OP_GET_FUNC(vkDestroyDescriptorPool);
    OP_GET_FUNC(vkDestroyPipeline);
    OP_GET_FUNC(vkDestroyPipelineLayout);
    OP_GET_FUNC(vkDestroyDescriptorSetLayout);
    OP_GET_FUNC(vkDestroyDevice);
    OP_GET_FUNC(vkGetInstanceProcAddr);
    OP_GET_FUNC(vkDestroyInstance);

    // destroy all the resources we created in reverse order
    vkDestroyCommandPool(device, commandPool, nullptr);
    
    for (auto b : shader.buffers) {
        vkFreeMemory(device, b.memory, nullptr);
        vkDestroyBuffer(device, b.buffer, nullptr);
    }

    // vkFreeCommandBuffers(device, commandPool, 1, &shader.commandBuffer);
    vkDestroyQueryPool(device, shader.queryPool, nullptr);
    vkDestroyDescriptorPool(device, shader.descriptorPool, nullptr);
    vkDestroyPipeline(device, shader.pipeline, nullptr);
    vkDestroyPipelineLayout(device, shader.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, shader.descriptorSetLayout, nullptr);
    vkDestroyDevice(device, nullptr);

    if (shader.reportcallback) {
        auto vkDestroyDebugReportCallbackEXT =
            reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
        if (vkDestroyDebugReportCallbackEXT)
            vkDestroyDebugReportCallbackEXT(instance, shader.reportcallback, nullptr);
    }

    if (shader.msgcallback) {
        auto vkDestroyDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

        if (vkDestroyDebugUtilsMessengerEXT)
            vkDestroyDebugUtilsMessengerEXT(instance, shader.msgcallback, nullptr);
    }

    vkDestroyInstance(instance, nullptr);
}

void OpCreateQueryPool(struct ComputeShader &shader)
{
    VkQueryPool queryPool;
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = 2;

    OP_GET_FUNC(vkCreateQueryPool);
    vkCreateQueryPool(shader.device, &queryPoolCreateInfo, nullptr, &queryPool);
    shader.queryPool = queryPool;
}

double OpGetTimestamp(struct ComputeShader &shader)
{
    VkQueryPool queryPool = shader.queryPool;
    VkQueryResultFlags flags = VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
    uint64_t timestamps[2];

    OP_GET_FUNC(vkGetQueryPoolResults);
    vkGetQueryPoolResults(shader.device, queryPool, 0, 2, 2 * sizeof(uint64_t),
                         timestamps, sizeof(uint64_t), flags);
    return (timestamps[1] - timestamps[0]) * shader.timestampPeriod * 1e-9;
}

#if VK_EXT_debug_utils
VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes __attribute__((unused)),
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData __attribute__((unused)))
{
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        fprintf(stderr, "%s [%d]: %s\n", pCallbackData->pMessageIdName,
            pCallbackData->messageIdNumber,
            pCallbackData->pMessage);
    } else {
        fprintf(stdout, "%s [%d]: %s\n", pCallbackData->pMessageIdName,
            pCallbackData->messageIdNumber,
            pCallbackData->pMessage);
    }
    return VK_FALSE;
}
VkResult OpCreateDebugUtilsCallback(struct ComputeShader &shader, VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT pfnCallback)
{
    VkResult error;
    OP_GET_FUNC(vkGetInstanceProcAddr);

    VkDebugUtilsMessengerCreateInfoEXT callbackCreateInfo;
    callbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    callbackCreateInfo.pNext = NULL;
    callbackCreateInfo.flags = 0;
    callbackCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    callbackCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    callbackCreateInfo.pfnUserCallback = pfnCallback;
    callbackCreateInfo.pUserData = nullptr;

    VkDebugUtilsMessengerEXT callback = VK_NULL_HANDLE;
    auto vkCreateDebugUtilsMessengerEXT =
      reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessengerEXT) {
        error = vkCreateDebugUtilsMessengerEXT(instance, &callbackCreateInfo, nullptr,
                                         &callback);
        if (error != VK_SUCCESS) {
            std::cerr << "Failed to create debug callback" << std::endl;
            return error;
        }
    }
    shader.msgcallback = callback;

    return error;
}
#else
VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
    VkDebugReportFlagsEXT flags __attribute__((unused)),
    VkDebugReportObjectTypeEXT objectType __attribute__((unused)),
    uint64_t object __attribute__((unused)),
    size_t location __attribute__((unused)),
    int32_t messageCode __attribute__((unused)),
    const char *pLayerPrefix __attribute__((unused)),
    const char *pMessage,
    void *pUserData __attribute__((unused)))
{
    fprintf(stderr, "%s\n", pMessage);
    return VK_FALSE;
}


VkResult OpCreateDebugReportCallback(struct ComputeShader &shader, VkInstance instance, PFN_vkDebugReportCallbackEXT pfnCallback)
{
    VkResult error;
    OP_GET_FUNC(vkGetInstanceProcAddr);
    VkDebugReportCallbackCreateInfoEXT callbackCreateInfo;
    callbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    callbackCreateInfo.pNext = nullptr;
    callbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
                                VK_DEBUG_REPORT_WARNING_BIT_EXT |
                                VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    callbackCreateInfo.pfnCallback = pfnCallback;
    callbackCreateInfo.pUserData = nullptr;
    VkDebugReportCallbackEXT callback = VK_NULL_HANDLE;
    auto vkCreateDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
          vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
    if (vkCreateDebugReportCallbackEXT) {
        error = vkCreateDebugReportCallbackEXT(instance, &callbackCreateInfo, nullptr,
                                         &callback);
        if (error != VK_SUCCESS) {
            std::cout << "Failed to create debug report callback" << std::endl;
        }
    }
    shader.reportcallback = callback;

    return error;
}
#endif

std::string findValidationLayerSupport() {
    uint32_t layerCount;
    OP_GET_FUNC(vkEnumerateInstanceLayerProperties);
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for (auto layer : availableLayers) {
        std::cout << "layer name " << layer.layerName << ": "<< layer.description << std::endl;
        // possible validation layers:
        // VK_LAYER_KHRONOS_validation
        // VK_LAYER_LUNARG_standard_validation
        if (strstr(layer.layerName, "validation")) {
            return std::string(layer.layerName);
        }
    }

    return nullptr;
}

void checkVulkanVersion()
{
    uint32_t version;
    OP_GET_FUNC(vkEnumerateInstanceVersion);
    vkEnumerateInstanceVersion(&version);
    std::cout << "version " << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version) << std::endl;
}

std::unique_ptr<std::map<std::string, void *>> loadLibrary(void) {
    std::unique_ptr<std::map<std::string, void *>> funcptr = std::make_unique<std::map<std::string, void *>>();
    const char *const name = "libvulkan.so.1";

    void *lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        std::cerr << "Failed to load library " << name << "," << dlerror() << std::endl;
        return nullptr;
    }
    vklib.lib = lib;
    const char *func_symbols[] = {
        "vkEnumerateInstanceVersion",
        "vkEnumerateInstanceLayerProperties",
        "vkCreateInstance",
        "vkEnumerateInstanceExtensionProperties",
        "vkGetInstanceProcAddr",
        "vkMapMemory",
        "vkUnmapMemory",
        "vkGetBufferMemoryRequirements",
        "vkGetPhysicalDeviceMemoryProperties",
        "vkAllocateMemory",
        "vkAllocateCommandBuffers",
        "vkBindBufferMemory",
        "vkCmdBindPipeline",
        "vkCmdDispatch",
        "vkCmdWriteTimestamp",
        "vkCmdBindDescriptorSets",
        "vkCmdResetQueryPool",
        "vkBeginCommandBuffer",
        "vkEndCommandBuffer",
        "vkQueueSubmit",
        "vkQueueWaitIdle",
        "vkCreateBuffer",
        "vkCreateQueryPool",
        "vkCreateDescriptorPool",
        "vkAllocateDescriptorSets",
        "vkUpdateDescriptorSets",
        "vkCreateCommandPool",
        "vkCreateComputePipelines",
        "vkCreateDevice",
        "vkGetDeviceQueue",
        "vkCreateDescriptorSetLayout",
        "vkCreatePipelineLayout",
        "vkDestroyBuffer",
        "vkDestroyQueryPool",
        "vkDestroyDescriptorPool",
        "vkDestroyPipeline",
        "vkDestroyPipelineLayout",
        "vkDestroyDescriptorSetLayout",
        "vkDestroyDevice",
        "vkDestroyInstance",
        "vkGetQueryPoolResults",
        "vkCreateShaderModule",
        "vkDestroyShaderModule",
        "vkDestroyCommandPool",
        "vkFreeMemory",
        "vkGetPhysicalDeviceQueueFamilyProperties",
        "vkGetPhysicalDeviceProperties2",
        "vkEnumeratePhysicalDevices",
        "vkEnumerateDeviceExtensionProperties",
        "vkResetCommandBuffer"
    };
    for (auto sym : func_symbols) {
        void *func = dlsym(lib, sym);
        if (!func) {
            std::cerr << "Failed to load symbol " << sym << "," << dlerror() << std::endl;
        }
        (*funcptr)[sym] = func;
    }
    return funcptr;
}

void unloadLibrary(void *lib) {
    dlclose(lib);
}

int main() {

    struct ComputeShader shader = {};
    vklib.symbols = loadLibrary();

    std::vector<const char *> enabledLayerNames;

    checkVulkanVersion();
    std::string str = findValidationLayerSupport();
    if (!str.empty()) {
        enabledLayerNames.push_back(str.c_str());
        std::cout << "validation layer found " << enabledLayerNames[0] << std::endl;
    }

    // checkInstanceExtension();
    VkInstance instance = OpCreateInstance(enabledLayerNames);
    if (!instance) {
        return -1;
    }

    const int num_element = 1024 * 1024;
    const uint32_t loop_count = 10000;

#if VK_EXT_debug_utils
    OpCreateDebugUtilsCallback(shader, instance, &debugUtilsCallback);
#else
    OpCreateDebugReportCallback(shader, instance, &debugReportCallback);
#endif

    OpCreateDevice(shader, instance, enabledLayerNames);
    
    // checkDeviceExtension(shader);

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings = OpDescriptorSetLayoutBinding();

    VkPipeline pipeline = OpCreatePipeline(shader, layoutBindings, loop_count);
    shader.pipeline = pipeline;

    if (OpCreateBuffers(shader, layoutBindings, num_element)) {
        std::cout << "Failed to create buffers" << std::endl;
        return -1;
    }

    OpCreateQueryPool(shader);

    OP_GET_FUNC(vkResetCommandBuffer);
    double duration = MAXFLOAT;
    for (int i = 0; i < 10; i++) {
        OpDispatchCommand(shader, num_element);
        OpSubmitWork(shader, num_element);

        duration = std::fmin(OpGetTimestamp(shader), duration);
        vkResetCommandBuffer(shader.commandBuffer, 0);
    }
    std::cout << "Duration: " << duration << "s" << std::endl;
    const double numOps = 2.f * 8.0f * double(num_element) * double(loop_count);
    double ops = numOps / duration;
    std::cout << "NumOps: " << ops << std::endl;
    std::cout << "Throughput: ";
    if (ops > 1.0f * 1e12) {
        std::cout << ops / 1e12 << "TFLOPS" << std::endl;
    } else if (ops > 1.0f * 1e9) {
        std::cout << ops / 1e9 << "GFLOPS" << std::endl;
    } else if (ops > 1.0f * 1e6) {
        std::cout << ops / 1e6 << "MFLOPS" << std::endl;
    } else if (ops > 1.0f * 1e3) {
        std::cout << ops / 1e3 << "KFLOPS" << std::endl;
    } else {
        std::cout << ops << "FLOPS" << std::endl;
    }

    OpVerifyWork(shader, num_element);
    OpDestroyHandles(shader, instance);
    unloadLibrary(vklib.lib);
    return 0;
}