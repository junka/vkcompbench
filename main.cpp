#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>
#include <iomanip>

#include <dlfcn.h>

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>

#include "benchmark.h"

class VulkanLib {
private:
    void *lib;
public:
    std::unique_ptr<std::map<std::string, void *>> symbols;
    VulkanLib() {
    symbols = std::make_unique<std::map<std::string, void *>>();
        const char *const name = "libvulkan.so.1";

        lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
        if (!lib) {
            std::cerr << "Failed to load library " << name << "," << dlerror() << std::endl;
            return ;
        }
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
            "vkResetCommandBuffer",
            "vkGetPhysicalDeviceFeatures",
            "vkGetPhysicalDeviceFeatures2"
        };
        for (auto sym : func_symbols) {
            void *func = dlsym(lib, sym);
            if (!func) {
                std::cerr << "Failed to load symbol " << sym << "," << dlerror() << std::endl;
            }
            (*symbols)[sym] = func;
        }
    }
    ~VulkanLib() {
        dlclose(lib);
    }

};

VulkanLib vklib;
#define OP_GET_FUNC(name) \
    auto name = reinterpret_cast<PFN_##name>(vklib.symbols->at(#name)); \
    if (!name) \
        std::cout << "fail to get function " << #name << std::endl

class ComputeBuffer {
public:
    VkBuffer buffer;
    VkDeviceMemory memory;
    ComputeBuffer(VkBuffer buffer, VkDeviceMemory memory): buffer(buffer), memory(memory) {};
    ~ComputeBuffer() {};
};

class ComputeDevice {
private:
    std::vector<VkExtensionProperties> ext_properties;
    VkPhysicalDeviceProperties deviceProperties;
    void checkDeviceDataTypeFeatures(void)
    {
        VkPhysicalDeviceFeatures deviceFeatures = {};
        OP_GET_FUNC(vkGetPhysicalDeviceFeatures);
        vkGetPhysicalDeviceFeatures( physicalDevice, &deviceFeatures);
        int64 = deviceFeatures.shaderInt64;
        fp64 = deviceFeatures.shaderFloat64;
        //fp32, int32 shuold be supported by default
        int16 = deviceFeatures.shaderInt16;

        OP_GET_FUNC(vkGetPhysicalDeviceFeatures2);
        VkPhysicalDeviceShaderFloat16Int8Features float16Int8Features = {};
        float16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
#if VK_KHR_shader_integer_dot_product
        VkPhysicalDeviceShaderIntegerDotProductFeatures integerDotProductFeatures = {};
        integerDotProductFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        float16Int8Features.pNext = &integerDotProductFeatures;
#endif


        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &float16Int8Features;

        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
        fp16 = float16Int8Features.shaderFloat16;
        int8 = float16Int8Features.shaderInt8;
#if VK_KHR_shader_integer_dot_product
        dot = integerDotProductFeatures.shaderIntegerDotProduct;
#else
        dot = false;
#endif
    }

    void checkDeviceExtension(void)
    {
        uint32_t extensionCount = 0;
        OP_GET_FUNC(vkEnumerateDeviceExtensionProperties);
        vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &extensionCount, NULL);
        ext_properties.resize(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &extensionCount, ext_properties.data());
        // std::cout << "Device Extensions:" << std::endl;
        // for (uint32_t i = 0; i < extensionCount; i++) {
        //     std::cout << ext_properties[i].extensionName << ":" << ext_properties[i].specVersion << std::endl;
        // }
    }
    bool checkDeviceExtensionFeature(const char *name)
    {
        for (auto ext : ext_properties) {
            if (std::string(ext.extensionName).compare(name) == 0) {
                return true;
            }
        }
        return false;
    }
#if VK_KHR_shader_integer_dot_product
    void check_shader_integer_dot_product_support() {

        VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR integerDotProductProperties = {};
        integerDotProductProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2 properties2 = {};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &integerDotProductProperties;

        OP_GET_FUNC(vkGetPhysicalDeviceProperties2);
        vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);

        // Check for supported integer dot product features
        int8dot = integerDotProductProperties.integerDotProduct8BitUnsignedAccelerated;
        int8dot4x8packed = integerDotProductProperties.integerDotProduct4x8BitPackedUnsignedAccelerated;
        int8dotaccsat = integerDotProductProperties.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
    }
#endif
    void getDeviceTimeLimits(void)
    {
        VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
        subgroup_properties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroup_properties.pNext = nullptr;
        VkPhysicalDeviceProperties2 properties2 = {};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &subgroup_properties;
        OP_GET_FUNC(vkGetPhysicalDeviceProperties2);
        vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
        deviceProperties = properties2.properties;
        timestampPeriod = deviceProperties.limits.timestampPeriod;
        
        std::cout << "GPU " << deviceProperties.deviceName << std::endl;
    }

    VkResult createDevice(std::vector<const char *> enabledLayerNames)
    {
        std::vector<uintptr_t> enabledFeatures;
        std::vector<const char *> enabledExtensions;
        VkPhysicalDeviceFeatures features = {};
        features.robustBufferAccess = VK_TRUE;
        if (int64)
            features.shaderInt64 = VK_TRUE;
        if (fp64)
            features.shaderFloat64 = VK_TRUE;
        if (int16)
            features.shaderInt16 = VK_TRUE;

        VkPhysicalDeviceFloat16Int8FeaturesKHR float16Int8Features = {};
        float16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;

        VkPhysicalDevice8BitStorageFeatures storage8bitFeatures = {};
        storage8bitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        storage8bitFeatures.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        storage8bitFeatures.storageBuffer8BitAccess = VK_TRUE;

#ifdef VK_KHR_16bit_storage
        VkPhysicalDevice16BitStorageFeatures storage16bitFeatures = {};
        storage16bitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage16bitFeatures.uniformAndStorageBuffer16BitAccess = VK_TRUE;
        storage16bitFeatures.storageBuffer16BitAccess = VK_TRUE;
        storage16bitFeatures.storageInputOutput16 = VK_TRUE;
#elif defined VK_VERSION_1_1
        VkPhysicalDeviceVulkan11Features storage16bitFeatures = {};
        storage16bitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        storage16bitFeatures.storageBuffer16BitAccess = VK_TRUE;
        storage16bitFeatures.storageInputOutput16 = VK_TRUE;
        storage16bitFeatures.uniformAndStorageBuffer16BitAccess = VK_TRUE;
#endif

#ifdef VK_KHR_shader_integer_dot_product
        VkPhysicalDeviceShaderIntegerDotProductFeatures shaderIntegerDotProductFeatures = {};
        shaderIntegerDotProductFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        shaderIntegerDotProductFeatures.shaderIntegerDotProduct = VK_TRUE;
#elif defined VK_VERSION_1_3
        VkPhysicalDeviceVulkan13Features features13 = {};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.shaderIntegerDotProduct = VK_TRUE;
#endif
        if (int8) {
            float16Int8Features.shaderInt8 = VK_TRUE;
            if (checkDeviceExtensionFeature(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
                enabledExtensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
                enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&storage8bitFeatures));
            }
        }
        if (fp16) {
            float16Int8Features.shaderFloat16 = VK_TRUE;
            if (checkDeviceExtensionFeature(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
                enabledExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
                if (deviceProperties.vendorID != 4318) {
                    // tested on Nvidia A2000, it supports 16bit storage feature but did not need to enable it
                    // enable it will cause validation error VK_ERROR_FEATURE_NOT_PRESENT
                    enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&storage16bitFeatures));
                }
            }
        }
        if (int8 || fp16) {
            if (checkDeviceExtensionFeature(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
                enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&float16Int8Features));
                enabledExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
            }
        }
        if (dot) {
#ifdef VK_KHR_shader_integer_dot_product
            if (checkDeviceExtensionFeature(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
                enabledExtensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
                enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&shaderIntegerDotProductFeatures));
            }
#elif defined VK_VERSION_1_3
            enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&features13));
#endif
        }

        struct GeneralFeature {
            VkStructureType sType;
            void*     pNext;
        };
        void* pFirst = nullptr;
        if (enabledFeatures.size() > 0) {
            pFirst = reinterpret_cast<void *>(enabledFeatures[0]);
            struct GeneralFeature* ptr = reinterpret_cast<struct GeneralFeature*>(pFirst);
            for (size_t i = 1; i < enabledFeatures.size(); i++) {
                struct GeneralFeature* feat = reinterpret_cast<struct GeneralFeature*>(enabledFeatures[i]);
                ptr->pNext = feat;
                ptr = feat;
            }
        }

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
        deviceCreateInfo.pEnabledFeatures = &features;
        deviceCreateInfo.enabledExtensionCount = enabledExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
        deviceCreateInfo.pNext = pFirst;

        OP_GET_FUNC(vkCreateDevice);
        VkResult error = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        return error;
    }

    void getDeviceQueue(void)
    {
        OP_GET_FUNC(vkGetDeviceQueue);
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

public:

    ComputeDevice(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, std::vector<const char *> enabledLayerNames):
        physicalDevice(physicalDevice), queueFamilyIndex(queueFamilyIndex) {
        checkDeviceDataTypeFeatures();
        checkDeviceExtension();
        getDeviceTimeLimits();
        VkResult err = createDevice(enabledLayerNames);
        if (err != VK_SUCCESS) {
            std::string errstrings[] = {
                "VK_ERROR_OUT_OF_HOST_MEMORY",
                "VK_ERROR_OUT_OF_DEVICE_MEMORY",
                "VK_ERROR_INITIALIZATION_FAILED",
                "VK_ERROR_DEVICE_LOST",
                "VK_ERROR_EXTENSION_NOT_PRESENT",
                "VK_ERROR_FEATURE_NOT_PRESENT",
                "VK_ERROR_TOO_MANY_OBJECTS",
            };
            int index = 0;
            if (err == VK_ERROR_OUT_OF_HOST_MEMORY) {
                index = 0;
            } else if (err == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
                index = 1;
            } else if (err == VK_ERROR_INITIALIZATION_FAILED) {
                index = 2;
            } else if (err == VK_ERROR_DEVICE_LOST) {
                index = 3;
            } else if (err ==VK_ERROR_EXTENSION_NOT_PRESENT) {
                index = 4;
            } else if (err == VK_ERROR_FEATURE_NOT_PRESENT) {
                index = 5;
            } else if (err == VK_ERROR_TOO_MANY_OBJECTS) {
                index = 6;
            }
            std::cout << "Failed to create device " << errstrings[index] << std::endl;
            throw 1;
        }
        getDeviceQueue();
#if VK_KHR_shader_integer_dot_product
        if (dot)
            check_shader_integer_dot_product_support();
#endif
    };
    ~ComputeDevice() {
        OP_GET_FUNC(vkDestroyDevice);
        vkDestroyDevice(device, nullptr);
    };

    VkDevice device;
    VkPhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    VkQueue queue;

    float timestampPeriod;

    bool int64;
    bool fp64;
    // int32 and fp32 should be supported by default
    bool int16;
    bool fp16;
    bool int8;
    bool dot;
#ifdef VK_KHR_shader_integer_dot_product
    bool int8dot;
    bool int8dot4x8packed;
    bool int8dotaccsat;
#endif

};

struct ComputeShader {
    std::shared_ptr<ComputeDevice> device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    std::vector<ComputeBuffer> buffers;

    VkQueryPool queryPool;
};

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

    // enable debug and validation layers
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

    VkInstance instance = VK_NULL_HANDLE;
    OP_GET_FUNC(vkCreateInstance);
    VkResult error = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    if (error != VK_SUCCESS) {
        std::cout << "fail to create instance " << error << std::endl;
        return nullptr;
    }

    return instance;
}

std::unique_ptr<ComputeDevice> OpCreateDevice(VkInstance instance, std::vector<const char *> enabledLayerNames)
{
    VkResult error;
    uint32_t count;

    OP_GET_FUNC(vkEnumeratePhysicalDevices);
    error = vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (error != VK_SUCCESS) {
        return nullptr;
    }
    std::vector<VkPhysicalDevice> physicalDevices(count);
    error = vkEnumeratePhysicalDevices(instance, &count, physicalDevices.data());
    if (error != VK_SUCCESS) {
        return nullptr;
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

    std::unique_ptr<ComputeDevice> dev = std::make_unique<ComputeDevice>(physicalDevice, queueFamilyIndex, enabledLayerNames);


    return dev;
}


VkPipelineLayout OpCreatePipelineLayout(struct ComputeShader &shader, std::vector<VkDescriptorSetLayoutBinding> &layoutBindings)
{
    VkDevice device = shader.device->device;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkResult error;

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
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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

VkPipeline OpCreatePipeline(struct ComputeShader &shader, std::vector<VkDescriptorSetLayoutBinding> &layoutBindings,
                            uint32_t loop_count, unsigned int code_size, unsigned char *code)
{
    VkDevice device = shader.device->device;
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
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(code);
    shaderModuleCreateInfo.codeSize = code_size;

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

VkResult __OpCreateBuffer(struct ComputeShader &shader, int bufferflags, int memoryflags, int num_element, size_t element_size)
{
    VkDevice device = shader.device->device;
    uint32_t queueFamilyIndex = shader.device->queueFamilyIndex;
    VkResult error;
    // create the buffers which will hold the data to be consumed by shader
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = element_size * num_element;
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
    vkGetPhysicalDeviceMemoryProperties(shader.device->physicalDevice, &memoryProperties);
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
    VkDevice device = shader.device->device;
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
    VkDevice device = shader.device->device;
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
    vkUpdateDescriptorSets(shader.device->device,1, &writeDescriptorSet, 0, nullptr);
    return VK_SUCCESS;
}

VkResult OpCreateBuffers(struct ComputeShader &shader, std::vector<VkDescriptorSetLayoutBinding> layoutBindings, int num_element, size_t element_size)
{
    VkDevice device = shader.device->device;
    VkResult error;

    error = __OpCreateBuffer(shader, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             num_element, element_size);
    if (error) {
        std::cout << "failed to create buffer1 !" << std::endl;
        return error;
    }

    error = __OpCreateBuffer(shader, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             num_element, element_size);
    if (error) {
        std::cout << "failed to create buffer2 !" << std::endl;
        return error;
    }

    error = __OpCreateBuffer(shader, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             num_element, element_size);
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
    commandPoolCreateInfo.queueFamilyIndex = shader.device->queueFamilyIndex;
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
    VkDevice device = shader.device->device;
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

template<typename T>
void OpSubmitWork(struct ComputeShader &shader, const int num_element)
{
    VkDevice device = shader.device->device;

    VkDeviceSize size = num_element * sizeof(T);

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

    T *aData = static_cast<T *>(aptr);
    T *bData = static_cast<T *>(bptr);
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, _Float64>
#ifdef HAVE_FLOAT16
                  || std::is_same_v<T, _Float16>
#endif
                  ) {
        for (auto i = 0; i < num_element; i++) {
            aData[i] = T((i % 9)+1) * T(0.1f);
            bData[i] = T((i % 5)+1) * T(1.f);
        }
    } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t> ||
                         std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t>) {
        for (auto i = 0; i < num_element; i++) {
           aData[i] = 1;
           bData[i] = 1;
        }
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
    vkCreateFence(device, &fence_create_info, nullptr, &fence);

    vkQueueSubmit(dev->queue, 1, &submitInfo, fence);
#else
    vkQueueSubmit(shader.device->queue, 1, &submitInfo, nullptr);
#endif
    vkQueueWaitIdle(shader.device->queue);
#if 0
    // std::cout << "waiting for fence" << std::endl;
    OP_GET_FUNC(vkWaitForFences);
    OP_GET_FUNC(vkDestroyFence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    // std::cout << "fence done" << std::endl;
#endif
}

template<typename T>
std::pair<float, float> OpVerifyWork(struct ComputeShader &shader, const int num_element, int loop_count)
{
    float diffmax = 0.0f;
    float precision = 0.0f;
    VkDevice device = shader.device->device;
    VkDeviceSize size = num_element * sizeof(T);

    void *cptr = nullptr;
    OP_GET_FUNC(vkMapMemory);
    OP_GET_FUNC(vkUnmapMemory);

    vkMapMemory(device, shader.buffers[2].memory, 0, size, 0, &cptr);

    T *rData = static_cast<T *>(cptr);
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>
                  || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t>) {
        for (auto i  = 0; i < num_element; i++) {
            if ((uint64_t)(rData[i]) != (uint64_t)(loop_count * 8 + 1)%((uint64_t)std::numeric_limits<T>::max()+1)) {
                std::cout << "Verification failed at index " << i << std::endl;
                std::cout << "Expected: " << (loop_count * 8 + 1)%(uint64_t(std::numeric_limits<T>::max())+1) << "\t";
                std::cout << "Got: " << uint64_t(rData[i]) << "  " <<uint64_t(std::numeric_limits<T>::max())+1<<std::endl;
                break;

            }
        }
    } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, _Float64>
#ifdef HAVE_FLOAT16
                  || std::is_same_v<T, _Float16>
#endif
                        ) {
        for (auto i  = 0; i < num_element; i++) {
            float diff = std::fabs(rData[i] - float((i % 5) + 1) * 1.f * (1.f / (1.f - float((i % 9) + 1) * (0.1f))));
            if (diffmax < diff) {
                diffmax = diff;
                precision = diff * 100.0 /float((i % 5) + 1) * 1.f * (1.f / (1.f - float((i % 9) + 1) * (0.1f)));
            }
            // relax the tolerance for float16
            if (diff > 0.2f) {
                std::cout << "Verification failed at index " << i << std::endl;
                std::cout << "Expected: " << float((i % 5) + 1) * (1.f) * (1.f / (1.f - float((i % 9) + 1) * 0.1f)) << "\t";
                std::cout << "Got: " << float(rData[i]) << std::endl;
                break;
            }
        }
    }

    vkUnmapMemory(device, shader.buffers[2].memory);
    return std::make_pair(diffmax, precision);
}

void OpDestroyShader(struct ComputeShader &shader)
{
    VkDevice device = shader.device->device;
    VkCommandPool commandPool = shader.commandPool;

    OP_GET_FUNC(vkDestroyCommandPool);
    OP_GET_FUNC(vkFreeMemory);
    OP_GET_FUNC(vkDestroyBuffer);
    OP_GET_FUNC(vkDestroyQueryPool);
    OP_GET_FUNC(vkDestroyDescriptorPool);
    OP_GET_FUNC(vkDestroyPipeline);
    OP_GET_FUNC(vkDestroyPipelineLayout);
    OP_GET_FUNC(vkDestroyDescriptorSetLayout);

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
}

void OpCreateQueryPool(struct ComputeShader &shader)
{
    VkQueryPool queryPool;
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = 2;

    OP_GET_FUNC(vkCreateQueryPool);
    vkCreateQueryPool(shader.device->device, &queryPoolCreateInfo, nullptr, &queryPool);
    shader.queryPool = queryPool;
}

double OpGetTimestamp(struct ComputeShader &shader)
{
    VkQueryPool queryPool = shader.queryPool;
    VkQueryResultFlags flags = VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
    uint64_t timestamps[2];

    OP_GET_FUNC(vkGetQueryPoolResults);
    vkGetQueryPoolResults(shader.device->device, queryPool, 0, 2, 2 * sizeof(uint64_t),
                         timestamps, sizeof(uint64_t), flags);
    return (timestamps[1] - timestamps[0]) * shader.device->timestampPeriod * 1e-9;
}

#if VK_EXT_debug_utils
VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    (void)messageTypes;
    (void)pUserData;
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
VkDebugUtilsMessengerEXT OpCreateDebugUtilsCallback(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT pfnCallback)
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
            return nullptr;
        }
    }
    return callback;
}
#else
VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT objectType,
    uint64_t object,
    size_t location,
    int32_t messageCode,
    const char *pLayerPrefix,
    const char *pMessage,
    void *pUserData)
{
    (void)flags;
    (void)objectType;
    (void)object;
    (void)location;
    (void)messageCode;
    (void)pLayerPrefix;
    (void)pUserData;
    fprintf(stderr, "%s\n", pMessage);
    return VK_FALSE;
}


VkDebugReportCallbackEXT OpCreateDebugReportCallback(VkInstance instance, PFN_vkDebugReportCallbackEXT pfnCallback)
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
            // std::cout << "Failed to create debug report callback" << std::endl;
            return nullptr;
        }
    }

    return callback;
}
#endif

std::string findValidationLayerSupport() {
    uint32_t layerCount;
    OP_GET_FUNC(vkEnumerateInstanceLayerProperties);
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for (auto layer : availableLayers) {
        // std::cout << "layer name " << layer.layerName << ": "<< layer.description << std::endl;
        // possible validation layers:
        // VK_LAYER_KHRONOS_validation
        // VK_LAYER_LUNARG_standard_validation
        if (std::string(layer.layerName).find("validation") != std::string::npos) {
            return std::string(layer.layerName);
        }
    }

    return {};
}

void checkVulkanVersion()
{
    uint32_t version;
    OP_GET_FUNC(vkEnumerateInstanceVersion);
    vkEnumerateInstanceVersion(&version);
    // std::cout << "version " << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version) << std::endl;
}


void OpBenchmarkResult(std::string name, double duration, uint64_t num_element, uint64_t loop_count, std::pair<float, float> result)
{
    std::cout << "Testcase: " << std::left << std::setw(20) << name << "\t";
    std::cout << "Duration: " << duration << "s" << "\t";
    const double numOps = 2.f * 8.0f * double(num_element) * double(loop_count);
    double ops = numOps / duration;
    // std::cout << "NumOps: " << ops << "\t";
    std::cout << "Throughput: ";
    std::string deli = "";
    if (name.find("fp")!= std::string::npos) {
        deli = "FL";
    }
    if (ops > 1.0f * 1e12) {
        std::cout << ops / 1e12 << " T";
    } else if (ops > 1.0f * 1e9) {
        std::cout << ops / 1e9 << " G";
    } else if (ops > 1.0f * 1e6) {
        std::cout << ops / 1e6 << " M";
    } else if (ops > 1.0f * 1e3) {
        std::cout << ops / 1e3 << " K";
    } else {
        std::cout << ops;
    }
    std::cout << deli << "OPS";
    if (result.first != 0.0f) {
        std::cout << "\tAccuracy: " << result.first << "(" << result.second <<"%)";
    }
    std::cout << std::endl;
}


struct testcase {
    std::string name;
    unsigned int code_size;
    unsigned char *code;
    bool enable;
};

template<typename T>
void OpRunShader(std::shared_ptr<ComputeDevice> dev,
                 std::vector<VkDescriptorSetLayoutBinding> layoutBindings, struct testcase t)
{
    const int num_element = 1024 * 1024;
    const uint32_t loop_count = 10000;
    struct ComputeShader shader = {};
    shader.device = dev;
    VkPipeline pipeline = OpCreatePipeline(shader, layoutBindings, loop_count, t.code_size, t.code);
    shader.pipeline = pipeline;

    if (OpCreateBuffers(shader, layoutBindings, num_element, sizeof(T))) {
        std::cout << "Failed to create buffers" << std::endl;
        return ;
    }
    OpCreateQueryPool(shader);

    OP_GET_FUNC(vkResetCommandBuffer);
    double duration = MAXFLOAT;
    for (int sloop = 0; sloop < 8; sloop++) {
        OpDispatchCommand(shader, num_element);
        OpSubmitWork<T>(shader, num_element);
        duration = std::fmin(OpGetTimestamp(shader), duration);
        vkResetCommandBuffer(shader.commandBuffer, 0);
    }
    std::pair<float, float> r = OpVerifyWork<T>(shader, num_element, loop_count);
    OpBenchmarkResult(t.name, duration, num_element, loop_count, r);
    OpDestroyShader(shader);
}

int main() {

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

#if VK_EXT_debug_utils
    VkDebugUtilsMessengerEXT callback = OpCreateDebugUtilsCallback(instance, &debugUtilsCallback);
#else
    VkDebugReportCallbackEXT callback = OpCreateDebugReportCallback(instance, &debugReportCallback);
#endif

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings = OpDescriptorSetLayoutBinding();

    {
        std::shared_ptr<ComputeDevice> dev = OpCreateDevice(instance, enabledLayerNames);

        struct testcase testcases[] = {
            {"int64", shaderint64_size, shaderint64_code, dev->int64},
            {"fp64", shaderfp64_size, shaderfp64_code, dev->fp64},
            {"int32", shaderint32_size, shaderint32_code,true},
            {"fp32", shaderfp32_size, shaderfp32_code, true},
            {"int16", shaderint16_size, shaderint16_code, dev->int16},
#ifdef HAVE_FLOAT16
            {"fp16", shaderfp16_size, shaderfp16_code, dev->fp16},
#endif
            {"int8", shaderint8_size, shaderint8_code, dev->int8},
#ifdef VK_KHR_shader_integer_dot_product
            {"int8dot", shaderint8dot_size, shaderint8dot_code,
                (dev->int8 && dev->dot && dev->int8dot)},
            {"int8dotaccsat", shaderint8dotaccsat_size, shaderint8dotaccsat_code,
                (dev->int8 && dev->dot && dev->int8dotaccsat)},
            {"int8dot4x8packed", shaderint8dot4x8packed_size, shaderint8dot4x8packed_code,
                (dev->int8 && dev->dot && dev->int8dot4x8packed)},
#endif
        };
        for (size_t i = 0; i < sizeof(testcases) / sizeof(testcases[0]); i++) {
            if (!testcases[i].enable) {
                std::cout << "Testcase: " << testcases[i].name << "\tNot Supported" << std::endl;
                continue;
            }

            if (testcases[i].name.compare("fp32")==0) {
                OpRunShader<float>(dev, layoutBindings, testcases[i]);
#ifdef HAVE_FLOAT16
            } else if (testcases[i].name.compare("fp16")==0) {
                OpRunShader<_Float16>(dev, layoutBindings, testcases[i]);
#endif
            } else if (testcases[i].name.compare("int32")==0) {
                OpRunShader<int>(dev, layoutBindings, testcases[i]);
            } else if (testcases[i].name.compare("int64")==0) {
                OpRunShader<int64_t>(dev, layoutBindings, testcases[i]);
            } else if (testcases[i].name.compare("fp64")==0) {
                OpRunShader<_Float64>(dev, layoutBindings, testcases[i]);
            } else if (testcases[i].name.compare("int16")==0) {
                OpRunShader<uint16_t>(dev, layoutBindings, testcases[i]);
            } else if (!testcases[i].name.compare("int8")
                       ||!testcases[i].name.compare("int8dot")
                       ||!testcases[i].name.compare("int8dotaccsat")
                       ||!testcases[i].name.compare("int8dot4x8packed")) {
                OpRunShader<uint8_t>(dev, layoutBindings, testcases[i]);
            }
            
        }
    }

    OP_GET_FUNC(vkGetInstanceProcAddr);
    OP_GET_FUNC(vkDestroyInstance);

    if (callback) {
#if VK_EXT_debug_utils
        auto vkDestroyDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

        if (vkDestroyDebugUtilsMessengerEXT)
            vkDestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
#else
        auto vkDestroyDebugReportCallbackEXT =
            reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
        if (vkDestroyDebugReportCallbackEXT)
            vkDestroyDebugReportCallbackEXT(instance, callback, nullptr);
#endif
    }
    vkDestroyInstance(instance, nullptr);

    return 0;
}