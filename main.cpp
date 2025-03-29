#include <__config>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <vector>
#include <iomanip>

#include <dlfcn.h>

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>

#include "benchmark.h"

#define VK_FUNCTION_LIST \
    PFN(vkEnumerateInstanceVersion) \
    PFN(vkEnumerateInstanceLayerProperties) \
    PFN(vkCreateInstance) \
    PFN(vkEnumerateInstanceExtensionProperties) \
    PFN(vkGetInstanceProcAddr) \
    PFN(vkMapMemory) \
    PFN(vkUnmapMemory) \
    PFN(vkGetBufferMemoryRequirements) \
    PFN(vkGetPhysicalDeviceMemoryProperties) \
    PFN(vkAllocateMemory) \
    PFN(vkAllocateCommandBuffers) \
    PFN(vkBindBufferMemory) \
    PFN(vkCmdBindPipeline) \
    PFN(vkCmdDispatch) \
    PFN(vkCmdWriteTimestamp) \
    PFN(vkCmdBindDescriptorSets) \
    PFN(vkCmdResetQueryPool) \
    PFN(vkBeginCommandBuffer) \
    PFN(vkEndCommandBuffer) \
    PFN(vkQueueSubmit) \
    PFN(vkQueueWaitIdle) \
    PFN(vkCreateBuffer) \
    PFN(vkCreateQueryPool) \
    PFN(vkCreateDescriptorPool) \
    PFN(vkAllocateDescriptorSets) \
    PFN(vkUpdateDescriptorSets) \
    PFN(vkCreateCommandPool) \
    PFN(vkCreateComputePipelines) \
    PFN(vkCreateDevice) \
    PFN(vkGetDeviceQueue) \
    PFN(vkCreateDescriptorSetLayout) \
    PFN(vkCreatePipelineLayout) \
    PFN(vkDestroyBuffer) \
    PFN(vkDestroyQueryPool) \
    PFN(vkDestroyDescriptorPool) \
    PFN(vkDestroyPipeline) \
    PFN(vkDestroyPipelineLayout) \
    PFN(vkDestroyDescriptorSetLayout) \
    PFN(vkDestroyDevice) \
    PFN(vkDestroyInstance) \
    PFN(vkGetQueryPoolResults) \
    PFN(vkCreateShaderModule) \
    PFN(vkDestroyShaderModule) \
    PFN(vkDestroyCommandPool) \
    PFN(vkFreeMemory) \
    PFN(vkGetPhysicalDeviceQueueFamilyProperties) \
    PFN(vkGetPhysicalDeviceProperties2) \
    PFN(vkEnumeratePhysicalDevices) \
    PFN(vkEnumerateDeviceExtensionProperties) \
    PFN(vkResetCommandBuffer) \
    PFN(vkFreeCommandBuffers) \
    PFN(vkGetPhysicalDeviceFeatures) \
    PFN(vkGetPhysicalDeviceFeatures2)

class VulkanLib {
private:
    void *lib;
    std::unique_ptr<std::map<std::string, void *> > symbols;
public:
    VulkanLib() {
        symbols = std::make_unique<std::map<std::string, void *> >();
#ifdef __APPLE__
        // const char *const name = "libMoltenVK.dylib";
        lib = dlopen("libvulkan.dylib", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
            lib = dlopen("libvulkan.1.dylib", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
		    lib = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!lib && getenv("DYLD_FALLBACK_LIBRARY_PATH") == nullptr)
            lib = dlopen("/usr/local/lib/libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
#elif defined __linux__
        lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib)
            lib = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
#endif
        if (!lib) {
            std::cerr << "Failed to load vulkan library ," << dlerror() << std::endl;
            return ;
        }
#define PFN(name) name = reinterpret_cast<PFN_##name>(dlsym(lib, #name));
            VK_FUNCTION_LIST
#undef PFN
    }
    ~VulkanLib() {
        dlclose(lib);
    }
    void *getSymbol(const char *name) {
        return symbols->at(name);
    }
#define PFN(name) PFN_##name name;
    VK_FUNCTION_LIST
#undef PFN
};

VulkanLib vklib;
#define OP(name) vklib.name

enum shader_feature {
    FEATURE_INT64 = 1 << 0,
    FEATURE_FP64 = 1 << 1,
    FEATURE_INT32 = 1 << 2,
    FEATURE_FP32 = 1 << 3,
    FEATURE_INT16 = 1 << 4,
    FEATURE_FP16 = 1 << 5,
    FEATURE_INT8 = 1 << 6,
    FEATURE_DOT = 1 << 7,
    FEATURE_INT8DOT = 1 << 8,
    FEATURE_INT8DOTACCSAT = 1 << 9,
    FEATURE_INT8DOT4X8PACKED = 1 << 10,
};

class ComputeDevice {
private:
    std::vector<VkExtensionProperties> ext_properties;
    VkPhysicalDeviceProperties deviceProperties;
    void checkDeviceDataTypeFeatures(void)
    {
        VkPhysicalDeviceFeatures deviceFeatures = {};
        OP(vkGetPhysicalDeviceFeatures)(physicalDevice, &deviceFeatures);
        this->features |= (deviceFeatures.shaderInt64 ? FEATURE_INT64 : 0);
        this->features |= (deviceFeatures.shaderFloat64 ? FEATURE_FP64 : 0);
        this->features |= FEATURE_FP32 | FEATURE_INT32;
        this->features |= (deviceFeatures.shaderInt16 ? FEATURE_INT16 : 0);

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

        OP(vkGetPhysicalDeviceFeatures2)(physicalDevice, &features2);
        this->features |= (float16Int8Features.shaderFloat16 ? FEATURE_FP16 : 0);
        this->features |= (float16Int8Features.shaderInt8 ? FEATURE_INT8 : 0);
#if VK_KHR_shader_integer_dot_product
        this->features |= (integerDotProductFeatures.shaderIntegerDotProduct ? FEATURE_DOT : 0);
#endif
    }

    void checkDeviceExtension(void)
    {
        uint32_t extensionCount = 0;
        OP(vkEnumerateDeviceExtensionProperties)(physicalDevice, NULL, &extensionCount, NULL);
        this->ext_properties.resize(extensionCount);
        OP(vkEnumerateDeviceExtensionProperties)(physicalDevice, NULL, &extensionCount, this->ext_properties.data());
        std::cout << "Device Extensions:" << std::endl;
        for (uint32_t i = 0; i < extensionCount; i++) {
            std::cout << ext_properties[i].extensionName << ":" << ext_properties[i].specVersion << std::endl;
        }
    }
    bool checkDeviceExtensionFeature(const char *name)
    {
        for (auto ext : this->ext_properties) {
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

        OP(vkGetPhysicalDeviceProperties2)(physicalDevice, &properties2);
        this->features |= (integerDotProductProperties.integerDotProduct8BitUnsignedAccelerated ? FEATURE_INT8DOT : 0);
        this->features |= (integerDotProductProperties.integerDotProduct4x8BitPackedUnsignedAccelerated ? FEATURE_INT8DOT4X8PACKED : 0);
        this->features |= (integerDotProductProperties.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated ? FEATURE_INT8DOTACCSAT : 0);
    }
#endif
    void getDeviceTimeLimits(void)
    {
        VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
        subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroup_properties.pNext = nullptr;
        VkPhysicalDeviceProperties2 properties2 = {};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &subgroup_properties;

        OP(vkGetPhysicalDeviceProperties2)(physicalDevice, &properties2);
        deviceProperties = properties2.properties;
        this->timestampPeriod = deviceProperties.limits.timestampPeriod;
        std::cout << "GPU " << deviceProperties.deviceName << std::endl;
    }

    VkResult createDevice(void)
    {
        std::vector<uintptr_t> enabledFeatures;
        std::vector<const char *> enabledExtensions;
        VkPhysicalDeviceFeatures features = {};
        features.robustBufferAccess = VK_TRUE;
        if (this->features & FEATURE_INT64)
            features.shaderInt64 = VK_TRUE;
        if (this->features & FEATURE_FP64)
            features.shaderFloat64 = VK_TRUE;
        if (this->features & FEATURE_INT16)
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
        if (this->features & FEATURE_INT8) {
            float16Int8Features.shaderInt8 = VK_TRUE;
            if (checkDeviceExtensionFeature(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
                enabledExtensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
                enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&storage8bitFeatures));
            }
        }
        if (this->features & FEATURE_FP16) {
            float16Int8Features.shaderFloat16 = VK_TRUE;
            if (checkDeviceExtensionFeature(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
                enabledExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
                if (deviceProperties.vendorID != 4318) {
                    // tested on Nvidia A2000, it supports 16bit storage feature but did not need to enable it
                    // enable it will cause validation error VK_ERROR_FEATURE_NOT_PRESENT
                    enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&storage16bitFeatures));
                }
            }
#if VK_AMD_gpu_shader_half_float
            if (deviceProperties.vendorID == 4098) {
                // for AMD card, do we really need this ? over VK_KHR_shader_float16_int8
                if (checkDeviceExtensionFeature(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME)) {
                    enabledExtensions.push_back(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
                }
            }
#endif
        }
        if (this->features & (FEATURE_INT8 |FEATURE_FP16)) {
            if (checkDeviceExtensionFeature(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
                enabledFeatures.push_back(reinterpret_cast<uintptr_t>(&float16Int8Features));
                enabledExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
            }
        }
        if (this->features & FEATURE_DOT) {
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
        deviceCreateInfo.enabledLayerCount = 0;
        deviceCreateInfo.ppEnabledLayerNames = nullptr;
        deviceCreateInfo.pEnabledFeatures = &features;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
        deviceCreateInfo.pNext = pFirst;

        return OP(vkCreateDevice)(this->physicalDevice, &deviceCreateInfo, nullptr, &this->device);
    }

    void getDeviceQueue(void)
    {
        OP(vkGetDeviceQueue)(device, queueFamilyIndex, 0, &this->queue);
    }

public:
    ComputeDevice(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex):
        physicalDevice(physicalDevice), queueFamilyIndex(queueFamilyIndex) {
        checkDeviceDataTypeFeatures();
        checkDeviceExtension();
        getDeviceTimeLimits();
        VkResult err = createDevice();
        if (err != VK_SUCCESS) {
            std::map<int, std::string> errstrings;
            errstrings[VK_ERROR_OUT_OF_HOST_MEMORY] = "VK_ERROR_OUT_OF_HOST_MEMORY";
            errstrings[VK_ERROR_OUT_OF_DEVICE_MEMORY] = "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            errstrings[VK_ERROR_INITIALIZATION_FAILED] = "VK_ERROR_INITIALIZATION_FAILED";
            errstrings[VK_ERROR_DEVICE_LOST] = "VK_ERROR_DEVICE_LOST";
            errstrings[VK_ERROR_EXTENSION_NOT_PRESENT] = "VK_ERROR_EXTENSION_NOT_PRESENT";
            errstrings[VK_ERROR_FEATURE_NOT_PRESENT] = "VK_ERROR_FEATURE_NOT_PRESENT";
            errstrings[VK_ERROR_TOO_MANY_OBJECTS] = "VK_ERROR_TOO_MANY_OBJECTS";
            throw std::runtime_error("Failed to create device " + errstrings[err]);
        }
        getDeviceQueue();
#if VK_KHR_shader_integer_dot_product
        if (this->features & FEATURE_DOT)
            check_shader_integer_dot_product_support();
#endif
    };
    ~ComputeDevice() {
        OP(vkDestroyDevice)(device, nullptr);
    };

    VkDevice device;
    VkPhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    VkQueue queue;
    float timestampPeriod;

    uint32_t features;
};

class ComputeBuffer {
private:
    int32_t findMemoryTypeFromProperties(uint32_t memoryTypeBits,
            VkPhysicalDeviceMemoryProperties properties,
            VkMemoryPropertyFlags requiredProperties)
    {
        for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
            if (((memoryTypeBits & (1 << index))) &&
                ((properties.memoryTypes[index].propertyFlags & requiredProperties) ==
                requiredProperties)) {
                return (int32_t)index;
            }
        }
        return -1;
    }
    VkResult __OpCreateBuffer(int bufferflags, int memoryflags, int num_element, size_t element_size)
    {
        uint32_t queueFamilyIndex = computedevice->queueFamilyIndex;
        VkDevice device = computedevice->device;
        VkDeviceMemory memory;
        VkBuffer buffer;
        VkResult error;

        // create the buffers which will hold the data to be consumed by shader
        VkBufferCreateInfo bufferCreateInfo = {};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = element_size * num_element;
        bufferCreateInfo.usage = bufferflags;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferCreateInfo.queueFamilyIndexCount = 1;
        bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

        error = OP(vkCreateBuffer)(device, &bufferCreateInfo, nullptr, &buffer);
        if (error) {
            std::cout << "failed to create buffer!" << std::endl;
            return error;
        }
        this->buffer = buffer;

        VkMemoryRequirements memoryRequirements;
        OP(vkGetBufferMemoryRequirements)(device, buffer, &memoryRequirements);

        VkPhysicalDeviceMemoryProperties memoryProperties;
        OP(vkGetPhysicalDeviceMemoryProperties)(computedevice->physicalDevice, &memoryProperties);

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

        error = OP(vkAllocateMemory)(device, &allocateInfo, nullptr, &memory);
        if (error) {
            std::cout << "failed to allocate memory!" << std::endl;
            return error;
        }

        error = OP(vkBindBufferMemory)(device, buffer, memory, 0);
        if (error) {
            return error;
        }
        this->memory = memory;

        return VK_SUCCESS;
    }
    
    std::shared_ptr<ComputeDevice> computedevice;
    VkBuffer buffer;
    VkDeviceMemory memory;
public:
    ComputeBuffer(std::shared_ptr<ComputeDevice> computedevice, int bufferflags, int memoryflags, int num_element, size_t element_size):
        computedevice(computedevice) {
        VkResult error = __OpCreateBuffer(bufferflags, memoryflags, num_element, element_size);
        if (error) {
            throw std::runtime_error("failed to create buffer1");
        }
    };
    ~ComputeBuffer() {
        if (buffer)
            OP(vkDestroyBuffer)(computedevice->device, buffer, nullptr);
        if (memory)
            OP(vkFreeMemory)(computedevice->device, memory, nullptr);
    };
    VkDeviceMemory getMemory() {
        return memory;
    };
    VkBuffer getBuffer() {
        return buffer;
    };
    void *getMemoryPtr(size_t size) {
        void *ptr;
        OP(vkMapMemory)(computedevice->device, memory, 0, size, 0, &ptr);
        return ptr;
    };
    void unmapMemory() {
        OP(vkUnmapMemory)(computedevice->device, memory);
    }
};

template<typename T>
class ComputeShader {
private:
    VkPipelineLayout OpCreatePipelineLayout(std::vector<VkDescriptorSetLayoutBinding> &layoutBindings)
    {
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkDevice device = computedevice->device;
        VkResult error;

        VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo = {};
        setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayoutCreateInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        setLayoutCreateInfo.pBindings = layoutBindings.data();

        error = OP(vkCreateDescriptorSetLayout)(device, &setLayoutCreateInfo, nullptr,
                                            &setLayout);
        if (error != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &setLayout;

        error = OP(vkCreatePipelineLayout)(device, &pipelineLayoutCreateInfo, nullptr,
                                        &pipelineLayout);
        if (error != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        this->descriptorSetLayout = setLayout;
        return pipelineLayout;
    }

    VkResult OpCreatePipeline(std::vector<VkDescriptorSetLayoutBinding> &layoutBindings,
                                uint32_t loop_count, const unsigned int code_size, const unsigned char *code)
    {
        VkDevice device = this->computedevice->device;
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkResult error;
        VkPipeline pipeline = VK_NULL_HANDLE;

        VkPipelineLayout layout = OpCreatePipelineLayout(layoutBindings);
        if (!layout) {
            std::cout << "failed to create pipeline layout!" << std::endl;
            return VK_ERROR_UNKNOWN;
        }
        this->layout = layout;

        VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(code);
        shaderModuleCreateInfo.codeSize = code_size;

        error = OP(vkCreateShaderModule)(device, &shaderModuleCreateInfo, nullptr,
                                    &shaderModule);
        if (error != VK_SUCCESS) {
            return error;
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
        pipelineCreateInfo.layout = layout;

        error = OP(vkCreateComputePipelines)(device, VK_NULL_HANDLE, 1,
                                        &pipelineCreateInfo, nullptr, &pipeline);
        if (error) {
            return error;
        }

        OP(vkDestroyShaderModule)(device, shaderModule, nullptr);
        this->pipeline = pipeline;
        return VK_SUCCESS;
    }

    VkResult OpCreateDescriptorPool(std::vector<VkDescriptorSetLayoutBinding> layoutBindings)
    {
        VkDevice device = this->computedevice->device;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = 1;
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(layoutBindings.size());
        descriptorPoolCreateInfo.pPoolSizes = &poolSize;
        OP(vkCreateDescriptorPool)(device, &descriptorPoolCreateInfo, nullptr,
                                &descriptorPool);
        this->descriptorPool = descriptorPool;
        return VK_SUCCESS;
    }

    VkResult OpAllocateDescriptorSets()
    {
        VkDevice device = this->computedevice->device;
        VkResult error;

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = this->descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &this->descriptorSetLayout;

        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        error = OP(vkAllocateDescriptorSets)(device, &descriptorSetAllocateInfo,
                                        &descriptorSet);
        if (error) {
            return error;
        }
        this->descriptorSet = descriptorSet;

        return VK_SUCCESS;
    }

    VkResult OpWriteDescriptorSets(void)
    {
        std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
        std::vector<VkWriteDescriptorSet> writeDescriptorSets(buffers.size());
        for (size_t i = 0; i < buffers.size(); i++) {
            VkDescriptorBufferInfo &bufferInfo = bufferInfos[i];
            VkWriteDescriptorSet &writeDescriptorSet = writeDescriptorSets[i];
            bufferInfo.buffer = buffers[i]->getBuffer();
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;

            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.dstSet = this->descriptorSet;
            writeDescriptorSet.dstBinding = static_cast<uint32_t>(i);
            writeDescriptorSet.dstArrayElement = 0;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        }

        OP(vkUpdateDescriptorSets)(computedevice->device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        return VK_SUCCESS;
    }

    VkResult OpCreateBuffers(std::vector<VkDescriptorSetLayoutBinding> layoutBindings, int num_element, size_t element_size)
    {
        VkDevice device = computedevice->device;
        VkResult error;

        ComputeBuffer *buffer1 = new ComputeBuffer(computedevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            num_element, element_size);
        this->buffers.push_back(buffer1);

        ComputeBuffer *buffer2 = new ComputeBuffer(computedevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            num_element, element_size);
        this->buffers.push_back(buffer2);

        ComputeBuffer *buffer3 = new ComputeBuffer(computedevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            num_element, element_size);
        this->buffers.push_back(buffer3);

        OpCreateDescriptorPool(layoutBindings);

        OpAllocateDescriptorSets();
        OpWriteDescriptorSets();

        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT|VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = computedevice->queueFamilyIndex;
        VkCommandPool commandPool = VK_NULL_HANDLE;

        error = OP(vkCreateCommandPool)(device, &commandPoolCreateInfo, nullptr,
                                    &commandPool);
        if (error) {
            std::cout << "failed to create command pool!" << std::endl;
            return error;
        }
        this->commandPool = commandPool;

        return VK_SUCCESS;
    }

    void OpDestroyShader()
    {
        VkDevice device = this->computedevice->device;

        OP(vkFreeCommandBuffers)(device, this->commandPool, 1, &this->commandBuffer);
        OP(vkDestroyCommandPool)(device, this->commandPool, nullptr);
        
        for (auto b : this->buffers) {
            delete b;
        }

        OP(vkDestroyQueryPool)(device, this->queryPool, nullptr);
        OP(vkDestroyDescriptorPool)(device, this->descriptorPool, nullptr);
        OP(vkDestroyPipeline)(device, this->pipeline, nullptr);
        OP(vkDestroyPipelineLayout)(device, this->layout, nullptr);
        OP(vkDestroyDescriptorSetLayout)(device, this->descriptorSetLayout, nullptr);
    }

    void OpCreateQueryPool()
    {
        VkQueryPool queryPool;
        VkQueryPoolCreateInfo queryPoolCreateInfo = {};
        queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolCreateInfo.queryCount = 2;

        OP(vkCreateQueryPool)(computedevice->device, &queryPoolCreateInfo, nullptr, &queryPool);
        this->queryPool = queryPool;
    }
    std::vector<VkDescriptorSetLayoutBinding> OpDescriptorSetLayoutBinding(void)
    {
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

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


public:
    double OpGetTimestamp(void)
    {
        VkQueryResultFlags flags = VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT;
        uint64_t timestamps[2];

        OP(vkGetQueryPoolResults)(computedevice->device, this->queryPool, 0, 2, 2 * sizeof(uint64_t),
                            timestamps, sizeof(uint64_t), flags);
        return (timestamps[1] - timestamps[0]) * computedevice->timestampPeriod * 1e-9;
    }

    VkResult OpDispatchCommand(const int num_element)
    {
        VkResult error;
        VkDevice device = computedevice->device;

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        error = OP(vkAllocateCommandBuffers)(device, &commandBufferAllocateInfo,
                                        &commandBuffer);
        if (error) {
            return error;
        }
        this->commandBuffer = commandBuffer;

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        OP(vkBeginCommandBuffer)(commandBuffer, &beginInfo);
        OP(vkCmdBindPipeline)(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        OP(vkCmdBindDescriptorSets)(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                this->layout, 0, 1, &descriptorSet, 0, nullptr);    

        OP(vkCmdResetQueryPool)(commandBuffer, queryPool, 0, 2);
        OP(vkCmdWriteTimestamp)(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
        OP(vkCmdDispatch)(commandBuffer, num_element/(4*16), 1, 1);
        OP(vkCmdWriteTimestamp)(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);

        error = OP(vkEndCommandBuffer)(commandBuffer);
        if (error) {
            return error;
        }
        return VK_SUCCESS;
    }

    explicit ComputeShader(std::shared_ptr<ComputeDevice> dev, const int code_size, const unsigned char *code,
                           int loop_count, int num_element): computedevice(dev) {
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings = OpDescriptorSetLayoutBinding();
        OpCreatePipeline(layoutBindings, loop_count, code_size, code);
        if (OpCreateBuffers(layoutBindings, num_element, sizeof(T))) {
            throw std::runtime_error("Failed to create buffers");
        }
        OpCreateQueryPool();
    };
    ~ComputeShader() {
        OpDestroyShader();
    };

    void OpSubmitWork(const int num_element)
    {
        VkDeviceSize size = num_element * sizeof(T);

        void *aptr = nullptr, *bptr = nullptr;

        aptr = buffers[0]->getMemoryPtr(size);
        if (!aptr) {
            std::cout << "failed to map memory!" << std::endl;
            return;
        }
        bptr = buffers[1]->getMemoryPtr(size);
        if (!bptr) {
            buffers[0]->unmapMemory();
            std::cout << "failed to map memory!" << std::endl;
            return;
        }

        T *aData = static_cast<T *>(aptr);
        T *bData = static_cast<T *>(bptr);
        if constexpr (std::is_same_v<T, float>
    #ifdef HAVE_FLOAT64
                    || std::is_same_v<T, _Float64>
    #endif
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

        buffers[0]->unmapMemory();
        buffers[1]->unmapMemory();

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &this->commandBuffer;

        OP(vkQueueSubmit)(computedevice->queue, 1, &submitInfo, nullptr);
        OP(vkQueueWaitIdle)(computedevice->queue);
    }

    std::pair<float, float> OpVerifyWork(const int num_element, int loop_count)
    {
        float diffmax = 0.0f;
        float precision = 0.0f;
        VkDeviceSize size = num_element * sizeof(T);

        void *cptr = nullptr;
        cptr = buffers[2]->getMemoryPtr(size);

        T *rData = static_cast<T *>(cptr);
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>
                    || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t>) {
            for (auto i  = 0; i < num_element; i++) {
                if ((uint64_t)(rData[i]) != (uint64_t)(loop_count * 8 + 1)%((uint64_t)std::numeric_limits<T>::max()+1)) {
                    std::cout << "Verification failed at index " << i << std::endl;
                    std::cout << "Expected: " << (loop_count * 8 + 1)%(uint64_t(std::numeric_limits<T>::max())+1) << "\t";
                    std::cout << "Got: " << uint64_t(rData[i]) << std::endl;
                    break;
                }
            }
        } else if constexpr (std::is_same_v<T, float> 
    #ifdef HAVE_FLOAT64
                    || std::is_same_v<T, _Float64>
    #endif
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

        buffers[2]->unmapMemory();
        return {diffmax, precision};
    }

private:
    std::shared_ptr<ComputeDevice> computedevice;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    std::vector<ComputeBuffer *> buffers;

    VkQueryPool queryPool;
};


class VulkanInstance {
private:
    VkInstance instance;

    std::string findValidationLayerSupport() {
        uint32_t layerCount;
        OP(vkEnumerateInstanceLayerProperties)(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        OP(vkEnumerateInstanceLayerProperties)(&layerCount, availableLayers.data());

        // first try VK_LAYER_KHRONOS_validation, it depreated VK_LAYER_LUNARG_standard_validation
        for (auto layer : availableLayers) {
            if (std::string(layer.layerName).find("VK_LAYER_KHRONOS_validation") != std::string::npos) {
                std::cout << "validation layer found " << layer.layerName << std::endl;
                return std::string(layer.layerName);
            }
        }
        for (auto layer : availableLayers) {
            if (std::string(layer.layerName).find("VK_LAYER_LUNARG_standard_validation") != std::string::npos) {
                std::cout << "validation layer found " << layer.layerName << std::endl;
                return std::string(layer.layerName);
            }
        }

        return {};
    }
    uint32_t getVulkanVersion(void)
    {
        uint32_t version = VK_API_VERSION_1_0;
        OP(vkEnumerateInstanceVersion)(&version);
        std::cout << "vulkan version " << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version) << std::endl;
        return version;
    }

    void checkInstanceExtension()
    {
        uint32_t pPropertyCount;
        OP(vkEnumerateInstanceExtensionProperties)(nullptr, &pPropertyCount, nullptr);
        ext_properties.resize(pPropertyCount);
        OP(vkEnumerateInstanceExtensionProperties)(nullptr, &pPropertyCount, ext_properties.data());
        for (auto ext : this->ext_properties) {
            std::cout << "instance extension " << ext.extensionName << std::endl;
        }
    }
    bool checkInstanceExtensionFeature(const char *name)
    {
        for (auto ext : this->ext_properties) {
            if (std::string(ext.extensionName).compare(name) == 0) {
                return true;
            }
        }
        return false;
    }

    VkInstance OpCreateInstance(std::vector<const char *> &enabledLayerNames) {
        uint32_t version = getVulkanVersion();
        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Vulkan Compute Shader Benchmark";
        // vkGetPhysicalDeviceProperties2 requires 1.1.0
        // SPV_KHR_vulkan_memory_model, use_vulkan_memory_model in spirv requires 1.2.0
        applicationInfo.apiVersion = version; // use the libvulkan version directly
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
        if (checkInstanceExtensionFeature(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            enabledExtensionNames.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
#endif
#if VK_EXT_debug_report
        if (enabledExtensionNames.empty() && checkInstanceExtensionFeature(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
            enabledExtensionNames.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
#endif
#if VK_KHR_get_physical_device_properties2
        if (checkInstanceExtensionFeature(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            enabledExtensionNames.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
#endif
#if VK_KHR_portability_enumeration
        if (checkInstanceExtensionFeature(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            enabledExtensionNames.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceCreateInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif
        if (enabledExtensionNames.size() > 0) {
            instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNames.size());
            instanceCreateInfo.ppEnabledExtensionNames = enabledExtensionNames.data();
        }
        VkInstance instance = VK_NULL_HANDLE;
        VkResult error = OP(vkCreateInstance)(&instanceCreateInfo, nullptr, &instance);
        if (error != VK_SUCCESS) {
            std::cout << "Fail to create instance " << error << std::endl;
            if (error == VK_ERROR_LAYER_NOT_PRESENT) {
                std::cout << "VK_ERROR_LAYER_NOT_PRESENT" << std::endl;
            } else if (error == VK_ERROR_INCOMPATIBLE_DRIVER) {
                std::cout << "VK_ERROR_INCOMPATIBLE_DRIVER" << std::endl;
            }
            return nullptr;
        }

        return instance;
    }

    
#if VK_EXT_debug_utils
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(
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
#endif
#if VK_EXT_debug_report
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
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
#endif

#if VK_EXT_debug_utils
    VkDebugUtilsMessengerEXT OpCreateDebugUtilsCallback()
    {
        PFN_vkDebugUtilsMessengerCallbackEXT pfnCallback = &debugUtilsCallback;
        VkResult error;

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
            OP(vkGetInstanceProcAddr)(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (vkCreateDebugUtilsMessengerEXT) {
            error = vkCreateDebugUtilsMessengerEXT(instance, &callbackCreateInfo, nullptr,
                                            &callback);
            if (error != VK_SUCCESS) {
                std::cerr << "Failed to create debug callback" << std::endl;
                return nullptr;
            }
        }
        this->callback.utils = callback;
        return callback;
    }
#endif
#if VK_EXT_debug_report
    VkDebugReportCallbackEXT OpCreateDebugReportCallback()
    {
        PFN_vkDebugReportCallbackEXT pfnCallback = &debugReportCallback;
        VkResult error;
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
            OP(vkGetInstanceProcAddr)(instance, "vkCreateDebugReportCallbackEXT"));
        if (vkCreateDebugReportCallbackEXT) {
            error = vkCreateDebugReportCallbackEXT(instance, &callbackCreateInfo, nullptr,
                                            &callback);
            if (error != VK_SUCCESS) {
                // std::cout << "Failed to create debug report callback" << std::endl;
                return nullptr;
            }
        }
        this->callback.report = callback;

        return callback;
    }
#endif

    std::vector<const char *> enabledExtensionNames;
    std::vector<VkExtensionProperties> ext_properties;
    union {
#if VK_EXT_debug_utils
        VkDebugUtilsMessengerEXT utils;
#endif
#if VK_EXT_debug_report
        VkDebugReportCallbackEXT report;
#endif
    } callback;
public:
    VulkanInstance() {
        std::string str = findValidationLayerSupport();
        std::vector<const char *> enabledLayerNames;
        if (!str.empty()) {
            enabledLayerNames.push_back(str.c_str());
        }
        checkInstanceExtension();
        instance = OpCreateInstance(enabledLayerNames);
        if (!instance) {
            throw std::runtime_error("Failed to create Vulkan instance.");            
        }
#if VK_EXT_debug_utils
        OpCreateDebugUtilsCallback();
#endif
#if VK_EXT_debug_report
        if (callback.utils == nullptr) {
            OpCreateDebugReportCallback();
        }
#endif
    }

    ~VulkanInstance() {
#if VK_EXT_debug_utils
        if (callback.utils && !enabledExtensionNames.empty() && 
            std::string(enabledExtensionNames[0]).compare(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            auto vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    OP(vkGetInstanceProcAddr)(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (vkDestroyDebugUtilsMessengerEXT)
                vkDestroyDebugUtilsMessengerEXT(instance, callback.utils, nullptr);
        }
#endif
#if VK_EXT_debug_report
        if (callback.report && !enabledExtensionNames.empty() &&
            std::string(enabledExtensionNames[0]).compare(VK_EXT_DEBUG_REPORT_EXTENSION_NAME) == 0) {
            auto vkDestroyDebugReportCallbackEXT = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
                    OP(vkGetInstanceProcAddr)(instance, "vkDestroyDebugReportCallbackEXT"));
            if (vkDestroyDebugReportCallbackEXT)
                vkDestroyDebugReportCallbackEXT(instance, callback.report, nullptr);
        }
#endif
        OP(vkDestroyInstance)(instance, nullptr);

    }

    std::pair<VkPhysicalDevice, uint32_t> getDeviceAndQeueue(void) {
        uint32_t count;

        VkResult error = OP(vkEnumeratePhysicalDevices)(instance, &count, nullptr);
        if (error != VK_SUCCESS) {
            return { nullptr, 0};
        }
        std::vector<VkPhysicalDevice> physicalDevices(count);
        error = OP(vkEnumeratePhysicalDevices)(instance, &count, physicalDevices.data());
        if (error != VK_SUCCESS) {
            return {nullptr, 0};
        }
        std::cout << "Found " << count << " physical devices." << std::endl;

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
        for (auto device : physicalDevices) {
            OP(vkGetPhysicalDeviceQueueFamilyProperties)(device, &count, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilyProperties(count);
            OP(vkGetPhysicalDeviceQueueFamilyProperties)(device, &count,
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
        return {physicalDevice, queueFamilyIndex};
    }
};
#ifdef HAVE_FLOAT64
#define TESTCASE_FP64 _(FP64)
#else
#define TESTCASE_FP64
#endif
#ifdef HAVE_FLOAT16
#define TESTCASE_FP16 _(FP16)
#else
#define TESTCASE_FP16
#endif
#if VK_KHR_shader_integer_dot_product
#define TESTCASE_DOT \
    _(INT8DOT) \
    _(INT8DOTACCSAT) \
    _(INT8DOT4X8PACKED)
#else
    #define TESTCASE_DOT
#endif

#define TESTCASES \
    _(INT64) \
    TESTCASE_FP64 \
    _(INT32) \
    _(FP32) \
    _(INT16) \
    TESTCASE_FP16 \
    _(INT8) \
    TESTCASE_DOT

template<typename T>
class optestcase {
    std::string name;
    const unsigned int code_size;
    const unsigned char *code;

    void OpBenchmarkResult(double duration, uint64_t num_element, uint64_t loop_count, std::pair<float, float> result)
    {
        std::cout << "Testcase: " << std::left << std::setw(20) << name << "\t";
        std::cout << "Duration: " << duration << "s" << "\t";
        const double numOps = 2.f * 8.0f * double(num_element) * double(loop_count);
        double ops = numOps / duration;
        std::cout << "Throughput: ";
        std::string deli = "";
        if (name.find("fp")!= std::string::npos) {
            deli = "FL";
        }
        const constexpr std::array<std::pair<double, const char*>, 4> units = {{
            {1e12, " T"}, {1e9, " G"}, {1e6, " M"}, {1e3, " K"}
        }};
        for (const auto& [threshold, suffix] : units) {
            if (ops > threshold) {
                std::cout << ops / threshold << suffix;
                break;
            }
        }
        std::cout << deli << "OPS";
        if (result.first != 0.0f) {
            std::cout << "\tAccuracy: " << result.first << "(" << result.second <<"%)";
        }
        std::cout << std::endl;
    }
public:
    void OpRunShader(std::shared_ptr<ComputeDevice> dev)
    {
        const constexpr int num_element = 1024;
        const constexpr uint32_t loop_count = 100;
        ComputeShader<T> shader(dev, code_size, code, loop_count, num_element);

        double duration = MAXFLOAT;
        for (int sloop = 0; sloop < 8; sloop++) {
            shader.OpDispatchCommand(num_element);
            shader.OpSubmitWork(num_element);
            duration = std::fmin(shader.OpGetTimestamp(), duration);
        }
        auto r = shader.OpVerifyWork(num_element, loop_count);
        OpBenchmarkResult(duration, num_element, loop_count, r);
    }

    optestcase(std::string name, const unsigned int code_size, const unsigned char *code, bool enable):
               name(name), code_size(code_size), code(code), enable(enable) {}
    ~optestcase() {}
    std::string getName() { return name; }
    bool enable;
};

enum testcase_type {
#define _(x) x,
    TESTCASES
#undef _
};
int main(int argc, char **argv) {
    std::tuple<optestcase<int64_t>,
#ifdef HAVE_FLOAT64
        optestcase<_Float64>,
#endif
        optestcase<int32_t>, optestcase<float>, optestcase<uint16_t>,
#ifdef HAVE_FLOAT16
        optestcase<_Float16>,
#endif
        optestcase<uint8_t>
#ifdef VK_KHR_shader_integer_dot_product
        , optestcase<uint8_t>, optestcase<uint8_t>, optestcase<uint8_t>
#endif
        > testcases = {
            optestcase<int64_t>("int64", shaderint64_size, shaderint64_code, false),
#ifdef HAVE_FLOAT64
            optestcase<_Float64>("fp64", shaderfp64_size, shaderfp64_code, false),
#endif
            optestcase<int32_t>("int32", shaderint32_size, shaderint32_code, true),
            optestcase<float>("fp32", shaderfp32_size, shaderfp32_code, true),
            optestcase<uint16_t>("int16", shaderint16_size, shaderint16_code, false),
#ifdef HAVE_FLOAT16
            optestcase<_Float16>("fp16", shaderfp16_size, shaderfp16_code, false),
#endif
            optestcase<uint8_t>("int8", shaderint8_size, shaderint8_code, false),
#ifdef VK_KHR_shader_integer_dot_product
            optestcase<uint8_t>("int8dot", shaderint8dot_size, shaderint8dot_code, false),
            optestcase<uint8_t>("int8dotaccsat", shaderint8dotaccsat_size, shaderint8dotaccsat_code, false),
            optestcase<uint8_t>("int8dot4x8packed", shaderint8dot4x8packed_size, shaderint8dot4x8packed_code, false),
#endif
    };

    std::string testname;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]).compare("--help") == 0) {
                std::cout << "Usage: " << argv[0] << " [--help] [--list] [--test <testname>]" << std::endl;
                return 0;
            } else if (std::string(argv[i]).compare("--list") == 0) {
                std::cout << "Available tests:" << std::endl;
                std::apply([](auto&&... test) {((std::cout << test.getName() << '\n'), ...);}, testcases);
                return 0;
            } else if (std::string(argv[i]).compare("--test") == 0) {
                testname = std::string(argv[i+1]);
                i++;
                bool found = false;
                std::apply([&](auto&&... test) {
                    ((((test.getName().compare(testname) == 0) ? (found = true, 0) : 0)), ...);
                }, testcases);

                if (found == false) {
                    std::cout << "invalid testname \"" << testname << "\"" << std::endl;
                    return -1;
                }
            } else {
                std::cout << "invalid argument " << argv[i] << std::endl;
                return -1;
            }
        }
    }

    VulkanInstance vulkanInstance;
    {
        auto r = vulkanInstance.getDeviceAndQeueue();
        auto dev = std::make_shared<ComputeDevice>(r.first, r.second);
#define _(x) std::get<x>(testcases).enable = dev->features & FEATURE_##x;
        TESTCASES
#undef _
        std::apply([&](auto&&... test) {
            (( (!test.enable || (!testname.empty() && test.getName().compare(testname) != 0)) ? 0 : (test.OpRunShader(dev), 0)), ...);
        }, testcases);        
    }

    return 0;
}
