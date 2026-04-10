// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef GHOST_VULKAN_PTR_H
#define GHOST_VULKAN_PTR_H

#include <vulkan/vulkan.h>

#include <memory>

namespace ghost {
namespace vk {

/// @brief Traits class mapping a Vulkan handle type to its destroy function.
///
/// Specialize for each handle type Ghost manages. The Vulkan API requires the
/// parent VkDevice to be passed to most destroy functions, so the trait takes
/// it as the first argument.
template <typename T>
class detail {};

template <>
class detail<VkBuffer> {
 public:
  static void release(VkDevice dev, VkBuffer v) {
    vkDestroyBuffer(dev, v, nullptr);
  }
};

template <>
class detail<VkDeviceMemory> {
 public:
  static void release(VkDevice dev, VkDeviceMemory v) {
    vkFreeMemory(dev, v, nullptr);
  }
};

template <>
class detail<VkImage> {
 public:
  static void release(VkDevice dev, VkImage v) {
    vkDestroyImage(dev, v, nullptr);
  }
};

template <>
class detail<VkImageView> {
 public:
  static void release(VkDevice dev, VkImageView v) {
    vkDestroyImageView(dev, v, nullptr);
  }
};

template <>
class detail<VkShaderModule> {
 public:
  static void release(VkDevice dev, VkShaderModule v) {
    vkDestroyShaderModule(dev, v, nullptr);
  }
};

template <>
class detail<VkPipeline> {
 public:
  static void release(VkDevice dev, VkPipeline v) {
    vkDestroyPipeline(dev, v, nullptr);
  }
};

template <>
class detail<VkPipelineLayout> {
 public:
  static void release(VkDevice dev, VkPipelineLayout v) {
    vkDestroyPipelineLayout(dev, v, nullptr);
  }
};

template <>
class detail<VkDescriptorSetLayout> {
 public:
  static void release(VkDevice dev, VkDescriptorSetLayout v) {
    vkDestroyDescriptorSetLayout(dev, v, nullptr);
  }
};

template <>
class detail<VkDescriptorPool> {
 public:
  static void release(VkDevice dev, VkDescriptorPool v) {
    vkDestroyDescriptorPool(dev, v, nullptr);
  }
};

template <>
class detail<VkCommandPool> {
 public:
  static void release(VkDevice dev, VkCommandPool v) {
    vkDestroyCommandPool(dev, v, nullptr);
  }
};

template <>
class detail<VkFence> {
 public:
  static void release(VkDevice dev, VkFence v) {
    vkDestroyFence(dev, v, nullptr);
  }
};

/// @brief Smart pointer for Vulkan handles whose destruction requires a parent
/// VkDevice.
///
/// Mirrors @c ghost::cu::ptr and @c ghost::opencl::ptr: the wrapper owns its
/// handle and calls the appropriate @c vkDestroyXxx in the destructor. Each
/// handle carries its own ownership flag, so aliased sub-resources (e.g.
/// image-from-image, sub-buffers) can be marked non-owning at the handle level
/// without needing per-class @c ownsHandles bookkeeping.
///
/// VkInstance and VkDevice are intentionally not specialized: their destroy
/// functions take no parent and they are owned only by @c DeviceVulkan, which
/// destroys them directly.
///
/// Not copyable. Move-only, like @c std::unique_ptr.
template <typename T, typename DETAIL = detail<T>>
class ptr {
 protected:
  VkDevice _device;
  T _value;
  bool _owned;

 public:
  /// @brief Construct an empty (null) handle wrapper.
  ptr() : _device(VK_NULL_HANDLE), _value(VK_NULL_HANDLE), _owned(false) {}

  /// @brief Construct from a device and (optionally) an existing handle.
  /// @param dev The parent device used for destruction.
  /// @param v The handle, or @c VK_NULL_HANDLE.
  /// @param retainObject If true, this wrapper takes ownership and will
  ///   destroy @p v on reset/destruction. Set to false for borrowed handles.
  explicit ptr(VkDevice dev, T v = VK_NULL_HANDLE, bool retainObject = true)
      : _device(dev), _value(v), _owned(retainObject && v != VK_NULL_HANDLE) {}

  ptr(const ptr&) = delete;
  ptr& operator=(const ptr&) = delete;

  ptr(ptr&& v) noexcept
      : _device(v._device), _value(v._value), _owned(v._owned) {
    v._value = VK_NULL_HANDLE;
    v._owned = false;
  }

  ptr& operator=(ptr&& v) noexcept {
    // Use std::addressof since &v would invoke our destroying operator&.
    if (this != std::addressof(v)) {
      destroy();
      _device = v._device;
      _value = v._value;
      _owned = v._owned;
      v._value = VK_NULL_HANDLE;
      v._owned = false;
    }
    return *this;
  }

  ~ptr() { destroy(); }

  /// @brief Destroy the held handle (if owned) and reset to null.
  void destroy() {
    if (_owned && _value != VK_NULL_HANDLE) {
      DETAIL::release(_device, _value);
    }
    _value = VK_NULL_HANDLE;
    _owned = false;
  }

  /// @brief Equivalent to destroy().
  void reset() { destroy(); }

  /// @brief Release ownership of the held handle without destroying it.
  /// @return The previously held handle.
  T release() {
    T v = _value;
    _value = VK_NULL_HANDLE;
    _owned = false;
    return v;
  }

  /// @brief Get the underlying Vulkan handle.
  T get() const { return _value; }

  /// @brief Get the parent device this handle belongs to.
  VkDevice device() const { return _device; }

  /// @brief Whether this wrapper owns the held handle.
  bool owned() const { return _owned; }

  /// @brief Implicit conversion to the underlying handle for use in Vulkan
  /// API calls that take a value parameter.
  operator T() const { return _value; }

  /// @brief Address-of for use as the out-parameter of @c vkCreateXxx.
  ///
  /// Destroys any previously held handle, then returns a pointer to the
  /// internal value. After @c vkCreateXxx writes to it, this wrapper owns
  /// the new handle. The device must already be set on this wrapper.
  ///
  /// For Vulkan APIs that take a @c const T* to an existing handle (e.g.
  /// @c vkWaitForFences, @c vkResetFences), use a local temporary
  /// instead — @c operator& destroys the held value, which is the wrong
  /// semantics for those calls.
  T* operator&() {
    destroy();
    _owned = true;
    return &_value;
  }

  /// @brief Set the parent device.
  ///
  /// Used when the handle is constructed before the device is known, e.g.
  /// when initializing members in a class constructor body.
  void setDevice(VkDevice dev) { _device = dev; }
};

}  // namespace vk
}  // namespace ghost

#endif
