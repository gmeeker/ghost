#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

// ---------------------------------------------------------------------------
// Device creation (parameterized across all backends)
// ---------------------------------------------------------------------------

class DeviceTest : public GhostTest {};

TEST_P(DeviceTest, CreationSucceeds) { EXPECT_NE(device_.get(), nullptr); }

TEST_P(DeviceTest, DefaultStreamSync) {
  // Syncing an empty stream should not hang or throw.
  EXPECT_NO_THROW(stream().sync());
}

TEST_P(DeviceTest, CreateStream) {
  auto s = device().createStream();
  EXPECT_NO_THROW(s.sync());
}

TEST_P(DeviceTest, ImplementationName) {
  auto attr = device().getAttribute(kDeviceImplementation);
  EXPECT_EQ(attr.type(), Attribute::Type_String);
  EXPECT_FALSE(attr.asString().empty());
}

TEST_P(DeviceTest, DeviceName) {
  auto attr = device().getAttribute(kDeviceName);
  EXPECT_EQ(attr.type(), Attribute::Type_String);
  EXPECT_FALSE(attr.asString().empty());
}

TEST_P(DeviceTest, DeviceVendor) {
  auto attr = device().getAttribute(kDeviceVendor);
  EXPECT_EQ(attr.type(), Attribute::Type_String);
  // Vendor may be empty for CPU backend, so just check type.
}

TEST_P(DeviceTest, DeviceMemory) {
  auto attr = device().getAttribute(kDeviceMemory);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asUInt64(), 0u);
}

TEST_P(DeviceTest, MaxThreads) {
  auto attr = device().getAttribute(kDeviceMaxThreads);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(DeviceTest, MaxWorkSize) {
  auto attr = device().getAttribute(kDeviceMaxWorkSize);
  EXPECT_TRUE(attr.valid());
  EXPECT_GE(attr.count(), 3u);
  EXPECT_GT(attr.intArray()[0], 0);
}

TEST_P(DeviceTest, UnifiedMemory) {
  auto attr = device().getAttribute(kDeviceUnifiedMemory);
  EXPECT_TRUE(attr.valid());
  EXPECT_EQ(attr.type(), Attribute::Type_Bool);
}

TEST_P(DeviceTest, QueryStringAttributes) {
  // These should all return valid string attributes (possibly empty).
  for (auto id : {kDeviceImplementation, kDeviceName, kDeviceVendor,
                  kDeviceDriverVersion}) {
    auto attr = device().getAttribute(id);
    EXPECT_EQ(attr.type(), Attribute::Type_String) << "Attribute id=" << id;
  }
}

TEST_P(DeviceTest, QueryBoolAttributes) {
  for (auto id : {kDeviceUnifiedMemory, kDeviceSupportsMappedBuffer,
                  kDeviceSupportsProgramConstants,
                  kDeviceSupportsProgramGlobals, kDeviceSupportsSubgroup}) {
    auto attr = device().getAttribute(id);
    EXPECT_EQ(attr.type(), Attribute::Type_Bool) << "Attribute id=" << id;
  }
}

TEST_P(DeviceTest, MemoryPoolDefault) {
  EXPECT_EQ(device().getMemoryPoolSize(), 0u);
}

TEST_P(DeviceTest, MemoryPoolSetGet) {
  device().setMemoryPoolSize(1024 * 1024);
  EXPECT_EQ(device().getMemoryPoolSize(), 1024u * 1024u);
  device().setMemoryPoolSize(0);
  EXPECT_EQ(device().getMemoryPoolSize(), 0u);
}

TEST_P(DeviceTest, ShareContext) {
  auto ctx = device().shareContext();
  // The context should contain at least a non-null device or context handle
  // for GPU backends.  CPU may return all nulls.
  if (backend() != Backend::CPU) {
    bool hasHandle = ctx.context != nullptr || ctx.device != nullptr;
    EXPECT_TRUE(hasHandle);
  }
}

// ---------------------------------------------------------------------------
// Extended device attributes (section 3a)
// ---------------------------------------------------------------------------

TEST_P(DeviceTest, MaxComputeUnits) {
  auto attr = device().getAttribute(kDeviceMaxComputeUnits);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(DeviceTest, MemoryAlignment) {
  auto attr = device().getAttribute(kDeviceMemoryAlignment);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(DeviceTest, BufferAlignment) {
  auto attr = device().getAttribute(kDeviceBufferAlignment);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(DeviceTest, MaxBufferSize) {
  auto attr = device().getAttribute(kDeviceMaxBufferSize);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asUInt64(), 0u);
}

TEST_P(DeviceTest, MaxConstantBufferSize) {
  auto attr = device().getAttribute(kDeviceMaxConstantBufferSize);
  EXPECT_TRUE(attr.valid());
  // Some backends may return 0 if constant buffers aren't supported.
  EXPECT_GE(attr.asUInt64(), 0u);
}

TEST_P(DeviceTest, TimestampPeriod) {
  auto attr = device().getAttribute(kDeviceTimestampPeriod);
  EXPECT_TRUE(attr.valid());
}

TEST_P(DeviceTest, SupportsProfilingTimer) {
  auto attr = device().getAttribute(kDeviceSupportsProfilingTimer);
  EXPECT_TRUE(attr.valid());
  EXPECT_EQ(attr.type(), Attribute::Type_Bool);
}

TEST_P(DeviceTest, SupportsCooperativeMatrix) {
  auto attr = device().getAttribute(kDeviceSupportsCooperativeMatrix);
  EXPECT_TRUE(attr.valid());
  EXPECT_EQ(attr.type(), Attribute::Type_Bool);
}

TEST_P(DeviceTest, DeviceFamily) {
  auto attr = device().getAttribute(kDeviceFamily);
  // All backends except OpenCL/CPU should return a non-empty family string.
  if (backend() != Backend::CPU && backend() != Backend::OpenCL) {
    EXPECT_TRUE(attr.valid());
    EXPECT_EQ(attr.type(), Attribute::Type_String);
    EXPECT_FALSE(attr.asString().empty());
  }
}

// ---------------------------------------------------------------------------
// Buffer alignment with sub-buffers
// ---------------------------------------------------------------------------

TEST_P(DeviceTest, SubBufferAtAlignment) {
  auto alignAttr = device().getAttribute(kDeviceBufferAlignment);
  if (!alignAttr.valid() || alignAttr.asInt() <= 0) {
    GTEST_SKIP() << "Buffer alignment not available";
  }

  size_t alignment = static_cast<size_t>(alignAttr.asInt());
  size_t parentSize = alignment * 4;
  auto parent = device().allocateBuffer(parentSize);

  try {
    auto sub = parent.createSubBuffer(alignment, alignment);
    EXPECT_EQ(sub.size(), alignment);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Sub-buffers not supported";
  }
}

// ---------------------------------------------------------------------------
// Multiple devices of same backend
// ---------------------------------------------------------------------------

TEST_P(DeviceTest, MultipleDevicesSameBackend) {
  std::vector<GpuInfo> devices;
  try {
    devices = enumerateDevices(backend());
  } catch (const std::exception&) {
    GTEST_SKIP() << "Enumeration failed";
  }

  if (devices.size() < 2) {
    GTEST_SKIP() << "Only " << devices.size() << " device(s) available";
  }

  // Create two devices and verify both are functional.
  auto dev1 = createDevice(backend());
  auto dev2 = createDevice(backend());
  ASSERT_NE(dev1.get(), nullptr);
  ASSERT_NE(dev2.get(), nullptr);

  auto buf1 = dev1->allocateBuffer(256);
  auto buf2 = dev2->allocateBuffer(256);
  EXPECT_EQ(buf1.size(), 256u);
  EXPECT_EQ(buf2.size(), 256u);
}

GHOST_INSTANTIATE_BACKEND_TESTS(DeviceTest);

// ---------------------------------------------------------------------------
// GpuInfo enumeration (parameterized across all backends)
// ---------------------------------------------------------------------------

class GpuInfoTest : public testing::TestWithParam<Backend> {};

TEST_P(GpuInfoTest, EnumerateDevices) {
  std::vector<GpuInfo> devices;
  try {
    devices = enumerateDevices(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << BackendName(GetParam())
                 << " enumeration failed: " << e.what();
  }

  // CPU always has at least 1 device.  GPU backends may have 0 if no
  // hardware is present (not a test failure).
  if (GetParam() == Backend::CPU) {
    ASSERT_GE(devices.size(), 1u);
  }

  for (const auto& info : devices) {
    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.implementation.empty());
    EXPECT_GT(info.memory, 0u);
  }
}

TEST_P(GpuInfoTest, CreateDeviceFromInfo) {
  std::vector<GpuInfo> devices;
  try {
    devices = enumerateDevices(GetParam());
  } catch (const std::exception&) {
    GTEST_SKIP() << BackendName(GetParam()) << " enumeration failed";
  }
  if (devices.empty()) {
    GTEST_SKIP() << BackendName(GetParam()) << " no devices found";
  }

  // Create a device from the first enumerated GpuInfo.
  std::unique_ptr<Device> dev;
  try {
    switch (GetParam()) {
      case Backend::CPU:
        dev = std::make_unique<DeviceCPU>(devices[0]);
        break;
#if WITH_METAL
      case Backend::Metal:
        dev = std::make_unique<DeviceMetal>(devices[0]);
        break;
#endif
#if WITH_OPENCL
      case Backend::OpenCL:
        dev = std::make_unique<DeviceOpenCL>(devices[0]);
        break;
#endif
#if WITH_CUDA
      case Backend::CUDA:
        dev = std::make_unique<DeviceCUDA>(devices[0]);
        break;
#endif
#if WITH_VULKAN
      case Backend::Vulkan:
        dev = std::make_unique<DeviceVulkan>(devices[0]);
        break;
#endif
#if WITH_DIRECTX
      case Backend::DirectX:
        dev = std::make_unique<DeviceDirectX>(devices[0]);
        break;
#endif
      default:
        GTEST_SKIP() << BackendName(GetParam()) << " not compiled";
        return;
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << BackendName(GetParam())
                 << " device creation failed: " << e.what();
  }

  ASSERT_NE(dev.get(), nullptr);
  // Verify the device is functional by querying its name.
  auto attr = dev->getAttribute(kDeviceName);
  EXPECT_FALSE(attr.asString().empty());
}

INSTANTIATE_TEST_SUITE_P(AllBackends, GpuInfoTest,
                         testing::ValuesIn(availableBackends()),
                         BackendNameGenerator());

// ---------------------------------------------------------------------------
// Cross-backend enumeration and preferred device selection
// ---------------------------------------------------------------------------

TEST(PreferredDeviceTest, EnumerateAll) {
  auto devices = ghost::enumerateDevices();
  // Should find at least one device (CPU is always available via backends).
  // But enumerateDevices() only includes compiled GPU backends, so it may be
  // empty if only CPU is compiled.  Just verify it doesn't throw.
  for (const auto& info : devices) {
    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.implementation.empty());
  }
}

TEST(PreferredDeviceTest, PreferredReturnsDevice) {
  auto devices = ghost::enumerateDevices();
  if (devices.empty()) {
    GTEST_SKIP() << "No GPU devices available";
  }

  auto best = ghost::preferredDevice();
  ASSERT_TRUE(best.has_value());
  EXPECT_FALSE(best->name.empty());
  EXPECT_GT(best->memory, 0u);
}

TEST(PreferredDeviceTest, PreferredFromList) {
  // Discrete should beat integrated, higher VRAM should beat lower.
  GpuInfo integrated;
  integrated.name = "Integrated";
  integrated.implementation = "Metal";
  integrated.memory = 1024;
  integrated.unifiedMemory = true;

  GpuInfo discrete;
  discrete.name = "Discrete";
  discrete.implementation = "Metal";
  discrete.memory = 4096;
  discrete.unifiedMemory = false;

  GpuInfo discreteSmall;
  discreteSmall.name = "DiscreteSmall";
  discreteSmall.implementation = "Metal";
  discreteSmall.memory = 2048;
  discreteSmall.unifiedMemory = false;

  std::vector<GpuInfo> devices = {integrated, discreteSmall, discrete};
  auto best = ghost::preferredDevice(devices);
  ASSERT_TRUE(best.has_value());
  EXPECT_EQ(best->name, "Discrete");
}

TEST(PreferredDeviceTest, PreferredFromEmptyList) {
  std::vector<GpuInfo> empty;
  auto best = ghost::preferredDevice(empty);
  EXPECT_FALSE(best.has_value());
}

TEST(PreferredDeviceTest, PreferredWithBackendFilter) {
  // Just verify it doesn't crash; may return nullopt if no Metal/CUDA devices.
  auto best = ghost::preferredDevice(Backend::Metal);
  // No assertion on value — hardware-dependent.
  (void)best;
}
