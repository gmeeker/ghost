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
                  kDeviceSupportsProgramConstants, kDeviceSupportsSubgroup}) {
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
