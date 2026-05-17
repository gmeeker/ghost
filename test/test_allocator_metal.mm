#if WITH_METAL

#import <Metal/Metal.h>

#include <ghost/allocator.h>

#include <atomic>
#include <vector>

#include "ghost_test.h"
#include "test_allocator_common.h"

using namespace ghost;
using namespace ghost::test;

namespace {

// Metal allocator that creates real id<MTLBuffer> / id<MTLTexture> resources
// and hands ownership to Ghost via __bridge_retained. Ghost transfers
// ownership back via __bridge_transfer in freeBuffer / freeImage.
class MetalAllocator : public Allocator {
public:
  id<MTLDevice> dev_;
  std::atomic<int> bufferAlloc{0};
  std::atomic<int> bufferFree{0};
  std::atomic<int> mappedAlloc{0};
  std::atomic<int> mappedFree{0};
  std::atomic<int> imageAlloc{0};
  std::atomic<int> imageFree{0};

  explicit MetalAllocator(id<MTLDevice> dev) : dev_(dev) {}

  void *allocateBuffer(size_t bytes, const BufferOptions &opts) override {
    (void)opts;
    bufferAlloc.fetch_add(1);
    id<MTLBuffer> buf = [dev_ newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
    return (__bridge_retained void *)buf;
  }

  void freeBuffer(void *handle, size_t bytes) override {
    (void)bytes;
    bufferFree.fetch_add(1);
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle;
    (void)buf;
  }

  void *allocateMappedBuffer(size_t bytes, const BufferOptions &opts) override {
    (void)opts;
    mappedAlloc.fetch_add(1);
    id<MTLBuffer> buf = [dev_ newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
    return (__bridge_retained void *)buf;
  }

  void freeMappedBuffer(void *handle, size_t bytes) override {
    (void)bytes;
    mappedFree.fetch_add(1);
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle;
    (void)buf;
  }

  void *allocateImage(const ImageDescription &descr) override {
    imageAlloc.fetch_add(1);
    MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2D;
    td.width = descr.size.x;
    td.height = descr.size.y;
    td.depth = 1;
    // The test uses PixelOrder_RGBA + DataType_Float, so 4-channel f32.
    td.pixelFormat = MTLPixelFormatRGBA32Float;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev_ newTextureWithDescriptor:td];
    return (__bridge_retained void *)tex;
  }

  void freeImage(void *handle, const ImageDescription &descr) override {
    (void)descr;
    imageFree.fetch_add(1);
    id<MTLTexture> tex = (__bridge_transfer id<MTLTexture>)handle;
    (void)tex;
  }
};

std::unique_ptr<Device> makeMetalDevice() {
  try {
    return createDevice(Backend::Metal);
  } catch (const std::exception &) {
    return nullptr;
  }
}

} // namespace

TEST(MetalAllocatorTest, BufferRoundTrip) {
  auto dev = makeMetalDevice();
  if (!dev)
    GTEST_SKIP() << "Metal unavailable";

  auto mtlDev = (__bridge id<MTLDevice>)dev->shareContext().device;
  auto a = std::make_shared<MetalAllocator>(mtlDev);
  dev->setAllocator(a);

  const size_t N = 64;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++)
    input[i] = i * 0.5f;

  {
    auto buf = dev->allocateBuffer(N * sizeof(float));
    EXPECT_EQ(a->bufferAlloc.load(), 1);
    buf.copy(dev->defaultStream(), input.data(), N * sizeof(float));
    buf.copyTo(dev->defaultStream(), output.data(), N * sizeof(float));
    dev->defaultStream().sync();
  }
  EXPECT_EQ(a->bufferFree.load(), 1);
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], i * 0.5f) << "i=" << i;
}

TEST(MetalAllocatorTest, MappedBufferRoundTrip) {
  auto dev = makeMetalDevice();
  if (!dev)
    GTEST_SKIP() << "Metal unavailable";

  auto mtlDev = (__bridge id<MTLDevice>)dev->shareContext().device;
  auto a = std::make_shared<MetalAllocator>(mtlDev);
  dev->setAllocator(a);

  const size_t N = 32;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++)
    input[i] = i * 2.0f;

  {
    auto mb = dev->allocateMappedBuffer(N * sizeof(float));
    EXPECT_EQ(a->mappedAlloc.load(), 1);
    void *p = mb.map(dev->defaultStream(), Access::ReadWrite);
    memcpy(p, input.data(), N * sizeof(float));
    std::vector<float> out(N, 0.0f);
    memcpy(out.data(), p, N * sizeof(float));
    mb.unmap(dev->defaultStream());
    for (size_t i = 0; i < N; i++)
      EXPECT_FLOAT_EQ(out[i], i * 2.0f);
  }
  EXPECT_EQ(a->mappedFree.load(), 1);
}

TEST(MetalAllocatorTest, ImageRoundTrip) {
  auto dev = makeMetalDevice();
  if (!dev)
    GTEST_SKIP() << "Metal unavailable";

  auto mtlDev = (__bridge id<MTLDevice>)dev->shareContext().device;
  auto a = std::make_shared<MetalAllocator>(mtlDev);
  dev->setAllocator(a);

  ImageDescription d(Size3(8, 4, 1), PixelOrder_RGBA, DataType_Float,
                     Stride2(0, 0));
  // RGBA * float = 16 bytes/pixel, 8*4*4 floats total.
  const size_t N = 8 * 4 * 4;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++)
    input[i] = static_cast<float>(i);

  {
    auto img = dev->allocateImage(d);
    EXPECT_EQ(a->imageAlloc.load(), 1);
    img.copy(dev->defaultStream(), input.data());
    img.copyTo(dev->defaultStream(), output.data());
    dev->defaultStream().sync();
  }
  EXPECT_EQ(a->imageFree.load(), 1);
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "i=" << i;
}

// ===========================================================================
// SharedBuffer / SharedImage wrap tests (Metal)
// ===========================================================================

TEST(MetalWrapTest, WrapBufferRoundTrip) {
  auto dev = makeMetalDevice();
  if (!dev)
    GTEST_SKIP() << "Metal unavailable";

  auto mtlDev = (__bridge id<MTLDevice>)dev->shareContext().device;
  const size_t N = 64;

  // Host owns the MTLBuffer end-to-end.
  id<MTLBuffer> hostBuf =
      [mtlDev newBufferWithLength:N * sizeof(float)
                          options:MTLResourceStorageModeShared];
  ASSERT_NE(hostBuf, nil);

  // Fill via the host pointer.
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++)
    input[i] = i * 0.75f;
  memcpy([hostBuf contents], input.data(), N * sizeof(float));

  {
    SharedBuffer sb{(__bridge void *)hostBuf, N * sizeof(float)};
    auto wrapped = dev->wrapBuffer(sb);

    std::vector<float> out(N, 0.0f);
    wrapped.copyTo(dev->defaultStream(), out.data(), N * sizeof(float));
    dev->defaultStream().sync();
    for (size_t i = 0; i < N; i++)
      EXPECT_FLOAT_EQ(out[i], i * 0.75f) << "i=" << i;
  }
  // Wrapper gone; hostBuf should still be alive and writable.
  memset([hostBuf contents], 0, N * sizeof(float));
  // Touching contents would crash if the buffer was freed.
}

TEST(MetalWrapTest, WrapImageRoundTrip) {
  auto dev = makeMetalDevice();
  if (!dev)
    GTEST_SKIP() << "Metal unavailable";

  auto mtlDev = (__bridge id<MTLDevice>)dev->shareContext().device;

  MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
  td.textureType = MTLTextureType2D;
  td.width = 8;
  td.height = 4;
  td.depth = 1;
  td.pixelFormat = MTLPixelFormatRGBA32Float;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> hostTex = [mtlDev newTextureWithDescriptor:td];
  ASSERT_NE(hostTex, nil);

  ImageDescription d(Size3(8, 4, 1), PixelOrder_RGBA, DataType_Float,
                     Stride2(0, 0));
  const size_t N = 8 * 4 * 4;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++)
    input[i] = static_cast<float>(i);

  {
    SharedImage si{(__bridge void *)hostTex, d};
    auto wrapped = dev->wrapImage(si);
    wrapped.copy(dev->defaultStream(), input.data());
    wrapped.copyTo(dev->defaultStream(), output.data());
    dev->defaultStream().sync();
  }
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "i=" << i;

  // hostTex must still be valid — we own it.
  EXPECT_EQ([hostTex width], 8u);
}

TEST(MetalAllocatorTest, DecliningFallsBackToDefault) {
  auto dev = makeMetalDevice();
  if (!dev)
    GTEST_SKIP() << "Metal unavailable";

  auto a = std::make_shared<DecliningAllocator>();
  dev->setAllocator(a);

  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++)
    input[i] = i + 100.0f;

  auto buf = dev->allocateBuffer(N * sizeof(float));
  EXPECT_GT(a->bufferCalls.load(), 0);

  buf.copy(dev->defaultStream(), input.data(), N * sizeof(float));
  buf.copyTo(dev->defaultStream(), output.data(), N * sizeof(float));
  dev->defaultStream().sync();

  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], i + 100.0f) << "i=" << i;
}

#endif // WITH_METAL
