#include <ghost/allocator.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ghost_test.h"
#include "test_allocator_common.h"

using namespace ghost;
using namespace ghost::test;

namespace {

// ---------------------------------------------------------------------------
// Host-memory allocator: returns a tagged buffer prefixed by a sentinel so we
// can verify that freeHostMemory was passed back our pointer.
// ---------------------------------------------------------------------------

class HostMemoryAllocator : public Allocator {
 public:
  static constexpr uint64_t kTag = 0xA110CA70DEADBEEFULL;

  std::atomic<int> allocCalls{0};
  std::atomic<int> freeCalls{0};
  std::atomic<size_t> outstandingBytes{0};

  void* allocateHostMemory(size_t bytes) override {
    allocCalls.fetch_add(1);
    outstandingBytes.fetch_add(bytes);
    // Prefix with the tag so freeHostMemory can validate it received one of
    // our allocations.
    void* raw = ::malloc(sizeof(uint64_t) + bytes);
    *reinterpret_cast<uint64_t*>(raw) = kTag;
    return static_cast<uint8_t*>(raw) + sizeof(uint64_t);
  }

  void freeHostMemory(void* ptr) override {
    freeCalls.fetch_add(1);
    if (!ptr) return;
    void* raw = static_cast<uint8_t*>(ptr) - sizeof(uint64_t);
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(raw), kTag)
        << "freeHostMemory called with a pointer that wasn't tagged by us";
    ::free(raw);
  }
};

}  // namespace

// ===========================================================================
// CPU-specific tests — the CPU backend supports allocator routing for plain
// buffer / image allocations without needing any backend-specific handle
// types.
// ===========================================================================

class CPUAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = std::unique_ptr<Device>(new DeviceCPU(SharedContext()));
  }

  Device& device() { return *device_; }

  Stream stream() { return device_->defaultStream(); }

  std::unique_ptr<Device> device_;
};

// CPU allocator returning real heap pointers. The "handle" the allocator
// returns IS the buffer's data pointer.
namespace {
class CpuAllocator : public Allocator {
 public:
  std::atomic<int> allocCalls{0};
  std::atomic<int> freeCalls{0};
  std::atomic<int> mappedAllocCalls{0};
  std::atomic<int> mappedFreeCalls{0};
  std::atomic<int> imageAllocCalls{0};
  std::atomic<int> imageFreeCalls{0};
  std::atomic<size_t> outstandingBytes{0};

  void* allocateBuffer(size_t bytes, const BufferOptions& opts) override {
    (void)opts;
    allocCalls.fetch_add(1);
    outstandingBytes.fetch_add(bytes);
    return ::malloc(bytes);
  }

  void freeBuffer(void* handle, size_t bytes) override {
    freeCalls.fetch_add(1);
    outstandingBytes.fetch_sub(bytes);
    ::free(handle);
  }

  void* allocateMappedBuffer(size_t bytes, const BufferOptions& opts) override {
    (void)opts;
    mappedAllocCalls.fetch_add(1);
    outstandingBytes.fetch_add(bytes);
    return ::malloc(bytes);
  }

  void freeMappedBuffer(void* handle, size_t bytes) override {
    mappedFreeCalls.fetch_add(1);
    outstandingBytes.fetch_sub(bytes);
    ::free(handle);
  }

  void* allocateImage(const ImageDescription& descr) override {
    imageAllocCalls.fetch_add(1);
    size_t bytes =
        descr.size.x * descr.size.y * descr.size.z * descr.pixelSize();
    outstandingBytes.fetch_add(bytes);
    return ::malloc(bytes);
  }

  void freeImage(void* handle, const ImageDescription& descr) override {
    imageFreeCalls.fetch_add(1);
    size_t bytes =
        descr.size.x * descr.size.y * descr.size.z * descr.pixelSize();
    outstandingBytes.fetch_sub(bytes);
    ::free(handle);
  }
};
}  // namespace

TEST_F(CPUAllocatorTest, NoAllocatorBehavesNormally) {
  // Without an allocator the default code path runs (no routing through
  // the allocator). Smoke test that a buffer round-trip still works.
  const size_t N = 64;
  std::vector<float> input(N, 1.5f), output(N, 0.0f);
  auto buf = device().allocateBuffer(N * sizeof(float));
  buf.copy(stream(), input.data(), N * sizeof(float));
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) EXPECT_FLOAT_EQ(output[i], 1.5f);
}

TEST_F(CPUAllocatorTest, DecliningAllocatorFallsBackToDefault) {
  // The declining allocator returns nullptr from every allocateBuffer; the
  // device should fall through to its default path and the data path should
  // still work.
  auto a = std::make_shared<DecliningAllocator>();
  device().setAllocator(a);

  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto buf = device().allocateBuffer(N * sizeof(float));
  EXPECT_GT(a->bufferCalls.load(), 0) << "allocator should have been asked";

  buf.copy(stream(), input.data(), N * sizeof(float));
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "i=" << i;
}

TEST_F(CPUAllocatorTest, BufferRoundTrip) {
  auto a = std::make_shared<CpuAllocator>();
  device().setAllocator(a);

  const size_t N = 64;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = i * 1.25f;

  {
    auto buf = device().allocateBuffer(N * sizeof(float));
    EXPECT_EQ(a->allocCalls.load(), 1);
    EXPECT_EQ(a->outstandingBytes.load(), N * sizeof(float));
    buf.copy(stream(), input.data(), N * sizeof(float));
    buf.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();
  }
  EXPECT_EQ(a->freeCalls.load(), 1);
  EXPECT_EQ(a->outstandingBytes.load(), 0u);

  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], i * 1.25f) << "i=" << i;
}

TEST_F(CPUAllocatorTest, MappedBufferRoundTrip) {
  auto a = std::make_shared<CpuAllocator>();
  device().setAllocator(a);

  const size_t N = 16;
  std::vector<int32_t> input(N), output(N, 0);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<int32_t>(i);

  {
    auto mb = device().allocateMappedBuffer(N * sizeof(int32_t));
    EXPECT_EQ(a->mappedAllocCalls.load(), 1);
    EXPECT_EQ(a->mappedFreeCalls.load(), 0);
    void* p = mb.map(stream(), Access::ReadWrite);
    memcpy(p, input.data(), N * sizeof(int32_t));
    memcpy(output.data(), p, N * sizeof(int32_t));
    mb.unmap(stream());
  }
  // mb goes out of scope → freeMappedBuffer called exactly once.
  EXPECT_EQ(a->mappedFreeCalls.load(), 1);

  for (size_t i = 0; i < N; i++)
    EXPECT_EQ(output[i], static_cast<int32_t>(i)) << "i=" << i;
}

TEST_F(CPUAllocatorTest, ImageRoundTrip) {
  auto a = std::make_shared<CpuAllocator>();
  device().setAllocator(a);

  ImageDescription d(Size3(8, 4, 1), PixelOrder_RGBA, DataType_Float,
                     Stride2(0, 0));
  // 4 channels (RGBA) * float = 16 bytes per pixel.
  const size_t N = 8 * 4 * 4;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = i + 0.5f;

  {
    auto img = device().allocateImage(d);
    EXPECT_EQ(a->imageAllocCalls.load(), 1);
    img.copy(stream(), input.data());
    img.copyTo(stream(), output.data());
    stream().sync();
  }
  EXPECT_EQ(a->imageFreeCalls.load(), 1);

  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], i + 0.5f) << "i=" << i;
}

TEST_F(CPUAllocatorTest, HostMemoryRoundsThroughAllocator) {
  auto a = std::make_shared<HostMemoryAllocator>();
  device().setAllocator(a);

  void* p = device().allocateHostMemory(128);
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(a->allocCalls.load(), 1);

  // Touch the memory to confirm it's writable.
  memset(p, 0xCC, 128);

  device().freeHostMemory(p);
  EXPECT_EQ(a->freeCalls.load(), 1);
}

TEST_F(CPUAllocatorTest, HostMemoryFallsBackWhenAllocatorDeclines) {
  auto a = std::make_shared<DecliningAllocator>();
  device().setAllocator(a);

  void* p = device().allocateHostMemory(64);
  EXPECT_NE(p, nullptr) << "default path should still succeed";
  EXPECT_EQ(a->hostMemCalls.load(), 1);

  // When an allocator is installed, freeHostMemory always routes through it.
  device().freeHostMemory(p);
  EXPECT_EQ(a->freeHostMemCalls.load(), 1);
}

// ===========================================================================
// SharedBuffer / SharedImage wrap tests (CPU)
// ===========================================================================

TEST_F(CPUAllocatorTest, WrapBufferRoundTrip) {
  const size_t N = 32;
  std::vector<float> host(N);
  for (size_t i = 0; i < N; i++) host[i] = i + 0.25f;

  SharedBuffer sb{host.data(), N * sizeof(float)};
  auto wrapped = device().wrapBuffer(sb);
  EXPECT_EQ(wrapped.size(), N * sizeof(float));

  // Copy out via Ghost APIs; the underlying memory is the host's vector.
  std::vector<float> out(N, 0.0f);
  wrapped.copyTo(stream(), out.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(out[i], i + 0.25f) << "i=" << i;
}

TEST_F(CPUAllocatorTest, WrapBufferGhostDoesNotFreeHostMemory) {
  // Allocate via ::malloc, wrap, destroy the wrapper, then verify the
  // pointer is still readable (we own it; Ghost must not have freed it).
  const size_t N = 16;
  void* p = ::malloc(N * sizeof(int));
  EXPECT_NE(p, nullptr);
  std::memset(p, 0xAB, N * sizeof(int));

  {
    SharedBuffer sb{p, N * sizeof(int)};
    auto wrapped = device().wrapBuffer(sb);
    EXPECT_EQ(wrapped.size(), N * sizeof(int));
  }
  // Wrapper is gone; our pointer must still be valid.
  uint8_t* bytes = static_cast<uint8_t*>(p);
  for (size_t i = 0; i < N * sizeof(int); i++) {
    EXPECT_EQ(bytes[i], 0xAB) << "byte " << i;
  }
  ::free(p);
}

TEST_F(CPUAllocatorTest, WrapImageRoundTrip) {
  ImageDescription d(Size3(4, 4, 1), PixelOrder_RGBA, DataType_Float,
                     Stride2(0, 0));
  // RGBA*float = 16 bytes/pixel, tight-packed.
  const size_t pixels = 4 * 4;
  const size_t N = pixels * 4;  // 4 channels
  std::vector<float> host(N);
  for (size_t i = 0; i < N; i++) host[i] = static_cast<float>(i);

  SharedImage si{host.data(), d};
  auto wrapped = device().wrapImage(si);

  std::vector<float> out(N, 0.0f);
  wrapped.copyTo(stream(), out.data());
  stream().sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(i)) << "i=" << i;
}

// Note: backend-specific allocator tests (Metal, etc.) live in their own
// translation units (e.g. test_allocator_metal.mm) because they need
// backend-only headers (Objective-C, Vulkan, D3D12, ...).
