#ifndef GHOST_TEST_ALLOCATOR_COMMON_H
#define GHOST_TEST_ALLOCATOR_COMMON_H

#include <ghost/allocator.h>

#include <atomic>
#include <cstdlib>

namespace ghost {
namespace test {

// Mock allocator that returns nullptr from every allocateX method, exercising
// Ghost's "decline and fall back to default" path while still counting calls.
class DecliningAllocator : public Allocator {
 public:
  std::atomic<int> bufferCalls{0};
  std::atomic<int> mappedCalls{0};
  std::atomic<int> imageCalls{0};
  std::atomic<int> hostMemCalls{0};
  std::atomic<int> freeHostMemCalls{0};

  void* allocateBuffer(size_t bytes, const BufferOptions& opts) override {
    (void)bytes;
    (void)opts;
    bufferCalls.fetch_add(1);
    return nullptr;
  }

  void* allocateMappedBuffer(size_t bytes, const BufferOptions& opts) override {
    (void)bytes;
    (void)opts;
    mappedCalls.fetch_add(1);
    return nullptr;
  }

  void* allocateImage(const ImageDescription& descr) override {
    (void)descr;
    imageCalls.fetch_add(1);
    return nullptr;
  }

  void* allocateHostMemory(size_t bytes) override {
    (void)bytes;
    hostMemCalls.fetch_add(1);
    return nullptr;
  }

  void freeHostMemory(void* ptr) override {
    freeHostMemCalls.fetch_add(1);
    if (ptr) ::free(ptr);
  }
};

}  // namespace test
}  // namespace ghost

#endif
