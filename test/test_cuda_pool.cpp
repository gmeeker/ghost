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

// CUDA-specific coverage for the memory-pool allocation path and deferred
// release: stream-ordered alloc/free (no host sync per pooled allocation,
// freed memory returns to the pool), and host-allocator interaction
// (allocator precedence over the pool; frees drained before handback).

#include "ghost_test.h"

#if WITH_CUDA

#include <cuda.h>
#include <ghost/allocator.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

using namespace ghost;
using namespace ghost::test;

namespace {

class CudaPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      device_ = createDevice(Backend::CUDA);
      if (!device_) {
        GTEST_SKIP() << "CUDA not compiled";
      }
    } catch (const std::exception& e) {
      GTEST_SKIP() << "CUDA unavailable: " << e.what();
    }
  }

  void TearDown() override {
    if (device_) {
      device_->setMemoryPoolSize(0);
      device_->setAllocator(nullptr);
    }
  }

  Device& device() { return *device_; }

  Stream stream() { return device_->defaultStream(); }

  std::unique_ptr<Device> device_;
};

// Regression for the pool leak: pooled allocations were created non-owning,
// so no cuMemFreeAsync ever returned them to the pool — every transient
// buffer permanently consumed device memory until pool destruction.
TEST_F(CudaPoolTest, PooledAllocationsReturnToPool) {
  const size_t kPoolBytes = 64ull << 20;
  const size_t kBufBytes = 32ull << 20;
  const int kIters = 40;
  device().setMemoryPoolSize(kPoolBytes);

  // One warm-up cycle so the pool's retained memory is in the baseline.
  {
    auto warm = device().allocateBuffer(kBufBytes, AllocHint::Transient);
    warm.fill(stream(), 0, kBufBytes, uint8_t(0));
    stream().sync();
  }
  ASSERT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);
  size_t freeBefore = 0, total = 0;
  ASSERT_EQ(cuMemGetInfo(&freeBefore, &total), CUDA_SUCCESS);

  for (int i = 0; i < kIters; ++i) {
    auto buf = device().allocateBuffer(kBufBytes, AllocHint::Transient);
    buf.fill(stream(), 0, kBufBytes, uint8_t(i));
    // Dropped with the fill still enqueued — the free must be deferred AND
    // must actually return the memory to the pool.
  }
  stream().sync();
  // Memory above the pool's release threshold is reclaimed at sync points.
  ASSERT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);
  size_t freeAfter = 0;
  ASSERT_EQ(cuMemGetInfo(&freeAfter, &total), CUDA_SUCCESS);

  // Pre-fix this leaked kIters * kBufBytes = 1.25 GB. Allow generous slack
  // for pool retention and allocation granularity.
  const size_t kSlack = kPoolBytes + (256ull << 20);
  EXPECT_GT(freeAfter + kSlack, freeBefore)
      << "pooled allocations were not returned to the pool: free memory "
      << "dropped by " << ((freeBefore - freeAfter) >> 20) << " MB";
}

// FR #25: pooled allocation is stream-ordered on the device's internal
// queue with no host sync; use on a different stream must see initialized,
// allocation-complete memory via the device-side ready-event dependency.
TEST_F(CudaPoolTest, PooledBufferCrossStreamRoundTrip) {
  device().setMemoryPoolSize(16ull << 20);
  auto other = device().createStream();

  const size_t N = 1 << 20;
  std::vector<uint8_t> in(N), out(N);
  for (size_t i = 0; i < N; ++i) in[i] = uint8_t(i * 31 + 7);

  for (int iter = 0; iter < 8; ++iter) {
    auto buf = device().allocateBuffer(N, AllocHint::Transient);
    std::fill(out.begin(), out.end(), 0);
    buf.copy(other, in.data(), N);
    buf.copyTo(other, out.data(), N);
    other.sync();
    ASSERT_EQ(out, in) << "iteration " << iter;
  }
}

// Pool reuse must not corrupt work still pending on the previous owner's
// stream: buffer A's readback races buffer B's fill of (likely) the same
// pool memory on another stream.
TEST_F(CudaPoolTest, PoolReuseDoesNotCorruptPendingWork) {
  device().setMemoryPoolSize(64ull << 20);
  auto s1 = device().createStream();
  auto s2 = device().createStream();

  const size_t N = 8ull << 20;
  std::vector<uint8_t> a(N, 0xAB), resultA(N), resultB(N);

  for (int iter = 0; iter < 10; ++iter) {
    std::fill(resultA.begin(), resultA.end(), 0);
    std::fill(resultB.begin(), resultB.end(), 0);
    {
      auto bufA = device().allocateBuffer(N, AllocHint::Transient);
      bufA.copy(s1, a.data(), N);
      bufA.copyTo(s1, resultA.data(), N);
      // bufA dropped — its free is ordered behind the s1 readback.
    }
    {
      auto bufB = device().allocateBuffer(N, AllocHint::Transient);
      bufB.fill(s2, 0, N, uint8_t(0xCD));
      bufB.copyTo(s2, resultB.data(), N);
    }
    s1.sync();
    s2.sync();
    for (size_t i = 0; i < N; i += N / 64) {
      ASSERT_EQ(resultA[i], 0xAB) << "iteration " << iter << " index " << i;
      ASSERT_EQ(resultB[i], 0xCD) << "iteration " << iter << " index " << i;
    }
  }
}

// FR #24: a buffer wrapper dropped while its stream still has work enqueued
// must defer the free behind that work (stream-ordered, no host callback).
TEST_F(CudaPoolTest, DroppedBufferFreeIsStreamOrdered) {
  const size_t N = 16ull << 20;
  std::vector<uint8_t> in(N, 0x77), out(N, 0);
  {
    auto buf = device().allocateBuffer(N);
    buf.copy(stream(), in.data(), N);
    buf.copyTo(stream(), out.data(), N);
    // Dropped before sync — the free must wait for both copies.
  }
  stream().sync();
  EXPECT_EQ(out, in);
}

// sharedImage donor lifetime: the image holds a non-owning view plus a
// keepalive of the donor buffer. Previously the image silently *stole*
// ownership from the buffer (cu::ptr lvalue-copy semantics), so one side
// freeing left the other dangling.

TEST_F(CudaPoolTest, SharedImageKeepsDonorBufferAlive) {
  const size_t W = 4, H = 4, C = 4;
  const size_t pixelCount = W * H * C;
  const size_t dataSize = pixelCount * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(int32_t(W * C * sizeof(float)), 0));

  std::vector<float> input(pixelCount), output(pixelCount, -1.0f);
  for (size_t i = 0; i < pixelCount; i++) input[i] = float(i);

  Image img(nullptr);
  {
    auto buf = device().allocateBuffer(dataSize);
    buf.copy(stream(), input.data(), dataSize);
    img = device().sharedImage(descr, buf);
    // Buffer wrapper dropped here — the image must keep the memory alive.
  }
  img.copyTo(stream(), output.data(), descr);
  stream().sync();
  for (size_t i = 0; i < pixelCount; i++) {
    ASSERT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

TEST_F(CudaPoolTest, DonorBufferSurvivesSharedImageDrop) {
  const size_t W = 4, H = 4, C = 4;
  const size_t pixelCount = W * H * C;
  const size_t dataSize = pixelCount * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(int32_t(W * C * sizeof(float)), 0));

  auto buf = device().allocateBuffer(dataSize);
  {
    Image img = device().sharedImage(descr, buf);
    // Image dropped here — it must NOT free the donor's memory (it used to
    // steal ownership and free at this point).
  }
  std::vector<float> input(pixelCount), output(pixelCount, -1.0f);
  for (size_t i = 0; i < pixelCount; i++) input[i] = float(i * 3 + 1);
  buf.copy(stream(), input.data(), dataSize);
  buf.copyTo(stream(), output.data(), dataSize);
  stream().sync();
  for (size_t i = 0; i < pixelCount; i++) {
    ASSERT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

class CountingDecliningAllocator : public Allocator {
 public:
  std::atomic<int> bufferAsks{0};

  void* allocateBuffer(size_t, const BufferOptions&) override {
    bufferAsks.fetch_add(1);
    return nullptr;  // decline — Ghost falls through to its own paths
  }
};

// A host-installed allocator overrides every Ghost-internal path: it must be
// consulted before the memory pool, and declining must fall through to a
// working pooled allocation.
TEST_F(CudaPoolTest, AllocatorConsultedBeforePool) {
  auto alloc = std::make_shared<CountingDecliningAllocator>();
  device().setAllocator(alloc);
  device().setMemoryPoolSize(16ull << 20);

  const size_t N = 1 << 20;
  std::vector<uint8_t> in(N, 0x42), out(N, 0);
  {
    auto buf = device().allocateBuffer(N, AllocHint::Transient);
    EXPECT_EQ(alloc->bufferAsks.load(), 1)
        << "allocator must be consulted before the memory pool";
    buf.copy(stream(), in.data(), N);
    buf.copyTo(stream(), out.data(), N);
    stream().sync();
  }
  EXPECT_EQ(out, in);
}

class CudaDeviceMemoryAllocator : public Allocator {
 public:
  std::atomic<int> allocCalls{0};
  std::atomic<int> freeCalls{0};

  void* allocateBuffer(size_t bytes, const BufferOptions&) override {
    CUdeviceptr p = 0;
    if (cuMemAlloc(&p, bytes) != CUDA_SUCCESS) return nullptr;
    allocCalls.fetch_add(1);
    return reinterpret_cast<void*>(static_cast<uintptr_t>(p));
  }

  void freeBuffer(void* handle, size_t) override {
    freeCalls.fetch_add(1);
    // The host may recycle (here: release) the memory immediately — Ghost
    // must have drained device work referencing it before this call.
    cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(handle)));
  }
};

// Dropping an allocator-backed buffer with copies in flight must drain the
// device before freeBuffer hands the memory back to the host. The readback
// is verified WITHOUT a stream sync — the destructor's drain is the only
// thing that makes `out` valid here.
TEST_F(CudaPoolTest, AllocatorFreeWaitsForPendingWork) {
  auto alloc = std::make_shared<CudaDeviceMemoryAllocator>();
  device().setAllocator(alloc);

  const size_t N = 32ull << 20;
  std::vector<uint8_t> in(N, 0x5A), out(N, 0);
  {
    auto buf = device().allocateBuffer(N);
    buf.copy(stream(), in.data(), N);
    buf.copyTo(stream(), out.data(), N);
  }
  EXPECT_EQ(alloc->allocCalls.load(), 1);
  EXPECT_EQ(alloc->freeCalls.load(), 1);
  EXPECT_EQ(out, in);
}

}  // namespace

#endif  // WITH_CUDA
