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

#include <ghost/binary_cache.h>
#include <ghost/implementation/impl_device.h>
#include <ghost/implementation/impl_function.h>

#include <filesystem>
#include <string>
#include <vector>

#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// On-disk BinaryCache tests
//
// These exercise BinaryCache::saveBinaries() / loadBinaries() — the disk
// cache used by every binary backend (CUDA cubin, OpenCL device binary,
// Vulkan SPIR-V, DirectX DXIL). Metal does not use this path; it serializes
// an MTLBinaryArchive into cachePath instead (covered separately below).
//
// Regression coverage for two historical bugs in binary_cache.cpp:
//   * saveBinaries hashed/wrote &binaries[i] (address of an 8-byte pointer
//     slot) for sizes[i] bytes -> out-of-bounds read -> access violation /
//     corrupt cache file.
//   * loadBinaries never stored the per-binary size it read back, so every
//     entry stayed size 0, the digest check always failed, and the cache
//     could never be reused (silent recompile every launch).
// ---------------------------------------------------------------------------

namespace {

// Unique temp directory per test instance that cleans itself up.
class ScopedCacheDir {
 public:
  explicit ScopedCacheDir(const void* salt) {
    path_ = fs::temp_directory_path() /
            ("ghost_cache_test_" +
             std::to_string(reinterpret_cast<uintptr_t>(salt)));
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  ~ScopedCacheDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  const fs::path& path() const { return path_; }

  size_t fileCount() const {
    std::error_code ec;
    size_t n = 0;
    if (!fs::is_directory(path_, ec)) return 0;
    for (const auto& e : fs::directory_iterator(path_, ec)) {
      (void)e;
      n++;
    }
    return n;
  }

 private:
  fs::path path_;
};

// RAII guard so a failed assertion can't leave a device's cache path pointing
// at a temp dir we are about to delete.
class ScopedCachePath {
 public:
  ScopedCachePath(BinaryCache& cache, fs::path p) : cache_(cache) {
    cache_.cachePath = std::move(p);
  }

  ~ScopedCachePath() { cache_.cachePath.clear(); }

 private:
  BinaryCache& cache_;
};

}  // namespace

class BinaryCacheTest : public GhostTest {};

// Direct save -> load round-trip on synthetic blobs. Deterministic on every
// backend (it only needs the device for the digest key), so it reliably
// catches both the out-of-bounds save and the dropped-size load regressions.
TEST_P(BinaryCacheTest, DiskRoundTrip) {
  auto& dev = *device().impl();
  int count = dev.getAttribute(kDeviceCount).asInt();
  // A sane device count is small; reject bogus values so the blob allocation
  // below can never run away (an unimplemented kDeviceCount must not spin).
  if (count < 1 || count > 64)
    GTEST_SKIP() << "device reports invalid kDeviceCount: " << count;

  ScopedCacheDir dir(this);
  auto& cache = device().binaryCache();
  ScopedCachePath guard(cache, dir.path());

  // saveBinaries writes sizes.size() as the entry count and loadBinaries
  // requires that to equal kDeviceCount, so produce exactly `count` blobs.
  std::vector<std::vector<unsigned char>> blobs(count);
  std::vector<unsigned char*> ptrs(count);
  std::vector<size_t> sizes(count);
  for (int i = 0; i < count; i++) {
    size_t len = 1024u + i * 257u + 3u;  // distinct, non-pointer-sized
    blobs[i].resize(len);
    for (size_t j = 0; j < len; j++)
      blobs[i][j] = static_cast<unsigned char>((j * 31u + i * 7u + 1u) & 0xFF);
    ptrs[i] = blobs[i].data();
    sizes[i] = len;
  }

  const char key[] = "synthetic-binary-cache-key";
  CompilerOptions options;

  cache.saveBinaries(dev, ptrs, sizes, key, sizeof(key), options);
  EXPECT_EQ(dir.fileCount(), 1u) << "save should create exactly one cache file";

  std::vector<std::vector<unsigned char>> outBlobs;
  std::vector<size_t> outSizes;
  bool ok =
      cache.loadBinaries(outBlobs, outSizes, dev, key, sizeof(key), options);
  ASSERT_TRUE(ok) << "round-trip load must succeed";
  ASSERT_EQ(outBlobs.size(), static_cast<size_t>(count));
  ASSERT_EQ(outSizes.size(), static_cast<size_t>(count));
  for (int i = 0; i < count; i++) {
    EXPECT_EQ(outSizes[i], sizes[i]) << "size mismatch for blob " << i;
    EXPECT_EQ(outBlobs[i], blobs[i]) << "content mismatch for blob " << i;
  }
}

// A different key must miss (and not crash / not return stale data).
TEST_P(BinaryCacheTest, LoadDifferentKeyMisses) {
  auto& dev = *device().impl();
  int count = dev.getAttribute(kDeviceCount).asInt();
  if (count < 1 || count > 64) GTEST_SKIP();

  ScopedCacheDir dir(this);
  auto& cache = device().binaryCache();
  ScopedCachePath guard(cache, dir.path());

  std::vector<std::vector<unsigned char>> blobs(count);
  std::vector<unsigned char*> ptrs(count);
  std::vector<size_t> sizes(count);
  for (int i = 0; i < count; i++) {
    blobs[i].assign(512u + i, static_cast<unsigned char>(0xA5 ^ i));
    ptrs[i] = blobs[i].data();
    sizes[i] = blobs[i].size();
  }
  const char keyA[] = "key-A";
  const char keyB[] = "key-B-different";
  CompilerOptions options;
  cache.saveBinaries(dev, ptrs, sizes, keyA, sizeof(keyA), options);

  std::vector<std::vector<unsigned char>> outBlobs;
  std::vector<size_t> outSizes;
  EXPECT_FALSE(
      cache.loadBinaries(outBlobs, outSizes, dev, keyB, sizeof(keyB), options));
}

// Loading from an empty cache returns false rather than crashing.
TEST_P(BinaryCacheTest, LoadMissOnEmptyCache) {
  auto& dev = *device().impl();
  ScopedCacheDir dir(this);
  auto& cache = device().binaryCache();
  ScopedCachePath guard(cache, dir.path());

  const char key[] = "never-saved";
  std::vector<std::vector<unsigned char>> outBlobs;
  std::vector<size_t> outSizes;
  EXPECT_FALSE(cache.loadBinaries(outBlobs, outSizes, dev, key, sizeof(key),
                                  CompilerOptions()));
}

GHOST_INSTANTIATE_BACKEND_TESTS(BinaryCacheTest);

// ---------------------------------------------------------------------------
// End-to-end: compiling with the cache enabled creates files and the second
// compile produces a working kernel. Covers the full wiring including Metal's
// MTLBinaryArchive path (which writes a .metalarchive into cachePath).
// ---------------------------------------------------------------------------

class BinaryCacheKernelTest : public GhostKernelTest {};

TEST_P(BinaryCacheKernelTest, CompilePopulatesCacheAndReuses) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP() << "No kernel source for " << BackendName(backend());

  ScopedCacheDir dir(this);
  auto& cache = device().binaryCache();
  ScopedCachePath guard(cache, dir.path());
  ASSERT_TRUE(cache.isEnabled());

  // First compile: should write a cache entry to disk.
  {
    auto lib = device().loadLibraryFromText(src);
    auto fn = lib.lookupFunction("mult_const_f");
    EXPECT_NE(fn.impl().get(), nullptr);
  }
  EXPECT_GT(dir.fileCount(), 0u)
      << "first compile should populate the on-disk cache";

  // Second compile of the same source hits the cache path; verify the kernel
  // it produces actually runs correctly (i.e. the cached binary is valid).
  auto lib2 = device().loadLibraryFromText(src);
  auto fn2 = lib2.lookupFunction("mult_const_f");
  ASSERT_NE(fn2.impl().get(), nullptr);

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);
  fn2(la, stream())(outBuf, inBuf, 3.0f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 3.0f) << "index " << i;

  device().purgeBinaries(0);
}

GHOST_INSTANTIATE_KERNEL_TESTS(BinaryCacheKernelTest);
