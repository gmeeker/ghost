#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

class ImageTest : public GhostTest {};

// ---------------------------------------------------------------------------
// 2D image allocation and round-trip
// ---------------------------------------------------------------------------

TEST_P(ImageTest, Allocate2DImage) {
  const size_t W = 4, H = 4, C = 4;
  size_t rowStride = W * C * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(static_cast<int32_t>(rowStride), 0));

  Image img(nullptr);
  try {
    img = device().allocateImage(descr);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Images not supported on " << BackendName(backend());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Image allocation failed: " << e.what();
  }
  EXPECT_NE(img.impl().get(), nullptr);
}

TEST_P(ImageTest, HostImageRoundTrip) {
  const size_t W = 4, H = 4, C = 4;
  const size_t pixelCount = W * H * C;
  size_t rowStride = W * C * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(static_cast<int32_t>(rowStride), 0));

  Image img(nullptr);
  try {
    img = device().allocateImage(descr);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Image allocation not supported";
  }

  // Fill input with known values.
  std::vector<float> input(pixelCount), output(pixelCount, -1.0f);
  for (size_t i = 0; i < pixelCount; i++) {
    input[i] = static_cast<float>(i) / static_cast<float>(pixelCount);
  }

  img.copy(stream(), input.data(), descr);
  img.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t i = 0; i < pixelCount; i++) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "pixel element " << i;
  }
}

// ---------------------------------------------------------------------------
// Buffer-backed shared image
// ---------------------------------------------------------------------------

TEST_P(ImageTest, SharedImageFromBuffer) {
  const size_t W = 4, H = 4, C = 4;
  const size_t pixelCount = W * H * C;
  const size_t dataSize = pixelCount * sizeof(float);
  size_t rowStride = W * C * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(static_cast<int32_t>(rowStride), 0));

  auto buf = device().allocateBuffer(dataSize);

  Image img(nullptr);
  try {
    img = device().sharedImage(descr, buf);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Shared images not supported on " << BackendName(backend());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Shared image failed: " << e.what();
  }

  // Write data to the buffer, then read from the image.
  std::vector<float> input(pixelCount), output(pixelCount, -1.0f);
  for (size_t i = 0; i < pixelCount; i++) {
    input[i] = static_cast<float>(i);
  }

  buf.copy(stream(), input.data(), dataSize);
  img.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t i = 0; i < pixelCount; i++) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

TEST_P(ImageTest, SharedImageFromBufferWithPool) {
  const size_t W = 4, H = 4, C = 4;
  const size_t pixelCount = W * H * C;
  const size_t dataSize = pixelCount * sizeof(float);
  size_t rowStride = W * C * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(static_cast<int32_t>(rowStride), 0));

  // Enable a memory pool so buffers may come from a heap/pool.
  try {
    device().setMemoryPoolSize(1024 * 1024);
  } catch (const std::exception&) {
    // Pool not supported on this backend — test still exercises Shared hint.
  }

  // Reset the pool on every exit path, including the GTEST_SKIP returns below.
  struct PoolGuard {
    Device& d;

    ~PoolGuard() {
      try {
        d.setMemoryPoolSize(0);
      } catch (const std::exception&) {
      }
    }
  } poolGuard{device()};

  // Default-hint buffer may be heap-allocated; sharedImage should either
  // work or throw unsupported_error (not hang).
  auto defaultBuf = device().allocateBuffer(dataSize);
  bool defaultWorks = false;
  try {
    auto img = device().sharedImage(descr, defaultBuf);
    defaultWorks = true;
  } catch (const std::exception&) {
    // Expected on Metal (heap buffers can't back textures) or backends
    // that don't support shared images at all.
  }

  // AllocHint::Shared bypasses the heap, so sharedImage must succeed
  // (on backends that support shared images at all).
  auto sharedBuf = device().allocateBuffer(dataSize, AllocHint::Shared);
  Image img(nullptr);
  try {
    img = device().sharedImage(descr, sharedBuf);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Shared images not supported on " << BackendName(backend());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Shared image failed: " << e.what();
  }

  // Verify data sharing: write to buffer, read from image.
  std::vector<float> input(pixelCount), output(pixelCount, -1.0f);
  for (size_t i = 0; i < pixelCount; i++) {
    input[i] = static_cast<float>(i);
  }
  sharedBuf.copy(stream(), input.data(), dataSize);
  img.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t i = 0; i < pixelCount; i++) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Image-to-image copy
// ---------------------------------------------------------------------------

TEST_P(ImageTest, ImageToImageCopy) {
  const size_t W = 4, H = 4, C = 4;
  const size_t pixelCount = W * H * C;
  size_t rowStride = W * C * sizeof(float);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(static_cast<int32_t>(rowStride), 0));

  Image src(nullptr), dst(nullptr);
  try {
    src = device().allocateImage(descr);
    dst = device().allocateImage(descr);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Image allocation not supported";
  }

  std::vector<float> input(pixelCount), output(pixelCount, -1.0f);
  for (size_t i = 0; i < pixelCount; i++) {
    input[i] = static_cast<float>(i);
  }

  src.copy(stream(), input.data(), descr);
  dst.copy(stream(), src);
  dst.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t i = 0; i < pixelCount; i++) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// UInt8 image format
// ---------------------------------------------------------------------------

TEST_P(ImageTest, UInt8ImageRoundTrip) {
  const size_t W = 8, H = 8, C = 4;
  const size_t pixelCount = W * H * C;
  size_t rowStride = W * C * sizeof(uint8_t);
  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_UInt8,
                         Stride2(static_cast<int32_t>(rowStride), 0));

  Image img(nullptr);
  try {
    img = device().allocateImage(descr);
  } catch (const std::exception&) {
    GTEST_SKIP() << "UInt8 image allocation not supported";
  }

  std::vector<uint8_t> input(pixelCount), output(pixelCount, 0);
  for (size_t i = 0; i < pixelCount; i++) {
    input[i] = static_cast<uint8_t>(i & 0xFF);
  }

  img.copy(stream(), input.data(), descr);
  img.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t i = 0; i < pixelCount; i++) {
    EXPECT_EQ(output[i], input[i]) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Image alignment query sanity check
// ---------------------------------------------------------------------------

TEST_P(ImageTest, ImageAlignmentIsReasonable) {
  auto attr = device().getAttribute(kDeviceMaxImageAlignment);
  auto alignment = attr.asUInt64();
  if (alignment == 0) {
    GTEST_SKIP() << "device does not support 2D images created from a buffer";
  }
  EXPECT_EQ(alignment & (alignment - 1), 0u)
      << "alignment " << alignment << " is not a power of two";
  // All known backends return at least 16 bytes.
  EXPECT_GE(alignment, 16u) << "alignment " << alignment
                            << " seems too small (expected bytes, not pixels?)";
}

// ---------------------------------------------------------------------------
// Format-aware imageAlignment()
// ---------------------------------------------------------------------------

TEST_P(ImageTest, ImageAlignmentFormatAware) {
  ImageDescription rgba_f32(Size3(16, 16, 1), PixelOrder_RGBA, DataType_Float,
                            Stride2(0, 0));
  auto align_rgba = device().imageAlignment(rgba_f32);
  EXPECT_GE(align_rgba, 1u);
  EXPECT_EQ(align_rgba & (align_rgba - 1), 0u)
      << "alignment " << align_rgba << " is not a power of two";

  ImageDescription r_f32(Size3(16, 16, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(0, 0));
  r_f32.channels = 1;
  auto align_r = device().imageAlignment(r_f32);
  EXPECT_GE(align_r, 1u);
  EXPECT_EQ(align_r & (align_r - 1), 0u)
      << "alignment " << align_r << " is not a power of two";

  // Single-channel alignment should be <= RGBA alignment (or equal if the
  // backend doesn't distinguish).
  EXPECT_LE(align_r, align_rgba);
}

// ---------------------------------------------------------------------------
// Shared image from buffer with odd (unaligned) width
// ---------------------------------------------------------------------------

TEST_P(ImageTest, SharedImageFromBufferOddWidth) {
  // Width of 3 RGBA Float pixels = 48 bytes per row, which is unlikely to
  // satisfy any alignment > 16 without explicit padding.
  const size_t W = 3, H = 4, C = 4;
  const size_t pixelSize = C * sizeof(float);
  const size_t tightRowBytes = W * pixelSize;  // 48

  // Use format-aware alignment for the actual pixel format.
  ImageDescription probe(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(0, 0));
  auto alignment = device().imageAlignment(probe);
  if (alignment == 0) {
    GTEST_SKIP() << "Backend reports zero image alignment";
  }

  // Align row stride up to the device's required alignment.
  const size_t alignedRowBytes =
      (tightRowBytes + alignment - 1) & ~(alignment - 1);
  const size_t dataSize = alignedRowBytes * H;

  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
                         Stride2(static_cast<int32_t>(alignedRowBytes), 0));

  // Use AllocHint::Shared so Metal bypasses the heap (required for
  // buffer-backed textures).
  auto buf = device().allocateBuffer(dataSize, AllocHint::Shared);

  Image img(nullptr);
  try {
    img = device().sharedImage(descr, buf);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Shared images not supported on " << BackendName(backend());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Shared image failed: " << e.what();
  }

  // Write padded rows into the buffer, read back via the image.
  std::vector<uint8_t> input(dataSize, 0);
  for (size_t y = 0; y < H; y++) {
    auto* row = reinterpret_cast<float*>(input.data() + y * alignedRowBytes);
    for (size_t x = 0; x < W * C; x++) {
      row[x] = static_cast<float>(y * W * C + x);
    }
  }

  buf.copy(stream(), input.data(), dataSize);

  // Read back with the same padded layout (output buffer must match stride).
  std::vector<uint8_t> output(dataSize, 0);
  img.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t y = 0; y < H; y++) {
    auto* expectedRow =
        reinterpret_cast<float*>(input.data() + y * alignedRowBytes);
    auto* outputRow =
        reinterpret_cast<float*>(output.data() + y * alignedRowBytes);
    for (size_t x = 0; x < W * C; x++) {
      EXPECT_FLOAT_EQ(outputRow[x], expectedRow[x]) << "y=" << y << " x=" << x;
    }
  }
}

// ---------------------------------------------------------------------------
// Shared image from buffer with single-channel UInt8 (small pixel size)
// ---------------------------------------------------------------------------

TEST_P(ImageTest, SharedImageFromBufferOddWidthUInt8) {
  // Width of 5 RGBA UInt8 pixels = 20 bytes per row, misaligned for most
  // backends (alignment typically >= 16).
  const size_t W = 5, H = 4, C = 4;
  const size_t pixelSize = C * sizeof(uint8_t);
  const size_t tightRowBytes = W * pixelSize;  // 20

  // Use format-aware alignment for the actual pixel format.
  ImageDescription probe(Size3(W, H, 1), PixelOrder_RGBA, DataType_UInt8,
                         Stride2(0, 0));
  auto alignment = device().imageAlignment(probe);
  if (alignment == 0) {
    GTEST_SKIP() << "Backend reports zero image alignment";
  }

  const size_t alignedRowBytes =
      (tightRowBytes + alignment - 1) & ~(alignment - 1);
  const size_t dataSize = alignedRowBytes * H;

  ImageDescription descr(Size3(W, H, 1), PixelOrder_RGBA, DataType_UInt8,
                         Stride2(static_cast<int32_t>(alignedRowBytes), 0));

  // Use AllocHint::Shared so Metal bypasses the heap (required for
  // buffer-backed textures).
  auto buf = device().allocateBuffer(dataSize, AllocHint::Shared);

  Image img(nullptr);
  try {
    img = device().sharedImage(descr, buf);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Shared images not supported on " << BackendName(backend());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Shared image failed: " << e.what();
  }

  // Write padded rows into the buffer, read back via the image.
  std::vector<uint8_t> input(dataSize, 0);
  for (size_t y = 0; y < H; y++) {
    auto* row = input.data() + y * alignedRowBytes;
    for (size_t x = 0; x < W * C; x++) {
      row[x] = static_cast<uint8_t>((y * W * C + x) & 0xFF);
    }
  }

  buf.copy(stream(), input.data(), dataSize);

  std::vector<uint8_t> output(dataSize, 0);
  img.copyTo(stream(), output.data(), descr);
  stream().sync();

  for (size_t y = 0; y < H; y++) {
    auto* expectedRow = input.data() + y * alignedRowBytes;
    auto* outputRow = output.data() + y * alignedRowBytes;
    for (size_t x = 0; x < W * C; x++) {
      EXPECT_EQ(outputRow[x], expectedRow[x]) << "y=" << y << " x=" << x;
    }
  }
}

// ---------------------------------------------------------------------------
// Shared image with tall narrow shapes
// ---------------------------------------------------------------------------

TEST_P(ImageTest, SharedImageTallNarrow) {
  // Single-channel R32Float. Test multiple widths.
  struct Shape {
    size_t w, h;
  };

  Shape shapes[] = {{7, 64},    {14, 64},   {28, 64},   {56, 64},
                    {7, 256},   {28, 256},  {7, 3584},  {14, 3584},
                    {28, 3584}, {56, 3584}, {112, 3584}};

  for (const auto& s : shapes) {
    const size_t C = 1;
    const size_t pixBytes = C * sizeof(float);

    ImageDescription probe(Size3(s.w, s.h, 1), PixelOrder_RGBA, DataType_Float,
                           Stride2(0, 0));
    probe.channels = C;
    auto alignment = device().imageAlignment(probe);
    if (alignment == 0) continue;

    size_t tightRow = s.w * pixBytes;
    size_t alignedRow = (tightRow + alignment - 1) & ~(alignment - 1);
    size_t dataSize = alignedRow * s.h;

    ImageDescription descr(Size3(s.w, s.h, 1), PixelOrder_RGBA, DataType_Float,
                           Stride2(static_cast<int32_t>(alignedRow), 0));
    descr.channels = C;

    auto buf = device().allocateBuffer(dataSize, AllocHint::Shared);
    Image img(nullptr);
    try {
      img = device().sharedImage(descr, buf);
    } catch (const ghost::unsupported_error&) {
      GTEST_SKIP() << "Shared images not supported on "
                   << BackendName(backend());
    } catch (const std::exception& e) {
      GTEST_SKIP() << "Shared image failed: " << e.what();
    }
    ASSERT_TRUE(img) << "W=" << s.w << " H=" << s.h;
  }
}

GHOST_INSTANTIATE_BACKEND_TESTS(ImageTest);

// ---------------------------------------------------------------------------
// Kernel-side sampling of a buffer-backed shared image (tex2D path)
// ---------------------------------------------------------------------------
//
// The SharedImageFromBuffer* tests above validate via img.copyTo() (host
// readback). On Windows CUDA that path passed even while kernel-side tex2D
// fetches returned all-zero: the bug was in the texture object the kernel
// sampled, not in the host copy. So those tests could not catch it.
//
// This fixture dispatches an actual sampling kernel (tex2D / read_imagef /
// texture.sample) against a buffer-backed shared image and checks the values.
// It is the regression guard for Ghost commit 24557d6 ("Fix releasing of CUDA
// textures"), which fixed two real causes of the all-zero reads:
//   1. pitchInBytes came from descr.stride.x, which is 0 for tight-packed
//      images -> degenerate texture. Now resolved to width*pixelSize.
//   2. the CUtexObject was destroyed at the end of execute(), before the async
//      kernel ran. Texture objects are now cached on the image and, when the
//      image dies with work in flight, destroyed via an event-guarded reap
//      list (cuTexObjectDestroy may not run in a cuLaunchHostFunc callback).
// (The "tight packing" variant below specifically exercises cause #1; the
// "drop before sync" variant exercises cause #2's replacement machinery.)

class ImageKernelTest : public GhostKernelTest {
 protected:
  void runSampleTest(bool tightPacking, bool dropBeforeSync = false) {
    const char* src = sampleImageSource();
    if (!src)
      GTEST_SKIP() << "No image-sampling kernel for " << BackendName(backend());

    const int W = 128, H = 8;
    const size_t pixelCount = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t rowBytes = static_cast<size_t>(W) * sizeof(float);
    const size_t dataSize = pixelCount * sizeof(float);

    // Single-channel (R) float image. stride.x == 0 means tight packing, which
    // here equals W*sizeof(float) == rowBytes.
    ImageDescription descr(
        Size3(W, H, 1), PixelOrder_RGBA, DataType_Float,
        Stride2(tightPacking ? 0 : static_cast<int32_t>(rowBytes), 0));
    descr.channels = 1;

    auto buf = device().allocateBuffer(dataSize);

    Image img(nullptr);
    try {
      img = device().sharedImage(descr, buf);
    } catch (const ghost::unsupported_error&) {
      GTEST_SKIP() << "Shared images not supported on "
                   << BackendName(backend());
    } catch (const std::exception& e) {
      GTEST_SKIP() << "Shared image failed: " << e.what();
    }

    std::vector<float> input(pixelCount);
    for (size_t i = 0; i < pixelCount; i++)
      input[i] = static_cast<float>(i + 1);  // all non-zero
    std::vector<float> output(pixelCount, -1.0f);

    buf.copy(stream(), input.data(), dataSize);

    Library lib(nullptr);
    Function fn(nullptr);
    try {
      lib = device().loadLibraryFromText(src);
      fn = lib.lookupFunction("sample_image");
    } catch (const std::exception& e) {
      GTEST_SKIP() << "Kernel compile/lookup failed: " << e.what();
    }

    auto outBuf = device().allocateBuffer(dataSize);

    LaunchArgs la;
    la.global_size(static_cast<size_t>(W), static_cast<size_t>(H))
        .local_size(16, 8);
    fn(la, stream())(outBuf, img, static_cast<int32_t>(W),
                     static_cast<int32_t>(H));
    if (dropBeforeSync) {
      // Drop the image (and donor buffer) wrappers with the kernel possibly
      // still in flight: the backend must keep the sampled memory AND its
      // texture state alive until the kernel completes.
      img = Image(nullptr);
      buf = Buffer(nullptr);
    }
    outBuf.copyTo(stream(), output.data(), dataSize);
    stream().sync();

    float maxVal = 0.0f;
    for (float v : output)
      if (v > maxVal) maxVal = v;
    EXPECT_GT(maxVal, 0.0f) << "kernel read all-zero from a buffer-backed "
                               "shared image (the Windows-CUDA tex2D bug)";
    for (size_t i = 0; i < pixelCount; i++)
      EXPECT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
};

TEST_P(ImageKernelTest, SharedImageKernelSampleReadsBufferValues) {
  runSampleTest(/*tightPacking=*/false);
}

TEST_P(ImageKernelTest, SharedImageKernelSampleTightPacking) {
  runSampleTest(/*tightPacking=*/true);
}

TEST_P(ImageKernelTest, SharedImageKernelSampleDropBeforeSync) {
  runSampleTest(/*tightPacking=*/false, /*dropBeforeSync=*/true);
}

GHOST_INSTANTIATE_KERNEL_TESTS(ImageKernelTest);
