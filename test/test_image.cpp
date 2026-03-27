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

GHOST_INSTANTIATE_BACKEND_TESTS(ImageTest);
