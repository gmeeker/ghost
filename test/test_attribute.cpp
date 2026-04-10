#include <ghost/attribute.h>
#include <ghost/function.h>
#include <gtest/gtest.h>

using namespace ghost;

// ---------------------------------------------------------------------------
// Attribute type constructors and accessors
// ---------------------------------------------------------------------------

TEST(AttributeTest, DefaultIsUnknown) {
  Attribute a;
  EXPECT_FALSE(a.valid());
  EXPECT_EQ(a.type(), Attribute::Type_Unknown);
}

TEST(AttributeTest, StringFromCString) {
  Attribute a("hello");
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(a.type(), Attribute::Type_String);
  EXPECT_EQ(a.asString(), "hello");
  EXPECT_EQ(a.count(), 1u);
}

TEST(AttributeTest, StringFromStdString) {
  std::string s = "world";
  Attribute a(s);
  EXPECT_EQ(a.type(), Attribute::Type_String);
  EXPECT_EQ(a.asString(), "world");
}

TEST(AttributeTest, FloatScalar) {
  Attribute a(3.14f);
  EXPECT_EQ(a.type(), Attribute::Type_Float);
  EXPECT_EQ(a.count(), 1u);
  EXPECT_FLOAT_EQ(a.asFloat(), 3.14f);
  EXPECT_NEAR(a.asDouble(), 3.14, 0.001);
}

TEST(AttributeTest, DoubleScalar) {
  Attribute a(2.718);
  EXPECT_EQ(a.type(), Attribute::Type_Float);
  EXPECT_EQ(a.count(), 1u);
  EXPECT_NEAR(a.asDouble(), 2.718, 1e-6);
}

TEST(AttributeTest, Int32Scalar) {
  Attribute a(int32_t(42));
  EXPECT_EQ(a.type(), Attribute::Type_Int);
  EXPECT_EQ(a.count(), 1u);
  EXPECT_EQ(a.asInt(), 42);
  EXPECT_EQ(a.asInt64(), 42);
}

TEST(AttributeTest, UInt32Scalar) {
  Attribute a(uint32_t(100));
  EXPECT_EQ(a.type(), Attribute::Type_UInt);
  EXPECT_EQ(a.asUInt(), 100u);
  EXPECT_EQ(a.asUInt64(), 100u);
}

TEST(AttributeTest, Int64Scalar) {
  Attribute a(int64_t(123456789LL));
  EXPECT_EQ(a.type(), Attribute::Type_Int);
  EXPECT_EQ(a.asInt64(), 123456789LL);
}

TEST(AttributeTest, UInt64Scalar) {
  Attribute a(uint64_t(9876543210ULL));
  EXPECT_EQ(a.type(), Attribute::Type_UInt);
  EXPECT_EQ(a.asUInt64(), 9876543210ULL);
}

TEST(AttributeTest, BoolScalar) {
  Attribute a(true);
  EXPECT_EQ(a.type(), Attribute::Type_Bool);
  EXPECT_EQ(a.count(), 1u);
  EXPECT_TRUE(a.asBool());

  Attribute b(false);
  EXPECT_FALSE(b.asBool());
}

TEST(AttributeTest, FloatVector2) {
  Attribute a(1.0f, 2.0f);
  EXPECT_EQ(a.type(), Attribute::Type_Float);
  EXPECT_EQ(a.count(), 2u);
  EXPECT_FLOAT_EQ(a.floatArray()[0], 1.0f);
  EXPECT_FLOAT_EQ(a.floatArray()[1], 2.0f);
}

TEST(AttributeTest, FloatVector3) {
  Attribute a(1.0f, 2.0f, 3.0f);
  EXPECT_EQ(a.count(), 3u);
  EXPECT_FLOAT_EQ(a.floatArray()[2], 3.0f);
}

TEST(AttributeTest, FloatVector4) {
  Attribute a(1.0f, 2.0f, 3.0f, 4.0f);
  EXPECT_EQ(a.count(), 4u);
  EXPECT_FLOAT_EQ(a.floatArray()[3], 4.0f);
}

TEST(AttributeTest, IntVector3) {
  Attribute a(int32_t(10), int32_t(20), int32_t(30));
  EXPECT_EQ(a.type(), Attribute::Type_Int);
  EXPECT_EQ(a.count(), 3u);
  EXPECT_EQ(a.intArray()[0], 10);
  EXPECT_EQ(a.intArray()[1], 20);
  EXPECT_EQ(a.intArray()[2], 30);
}

TEST(AttributeTest, FloatFromArray) {
  float v[] = {10.0f, 20.0f, 30.0f};
  Attribute a(v, 3);
  EXPECT_EQ(a.type(), Attribute::Type_Float);
  EXPECT_EQ(a.count(), 3u);
  EXPECT_FLOAT_EQ(a.floatArray()[0], 10.0f);
  EXPECT_FLOAT_EQ(a.floatArray()[1], 20.0f);
  EXPECT_FLOAT_EQ(a.floatArray()[2], 30.0f);
}

TEST(AttributeTest, LocalMem) {
  Attribute a;
  a.localMem(1024);
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(a.type(), Attribute::Type_LocalMem);
  EXPECT_EQ(a.asUInt(), 1024u);
}

TEST(AttributeTest, BufferPointer) {
  // We can't easily create a real Buffer without a device, but we can test
  // the Attribute type tagging with a null cast.
  Buffer* bp = nullptr;
  Attribute a(bp);
  EXPECT_EQ(a.type(), Attribute::Type_Buffer);
  EXPECT_EQ(a.asBuffer(), nullptr);
}

TEST(AttributeTest, ImagePointer) {
  Image* ip = nullptr;
  Attribute a(ip);
  EXPECT_EQ(a.type(), Attribute::Type_Image);
  EXPECT_EQ(a.asImage(), nullptr);
}

// ---------------------------------------------------------------------------
// LaunchArgs
// ---------------------------------------------------------------------------

TEST(LaunchArgsTest, DefaultState) {
  LaunchArgs la;
  EXPECT_EQ(la.dims(), 0u);
  EXPECT_FALSE(la.is_local_defined());
  EXPECT_FALSE(la.is_cooperative());
  EXPECT_EQ(la.global_size()[0], 1u);
  EXPECT_EQ(la.global_size()[1], 1u);
  EXPECT_EQ(la.global_size()[2], 1u);
  EXPECT_EQ(la.local_size()[0], 1u);
}

TEST(LaunchArgsTest, GlobalSize1D) {
  LaunchArgs la;
  la.global_size(256);
  EXPECT_EQ(la.dims(), 1u);
  EXPECT_EQ(la.global_size()[0], 256u);
  EXPECT_EQ(la.global_size()[1], 1u);
  EXPECT_EQ(la.global_size()[2], 1u);
}

TEST(LaunchArgsTest, GlobalSize2D) {
  LaunchArgs la;
  la.global_size(512, 512);
  EXPECT_EQ(la.dims(), 2u);
  EXPECT_EQ(la.global_size()[0], 512u);
  EXPECT_EQ(la.global_size()[1], 512u);
}

TEST(LaunchArgsTest, GlobalSize3D) {
  LaunchArgs la;
  la.global_size(64, 64, 64);
  EXPECT_EQ(la.dims(), 3u);
  EXPECT_EQ(la.global_size()[0], 64u);
  EXPECT_EQ(la.global_size()[1], 64u);
  EXPECT_EQ(la.global_size()[2], 64u);
}

TEST(LaunchArgsTest, LocalSize) {
  LaunchArgs la;
  la.global_size(256).local_size(64);
  EXPECT_TRUE(la.is_local_defined());
  EXPECT_EQ(la.local_size()[0], 64u);
}

TEST(LaunchArgsTest, FluentChaining) {
  LaunchArgs la;
  auto& ref = la.global_size(1024).local_size(256);
  EXPECT_EQ(&ref, &la);
  EXPECT_EQ(la.global_size()[0], 1024u);
  EXPECT_EQ(la.local_size()[0], 256u);
}

TEST(LaunchArgsTest, Count1D) {
  LaunchArgs la;
  la.global_size(100).local_size(32);
  // ceil(100/32) = 4
  EXPECT_EQ(la.count(0), 4u);
  EXPECT_EQ(la.count(), 4u);
}

TEST(LaunchArgsTest, Count2D) {
  LaunchArgs la;
  la.global_size(100, 200).local_size(32, 16);
  // ceil(100/32) * ceil(200/16) = 4 * 13 = 52
  EXPECT_EQ(la.count(0), 4u);
  EXPECT_EQ(la.count(1), 13u);
  EXPECT_EQ(la.count(), 52u);
}

TEST(LaunchArgsTest, Cooperative) {
  LaunchArgs la;
  EXPECT_FALSE(la.is_cooperative());
  la.cooperative();
  EXPECT_TRUE(la.is_cooperative());
  la.cooperative(false);
  EXPECT_FALSE(la.is_cooperative());
}

TEST(LaunchArgsTest, GlobalSize64Bit) {
  // Element counts beyond UINT32_MAX must round-trip through LaunchArgs
  // intact; backends are responsible for narrowing at dispatch time.
  LaunchArgs la;
  const size_t big = (size_t(1) << 33) + 7;
  la.global_size(big);
  EXPECT_EQ(la.global_size()[0], big);
}

TEST(LaunchArgsTest, RequiredSubgroupSizeDefault) {
  LaunchArgs la;
  EXPECT_EQ(la.requiredSubgroupSize(), 0u);
  la.requireSubgroupSize(32);
  EXPECT_EQ(la.requiredSubgroupSize(), 32u);
  la.requireSubgroupSize(0);
  EXPECT_EQ(la.requiredSubgroupSize(), 0u);
}
