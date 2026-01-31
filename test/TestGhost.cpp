#include <ghost/cuda/device.h>
#include <ghost/device.h>
#include <ghost/metal/device.h>
#include <ghost/opencl/device.h>

using namespace ghost;

const float test_input[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
const float test_output[] = {
    0.f,   1.5f,  3.f,   4.5f,  6.f,   7.5f,  9.f,   10.5f, 12.f,  13.5f, 15.f,
    16.5f, 18.f,  19.5f, 21.f,  22.5f, 24.f,  25.5f, 27.f,  28.5f, 30.f,  31.5f,
    33.f,  34.5f, 36.f,  37.5f, 39.f,  40.5f, 42.f,  43.5f, 45.f,  46.5f};

#if WITH_CUDA
void testCUDA() {
  static const char* source =
      "\n\
";
  DeviceCUDA dev;
  auto program = dev.loadLibraryFromText(source);
  auto kernel = program.lookupFunction("kmain");
  auto input = dev.allocateBuffer(32 * sizeof(float));
  auto output = dev.allocateBuffer(32 * sizeof(float));
  LaunchArgs launch;
  launch.global_size(32).local_size(1);
  kernel(dev.defaultStream(), launch, input, output, 1.5f);
  dev.defaultStream().sync();
}
#endif

#if WITH_METAL
void testMetal() {
  static const char* source =
      "\n\
#include <metal_stdlib>\n\
using namespace metal;\n\
\n\
kernel void mult_const_f(device float* out [[buffer(0)]],\n\
                         device const float* A [[buffer(1)]],\n\
                         uint index [[thread_position_in_grid]],\n\
                         constant float& scale [[buffer(2)]]) {\n\
    out[index] = A[index] * scale;\n\
}";
  DeviceMetal dev;
  auto program = dev.loadLibraryFromText(source);
  auto kernel = program.lookupFunction("mult_const_f");
  auto input = dev.allocateBuffer(32 * sizeof(float));
  auto output = dev.allocateBuffer(32 * sizeof(float));
  LaunchArgs launch;
  launch.global_size(32).local_size(1);
  kernel(dev.defaultStream(), launch, input, output, 1.5f);
  dev.defaultStream().sync();
}
#endif

#if WITH_OPENCL
void testOpenCL() {
  static const char* source =
      "\n\
__kernel void mult_const_f(__global float *out, __global const float *A, float scale) {\n\
    int tid = get_global_id(0);\n\
\n\
    out[tid] = A[tid] * scale;\n\
}";
  DeviceOpenCL dev;
  auto program = dev.loadLibraryFromText(source);
  auto kernel = program.lookupFunction("mult_const_f");
  auto input = dev.allocateBuffer(32 * sizeof(float));
  auto output = dev.allocateBuffer(32 * sizeof(float));
  LaunchArgs launch;
  launch.global_size(32).local_size(1);
  kernel(dev.defaultStream(), launch, input, output, 1.5f);
  dev.defaultStream().sync();
}
#endif

int main(int argc, const char** argv) {
#if WITH_CUDA
  DeviceCUDA cuda;
#endif
#if WITH_METAL
  testMetal();
#endif
#if WITH_OPENCL
  testOpenCL();
#endif

  return 0;
}
