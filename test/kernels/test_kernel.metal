// Metal kernel source for Ghost test suite.
// Compiled to metallib at build time via xcrun metal + xcrun metallib.

#include <metal_stdlib>
using namespace metal;

kernel void mult_const_f(device float* out [[buffer(0)]],
                         const device float* A [[buffer(1)]],
                         uint index [[thread_position_in_grid]],
                         constant float& scale [[buffer(2)]]) {
  out[index] = A[index] * scale;
}

kernel void add_buffers(device float* out [[buffer(0)]],
                        const device float* A [[buffer(1)]],
                        const device float* B [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  out[index] = A[index] + B[index];
}
