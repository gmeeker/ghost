// OpenCL kernel source for Ghost test suite.
// Can be compiled to SPIR-V at build time via clang + llvm-spirv.

__kernel void mult_const_f(__global float* out, const __global float* A,
                           float scale) {
  int tid = get_global_id(0);
  out[tid] = A[tid] * scale;
}

__kernel void add_buffers(__global float* out, const __global float* A,
                          const __global float* B) {
  int tid = get_global_id(0);
  out[tid] = A[tid] + B[tid];
}
