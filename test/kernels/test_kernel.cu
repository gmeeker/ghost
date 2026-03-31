// CUDA kernel source for Ghost test suite.
// Compiled to PTX at build time via nvcc -ptx.

extern "C" __global__ void mult_const_f(float* out, const float* A,
                                        float scale) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  out[tid] = A[tid] * scale;
}

extern "C" __global__ void add_buffers(float* out, const float* A,
                                       const float* B) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  out[tid] = A[tid] + B[tid];
}
