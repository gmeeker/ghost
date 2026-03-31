// CPU kernel shared library for Ghost test suite.
// These functions match the FunctionCPU::Type signature:
//   void (*)(size_t i, size_t n, const std::vector<Attribute>& args)
//
// Buffer data is accessed via args[i].asBuffer()->impl() cast to BufferCPU*.

#include <ghost/attribute.h>
#include <ghost/cpu/impl_device.h>
#include <ghost/device.h>

using namespace ghost;
using namespace ghost::implementation;

#if defined(_WIN32)
#define GHOST_EXPORT __declspec(dllexport)
#else
#define GHOST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// mult_const_f: out[i] = A[i] * scale
// args[0] = out buffer, args[1] = A buffer, args[2] = scale (float)
GHOST_EXPORT void mult_const_f(size_t i, size_t n,
                               const std::vector<Attribute>& args) {
  auto* out = static_cast<float*>(
      static_cast<BufferCPU*>(args[0].asBuffer()->impl().get())->ptr);
  auto* A = static_cast<const float*>(
      static_cast<BufferCPU*>(args[1].asBuffer()->impl().get())->ptr);
  float scale = args[2].asFloat();
  out[i] = A[i] * scale;
}

// add_buffers: out[i] = A[i] + B[i]
// args[0] = out buffer, args[1] = A buffer, args[2] = B buffer
GHOST_EXPORT void add_buffers(size_t i, size_t n,
                              const std::vector<Attribute>& args) {
  auto* out = static_cast<float*>(
      static_cast<BufferCPU*>(args[0].asBuffer()->impl().get())->ptr);
  auto* A = static_cast<const float*>(
      static_cast<BufferCPU*>(args[1].asBuffer()->impl().get())->ptr);
  auto* B = static_cast<const float*>(
      static_cast<BufferCPU*>(args[2].asBuffer()->impl().get())->ptr);
  out[i] = A[i] + B[i];
}

}  // extern "C"
