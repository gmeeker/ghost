// HLSL compute shader for Ghost test suite.
// Compiled to SPIR-V (Vulkan) via dxc -spirv, or to DXIL (DirectX) via dxc.
//
// The [[vk::binding(N)]] annotations are ignored when compiling to DXIL
// (DirectX uses register(u#/t#/b#)), and override SPIR-V binding numbers
// when compiling via dxc -spirv. This avoids dxc's default behavior of
// numbering t# and u# from the same base, which would cause a collision.
// The chosen scheme — UAV at binding 0, SRV at binding 1, CBV at binding 2 —
// matches the (u, t, b) sort order Ghost uses for positional argument
// matching, so the same C++ call site works for both backends.

[[vk::binding(0)]] RWStructuredBuffer<float> out_buf : register(u0);
[[vk::binding(1)]] StructuredBuffer<float>  A_buf  : register(t0);

[[vk::binding(2)]] cbuffer Params : register(b0) {
    float scale;
};

[numthreads(64, 1, 1)]
void mult_const_f(uint3 tid : SV_DispatchThreadID) {
    out_buf[tid.x] = A_buf[tid.x] * scale;
}
