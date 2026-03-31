// HLSL compute shader for Ghost test suite.
// Compiled to SPIR-V (Vulkan) via dxc -spirv, or to DXIL (DirectX) via dxc.

RWStructuredBuffer<float> out_buf : register(u0);
StructuredBuffer<float> A_buf : register(t0);

cbuffer Params : register(b0) {
    float scale;
};

[numthreads(64, 1, 1)]
void mult_const_f(uint3 tid : SV_DispatchThreadID) {
    out_buf[tid.x] = A_buf[tid.x] * scale;
}
