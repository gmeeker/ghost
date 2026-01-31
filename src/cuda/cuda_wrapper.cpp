// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

/*
 * This wrapper only current up to CUDA 8.0, but can be easily upgraded
 * when newer functions are required.
 */

#if WITH_CUDA && !WITH_CUDA_LINK

#include <cuda.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <vector>

// Expand CUDA's #define functions.
#define EXPAND_DEFINE(a) #a
#define INIT_FUNCTION(a) m_##a = (f_##a)getCUDAFunction(EXPAND_DEFINE(a))

namespace {
#if defined(_WIN32)
static const std::vector<const TCHAR*> default_so_paths = {
    __TEXT("nvcuda.dll")};
#elif defined(__linux__)
static const std::vector<const char*> default_so_paths = {"libcuda.so"};
#endif

// Function pointers declaration
using f_cuGetErrorString = CUresult(CUDAAPI*)(CUresult error,
                                              const char** pStr);
using f_cuGetErrorName = CUresult(CUDAAPI*)(CUresult error, const char** pStr);
using f_cuInit = CUresult(CUDAAPI*)(unsigned int Flags);
using f_cuDriverGetVersion = CUresult(CUDAAPI*)(int* driverVersion);
using f_cuDeviceGet = CUresult(CUDAAPI*)(CUdevice* device, int ordinal);
using f_cuDeviceGetCount = CUresult(CUDAAPI*)(int* count);
using f_cuDeviceGetName = CUresult(CUDAAPI*)(char* name, int len, CUdevice dev);
using f_cuDeviceTotalMem = CUresult(CUDAAPI*)(size_t* bytes, CUdevice dev);
using f_cuDeviceGetAttribute = CUresult(CUDAAPI*)(int* pi,
                                                  CUdevice_attribute attrib,
                                                  CUdevice dev);
using f_cuDeviceGetProperties = CUresult(CUDAAPI*)(CUdevprop* prop,
                                                   CUdevice dev);
using f_cuDeviceComputeCapability = CUresult(CUDAAPI*)(int* major, int* minor,
                                                       CUdevice dev);
#if CUDA_VERSION >= 13000
using f_cuCtxCreate = CUresult(CUDAAPI*)(CUcontext* pctx,
                                         CUctxCreateParams* params,
                                         unsigned int flags, CUdevice dev);
#else
using f_cuCtxCreate = CUresult(CUDAAPI*)(CUcontext* pctx, unsigned int flags,
                                         CUdevice dev);
#endif
using f_cuCtxDestroy = CUresult(CUDAAPI*)(CUcontext ctx);
using f_cuCtxPushCurrent = CUresult(CUDAAPI*)(CUcontext ctx);
using f_cuCtxPopCurrent = CUresult(CUDAAPI*)(CUcontext* pctx);
using f_cuCtxSetCurrent = CUresult(CUDAAPI*)(CUcontext ctx);
using f_cuCtxGetCurrent = CUresult(CUDAAPI*)(CUcontext* pctx);
using f_cuCtxGetDevice = CUresult(CUDAAPI*)(CUdevice* device);
using f_cuCtxSynchronize = CUresult(CUDAAPI*)(void);
using f_cuCtxSetLimit = CUresult(CUDAAPI*)(CUlimit limit, size_t value);
using f_cuCtxGetLimit = CUresult(CUDAAPI*)(size_t* pvalue, CUlimit limit);
using f_cuCtxGetCacheConfig = CUresult(CUDAAPI*)(CUfunc_cache* pconfig);
using f_cuCtxSetCacheConfig = CUresult(CUDAAPI*)(CUfunc_cache config);
using f_cuCtxGetSharedMemConfig = CUresult(CUDAAPI*)(CUsharedconfig* pConfig);
using f_cuCtxSetSharedMemConfig = CUresult(CUDAAPI*)(CUsharedconfig config);
using f_cuCtxGetApiVersion = CUresult(CUDAAPI*)(CUcontext ctx,
                                                unsigned int* version);
using f_cuCtxGetStreamPriorityRange = CUresult(CUDAAPI*)(int* leastPriority,
                                                         int* greatestPriority);
using f_cuCtxAttach = CUresult(CUDAAPI*)(CUcontext* pctx, unsigned int flags);
using f_cuCtxDetach = CUresult(CUDAAPI*)(CUcontext ctx);
using f_cuModuleLoad = CUresult(CUDAAPI*)(CUmodule* module, const char* fname);
using f_cuModuleLoadData = CUresult(CUDAAPI*)(CUmodule* module,
                                              const void* image);
using f_cuModuleLoadDataEx = CUresult(CUDAAPI*)(CUmodule* module,
                                                const void* image,
                                                unsigned int numOptions,
                                                CUjit_option* options,
                                                void** optionValues);
using f_cuModuleLoadFatBinary = CUresult(CUDAAPI*)(CUmodule* module,
                                                   const void* fatCubin);
using f_cuModuleUnload = CUresult(CUDAAPI*)(CUmodule hmod);
using f_cuModuleGetFunction = CUresult(CUDAAPI*)(CUfunction* hfunc,
                                                 CUmodule hmod,
                                                 const char* name);
using f_cuModuleGetGlobal = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t* bytes,
                                               CUmodule hmod, const char* name);
using f_cuModuleGetTexRef = CUresult(CUDAAPI*)(CUtexref* pTexRef, CUmodule hmod,
                                               const char* name);
using f_cuModuleGetSurfRef = CUresult(CUDAAPI*)(CUsurfref* pSurfRef,
                                                CUmodule hmod,
                                                const char* name);
using f_cuLinkCreate = CUresult(CUDAAPI*)(unsigned int numOptions,
                                          CUjit_option* options,
                                          void** optionValues,
                                          CUlinkState* stateOut);
using f_cuLinkAddData = CUresult(CUDAAPI*)(CUlinkState state,
                                           CUjitInputType type, void* data,
                                           size_t size, const char* name,
                                           unsigned int numOptions,
                                           CUjit_option* options,
                                           void** optionValues);
using f_cuLinkAddFile = CUresult(CUDAAPI*)(
    CUlinkState state, CUjitInputType type, const char* path,
    unsigned int numOptions, CUjit_option* options, void** optionValues);
using f_cuLinkComplete = CUresult(CUDAAPI*)(CUlinkState state, void** cubinOut,
                                            size_t* sizeOut);
using f_cuLinkDestroy = CUresult(CUDAAPI*)(CUlinkState state);
using f_cuMemGetInfo = CUresult(CUDAAPI*)(size_t* free, size_t* total);
using f_cuMemAlloc = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t bytesize);
using f_cuMemAllocPitch = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t* pPitch,
                                             size_t WidthInBytes, size_t Height,
                                             unsigned int ElementSizeBytes);
using f_cuMemFree = CUresult(CUDAAPI*)(CUdeviceptr dptr);
using f_cuMemGetAddressRange = CUresult(CUDAAPI*)(CUdeviceptr* pbase,
                                                  size_t* psize,
                                                  CUdeviceptr dptr);
using f_cuMemAllocHost = CUresult(CUDAAPI*)(void** pp, size_t bytesize);
using f_cuMemFreeHost = CUresult(CUDAAPI*)(void* p);
using f_cuMemHostAlloc = CUresult(CUDAAPI*)(void** pp, size_t bytesize,
                                            unsigned int Flags);
using f_cuMemHostGetDevicePointer = CUresult(CUDAAPI*)(CUdeviceptr* pdptr,
                                                       void* p,
                                                       unsigned int Flags);
using f_cuMemHostGetFlags = CUresult(CUDAAPI*)(unsigned int* pFlags, void* p);
using f_cuMemAllocManaged = CUresult(CUDAAPI*)(CUdeviceptr* dptr,
                                               size_t bytesize,
                                               unsigned int flags);
using f_cuDeviceGetByPCIBusId = CUresult(CUDAAPI*)(CUdevice* dev,
                                                   const char* pciBusId);
using f_cuDeviceGetPCIBusId = CUresult(CUDAAPI*)(char* pciBusId, int len,
                                                 CUdevice dev);
using f_cuIpcGetEventHandle = CUresult(CUDAAPI*)(CUipcEventHandle* pHandle,
                                                 CUevent event);
using f_cuIpcOpenEventHandle = CUresult(CUDAAPI*)(CUevent* phEvent,
                                                  CUipcEventHandle handle);
using f_cuIpcGetMemHandle = CUresult(CUDAAPI*)(CUipcMemHandle* pHandle,
                                               CUdeviceptr dptr);
using f_cuIpcOpenMemHandle = CUresult(CUDAAPI*)(CUdeviceptr* pdptr,
                                                CUipcMemHandle handle,
                                                unsigned int Flags);
using f_cuIpcCloseMemHandle = CUresult(CUDAAPI*)(CUdeviceptr dptr);
using f_cuMemHostRegister = CUresult(CUDAAPI*)(void* p, size_t bytesize,
                                               unsigned int Flags);
using f_cuMemHostUnregister = CUresult(CUDAAPI*)(void* p);
using f_cuMemcpy = CUresult(CUDAAPI*)(CUdeviceptr dst, CUdeviceptr src,
                                      size_t ByteCount);
using f_cuMemcpyPeer = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                          CUcontext dstContext,
                                          CUdeviceptr srcDevice,
                                          CUcontext srcContext,
                                          size_t ByteCount);
using f_cuMemcpyHtoD = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                          const void* srcHost,
                                          size_t ByteCount);
using f_cuMemcpyDtoH = CUresult(CUDAAPI*)(void* dstHost, CUdeviceptr srcDevice,
                                          size_t ByteCount);
using f_cuMemcpyDtoD = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                          CUdeviceptr srcDevice,
                                          size_t ByteCount);
using f_cuMemcpyDtoA = CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset,
                                          CUdeviceptr srcDevice,
                                          size_t ByteCount);
using f_cuMemcpyAtoD = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                          CUarray srcArray, size_t srcOffset,
                                          size_t ByteCount);
using f_cuMemcpyHtoA = CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset,
                                          const void* srcHost,
                                          size_t ByteCount);
using f_cuMemcpyAtoH = CUresult(CUDAAPI*)(void* dstHost, CUarray srcArray,
                                          size_t srcOffset, size_t ByteCount);
using f_cuMemcpyAtoA = CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset,
                                          CUarray srcArray, size_t srcOffset,
                                          size_t ByteCount);
using f_cuMemcpy2D = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D* pCopy);
using f_cuMemcpy2DUnaligned = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D* pCopy);
using f_cuMemcpy3D = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D* pCopy);
using f_cuMemcpy3DPeer = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D_PEER* pCopy);
using f_cuMemcpyAsync = CUresult(CUDAAPI*)(CUdeviceptr dst, CUdeviceptr src,
                                           size_t ByteCount, CUstream hStream);
using f_cuMemcpyPeerAsync = CUresult(CUDAAPI*)(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
    CUcontext srcContext, size_t ByteCount, CUstream hStream);
using f_cuMemcpyHtoDAsync = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                               const void* srcHost,
                                               size_t ByteCount,
                                               CUstream hStream);
using f_cuMemcpyDtoHAsync = CUresult(CUDAAPI*)(void* dstHost,
                                               CUdeviceptr srcDevice,
                                               size_t ByteCount,
                                               CUstream hStream);
using f_cuMemcpyDtoDAsync = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                               CUdeviceptr srcDevice,
                                               size_t ByteCount,
                                               CUstream hStream);
using f_cuMemcpyHtoAAsync = CUresult(CUDAAPI*)(CUarray dstArray,
                                               size_t dstOffset,
                                               const void* srcHost,
                                               size_t ByteCount,
                                               CUstream hStream);
using f_cuMemcpyAtoHAsync = CUresult(CUDAAPI*)(void* dstHost, CUarray srcArray,
                                               size_t srcOffset,
                                               size_t ByteCount,
                                               CUstream hStream);
using f_cuMemcpy2DAsync = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D* pCopy,
                                             CUstream hStream);
using f_cuMemcpy3DAsync = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D* pCopy,
                                             CUstream hStream);
using f_cuMemcpy3DPeerAsync =
    CUresult(CUDAAPI*)(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream);
using f_cuMemsetD8 = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned char uc,
                                        size_t N);
using f_cuMemsetD16 = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                         unsigned short us, size_t N);
using f_cuMemsetD32 = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned int ui,
                                         size_t N);
using f_cuMemsetD2D8 = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                          size_t dstPitch, unsigned char uc,
                                          size_t Width, size_t Height);
using f_cuMemsetD2D16 = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                           size_t dstPitch, unsigned short us,
                                           size_t Width, size_t Height);
using f_cuMemsetD2D32 = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                           size_t dstPitch, unsigned int ui,
                                           size_t Width, size_t Height);
using f_cuMemsetD8Async = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                             unsigned char uc, size_t N,
                                             CUstream hStream);
using f_cuMemsetD16Async = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                              unsigned short us, size_t N,
                                              CUstream hStream);
using f_cuMemsetD32Async = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                              unsigned int ui, size_t N,
                                              CUstream hStream);
using f_cuMemsetD2D8Async = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                               size_t dstPitch,
                                               unsigned char uc, size_t Width,
                                               size_t Height, CUstream hStream);
using f_cuMemsetD2D16Async = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                                size_t dstPitch,
                                                unsigned short us, size_t Width,
                                                size_t Height,
                                                CUstream hStream);
using f_cuMemsetD2D32Async = CUresult(CUDAAPI*)(CUdeviceptr dstDevice,
                                                size_t dstPitch,
                                                unsigned int ui, size_t Width,
                                                size_t Height,
                                                CUstream hStream);
using f_cuArrayCreate = CUresult(CUDAAPI*)(
    CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray);
using f_cuArrayGetDescriptor =
    CUresult(CUDAAPI*)(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
using f_cuArrayDestroy = CUresult(CUDAAPI*)(CUarray hArray);
using f_cuArray3DCreate = CUresult(CUDAAPI*)(
    CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
using f_cuArray3DGetDescriptor = CUresult(CUDAAPI*)(
    CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
using f_cuMipmappedArrayCreate =
    CUresult(CUDAAPI*)(CUmipmappedArray* pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                       unsigned int numMipmapLevels);
using f_cuMipmappedArrayGetLevel = CUresult(CUDAAPI*)(
    CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
using f_cuMipmappedArrayDestroy =
    CUresult(CUDAAPI*)(CUmipmappedArray hMipmappedArray);
using f_cuPointerGetAttribute = CUresult(CUDAAPI*)(
    void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
using f_cuPointerSetAttribute = CUresult(CUDAAPI*)(
    const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);
using f_cuStreamCreate = CUresult(CUDAAPI*)(CUstream* phStream,
                                            unsigned int Flags);
using f_cuStreamCreateWithPriority = CUresult(CUDAAPI*)(CUstream* phStream,
                                                        unsigned int flags,
                                                        int priority);
using f_cuStreamGetPriority = CUresult(CUDAAPI*)(CUstream hStream,
                                                 int* priority);
using f_cuStreamGetFlags = CUresult(CUDAAPI*)(CUstream hStream,
                                              unsigned int* flags);
using f_cuStreamWaitEvent = CUresult(CUDAAPI*)(CUstream hStream, CUevent hEvent,
                                               unsigned int Flags);
using f_cuStreamAddCallback = CUresult(CUDAAPI*)(CUstream hStream,
                                                 CUstreamCallback callback,
                                                 void* userData,
                                                 unsigned int flags);
using f_cuStreamAttachMemAsync = CUresult(CUDAAPI*)(CUstream hStream,
                                                    CUdeviceptr dptr,
                                                    size_t length,
                                                    unsigned int flags);
using f_cuStreamQuery = CUresult(CUDAAPI*)(CUstream hStream);
using f_cuStreamSynchronize = CUresult(CUDAAPI*)(CUstream hStream);
using f_cuStreamDestroy = CUresult(CUDAAPI*)(CUstream hStream);
using f_cuEventCreate = CUresult(CUDAAPI*)(CUevent* phEvent,
                                           unsigned int Flags);
using f_cuEventRecord = CUresult(CUDAAPI*)(CUevent hEvent, CUstream hStream);
using f_cuEventQuery = CUresult(CUDAAPI*)(CUevent hEvent);
using f_cuEventSynchronize = CUresult(CUDAAPI*)(CUevent hEvent);
using f_cuEventDestroy = CUresult(CUDAAPI*)(CUevent hEvent);
using f_cuEventElapsedTime = CUresult(CUDAAPI*)(float* pMilliseconds,
                                                CUevent hStart, CUevent hEnd);
using f_cuFuncGetAttribute = CUresult(CUDAAPI*)(int* pi,
                                                CUfunction_attribute attrib,
                                                CUfunction hfunc);
using f_cuFuncSetAttribute = CUresult(CUDAAPI*)(CUfunction hfunc,
                                                CUfunction_attribute attrib,
                                                int value);
using f_cuFuncSetCacheConfig = CUresult(CUDAAPI*)(CUfunction hfunc,
                                                  CUfunc_cache config);
using f_cuFuncSetSharedMemConfig = CUresult(CUDAAPI*)(CUfunction hfunc,
                                                      CUsharedconfig config);
using f_cuLaunchKernel = CUresult(CUDAAPI*)(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra);
using f_cuFuncSetBlockShape = CUresult(CUDAAPI*)(CUfunction hfunc, int x, int y,
                                                 int z);
using f_cuFuncSetSharedSize = CUresult(CUDAAPI*)(CUfunction hfunc,
                                                 unsigned int bytes);
using f_cuParamSetSize = CUresult(CUDAAPI*)(CUfunction hfunc,
                                            unsigned int numbytes);
using f_cuParamSeti = CUresult(CUDAAPI*)(CUfunction hfunc, int offset,
                                         unsigned int value);
using f_cuParamSetf = CUresult(CUDAAPI*)(CUfunction hfunc, int offset,
                                         float value);
using f_cuParamSetv = CUresult(CUDAAPI*)(CUfunction hfunc, int offset,
                                         void* ptr, unsigned int numbytes);
using f_cuLaunch = CUresult(CUDAAPI*)(CUfunction f);
using f_cuLaunchGrid = CUresult(CUDAAPI*)(CUfunction f, int grid_width,
                                          int grid_height);
using f_cuLaunchGridAsync = CUresult(CUDAAPI*)(CUfunction f, int grid_width,
                                               int grid_height,
                                               CUstream hStream);
using f_cuParamSetTexRef = CUresult(CUDAAPI*)(CUfunction hfunc, int texunit,
                                              CUtexref hTexRef);
using f_cuOccupancyMaxActiveBlocksPerMultiprocessor = CUresult(CUDAAPI*)(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
using f_cuOccupancyMaxPotentialBlockSize =
    CUresult(CUDAAPI*)(int* minGridSize, int* blockSize, CUfunction func,
                       CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                       size_t dynamicSMemSize, int blockSizeLimit);
using f_cuTexRefSetArray = CUresult(CUDAAPI*)(CUtexref hTexRef, CUarray hArray,
                                              unsigned int Flags);
using f_cuTexRefSetMipmappedArray = CUresult(CUDAAPI*)(
    CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
using f_cuTexRefSetAddress = CUresult(CUDAAPI*)(size_t* ByteOffset,
                                                CUtexref hTexRef,
                                                CUdeviceptr dptr, size_t bytes);
using f_cuTexRefSetAddress2D =
    CUresult(CUDAAPI*)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc,
                       CUdeviceptr dptr, size_t Pitch);
using f_cuTexRefSetFormat = CUresult(CUDAAPI*)(CUtexref hTexRef,
                                               CUarray_format fmt,
                                               int NumPackedComponents);
using f_cuTexRefSetAddressMode = CUresult(CUDAAPI*)(CUtexref hTexRef, int dim,
                                                    CUaddress_mode am);
using f_cuTexRefSetFilterMode = CUresult(CUDAAPI*)(CUtexref hTexRef,
                                                   CUfilter_mode fm);
using f_cuTexRefSetMipmapFilterMode = CUresult(CUDAAPI*)(CUtexref hTexRef,
                                                         CUfilter_mode fm);
using f_cuTexRefSetMipmapLevelBias = CUresult(CUDAAPI*)(CUtexref hTexRef,
                                                        float bias);
using f_cuTexRefSetMipmapLevelClamp = CUresult(CUDAAPI*)(
    CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
using f_cuTexRefSetMaxAnisotropy = CUresult(CUDAAPI*)(CUtexref hTexRef,
                                                      unsigned int maxAniso);
using f_cuTexRefSetFlags = CUresult(CUDAAPI*)(CUtexref hTexRef,
                                              unsigned int Flags);
using f_cuTexRefGetAddress = CUresult(CUDAAPI*)(CUdeviceptr* pdptr,
                                                CUtexref hTexRef);
using f_cuTexRefGetArray = CUresult(CUDAAPI*)(CUarray* phArray,
                                              CUtexref hTexRef);
using f_cuTexRefGetMipmappedArray =
    CUresult(CUDAAPI*)(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
using f_cuTexRefGetAddressMode = CUresult(CUDAAPI*)(CUaddress_mode* pam,
                                                    CUtexref hTexRef, int dim);
using f_cuTexRefGetFilterMode = CUresult(CUDAAPI*)(CUfilter_mode* pfm,
                                                   CUtexref hTexRef);
using f_cuTexRefGetFormat = CUresult(CUDAAPI*)(CUarray_format* pFormat,
                                               int* pNumChannels,
                                               CUtexref hTexRef);
using f_cuTexRefGetMipmapFilterMode = CUresult(CUDAAPI*)(CUfilter_mode* pfm,
                                                         CUtexref hTexRef);
using f_cuTexRefGetMipmapLevelBias = CUresult(CUDAAPI*)(float* pbias,
                                                        CUtexref hTexRef);
using f_cuTexRefGetMipmapLevelClamp = CUresult(CUDAAPI*)(
    float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
using f_cuTexRefGetMaxAnisotropy = CUresult(CUDAAPI*)(int* pmaxAniso,
                                                      CUtexref hTexRef);
using f_cuTexRefGetFlags = CUresult(CUDAAPI*)(unsigned int* pFlags,
                                              CUtexref hTexRef);
using f_cuTexRefCreate = CUresult(CUDAAPI*)(CUtexref* pTexRef);
using f_cuTexRefDestroy = CUresult(CUDAAPI*)(CUtexref hTexRef);
using f_cuSurfRefSetArray = CUresult(CUDAAPI*)(CUsurfref hSurfRef,
                                               CUarray hArray,
                                               unsigned int Flags);
using f_cuSurfRefGetArray = CUresult(CUDAAPI*)(CUarray* phArray,
                                               CUsurfref hSurfRef);
using f_cuTexObjectCreate = CUresult(CUDAAPI*)(
    CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc,
    const CUDA_TEXTURE_DESC* pTexDesc,
    const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
using f_cuTexObjectDestroy = CUresult(CUDAAPI*)(CUtexObject texObject);
using f_cuTexObjectGetResourceDesc =
    CUresult(CUDAAPI*)(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
using f_cuTexObjectGetTextureDesc =
    CUresult(CUDAAPI*)(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
using f_cuTexObjectGetResourceViewDesc = CUresult(CUDAAPI*)(
    CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
using f_cuSurfObjectCreate = CUresult(CUDAAPI*)(
    CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc);
using f_cuSurfObjectDestroy = CUresult(CUDAAPI*)(CUsurfObject surfObject);
using f_cuSurfObjectGetResourceDesc =
    CUresult(CUDAAPI*)(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
using f_cuDeviceCanAccessPeer = CUresult(CUDAAPI*)(int* canAccessPeer,
                                                   CUdevice dev,
                                                   CUdevice peerDev);
using f_cuCtxEnablePeerAccess = CUresult(CUDAAPI*)(CUcontext peerContext,
                                                   unsigned int Flags);
using f_cuCtxDisablePeerAccess = CUresult(CUDAAPI*)(CUcontext peerContext);
using f_cuGraphicsUnregisterResource =
    CUresult(CUDAAPI*)(CUgraphicsResource resource);
using f_cuGraphicsSubResourceGetMappedArray =
    CUresult(CUDAAPI*)(CUarray* pArray, CUgraphicsResource resource,
                       unsigned int arrayIndex, unsigned int mipLevel);
using f_cuGraphicsResourceGetMappedMipmappedArray = CUresult(CUDAAPI*)(
    CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource);
using f_cuGraphicsResourceGetMappedPointer = CUresult(CUDAAPI*)(
    CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
using f_cuGraphicsResourceSetMapFlags =
    CUresult(CUDAAPI*)(CUgraphicsResource resource, unsigned int flags);
using f_cuGraphicsMapResources = CUresult(CUDAAPI*)(
    unsigned int count, CUgraphicsResource* resources, CUstream hStream);
using f_cuGraphicsUnmapResources = CUresult(CUDAAPI*)(
    unsigned int count, CUgraphicsResource* resources, CUstream hStream);
using f_cuGetExportTable = CUresult(CUDAAPI*)(const void** ppExportTable,
                                              const CUuuid* pExportTableId);

using f_cuDevicePrimaryCtxRetain = CUresult(CUDAAPI*)(CUcontext* pctx,
                                                      CUdevice dev);
using f_cuDevicePrimaryCtxRelease = CUresult(CUDAAPI*)(CUdevice dev);
using f_cuDevicePrimaryCtxSetFlags = CUresult(CUDAAPI*)(CUdevice dev,
                                                        unsigned int flags);
using f_cuDevicePrimaryCtxGetState = CUresult(CUDAAPI*)(CUdevice dev,
                                                        unsigned int* flags,
                                                        int* active);
using f_cuDevicePrimaryCtxReset = CUresult(CUDAAPI*)(CUdevice dev);
using f_cuCtxGetFlags = CUresult(CUDAAPI*)(unsigned int* flags);
using f_cuPointerGetAttributes = CUresult(CUDAAPI*)(
    unsigned int numAttributes, CUpointer_attribute* attributes, void** data,
    CUdeviceptr ptr);

using f_cuMemPrefetchAsync = CUresult(CUDAAPI*)(CUdeviceptr devPtr,
                                                size_t count,
                                                CUdevice dstDevice,
                                                CUstream hStream);
using f_cuMemAdvise = CUresult(CUDAAPI*)(CUdeviceptr devPtr, size_t count,
                                         CUmem_advise advice, CUdevice device);
using f_cuMemRangeGetAttribute = CUresult(CUDAAPI*)(
    void* data, size_t dataSize, CUmem_range_attribute attribute,
    CUdeviceptr devPtr, size_t count);
using f_cuMemRangeGetAttributes = CUresult(CUDAAPI*)(
    void** data, size_t* dataSizes, CUmem_range_attribute* attributes,
    size_t numAttributes, CUdeviceptr devPtr, size_t count);
using f_cuStreamWaitValue32 = CUresult(CUDAAPI*)(CUstream stream,
                                                 CUdeviceptr addr,
                                                 cuuint32_t value,
                                                 unsigned int flags);
using f_cuStreamWriteValue32 = CUresult(CUDAAPI*)(CUstream stream,
                                                  CUdeviceptr addr,
                                                  cuuint32_t value,
                                                  unsigned int flags);
using f_cuStreamBatchMemOp = CUresult(CUDAAPI*)(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray,
    unsigned int flags);

class LibCUDAWrapper {
 public:
  static LibCUDAWrapper& getInstance() {
    static LibCUDAWrapper instance;
    instance.loadCUDAFunctions();
    return instance;
  }

  LibCUDAWrapper(const LibCUDAWrapper&) = delete;
  LibCUDAWrapper& operator=(const LibCUDAWrapper&) = delete;

  void* getCUDAFunction(const char* funcName) {
    if (m_libHandler == nullptr) openLibCUDA();
#if defined(_WIN32)
    return GetProcAddress(m_libHandler, funcName);
#else
    return dlsym(m_libHandler, funcName);
#endif
  }

  f_cuGetErrorString m_cuGetErrorString = nullptr;
  f_cuGetErrorName m_cuGetErrorName = nullptr;
  f_cuInit m_cuInit = nullptr;
  f_cuDriverGetVersion m_cuDriverGetVersion = nullptr;
  f_cuDeviceGet m_cuDeviceGet = nullptr;
  f_cuDeviceGetCount m_cuDeviceGetCount = nullptr;
  f_cuDeviceGetName m_cuDeviceGetName = nullptr;
  f_cuDeviceTotalMem m_cuDeviceTotalMem = nullptr;
  f_cuDeviceGetAttribute m_cuDeviceGetAttribute = nullptr;
  f_cuDeviceGetProperties m_cuDeviceGetProperties = nullptr;
  f_cuDeviceComputeCapability m_cuDeviceComputeCapability = nullptr;
  f_cuCtxCreate m_cuCtxCreate = nullptr;
  f_cuCtxDestroy m_cuCtxDestroy = nullptr;
  f_cuCtxPushCurrent m_cuCtxPushCurrent = nullptr;
  f_cuCtxPopCurrent m_cuCtxPopCurrent = nullptr;
  f_cuCtxSetCurrent m_cuCtxSetCurrent = nullptr;
  f_cuCtxGetCurrent m_cuCtxGetCurrent = nullptr;
  f_cuCtxGetDevice m_cuCtxGetDevice = nullptr;
  f_cuCtxSynchronize m_cuCtxSynchronize = nullptr;
  f_cuCtxSetLimit m_cuCtxSetLimit = nullptr;
  f_cuCtxGetLimit m_cuCtxGetLimit = nullptr;
  f_cuCtxGetCacheConfig m_cuCtxGetCacheConfig = nullptr;
  f_cuCtxSetCacheConfig m_cuCtxSetCacheConfig = nullptr;
  f_cuCtxGetSharedMemConfig m_cuCtxGetSharedMemConfig = nullptr;
  f_cuCtxSetSharedMemConfig m_cuCtxSetSharedMemConfig = nullptr;
  f_cuCtxGetApiVersion m_cuCtxGetApiVersion = nullptr;
  f_cuCtxGetStreamPriorityRange m_cuCtxGetStreamPriorityRange = nullptr;
  f_cuCtxAttach m_cuCtxAttach = nullptr;
  f_cuCtxDetach m_cuCtxDetach = nullptr;
  f_cuModuleLoad m_cuModuleLoad = nullptr;
  f_cuModuleLoadData m_cuModuleLoadData = nullptr;
  f_cuModuleLoadDataEx m_cuModuleLoadDataEx = nullptr;
  f_cuModuleLoadFatBinary m_cuModuleLoadFatBinary = nullptr;
  f_cuModuleUnload m_cuModuleUnload = nullptr;
  f_cuModuleGetFunction m_cuModuleGetFunction = nullptr;
  f_cuModuleGetGlobal m_cuModuleGetGlobal = nullptr;
  f_cuModuleGetTexRef m_cuModuleGetTexRef = nullptr;
  f_cuModuleGetSurfRef m_cuModuleGetSurfRef = nullptr;
  f_cuLinkCreate m_cuLinkCreate = nullptr;
  f_cuLinkAddData m_cuLinkAddData = nullptr;
  f_cuLinkAddFile m_cuLinkAddFile = nullptr;
  f_cuLinkComplete m_cuLinkComplete = nullptr;
  f_cuLinkDestroy m_cuLinkDestroy = nullptr;
  f_cuMemGetInfo m_cuMemGetInfo = nullptr;
  f_cuMemAlloc m_cuMemAlloc = nullptr;
  f_cuMemAllocPitch m_cuMemAllocPitch = nullptr;
  f_cuMemFree m_cuMemFree = nullptr;
  f_cuMemGetAddressRange m_cuMemGetAddressRange = nullptr;
  f_cuMemAllocHost m_cuMemAllocHost = nullptr;
  f_cuMemFreeHost m_cuMemFreeHost = nullptr;
  f_cuMemHostAlloc m_cuMemHostAlloc = nullptr;
  f_cuMemHostGetDevicePointer m_cuMemHostGetDevicePointer = nullptr;
  f_cuMemHostGetFlags m_cuMemHostGetFlags = nullptr;
  f_cuMemAllocManaged m_cuMemAllocManaged = nullptr;
  f_cuDeviceGetByPCIBusId m_cuDeviceGetByPCIBusId = nullptr;
  f_cuDeviceGetPCIBusId m_cuDeviceGetPCIBusId = nullptr;
  f_cuIpcGetEventHandle m_cuIpcGetEventHandle = nullptr;
  f_cuIpcOpenEventHandle m_cuIpcOpenEventHandle = nullptr;
  f_cuIpcGetMemHandle m_cuIpcGetMemHandle = nullptr;
  f_cuIpcOpenMemHandle m_cuIpcOpenMemHandle = nullptr;
  f_cuIpcCloseMemHandle m_cuIpcCloseMemHandle = nullptr;
  f_cuMemHostRegister m_cuMemHostRegister = nullptr;
  f_cuMemHostUnregister m_cuMemHostUnregister = nullptr;
  f_cuMemcpy m_cuMemcpy = nullptr;
  f_cuMemcpyPeer m_cuMemcpyPeer = nullptr;
  f_cuMemcpyHtoD m_cuMemcpyHtoD = nullptr;
  f_cuMemcpyDtoH m_cuMemcpyDtoH = nullptr;
  f_cuMemcpyDtoD m_cuMemcpyDtoD = nullptr;
  f_cuMemcpyDtoA m_cuMemcpyDtoA = nullptr;
  f_cuMemcpyAtoD m_cuMemcpyAtoD = nullptr;
  f_cuMemcpyHtoA m_cuMemcpyHtoA = nullptr;
  f_cuMemcpyAtoH m_cuMemcpyAtoH = nullptr;
  f_cuMemcpyAtoA m_cuMemcpyAtoA = nullptr;
  f_cuMemcpy2D m_cuMemcpy2D = nullptr;
  f_cuMemcpy2DUnaligned m_cuMemcpy2DUnaligned = nullptr;
  f_cuMemcpy3D m_cuMemcpy3D = nullptr;
  f_cuMemcpy3DPeer m_cuMemcpy3DPeer = nullptr;
  f_cuMemcpyAsync m_cuMemcpyAsync = nullptr;
  f_cuMemcpyPeerAsync m_cuMemcpyPeerAsync = nullptr;
  f_cuMemcpyHtoDAsync m_cuMemcpyHtoDAsync = nullptr;
  f_cuMemcpyDtoHAsync m_cuMemcpyDtoHAsync = nullptr;
  f_cuMemcpyDtoDAsync m_cuMemcpyDtoDAsync = nullptr;
  f_cuMemcpyHtoAAsync m_cuMemcpyHtoAAsync = nullptr;
  f_cuMemcpyAtoHAsync m_cuMemcpyAtoHAsync = nullptr;
  f_cuMemcpy2DAsync m_cuMemcpy2DAsync = nullptr;
  f_cuMemcpy3DAsync m_cuMemcpy3DAsync = nullptr;
  f_cuMemcpy3DPeerAsync m_cuMemcpy3DPeerAsync = nullptr;
  f_cuMemsetD8 m_cuMemsetD8 = nullptr;
  f_cuMemsetD16 m_cuMemsetD16 = nullptr;
  f_cuMemsetD32 m_cuMemsetD32 = nullptr;
  f_cuMemsetD2D8 m_cuMemsetD2D8 = nullptr;
  f_cuMemsetD2D16 m_cuMemsetD2D16 = nullptr;
  f_cuMemsetD2D32 m_cuMemsetD2D32 = nullptr;
  f_cuMemsetD8Async m_cuMemsetD8Async = nullptr;
  f_cuMemsetD16Async m_cuMemsetD16Async = nullptr;
  f_cuMemsetD32Async m_cuMemsetD32Async = nullptr;
  f_cuMemsetD2D8Async m_cuMemsetD2D8Async = nullptr;
  f_cuMemsetD2D16Async m_cuMemsetD2D16Async = nullptr;
  f_cuMemsetD2D32Async m_cuMemsetD2D32Async = nullptr;
  f_cuArrayCreate m_cuArrayCreate = nullptr;
  f_cuArrayGetDescriptor m_cuArrayGetDescriptor = nullptr;
  f_cuArrayDestroy m_cuArrayDestroy = nullptr;
  f_cuArray3DCreate m_cuArray3DCreate = nullptr;
  f_cuArray3DGetDescriptor m_cuArray3DGetDescriptor = nullptr;
  f_cuMipmappedArrayCreate m_cuMipmappedArrayCreate = nullptr;
  f_cuMipmappedArrayGetLevel m_cuMipmappedArrayGetLevel = nullptr;
  f_cuMipmappedArrayDestroy m_cuMipmappedArrayDestroy = nullptr;
  f_cuPointerGetAttribute m_cuPointerGetAttribute = nullptr;
  f_cuPointerSetAttribute m_cuPointerSetAttribute = nullptr;
  f_cuStreamCreate m_cuStreamCreate = nullptr;
  f_cuStreamCreateWithPriority m_cuStreamCreateWithPriority = nullptr;
  f_cuStreamGetPriority m_cuStreamGetPriority = nullptr;
  f_cuStreamGetFlags m_cuStreamGetFlags = nullptr;
  f_cuStreamWaitEvent m_cuStreamWaitEvent = nullptr;
  f_cuStreamAddCallback m_cuStreamAddCallback = nullptr;
  f_cuStreamAttachMemAsync m_cuStreamAttachMemAsync = nullptr;
  f_cuStreamQuery m_cuStreamQuery = nullptr;
  f_cuStreamSynchronize m_cuStreamSynchronize = nullptr;
  f_cuStreamDestroy m_cuStreamDestroy = nullptr;
  f_cuEventCreate m_cuEventCreate = nullptr;
  f_cuEventRecord m_cuEventRecord = nullptr;
  f_cuEventQuery m_cuEventQuery = nullptr;
  f_cuEventSynchronize m_cuEventSynchronize = nullptr;
  f_cuEventDestroy m_cuEventDestroy = nullptr;
  f_cuEventElapsedTime m_cuEventElapsedTime = nullptr;
  f_cuFuncGetAttribute m_cuFuncGetAttribute = nullptr;
  f_cuFuncSetAttribute m_cuFuncSetAttribute = nullptr;
  f_cuFuncSetCacheConfig m_cuFuncSetCacheConfig = nullptr;
  f_cuFuncSetSharedMemConfig m_cuFuncSetSharedMemConfig = nullptr;
  f_cuLaunchKernel m_cuLaunchKernel = nullptr;
  f_cuFuncSetBlockShape m_cuFuncSetBlockShape = nullptr;
  f_cuFuncSetSharedSize m_cuFuncSetSharedSize = nullptr;
  f_cuParamSetSize m_cuParamSetSize = nullptr;
  f_cuParamSeti m_cuParamSeti = nullptr;
  f_cuParamSetf m_cuParamSetf = nullptr;
  f_cuParamSetv m_cuParamSetv = nullptr;
  f_cuLaunch m_cuLaunch = nullptr;
  f_cuLaunchGrid m_cuLaunchGrid = nullptr;
  f_cuLaunchGridAsync m_cuLaunchGridAsync = nullptr;
  f_cuParamSetTexRef m_cuParamSetTexRef = nullptr;
  f_cuOccupancyMaxActiveBlocksPerMultiprocessor
      m_cuOccupancyMaxActiveBlocksPerMultiprocessor = nullptr;
  f_cuOccupancyMaxPotentialBlockSize m_cuOccupancyMaxPotentialBlockSize =
      nullptr;
  f_cuTexRefSetArray m_cuTexRefSetArray = nullptr;
  f_cuTexRefSetMipmappedArray m_cuTexRefSetMipmappedArray = nullptr;
  f_cuTexRefSetAddress m_cuTexRefSetAddress = nullptr;
  f_cuTexRefSetAddress2D m_cuTexRefSetAddress2D = nullptr;
  f_cuTexRefSetFormat m_cuTexRefSetFormat = nullptr;
  f_cuTexRefSetAddressMode m_cuTexRefSetAddressMode = nullptr;
  f_cuTexRefSetFilterMode m_cuTexRefSetFilterMode = nullptr;
  f_cuTexRefSetMipmapFilterMode m_cuTexRefSetMipmapFilterMode = nullptr;
  f_cuTexRefSetMipmapLevelBias m_cuTexRefSetMipmapLevelBias = nullptr;
  f_cuTexRefSetMipmapLevelClamp m_cuTexRefSetMipmapLevelClamp = nullptr;
  f_cuTexRefSetMaxAnisotropy m_cuTexRefSetMaxAnisotropy = nullptr;
  f_cuTexRefSetFlags m_cuTexRefSetFlags = nullptr;
  f_cuTexRefGetAddress m_cuTexRefGetAddress = nullptr;
  f_cuTexRefGetArray m_cuTexRefGetArray = nullptr;
  f_cuTexRefGetMipmappedArray m_cuTexRefGetMipmappedArray = nullptr;
  f_cuTexRefGetAddressMode m_cuTexRefGetAddressMode = nullptr;
  f_cuTexRefGetFilterMode m_cuTexRefGetFilterMode = nullptr;
  f_cuTexRefGetFormat m_cuTexRefGetFormat = nullptr;
  f_cuTexRefGetMipmapFilterMode m_cuTexRefGetMipmapFilterMode = nullptr;
  f_cuTexRefGetMipmapLevelBias m_cuTexRefGetMipmapLevelBias = nullptr;
  f_cuTexRefGetMipmapLevelClamp m_cuTexRefGetMipmapLevelClamp = nullptr;
  f_cuTexRefGetMaxAnisotropy m_cuTexRefGetMaxAnisotropy = nullptr;
  f_cuTexRefGetFlags m_cuTexRefGetFlags = nullptr;
  f_cuTexRefCreate m_cuTexRefCreate = nullptr;
  f_cuTexRefDestroy m_cuTexRefDestroy = nullptr;
  f_cuSurfRefSetArray m_cuSurfRefSetArray = nullptr;
  f_cuSurfRefGetArray m_cuSurfRefGetArray = nullptr;
  f_cuTexObjectCreate m_cuTexObjectCreate = nullptr;
  f_cuTexObjectDestroy m_cuTexObjectDestroy = nullptr;
  f_cuTexObjectGetResourceDesc m_cuTexObjectGetResourceDesc = nullptr;
  f_cuTexObjectGetTextureDesc m_cuTexObjectGetTextureDesc = nullptr;
  f_cuTexObjectGetResourceViewDesc m_cuTexObjectGetResourceViewDesc = nullptr;
  f_cuSurfObjectCreate m_cuSurfObjectCreate = nullptr;
  f_cuSurfObjectDestroy m_cuSurfObjectDestroy = nullptr;
  f_cuSurfObjectGetResourceDesc m_cuSurfObjectGetResourceDesc = nullptr;
  f_cuDeviceCanAccessPeer m_cuDeviceCanAccessPeer = nullptr;
  f_cuCtxEnablePeerAccess m_cuCtxEnablePeerAccess = nullptr;
  f_cuCtxDisablePeerAccess m_cuCtxDisablePeerAccess = nullptr;
  f_cuGraphicsUnregisterResource m_cuGraphicsUnregisterResource = nullptr;
  f_cuGraphicsSubResourceGetMappedArray m_cuGraphicsSubResourceGetMappedArray =
      nullptr;
  f_cuGraphicsResourceGetMappedMipmappedArray
      m_cuGraphicsResourceGetMappedMipmappedArray = nullptr;
  f_cuGraphicsResourceGetMappedPointer m_cuGraphicsResourceGetMappedPointer =
      nullptr;
  f_cuGraphicsResourceSetMapFlags m_cuGraphicsResourceSetMapFlags = nullptr;
  f_cuGraphicsMapResources m_cuGraphicsMapResources = nullptr;
  f_cuGraphicsUnmapResources m_cuGraphicsUnmapResources = nullptr;
  f_cuGetExportTable m_cuGetExportTable = nullptr;

  f_cuDevicePrimaryCtxRetain m_cuDevicePrimaryCtxRetain = nullptr;
  f_cuDevicePrimaryCtxRelease m_cuDevicePrimaryCtxRelease = nullptr;
  f_cuDevicePrimaryCtxSetFlags m_cuDevicePrimaryCtxSetFlags = nullptr;
  f_cuDevicePrimaryCtxGetState m_cuDevicePrimaryCtxGetState = nullptr;
  f_cuDevicePrimaryCtxReset m_cuDevicePrimaryCtxReset = nullptr;
  f_cuCtxGetFlags m_cuCtxGetFlags = nullptr;
  f_cuPointerGetAttributes m_cuPointerGetAttributes = nullptr;

  f_cuMemPrefetchAsync m_cuMemPrefetchAsync = nullptr;
  f_cuMemAdvise m_cuMemAdvise = nullptr;
  f_cuMemRangeGetAttribute m_cuMemRangeGetAttribute = nullptr;
  f_cuMemRangeGetAttributes m_cuMemRangeGetAttributes = nullptr;
  f_cuStreamWaitValue32 m_cuStreamWaitValue32 = nullptr;
  f_cuStreamWriteValue32 m_cuStreamWriteValue32 = nullptr;
  f_cuStreamBatchMemOp m_cuStreamBatchMemOp = nullptr;

 private:
  LibCUDAWrapper() {}

  ~LibCUDAWrapper() {
#if defined(_WIN32)
    if (m_libHandler) FreeLibrary(m_libHandler);
#else
    if (m_libHandler) dlclose(m_libHandler);
#endif
  }

  void openLibCUDA() {
    for (const auto it : default_so_paths) {
#if defined(_WIN32)
      m_libHandler = LoadLibrary(it);
#else
      m_libHandler = dlopen(it, RTLD_LAZY);
#endif
      if (m_libHandler != nullptr) return;
    }
    ICHECK(m_libHandler != nullptr) << "Error! Cannot open libcuda!";
  }

  void loadCUDAFunctions() {
    if (m_failed) return;
    if (m_libHandler == nullptr) openLibCUDA();
    if (!m_libHandler) {
      m_failed = true;
      return;
    }
    INIT_FUNCTION(cuGetErrorString);
    INIT_FUNCTION(cuGetErrorName);
    INIT_FUNCTION(cuInit);
    INIT_FUNCTION(cuDriverGetVersion);
    INIT_FUNCTION(cuDeviceGet);
    INIT_FUNCTION(cuDeviceGetCount);
    INIT_FUNCTION(cuDeviceGetName);
    INIT_FUNCTION(cuDeviceTotalMem);
    INIT_FUNCTION(cuDeviceGetAttribute);
    INIT_FUNCTION(cuDeviceGetProperties);
    INIT_FUNCTION(cuDeviceComputeCapability);
    INIT_FUNCTION(cuCtxCreate);
    INIT_FUNCTION(cuCtxDestroy);
    INIT_FUNCTION(cuCtxPushCurrent);
    INIT_FUNCTION(cuCtxPopCurrent);
    INIT_FUNCTION(cuCtxSetCurrent);
    INIT_FUNCTION(cuCtxGetCurrent);
    INIT_FUNCTION(cuCtxGetDevice);
    INIT_FUNCTION(cuCtxSynchronize);
    INIT_FUNCTION(cuCtxSetLimit);
    INIT_FUNCTION(cuCtxGetLimit);
    INIT_FUNCTION(cuCtxGetCacheConfig);
    INIT_FUNCTION(cuCtxSetCacheConfig);
    INIT_FUNCTION(cuCtxGetSharedMemConfig);
    INIT_FUNCTION(cuCtxSetSharedMemConfig);
    INIT_FUNCTION(cuCtxGetApiVersion);
    INIT_FUNCTION(cuCtxGetStreamPriorityRange);
    INIT_FUNCTION(cuCtxAttach);
    INIT_FUNCTION(cuCtxDetach);
    INIT_FUNCTION(cuModuleLoad);
    INIT_FUNCTION(cuModuleLoadData);
    INIT_FUNCTION(cuModuleLoadDataEx);
    INIT_FUNCTION(cuModuleLoadFatBinary);
    INIT_FUNCTION(cuModuleUnload);
    INIT_FUNCTION(cuModuleGetFunction);
    INIT_FUNCTION(cuModuleGetGlobal);
    INIT_FUNCTION(cuModuleGetTexRef);
    INIT_FUNCTION(cuModuleGetSurfRef);
    INIT_FUNCTION(cuLinkCreate);
    INIT_FUNCTION(cuLinkAddData);
    INIT_FUNCTION(cuLinkAddFile);
    INIT_FUNCTION(cuLinkComplete);
    INIT_FUNCTION(cuLinkDestroy);
    INIT_FUNCTION(cuMemGetInfo);
    INIT_FUNCTION(cuMemAlloc);
    INIT_FUNCTION(cuMemAllocPitch);
    INIT_FUNCTION(cuMemFree);
    INIT_FUNCTION(cuMemGetAddressRange);
    INIT_FUNCTION(cuMemAllocHost);
    INIT_FUNCTION(cuMemFreeHost);
    INIT_FUNCTION(cuMemHostAlloc);
    INIT_FUNCTION(cuMemHostGetDevicePointer);
    INIT_FUNCTION(cuMemHostGetFlags);
    INIT_FUNCTION(cuMemAllocManaged);
    INIT_FUNCTION(cuDeviceGetByPCIBusId);
    INIT_FUNCTION(cuDeviceGetPCIBusId);
    INIT_FUNCTION(cuIpcGetEventHandle);
    INIT_FUNCTION(cuIpcOpenEventHandle);
    INIT_FUNCTION(cuIpcGetMemHandle);
    INIT_FUNCTION(cuIpcOpenMemHandle);
    INIT_FUNCTION(cuIpcCloseMemHandle);
    INIT_FUNCTION(cuMemHostRegister);
    INIT_FUNCTION(cuMemHostUnregister);
    INIT_FUNCTION(cuMemcpy);
    INIT_FUNCTION(cuMemcpyPeer);
    INIT_FUNCTION(cuMemcpyHtoD);
    INIT_FUNCTION(cuMemcpyDtoH);
    INIT_FUNCTION(cuMemcpyDtoD);
    INIT_FUNCTION(cuMemcpyDtoA);
    INIT_FUNCTION(cuMemcpyAtoD);
    INIT_FUNCTION(cuMemcpyHtoA);
    INIT_FUNCTION(cuMemcpyAtoH);
    INIT_FUNCTION(cuMemcpyAtoA);
    INIT_FUNCTION(cuMemcpy2D);
    INIT_FUNCTION(cuMemcpy2DUnaligned);
    INIT_FUNCTION(cuMemcpy3D);
    INIT_FUNCTION(cuMemcpy3DPeer);
    INIT_FUNCTION(cuMemcpyAsync);
    INIT_FUNCTION(cuMemcpyPeerAsync);
    INIT_FUNCTION(cuMemcpyHtoDAsync);
    INIT_FUNCTION(cuMemcpyDtoHAsync);
    INIT_FUNCTION(cuMemcpyDtoDAsync);
    INIT_FUNCTION(cuMemcpyHtoAAsync);
    INIT_FUNCTION(cuMemcpyAtoHAsync);
    INIT_FUNCTION(cuMemcpy2DAsync);
    INIT_FUNCTION(cuMemcpy3DAsync);
    INIT_FUNCTION(cuMemcpy3DPeerAsync);
    INIT_FUNCTION(cuMemsetD8);
    INIT_FUNCTION(cuMemsetD16);
    INIT_FUNCTION(cuMemsetD32);
    INIT_FUNCTION(cuMemsetD2D8);
    INIT_FUNCTION(cuMemsetD2D16);
    INIT_FUNCTION(cuMemsetD2D32);
    INIT_FUNCTION(cuMemsetD8Async);
    INIT_FUNCTION(cuMemsetD16Async);
    INIT_FUNCTION(cuMemsetD32Async);
    INIT_FUNCTION(cuMemsetD2D8Async);
    INIT_FUNCTION(cuMemsetD2D16Async);
    INIT_FUNCTION(cuMemsetD2D32Async);
    INIT_FUNCTION(cuArrayCreate);
    INIT_FUNCTION(cuArrayGetDescriptor);
    INIT_FUNCTION(cuArrayDestroy);
    INIT_FUNCTION(cuArray3DCreate);
    INIT_FUNCTION(cuArray3DGetDescriptor);
    INIT_FUNCTION(cuMipmappedArrayCreate);
    INIT_FUNCTION(cuMipmappedArrayGetLevel);
    INIT_FUNCTION(cuMipmappedArrayDestroy);
    INIT_FUNCTION(cuPointerGetAttribute);
    INIT_FUNCTION(cuPointerSetAttribute);
    INIT_FUNCTION(cuStreamCreate);
    INIT_FUNCTION(cuStreamCreateWithPriority);
    INIT_FUNCTION(cuStreamGetPriority);
    INIT_FUNCTION(cuStreamGetFlags);
    INIT_FUNCTION(cuStreamWaitEvent);
    INIT_FUNCTION(cuStreamAddCallback);
    INIT_FUNCTION(cuStreamAttachMemAsync);
    INIT_FUNCTION(cuStreamQuery);
    INIT_FUNCTION(cuStreamSynchronize);
    INIT_FUNCTION(cuStreamDestroy);
    INIT_FUNCTION(cuEventCreate);
    INIT_FUNCTION(cuEventRecord);
    INIT_FUNCTION(cuEventQuery);
    INIT_FUNCTION(cuEventSynchronize);
    INIT_FUNCTION(cuEventDestroy);
    INIT_FUNCTION(cuEventElapsedTime);
    INIT_FUNCTION(cuFuncGetAttribute);
    INIT_FUNCTION(cuFuncSetAttribute);
    INIT_FUNCTION(cuFuncSetCacheConfig);
    INIT_FUNCTION(cuFuncSetSharedMemConfig);
    INIT_FUNCTION(cuLaunchKernel);
    INIT_FUNCTION(cuFuncSetBlockShape);
    INIT_FUNCTION(cuFuncSetSharedSize);
    INIT_FUNCTION(cuParamSetSize);
    INIT_FUNCTION(cuParamSeti);
    INIT_FUNCTION(cuParamSetf);
    INIT_FUNCTION(cuParamSetv);
    INIT_FUNCTION(cuLaunch);
    INIT_FUNCTION(cuLaunchGrid);
    INIT_FUNCTION(cuLaunchGridAsync);
    INIT_FUNCTION(cuParamSetTexRef);
    INIT_FUNCTION(cuOccupancyMaxActiveBlocksPerMultiprocessor);
    INIT_FUNCTION(cuOccupancyMaxPotentialBlockSize);
    INIT_FUNCTION(cuTexRefSetArray);
    INIT_FUNCTION(cuTexRefSetMipmappedArray);
    INIT_FUNCTION(cuTexRefSetAddress);
    INIT_FUNCTION(cuTexRefSetAddress2D);
    INIT_FUNCTION(cuTexRefSetFormat);
    INIT_FUNCTION(cuTexRefSetAddressMode);
    INIT_FUNCTION(cuTexRefSetFilterMode);
    INIT_FUNCTION(cuTexRefSetMipmapFilterMode);
    INIT_FUNCTION(cuTexRefSetMipmapLevelBias);
    INIT_FUNCTION(cuTexRefSetMipmapLevelClamp);
    INIT_FUNCTION(cuTexRefSetMaxAnisotropy);
    INIT_FUNCTION(cuTexRefSetFlags);
    INIT_FUNCTION(cuTexRefGetAddress);
    INIT_FUNCTION(cuTexRefGetArray);
    INIT_FUNCTION(cuTexRefGetMipmappedArray);
    INIT_FUNCTION(cuTexRefGetAddressMode);
    INIT_FUNCTION(cuTexRefGetFilterMode);
    INIT_FUNCTION(cuTexRefGetFormat);
    INIT_FUNCTION(cuTexRefGetMipmapFilterMode);
    INIT_FUNCTION(cuTexRefGetMipmapLevelBias);
    INIT_FUNCTION(cuTexRefGetMipmapLevelClamp);
    INIT_FUNCTION(cuTexRefGetMaxAnisotropy);
    INIT_FUNCTION(cuTexRefGetFlags);
    INIT_FUNCTION(cuTexRefCreate);
    INIT_FUNCTION(cuTexRefDestroy);
    INIT_FUNCTION(cuSurfRefSetArray);
    INIT_FUNCTION(cuSurfRefGetArray);
    INIT_FUNCTION(cuTexObjectCreate);
    INIT_FUNCTION(cuTexObjectDestroy);
    INIT_FUNCTION(cuTexObjectGetResourceDesc);
    INIT_FUNCTION(cuTexObjectGetTextureDesc);
    INIT_FUNCTION(cuTexObjectGetResourceViewDesc);
    INIT_FUNCTION(cuSurfObjectCreate);
    INIT_FUNCTION(cuSurfObjectDestroy);
    INIT_FUNCTION(cuSurfObjectGetResourceDesc);
    INIT_FUNCTION(cuDeviceCanAccessPeer);
    INIT_FUNCTION(cuCtxEnablePeerAccess);
    INIT_FUNCTION(cuCtxDisablePeerAccess);
    INIT_FUNCTION(cuGraphicsUnregisterResource);
    INIT_FUNCTION(cuGraphicsSubResourceGetMappedArray);
    INIT_FUNCTION(cuGraphicsResourceGetMappedMipmappedArray);
    INIT_FUNCTION(cuGraphicsResourceGetMappedPointer);
    INIT_FUNCTION(cuGraphicsResourceSetMapFlags);
    INIT_FUNCTION(cuGraphicsMapResources);
    INIT_FUNCTION(cuGraphicsUnmapResources);
    INIT_FUNCTION(cuGetExportTable);

    INIT_FUNCTION(cuDevicePrimaryCtxRetain);
    INIT_FUNCTION(cuDevicePrimaryCtxRelease);
    INIT_FUNCTION(cuDevicePrimaryCtxSetFlags);
    INIT_FUNCTION(cuDevicePrimaryCtxGetState);
    INIT_FUNCTION(cuDevicePrimaryCtxReset);
    INIT_FUNCTION(cuCtxGetFlags);
    INIT_FUNCTION(cuPointerGetAttributes);

    INIT_FUNCTION(cuMemPrefetchAsync);
    INIT_FUNCTION(cuMemAdvise);
    INIT_FUNCTION(cuMemRangeGetAttribute);
    INIT_FUNCTION(cuMemRangeGetAttributes);
    INIT_FUNCTION(cuStreamWaitValue32);
    INIT_FUNCTION(cuStreamWriteValue32);
    INIT_FUNCTION(cuStreamBatchMemOp);
  }

 private:
#if defined(_WIN32)
  HMODULE m_libHandler = nullptr;
#else
  void* m_libHandler = nullptr;
#endif
  bool m_failed = false;
};
}  // namespace

CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGetErrorString;
  if (func) {
    return func(error, pStr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGetErrorName;
  if (func) {
    return func(error, pStr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuInit;
  if (func) {
    return func(Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDriverGetVersion(int* driverVersion) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDriverGetVersion;
  if (func) {
    return func(driverVersion);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGet;
  if (func) {
    return func(device, ordinal);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGetCount(int* count) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGetCount;
  if (func) {
    return func(count);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGetName;
  if (func) {
    return func(name, len, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceTotalMem;
  if (func) {
    return func(bytes, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGetAttribute;
  if (func) {
    return func(pi, attrib, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGetProperties;
  if (func) {
    return func(prop, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor,
                                           CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceComputeCapability;
  if (func) {
    return func(major, minor, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

#if CUDA_VERSION >= 13000

CUresult CUDAAPI cuCtxCreate(CUcontext* pctx, CUctxCreateParams* params,
                             unsigned int flags, CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxCreate;
  if (func) {
    return func(pctx, params, flags, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

#else

CUresult CUDAAPI cuCtxCreate(CUcontext* pctx, unsigned int flags,
                             CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxCreate;
  if (func) {
    return func(pctx, flags, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

#endif

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxDestroy;
  if (func) {
    return func(ctx);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxPushCurrent;
  if (func) {
    return func(ctx);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxPopCurrent;
  if (func) {
    return func(pctx);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxSetCurrent;
  if (func) {
    return func(ctx);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetCurrent;
  if (func) {
    return func(pctx);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetDevice;
  if (func) {
    return func(device);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxSynchronize(void) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxSynchronize;
  if (func) {
    return func();
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxSetLimit;
  if (func) {
    return func(limit, value);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetLimit(size_t* pvalue, CUlimit limit) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetLimit;
  if (func) {
    return func(pvalue, limit);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache* pconfig) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetCacheConfig;
  if (func) {
    return func(pconfig);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxSetCacheConfig;
  if (func) {
    return func(config);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetSharedMemConfig;
  if (func) {
    return func(pConfig);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxSetSharedMemConfig;
  if (func) {
    return func(config);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetApiVersion;
  if (func) {
    return func(ctx, version);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int* leastPriority,
                                             int* greatestPriority) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetStreamPriorityRange;
  if (func) {
    return func(leastPriority, greatestPriority);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxAttach(CUcontext* pctx, unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxAttach;
  if (func) {
    return func(pctx, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxDetach;
  if (func) {
    return func(ctx);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleLoad;
  if (func) {
    return func(module, fname);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleLoadData;
  if (func) {
    return func(module, image);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image,
                                    unsigned int numOptions,
                                    CUjit_option* options,
                                    void** optionValues) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleLoadDataEx;
  if (func) {
    return func(module, image, numOptions, options, optionValues);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleLoadFatBinary;
  if (func) {
    return func(module, fatCubin);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleUnload;
  if (func) {
    return func(hmod);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                                     const char* name) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleGetFunction;
  if (func) {
    return func(hfunc, hmod, name);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                                   CUmodule hmod, const char* name) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleGetGlobal;
  if (func) {
    return func(dptr, bytes, hmod, name);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod,
                                   const char* name) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleGetTexRef;
  if (func) {
    return func(pTexRef, hmod, name);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod,
                                    const char* name) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuModuleGetSurfRef;
  if (func) {
    return func(pSurfRef, hmod, name);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                              void** optionValues, CUlinkState* stateOut) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLinkCreate;
  if (func) {
    return func(numOptions, options, optionValues, stateOut);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void* data, size_t size, const char* name,
                               unsigned int numOptions, CUjit_option* options,
                               void** optionValues) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLinkAddData;
  if (func) {
    return func(state, type, data, size, name, numOptions, options,
                optionValues);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char* path, unsigned int numOptions,
                               CUjit_option* options, void** optionValues) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLinkAddFile;
  if (func) {
    return func(state, type, path, numOptions, options, optionValues);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void** cubinOut,
                                size_t* sizeOut) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLinkComplete;
  if (func) {
    return func(state, cubinOut, sizeOut);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLinkDestroy;
  if (func) {
    return func(state);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemGetInfo(size_t* free, size_t* total) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemGetInfo;
  if (func) {
    return func(free, total);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemAlloc;
  if (func) {
    return func(dptr, bytesize);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemAllocPitch;
  if (func) {
    return func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemFree;
  if (func) {
    return func(dptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize,
                                      CUdeviceptr dptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemGetAddressRange;
  if (func) {
    return func(pbase, psize, dptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemAllocHost;
  if (func) {
    return func(pp, bytesize);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemFreeHost(void* p) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemFreeHost;
  if (func) {
    return func(p);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize,
                                unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemHostAlloc;
  if (func) {
    return func(pp, bytesize, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p,
                                           unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemHostGetDevicePointer;
  if (func) {
    return func(pdptr, p, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int* pFlags, void* p) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemHostGetFlags;
  if (func) {
    return func(pFlags, p);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize,
                                   unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemAllocManaged;
  if (func) {
    return func(dptr, bytesize, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGetByPCIBusId;
  if (func) {
    return func(dev, pciBusId);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceGetPCIBusId;
  if (func) {
    return func(pciBusId, len, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuIpcGetEventHandle;
  if (func) {
    return func(pHandle, event);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent* phEvent,
                                      CUipcEventHandle handle) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuIpcOpenEventHandle;
  if (func) {
    return func(phEvent, handle);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuIpcGetMemHandle;
  if (func) {
    return func(pHandle, dptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuIpcOpenMemHandle;
  if (func) {
    return func(pdptr, handle, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuIpcCloseMemHandle;
  if (func) {
    return func(dptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemHostRegister(void* p, size_t bytesize,
                                   unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemHostRegister;
  if (func) {
    return func(p, bytesize, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemHostUnregister(void* p) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemHostUnregister;
  if (func) {
    return func(p);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy;
  if (func) {
    return func(dst, src, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyPeer;
  if (func) {
    return func(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost,
                              size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyHtoD;
  if (func) {
    return func(dstDevice, srcHost, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyDtoH;
  if (func) {
    return func(dstHost, srcDevice, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyDtoD;
  if (func) {
    return func(dstDevice, srcDevice, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                              CUdeviceptr srcDevice, size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyDtoA;
  if (func) {
    return func(dstArray, dstOffset, srcDevice, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                              size_t srcOffset, size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyAtoD;
  if (func) {
    return func(dstDevice, srcArray, srcOffset, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                              const void* srcHost, size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyHtoA;
  if (func) {
    return func(dstArray, dstOffset, srcHost, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyAtoH;
  if (func) {
    return func(dstHost, srcArray, srcOffset, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                              CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyAtoA;
  if (func) {
    return func(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy2D;
  if (func) {
    return func(pCopy);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy2DUnaligned;
  if (func) {
    return func(pCopy);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy3D;
  if (func) {
    return func(pCopy);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy3DPeer;
  if (func) {
    return func(pCopy);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyAsync;
  if (func) {
    return func(dst, src, ByteCount, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyPeerAsync;
  if (func) {
    return func(dstDevice, dstContext, srcDevice, srcContext, ByteCount,
                hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                                   size_t ByteCount, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyHtoDAsync;
  if (func) {
    return func(dstDevice, srcHost, ByteCount, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyDtoHAsync;
  if (func) {
    return func(dstHost, srcDevice, ByteCount, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyDtoDAsync;
  if (func) {
    return func(dstDevice, srcDevice, ByteCount, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                   const void* srcHost, size_t ByteCount,
                                   CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyHtoAAsync;
  if (func) {
    return func(dstArray, dstOffset, srcHost, ByteCount, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpyAtoHAsync;
  if (func) {
    return func(dstHost, srcArray, srcOffset, ByteCount, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy2DAsync;
  if (func) {
    return func(pCopy, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy3DAsync;
  if (func) {
    return func(pCopy, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy,
                                     CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemcpy3DPeerAsync;
  if (func) {
    return func(pCopy, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD8;
  if (func) {
    return func(dstDevice, uc, N);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD16;
  if (func) {
    return func(dstDevice, us, N);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD32;
  if (func) {
    return func(dstDevice, ui, N);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                              unsigned char uc, size_t Width, size_t Height) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD2D8;
  if (func) {
    return func(dstDevice, dstPitch, uc, Width, Height);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned short us, size_t Width, size_t Height) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD2D16;
  if (func) {
    return func(dstDevice, dstPitch, us, Width, Height);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned int ui, size_t Width, size_t Height) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD2D32;
  if (func) {
    return func(dstDevice, dstPitch, ui, Width, Height);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD8Async;
  if (func) {
    return func(dstDevice, uc, N, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD16Async;
  if (func) {
    return func(dstDevice, us, N, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD32Async;
  if (func) {
    return func(dstDevice, ui, N, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD2D8Async;
  if (func) {
    return func(dstDevice, dstPitch, uc, Width, Height, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD2D16Async;
  if (func) {
    return func(dstDevice, dstPitch, us, Width, Height, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemsetD2D32Async;
  if (func) {
    return func(dstDevice, dstPitch, ui, Width, Height, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuArrayCreate(CUarray* pHandle,
                               const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuArrayCreate;
  if (func) {
    return func(pHandle, pAllocateArray);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                      CUarray hArray) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuArrayGetDescriptor;
  if (func) {
    return func(pArrayDescriptor, hArray);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuArrayDestroy;
  if (func) {
    return func(hArray);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuArray3DCreate(
    CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuArray3DCreate;
  if (func) {
    return func(pHandle, pAllocateArray);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuArray3DGetDescriptor;
  if (func) {
    return func(pArrayDescriptor, hArray);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI
cuMipmappedArrayCreate(CUmipmappedArray* pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMipmappedArrayCreate;
  if (func) {
    return func(pHandle, pMipmappedArrayDesc, numMipmapLevels);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray* pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMipmappedArrayGetLevel;
  if (func) {
    return func(pLevelArray, hMipmappedArray, level);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMipmappedArrayDestroy;
  if (func) {
    return func(hMipmappedArray);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuPointerGetAttribute(void* data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuPointerGetAttribute;
  if (func) {
    return func(data, attribute, ptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuPointerSetAttribute(const void* value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuPointerSetAttribute;
  if (func) {
    return func(value, attribute, ptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamCreate;
  if (func) {
    return func(phStream, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* phStream,
                                            unsigned int flags, int priority) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamCreateWithPriority;
  if (func) {
    return func(phStream, flags, priority);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int* priority) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamGetPriority;
  if (func) {
    return func(hStream, priority);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamGetFlags;
  if (func) {
    return func(hStream, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamWaitEvent;
  if (func) {
    return func(hStream, hEvent, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void* userData,
                                     unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamAddCallback;
  if (func) {
    return func(hStream, callback, userData, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamAttachMemAsync;
  if (func) {
    return func(hStream, dptr, length, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamQuery;
  if (func) {
    return func(hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamSynchronize;
  if (func) {
    return func(hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamDestroy;
  if (func) {
    return func(hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuEventCreate;
  if (func) {
    return func(phEvent, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuEventRecord;
  if (func) {
    return func(hEvent, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuEventQuery;
  if (func) {
    return func(hEvent);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuEventSynchronize;
  if (func) {
    return func(hEvent);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuEventDestroy;
  if (func) {
    return func(hEvent);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuEventElapsedTime(float* pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuEventElapsedTime;
  if (func) {
    return func(pMilliseconds, hStart, hEnd);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuFuncGetAttribute(int* pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuFuncGetAttribute;
  if (func) {
    return func(pi, attrib, hfunc);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuFuncSetAttribute;
  if (func) {
    return func(hfunc, attrib, value);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuFuncSetCacheConfig;
  if (func) {
    return func(hfunc, config);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuFuncSetSharedMemConfig;
  if (func) {
    return func(hfunc, config);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLaunchKernel;
  if (func) {
    return func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuFuncSetBlockShape;
  if (func) {
    return func(hfunc, x, y, z);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuFuncSetSharedSize;
  if (func) {
    return func(hfunc, bytes);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuParamSetSize;
  if (func) {
    return func(hfunc, numbytes);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuParamSeti;
  if (func) {
    return func(hfunc, offset, value);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuParamSetf;
  if (func) {
    return func(hfunc, offset, value);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void* ptr,
                             unsigned int numbytes) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuParamSetv;
  if (func) {
    return func(hfunc, offset, ptr, numbytes);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLaunch(CUfunction f) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLaunch;
  if (func) {
    return func(f);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLaunchGrid;
  if (func) {
    return func(f, grid_width, grid_height);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width,
                                   int grid_height, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuLaunchGridAsync;
  if (func) {
    return func(f, grid_width, grid_height, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit,
                                  CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuParamSetTexRef;
  if (func) {
    return func(hfunc, texunit, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func_ = lib.m_cuOccupancyMaxActiveBlocksPerMultiprocessor;
  if (func_) {
    return func_(numBlocks, func, blockSize, dynamicSMemSize);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func_ = lib.m_cuOccupancyMaxPotentialBlockSize;
  if (func_) {
    return func_(minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                 dynamicSMemSize, blockSizeLimit);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetArray;
  if (func) {
    return func(hTexRef, hArray, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetMipmappedArray;
  if (func) {
    return func(hTexRef, hMipmappedArray, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetAddress;
  if (func) {
    return func(ByteOffset, hTexRef, dptr, bytes);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR* desc,
                                      CUdeviceptr dptr, size_t Pitch) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetAddress2D;
  if (func) {
    return func(hTexRef, desc, dptr, Pitch);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetFormat;
  if (func) {
    return func(hTexRef, fmt, NumPackedComponents);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetAddressMode;
  if (func) {
    return func(hTexRef, dim, am);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetFilterMode;
  if (func) {
    return func(hTexRef, fm);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetMipmapFilterMode;
  if (func) {
    return func(hTexRef, fm);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetMipmapLevelBias;
  if (func) {
    return func(hTexRef, bias);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetMipmapLevelClamp;
  if (func) {
    return func(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetMaxAnisotropy;
  if (func) {
    return func(hTexRef, maxAniso);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefSetFlags;
  if (func) {
    return func(hTexRef, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetAddress;
  if (func) {
    return func(pdptr, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetArray;
  if (func) {
    return func(phArray, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray,
                                           CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetMipmappedArray;
  if (func) {
    return func(phMipmappedArray, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef,
                                        int dim) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetAddressMode;
  if (func) {
    return func(pam, hTexRef, dim);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetFilterMode;
  if (func) {
    return func(pfm, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels,
                                   CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetFormat;
  if (func) {
    return func(pFormat, pNumChannels, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm,
                                             CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetMipmapFilterMode;
  if (func) {
    return func(pfm, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetMipmapLevelBias;
  if (func) {
    return func(pbias, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp,
                                             float* pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetMipmapLevelClamp;
  if (func) {
    return func(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetMaxAnisotropy;
  if (func) {
    return func(pmaxAniso, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefGetFlags;
  if (func) {
    return func(pFlags, hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefCreate(CUtexref* pTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefCreate;
  if (func) {
    return func(pTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexRefDestroy;
  if (func) {
    return func(hTexRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuSurfRefSetArray;
  if (func) {
    return func(hSurfRef, hArray, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuSurfRefGetArray;
  if (func) {
    return func(phArray, hSurfRef);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI
cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc,
                  const CUDA_TEXTURE_DESC* pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexObjectCreate;
  if (func) {
    return func(pTexObject, pResDesc, pTexDesc, pResViewDesc);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexObjectDestroy;
  if (func) {
    return func(texObject);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc,
                                            CUtexObject texObject) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexObjectGetResourceDesc;
  if (func) {
    return func(pResDesc, texObject);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc,
                                           CUtexObject texObject) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexObjectGetTextureDesc;
  if (func) {
    return func(pTexDesc, texObject);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuTexObjectGetResourceViewDesc;
  if (func) {
    return func(pResViewDesc, texObject);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject* pSurfObject,
                                    const CUDA_RESOURCE_DESC* pResDesc) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuSurfObjectCreate;
  if (func) {
    return func(pSurfObject, pResDesc);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuSurfObjectDestroy;
  if (func) {
    return func(surfObject);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc,
                                             CUsurfObject surfObject) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuSurfObjectGetResourceDesc;
  if (func) {
    return func(pResDesc, surfObject);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDeviceCanAccessPeer;
  if (func) {
    return func(canAccessPeer, dev, peerDev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxEnablePeerAccess;
  if (func) {
    return func(peerContext, Flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxDisablePeerAccess;
  if (func) {
    return func(peerContext);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsUnregisterResource;
  if (func) {
    return func(resource);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsSubResourceGetMappedArray;
  if (func) {
    return func(pArray, resource, arrayIndex, mipLevel);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsResourceGetMappedMipmappedArray;
  if (func) {
    return func(pMipmappedArray, resource);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsResourceGetMappedPointer;
  if (func) {
    return func(pDevPtr, pSize, resource);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsResourceSetMapFlags;
  if (func) {
    return func(resource, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource* resources,
                                        CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsMapResources;
  if (func) {
    return func(count, resources, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource* resources,
                                          CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGraphicsUnmapResources;
  if (func) {
    return func(count, resources, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuGetExportTable(const void** ppExportTable,
                                  const CUuuid* pExportTableId) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuGetExportTable;
  if (func) {
    return func(ppExportTable, pExportTableId);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDevicePrimaryCtxRetain;
  if (func) {
    return func(pctx, dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDevicePrimaryCtxRelease;
  if (func) {
    return func(dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDevicePrimaryCtxSetFlags;
  if (func) {
    return func(dev, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags,
                                            int* active) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDevicePrimaryCtxGetState;
  if (func) {
    return func(dev, flags, active);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuDevicePrimaryCtxReset;
  if (func) {
    return func(dev);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int* flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuCtxGetFlags;
  if (func) {
    return func(flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute* attributes,
                                        void** data, CUdeviceptr ptr) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuPointerGetAttributes;
  if (func) {
    return func(numAttributes, attributes, data, ptr);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemPrefetchAsync;
  if (func) {
    return func(devPtr, count, dstDevice, hStream);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemAdvise;
  if (func) {
    return func(devPtr, count, advice, device);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemRangeGetAttribute(void* data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemRangeGetAttribute;
  if (func) {
    return func(data, dataSize, attribute, devPtr, count);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuMemRangeGetAttributes(void** data, size_t* dataSizes,
                                         CUmem_range_attribute* attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuMemRangeGetAttributes;
  if (func) {
    return func(data, dataSizes, attributes, numAttributes, devPtr, count);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamWaitValue32;
  if (func) {
    return func(stream, addr, value, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamWriteValue32;
  if (func) {
    return func(stream, addr, value, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams* paramArray,
                                    unsigned int flags) {
  auto& lib = LibCUDAWrapper::getInstance();
  auto func = lib.m_cuStreamBatchMemOp;
  if (func) {
    return func(stream, count, paramArray, flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

#endif