
/* NVIDIA Performance Primitives (NPP) 
 *
 * https://docs.nvidia.com/cuda/npp/group__image__fourier__transforms.html
 *
 * may be useful to try :
 *  g++ foo.c  -lnppi_static -lculibos -lcudart_static -lpthread -ldl 
 *   -I <cuda-toolkit-path>/include -L <cuda-toolkit-path>/lib64 -o foo
 */

#include <memory>
#include <iostream>
#include <npp.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

int main(int argc, char **argv)
{
    int dev;
    int deviceCount = 0;
    int driverVersion = 0;
    int runtimeVersion = 0;

    cudaError_t cuda_err_status;

    printf("\n *** CUDA Device Query (Runtime API) ***\n");

    #ifdef CUDART_VERSION
        printf("INFO : CUDART_VERSION defined as %d\n", CUDART_VERSION);
        if ( CUDART_VERSION < 9020 ){
            /* old CUDA runtime detected.
             * Bail out. getCudaAttribute not defined. */
            fprintf(stderr,"FAIL : need CUDART_VERSION at least 9020\n");
            return ( EXIT_FAILURE );
        }
        printf("INFO : we may be using CUDART static linking\n");
    #else
        fprintf(stderr,"FAIL : CUDART_VERSION is not defined?\n");
        return ( EXIT_FAILURE );
    #endif

    #ifdef NPP_VERSION_MAJOR
        printf("INFO : NPP_VERSION_MAJOR defined as %d\n", NPP_VERSION_MAJOR);
        #ifdef NPP_VERSION_MINOR
            fprintf(stderr,"INFO : NPP_VERSION_MINOR defined as %d\n", NPP_VERSION_MINOR);
            /* do a quick check for the npp ver
            */
            if ( ( (NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4) ) >= 0x6000 ){
                fprintf(stderr,"     : (NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4) >= 0x6000\n");
            }
        #endif
        fprintf(stderr,"     : signal and image processing functions are available.\n");
    #else
        printf("INFO : NPP_VERSION_MAJOR is not defined.\n");
    #endif

    cuda_err_status = cudaGetDeviceCount(&deviceCount);
    if (cuda_err_status != cudaSuccess)
    {
        fprintf(stderr,"FAIL : cudaGetDeviceCount returned %d\n-> %s\n",
                           (int)cuda_err_status, cudaGetErrorString(cuda_err_status));
        exit(EXIT_FAILURE);
    }

    /* returns 0 if there are no CUDA capable devices */
    if (deviceCount == 0)
    {
        fprintf(stderr,"WARN : no available device that supports CUDA\n");
        fprintf(stderr,"     : sorry. quitting gracefully.\n");
        exit(EXIT_FAILURE);
    } else {
        printf("INFO : Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    /*****************************************************************

      __cudart_builtin__ cudaError_t cudaGetDeviceProperties
                            (struct cudaDeviceProp * prop, int device)

      Returns:
          cudaSuccess, cudaErrorInvalidDevice

      if cudaSuccess then *prop contains the properties of device dev.

      the cudaDeviceProp structure is defined as:

          struct cudaDeviceProp {
              char name[256];
              size_t totalGlobalMem;
              size_t sharedMemPerBlock;
              int regsPerBlock;
              int warpSize;
              size_t memPitch;
              int maxThreadsPerBlock;
              int maxThreadsDim[3];
              int maxGridSize[3];
              int clockRate;
              size_t totalConstMem;
              int major;
              int minor;
              size_t textureAlignment;
              size_t texturePitchAlignment;
              int deviceOverlap;
              int multiProcessorCount;
              int kernelExecTimeoutEnabled;
              int integrated;
              int canMapHostMemory;
              int computeMode;
              int maxTexture1D;
              int maxTexture1DMipmap;
              int maxTexture1DLinear;
              int maxTexture2D[2];
              int maxTexture2DMipmap[2];
              int maxTexture2DLinear[3];
              int maxTexture2DGather[2];
              int maxTexture3D[3];
              int maxTexture3DAlt[3];
              int maxTextureCubemap;
              int maxTexture1DLayered[2];
              int maxTexture2DLayered[3];
              int maxTextureCubemapLayered[2];
              int maxSurface1D;
              int maxSurface2D[2];
              int maxSurface3D[3];
              int maxSurface1DLayered[2];
              int maxSurface2DLayered[3];
              int maxSurfaceCubemap;
              int maxSurfaceCubemapLayered[2];
              size_t surfaceAlignment;
              int concurrentKernels;
              int ECCEnabled;
              int pciBusID;
              int pciDeviceID;
              int pciDomainID;
              int tccDriver;
              int asyncEngineCount;
              int unifiedAddressing;
              int memoryClockRate;
              int memoryBusWidth;
              int l2CacheSize;
              int maxThreadsPerMultiProcessor;
              int streamPrioritiesSupported;
              int globalL1CacheSupported;
              int localL1CacheSupported;
              size_t sharedMemPerMultiprocessor;
              int regsPerMultiprocessor;
              int managedMemory;
              int isMultiGpuBoard;
              int multiGpuBoardGroupID;
              int singleToDoublePrecisionPerfRatio;
              int pageableMemoryAccess;
              int concurrentManagedAccess;
              int computePreemptionSupported;
              int canUseHostPointerForRegisteredMem;
              int cooperativeLaunch;
              int cooperativeMultiDeviceLaunch;
              int pageableMemoryAccessUsesHostPageTables;
              int directManagedMemAccessFromHost;
          }
    *******************************************************************/

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        char buff[256];

        cudaSetDevice(dev);
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("INFO : dev number %d: name = \"%s\"\n", dev, deviceProp.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("     : CUDA Driver Version    %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
        printf("     : CUDA Runtime Version   %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("     : CUDA Capability Major/Minor version %d.%d\n", deviceProp.major, deviceProp.minor);

        sprintf(buff,
                "     : Total global memory: %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f,
                (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", buff);

        printf("     : (%2d) Multiprocessors, (%3d) CUDA Cores/MP: %d CUDA Cores\n",
                   deviceProp.multiProcessorCount,

                   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),

                   (
                      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
                      * deviceProp.multiProcessorCount
                   )
              );

        printf("     : GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

        printf("     : Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("     : Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize){
            printf("     : L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
        }

        printf("     : Maximum Texture Dimension Size (x,y,z)\n");
        printf("     :     1D=(%d)\n", deviceProp.maxTexture1D );

        printf("     :     2D=(%d, %d)\n",
                                       deviceProp.maxTexture2D[0],
                                       deviceProp.maxTexture2D[1]);

        printf("     :     3D=(%d, %d, %d)\n",
                                       deviceProp.maxTexture3D[0],
                                       deviceProp.maxTexture3D[1],
                                       deviceProp.maxTexture3D[2]);

        printf("     : Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0],
               deviceProp.maxTexture1DLayered[1]);

        printf("     : Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0],
               deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);


        printf("     : constant memory: %lu bytes\n", deviceProp.totalConstMem);

        printf("     : shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);

        printf("     : registers available per block: %d\n", deviceProp.regsPerBlock);

        printf("     : warp size: %d\n", deviceProp.warpSize);

        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);

        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
                (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n", deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
                deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2)
    {
        cudaDeviceProp prop[64];
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;

        for (int i=0; i < deviceCount; i++)
        {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to support this
                && prop[i].tccDriver
#endif
               )
            {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2)
        {
            for (int i = 0; i < gpu_p2p_count; i++)
            {
                for (int j = 0; j < gpu_p2p_count; j++)
                {
                    if (gpuid[i] == gpuid[j])
                    {
                        continue;
                    }
                    checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid[i]].name, gpuid[i],
                           prop[gpuid[j]].name, gpuid[j] ,
                           can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;
    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");

    // finish
    exit(EXIT_SUCCESS);
}
