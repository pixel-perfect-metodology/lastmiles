
/* NVIDIA Performance Primitives (NPP) 
 *
 * https://docs.nvidia.com/cuda/npp/group__image__fourier__transforms.html
 *
 * may be useful to try :
 *  g++ foo.c  -lnppi_static -lculibos -lcudart_static -lpthread -ldl 
 *   -I <cuda-toolkit-path>/include -L <cuda-toolkit-path>/lib64 -o foo
 *
 *   This was hacked together by Dennis Clarke with reference material
 *   here there and everywhere. It is still not complete and no one
 *   is happy with the code format.
 *
 *   Lines are too darn long.
 *
 */

#include <memory>
#include <iostream>
#include <npp.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

int main(int argc, char **argv)
{
    int dev;
    int dev_count = 0;
    int driver_ver = 0;
    int runtime_ver = 0;

    const char *compute_mode_type_string[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };

    cudaError_t cuda_err_status;
    cudaDeviceProp dev_prop;

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

    cuda_err_status = cudaGetDeviceCount(&dev_count);
    if (cuda_err_status != cudaSuccess)
    {
        fprintf(stderr,"FAIL : cudaGetDeviceCount returned %d\n-> %s\n",
                           (int)cuda_err_status,
                           cudaGetErrorString(cuda_err_status));

        exit(EXIT_FAILURE);
    }

    /* returns 0 if there are no CUDA capable devices */
    if (dev_count == 0)
    {
        fprintf(stderr,"WARN : no CUDA device found.\n");
        exit(EXIT_FAILURE);
    } else {
        printf("INFO : Detected %d CUDA Capable device(s)\n", dev_count);
    }

    /*****************************************************************
     *
     *  Pulled this directly out of the man page and we may as well
     *  dig into the struct for all its worth.
     *
     *  __cudart_builtin__ cudaError_t cudaGetDeviceProperties
     *                       (struct cudaDeviceProp * prop,
     *                       int device)
     *
     *  Returns:
     *      cudaSuccess, cudaErrorInvalidDevice
     *
     *  if cudaSuccess then *prop contains the properties
     *  of device dev.
     *
     *  the cudaDeviceProp structure is defined as:
     *
     *     struct cudaDeviceProp {
     *         char name[256];
     *         size_t totalGlobalMem;
     *         size_t sharedMemPerBlock;
     *         int regsPerBlock;
     *         int warpSize;
     *         size_t memPitch;
     *         int maxThreadsPerBlock;
     *         int maxThreadsDim[3];
     *         int maxGridSize[3];
     *         int clockRate;
     *         size_t totalConstMem;
     *         int major;
     *         int minor;
     *         size_t textureAlignment;
     *         size_t texturePitchAlignment;
     *         int deviceOverlap;
     *         int multiProcessorCount;
     *         int kernelExecTimeoutEnabled;
     *         int integrated;
     *         int canMapHostMemory;
     *         int computeMode;
     *         int maxTexture1D;
     *         int maxTexture1DMipmap;
     *         int maxTexture1DLinear;
     *         int maxTexture2D[2];
     *         int maxTexture2DMipmap[2];
     *         int maxTexture2DLinear[3];
     *         int maxTexture2DGather[2];
     *         int maxTexture3D[3];
     *         int maxTexture3DAlt[3];
     *         int maxTextureCubemap;
     *         int maxTexture1DLayered[2];
     *         int maxTexture2DLayered[3];
     *         int maxTextureCubemapLayered[2];
     *         int maxSurface1D;
     *         int maxSurface2D[2];
     *         int maxSurface3D[3];
     *         int maxSurface1DLayered[2];
     *         int maxSurface2DLayered[3];
     *         int maxSurfaceCubemap;
     *         int maxSurfaceCubemapLayered[2];
     *         size_t surfaceAlignment;
     *         int concurrentKernels;
     *         int ECCEnabled;
     *         int pciBusID;
     *         int pciDeviceID;
     *         int pciDomainID;
     *         int tccDriver;
     *         int asyncEngineCount;
     *         int unifiedAddressing;
     *         int memoryClockRate;
     *         int memoryBusWidth;
     *         int l2CacheSize;
     *         int maxThreadsPerMultiProcessor;
     *         int streamPrioritiesSupported;
     *         int globalL1CacheSupported;
     *         int localL1CacheSupported;
     *         size_t sharedMemPerMultiprocessor;
     *         int regsPerMultiprocessor;
     *         int managedMemory;
     *         int isMultiGpuBoard;
     *         int multiGpuBoardGroupID;
     *         int singleToDoublePrecisionPerfRatio;
     *         int pageableMemoryAccess;
     *         int concurrentManagedAccess;
     *         int computePreemptionSupported;
     *         int canUseHostPointerForRegisteredMem;
     *         int cooperativeLaunch;
     *         int cooperativeMultiDeviceLaunch;
     *         int pageableMemoryAccessUsesHostPageTables;
     *         int directManagedMemAccessFromHost;
     *     }
     * 
     *  Some of the above may even be true.
     *
     ****************************************************************/

    for (dev = 0; dev < dev_count; ++dev) {
        char buff[256];

        cudaSetDevice(dev);
        cudaGetDeviceProperties(&dev_prop, dev);

        printf("INFO : dev number %d: name = \"%s\"\n",
                                                   dev, dev_prop.name);

        cudaDriverGetVersion(&driver_ver);
        cudaRuntimeGetVersion(&runtime_ver);

        printf("     : CUDA Driver Version    %d.%d\n",
                                 driver_ver/1000, (driver_ver%100)/10);

        printf("     : CUDA Runtime Version   %d.%d\n",
                               runtime_ver/1000, (runtime_ver%100)/10);

        printf("     : CUDA Capability Major/Minor version %d.%d\n",
                                       dev_prop.major, dev_prop.minor);


        printf("     : Total global memory: %llu bytes\n",
                         (unsigned long long) dev_prop.totalGlobalMem);

        printf("     : constant memory: %llu bytes\n",
                                               dev_prop.totalConstMem);

        printf("     : multiprocessor count = %i\n",
                                         dev_prop.multiProcessorCount);

        printf("     : CUDA cores per multiprocessor = %d\n",
                  _ConvertSMVer2Cores(dev_prop.major, dev_prop.minor));

        printf("     : total CUDA cores = %d\n",
                  ( _ConvertSMVer2Cores(dev_prop.major, dev_prop.minor)
                    * dev_prop.multiProcessorCount ) );

        printf("     : GPU max graphics clock rate: %i kHz\n",
                                                  dev_prop.clockRate );

        printf("     : Memory Clock rate: %i khz\n",
                                    dev_prop.memoryClockRate );

        printf("     : Memory Bus Width: %i-bit\n",
                                              dev_prop.memoryBusWidth);

        if (dev_prop.l2CacheSize){
            printf("     : L2 Cache Size: %i bytes\n",
                                                 dev_prop.l2CacheSize);
        }

        printf("     : Maximum Texture Dimension Size (x,y,z)\n");
        printf("     :     1D=(%d)\n", dev_prop.maxTexture1D );

        printf("     :     2D=(%d, %d)\n",
                                              dev_prop.maxTexture2D[0],
                                             dev_prop.maxTexture2D[1]);

        printf("     :     3D=(%d, %d, %d)\n",
                                              dev_prop.maxTexture3D[0],
                                              dev_prop.maxTexture3D[1],
                                             dev_prop.maxTexture3D[2]);

        printf("     : Maximum Layered 1D Texture Size,");
        printf(" (num) layers  1D=(%d), %d layers\n",
                                       dev_prop.maxTexture1DLayered[0],
                                      dev_prop.maxTexture1DLayered[1]);

        printf("     : Maximum Layered 2D Texture Size,");
        printf(" (num) layers  2D=(%d, %d), %d layers\n",
                                       dev_prop.maxTexture2DLayered[0],
                                       dev_prop.maxTexture2DLayered[1],
                                      dev_prop.maxTexture2DLayered[2]);

        printf("     : shared memory per block: %lu bytes\n",
                                           dev_prop.sharedMemPerBlock);

        printf("     : registers available per block: %d\n",
                                                dev_prop.regsPerBlock);

        printf("     : warp size: %d\n", dev_prop.warpSize);

        printf("     : max num threads per multiprocessor: %d\n",
                                 dev_prop.maxThreadsPerMultiProcessor);

        printf("     : max numb of threads per block: %d\n",
                                          dev_prop.maxThreadsPerBlock);

        printf("     : max dimension size of a thread block (x,y,z)");
        printf(" = (%d, %d, %d)\n", dev_prop.maxThreadsDim[0],
                                        dev_prop.maxThreadsDim[1],
                                            dev_prop.maxThreadsDim[2]);

        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               dev_prop.maxGridSize[0],
               dev_prop.maxGridSize[1],
               dev_prop.maxGridSize[2]);

        printf("  Maximum memory pitch:                          %lu bytes\n", dev_prop.memPitch);

        printf("  Texture alignment:                             %lu bytes\n", dev_prop.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
                (dev_prop.deviceOverlap ? "Yes" : "No"), dev_prop.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", dev_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", dev_prop.integrated ? "Yes" : "No");
        printf("   : host page-locked memory mapping: %s\n", dev_prop.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", dev_prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Device supports Unified Addressing (UVA):      %s\n", dev_prop.unifiedAddressing ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", dev_prop.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n", dev_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
                dev_prop.pciDomainID, dev_prop.pciBusID, dev_prop.pciDeviceID);

        const char *compute_mode_type_string[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", compute_mode_type_string[dev_prop.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (dev_count >= 2)
    {
        cudaDeviceProp prop[64];
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;

        for (int i=0; i < dev_count; i++)
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
    sprintf_s(cTemp, 10, "%d.%d", driver_ver/1000, (driver_ver%100)/10);
#else
    sprintf(cTemp, "%d.%d", driver_ver/1000, (driver_ver%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtime_ver/1000, (runtime_ver%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtime_ver/1000, (runtime_ver%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", dev_count);
#else
    sprintf(cTemp, "%d", dev_count);
#endif
    sProfileString += cTemp;
    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");

    // finish
    exit(EXIT_SUCCESS);
}
