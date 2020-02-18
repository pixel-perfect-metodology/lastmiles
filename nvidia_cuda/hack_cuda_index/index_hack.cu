/**
 * hacked at by Dennis Clarke for some twitch fun stuff
 * okay hacked a LOT.
 *
 * Lets get system info and memory reports for local host
 * and then use the system clock to init the drand prng pool.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <locale.h>
#include <math.h>
#include <errno.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include <omp.h>

#define NUM_ELEMENTS 1024
#define THREADS_PER_BLOCK 1024

int sysinfo(void);
uint64_t system_memory();
uint64_t timediff( struct timespec st, struct timespec en );

/**
 * CUDA Kernel Device code
 */
__global__ void
vector_idx_dump(uint64_t *C, int num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = ( blockDim.x << 20 ) + ( blockIdx.x << 10 ) + threadIdx.x;
}

int main(int argc, char *argv[])
{
    struct timespec t0, t1;
    uint64_t tdelta_nsec;

    cudaError_t err = cudaSuccess;
    int num_elements = NUM_ELEMENTS;
    size_t size = num_elements * sizeof(uint64_t);

    /* do we even have a NVidia Quadro GPU ? */
    int num_gpus = 0;

    setlocale( LC_ALL, "C" );
    sysinfo();

    /* Get the CLOCK_REALTIME time in a timespec struct */
    if ( clock_gettime( CLOCK_REALTIME, &t0 ) == -1 ) {
        /* We could not get the clock. Bail out. */
        fprintf(stderr,"ERROR : could not attain CLOCK_REALTIME\n");
        return(EXIT_FAILURE);
    }

    /* determine the number of CUDA capable GPUs */
    cudaGetDeviceCount(&num_gpus);
    if ( num_gpus < 1 )
    {
        printf("INFO : no CUDA capable devices were detected\n");
        return EXIT_FAILURE;
    }

    /* display CPU and GPU configuration */
    printf("INFO : number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("INFO : number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("     :    %d: %s\n", i, dprop.name);
    }

    printf("INFO : firing off %i cuda core code chunks\n", num_elements);
    printf("     : Memory size of array is %ld bytes\n", size );

    uint64_t *h_C = (uint64_t *)malloc(size);

    if ( h_C == NULL )
    {
        fprintf(stderr, "FAIL : memory allocate\n");
        exit(EXIT_FAILURE);
    }

    uint64_t *d_C = NULL;
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA failed to allocate vector C\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : time to here with cudaMalloc %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    /* fire off the cuda "kernel" code blocks */
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid =( num_elements + threadsPerBlock - 1 ) / threadsPerBlock;
    printf("INFO : CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vector_idx_dump<<<blocksPerGrid, threadsPerBlock>>>(d_C, num_elements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : CUDA failed\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : done.\n");
    clock_gettime( CLOCK_REALTIME, &t0 );
    tdelta_nsec = timediff( t1, t0);
    printf("     : time seems to be %" PRIu64 " nsecs  %9.7g secs\n",
                            tdelta_nsec, (float)tdelta_nsec/1.0e9);

    /* Copy the device result memory d_C to the host result h_C memory */
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : CUDA failed to copy result vector C from device to host\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy result C from device to host done.\n");

    /* for all of the data what was the values in C ? */
    uint64_t foo, bar;
    for (int i = 0; i < num_elements; ++i)
    {
        /* Lets undo this hackary */
        printf("%-10i    ", i);

        foo = h_C[i] >> 20;
        printf("blockDim.x = %" PRIu64 "    ", foo );

        bar = ( ( h_C[i] >> 10 ) & 0x03ff );
        printf("blockIdx.x = %" PRIu64 "    ", bar );

        printf("threadIdx.x = %" PRIu64 "\n", h_C[i] & 0x03ff );

    }

    /* Free device memory */
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : Failed to free memory on device for vector C\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Free host memory */
    free(h_C);

    printf("INFO : host memory free and we are done\n");
    cudaProfilerStop();
    return EXIT_SUCCESS;

}

uint64_t timediff( struct timespec st, struct timespec en )
{
    /* return the delta time as a 64-bit positive number of
     * nanoseconds.  Regardless of the time direction between
     * start and end we always get a positive result. */

    struct timespec temp;
    uint64_t s, n;

    if ( ( en.tv_nsec - st.tv_nsec ) < 0 ) {
        /* make a full second adjustment to tv_sec */
        temp.tv_sec = en.tv_sec - st.tv_sec - 1;
        /* we have to add a full second to temp.tv_nsec */
        temp.tv_nsec = 1000000000 + en.tv_nsec - st.tv_nsec;
    } else {
        temp.tv_sec = en.tv_sec - st.tv_sec;
        temp.tv_nsec = en.tv_nsec - st.tv_nsec;
    }
    s = (uint64_t) temp.tv_sec;
    n = (uint64_t) temp.tv_nsec;
    return ( s * (uint64_t)1000000000 + n );
}

uint64_t system_memory()
{
    /* should return the amount of memory available in bytes */
    long en;
    uint64_t pages, page_size;

    en = sysconf(_SC_PHYS_PAGES);
    if ( en < 0 ){
        perror("sysconf(_SC_PHYS_PAGES) : ");
        exit(EXIT_FAILURE);
    }
    pages = (uint64_t) en;

    page_size = (uint64_t)sysconf(_SC_PAGE_SIZE);
    return ( pages * page_size );
}

int sysinfo(void) {

    struct utsname uname_data;

    uint64_t sysmem = system_memory();
    uint64_t pagesize = (uint64_t)sysconf(_SC_PAGESIZE);

    setlocale( LC_MESSAGES, "C" );
    if ( uname( &uname_data ) < 0 ) {
        fprintf ( stderr,
                 "WARNING : Could not attain system uname data.\n" );
        perror ( "uname" );
    } else {
        printf ( "-------------------------------" );
        printf ( "------------------------------\n" );
        printf ( "        system name = %s\n", uname_data.sysname );
        printf ( "          node name = %s\n", uname_data.nodename );
        printf ( "            release = %s\n", uname_data.release );
        printf ( "            version = %s\n", uname_data.version );
        printf ( "            machine = %s\n", uname_data.machine );
        printf ( "          page size = %" PRIu64 "\n", pagesize );
        printf ( "       avail memory = %" PRIu64 "\n", sysmem );
        printf ( "                    = %" PRIu64 " kB\n", sysmem/1024 );
        printf ( "                    = %" PRIu64 " MB\n", sysmem/1048576 );
        /*
         *  this doesn't really work for memory size near GB boundaries
         *
         *  if ( sysmem > ( 1024 * 1048576 ) ) {
         *      printf ( "                    = %" PRIu64 " GB\n",
         *              sysmem/( 1024 * 1048576 ) );
         *  }
        */
        printf ( "-------------------------------" );
        printf ( "------------------------------" );
    }
    printf ("\n");

    return ( EXIT_SUCCESS );

}

