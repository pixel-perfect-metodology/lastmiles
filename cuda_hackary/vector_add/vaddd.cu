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
#include <errno.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include <omp.h>

#include "dat.h"

int sysinfo(void);
uint64_t system_memory();
uint64_t timediff( struct timespec st, struct timespec en );

/**
 * CUDA Kernel Device code
 */
__global__ void
vector_add(const double *A, const double *B, double *C, int num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < num_elements )
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    struct timespec t0, t1;
    uint64_t tdelta_nsec;

    cudaError_t err = cudaSuccess;
    int num_elements = NUM_ELEMENTS;
    size_t size = num_elements * sizeof(double);

    /* do we even have a NVidia Quadro GPU ? */
    int num_gpus = 0;

    setlocale( LC_ALL, "C" );
    sysinfo();

    /* Get the CLOCK_REALTIME time in a timespec struct */
    if ( clock_gettime( CLOCK_REALTIME, &t0 ) == -1 ) {
        /* We could not get the clock. Bail out. */
        fprintf(stderr,"ERROR : could not attain CLOCK_REALTIME\n");
        return(EXIT_FAILURE);
    } else {
        /* call srand48() with the sub-second time data */
        srand48( (long) t0.tv_nsec );
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

    printf("INFO : Vector addition of %d double FP64 elements\n", num_elements);
    printf("     : Memory size of each array is %ld bytes\n", size );

    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "FAIL : memory allocate\n");
        exit(EXIT_FAILURE);
    }

    /* fill the arrays A and B with random data */
    clock_gettime( CLOCK_REALTIME, &t0 );
    for (int i = 0; i < num_elements; ++i)
    {
        h_A[i] = drand48();
        h_B[i] = drand48();
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : random data loaded %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);

    /* Allocate the device input memory */
    double *d_A = NULL;
    if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA failed to allocate vector A\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t0 );
    tdelta_nsec = timediff( t1, t0);
    printf("     : cudaMalloc(A) %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    double *d_B = NULL;
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA failed to allocate vector B\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMalloc(B) %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    double *d_C = NULL;
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA failed to allocate vector C\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t0 );
    tdelta_nsec = timediff( t1, t0);
    printf("     : cudaMalloc(C) %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    /* Copy the host input A and B arrays in host memory
     * to the device memory */
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA failed to copy vector A from host to device\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy of vector A from host to device done.\n");
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMemcpy() %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA failed to copy vector B from host to device\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy of vector B from host to device done.\n");
    clock_gettime( CLOCK_REALTIME, &t0 );
    tdelta_nsec = timediff( t1, t0);
    printf("     : cudaMemcpy() %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    /* fire off the cuda "kernel" code blocks */
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid =( num_elements + threadsPerBlock - 1 ) / threadsPerBlock;
    printf("INFO : CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : CUDA failed vectorAdd\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : vector_add done.\n");
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : vector_add %" PRIu64 " nsecs  %9.7g secs\n",
                            tdelta_nsec, (float)tdelta_nsec/1.0e9);

    /* Copy the device result memory d_C to the host result h_C memory */
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : CUDA failed to copy result vector C from device to host\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy result A from device to host done.\n");
    clock_gettime( CLOCK_REALTIME, &t0 );
    tdelta_nsec = timediff( t1, t0 );
    printf("     : cudaMemcpy() %" PRIu64 " nsecs  %9.7g secs\n",
                               tdelta_nsec, (float)tdelta_nsec/1.0e9);


    /* test that result vector is correct within epsilon error */
    for (int i = 0; i < num_elements; ++i)
    {
        if ( fabs(h_A[i] + h_B[i] - h_C[i]) > EPSILON )
        {
            fprintf(stderr, "FAIL : Result verification failed at element %d!\n", i);
            /* TODO : maybe not be a prick and free() all the local host
               memory and the device memory? Ya think? */
            exit(EXIT_FAILURE);
        }
    }

    printf("INFO : A + B correct within error epsilon = %16.9e\n", EPSILON);
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : epsilon %" PRIu64 " nsecs  %9.7g secs\n",
                          tdelta_nsec, (float)tdelta_nsec/1.0e9);


    /* Free device memory */
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : Failed to free memory on device for vector A\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : Failed to free memory on device for vector B\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : Failed to free memory on device for vector C\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Free host memory */
    free(h_A);
    free(h_B);
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

