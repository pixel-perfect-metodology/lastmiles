/**
 * hacked at by Dennis Clarke for some twitch fun stuff
 * okay hacked a LOT.  Then a lot more. Currently we may
 * have a problem with needing cudaDeviceSynchronize over
 * and over. Maybe. Also perhaps a device reset may be a
 * need right up front. Not sure yet.  20200516094734
 *
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
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

#define NUM_ELEMENTS 16777216
#define THREADS_PER_BLOCK 2048
#define BAIL_OUT 4096

int sysinfo(void);
uint64_t system_memory();
uint64_t timediff( struct timespec st, struct timespec en );
uint32_t cpu_mbrot( double c_r, double c_i, uint32_t bail_out );

/**
 * CUDA Kernel Device code
 */
__global__ void
gpu_mbrot( const double *c_r, const double *c_i,
       uint32_t *mval, int num_elements )
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < num_elements ) {

        /* point c belongs to the Mandelbrot set if and only if
         * the magnitude of the f(c) <= 2.0 */
        uint32_t height = 0;
        double zr = 0.0;
        double zi = 0.0;
        double tmp_r, tmp_i;
        double mag = 0.0;

        while ( ( height < BAIL_OUT ) && ( mag < 4.0 ) ) {
            tmp_r = ( zr * zr ) - ( zi * zi );
            tmp_i = ( zr * zi ) + ( zr * zi );
            zr = tmp_r + c_r[i];
            zi = tmp_i + c_i[i];
            mag = zr * zr + zi * zi;
            height += 1;
        }

        mval[i] = height;

    }

}

int main(int argc, char *argv[])
{
    struct timespec t0, t1;
    uint64_t tdelta_nsec;

    cudaError_t err = cudaSuccess;
    int num_elements = NUM_ELEMENTS;
    size_t size_coord = num_elements * sizeof(double);
    size_t size_height = num_elements * sizeof(uint32_t);

    double *host_r, *host_i, *device_r, *device_i;
    uint32_t *host_mval, *device_mval, check_val;

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
        srand48((long int)t0.tv_nsec);
        /* srand48(((long int)123456789)); */
    }

    /* determine the number of CUDA capable GPUs */
    cudaGetDeviceCount(&num_gpus);
    if ( num_gpus < 1 ) {
        printf("INFO : no CUDA capable devices were detected\n");
        return EXIT_FAILURE;
    }

    /* display CPU and GPU configuration */
    printf("INFO : number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("INFO : number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("     :    %d: %s\n", i, dprop.name);
    }

    printf("INFO : firing off %i cuda core code chunks\n",
                                                         num_elements);

    printf("     : mem size of double data array is %ld bytes\n",
                                                          size_coord );

    printf("     : mem size of uint32_t height array is %ld bytes\n",
                                                         size_height );

    host_r = (double *)malloc(size_coord);
    host_i = (double *)malloc(size_coord);
    host_mval = (uint32_t *)calloc(num_elements,sizeof(uint32_t));

    if ( ( host_r == NULL )
            || ( host_i == NULL )
            || ( host_mval == NULL ) ) {
        fprintf(stderr, "FAIL : memory allocate\n");
        exit(EXIT_FAILURE);
    }

    /* fill the arrays host real and imaginary with random double
     * floating point values within the range (-2.0, +2.0) */
    clock_gettime( CLOCK_REALTIME, &t0 );
    for (int i = 0; i < num_elements; ++i) {
        host_r[i] = -2.0 + 4.0 * drand48();
        host_i[i] = -2.0 + 4.0 * drand48();
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : random data load %" PRIu64 " nsecs\n", tdelta_nsec);

    clock_gettime( CLOCK_REALTIME, &t0 );
    device_r = NULL;
    if (cudaMalloc((void **)&device_r, size_coord) != cudaSuccess) {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA fail allocate real array\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMalloc device_r %" PRIu64 " nsecs\n",
                                                         tdelta_nsec);


    clock_gettime( CLOCK_REALTIME, &t0 );
    device_i = NULL;
    if (cudaMalloc((void **)&device_i, size_coord) != cudaSuccess) {
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA fail allocate imaginary array\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMalloc device_i %" PRIu64 " nsecs\n",
                                                          tdelta_nsec);

    fprintf( stderr,"DBUG : at %d in %s\n", __LINE__, __FILE__);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed cudaDeviceSynchronize()\n");
        fprintf(stderr, "err = %0x\n", err );
        exit(EXIT_FAILURE);
    }

    clock_gettime( CLOCK_REALTIME, &t0 );
    device_mval = NULL;
    if (cudaMalloc((void **)&device_mval, size_height) != cudaSuccess){
        err = cudaGetLastError();
        fprintf(stderr, "FAIL : CUDA fail allocate height array\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMalloc device_mval %" PRIu64 " nsecs\n",
                                                          tdelta_nsec);


    fprintf( stderr,"DBUG : at %d in %s\n", __LINE__, __FILE__);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed cudaDeviceSynchronize()\n");
        fprintf(stderr, "err = %0x\n", err );
        exit(EXIT_FAILURE);
    }

    /* Copy the host input arrays in host memory
     * to the device memory */
    clock_gettime( CLOCK_REALTIME, &t0 );
    /* possible stuff that can happen with cudaMemcpy()
     *
     *   cudaErrorInvalidValue
     *   cudaErrorInvalidDevicePointer
     *   cudaErrorInvalidMemcpyDirection
     */
    err = cudaMemcpy(device_r,host_r,size_coord,cudaMemcpyHostToDevice);
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed memcopy host to device\n");
        fprintf(stderr, "FAIL : error ");

        if ( err == cudaErrorInvalidValue ) {
            fprintf(stderr, "cudaErrorInvalidValue\n");
        } else if ( err == cudaErrorInvalidDevicePointer ) {
            fprintf(stderr, "cudaErrorInvalidDevicePointer\n");
        } else if ( err == cudaErrorInvalidMemcpyDirection ) {
            fprintf(stderr, "cudaErrorInvalidMemcpyDirection\n");
        } else {
            fprintf(stderr, "what the absolute fuck?\n");
            fprintf(stderr, "err = %0x\n", err );
        }
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy of real data from host to device done.\n");
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMemcpy() %" PRIu64 "nsecs\n", tdelta_nsec);

    fprintf( stderr,"DBUG : at %d in %s\n", __LINE__, __FILE__);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed cudaDeviceSynchronize()\n");
        fprintf(stderr, "err = %0x\n", err );
        exit(EXIT_FAILURE);
    }

    clock_gettime( CLOCK_REALTIME, &t0 );

    err = cudaMemcpy(device_i,host_i,size_coord,cudaMemcpyHostToDevice);
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed memcopy host to device\n");
        fprintf(stderr, "FAIL : error ");

        if ( err == cudaErrorInvalidValue ) {
            fprintf(stderr, "cudaErrorInvalidValue\n");
        } else if ( err == cudaErrorInvalidDevicePointer ) {
            fprintf(stderr, "cudaErrorInvalidDevicePointer\n");
        } else if ( err == cudaErrorInvalidMemcpyDirection ) {
            fprintf(stderr, "cudaErrorInvalidMemcpyDirection\n");
        } else {
            fprintf(stderr, "what the absolute fuck?\n");
            fprintf(stderr, "err = %0x\n", err );
        }
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy of imaginary data from host to device done.\n");
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : cudaMemcpy() %" PRIu64 " nsecs\n", tdelta_nsec);


    /* fire off the cuda "kernel" code blocks */
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = ( num_elements + threadsPerBlock - 1 )
                        / threadsPerBlock;

    printf("INFO : CUDA kernel launch with %d blocks of %d threads\n",
                                       blocksPerGrid, threadsPerBlock);

    fprintf( stderr,"DBUG : at %d in %s\n", __LINE__, __FILE__);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed cudaDeviceSynchronize()\n");
        fprintf(stderr, "err = %0x\n", err );
        exit(EXIT_FAILURE);
    }

    clock_gettime( CLOCK_REALTIME, &t0 );

    /* the NVidia CUDA code format is a bit special */
    gpu_mbrot<<<blocksPerGrid, threadsPerBlock>>>( device_r, device_i, device_mval, num_elements );

    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
        fprintf(stderr, "FAIL : CUDA failed\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        fprintf(stderr, "FAIL : err = %0x\n", err );
        exit(EXIT_FAILURE);
    }
    printf("INFO : done.\n");
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : gpu_mbrot time delta %" PRIu64 " nsecs\n", tdelta_nsec);

    fprintf( stderr,"DBUG : at %d in %s\n", __LINE__, __FILE__);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed cudaDeviceSynchronize()\n");
        fprintf(stderr, "err = %0x\n", err );
        exit(EXIT_FAILURE);
    }

    /* Copy the device result memory to the host result memory */
    clock_gettime( CLOCK_REALTIME, &t0 );

    err = cudaMemcpy(host_mval, device_mval, size_height, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess) {
        fprintf(stderr, "DBUG : at %d in %s\n", __LINE__, __FILE__);
        fprintf(stderr, "FAIL : CUDA fail copy result to host\n");
        fprintf(stderr, "FAIL : error ");

        if ( err == cudaErrorInvalidValue ) {
            fprintf(stderr, "cudaErrorInvalidValue\n");
        } else if ( err == cudaErrorInvalidDevicePointer ) {
            fprintf(stderr, "cudaErrorInvalidDevicePointer\n");
        } else if ( err == cudaErrorInvalidMemcpyDirection ) {
            fprintf(stderr, "cudaErrorInvalidMemcpyDirection\n");
        } else {
            fprintf(stderr, "what the absolute fuck?\n");
            fprintf(stderr, "err = %0x\n", err );
        }
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy result from device to host done.\n");
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : copy device result done %" PRIu64 " nsecs\n",
                                                          tdelta_nsec);
    fprintf( stderr,"DBUG : at %d in %s\n", __LINE__, __FILE__);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess) {
        fprintf(stderr, "FAIL : CUDA failed cudaDeviceSynchronize()\n");
        fprintf(stderr, "err = %0x\n", err );
        exit(EXIT_FAILURE);
    }

    /* Free device memory */
    err = cudaFree(device_r);
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL : free memory on device for real\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(device_i);
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL : free memory on device for imaginary\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(device_mval);
    if (err != cudaSuccess) {
        fprintf(stderr, "FAIL : free memory on device for mbrot value\n");
        fprintf(stderr, "FAIL : error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* for all of the data what was the height values in host_mval */
    clock_gettime( CLOCK_REALTIME, &t0 );
    int error_count = 0;
    uint32_t delta_error_sum = 0;
    for (int i = 0; i < num_elements; ++i)
    {

        check_val = cpu_mbrot( host_r[i], host_i[i], (uint32_t)BAIL_OUT );

        if ( host_mval[i] != check_val ) {

            printf("%-9i    :     ( %-+18.12e , %-+18.12e ) == %-6i",
                               i, host_r[i], host_i[i], host_mval[i] );
            printf("    ERROR %-6i    DELTA = ", check_val);

            if ( host_mval[i] < check_val ){
                delta_error_sum += check_val - host_mval[i];
                printf("%i\n", check_val - host_mval[i]);
            } else {
                delta_error_sum += host_mval[i] - check_val;
                printf("%i\n", host_mval[i] - check_val);
            }
            error_count += 1;
        }
    }
    printf("     : Total error count = %i\n", error_count);
    if ( error_count > 0 ) {
        printf("     : Total delta error = %i\n", delta_error_sum);
        printf("     : Mean error = %10.6f\n",
                           (float)(delta_error_sum) / error_count);
    }
    clock_gettime( CLOCK_REALTIME, &t1 );
    tdelta_nsec = timediff( t0, t1);
    printf("     : data check done %" PRIu64 " nsecs\n", tdelta_nsec);

    /* Free host memory */
    free(host_r);
    free(host_i);
    free(host_mval);

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

uint32_t cpu_mbrot( double c_r, double c_i, uint32_t bail_out )
{

    /* point c belongs to the Mandelbrot set if and only if
     * the magnitude of the f(c) <= 2.0 */
    uint32_t height = 0;
    double zr = 0.0;
    double zi = 0.0;
    double tmp_r, tmp_i;
    double mag = 0.0;

    while ( ( height < bail_out ) && ( mag < 4.0 ) ) {
        tmp_r = ( zr * zr ) - ( zi * zi );
        tmp_i = ( zr * zi ) + ( zr * zi );
        zr = tmp_r + c_r;
        zi = tmp_i + c_i;

        /* mag = sqrt( zr * zr + zi * zi ); 
         *
         * lets speed this up a little and lose the sqrt
         *
         * mag = zr * zr + zi * zi;
         */

        mag = zr * zr + zi * zi;

        height += 1;
    }

    return ( height );

}

