/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <omp.h>

/* tested with K4200 GPU */
#define NUM_ELEMENTS 500000
#define THREADS_PER_BLOCK 1024
#define EPSILON_ERROR 1e-9

/**
 * CUDA Kernel Device code
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
    float epsilon = EPSILON_ERROR;
    int numElements = NUM_ELEMENTS;
    size_t size = numElements * sizeof(float);

    int num_gpus = 0;   // number of CUDA GPUs

    printf("\n---------+---------+---------+---------+");
    printf("---------+---------+---------+--\n");

    /* determine the number of CUDA capable GPUs */
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1)
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
    printf("---------+---------+---------+---------+");
    printf("---------+---------+---------+--\n");

    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    /* Init input vectors with random data */
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    /* Allocate the device input vectors */
    float *d_A = NULL;
    if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy the host input vectors A and B in host memory
     * to the device input vectors in device memory */
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy of vector A from host to device done.\n");

    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        err = cudaGetLastError();
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy of vector B from host to device done.\n");

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "FAIL : vectorAdd (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : vectorAdd done.\n");

    /* Copy the device result vector in device memory to the host
     * result vector in host memory */
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("INFO : Copy output data from CUDA device to host memory done.\n");

    /* test that result vector is correct within epsilon error */
    for (int i = 0; i < numElements; ++i)
    {
        if ( fabs(h_A[i] + h_B[i] - h_C[i]) > epsilon )
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("INFO : A + B correct within error epsilon = %g\n", epsilon);

    /* Free device global memory */
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return EXIT_SUCCESS;

}

