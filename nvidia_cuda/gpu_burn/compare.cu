
/* rounding errors due to results being added
 * Therefore EPSILON = 0.0f is silly but worth while to
 * test with. What can go wrong?  Right? */

/* ordinary 32-bit float */
#define EPSILON 0.001f

/* more useful 64-bit double floating point */
#define EPSILOND 0.0000001

extern "C" __global__ void compare(float *C, int *faultyElems, size_t iters) {

    size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;

    /* please excuse the silly comments that are inside the lines */
    size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)* /* Y */
        gridDim.x*blockDim.x +                              /* W */
        blockIdx.x*blockDim.x + threadIdx.x;                /* X */

    int myFaulty = 0;

    for (size_t i = 1; i < iters; ++i) {
        if (fabsf( C[myIndex] - C[myIndex + i*iterStep]) > EPSILON) {
            /* well we hit an error so lets count it */
            myFaulty++;
        }
    }

    /* the atomic add ensures that no other thread anywhere can
     * mess with the data we are incrementing */
    atomicAdd(faultyElems, myFaulty);

}

/* same deal as above just for double precision data */
extern "C" __global__ void compareD(double *C, int *faultyElems, size_t iters) {

    size_t iterStep = blockDim.x*blockDim.y*gridDim.x*gridDim.y;

    size_t myIndex = (blockIdx.y*blockDim.y + threadIdx.y)*
        gridDim.x*blockDim.x +
        blockIdx.x*blockDim.x + threadIdx.x;

    int myFaulty = 0;

    for (size_t i = 1; i < iters; ++i) {
        if (fabs(C[myIndex] - C[myIndex + i*iterStep]) > EPSILOND) {
            /* we hit an error so lets count it */
            myFaulty++;
        }
    }

    atomicAdd(faultyElems, myFaulty);
}

