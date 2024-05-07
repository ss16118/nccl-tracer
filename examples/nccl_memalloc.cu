#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CALL(func) \
    do { \
        cudaError_t err = (func); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    ncclResult_t nccl_result = ncclGetUniqueId(&id);
    if (nccl_result != ncclSuccess) {
        fprintf(stderr, "NCCL GetUniqueId failed\n");
        exit(EXIT_FAILURE);
    }
    nccl_result = ncclCommInitRank(&comm, 1, id, 0);
    if (nccl_result != ncclSuccess) {
        fprintf(stderr, "NCCL Init failed\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory using ncclMemAlloc
    size_t size = 1024 * sizeof(float);
    float* data;
    nccl_result = ncclMemAlloc((void**)&data, size);
    if (nccl_result != ncclSuccess) {
        fprintf(stderr, "ncclMemAlloc failed\n");
        exit(EXIT_FAILURE);
    }
    printf("[DEBUG] Allocated memory\n");
    // Use the allocated memory
    // for (int i = 0; i < 1024; ++i) {
    //     data[i] = i;
    // }

    // Free the allocated memory
    nccl_result = ncclMemFree(data);
    if (nccl_result != ncclSuccess) {
        fprintf(stderr, "ncclMemFree failed\n");
        exit(EXIT_FAILURE);
    }

    // Finalize NCCL
    nccl_result = ncclCommDestroy(comm);
    if (nccl_result != ncclSuccess) {
        fprintf(stderr, "NCCL Destroy failed\n");
        exit(EXIT_FAILURE);
    }

    return 0;
}
