// File: nccl_broadcast_example.c

#include <stdint.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_MPI(call)                                                         \
    if ((call) != MPI_SUCCESS) {                                                \
        fprintf(stderr, "MPI error calling \"%s\"\n", #call);                   \
        exit(-1);                                                               \
    }

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), #call);     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_NCCL(call)                                                        \
    do {                                                                        \
        ncclResult_t err = call;                                                \
        if (err != ncclSuccess) {                                               \
            fprintf(stderr, "NCCL error at %s:%d '%s'\n", __FILE__, __LINE__,   \
                    ncclGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(int argc, char* argv[]) {
    int rank, size;
    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t stream;
    int ngpus = 1; // Number of GPUs per process
    int gpu;
    
    // Initialize MPI
    CHECK_MPI(MPI_Init(&argc, &argv));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

    // Select GPU
    gpu = rank;
    CHECK_CUDA(cudaSetDevice(gpu));
    printf("Rank %d uses GPU %d\n", rank, gpu);
    // Get NCCL unique ID at rank 0 and broadcast it to all others
    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&id));
    }
    CHECK_MPI(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL communicator
    CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));
    printf("[DEBUG] %d, nccl communicator: %p\n", rank, (uintptr_t) comm);
    // Create CUDA stream
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Allocate device memory
    size_t N = 1024;
    float* d_buf;
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(float)));

    // Initialize buffer
    if (rank == 0) {
        // Rank 0 initializes the buffer with data
        float* h_buf = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; i++) {
            h_buf[i] = (float)i;
        }
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, N * sizeof(float), cudaMemcpyHostToDevice));
        free(h_buf);
    } else {
        // Other ranks initialize the buffer to zero
        CHECK_CUDA(cudaMemset(d_buf, 0, N * sizeof(float)));
    }

    // Perform NCCL broadcast (from rank 0 to all others)
    CHECK_NCCL(ncclBroadcast((const void*)d_buf, (void*)d_buf, N, ncclFloat, 0, comm, stream));
    printf("[DEBUG] %d, nccl communicator after bcast: %p\n", rank, (uintptr_t) comm);

    // Synchronize
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Check results (copy data back to host and print first few elements)
    float* h_recv_buf = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_recv_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the broadcasted data
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_recv_buf[i] != (float)i) {
            errors++;
            if (errors < 10) {
                printf("Rank %d: Mismatch at index %zu: expected %f, got %f\n",
                       rank, i, (float)i, h_recv_buf[i]);
            }
        }
    }

    if (errors == 0) {
        printf("Rank %d: Broadcast successful. Data verified.\n", rank);
    } else {
        printf("Rank %d: Broadcast failed with %d errors.\n", rank, errors);
    }

    free(h_recv_buf);

    // Clean up
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(comm));
    CHECK_MPI(MPI_Finalize());

    return 0;
}
