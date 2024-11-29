#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

void print_nccl_unique_id(ncclUniqueId id) {
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
        printf("%02x", id.internal[i]);
    }
}


int main(int argc, char* argv[]) {
    int rank, size;
    // Initialize MPI for inter-process communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Error checking: Ensure we have exactly 4 processes
    if (size != 4) {
        if (rank == 0) printf("This example requires exactly 4 processes.\n");
        MPI_Finalize();
        return -1;
    }

    // Set GPU device based on rank
    cudaSetDevice(rank);

    // Define two groups: Group 0 (ranks 0,1), Group 1 (ranks 2,3)
    int group = rank / 2;         // Group ID
    int groupRank = rank % 2;     // Rank within the group
    int groupSize = 2;            // Size of each group
    
    // Generate a unique NCCL ID for each group
    ncclUniqueId groupId;
    printf("Rank %d in group %d generating NCCL unique ID.\n", rank, group);
    if (groupRank == 0) {         // Only the first rank in each group generates the ID
        ncclGetUniqueId(&groupId);
    }
    
    // Broadcast the unique ID to all members of the group
    MPI_Bcast(&groupId, sizeof(groupId), MPI_BYTE, group * 2, MPI_COMM_WORLD);

    // Initialize a communicator for the group
    ncclComm_t groupComm;
    printf("Group rank %d in group %d initializing NCCL communicator.\n", rank, group);
    ncclGroupStart();
    ncclCommInitRank(&groupComm, groupSize, groupId, groupRank);
    ncclGroupEnd();

    printf("Rank %d in group %d initialized NCCL communicator [group rank: %u, groupId: %d].\n", rank, group, groupRank, groupId);
    // printf("[UNIQUE ID] Rank %d in group %d: ", rank, group);
    // print_nccl_unique_id(groupId);
    // printf("\n");

    // Example: Group-specific collective communication (e.g., All-Reduce)
    float sendBuf = (float)rank;
    printf("Rank %d in group %d sending data: %f\n", rank, group, sendBuf);
    float recvBuf = 0.0;
    float* d_recvBuf;
    cudaMalloc(&d_recvBuf, sizeof(float));
    // cudaMemcpy(d_recvBuf, &recvBuf, sizeof(float), cudaMemcpyHostToDevice);

    float *d_sendBuf;
    cudaMalloc(&d_sendBuf, sizeof(float));
    cudaMemcpy(d_sendBuf, &sendBuf, sizeof(float), cudaMemcpyHostToDevice);
    int nccl_result;
    ncclGroupStart();
    nccl_result = ncclAllReduce(d_sendBuf, d_recvBuf, 1, ncclFloat, ncclSum, groupComm, cudaStreamDefault);
    ncclGroupEnd();
    // printf("result: %d\n", nccl_result);
    cudaMemcpy(&recvBuf, d_recvBuf, sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(cudaStreamDefault);
    printf("Rank %d in group %d received result: %f\n", rank, group, recvBuf);

    // Finalize the NCCL communicator for this group
    ncclCommDestroy(groupComm);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
