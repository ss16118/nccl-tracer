/*************************************************************************
 * NCCL Tracer
 * Siyuan Shen - ETH Zurich
 *************************************************************************/
#define _GNU_SOURCE 
#include <stdio.h>
#include <dlfcn.h>
#include <nccl.h>

// Define the type of pointer to the ncclBcast function
typedef ncclResult_t (*ncclBcast_func_t)(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

// Wrapper function for ncclBcast
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    static ncclBcast_func_t real_ncclBcast = NULL;

    if (!real_ncclBcast) {
        // Load the real ncclBcast function
        real_ncclBcast = (ncclBcast_func_t)dlsym(RTLD_NEXT, "ncclBcast");
        if (!real_ncclBcast) {
            fprintf(stderr, "Error loading ncclBcast: %s\n", dlerror());
            return ncclSystemError;
        }
    }

    // Custom code before the real call
    printf("ncclBcast intercepted!\n");

    // Call the real ncclBcast function
    return real_ncclBcast(buff, count, datatype, root, comm, stream);
}


