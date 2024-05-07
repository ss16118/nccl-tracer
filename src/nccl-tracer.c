/*************************************************************************
 * NCCL Tracer
 * Siyuan Shen - ETH Zurich
 *************************************************************************/
#define _GNU_SOURCE 
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>
#include <nccl.h>
#include <time.h>

// TODO: Replace with writing to file
#define WRITE_TRACE(fmt, args...) printf(fmt, args)


/**
 * A helper function that returns a single timestamp to indicate
 * the current time. Only works for POSIX systems.
 **/
double get_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Wrapper function for ncclMemAlloc
ncclResult_t ncclMemAlloc(void** ptr, size_t size)
{
  ncclResult_t nccl_result;

  double start_time = get_time();
  
  nccl_result = pncclMemAlloc(ptr, size);

  double end_time = get_time();
  
  WRITE_TRACE("ncclMemAlloc:%f:%lx:%d:%f\n", start_time, (uintptr_t)*ptr, size, end_time);
  
  return nccl_result;
}

// Wrapper function for ncclMemFree
ncclResult_t ncclMemFree(void *ptr)
{
  ncclResult_t nccl_result;

  double start_time = get_time();
  
  nccl_result = pncclMemFree(ptr);

  double end_time = get_time();
  
  WRITE_TRACE("ncclMemFree:%f:%lx:%f\n", start_time, (uintptr_t)ptr, end_time);
  
  return nccl_result;
}


// Wrapper function for ncclCommInitRankConfig
/* ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
				    ncclConfig_t* config)
{
  ncclResult_t nccl_result;

  double start_time = get_time();
  
  nccl_result = pncclBcast(buff, count, datatype, root, comm, stream);

  double end_time = get_time();
} */


// Wrapper function for ncclBcast
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
		       ncclComm_t comm, cudaStream_t stream)
{
  ncclResult_t nccl_result;

  double start_time = get_time();
  
  nccl_result = pncclBcast(buff, count, datatype, root, comm, stream);

  double end_time = get_time();
  
  WRITE_TRACE("ncclBroadcast:%f:%lx:%d:%d:%d:%lx:%lx:%f\n",
	      start_time, (uintptr_t)buff, count, datatype, root, &comm, &stream, end_time);
  
  return nccl_result;
}


