/*************************************************************************
 * NCCL Tracer
 * Siyuan Shen - ETH Zurich
 *************************************************************************/
#define _GNU_SOURCE 
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>
#include <nccl.h>
#include <assert.h>
#include <time.h>

#define WRITE_TRACE(fmt, args...) assert(file_ptr != NULL); fprintf(file_ptr, fmt, args)
int initialized = 0;
FILE *file_ptr;
int proc_rank = -1;
ncclComm_t initial_comm = NULL;


/**
 * Converts the given ncclUniqueId to a string stored in the given buffer
 * as per the first 8 bytes of the internal array.
 * Does not guarantee uniqueness.
 */
static void unique_id_to_string(ncclUniqueId id, char *str) {
    for (int i = 0; i < 4; i++) {
        sprintf(str + i * 2, "%02x", id.internal[i]);
    }
    str[8] = '\0';
}


/**
 * A helper function that waits for the initialization of the given communicator
 * in case ncclCommInitRankConfig is used and the 'blocking' attribute is set to 0.
 */
static void wait_for_init(ncclComm_t comm)
{
  ncclResult_t nccl_result;
  do
  {
    pncclCommGetAsyncError(comm, &nccl_result);
  } while (nccl_result == ncclInProgress);

  if (nccl_result != ncclSuccess)
  {
    printf("Error in pncclCommGetAsyncError\n");
    exit(1);
  }
}


static void init_tracer()
{
  if (initialized == 0)
  {
    char file_name[PATH_MAX];
    char *env = getenv("NCCL_TRACE_PREFIX");
    int device;
    cudaGetDevice(&device);
    printf("********* NCCL Tracer initialized for device: %d *********\n", device);
    if (env == NULL)
    {
      snprintf(file_name, PATH_MAX, "nccl_trace_%d.txt", device);
    }
    else
    {
      snprintf(file_name, PATH_MAX, "%s_%d.txt", env, device);
    } 
    printf("[NCCL DEBUG] Rank %d writing to %s\n", device, file_name);
    // Deletes the file if it already exists
    if (remove(file_name) != 0)
    {
      printf("Error deleting file %s\n", file_name);
    }
    FILE *tmp = fopen(file_name, "a");
    assert(tmp != NULL);

    // Reads from the temp file and writes to the output file
    // size_t trace_size = ftell(file_ptr);
    // fseek(file_ptr, 0, SEEK_SET);

    // void *buffer = malloc(trace_size * sizeof(char));
    // size_t bytes_read = fread(buffer, 1, trace_size, file_ptr);
    // assert(bytes_read == trace_size);
    // // Writes the content of the temp file to the output file
    // fwrite(buffer, 1, trace_size, tmp);
    file_ptr = tmp;
    
    // free(buffer);
    initialized = 1;
  }
  // else
  // {
  //   // If the rank is not set, we cannot write to the file
  //   // So we store the trace in a temporary file
  //   file_ptr = tmpfile();
  //   assert(file_ptr != NULL);
  // }
}

static void write_to_file()
{
  assert(file_ptr != NULL);
  assert(initialized == 1);

  size_t trace_size = ftell(file_ptr);
  fseek(file_ptr, 0, SEEK_SET);

  char file_name[PATH_MAX];
  char *env = getenv("NCCL_TRACE_PREFIX");
  if (env == NULL)
  {
    snprintf(file_name, PATH_MAX, "nccl_trace_%d.txt", proc_rank);
  }
  else
  {
    snprintf(file_name, PATH_MAX, "%s_%d.txt", env, proc_rank);
  }

  FILE *output_file = fopen(file_name, "w");
  assert(output_file != NULL);
  // Reads from teh temp file and writes to the output file
  void *buffer = malloc(trace_size * sizeof(char));
  size_t bytes_read = fread(buffer, 1, trace_size, file_ptr);
  assert(bytes_read == trace_size);
  fwrite(buffer, 1, trace_size, output_file);
  fclose(output_file);
  fclose(file_ptr);
  free(buffer);
  printf("*** NCCL trace of rank %d is written to %s: [size: %d] ***\n", proc_rank, file_name, trace_size);
}



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

