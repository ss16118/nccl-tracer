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

