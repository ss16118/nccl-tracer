# NCCL Tracer

This is a simple tool to trace the NCCL calls in a program based on the [pNCCL interface](https://github.com/NVIDIA/nccl/issues/344). It leverages the LD_PRELOAD mechanism to intercept the NCCL calls and records information about the calls, including the arguments and the return values. The information is then written to text files.


### Build

To generate the wrapper file that contains the tracing code, run the following command:

```bash
python3 gencode.py
```

This will generate the `nccl_wrapper.c` file. To build the tracer, run the following command. In addition, make sure to change the `NCCL_INSTALL` and `CUDA_INSTALL` variables in the `Makefile` to the correct paths on your system.

```bash
make clean && make
```
You should see the `libncclprof.so` file in the current directory. There are several example programs in the __examples__ directory that you can use to test the tracer.

### Usage

To trace an application, set the `LD_PRELOAD` environment variable to the path of the `libncclprof.so` file. For example, to trace the `mpi_nccl_bcast` application, run the following command:

```bash
LD_LIBRARY_PATH=<path-to-libcuda-and-libnccl> LD_PRELOAD=./libncclprof.so ./mpi_nccl_bcast
```

The output of each GPU rank is written to a separate text file in the current directory. The text files are named `nccl_trace_<rank>.txt`, where `<rank>` is the rank of the GPU. The text files contain the following information:
1. The name of the NCCL function that was called.
2. The starting and ending times of the function call.
3. The arguments of the function.
For more details, please refer to the source code in `nccl_wrapper.c` file.

You can change the output directory by setting the `NCCL_TRACE_PREFIX` environment variable. For example, to write the output to the `/tmp` directory, run the following command:

```bash
NCCL_TRACE_PREFIX=/tmp LD_LIBRARY_PATH=<path-to-libcuda-and-libnccl> LD_PRELOAD=./libncclprof.so ./mpi_nccl_bcast
```

### Customization

To customize the tracer, you can modify the `nccl_wrapper.c` file. The file contains the implementation of the NCCL functions that are intercepted by the tracer. You can add additional code to the functions to record more information about the calls.

If you want to change the way the information is recorded en masse, you can modify the `gencode.py` script. The script generates the `nccl_wrapper.c` file based on the `nccl.h` header file. You can modify the script to generate the code in a different format.