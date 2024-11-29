CC=gcc
C_SRC=nccl_wrapper.c
NCCL_INSTALL=/scratch/sshen/miniconda3/pkgs/nccl-2.23.4.1-h2b5d15b_3
CUDA_INSTALL=/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d
CFLAGS=-I${NCCL_INSTALL}/include -I${CUDA_INSTALL}/include -O2 -g
LDFLAGS=-L${NCCL_INSTALL}/lib -L${CUDA_INSTALL}/lib64 -L${CUDA_INSTALL}/targets/x86_64-linux/lib/stubs
LIBS=-lcuda -lcudart -lnccl

tracer: nccl-tracer.o
	${CC} -shared -fPIC -Wl,-soname,libncclprof.so.1 -o libncclprof.so ${LDFLAGS} ${LIBS} $<

nccl-tracer.o:
	echo "[INFO] NCCL install dir: ${NCCL_INSTALL}"
	${CC} -fPIC ${CFLAGS} -o $@ -c ${C_SRC}

clean:
	rm -f nccl-tracer.o
	rm -f libncclprof.so
