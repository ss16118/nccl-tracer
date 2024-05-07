CC=mpicc
C_SRC=nccl_wrapper.c
CFLAGS=-I${NCCL_INSTALL}/include -I/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/include -O2 -g
LDFLAGS=-L${NCCL_INSTALL}/lib -L/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/lib64
LIBS=-lcuda -lcudart -lnccl

tracer: nccl-tracer.o
	${CC} -shared -fPIC -Wl,-soname,libncclprof.so.1 -o libncclprof.so ${LDFLAGS} ${LIBS} $<

nccl-tracer.o:
	echo "[INFO] NCCL install dir: ${NCCL_INSTALL}"
	${CC} -fPIC ${CFLAGS} -o $@ -c ${C_SRC}

clean:
	rm -f nccl-tracer.o
	rm -f libncclprof.so
