CC=nvcc
CFLAGS=-I${NCCL_INSTALL}/include -I/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/include -O2
LDFLAGS=-L${NCCL_INSTALL}/lib -L/apps/ault/spack/opt/spack/linux-centos8-zen2/gcc-10.2.0/cuda-11.4.0-udyaakpt7oztg7gnj764dhkhdf5ory5d/lib64
LIBS=-lcuda -lcudart -lnccl

all: bcast reduce memalloc

bcast: bcast.o
	${CC} -o $@ ${LDFLAGS} ${LIBS} $<

reduce: reduce.o
	${CC} -o $@ ${LDFLAGS} ${LIBS} $<

memalloc: memalloc.o
	${CC} -o $@ ${LDFLAGS} ${LIBS} $<

bcast.o: nccl_bcast.cu
	${CC} ${CFLAGS} -o $@ -c $<

reduce.o: nccl_reduce.cu
	${CC} ${CFLAGS} -o $@ -c $<

memalloc.o: nccl_memalloc.cu
	${CC} ${CFLAGS} -o $@ -c $<

clean:
	rm -f bcast.o bcast
	rm -f reduce.o reduce
	rm -f memalloc.o memalloc
