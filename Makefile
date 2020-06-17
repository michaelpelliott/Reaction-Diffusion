CC = gcc
MCC = mpicc
CFLAGS = -std=c99

final: spots_and_stripes.c cg_BE_parallel.c
	${MCC} ${CFLAGS} -o $@ -O2 $^ -lm

poisson_parallel: solve_Poisson_parallel.c cg_parallel.c
	${MCC} -o $@ $^ -lm

poisson_serial: solve_Poisson_serial.c cg_serial.c
	${CC} ${CFLAGS} -o $@ $^ -lm 

spiral: spiral_serial.c
	${CC} -o spiral -O2 spiral_serial.c -lm

spiral_parallel: spiral_parallel.c
	${MCC} -o $@ $< -lm

1dparallel: heat_parallel.c
	${MCC} -o 1dparallel heat_parallel.c -lm

clean:
	rm -rf final poisson_parallel 1dparallel spiral spiral_parallel poisson_serial *.out

