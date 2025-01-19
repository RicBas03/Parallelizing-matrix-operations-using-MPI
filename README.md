### Repository contents
- `homework2.c`: code containing all the implementations
described.
- `mpiC.pbs`: that runs the program for every matrix
size and number of processes.

### Instructions
1) Install gcc 9.1.0 and mpich-3.2.1--gcc-9.1.0 compiler version
2) Clone the repository
3) Compile the program with `mpicc -o es homework2.c `
4) Run the program with `mpirun -np n ./es N`, where n is the number of processes and N is the matrix
size.
5) Alternatively run the PBS with `qsub mpiC.pbs`
(remember to modify the directory path).
