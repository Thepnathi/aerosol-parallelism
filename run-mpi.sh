#!/bin/bash -l

# Load the Intel modules required to run the code
module load compilers/intel/2019u5 
module load mpi/intel-mpi/2019u5/bin


echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of threads or processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"


SRC=mpi-aerosol.c
EXE=${SRC%%.c}.exe
echo compiling $SRC to $EXE

export numMPI=${SLURM_NTASKS:-1}
# export numMPI=${SLURM_NTASKS:-1} # if '-n' not used then default to 1

mpiicc -O0 -qopenmp $SRC -o $EXE && \
      (
      echo Using ${numMPI} MPI processes
      mpirun -np ${numMPI} ./${EXE};echo
  ) \
      || echo $SRC did not built to $EXE



## if wanted to check processor core speeds
# # run 3 times
# grep MHz /proc/cpuinfo|sort -nr|uniq -c; ./${EXE};echo
# grep MHz /proc/cpuinfo|sort -nr|uniq -c; ./${EXE};echo
# grep MHz /proc/cpuinfo|sort -nr|uniq -c; ./${EXE};echo
# grep MHz /proc/cpuinfo|sort -nr|uniq -c; 

