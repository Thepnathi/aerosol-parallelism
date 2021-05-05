#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h> 

int main(int argc, char* argv[]) {

    MPI_Init(NULL, NULL);

    int numProcesses, rankNum;

    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankNum);

    int i;
    int num = 6;
    int numbers[6] = {1, 2, 3, 4, 5, 6}; 

    int partitionSizePerProcess = num/numProcesses;  // partion the particles for each process
    if (rankNum == numProcesses-1) { // If it is the last rank
      partitionSizePerProcess = num - (partitionSizePerProcess * rankNum);
    } 
    int startIndex = rankNum * partitionSizePerProcess;
    int endIndex = startIndex + partitionSizePerProcess - 1;

    int tempArray[partitionSizePerProcess];
    int start = 0;

    for (i=startIndex; i <= endIndex; i++) {
        numbers[i] = numbers[i] * 2;
        tempArray[start] = numbers[i];
        start+=1;
    }

    int gathered[6];
    MPI_Allgather(&tempArray, partitionSizePerProcess, MPI_INT, gathered, partitionSizePerProcess, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Rank %d\n", rankNum);
    for (i=0; i < num; i++) {
        printf("%d ", gathered[i]);
    }
    printf("\n");
    MPI_Finalize();
}