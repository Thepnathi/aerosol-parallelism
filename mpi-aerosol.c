
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h> 

double liquid_mass=2.0, gas_mass=0.3, k=0.00001;

int init(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);
double calc_system_energy(double, double*, double*, double*, int);
void output_particles(double*, double*, double*, double*, double*, double*, double*, double*, int);
void calc_centre_mass(double*, double*, double*, double*, double*, double, int);
int debugParticle(double *x, double *y, double *z, int num);

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);

  int numProcesses, rankNum;

  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankNum);

  int i, j;
  int num;     // user defined (argv[1]) total number of gas molecules in simulation
  int time, timesteps; // for time stepping, including user defined (argv[2]) number of timesteps to integrate
  int rc;      // return code
  double *mass, *x, *y, *z, *vx, *vy, *vz;  // 1D array for mass, (x,y,z) position, (vx, vy, vz) velocity
  double dx, dy, dz, d, F, GRAVCONST=0.001, T=300;
  double ax, ay, az;
  double *gas, *liquid, *loss_rate;       // 1D array for each particle's component that will evaporate
  double *old_x, *old_y, *old_z, *old_mass;            // save previous values whilst doing global updates
  double totalMass, localMass, systemEnergy;  // for stats

  double start=omp_get_wtime();   // we make use of the simple wall clock timer available in OpenMP 

  /* if avail, input size of system */
  if (argc > 1 ) {
    num = atoi(argv[1]);
    timesteps = atoi(argv[2]);
  }
  else {
    num = 20000;
    timesteps = 50;
  }

  printf("Initializing for %d particles in x,y,z space...", num);

  /* malloc arrays and pass ref to init(). NOTE: init() uses random numbers */
  mass = (double *) malloc(num * sizeof(double));
  x =  (double *) malloc(num * sizeof(double));
  y =  (double *) malloc(num * sizeof(double));
  z =  (double *) malloc(num * sizeof(double));
  vx = (double *) malloc(num * sizeof(double));
  vy = (double *) malloc(num * sizeof(double));
  vz = (double *) malloc(num * sizeof(double));
  gas = (double *) malloc(num * sizeof(double));
  liquid = (double *) malloc(num * sizeof(double));
  loss_rate = (double *) malloc(num * sizeof(double));
  old_x = (double *) malloc(num * sizeof(double));
  old_y = (double *) malloc(num * sizeof(double));
  old_z = (double *) malloc(num * sizeof(double));
  old_mass = (double *) malloc(num * sizeof(double));

  // initialise
  rc = init(mass, x, y, z, vx, vy, vz, gas, liquid, loss_rate, num);

  totalMass = 0.0; // using MPI_Allreduce here breaks the mass data for some reason...
  for (i=0; i<num; i++) {
    mass[i] = gas[i]*gas_mass + liquid[i]*liquid_mass;
    totalMass += mass[i];
  }

  systemEnergy = calc_system_energy(totalMass, vx, vy, vz, num);
  printf("Time 0. System energy=%g\n", systemEnergy);


  printf("Now to integrate for %d timesteps\n", timesteps);

  int partitionSizePerProcess = num/numProcesses;  // partion the particles for each process
  if (rankNum == numProcesses-1) { // If it is the last rank
    partitionSizePerProcess = num - (partitionSizePerProcess * rankNum);
  } 
  int startIndex = rankNum * partitionSizePerProcess;
  int endIndex = startIndex + partitionSizePerProcess - 1;

  // time=0 was initial conditions
  for (time=1; time<=timesteps; time++) {

    double temp_old_x[partitionSizePerProcess];
    double temp_old_y[partitionSizePerProcess];
    double temp_old_z[partitionSizePerProcess];
    double temp_old_mass[partitionSizePerProcess];
    
    int temp_index = 0;

    // LOOP1: take snapshot to use on RHS when looping for updates
    for (i=startIndex; i<=endIndex; i++) {
      old_x[i] = x[i];
      old_y[i] = y[i];
      old_z[i] = z[i];
      old_mass[i] = mass[i];
      temp_old_x[temp_index] = old_x[i];
      temp_old_y[temp_index] = old_y[i];
      temp_old_z[temp_index] = old_z[i];
      temp_old_mass[temp_index] = old_mass[i];
      // printf("\nIndex=%d and the mass=%lf\n", i, old_mass[i]);
      temp_index += 1;
    }

    double *old_x_buffer;
    double *old_y_buffer;
    double *old_z_buffer;
    double *old_mass_buffer;

    old_x_buffer = (double *) malloc(num * sizeof(double));
    old_y_buffer = (double *) malloc(num * sizeof(double));
    old_z_buffer = (double *) malloc(num * sizeof(double));
    old_mass_buffer = (double *) malloc(num * sizeof(double));

    MPI_Allgather(&temp_old_x, partitionSizePerProcess, MPI_DOUBLE, old_x_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&temp_old_y, partitionSizePerProcess, MPI_DOUBLE, old_y_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&temp_old_z, partitionSizePerProcess, MPI_DOUBLE, old_z_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&temp_old_mass, partitionSizePerProcess, MPI_DOUBLE, old_mass_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);

  // Wait until all process has finish before proceeding to use the buffer
    MPI_Barrier(MPI_COMM_WORLD);

        // Now we will update the old x, y, z and mass of the particles across all the ranks
    for (i=0; i<num; i++) {
      old_x[i] = old_x_buffer[i];
      old_y[i] = old_y_buffer[i];
      old_z[i] = old_z_buffer[i];
      old_mass[i] = old_mass_buffer[i];
    }

    free(old_x_buffer);
    free(old_y_buffer);
    free(old_z_buffer);
    free(old_mass_buffer);

    double temp_d, temp_z;

    double temp_vx[partitionSizePerProcess];
    double temp_vy[partitionSizePerProcess];
    double temp_vz[partitionSizePerProcess];
    double x_copy[partitionSizePerProcess];
    double y_copy[partitionSizePerProcess];
    double z_copy[partitionSizePerProcess];
    double temp_mass[partitionSizePerProcess];
    temp_index = 0;

    // LOOP2: update position etc per particle (based on old data)
    for(i=startIndex; i<=endIndex; i++) {
      // calc forces on body i due to particles (j != i)
      for (j=0; j<num; j++) {
        if (j != i) {
          dx = old_x[j] - x[i];
          dy = old_y[j] - y[i];
	        dz = old_z[j] - z[i];
          temp_d =sqrt(dx*dx + dy*dy + dz*dz);
          d = temp_d>0.01 ? temp_d : 0.01;
          F = GRAVCONST * mass[i] * old_mass[j] / (d*d);
	        // calculate acceleration due to the force, F
          ax = (F/mass[i]) * dx/d;
          ay = (F/mass[i]) * dy/d;
	        az = (F/mass[i]) * dz/d;
	        // approximate velocities in "unit time"
          vx[i] += ax;
          vy[i] += ay;
	        vz[i] += az;
        }
      } 
      // calc new position 
      x[i] = old_x[i] + vx[i];
      y[i] = old_y[i] + vy[i];
      z[i] = old_z[i] + vz[i];
      // temp-dependent condensation from gas to liquid
      gas[i] *= loss_rate[i] * exp(-k*T);
      liquid[i] = 1.0 - gas[i];
      mass[i] = gas[i]*gas_mass + liquid[i]*liquid_mass;
      // conserve energy means 0.5*m*v*v remains constant
      double v_squared = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
      double factor = sqrt(old_mass[i]*v_squared/mass[i])/sqrt(v_squared);
      vx[i] *= factor;
      vy[i] *= factor;
      vz[i] *= factor;
      temp_vx[temp_index] = vx[i];
      temp_vy[temp_index] = vy[i];
      temp_vz[temp_index] = vz[i];
      x_copy[temp_index] = x[i];
      y_copy[temp_index] = y[i];
      z_copy[temp_index] = z[i];
      temp_mass[temp_index] = mass[i];
      temp_index += 1;
    } // end of LOOP 2

    double *vx_buffer, *vy_buffer, *vz_buffer;
    double *x_buffer, *y_buffer, *z_buffer;
    double *mass_buffer;

    vx_buffer = (double *) malloc(num * sizeof(double));
    vy_buffer = (double *) malloc(num * sizeof(double));
    vz_buffer = (double *) malloc(num * sizeof(double));
    x_buffer = (double *) malloc(num * sizeof(double));
    y_buffer = (double *) malloc(num * sizeof(double));
    z_buffer = (double *) malloc(num * sizeof(double));
    mass_buffer = (double *) malloc(num * sizeof(double));

    MPI_Allgather(&temp_vx, partitionSizePerProcess, MPI_DOUBLE, vx_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&temp_vy, partitionSizePerProcess, MPI_DOUBLE, vy_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&temp_vz, partitionSizePerProcess, MPI_DOUBLE, vz_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgather(&x_copy, partitionSizePerProcess, MPI_DOUBLE, x_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&y_copy, partitionSizePerProcess, MPI_DOUBLE, y_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&z_copy, partitionSizePerProcess, MPI_DOUBLE, z_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgather(&temp_mass, partitionSizePerProcess, MPI_DOUBLE, mass_buffer, partitionSizePerProcess, MPI_DOUBLE, MPI_COMM_WORLD);

    // Update and combine vx, vy, vz for all the ranks
    for (i=0; i < num; i++) {
      vx[i] = vx_buffer[i];
      vy[i] = vy_buffer[i];
      vz[i] = vz_buffer[i];
      x[i] = x_buffer[i];
      y[i] = y_buffer[i];
      z[i] = z_buffer[i];
      mass[i] = mass_buffer[i];
    }

    free(vx_buffer);
    free(vy_buffer);
    free(vz_buffer);
    free(x_buffer);
    free(y_buffer);
    free(z_buffer);
    free(mass_buffer);

    totalMass = 0.0;
    localMass = 0.0;
    for (i=startIndex; i<=endIndex; i++) {
      localMass += mass[i];
    }
    MPI_Allreduce(&localMass, &totalMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // printf("\nRank is %d and the total mass is %lf\n", rankNum, totalMass);

    systemEnergy = calc_system_energy(totalMass, vx, vy, vz, num);
    if (rankNum == 0) {
      printf("At end of timestep %d with temp %f the system energy=%g and total aerosol mass=%g\n", time, T, systemEnergy, totalMass);
    }
    // temperature drops per timestep
    T *= 0.99999;
  } // time steps
  
  MPI_Barrier(MPI_COMM_WORLD);
  if (rankNum == 0) {
    printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, omp_get_wtime()-start);
    // output a metric (centre of mass) for checking
    double com[3];
    calc_centre_mass(com, x,y,z,mass,totalMass,num);
    printf("Centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);
  }
} // main

int debugParticle(double *x, double *y, double *z, int num) {
  int i;
  printf("Printing the x, y, z before calc centre mass \n");
  for (i=0;i<num;i++) {
        printf("Index %d with x=%lf, y=%lf, z=%lf\n", i, x[i], y[i], z[i]);
  }
  printf("\n");
}


// init() will return 0 only if successful
int init(double *mass, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *gas, double* liquid, double* loss_rate, int num) {
  // random numbers to set initial conditions - do not parallelise or amend order of random number usage
  int i;
  double comp;
  double min_pos = -50.0, mult = +100.0, maxVel = +10.0;
  double recip = 1.0 / (double)RAND_MAX;

  // create all random numbers 
  int numToCreate = num*8;
  double *ranvec;
  ranvec = (double *) malloc(numToCreate * sizeof(double));
  if (ranvec == NULL) {
    printf("\n ERROR in malloc ranvec within init()\n");
    return -99;
  } 
  for (i=0; i<numToCreate; i++) {
    ranvec[i] = (double) rand();
  }

  // requirement to access ranvec in same order as if we had just used rand()
  for (i=0; i<num; i++) {
    x[i] = min_pos + mult*ranvec[8*i+0] * recip;  
    y[i] = min_pos + mult*ranvec[8*i+1] * recip;  
    z[i] = 0.0 + mult*ranvec[8*i+2] * recip;   
    vx[i] = -maxVel + 2.0*maxVel*ranvec[8*i+3] * recip;   
    vy[i] = -maxVel + 2.0*maxVel*ranvec[8*i+4] * recip;   
    vz[i] = -maxVel + 2.0*maxVel*ranvec[8*i+5] * recip;   
    // proportion of aerosol that evaporates
    comp = .5 + ranvec[8*i+6]*recip/2.0;
    loss_rate[i] = 1.0 - ranvec[8*i+7]*recip/25.0;
    // aerosol is component of gas and (1-comp) of liquid
    gas[i] = comp;
    liquid[i] = (1.0-comp);
  }

  // release temp memory for ranvec which is no longer required
  free(ranvec);

  return 0;
} // init


double calc_system_energy(double mass, double *vx, double *vy, double *vz, int num) {
  /* 
     energy is sum of 0.5*mass*velocity^2
     where velocity^2 is sum of squares of components
  */
  int i;
  double totalEnergy = 0.0, systemEnergy;
  for (i=0; i<num; i++) {
    totalEnergy += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
  }
  totalEnergy = 0.5 * mass * totalEnergy;
  systemEnergy = totalEnergy / (double) num;
  return systemEnergy;
}


void output_particles(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *gas, double *liquid, int n) {
  int i;
  printf("num \t position (x,y,z) \t velocity (vx, vy, vz) \t mass (gas, liquid)\n");
  for (i=0; i<n; i++) {
    printf("%d \t %f %f %f \t %f %f %f \t %f %f\n", i, x[i], y[i], z[i], vx[i], vy[i], vz[i], gas[i]*gas_mass, liquid[i]*liquid_mass);
  }
}


void calc_centre_mass(double *com, double *x, double *y, double *z, double *mass, double totalMass, int N) {
  int i, axis;
   // calculate the centre of mass, com(x,y,z)
  for (axis=0; axis<2; axis++) {
    com[0] = 0.0;     com[1] = 0.0;     com[2] = 0.0; 
    for (i=0; i<N; i++) {
      com[0] += mass[i]*x[i];
      com[1] += mass[i]*y[i];
      com[2] += mass[i]*z[i];
    }
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
  }
  return;
}
