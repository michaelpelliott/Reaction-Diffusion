#include <mpi.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>   /* For atoi */
#include <stdio.h>

#include "cg_BE_parallel.h"   /* For f_t def */

#define MIN -0.5
#define MAX 0.5

void mpi_debug();

/* returns a random double between min and max */
double double_rand(double min, double max) {
    return ( (double)rand() * (max - min) )/(double)RAND_MAX + min;
}

/* initial condition: random double between -1/2 and 1/2 */
double init() {
    return double_rand(MIN,MAX);
}

typedef struct
{
    double t; 
    MPI_Datatype localarray_t;
} struct_timeinfo_t;

/* usage: 
 * mpicc compiler
 * mpirun -n [nprocs] ./solve_Poisson_parallel.c [N] [tol] [kmax] [prt] [prows] [pcols] 
 * note: prows * pcols must = nprocs
 * prt = 1 prints out some information, some useful some not.
 * prt = 10 prints out a lot of troubleshooting info in matmult.
 */
typedef struct 
{
    double xgrid[2];
    double ygrid[2]; 
    int N[2];
    int nout;
} struct_header_t;

void create_header_type(MPI_Datatype *header_t)
{
    int blocksize = 4;
    int block_lengths[4] = {2,2,2,1}; /* Use blocksize to dimension array */

    /* Set up types */
    MPI_Datatype typelist[4];
    typelist[0] = MPI_INT;
    typelist[1] = MPI_DOUBLE;
    typelist[2] = MPI_DOUBLE;
    typelist[3] = MPI_INT;

    /* Set up displacements */
    MPI_Aint disp[4];
    disp[0] = offsetof(struct_header_t,N);
    disp[1] = offsetof(struct_header_t,xgrid);
    disp[2] = offsetof(struct_header_t,ygrid);
    disp[3] = offsetof(struct_header_t,nout);

    MPI_Type_create_struct(blocksize,block_lengths, disp, typelist, header_t);
    MPI_Type_commit(header_t);
}

void create_timeinfo_type(MPI_Datatype localarray_t, MPI_Datatype *timeinfo_t)
{
    int blocksize = 2;
    int block_lengths[2] = {1,1}; /* Use blocksize to dimension array */


    /* Set up types */
    MPI_Datatype typelist[2];
    typelist[0] = MPI_DOUBLE;
    typelist[1] = localarray_t;

    /* Set up displacements */
    MPI_Aint disp[4];
    disp[0] = offsetof(struct_timeinfo_t,t);
    disp[1] = offsetof(struct_timeinfo_t,localarray_t);

    MPI_Type_create_struct(blocksize,block_lengths, disp, typelist, timeinfo_t);
    MPI_Type_commit(timeinfo_t);
}
    


/* N is along x direction, M is along y direction */
double** allocate_2d(int N, int M, int numGhosts)
{
    int rows = N + 1 + 2*numGhosts;
    int cols = M + 1 + 2*numGhosts;
    double *qmem = malloc(rows*cols*sizeof(double));
    double **qrows = malloc(rows*sizeof(double *));
    for (int i = 0; i < rows; i++) {
        qrows[i] = &qmem[cols*i + numGhosts];
    }
    double** q = &qrows[numGhosts];
    return q;
}

void delete_2d(int mbc, double ***q)
{
    free(&(*q)[-mbc][-mbc]);
    free(&(*q)[-mbc]);
    *q = NULL;
}

double rhs_v(double u, double v,double beta, double alpha, double t1, double gamma, double t2)
{
    double rhsV = beta*v*(1 + alpha*t1/beta*u*v) + u*(gamma + t2*v);
    return rhsV;
}

double rhs_u(double u, double v, double t1, double t2, double alpha)
{
    double rhsU = alpha*u*(1-t1*v*v)+v*(1-t2*u);
    return rhsU;
}
static double s_D;
static double s_delta;
//static double s_dt;
//static double s_dx;
//static double s_dy;
/* int u is true false flag: true if 1, means this is for calculating u; else calculate for v
    this is used for Alpha; Alpha changes depending upon calculating u or v */
void matmult(int M, int N, double** x,double **Bx, int* location, MPI_Comm comm_cart, int prt, int rank,int isU,double dx2, double dy2, double dt)
{
    double Alpha;
//    double dx2 = s_dx*s_dx;
//    double dy2 = s_dy*s_dy;

    if( isU == 1 ) {
        Alpha = s_D*s_delta*dt;
    }
    else {
        Alpha = s_delta*dt;
    }
    //double bv = 0.0; /* boundary value, problem specific */
    /* Buffers to hand left, right, up, down */
    double up[M+1], down[M+1];
    double left[N+1], right[N+1];
/* cannot initialize the boundary values the serial way,
 * because we are handing in local N, and local M,
 * the serial way of boundary value initialization will ruin data. */
    /* 1) initialize up,down,left,right buffers
     * 2) Shift and send/recv buffers
     * 3) after each shift, check for proc_null, set boundary vals
     * 4) write out Ax[i][j] */
    for(int i = 0; i <= M; i++) {
        up[i] = x[i][N-1];  
        down[i] = x[i][1];
    }
    for(int j = 0; j <= N; j++) {
        right[j] = x[M-1][j];
        left[j] = x[1][j];
    }
    /* MPI shift variables */
    int dir, source, dest, shift;
    int tag = 0;
    if(prt == 10) {
        printf("Rank: %d\tline 101 matmult\tN: %d\tM: %d\n",rank,N,M);
    }
/* For some reason, this program wasn't working. I added all of the print
 * check statements, and that somehow fixed the problem for 2x2 processor
 * layouts. (prows = pcols = 2). I have no idea why that fixed it... maybe
 * i fixed an equals sign somewhere that I forgot about..? */
    dir = 0; /* shift sideways */
    shift = 1; /* send right, receive left */
    MPI_Cart_shift(comm_cart,dir,shift,&source,&dest);
    MPI_Sendrecv_replace(&right,N+1,MPI_DOUBLE,dest,tag,source,tag,comm_cart,MPI_STATUS_IGNORE);
    for(int j = location[2]; j <= location[3]; j++) {
        if(source == MPI_PROC_NULL) { 
            /* on left edge of grid */
            if(prt == 10 && j == location[2]) {
                printf("rank: %d\tright boundary value, sending left\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            x[-1][j] = x[1][j];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\tj: %d~~~~~~~~\n"
                       "x[0][%d] =  %f\tExpected: 0.0\n",rank,j,j,x[0][j]);
            }
        }
        else {
            if(prt == 10 && j == location[2]) {
                printf("rank: %d\tleft ghost cell, sending right\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            /* ghost cell transfer */
            x[-1][j] = right[j];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\tj: %d~~~~~~~~\n"
                       " x[-1][%d] =  %f = right[%d]\n",rank,j,j,x[-1][j],j);
            }
        }
    }
    shift = -1; /* send left, receive right */
    MPI_Cart_shift(comm_cart,dir,shift,&source,&dest);
    MPI_Sendrecv_replace(&left,N+1,MPI_DOUBLE,dest,tag,source,tag,comm_cart,MPI_STATUS_IGNORE);
    for(int j = location[2]; j <= location[3]; j++) {
        if(source == MPI_PROC_NULL) {
            if(prt == 10 && j == location[2]) {
                printf("rank: %d\tright boundary value, sending left\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            /* on right edge */
            x[M+1][j] = x[M-1][j];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\tj: %d~~~~~~~~\n"
                        "x[N][%d] =  %f\tExpected: 0.0\n",rank,j,j,x[N][j]);
            }
        }
        else {
            /* ghost cell transfer */
            if(prt == 10 && j == location[2]) {
                printf("rank: %d\tright boundary ghost cell, sending left\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            x[M+1][j] = left[j];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\tj: %d~~~~~~~~\n"
                        "x[N+1][%d] =  %f\tExpected: 0.0\n",rank,j,j,x[N+1][j]);
            }
        }
    }
    /* shift vertical */
    dir = 1;
    /* send down, receive from above */
    MPI_Cart_shift(comm_cart,dir,shift,&source,&dest);
    MPI_Sendrecv_replace(&down,M+1,MPI_DOUBLE,dest,tag,source,tag,comm_cart,MPI_STATUS_IGNORE);
    for(int i = location[0]; i <= location[1]; i++) {
        if(source == MPI_PROC_NULL) {
            /* on top boundary, boundary val = 0 */
            if(prt == 10 && i == location[0]) {
                printf("rank: %d\ttop boundary value, sending down\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            x[i][N+1] = x[i][N-1];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\ti: %d~~~~~~~~\n"
                        "x[%d][M] =  %f\tExpected: 0.0\n",rank,i,i,x[i][M]);
            }
        }
        else {
            /* ghost cell transfer */
            if(prt == 10 && i == location[0]) {
                printf("rank: %d\ttop boundary value, sending down\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            x[i][N+1] = down[i];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\ti: %d~~~~~~~~\n"
                        "x[%d][M+1] =  %f\tExpected: 0.0\n",rank,i,i,x[i][M+1]);
            }
        }
    }

    shift = 1; /* send up, receive from below */
    MPI_Cart_shift(comm_cart,dir,shift,&source,&dest);
    MPI_Sendrecv_replace(&up,M+1,MPI_DOUBLE,dest,tag,source,tag,comm_cart,MPI_STATUS_IGNORE);
    for(int i = location[0]; i <= location[1]; i++) {
        if(source == MPI_PROC_NULL) {
            /* bottom boundary */
            if(prt == 10 && i == location[0]) {
                printf("rank: %d\tbottom boundary value, sending up\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            x[i][-1] = x[i][1];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\ti: %d~~~~~~~~\n"
                        "x[%d][0] =  %f\tExpected: 0.0\n",rank,i,i,x[i][0]);
            }
        }
        else {
            /* ghost cell transfer */
            if(prt != 0 && i == location[0]) {
                printf("rank: %d\tbottom boundary value, sending upward\tshift: %d\tsource: %d\n",rank,shift,source);
            }
            x[i][-1] = up[i];
            if(prt == 10) {
                printf(" ~~~~~~~~ rank: %d\ti: %d~~~~~~~~\n"
                        "x[%d][-1] =  %f\tExpected: 0.0\n",rank,i,i,x[i][-1]);
            }
        }
    }
    /* print x...? */
    if(prt == 10) {
        printf("\nrank: %d\tline 284 matmult\n",rank);
        printf("\n\nrank: %d\tloc0: %d\t loc1: %d\tloc2: %d\t:loc3: %d\n\n",rank,location[0],location[1],location[2],location[3]);
    }
    for(int i = location[0]; i <= location[1]; i++)
    {   
        for(int j = location[2]; j <= location[3]; j++)
        {
            double uxx = (x[i-1][j] - 2*x[i][j] + x[i+1][j])/dx2;
            double uyy = (x[i][j-1] - 2*x[i][j] + x[i][j+1])/dy2;
            double Ax = uxx+uyy;
            Bx[i][j] = x[i][j] - Alpha*Ax;
            /*
            if(i == j) {
                printf("\n\n\n matmult: Bx[i][j] = %f\n\n\n",Bx[i][j]);
            }
            */
        }
    }    
}

int main(int argc, char** argv)
{
   /* ------------------------------- MPI Initialization ------------------------------ */
    MPI_Init(&argc, &argv);
 
 
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);



    /* ------------------------------- Read command line ------------------------------ */
    int p = atoi(argv[1]);
    int prows = atoi(argv[2]);   /* Number of procs in x direction */
    int pcols = atoi(argv[3]);   /* Number of procs in y direction */
    int nout = atoi(argv[4]);    /* number of output steps per processor */
    // int debug = atoi(argv[5]);
    double delta = atof(argv[5]);
    s_delta = delta;
    double D = atof(argv[6]);
    s_D = D;
    double tau1 = atof(argv[7]);
    double tau2 = atof(argv[8]);
    double alpha = atof(argv[9]);
    double gamma = -1*alpha;
    double beta = atof(argv[10]);
    int prt = atoi(argv[11]);
    double Tfinal = atof(argv[12]);
    int kmax = atoi(argv[13]);
    double tol = atof(argv[14]); 
    int id = atoi(argv[15]);


#if 0
    if (debug)
    {
        mpi_debug();        
    }
#endif    

    if (prows*pcols != nprocs)
    {
        printf("\n\n\n nprocs = %d\n\n\n",nprocs);
        printf("prows*pcols != nprocs\n");
        exit(0);
    }
   
    /* ------------------------------- Other user parameters -------------------------- */
    /* Always print first and last time step, at a minimum */
    nout = (nout < 2) ? 2 : nout;

    /* Number of mesh cells in each direction, belongs to output.c */
    int Nx = pow(2,p);
    int Ny = Nx;
    

    /* number of ghost cells */
    int numGhosts = 1;
    /* --------------------------------- Communicator --------------------------------- */
    
    MPI_Comm comm_cart;
    int ndim = 2;
    int dims[2] = {prows,pcols};  /* (0,0) is lower left corner */
    int periodicity[2] = {0,0};
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims,  periodicity, reorder, &comm_cart);


    int mycoords[2];
    MPI_Cart_coords(comm_cart,rank,2,mycoords);

    /* ------------------------------- Numerical parameters --------------------------- */
    
    int Nx_local = Nx/dims[0];
    int Ny_local = Ny/dims[1];
    double L = 1;
    double ax = -L;
    double bx = L;
    double ay = -L;
    double by = L;
    double dx = (bx - ax)/Nx;
 //   s_dx = dx;
    double dx2 = dx*dx;
    double dy = (by - ay)/Ny;
//    s_dy = dy;
    double dy2 = dy*dy;
    double dt_stable = dx;
    int M = ceil(Tfinal/dt_stable) + 1;
    double w[2] = {(bx-ax)/dims[0], (by-ay)/dims[1]};
    double dt = Tfinal/M; // Tfinal/M;
//    s_dt = dt;
    
    int location[4] = {0, Nx_local, 0, Ny_local}; /* general case initially */
    /* corner cases for location[] */
#if 0  
    if(mycoords[0] == 0) {
        location[0] = 1;
    }
    if(mycoords[0] == dims[0]-1) {
        location[1] = Nx_local-1;
    }
    if(mycoords[1] == 0) {
        location[2] = 1;
    }
    if(mycoords[1] == dims[1]-1) {
        location[3] = Ny_local-1;
    }
#endif
    /* --------------------------------- Header ----------------------------------------*/
    /* Borrowed from Jenny , neat way to write different files for different nproc conditions */
    
    char filename[32];
    sprintf(filename,"%02d_%01dp_%02dprocs_spots_stripes.out",id,p,nprocs);
    MPI_File file;
    
    MPI_File_open(MPI_COMM_WORLD, filename, 
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);
    
    MPI_Datatype header_t;
    create_header_type(&header_t);
    
    struct_header_t header;    
    if (rank == 0)
    {
        header.xgrid[0] = ax;
        header.xgrid[1] = bx;
        header.ygrid[0] = ay;
        header.ygrid[1] = by;
        header.N[0] = Nx;
        header.N[1] = Ny;
        header.nout = nout;

        MPI_File_write(file,&header,1,header_t, MPI_STATUS_IGNORE);  
    }

    /* --------------------- Create view for this processor into file ------------------*/

    /* Create a file type : Data is distributed in a grid across processors */
    int ndims = 2;
    int globalsize[2] = {Nx+1,Ny+1};
    int localsize[2] = {Nx_local+1,Ny_local+1};
    int starts[2] = {mycoords[0]*Nx_local, mycoords[1]*Ny_local};
    int order = MPI_ORDER_C;  /* MPI_ORDER_FORTRAN */

    MPI_Datatype localarray_t;
    MPI_Type_create_subarray(ndims, globalsize, localsize, starts, order, MPI_DOUBLE, &localarray_t);
    MPI_Type_commit(&localarray_t);

    MPI_Datatype timeinfo_t;
    create_timeinfo_type(localarray_t,&timeinfo_t);

    MPI_Aint extent, lb;
    MPI_Type_get_extent(header_t,&lb,&extent); 
    MPI_Offset disp = extent;   /* Should be 48 bytes (4*8 + 3*4 + 4(extra)) */
    MPI_File_set_view(file, disp,  MPI_DOUBLE, localarray_t, 
                           "native", MPI_INFO_NULL);


    /* ----------------------------------- Initialize data ---------------------------- */


    double **u = allocate_2d(Nx_local,Ny_local,1);
    double **v = allocate_2d(Nx_local,Ny_local,1);
    double **output = allocate_2d(Nx_local,Ny_local,0);

    /* initializing u and v */
    for(int j = 0; j <= Ny_local; j++)
    {
        for(int i = 0; i <= Nx_local; i++)
        {
            u[i][j] = init();
            v[i][j] = init();
        }
    }
    
    /* ------------------------------------ "Time step" ------------------------------- */

    /* Write out initial time and solution */
    double t = 0;
    int k = 0; /* number of output files created */
    int *noutsteps = (int *) malloc(nout*sizeof(int));
    double dM = ((double) M-1)/(nout-1);
    dM = (dM < 1) ? 1 : dM;
    for(int m = 0; m <= nout-1; m++) {
        noutsteps[m] = (int) floor(m*dM);
    }

    MPI_Offset offset;
    MPI_Aint extent_la, lb_la;
    MPI_Type_get_extent(localarray_t,&lb_la,&extent_la); 

    /* Write out the time and reposition the file handle */
    if (rank == 0)
    {
        MPI_File_write(file,&t, 1, MPI_DOUBLE,MPI_STATUS_IGNORE); 
    }
    disp += sizeof(double);
    MPI_File_set_view(file, disp,  MPI_DOUBLE, localarray_t, 
                      "native", MPI_INFO_NULL);
    for(int j = 0; j <= Ny_local; j++) {
        for(int i = 0; i <= Nx_local; i++) {
            output[i][j] = u[i][j];
        }
    }
    
    /* Output local array and update displacement */
    int outsize = (Nx_local+1)*(Ny_local+1);
    MPI_File_write_all(file, &output[0][0], outsize, MPI_DOUBLE, MPI_STATUS_IGNORE);
    disp += extent_la;
    k++;
    /* do the work */

    double **Fu = allocate_2d(Nx_local, Ny_local,numGhosts);
    double **Fv = allocate_2d(Nx_local, Ny_local,numGhosts);

    int dir, source, dest, shift, it_cnt;
  //  int tag = 0;

    for(int n = 1; n < M; n++)
    {
        t += dt;
        for(int i = 0; i <= Nx_local; i++) {
            for(int j = 0; j <= Ny_local; j++) {
                double uij = u[i][j];
                double vij = v[i][j];
                Fu[i][j] = uij + dt*rhs_u(uij,vij,tau1,tau2,alpha);
                Fv[i][j] = vij + dt*rhs_v(uij,vij,beta,alpha,tau1,gamma,tau2);
            }
        }
        
#if 1

    if(mycoords[1] == 0) {
        //noflux boundary on bottom with Fu and Fv;
        for(int i = 0; i <= Nx_local; i++) {
            Fu[i][-1] = Fu[i][1];
            Fv[i][-1] = Fv[i][1];
        }
    }
    if(mycoords[1] == Ny) {
        //noflux boundary on top with Fu and Fv;
        for(int i = 0; i <= Nx_local; i++) {
            Fu[i][Ny_local+1] = Fu[i][Ny_local-1];
            Fv[i][Ny_local+1] = Fv[i][Ny_local-1];
        }
    }
    if(mycoords[0] == 0) {
        //noflux boundary on left with Fu and Fv;
        for(int j = 0; j <= Ny_local; j++) {
            Fu[-1][j] = Fu[1][j];
            Fv[-1][j] = Fv[1][j];
        }
    }
    if(mycoords[0] == Nx) {
        //noflux boundary on right with Fu and Fv;
        for(int j = 0; j <= Ny_local; j++) {
            Fu[Nx_local+1][j] = Fu[Nx_local-1][j];
            Fv[Nx_local+1][j] = Fv[Nx_local-1][j];
        }
    }

#endif

#if 0
/* if on boundary exterior boundary, we do this, else we dont */
        for(int j = 0; j <= Ny_local; j++) { 
            Fu[-1][j] = Fu[1][j];
            Fu[Nx_local+1][j] = Fu[Nx_local-1][j];
            Fv[-1][j] = Fv[1][j];
            Fv[Nx_local+1][j] = Fv[Nx_local-1][j];
        }
        for(int i = 0; i <= Nx_local; i++) {
            Fu[i][-1] = Fu[i][1];
            Fu[i][Ny_local+1] = Fu[i][Ny_local-1];
            Fv[i][-1] = Fv[i][1];
            Fv[i][Ny_local+1] = Fv[i][Ny_local-1];
        }
#endif
        /* update routine */
        
        cg_2d(Nx_local, Ny_local,numGhosts,kmax,tol,prt,matmult,Fu,u,&it_cnt,location,comm_cart,rank,1,dx2,dy2,dt);
        cg_2d(Nx_local, Ny_local,numGhosts,kmax,tol,prt,matmult,Fv,v,&it_cnt,location,comm_cart,rank,0,dx2,dy2,dt);
        
        /* Write out the time and reposition the file handle */
        
        if(n == noutsteps[k]) { 
            if (prt == 1) {
                printf("\n\tline 591 in driver: n = %d\toutsize: %d\n\n",n,outsize);
            }
            if (rank == 0)
            {
                MPI_File_write(file,&t, 1, MPI_DOUBLE,MPI_STATUS_IGNORE); 
            }
            disp += sizeof(double);
            MPI_File_set_view(file, disp,  MPI_DOUBLE, localarray_t, 
                          "native", MPI_INFO_NULL); 
            /* update routine */
            for(int j = 0; j <= Ny_local; j++) {
                for(int i = 0; i <= Nx_local; i++) {
                    output[i][j] = u[i][j];
                }
            }
            
            /* Collective write; update file displacement */
            MPI_File_write_all(file, &output[0][0], outsize, MPI_DOUBLE, MPI_STATUS_IGNORE);
            disp += extent_la; 
            k++;
        }
        /* swap pointers */
        /* section removed because cg does this */
    }
    /* ------------------------------- Clean up --------------------------------------- */

    MPI_Type_free(&localarray_t);
    MPI_Type_free(&header_t);

    delete_2d(numGhosts,(double***) &u);
    delete_2d(numGhosts,(double***) &v);
    delete_2d(numGhosts, (double***) &Fu);
    delete_2d(numGhosts, (double***) &Fv);
    delete_2d(0,(double***) &output);

    MPI_File_close(&file);

    MPI_Finalize();
    free(noutsteps);
    return 0;
}