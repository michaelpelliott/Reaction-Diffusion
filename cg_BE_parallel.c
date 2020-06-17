#include <mpi.h>
#include <math.h>
#include <stdlib.h>   /* For atoi */
#include <stdio.h>

#include "cg_BE_parallel.h"
/* cg_2d only needs rank for printing */
void cg_2d(int N, int M, int mbc, int kmax, double tol, int prt, matmult_t matmult, double **F, double **u, int* it_cnt, int *location, MPI_Comm comm_cart, int rank,int isU, double dx2, double dy2, double dt)
{

    /* ----------------------------------------------------------------
       Set up arrays and other vars needed for iterative method
    ---------------------------------------------------------------- */
    double **Auk  = allocate_2d(N,M,mbc);
    double **rk   = allocate_2d(N,M,mbc);
    double **rkp1 = allocate_2d(N,M,mbc);
    double **uk   = allocate_2d(N,M,mbc);
    double **ukp1 = allocate_2d(N,M,mbc);
    double **pk   = allocate_2d(N,M,mbc);
    double **wk   = allocate_2d(N,M,mbc);


    /* ----------------------------------------------------------------
       Start iterations
    ---------------------------------------------------------------- */
    for(int i = 0; i < N+1; i++)
    {
        for(int j = 0; j < M+1; j++) {
            uk[i][j] = 0;    /* or use memset */
        }
    }
     /* matmult usage: int N, double** x, double **Ax, int* location,
     *          MPI_comm comm_cart, int rank */
    matmult(N,M,uk,Auk,location,comm_cart,prt,rank,isU,dx2,dy2,dt);
    for(int i = location[0]; i <= location[1]; i++)
    {   
        for(int j = location[2]; j <= location[3]; j++) 
        {
            /* Compute residual rk = F - A*uk */
            rk[i][j] = F[i][j] - Auk[i][j];
            pk[i][j] = rk[i][j];
        }
    }        

    *it_cnt = 0;
    if(prt != 0) {
        //printf("rank: %d\tlocation: %d %d %d %d\t\n",rank,location[0],location[1],location[2],location[3]);
    }
    for(int k = 0; k < kmax; k++)
    {
        matmult(N,M,pk,wk,location,comm_cart,prt,rank,isU,dx2,dy2,dt);

        double a[2] = {0,0};
        for(int i = location[0]; i <= location[1]; i++)
        {
            for(int j = location[2]; j <= location[3]; j++) {
                a[0] += rk[i][j]*rk[i][j];
                a[1] += pk[i][j]*wk[i][j];
            }
        }
        double aSum[2];
        MPI_Allreduce(&a,&aSum,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        double alpha = aSum[0]/aSum[1];

        double b[2] = {0,0};
        double norm_zk = 0, zk;
        for(int i = location[0]; i <= location[1]; i++)
        {
            for(int j = location[2]; j <= location[3]; j++) {
                zk = alpha*pk[i][j];
                ukp1[i][j] = uk[i][j] + zk;
                rkp1[i][j] = rk[i][j] - alpha*wk[i][j];
                b[0] += rkp1[i][j]*rkp1[i][j];
                b[1] += rk[i][j]*rk[i][j];
                norm_zk = fabs(zk) > norm_zk ? fabs(zk) : norm_zk;
            }
        }
        double bSum[2];
        MPI_Allreduce(&b,&bSum,2, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        double beta = bSum[0]/bSum[1];

        double norm_zkMax;
        MPI_Allreduce(&norm_zk,&norm_zkMax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); 
        if(prt == 9 && rank !=0) {
            printf("rank: %d\tk: %d\tnorm_zk: %16.8e\tnorm_zkMax: %16.8e\n",rank,k,norm_zk,norm_zkMax);
        }
        /* print out if prt != 0 */
        if(prt != 0 && rank == 0) {
            /* serial prints k, norm_zk,
             * parallel prints k, norm_zk, norm_zkMax */
            printf("\n\n cg_BE_parallel line 91\n");
            printf("%8d %16.8e %16.8e\n",k,norm_zk,norm_zkMax);
        }
        /* save results for output */
        *it_cnt = k+1;
        for(int i = location[0]; i <= location[1]; i++)
        {
            for(int j = location[2]; j <= location[3]; j++) {
                pk[i][j] = rkp1[i][j] + beta*pk[i][j];
                rk[i][j] = rkp1[i][j];
                uk[i][j] = ukp1[i][j];
            }
        }
        if (norm_zkMax < tol)
        {
            break;
        }
    }
    
    for(int i = location[0]; i <= location[1]; i++)
    {
        for(int j = location[2]; j <= location[3]; j++) {
            u[i][j] = uk[i][j];
        }
    }
    
    delete_2d(mbc,(double***) &Auk);
    delete_2d(mbc,(double***) &rk);
    delete_2d(mbc,(double***) &rkp1);
    delete_2d(mbc,(double***) &uk);
    delete_2d(mbc,(double***) &ukp1);
    delete_2d(mbc,(double***) &pk);
    delete_2d(mbc,(double***) &wk);
}