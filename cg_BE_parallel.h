#ifndef CG_SERIAL_H
#define CG_SERIAL_H


#ifdef __cplusplus
extern "C"
{
#if 0
}
#endif
#endif

typedef void (*matmult_t)(int N, int M,double** x, double **Ax, int *location, MPI_Comm comm_cart,int prt, int rank, int isU, double dx2, double dy2, double dt);

void cg_2d(int N, int M, int mbc, int kmax, double tol, int prt, matmult_t matmult,
        double **F, double **u, int* it_cnt, int *location, MPI_Comm comm_cart,int rank, int isU,double dx2, double dy2, double dt);

double** allocate_2d(int N, int M, int numGhosts);

void delete_2d(int mbc, double*** q);

double double_rand(double min, double max);

double init();

double rhs_v(double u, double v, double beta, double alpha, double t1, double gamma, double t2);

double rhs_u(double u, double v, double t1, double t2, double alph);

#ifdef __cplusplus
#if 0
{
#endif
}
#endif


#endif