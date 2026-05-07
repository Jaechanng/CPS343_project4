/*
 * This program reads a matrix from an HDF5 file and estimates the dominant eigenvalue using the power
 * method. This version uses the CUBLAS library for the matrix-vector multiply and vector operations.
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>   
#include <cstdlib>    
#include <cstdio>
#include <cstring>
#include <cmath>
#include "wtime.c"
#include <hdf5.h>
extern "C" {
    #include <cblas.h>
}
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda.h>

// Check return values from HDF5 routines
#define CHKERR(status, name) \
    if ((status) < 0) { \
        fprintf(stderr, "Error: failure in %s\n", name); \
        exit(EXIT_FAILURE); \
    }

// Macro to index matrices in column-major (Fortran) order
#define IDX(i,j,stride) ((i)+(j)*(stride))  // column major

// Alternative macro for 1-based indexing (Fortran style)
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void  read_matrix_hdf5(char* fname, const char* name, double** A, int& n) {
    // read the matrix from the file and store it in column-major order
    hid_t file_id, dataset_id, file_dataspace_id;
    herr_t status;
    hsize_t dims[2];
    int rank;
    int ndims;
    hsize_t num_elem;

    //open the file
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHKERR(file_id, "H5Fopen");

    //open the dataset
    dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
    CHKERR(dataset_id, "H5Dopen");

    //dermine the dimensions of the dataset
    file_dataspace_id = H5Dget_space(dataset_id);
    CHKERR(file_dataspace_id, "H5Dget_space");
    rank  = H5Sget_simple_extent_ndims(file_dataspace_id);
    ndims = H5Sget_simple_extent_dims(file_dataspace_id, dims, nullptr);
    if (rank != 2) {
        fprintf(stderr, "Error: dataset is not 2-dimensional\n");
        exit(EXIT_FAILURE);
    }
    if(ndims < 0) {
        fprintf(stderr, "Error: unable to determine the dimensions of the dataset\n");
        exit(EXIT_FAILURE);
    }

    
    //allocate memory for the matrix
    num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
    *A = new double[num_elem];
    if(dims[0] != dims[1]) {
        fprintf(stderr, "Error: dataset is not square\n");
        exit(EXIT_FAILURE);
    }
    n = (int)dims[0];

    //read the dataset into the matrix
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *A); 
    CHKERR(status, "H5Dread");

    //close resources
    status = H5Sclose(file_dataspace_id); CHKERR(status, "H5Sclose");
    status = H5Dclose(dataset_id); CHKERR(status, "H5Dclose");
    status = H5Fclose(file_id); CHKERR(status, "H5Fclose");
}

void power_method(double* A, int n, double tol, int max_iter, double* lambda, int* iters, double* iter_time, int verbose) {
    // create cuBLAS context
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // allocate GPU memory
    double *d_A, *x_d, *y_d;
    cudaMalloc((void**)&d_A, n*n*sizeof(double));
    cudaMalloc((void**)&x_d, n*sizeof(double));
    cudaMalloc((void**)&y_d, n*sizeof(double));
    
    // copy matrix A to device
    cublasSetMatrix(n, n, sizeof(double), A, n, d_A, n);
    
    // initialize x on host, then copy to device
    double* x = new double[n];
    for(int i = 0; i < n; i++) {
        x[i] = 1.0;
    }
    cublasSetVector(n, sizeof(double), x, 1, x_d, 1);
    delete[] x;  // host x is no longer needed after this point
    
    // normalize x
    double normx;
    cublasDnrm2(handle, n, x_d, 1, &normx);
    double alpha = 1.0 / normx;
    cublasDscal(handle, n, &alpha, x_d, 1);
    
    // initialize values for eigenvalue tracking
    double lambda_new = 0.0;
    double lambda_old = lambda_new + 2.0 * tol;
    double delta = fabs(lambda_new - lambda_old);
    int iter = 0;

    // start the timer
    double start_time = wtime();

    // power iteration loop
    const double one = 1.0;
    const double zero = 0.0;
    
    while(delta >= tol && iter <= max_iter) {
        iter++;
        
        // compute y = A*x
        cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, d_A, n, x_d, 1, &zero, y_d, 1);

        // update eigenvalue estimate
        lambda_old = lambda_new;
        cublasDdot(handle, n, x_d, 1, y_d, 1, &lambda_new);

        // compute norm of y
        double norm_y;
        cublasDnrm2(handle, n, y_d, 1, &norm_y);
        
        // x = y / norm_y
        cublasDcopy(handle, n, y_d, 1, x_d, 1);
        double scale = 1.0 / norm_y;
        cublasDscal(handle, n, &scale, x_d, 1);

        delta = fabs(lambda_new - lambda_old);
        if (verbose) {
            printf("%3d: lambda = %12.9f, delta = %.4e\n", iter, lambda_new, delta);
        }
    }
    
    // stop the timer
    double end_time = wtime();

    // report results
    *iter_time = end_time - start_time;
    *lambda = lambda_new;
    *iters = iter;

    // free device memory
    cudaFree(d_A);
    cudaFree(x_d);
    cudaFree(y_d);
    
    cublasDestroy(handle);
}

int usage(char *progname) {
    fprintf(stderr, "Usage: %s [-q] [-v] [-e tol] [-m maxiter] filename\n", progname);
    return EXIT_FAILURE;
}



int main(int argc, char *argv[]) {
    // read the matrix from the file
    // estimate the dominant eigenvalue using the power method
    // output the results
    double* A; // pointer to the matrix stored in column-major order
    int n; // number of rows in the matrix
    
    char* filename = NULL; // name of the file containing the matrix
    int verbose = 0; // flag for verbose output
    double tol = 1e-6; // tolerance for convergence
    int max_iter = 1000; // maximum number of iterations
    int quiet = 0; // flag for quiet output

    //parse the command line arguments
    int ch;
    while((ch = getopt(argc, argv, "e:m:qv")) != -1) {
        switch(ch) {
            case 'e':
                tol = atof(optarg);
                break;
            case 'm':
                max_iter = atoi(optarg);
                break;
            case 'q':
                quiet = 1;
                break;  
            case 'v':
                verbose = 1;
                break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }
    
    if (optind >= argc) {
    usage(argv[0]);
    return EXIT_FAILURE;
    }
    filename = argv[optind];

    double start_time = wtime();
    read_matrix_hdf5(filename, "/A/value", &A, n);
    double end_time = wtime();
    double read_time = end_time - start_time;

    // //print matrix
    // printf("Matrix A:\n");
    // dumpMatrix(A, n, n, n);

    //call the power method
    //this is where I need to call the kernel function 
    double lambda;
    int iters;
    double compute_time;
    power_method(A, n, tol, max_iter, &lambda, &iters, &compute_time, verbose);

    //report
    //handle non-convergence
    if (iters > max_iter) {
        fprintf(stderr, "*** WARNING ****: maximum number of iterations exceeded\n");
    }

    if (quiet) {
        printf("%5d x %5d %5d %10.6f %10.6f %10.6f\n", n, n, iters, lambda, read_time, compute_time);
    } else {
        printf("matrix dimensions: %d x %d; tolerance: %8.4e; max iterations: %d\n",
            n, n, tol, max_iter);
        printf("elapsed HDF5 read time = %10.6f seconds\n", read_time);
        printf("elapsed compute time   = %10.6f seconds\n", compute_time);
        printf("eigenvalue = %.6f found in %d iterations\n", lambda, iters);
    }

    delete[] A;
}
