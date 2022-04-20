#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define BLOCK_SIZE 32  

__global__ void mmkernel(double* m, double* A, const double* B, long N){
  
  int r = (blockIdx.y) * blockDim.y + threadIdx.y;
  int c = (blockIdx.x) * blockDim.x + threadIdx.x;
  double tmp = 0.0;
  if(r < N && c < N) {
    for(int i=0; i<N; i++) {
        tmp += A[r*N+i]*B[i*N+c];
    }
  }
  m[r*N + c] = tmp;
}

int main() {
  long N = 2048;   
  double *A, *B, *C, *C0;
  cudaMallocHost((void**)&A, N*N* sizeof(double));
  cudaMallocHost((void**)&B, N*N* sizeof(double));
  cudaMallocHost((void**)&C, N*N* sizeof(double));
  cudaMallocHost((void**)&C0, N*N* sizeof(double));
  #pragma omp parallel for schedule(static) collapse(2)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
        A[i*N+j] = 1.0/(i+1);
        B[i*N+j] = 1.0/(i+1);
    }
  }
  

  double tt = omp_get_wtime();
  #pragma omp parallel for schedule(static) collapse(2)
    for(int row=0; row<N; row++) {
      for(int col=0; col<N; col++) {
          double sum = 0.00;
          for (long i = 0; i < N; i++) 
              sum += A[row*N+i] * B[col+i*N];
          C0[row*N+col] = sum;
      }
    }
  printf("CPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *a, *b, *c;
  cudaMalloc(&a, N*N*sizeof(double));
  cudaMalloc(&b, N*N*sizeof(double));
  cudaMalloc(&c, N*N*sizeof(double));

  cudaMemcpyAsync(a, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(b, B, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  int block = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //dim 3 Pre-defined variables
  dim3 Blocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Grids(block,block);
  mmkernel<<<Grids,Blocks>>>(c, a, b, N);

  cudaMemcpyAsync(C, c, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  double err = 0.0; 

  #pragma omp parallel for schedule(static) collapse(2)
  for(int row=0; row<N; row++) {
    for(int col=0; col<N; col++) {
        double tmp = fabs(C[row*N+col]-C0[row*N+col]);
        err += tmp;
    }
  }
  printf("Error = %f\n", err);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
  cudaFreeHost(C0);
  
  return 0;
}
