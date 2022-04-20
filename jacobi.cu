#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define BLOCK_SIZE 32


__global__ void mmul_kernel(double* new_X, double* X, long N){
  
  int r = (blockIdx.y) * blockDim.y + threadIdx.y;
  int c = (blockIdx.x) * blockDim.x + threadIdx.x;
  double h= (1.0/N)*(1.0/N);
  double sum = 0.0;
  if(r > 0 && c> 0 && r < N-1 && c < N-1) 
    sum = 0.25*(h+X[r*N+(c-1)]+X[r*N+(c+1)]+X[(r-1)*N+c]+X[(r+1)*N+c]);
  new_X[r*N + c] = sum;   
}

int main() {
  long N = 1024;//(1UL<<11)+1;  
  double *X, *new_X;
  cudaMallocHost((void**)&X, N*N* sizeof(double));
  cudaMallocHost((void**)&new_X, N*N* sizeof(double));
  #pragma omp parallel for schedule(static) collapse(2)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
        X[i*N+j] = 0.0;
        new_X[i*N+j] = 0.0;
    }
  }

  double *X_d, *new_X_d;
  cudaMalloc(&X_d, N*N*sizeof(double));
  cudaMalloc(&new_X_d, N*N*sizeof(double));

  cudaMemcpyAsync(X_d, X, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(new_X_d, new_X_d, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double tt = omp_get_wtime();

  int block = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  dim3 Blocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Grids(block,block);

  #pragma omp parallel for schedule(static) collapse(1)
  for (int i=0; i<=100; i++){
    if(i%2==0) mmul_kernel<<<Grids,Blocks>>>(new_X_d, X_d, N);
    else mmul_kernel<<<Grids,Blocks>>>(X_d, new_X_d, N);

  }

  cudaMemcpyAsync(new_X, new_X_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  
  double err = 0.0;
  //#pragma omp parallel for schedule(static) collapse(2)
  for (long i = 0; i < N; i++) {
    int tmp=0;
    for (long j = 0; j < N; j++) {
        tmp += new_X[i*N+j];
    }
    err += abs(tmp);
  }


  printf("Error = %f\n", err);
  cudaFree(X_d);
  cudaFree(new_X_d);
  cudaFreeHost(X);
  cudaFreeHost(new_X);

  return 0;
}
