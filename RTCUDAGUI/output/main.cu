#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
void checkCudaError(const char *msg)
{
        cudaError_t err = cudaGetLastError();
        if(cudaSuccess != err){
                printf("%s(%i) : CUDA error : %s : (%d) %s\n", __FILE__, __LINE__, msg, (int)err, cudaGetErrorString(err));
                exit (-1);
        }
}
#include "rcuda.h"
#include "kernel1params.h"
#include "kernel1.cu"
#include "kernel2params.h"
#include "kernel2.cu"

int main(int argc,char *argv[]){
	int N=1024;
	int GPU=2;
	if(argc>1)N=atoi(argv[1]);
	
	if(argc>2)GPU=atoi(argv[2]);
	
	cudaSetDevice (GPU);
	float *A,*B,*C;
	int memsize=N*N*sizeof(float );
	cudaMallocManaged(&A,memsize);
	
	cudaMallocManaged(&B,memsize);
	
	cudaMallocManaged(&C,memsize);
	
	dim3 threads(matrix_addBLOCKSIZE,1);
	dim3 grid(N*N/matrix_addBLOCKSIZE/matrix_addMERGE_LEVEL/matrix_addSKEW_LEVEL,1);
	matrix_add<<<grid,threads>>>(C,A,B,N);
	cudaDeviceSynchronize();
	
	checkCudaError("matrix_add error: ");
	
	dim3 threads(matrix_subBLOCKSIZE,1);
	dim3 grid(N*N/matrix_subBLOCKSIZE/matrix_subMERGE_LEVEL/matrix_subSKEW_LEVEL,1);
	matrix_sub<<<grid,threads>>>(C,A,B,N);
	cudaDeviceSynchronize();
	
	checkCudaError("matrix_sub error: ");
	
	cudaFree (A);
	cudaFree (B);
	cudaFree (C);
	cudaThreadExit();
	
}
