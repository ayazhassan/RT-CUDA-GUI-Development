__global__ void matrix_add(float *C,float const *__restrict__ A,float const *__restrict__ B,int N){
	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int ij=bid*matrix_addBLOCKSIZE+tid;
	{
		int i=(ij/N)*matrix_addMERGE_LEVEL;
		int j=(ij%N)*matrix_addSKEW_LEVEL;
		for(int m=0;m<matrix_addMERGE_LEVEL;m++)
			for(int n=0;n<matrix_addSKEW_LEVEL;n++)
				C[((i+m))*N+((j+n))]=A[((i+m))*N+((j+n))]+B[((i+m))*N+((j+n))];
				
	}
}
