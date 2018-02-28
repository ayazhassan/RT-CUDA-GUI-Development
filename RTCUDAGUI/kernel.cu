__global__ void matrix_add(float *C,float const *__restrict__ A,float const *__restrict__ B,int N){
	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int ij=bid*BLOCKSIZE+tid;
	{
		int i=(ij/N)*MERGE_LEVEL;
		int j=(ij%N)*SKEW_LEVEL;
		for(int m=0;m<MERGE_LEVEL;m++)
			for(int n=0;n<SKEW_LEVEL;n++)
				C[((i+m))*N+((j+n))]=A[((i+m))*N+((j+n))]+B[((i+m))*N+((j+n))];
				
	}
}
