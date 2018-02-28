void matrix_sub(float *C, float * restrict A, float * restrict B, int N)
{
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            C[i][j] = A[i][j] - B[i][j];
}
