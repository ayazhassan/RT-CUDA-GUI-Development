int main(int argc, char *argv[]) {
    int N = 1024;
    int GPU = 2;
    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        GPU = atoi(argv[2]);
    cudaSetDevice(GPU);

    float *A, *B, *C;
    int memsize = N*N*sizeof(float);
    A = (float *)malloc(memsize);
    B = (float *)malloc(memsize);
    C = (float *)malloc(memsize);

    matrix_add(C, A, B, N);
    matrix_sub(C, A, B, N);
    
    free(A);
    free(B);
    free(C);
    exit(0);
}
