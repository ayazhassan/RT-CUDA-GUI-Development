#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "mmio.h"

#define RTCSR 0
#define RTBSR 1
#define RTHYB 2

cublasStatus_t _rtcuda_cublas_status;
cublasHandle_t _rtcuda_cublas_handle;
cusparseHandle_t _rtcuda_cusparse_handle;
cusparseMatDescr_t _rtcuda_matrix_descr=0;
cusparseStatus_t _rtcuda_cusparse_status;
cudaEvent_t _rtcuda_time_event_start, _rtcuda_time_event_stop;

typedef struct{
    int *row_indices;
    int *column_indices;
    float *values;
    int nnz;
} RTspSArray;

typedef struct{
    int *row_indices;
    int *column_indices;
    double *values;
    int nnz;    
} RTspDArray;

void RTAPIInit(){
    _rtcuda_cublas_status = cublasCreate(&_rtcuda_cublas_handle);
    switch(_rtcuda_cublas_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTAPIInit error: the CUDA Runtime initialization failed.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTAPIInit error: the resources could not be allocated.\n");
            exit(1);
            break;
    }
    
    _rtcuda_cusparse_status = cusparseCreate(&_rtcuda_cusparse_handle);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTAPIInit error: the CUDA Runtime initialization failed.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTAPIInit error: the resources could not be allocated.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTAPIInit error: the device compute capability (CC) is less than 1.1. The CC of at least 1.1 is required.\n");
            exit(1);
            break;
    }
    _rtcuda_cusparse_status = cusparseCreateMatDescr(&_rtcuda_matrix_descr);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTAPIInit error: the resources could not be allocated.\n");
            exit(1);
            break;
    }
    _rtcuda_cusparse_status = cusparseSetMatType(_rtcuda_matrix_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTAPIInit error: An invalid type parameter was passed.\n");
            exit(1);
            break;
    }
    _rtcuda_cusparse_status = cusparseSetMatIndexBase(_rtcuda_matrix_descr,CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTAPIInit error: An invalid base parameter was passed.\n");
            exit(1);
            break;
    }    
    cudaEventCreate(&_rtcuda_time_event_start);
    cudaEventCreate(&_rtcuda_time_event_stop);
}

void RTAPIFinalize(){
    _rtcuda_cublas_status = cublasDestroy(_rtcuda_cublas_handle);
    switch(_rtcuda_cublas_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTAPIFinalize error: the library was not initialized.\n");
            exit(1);
            break;
    }
    
    _rtcuda_cusparse_status = cusparseDestroyMatDescr(_rtcuda_matrix_descr);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
    }    
    _rtcuda_cusparse_status = cusparseDestroy(_rtcuda_cusparse_handle);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTAPIFinalize error: the library was not initialized.\n");
            exit(1);
            break;
    }
    cudaEventDestroy(_rtcuda_time_event_start);
    cudaEventDestroy(_rtcuda_time_event_stop);
}

void RTdSMM(float *C, const float *A, const float *B, int m, int n, int k){
    const float alpha = 1.0;
    const float beta = 0.0;
    _rtcuda_cublas_status = cublasSgemm(_rtcuda_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B, k, A, m, &beta, C, m);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMM error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdSMM error: the parameters m,n,k<0");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdSMM error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdSMM error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdDMM(double *C, const double *A, const double *B, int m, int n, int k){
    const double alpha = 1.0;
    const double beta = 0.0;
    _rtcuda_cublas_status = cublasDgemm(_rtcuda_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B, k, A, m, &beta, C, m);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMM error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdDMM error: the parameters m,n,k<0");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdDMM error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdDMM error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdSMV(float *C, const float *A, const float *B, int m, int n){
    const float alpha = 1.0;
    const float beta = 0.0;
    _rtcuda_cublas_status = cublasSgemv(_rtcuda_cublas_handle, CUBLAS_OP_T, m, n, &alpha, A, m, B, 1, &beta, C, 1);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMV error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdSMV error: the parameters m,n<0 or incx,incy=0");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdSMV error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdSMV error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdDMV(double *C, const double *A, const double *B, int m, int n){
    const double alpha = 1.0;
    const double beta = 0.0;
    _rtcuda_cublas_status = cublasDgemv(_rtcuda_cublas_handle, CUBLAS_OP_T, m, n, &alpha, A, m, B, 1, &beta, C, 1);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMV error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdDMV error: the parameters m,n<0 or incx,incy=0");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdDMV error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdDMV error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdSMT(float *C, const float *A, int m, int n){
    const float alpha = 1.0;
    const float beta = 0.0;
    _rtcuda_cublas_status = cublasSgeam(_rtcuda_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, m, &beta, A, m, C, m);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMT error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdSMT error: the parameters m,n<0, alpha,beta=NULL or improper settings of in-place mode");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdSMT error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdSMT error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdDMT(double *C, const double *A, int m, int n){
    const double alpha = 1.0;
    const double beta = 0.0;
    _rtcuda_cublas_status = cublasDgeam(_rtcuda_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, m, &beta, A, m, C, m);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMT error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdDMT error: the parameters m,n<0, alpha,beta=NULL or improper settings of in-place mode");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdDMT error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdDMT error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdSVV(float *C, const float *A, int m, const float *B, int n){
    const float alpha = 1.0;
    const float beta = 0.0;
    _rtcuda_cublas_status = cublasSgemm(_rtcuda_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 1, &alpha, A, m, B, n, &beta, C, m);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSVV error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdSVV error: the parameters m,n,k<0");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdSVV error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdSVV error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdDVV(double *C, const double *A, int m, const double *B, int n){
    const double alpha = 1.0;
    const double beta = 0.0;
    _rtcuda_cublas_status = cublasDgemm(_rtcuda_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 1, &alpha, A, m, B, n, &beta, C, m);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDVV error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("RTdDVV error: the parameters m,n,k<0");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdDVV error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdDVV error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdSDOT(const float *C, const float *A, int n, float *r){
    _rtcuda_cublas_status = cublasSdot(_rtcuda_cublas_handle, n, C, 1, A, 1, r);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSDOT error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RTdSDOT error: the reduction buffer could not be allocated");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdSDOT error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdSDOT error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTdDDOT(const double *C, const double *A, int n, double *r){
    _rtcuda_cublas_status = cublasDdot(_rtcuda_cublas_handle, n, C, 1, A, 1, r);
    switch(_rtcuda_cublas_status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDDOT error: the library was not initialized");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RTdDDOT error: the reduction buffer could not be allocated");
            exit(1);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("RTdDDOT error: the device does not support double-precision");
            exit(1);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("RTdDDOT error: the function failed to launch on the GPU");
            exit(1);
            break;
    }
}

void RTspSArrayCreate(float *A, RTspSArray *array, int m, int n){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA,(m+1)*sizeof(int));
    int *nnzPerRowA;
    cudaMalloc((void **)&nnzPerRowA, m*sizeof(int));
    int nnzA;
    _rtcuda_cusparse_status = cusparseSnnz(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, n, _rtcuda_matrix_descr, A, m, nnzPerRowA, &nnzA);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSArrayCreate error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspSArrayCreate error: the resources could not be allocated.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSArrayCreate error: invalid parameters were passed (m, n<0).\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspSArrayCreate error: the device does not support double precision.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSArrayCreate error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspSArrayCreate error: an internal operation failed.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspSArrayCreate error: the matrix type is not supported.\n");
            exit(1);
            break;
    }
    array = (RTspSArray *)malloc(sizeof(RTspSArray));
    cudaMalloc((void **)&(array->values), nnzA*sizeof(float));
    cudaMalloc((void **)&(array->column_indices), nnzA*sizeof(int));
    _rtcuda_cusparse_status = cusparseSdense2csr(_rtcuda_cusparse_handle, m, n, _rtcuda_matrix_descr, A, m, nnzPerRowA, array->values, csrRowPtrA, array->column_indices);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSArrayCreate error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspSArrayCreate error: the resources could not be allocated.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSArrayCreate error: invalid parameters were passed (m, n<0).\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspSArrayCreate error: the device does not support double precision.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSArrayCreate error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspSArrayCreate error: the matrix type is not supported.\n");
            exit(1);
            break;
    }
    cudaMalloc((void **)&(array->row_indices), nnzA*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcsr2coo(_rtcuda_cusparse_handle, csrRowPtrA, nnzA, m, array->row_indices, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSArrayCreate error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSArrayCreate error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSArrayCreate error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    array->nnz = nnzA;
    cudaFree(csrRowPtrA);
    cudaFree(nnzPerRowA);
}

void RTspDArrayCreate(double *A, RTspDArray *array, int m, int n){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA,(m+1)*sizeof(int));
    int *nnzPerRowA;
    cudaMalloc((void **)&nnzPerRowA, m*sizeof(int));
    int nnzA;
    _rtcuda_cusparse_status = cusparseDnnz(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, n, _rtcuda_matrix_descr, A, m, nnzPerRowA, &nnzA);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDArrayCreate error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDArrayCreate error: the resources could not be allocated.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDArrayCreate error: invalid parameters were passed (m, n<0).\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDArrayCreate error: the device does not support double precision.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDArrayCreate error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspDArrayCreate error: an internal operation failed.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDArrayCreate error: the matrix type is not supported.\n");
            exit(1);
            break;
    }
    array = (RTspDArray *)malloc(sizeof(RTspDArray));
    cudaMalloc((void **)&(array->values), nnzA*sizeof(double));
    cudaMalloc((void **)&(array->column_indices), nnzA*sizeof(int));
    _rtcuda_cusparse_status = cusparseDdense2csr(_rtcuda_cusparse_handle, m, n, _rtcuda_matrix_descr, A, m, nnzPerRowA, array->values, csrRowPtrA, array->column_indices);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDArrayCreate error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDArrayCreate error: the resources could not be allocated.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDArrayCreate error: invalid parameters were passed (m, n<0).\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDArrayCreate error: the device does not support double precision.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDArrayCreate error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDArrayCreate error: the matrix type is not supported.\n");
            exit(1);
            break;
    }
    cudaMalloc((void **)&(array->row_indices), nnzA*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcsr2coo(_rtcuda_cusparse_handle, csrRowPtrA, nnzA, m, array->row_indices, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDArrayCreate error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDArrayCreate error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDArrayCreate error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    array->nnz = nnzA;
    cudaFree(csrRowPtrA);
    cudaFree(nnzPerRowA);
}

void RTspSArrayLoadFromFile(char *filename, RTspSArray *array){
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i;
    
    if ((f = fopen(filename, "r")) == NULL) 
            exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("RTspSArrayLoadFromFile error: Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("RTspSArrayLoadFromFile error: Sorry, this application does not support ");
        printf("RTspSArrayLoadFromFile error: Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    if(N == 1)
            exit(0);

    array->column_indices = (int *) malloc(nz * sizeof(int));
    array->row_indices = (int *) malloc(nz * sizeof(int));
    array->values = (float *) malloc(nz * sizeof(float));
    array->nnz = nz;

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &(array->column_indices[i]), &(array->row_indices[i]), &(array->values[i]));
        array->column_indices[i]--;  /* adjust from 1-based to 0-based */
        array->row_indices[i]--;
    }

    if (f !=stdin) fclose(f);
}

void RTspDArrayLoadFromFile(char *filename, RTspDArray *array){
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i;
    
    if ((f = fopen(filename, "r")) == NULL) 
            exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("RTspSArrayLoadFromFile error: Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("RTspSArrayLoadFromFile error: Sorry, this application does not support ");
        printf("RTspSArrayLoadFromFile error: Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    if(N == 1)
            exit(0);

    array->column_indices = (int *) malloc(nz * sizeof(int));
    array->row_indices = (int *) malloc(nz * sizeof(int));
    array->values = (double *) malloc(nz * sizeof(double));
    array->nnz = nz;

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &(array->column_indices[i]), &(array->row_indices[i]), &(array->values[i]));
        array->column_indices[i]--;  /* adjust from 1-based to 0-based */
        array->row_indices[i]--;
    }

    if (f !=stdin) fclose(f);
}

void RTspSArrayDestroy(RTspSArray *array){
    cudaFree(array->column_indices);
    cudaFree(array->row_indices);
    cudaFree(array->values);
}

void RTspDArrayDestroy(RTspDArray *array){
    cudaFree(array->column_indices);
    cudaFree(array->row_indices);
    cudaFree(array->values);
}

void RTspSMM(RTspSArray *C, RTspSArray *A, const RTspSArray *B, int m, int n, int k, int format=RTCSR){
    int *csrRowPtrC;
    cudaMalloc((void **)&csrRowPtrC, (m+1)*sizeof(int));
    int nnzC;
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    int *csrRowPtrB;
    cudaMalloc((void **)&csrRowPtrB, (k+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, B->row_indices, B->nnz, k, csrRowPtrB, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    cusparseXcsrgemmNnz(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, _rtcuda_matrix_descr, A->nnz, csrRowPtrA, 
                            A->column_indices, _rtcuda_matrix_descr, B->nnz, csrRowPtrB, B->column_indices, _rtcuda_matrix_descr, csrRowPtrC, &nnzC);
    cudaMalloc((void **)&C->values, nnzC*sizeof(float));
    cudaMalloc((void **)&C->column_indices, nnzC*sizeof(int));
    C->nnz = nnzC;
    _rtcuda_cusparse_status = cusparseScsrgemm(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
                                _rtcuda_matrix_descr, B->nnz, B->values, csrRowPtrB, B->column_indices, _rtcuda_matrix_descr, A->nnz, A->values, csrRowPtrA, A->column_indices, _rtcuda_matrix_descr, C->values, csrRowPtrC, C->column_indices);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspMM error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspMM error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspMM error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspMM error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspMM error: an internal operation failed.\n");
            exit(1);
    }
    _rtcuda_cusparse_status = cusparseXcsr2coo(_rtcuda_cusparse_handle, csrRowPtrC, nnzC, m, C->row_indices, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspMM error: the function failed to launch on the GPU.\n");
            exit(1);
    }
    cudaFree(csrRowPtrC);
    cudaFree(csrRowPtrA);
    cudaFree(csrRowPtrB);
}

void RTspDMM(RTspDArray *C, RTspDArray *A, const RTspDArray *B, int m, int n, int k, int format=RTCSR){
    int *csrRowPtrC;
    cudaMalloc((void **)&csrRowPtrC, (m+1)*sizeof(int));
    int nnzC;
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    int *csrRowPtrB;
    cudaMalloc((void **)&csrRowPtrB, (k+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, B->row_indices, B->nnz, k, csrRowPtrB, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    cusparseXcsrgemmNnz(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, _rtcuda_matrix_descr, A->nnz, csrRowPtrA, 
                            A->column_indices, _rtcuda_matrix_descr, B->nnz, csrRowPtrB, B->column_indices, _rtcuda_matrix_descr, csrRowPtrC, &nnzC);
    cudaMalloc((void **)&C->values, nnzC*sizeof(double));
    cudaMalloc((void **)&C->column_indices, nnzC*sizeof(int));
    C->nnz = nnzC;
    _rtcuda_cusparse_status = cusparseDcsrgemm(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
                                _rtcuda_matrix_descr, B->nnz, B->values, csrRowPtrB, B->column_indices, _rtcuda_matrix_descr, A->nnz, A->values, csrRowPtrA, A->column_indices, _rtcuda_matrix_descr, C->values, csrRowPtrC, C->column_indices);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDMM error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDMM error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMM error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDMM error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspDMM error: an internal operation failed.\n");
            exit(1);
    }
    _rtcuda_cusparse_status = cusparseXcsr2coo(_rtcuda_cusparse_handle, csrRowPtrC, nnzC, m, C->row_indices, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMM error: the function failed to launch on the GPU.\n");
            exit(1);
    }
    cudaFree(csrRowPtrC);
    cudaFree(csrRowPtrA);
    cudaFree(csrRowPtrB);
}

void RTspdSMMCSR(float *C, RTspSArray *A, const float *B, int m, int n, int k){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdSMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdSMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdSMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    const float alpha = 1.0; const float beta = 0.0;
    _rtcuda_cusparse_status = cusparseScsrmm2(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
                                                A->nnz, &alpha, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, B, k, &beta, C, m);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdSMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspdSMM error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdSMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspdSMM error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdSMM error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspdSMM error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspdSMM error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(csrRowPtrA);    
}

void RTspdSMMBSR(float *C, RTspSArray *A, const float *B, int m, int n, int k, int blocksize=4){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdSMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdSMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdSMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    int base, nnzb;
    int mb = (m + blocksize - 1) / blocksize;
    int kb = (k + blocksize - 1) / blocksize;
    int *bsrRowPtrA;
    cudaMalloc((void**)&bsrRowPtrA, sizeof(int)*(mb+1));
    int *nnzHost = &nnzb;
    cusparseXcsr2bsrNnz(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, k, _rtcuda_matrix_descr, csrRowPtrA, A->column_indices, blocksize, _rtcuda_matrix_descr, 
                        bsrRowPtrA, nnzHost);
    if (NULL != nnzHost){ 
            nnzb = *nnzHost; 
    }
    else{ 
            cudaMemcpy(&nnzb, bsrRowPtrA+mb, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(&base, bsrRowPtrA, sizeof(int), cudaMemcpyDeviceToHost); 
            nnzb -= base; 
    }

    int *bsrColIndA;
    cudaMalloc((void**)&bsrColIndA, sizeof(int)*nnzb);
    float *bsrValA;
    cudaMalloc((void**)&bsrValA, sizeof(float)*(blocksize*blocksize)*nnzb);
    cusparseScsr2bsr(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, k, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, blocksize, 
                        _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, bsrColIndA);
    cudaFree(csrRowPtrA);    

    const float alpha = 1.0; const float beta = 0.0;
    _rtcuda_cusparse_status = cusparseSbsrmm(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, mb, n, kb, nnzb, &alpha, _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, 
                                                bsrColIndA, blocksize, B, k, &beta, C, m);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdSMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspdSMM error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdSMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspdSMM error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdSMM error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspdSMM error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspdSMM error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(bsrRowPtrA);
    cudaFree(bsrColIndA);
    cudaFree(bsrValA);
}

void RTspdSMM(float *C, RTspSArray *A, const float *B, int m, int n, int k, int format=RTCSR){
    if(format == RTBSR)
        RTspdSMMBSR(C, A, B, m, n, k);
    else
        RTspdSMMCSR(C, A, B, m, n, k);
}

void RTspdDMMCSR(double *C, RTspDArray *A, const double *B, int m, int n, int k){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdDMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdDMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdDMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    const double alpha = 1.0; const double beta = 0.0;
    _rtcuda_cusparse_status = cusparseDcsrmm2(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
                                                A->nnz, &alpha, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, B, k, &beta, C, m);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdDMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspdDMM error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdDMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspdDMM error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdDMM error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspdDMM error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspdDMM error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(csrRowPtrA);    
}

void RTspdDMMBSR(double *C, RTspDArray *A, const double *B, int m, int n, int k, int blocksize=4){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdDMM error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdDMM error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdDMM error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    int base, nnzb;
    int mb = (m + blocksize - 1) / blocksize;
    int kb = (k + blocksize - 1) / blocksize;
    int *bsrRowPtrA;
    cudaMalloc((void**)&bsrRowPtrA, sizeof(int)*(mb+1));
    int *nnzHost = &nnzb;
    cusparseXcsr2bsrNnz(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, k, _rtcuda_matrix_descr, csrRowPtrA, A->column_indices, blocksize, _rtcuda_matrix_descr, 
                        bsrRowPtrA, nnzHost);
    if (NULL != nnzHost){ 
            nnzb = *nnzHost; 
    }
    else{ 
            cudaMemcpy(&nnzb, bsrRowPtrA+mb, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(&base, bsrRowPtrA, sizeof(int), cudaMemcpyDeviceToHost); 
            nnzb -= base; 
    }

    int *bsrColIndA;
    cudaMalloc((void**)&bsrColIndA, sizeof(int)*nnzb);
    double *bsrValA;
    cudaMalloc((void**)&bsrValA, sizeof(double)*(blocksize*blocksize)*nnzb);
    cusparseDcsr2bsr(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, k, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, blocksize, 
                        _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, bsrColIndA);
    cudaFree(csrRowPtrA);    

    const double alpha = 1.0; const double beta = 0.0;
    _rtcuda_cusparse_status = cusparseDbsrmm(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, mb, n, kb, nnzb, &alpha, _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, 
                                                bsrColIndA, blocksize, B, k, &beta, C, m);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspdDMM error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspdDMM error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspdDMM error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspdDMM error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspdDMM error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspdDMM error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspdDMM error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(bsrRowPtrA);
    cudaFree(bsrColIndA);
    cudaFree(bsrValA);
}

void RTspdDMM(double *C, RTspDArray *A, const double *B, int m, int n, int k, int format=RTCSR){
    if(format == RTBSR)
        RTspdDMMBSR(C, A, B, m, n, k);
    else
        RTspdDMMCSR(C, A, B, m, n, k);
}

void RTspSMVCSR(float *C, const RTspSArray *A, const float *B, int m, int n){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    const float alpha=1.0; const float beta=0.0;
    _rtcuda_cusparse_status = cusparseScsrmv(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, m, n, A->nnz, &alpha, _rtcuda_matrix_descr, 
                                                A->values, csrRowPtrA, A->column_indices, B, &beta, C);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspSMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspSMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspSMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspSMV error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(csrRowPtrA);        
}

void RTspSMVBSR(float *C, const RTspSArray *A, const float *B, int m, int n, int blocksize=4){
        int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    int base, nnzb;
    int mb = (m + blocksize - 1) / blocksize;
    int kb = (n + blocksize - 1) / blocksize;
    int *bsrRowPtrA;
    cudaMalloc((void**)&bsrRowPtrA, sizeof(int)*(mb+1));
    int *nnzHost = &nnzb;
    cusparseXcsr2bsrNnz(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, n, _rtcuda_matrix_descr, csrRowPtrA, A->column_indices, blocksize, _rtcuda_matrix_descr, 
                        bsrRowPtrA, nnzHost);
    if (NULL != nnzHost){ 
            nnzb = *nnzHost; 
    }
    else{ 
            cudaMemcpy(&nnzb, bsrRowPtrA+mb, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(&base, bsrRowPtrA, sizeof(int), cudaMemcpyDeviceToHost); 
            nnzb -= base; 
    }

    int *bsrColIndA;
    cudaMalloc((void**)&bsrColIndA, sizeof(int)*nnzb);
    float *bsrValA;
    cudaMalloc((void**)&bsrValA, sizeof(float)*(blocksize*blocksize)*nnzb);
    cusparseScsr2bsr(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, n, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, blocksize, 
                        _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, bsrColIndA);
    cudaFree(csrRowPtrA);
    
    const float alpha=1.0; const float beta=0.0;
    _rtcuda_cusparse_status = cusparseSbsrmv(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_TRANSPOSE, mb, kb, nnzb, &alpha, 
                                                _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, bsrColIndA, blocksize, B, &beta, C);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspSMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspSMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspSMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspSMV error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(bsrColIndA);
    cudaFree(bsrRowPtrA);
    cudaFree(bsrValA);
}

void RTspSMVHYB(float *C, const RTspSArray *A, const float *B, int m, int n){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    cusparseHybMat_t hybA;
    cusparseCreateHybMat(&hybA);
    int ELLWidth = 2; //should be less than the maximum number of nonzeros per row, not used if CUSPARSE_HYB_PARTITION_AUTO
    _rtcuda_cusparse_status = cusparseScsr2hyb(_rtcuda_cusparse_handle, m, n, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, hybA, 
                                                ELLWidth, CUSPARSE_HYB_PARTITION_AUTO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspSMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspSMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspSMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspSMV error: an internal operation failed.\n");
            exit(1);
    }
    const float alpha=1.0; const float beta=0.0;
    _rtcuda_cusparse_status = cusparseShybmv(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, _rtcuda_matrix_descr, hybA, B, &beta, C);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspSMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspSMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspSMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspSMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspSMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspSMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspSMV error: an internal operation failed.\n");
            exit(1);
    }    
    cudaFree(csrRowPtrA);
    cusparseDestroyHybMat(hybA);
}

void RTspSMV(float *C, const RTspSArray *A, const float *B, int m, int n, int format=RTHYB){
    if(format == RTCSR)
        RTspSMVCSR(C, A, B, m, n);
    else if(format == RTBSR)
        RTspSMVBSR(C, A, B, m, n);
    else
        RTspSMVHYB(C, A, B, m, n);
}

void RTspDMVCSR(double *C, const RTspDArray *A, const double *B, int m, int n){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    const double alpha=1.0; const double beta=0.0;
    _rtcuda_cusparse_status = cusparseDcsrmv(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, m, n, A->nnz, &alpha, _rtcuda_matrix_descr, 
                                                A->values, csrRowPtrA, A->column_indices, B, &beta, C);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspDMV error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(csrRowPtrA);        
}

void RTspDMVBSR(double *C, const RTspDArray *A, const double *B, int m, int n, int blocksize=4){
        int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    int base, nnzb;
    int mb = (m + blocksize - 1) / blocksize;
    int kb = (n + blocksize - 1) / blocksize;
    int *bsrRowPtrA;
    cudaMalloc((void**)&bsrRowPtrA, sizeof(int)*(mb+1));
    int *nnzHost = &nnzb;
    cusparseXcsr2bsrNnz(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, n, _rtcuda_matrix_descr, csrRowPtrA, A->column_indices, blocksize, _rtcuda_matrix_descr, 
                        bsrRowPtrA, nnzHost);
    if (NULL != nnzHost){ 
            nnzb = *nnzHost; 
    }
    else{ 
            cudaMemcpy(&nnzb, bsrRowPtrA+mb, sizeof(int), cudaMemcpyDeviceToHost); 
            cudaMemcpy(&base, bsrRowPtrA, sizeof(int), cudaMemcpyDeviceToHost); 
            nnzb -= base; 
    }

    int *bsrColIndA;
    cudaMalloc((void**)&bsrColIndA, sizeof(int)*nnzb);
    double *bsrValA;
    cudaMalloc((void**)&bsrValA, sizeof(double)*(blocksize*blocksize)*nnzb);
    cusparseDcsr2bsr(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, m, n, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, blocksize, 
                        _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, bsrColIndA);
    cudaFree(csrRowPtrA);
    
    const double alpha=1.0; const double beta=0.0;
    _rtcuda_cusparse_status = cusparseDbsrmv(_rtcuda_cusparse_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_TRANSPOSE, mb, kb, nnzb, &alpha, 
                                                _rtcuda_matrix_descr, bsrValA, bsrRowPtrA, bsrColIndA, blocksize, B, &beta, C);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspDMV error: an internal operation failed.\n");
            exit(1);
    }
    cudaFree(bsrColIndA);
    cudaFree(bsrRowPtrA);
    cudaFree(bsrValA);
}

void RTspDMVHYB(double *C, const RTspDArray *A, const double *B, int m, int n){
    int *csrRowPtrA;
    cudaMalloc((void **)&csrRowPtrA, (m+1)*sizeof(int));
    _rtcuda_cusparse_status = cusparseXcoo2csr(_rtcuda_cusparse_handle, A->row_indices, A->nnz, m, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.\n");
            exit(1);
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
            break;
    }
    cusparseHybMat_t hybA;
    cusparseCreateHybMat(&hybA);
    int ELLWidth = 2; //should be less than the maximum number of nonzeros per row, not used if CUSPARSE_HYB_PARTITION_AUTO
    _rtcuda_cusparse_status = cusparseDcsr2hyb(_rtcuda_cusparse_handle, m, n, _rtcuda_matrix_descr, A->values, csrRowPtrA, A->column_indices, hybA, 
                                                ELLWidth, CUSPARSE_HYB_PARTITION_AUTO);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspDMV error: an internal operation failed.\n");
            exit(1);
    }
    const double alpha=1.0; const double beta=0.0;
    _rtcuda_cusparse_status = cusparseDhybmv(_rtcuda_cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, _rtcuda_matrix_descr, hybA, B, &beta, C);
    switch(_rtcuda_cusparse_status){
        case CUSPARSE_STATUS_SUCCESS:
            break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            printf("RTspDMV error: the library was not initialized.\n");
            exit(1);
        case CUSPARSE_STATUS_ALLOC_FAILED:
            printf("RTspDMV error: the resources could not be allocated.\n");
            exit(1);
        case CUSPARSE_STATUS_INVALID_VALUE:
            printf("RTspDMV error: invalid parameters were passed (m,n,k<0; IndexBase of descrA,descrB,descrC is not base-0 or base-1; or alpha or beta is nil )).\n");
            exit(1);
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            printf("RTspDMV error: the device does not support double precision.\n");
            exit(1);
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            printf("RTspDMV error: the function failed to launch on the GPU.\n");
            exit(1);
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            printf("RTspDMV error: the matrix type is not supported.\n");
            exit(1);
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            printf("RTspDMV error: an internal operation failed.\n");
            exit(1);
    }    
    cudaFree(csrRowPtrA);
    cusparseDestroyHybMat(hybA);
}

void RTspDMV(double *C, const RTspDArray *A, const double *B, int m, int n, int format=RTHYB){
    if(format == RTCSR)
        RTspDMVCSR(C, A, B, m, n);
    else if(format == RTBSR)
        RTspDMVBSR(C, A, B, m, n);
    else
        RTspDMVHYB(C, A, B, m, n);
}

void RTWtimeInit()
{
    cudaEventRecord(_rtcuda_time_event_start, 0);    
}

float RTWtimeFinalize()
{
    float time=0.0;
    cudaEventRecord(_rtcuda_time_event_stop, 0);
    cudaEventSynchronize(_rtcuda_time_event_stop);
    cudaEventElapsedTime(&time, _rtcuda_time_event_start, _rtcuda_time_event_stop);
    return time;
}
