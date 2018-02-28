#include <cublas_v2.h>

void RTdSMM(float *C, const float *A, const float *B, int m, int n, int k){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMM error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdSMM error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const float alpha = 1.0;
    const float beta = 0.0;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B, k, A, m, &beta, C, m);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMM error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdDMM(double *C, const double *A, const double *B, int m, int n, int k){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMM error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdDMM error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const double alpha = 1.0;
    const double beta = 0.0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B, k, A, m, &beta, C, m);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMM error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdSMV(float *C, const float *A, const float *B, int m, int n){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMV error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdSMV error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const float alpha = 1.0;
    const float beta = 0.0;
    status = cublasSgemv(handle, CUBLAS_OP_T, m, n, &alpha, A, m, B, 1, &beta, C, 1);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMV error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdDMV(double *C, const double *A, const double *B, int m, int n){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMV error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdDMV error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const double alpha = 1.0;
    const double beta = 0.0;
    status = cublasDgemv(handle, CUBLAS_OP_T, m, n, &alpha, A, m, B, 1, &beta, C, 1);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMV error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdSMT(float *C, const float *A, int m, int n){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMT error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdSMT error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const float alpha = 1.0;
    const float beta = 0.0;
    status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, m, &beta, A, m, C, m);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSMT error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdDMT(double *C, const double *A, int m, int n){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMT error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdDMT error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const double alpha = 1.0;
    const double beta = 0.0;
    status = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, m, &beta, A, m, C, m);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDMT error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdSVV(float *C, const float *A, int m, const float *B, int n){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSVV error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdSVV error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const float alpha = 1.0;
    const float beta = 0.0;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 1, &alpha, A, m, B, n, &beta, C, m);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdSVV error: the library was not initialized");
            exit(1);
            break;
    }
}

void RTdDVV(double *C, const double *A, int m, const double *B, int n){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDVV error: the CUDA Runtime Initialization Failed");
            exit(1);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("RtdDVV error: the resources could not be allocated");
            exit(1);
            break;
    }
    
    const double alpha = 1.0;
    const double beta = 0.0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 1, &alpha, A, m, B, n, &beta, C, m);
    switch(status){
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
    
    status = cublasDestroy(handle);
    switch(status){
        case CUBLAS_STATUS_SUCCESS:
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("RTdDVV error: the library was not initialized");
            exit(1);
            break;
    }
}
