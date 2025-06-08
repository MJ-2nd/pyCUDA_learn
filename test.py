import numpy as np
from cuda.bindings import runtime as cudart
from cuda.bindings import cublas

# 행렬 크기
M = N = K = 1024
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

# CUDA malloc
a_gpu = cudart.cudaMalloc(A.nbytes)[1]
b_gpu = cudart.cudaMalloc(B.nbytes)[1]
c_gpu = cudart.cudaMalloc(C.nbytes)[1]

# Host to Device 복사
cudart.cudaMemcpy(a_gpu, A.ctypes.data, A.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
cudart.cudaMemcpy(b_gpu, B.ctypes.data, B.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

# cuBLAS handle 생성
handle = cublas.cublasCreate_v2()[1]

alpha = np.array([1.0], dtype=np.float32)
beta = np.array([0.0], dtype=np.float32)

# cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
# 단, column-major를 기준으로 하므로 매개변수 주의!
status = cublas.cublasSgemm_v2(
    handle,
    cublas.CUBLAS_OP_N,
    cublas.CUBLAS_OP_N,
    N, M, K,
    alpha.ctypes.data,
    b_gpu, N,
    a_gpu, K,
    beta.ctypes.data,
    c_gpu, N
)
assert status == cublas.CUBLAS_STATUS_SUCCESS

# Device to Host 복사
cudart.cudaMemcpy(C.ctypes.data, c_gpu, C.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

# 핸들 및 메모리 해제
cublas.cublasDestroy_v2(handle)
cudart.cudaFree(a_gpu)
cudart.cudaFree(b_gpu)
cudart.cudaFree(c_gpu)

print("Done. Result matrix C shape:", C.shape)
