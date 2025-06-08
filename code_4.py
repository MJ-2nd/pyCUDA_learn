import numpy as np
from cuda import cuda, cublas
import ctypes

# cuBLAS handle 생성
status, handle = cublas.cublasCreate()
assert status == cuda.CUDA_SUCCESS

# 행렬 사이즈
N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# GPU 메모리 할당 및 복사
A_d = cuda.cuMemAlloc(A.nbytes)[1]
B_d = cuda.cuMemAlloc(B.nbytes)[1]
C_d = cuda.cuMemAlloc(C.nbytes)[1]

cuda.cuMemcpyHtoD(A_d, A.ctypes.data, A.nbytes)
cuda.cuMemcpyHtoD(B_d, B.ctypes.data, B.nbytes)

# SGEMM 파라미터
alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)
lda = ldb = ldc = N

# SGEMM 실행: C = alpha * A * B + beta * C
status = cublas.cublasSgemm(
    handle,
    cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
    N, N, N,
    ctypes.byref(alpha),
    A_d, lda,
    B_d, ldb,
    ctypes.byref(beta),
    C_d, ldc
)
assert status == cuda.CUDA_SUCCESS

# 결과를 host로 복사
cuda.cuMemcpyDtoH(C.ctypes.data, C_d, C.nbytes)

# 자원 해제
cublas.cublasDestroy(handle)
cuda.cuMemFree(A_d)
cuda.cuMemFree(B_d)
cuda.cuMemFree(C_d)

# 결과 확인
print("C[0,0]:", C[0, 0])
