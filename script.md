# Python CUDA 세미나 대본

## 1. 머신러닝 프레임워크의 진화와 Python CUDA의 혁신

### 1.1 기존 프레임워크들의 GPU 활용 방식
- TensorFlow와 PyTorch의 GPU 연산 처리 방식
  * 자동 그래프 최적화
  * 추상화된 GPU 연산
  * 제한된 커스터마이징
- 예시 코드:
```python
# PyTorch의 일반적인 GPU 사용
import torch
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)  # GPU에서 자동 실행
```

### 1.2 "Black Box" GPU 연산의 한계
- 문제점:
  * 메모리 관리의 제한적 제어
  * 커스텀 연산 구현의 어려움
  * 디버깅의 어려움
- 실제 사례:
  * 대규모 행렬 연산에서의 메모리 누수
  * 특수 연산 구현 시 성능 저하
  * 프레임워크 업데이트로 인한 호환성 문제

### 1.3 Python CUDA가 가져온 혁신적 변화
- 직접적인 GPU 프로그래밍 가능
- 메모리 관리의 완전한 제어
- 커스텀 커널 구현의 자유
- 예시 코드:
```python
# PyCUDA를 사용한 직접적인 GPU 프로그래밍
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")
```

### 1.4 실제 성능 비교 데이터 시연
- 동일한 행렬 곱셈 연산을 다음 세 가지 방식으로 구현:
  1. NumPy (CPU)
  2. PyTorch (GPU)
  3. PyCUDA (GPU)

예상되는 성능 비교 결과:
- NumPy (CPU): ~10-15초
- PyTorch (GPU): ~1-2초
- PyCUDA (GPU): ~0.5-1초

시연 시 주의사항:
1. 실제 하드웨어 사양에 따라 결과가 달라질 수 있음
2. 메모리 전송 시간을 고려한 전체 실행 시간 측정
3. 여러 번 실행하여 평균값 제시
4. 에러 처리와 예외 상황 대비


문제
1. pycuda와 python-cuda는 다르다.
pycuda는 개발자들이 작성하던 오픈소스, python-cuda는 엔비디아 공식
pycuda는 설치하려면 오류가 많이 나서 사용이 쉽지가 않음

2. 자주 업데이트 되어 사용법이 아직 확정되지 않음
from cuda import cudart, cublas
를 사용하면 된다고 해서 썻는데
```
(pycuda_venv) mj-main@mj-main:~/shared/pyCUDA_learn$ python3 test.py 
<frozen importlib._bootstrap_external>:1297: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
Traceback (most recent call last):
  File "/home/mj-main/shared/pyCUDA_learn/test.py", line 2, in <module>
    from cuda import cudart, cublas
ImportError: cannot import name 'cublas' from 'cuda' (unknown location)
```

이런 오류만 남

최신 기법은
```
from cuda.bindings import runtime as cudart
from cuda.bindings import cublas
```
를 써야한다.


행렬곱! - https://blog.naver.com/lis0517/220906050490

해당 방법으로 code_1.py를 비교해보면
```
(pycuda_venv) mj-main@mj-main:~/shared/pyCUDA_learn$ python3 code_1.py 
========== 실행 시간 (평균) ==========
NumPy (CPU):       0.194304 sec
Numba CUDA (GPU):  0.539669 sec
PyTorch (GPU):     0.017443 sec

========== 정확도 비교 (Max 오차) ==========
Numba vs NumPy:    1.525879e-04
PyTorch vs NumPy:  5.950928e-04
```

의외로 numba가 제일 느림



이유

이 방식은 다음을 고려하지 않았습니다:
Shared memory 사용 없음
현재 구현: 각 스레드가 매 반복마다 전역 메모리(Global Memory)에서 데이터를 읽어옴

개선 방향:
Shared Memory를 사용하여 자주 접근하는 데이터를 캐싱
블록 단위로 데이터를 Shared Memory에 로드하여 재사용

현재
```
@cuda.jit
def matmul_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]  # 매번 전역 메모리에서 데이터를 읽어옴
        C[i, j] = tmp
```

예시
```
@cuda.jit
def matmul_shared_kernel(A, B, C):
    # Shared Memory 선언
    shared_A = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    shared_B = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    
    i, j = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    
    tmp = 0.0
    for m in range(0, A.shape[1], BLOCK_SIZE):
        # Shared Memory에 데이터 로드
        shared_A[tx, ty] = A[i, m + ty]
        shared_B[tx, ty] = B[m + tx, j]
        cuda.syncthreads()
        
        # Shared Memory에서 계산
        for k in range(BLOCK_SIZE):
            tmp += shared_A[tx, k] * shared_B[k, ty]
        cuda.syncthreads()
    
    C[i, j] = tmp
```


memory coalescing 없음
현재 구현: 각 스레드가 불규칙한 메모리 위치에 접근
memory colescing > GPU에서 여러 스레드가 동시에 메모리에 접근할 때, 이 접근들을 하나의 메모리 트랜잭션으로 병합하는 기술입니다.
(NVIDIA GPU는 32개의 스레드를 하나의 워프(warp)로 그룹화하여 실행합니다.)

문제점:
메모리 접근이 비효율적
메모리 대역폭을 최대한 활용하지 못함

워프 1의 스레드들:
스레드 0: A[0,0], B[0,0]
스레드 1: A[0,1], B[1,0]
스레드 2: A[0,2], B[2,0]
...
스레드 31: A[0,31], B[31,0]



개선 방향:
연속된 메모리 접근 패턴 사용
32바이트 정렬된 메모리 접근

# 전치 행렬 사용
B_transposed = np.transpose(B)

워프 1의 스레드들:
스레드 0: A[0,0], B[0,0]
스레드 1: A[0,1], B[0,1]
스레드 2: A[0,2], B[0,2]
...
스레드 31: A[0,31], B[0,31]



thread-block tiling 없음
현재 구현: 단순한 2D 그리드 구조
문제점:
캐시 활용도가 낮음
메모리 접근이 비효율적
개선 방향:
적절한 블록 크기 선택 (예: 16x16, 32x32)
타일링을 통한 데이터 재사용

Thread Block Tiling이란?
큰 행렬을 작은 타일(블록)로 나누어 처리하는 기술입니다.
각 블록은 Shared Memory에 로드되어 재사용됩니다.
메모리 접근 횟수를 줄이고 데이터 재사용성을 높입니다.

타일 A[i,j]는 B[j,k]와 곱해질 때 여러 번 재사용됨
- A[i,j]를 Shared Memory에 한 번만 로드
- 여러 B[j,k] 타일과 곱셈 수행


개선된 예시
```
@cuda.jit
def matmul_tiled_kernel(A, B, C):
    # Shared Memory 선언
    shared_A = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    shared_B = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    
    # 스레드 인덱스
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # 결과 행렬의 위치 계산
    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx
    
    # 누적 합을 저장할 변수
    tmp = 0.0
    
    # 타일 단위로 반복
    for m in range(0, A.shape[1], TILE_SIZE):
        # Shared Memory에 데이터 로드
        if row < A.shape[0] and m + tx < A.shape[1]:
            shared_A[ty, tx] = A[row, m + tx]
        if m + ty < B.shape[0] and col < B.shape[1]:
            shared_B[ty, tx] = B[m + ty, col]
        cuda.syncthreads()
        
        # 타일 내에서 계산
        if row < C.shape[0] and col < C.shape[1]:
            for k in range(TILE_SIZE):
                if m + k < A.shape[1]:
                    tmp += shared_A[ty, k] * shared_B[k, tx]
        cuda.syncthreads()
    
    # 결과 저장
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp
```


SIMD (Single Instruction Multiple Data)  명령어 사용 없음


PyTorch는 내부적으로 다음을 사용합니다:
cuBLAS, cuDNN, CUTLASS 등 NVIDIA에서 최적화한 라이브러리
Tensor Core (FP16/TF32) 연산 자동 활용 (가능한 경우)
memory layout, thread-block tiling, pipeline 전부 적용


cuBLAS와 비교
PyTorch는 내부적으로 cuBLAS (NVIDIA의 highly optimized matrix multiply library)를 사용합니다.
cuBLAS는:
최적화된 tiling, warp scheduling
Tensor Core (FP16/FP32 혼합 연산)
L2/L1 캐시 및 shared memory 자동 튜닝
instruction-level parallelism 등을 수년간 최적화해온 코드입니다.

단순히 GPU로 옮기는 것은 충분하지 않고, 블럭 연산을 어떻게 수행할지, shared memory는 어떻게 할지





cuda-python만 가지고는 nvidia 주요 라이브러리 사용 불가
바인딩을 해서 가져와서 사용해야 함 (사용방법을 알아야 함)


cuda-python은 C로 된 cuBLAS API를 Python에서 직접 호출할 수 있도록 바인딩한 라이브러리입니다.
이때 cuBLAS는 이미 컴파일된 라이브러리 (libcublas.so)이므로, cuda-python은 단순히 함수 호출 인터페이스만 연결합니다.
예를 들어 cuda.cublas.cublasSgemm(...) 호출 시 내부적으로는 미리 컴파일된 libcublas.so에 있는 C 함수가 불립니다.


예시

cuBLAS(CUDA Basic Linear Algebra Subroutines)

NVIDIA의 기본 선형대수 라이브러리
BLAS(기본 선형대수 서브루틴)의 GPU 구현
행렬 곱셈, 벡터 연산 등 기본적인 선형대수 연산 제공


cuDNN (CUDA Deep Neural Network)
딥러닝을 위한 고성능 프리미티브 라이브러리
컨볼루션, 풀링, 활성화 함수 등 딥러닝 연산 최적화
TensorFlow, PyTorch 등 주요 딥러닝 프레임워크에서 사용

CUTLASS (CUDA Templates for Linear Algebra Subroutines)
NVIDIA의 고성능 선형대수 라이브러리
템플릿 기반으로 다양한 데이터 타입과 연산 지원
커스텀 커널 개발을 위한 유연한 프레임워크


딥러닝을 위한 고성능 프리미티브 라이브러리
컨볼루션, 풀링, 활성화 함수 등 딥러닝 연산 최적화
TensorFlow, PyTorch 등 주요 딥러닝 프레임워크에서 사용




그럼 단점만 있는 방식인가...?
GPU 연산에 대한 지식이 있다면, 아주 나쁜 방법은 아니다...

기존의 연산은 추상화되어 있어 추적이 어려움
```
# PyTorch 예시
import torch

# 단순한 행렬 곱셈
a = torch.randn(1000, 1000).cuda()
b = torch.randn(1000, 1000).cuda()
c = torch.matmul(a, b)  # 내부 구현을 알 수 없음
```

>> 직접적으로 제어 가능 (성능의 여부를 떠나)
```

# 메모리 할당과 관리
def custom_gpu_operation():
    # GPU 메모리 직접 할당
    a_gpu = cudart.cudaMalloc(a.nbytes)[1]
    
    # 메모리 전송 직접 제어
    cudart.cudaMemcpy(a_gpu, a.ctypes.data, a.nbytes, 
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    
    # 연산 직접 수행
    handle = cublas.cublasCreate()[1]
    cublas.cublasSgemm(...)
```


커스텀을 하고 싶어도 하기가 어려움
```
# TensorFlow 예시
import tensorflow as tf

# 컨볼루션 연산
x = tf.random.normal([32, 28, 28, 3])
y = tf.keras.layers.Conv2D(64, 3)(x)  # 내부 최적화 방법을 수정할 수 없음
```

행렬 연산 직접 구현 가능
```
def gpu_matmul(a, b, num_runs=10, blockdim=(16,16)):
    d_A = cuda.to_device(a)
    d_B = cuda.to_device(b)
    d_C = cuda.device_array((a.shape[0], b.shape[1]), dtype=np.float32)

    threads_per_block = blockdim
    blocks_per_grid = (
        (a.shape[0] + blockdim[0] - 1) // blockdim[0],
        (b.shape[1] + blockdim[1] - 1) // blockdim[1]
    )

    gpu_times = []
    for _ in range(num_runs):
        start = time.time()
        matmul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()
        gpu_times.append(time.time() - start)

    c_res = d_C.copy_to_host()
    return c_res, gpu_times
```

HW 최적화 가능
```
# GPU 특성에 맞는 최적화
def get_optimal_block_size():
    # GPU 정보 조회
    device = cudart.cudaGetDeviceProperties(0)[1]
    
    # 하드웨어 특성에 맞는 블록 크기 선택
    max_threads_per_block = device.maxThreadsPerBlock
    warp_size = device.warpSize
    
    return (warp_size, warp_size)  # 32x32 블록
```


디버깅이 어려움
```
# 에러 발생 시 추적이 어려움
try:
    output = model(input)
except RuntimeError as e:
    print(e)  # GPU 연산의 구체적인 문제 파악이 어려움
```


예시 cuBLAS 핸들러를 가져와서 오류 코드를 직접 관리 가능
```
# 상세한 에러 추적
try:
    # GPU 연산 수행
    result = cublas.cublasSgemm(...)
except Exception as e:
    # 구체적인 에러 정보
    print(f"GPU 연산 실패: {e}")
    print(f"메모리 상태: {cudart.cudaGetLastError()}")
```


메모리 복사, 연산 등 각 과정에 정확한 소요시간 측정 가능


tensorflow, pytorch 등은 아직 C++/C 기반이지만, 프로파일링/디버깅에 cuda-python을 도입하는 것은 검토 중





## 마무리 JIT
