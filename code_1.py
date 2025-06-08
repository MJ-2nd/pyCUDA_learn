import numpy as np
import torch
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

def benchmark_matrix_multiplication(size=5000, num_runs=3):
    """
    행렬 곱셈 성능을 세 가지 방식으로 비교하는 벤치마크 함수
    
    Args:
        size (int): 행렬의 크기 (size x size)
        num_runs (int): 각 방식별 실행 횟수
    """
    print(f"\n행렬 크기: {size}x{size}")
    print("-" * 50)
    
    # 테스트 데이터 준비
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # 1. NumPy (CPU)
    numpy_times = []
    for _ in range(num_runs):
        start = time.time()
        c_numpy = np.matmul(a, b)
        numpy_times.append(time.time() - start)
    
    # 2. PyTorch (GPU)
    torch_times = []
    for _ in range(num_runs):
        a_torch = torch.tensor(a).cuda()
        b_torch = torch.tensor(b).cuda()
        start = time.time()
        c_torch = torch.matmul(a_torch, b_torch)
        cuda.synchronize()  # GPU 연산 완료 대기
        torch_times.append(time.time() - start)
    
    # 3. PyCUDA (GPU)
    # CUDA 커널 정의
    mod = SourceModule("""
    __global__ void matrix_multiply(float *C, float *A, float *B, int size)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < size && col < size) {
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                sum += A[row * size + i] * B[i * size + col];
            }
            C[row * size + col] = sum;
        }
    }
    """)
    
    matrix_multiply = mod.get_function("matrix_multiply")
    cuda_times = []
    
    for _ in range(num_runs):
        # GPU 메모리 할당
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(a.nbytes)
        
        # 데이터를 GPU로 복사
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)
        
        # 커널 실행
        block_size = 16
        grid_size = (size // block_size + 1, size // block_size + 1)
        block = (block_size, block_size, 1)
        
        start = time.time()
        matrix_multiply(
            c_gpu, a_gpu, b_gpu, np.int32(size),
            block=block, grid=grid_size
        )
        cuda.synchronize()
        cuda_times.append(time.time() - start)
    
    # 결과 출력
    print(f"NumPy (CPU) 평균 실행 시간: {np.mean(numpy_times):.3f}초")
    print(f"PyTorch (GPU) 평균 실행 시간: {np.mean(torch_times):.3f}초")
    print(f"PyCUDA (GPU) 평균 실행 시간: {np.mean(cuda_times):.3f}초")
    
    # 결과 검증
    c_numpy = np.matmul(a, b)
    c_torch = torch.matmul(torch.tensor(a).cuda(), torch.tensor(b).cuda()).cpu().numpy()
    c_cuda = np.empty_like(a)
    cuda.memcpy_dtoh(c_cuda, c_gpu)
    
    print("\n결과 검증:")
    print(f"NumPy vs PyTorch 오차: {np.max(np.abs(c_numpy - c_torch)):.2e}")
    print(f"NumPy vs PyCUDA 오차: {np.max(np.abs(c_numpy - c_cuda)):.2e}")

if __name__ == "__main__":
    try:
        # GPU 정보 출력
        print("GPU 정보:")
        print(f"사용 가능한 GPU: {cuda.Device.count()}")
        for i in range(cuda.Device.count()):
            device = cuda.Device(i)
            print(f"GPU {i}: {device.name()}")
        print("-" * 50)
        
        # 다양한 크기로 벤치마크 실행
        sizes = [1000, 2000, 5000]
        for size in sizes:
            benchmark_matrix_multiplication(size)
            
    except Exception as e:
        print(f"에러 발생: {str(e)}")
