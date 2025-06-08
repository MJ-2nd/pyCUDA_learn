import numpy as np
import time
from numba import cuda, float32
import torch

@cuda.jit
def matmul_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

def gpu_matmul(a, b, num_runs=10, blockdim=(32,32)):
    d_A = cuda.to_device(a)
    d_B = cuda.to_device(b)
    d_C = cuda.device_array((a.shape[0], b.shape[1]), dtype=np.float32)

    threads_per_block = blockdim
    blocks_per_grid = (
        (a.shape[0] + blockdim[0] - 1) // blockdim[0],
        (b.shape[1] + blockdim[1] - 1) // blockdim[1]
    )

    # 커널 실행 시간 측정 (데이터 전송 제외)
    gpu_kernel_times = []
    for _ in range(num_runs):
        start = time.time()
        matmul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()
        gpu_kernel_times.append(time.time() - start)
    
    # 전체 GPU 실행 시간 (데이터 전송 포함)
    gpu_total_times = []
    for _ in range(num_runs):
        start = time.time()
        d_A = cuda.to_device(a)
        d_B = cuda.to_device(b)
        matmul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()
        c_res = d_C.copy_to_host()
        gpu_total_times.append(time.time() - start)
    
    return c_res, gpu_kernel_times, gpu_total_times

if __name__ == "__main__":
    size = 10000
    num_runs = 5

    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # 1. CPU NumPy 측정
    numpy_times = []
    for _ in range(num_runs):
        start = time.time()
        c_cpu = np.matmul(a, b)
        numpy_times.append(time.time() - start)

    # 2. GPU Numba CUDA 측정
    c_gpu, gpu_kernel_times, gpu_total_times = gpu_matmul(a, b, num_runs=num_runs)

    # 3. PyTorch GPU 측정
    torch_times = []
    for _ in range(num_runs):
        a_torch = torch.tensor(a).cuda()
        b_torch = torch.tensor(b).cuda()
        torch.cuda.synchronize()
        start = time.time()
        c_torch = torch.matmul(a_torch, b_torch)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start)

    print("========== 실행 시간 (평균) ==========")
    print(f"NumPy (CPU):       {np.mean(numpy_times):.6f} sec")
    print(f"Numba CUDA (커널):  {np.mean(gpu_kernel_times):.6f} sec")
    print(f"Numba CUDA (전체):  {np.mean(gpu_total_times):.6f} sec")
    print(f"PyTorch (GPU):     {np.mean(torch_times):.6f} sec")

    print("\n========== 정확도 비교 (Max 오차) ==========")
    print(f"Numba vs NumPy:    {np.max(np.abs(c_cpu - c_gpu))}")
    print(f"PyTorch vs NumPy:  {np.max(np.abs(c_cpu - c_torch.cpu().numpy()))}")
