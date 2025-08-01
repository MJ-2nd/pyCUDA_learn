import numpy as np
import time
from numba import cuda, float32
import torch

# GPU용 naive 행렬 곱 커널 정의
@cuda.jit
def matmul_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

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

if __name__ == "__main__":
    size = 4096
    num_runs = 5

    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # 1️⃣ CPU NumPy
    numpy_times = []
    for _ in range(num_runs):
        start = time.time()
        c_cpu = np.matmul(a, b)
        numpy_times.append(time.time() - start)

    # 2️⃣ GPU Numba CUDA
    c_gpu, gpu_times = gpu_matmul(a, b, num_runs=num_runs)

    # 3️⃣ GPU PyTorch
    torch_times = []
    for _ in range(num_runs):
        a_torch = torch.tensor(a, device='cuda')
        b_torch = torch.tensor(b, device='cuda')
        start = time.time()
        c_torch = torch.matmul(a_torch, b_torch)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start)
    c_torch_cpu = c_torch.cpu().numpy()

    # 결과 출력
    print("========== 실행 시간 (평균) ==========")
    print("NumPy (CPU):       {:.6f} sec".format(np.mean(numpy_times)))
    print("Numba CUDA (GPU):  {:.6f} sec".format(np.mean(gpu_times)))
    print("PyTorch (GPU):     {:.6f} sec".format(np.mean(torch_times)))

    print("\n========== 정확도 비교 (Max 오차) ==========")
    print("Numba vs NumPy:    {:.6e}".format(np.max(np.abs(c_cpu - c_gpu))))
    print("PyTorch vs NumPy:  {:.6e}".format(np.max(np.abs(c_cpu - c_torch_cpu))))
