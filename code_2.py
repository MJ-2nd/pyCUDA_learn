import numpy as np
import time
from numba import cuda, float32

# 블록 사이즈 (타일 크기)
TPB = 16  # Threads Per Block

@cuda.jit
def matmul_shared_mem(A, B, C):
    sA = cuda.shared.array((TPB, TPB), float32)
    sB = cuda.shared.array((TPB, TPB), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TPB + ty
    col = cuda.blockIdx.x * TPB + tx

    tmp = 0.0

    for m in range((A.shape[1] + TPB - 1) // TPB):
        if row < A.shape[0] and (m * TPB + tx) < A.shape[1]:
            sA[ty, tx] = A[row, m * TPB + tx]
        else:
            sA[ty, tx] = 0.0

        if col < B.shape[1] and (m * TPB + ty) < B.shape[0]:
            sB[ty, tx] = B[m * TPB + ty, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

def gpu_matmul_shared(a, b, num_runs=10):
    d_A = cuda.to_device(a)
    d_B = cuda.to_device(b)
    d_C = cuda.device_array((a.shape[0], b.shape[1]), dtype=np.float32)

    threads_per_block = (TPB, TPB)
    blocks_per_grid = (
        (b.shape[1] + TPB - 1) // TPB,
        (a.shape[0] + TPB - 1) // TPB
    )

    gpu_times = []
    for _ in range(num_runs):
        start = time.time()
        matmul_shared_mem[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        cuda.synchronize()
        gpu_times.append(time.time() - start)

    c_res = d_C.copy_to_host()
    return c_res, gpu_times

if __name__ == "__main__":
    import torch

    size = 4096
    num_runs = 5

    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # NumPy
    numpy_times = []
    for _ in range(num_runs):
        start = time.time()
        c_numpy = np.matmul(a, b)
        numpy_times.append(time.time() - start)

    # Numba 최적화 버전
    c_gpu_opt, gpu_times_opt = gpu_matmul_shared(a, b, num_runs=num_runs)

    # PyTorch
    torch_times = []
    for _ in range(num_runs):
        a_torch = torch.tensor(a).cuda()
        b_torch = torch.tensor(b).cuda()
        start = time.time()
        c_torch = torch.matmul(a_torch, b_torch)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start)

    c_torch_cpu = c_torch.cpu().numpy()

    print("\n========== 실행 시간 (평균) ==========")
    print(f"NumPy (CPU):       {np.mean(numpy_times):.6f} sec")
    print(f"Numba CUDA (최적): {np.mean(gpu_times_opt):.6f} sec")
    print(f"PyTorch (GPU):     {np.mean(torch_times):.6f} sec")

    print("\n========== 정확도 비교 (Max 오차) ==========")
    print("Numba vs NumPy:   ", np.max(np.abs(c_numpy - c_gpu_opt)))
    print("PyTorch vs NumPy: ", np.max(np.abs(c_numpy - c_torch_cpu)))
