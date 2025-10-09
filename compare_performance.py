import numpy as np
import time
from numba import cuda
from matrix_multiply_cpu import matrix_multiply_cpu
from matrix_multiply_gpu import gpu_matrix_multiply


def compare():
    N = 1024
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # CPU
    start_cpu = time.time()
    matrix_multiply_cpu(A, B)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # GPU
    start_gpu = time.time()
    gpu_matrix_multiply(A, B)
    cuda.synchronize()
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")


if __name__ == "__main__":
    compare()
