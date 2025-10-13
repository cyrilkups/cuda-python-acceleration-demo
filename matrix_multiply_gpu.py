import numpy as np
from numba import cuda
import math
import time

@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def gpu_matrix_multiply(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(N / threads_per_block[0])
    blocks_per_grid_y = math.ceil(N / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((N, N))

    matmul_kernel[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_global_mem)

    C_global_mem.copy_to_host(C)
    return C

if __name__ == "__main__":
    N = 1024
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    start = time.time()
    C = gpu_matrix_multiply(A, B)
    cuda.synchronize()
    end = time.time()

    print(f"GPU computation completed in {end - start:.4f} seconds")
