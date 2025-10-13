import numpy as np
import time

def matrix_multiply_cpu(A, B):
    return np.dot(A, B)

if __name__ == "__main__":
    N = 1024
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    start = time.time()
    C = matrix_multiply_cpu(A, B)
    end = time.time()

    print(f"CPU computation completed in {end - start:.4f} seconds")
