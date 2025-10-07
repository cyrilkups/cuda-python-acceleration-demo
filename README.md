# cuda-python-acceleration-demo
GPU Acceleration Demo — Fundamentals of Accelerated Computing with Python

This project demonstrates the fundamentals of GPU acceleration in Python using CUDA and Numba, highlighting performance optimization through parallel computing.
It is inspired by NVIDIA’s Fundamentals of Accelerated Computing with Python course and focuses on understanding how to leverage GPU resources to achieve massive computation speedups.

🚀 Overview

This demo implements a GPU-accelerated matrix multiplication using Numba’s @cuda.jit compiler.
The goal is to show how moving compute-intensive operations from the CPU to the GPU can deliver significant performance improvements.

🧠 Key Concepts

CUDA Programming: Parallel execution of code on NVIDIA GPUs.

Numba CUDA JIT: Just-in-time compilation for Python functions to run on GPUs.

Memory Management: Transferring data between host (CPU) and device (GPU) memory.

Kernel Profiling: Analyzing performance using NVIDIA Nsight Systems.

🧩 Features

Implemented GPU-accelerated matrix multiplication using Numba CUDA JIT

Achieved up to 10× speedup compared to CPU-based execution

Profiled GPU performance using NVIDIA Nsight Systems for kernel execution and memory throughput analysis

Demonstrates baseline CPU vs. GPU benchmarking with reproducible scripts

📁 Repository Structure
gpu-acceleration-demo/
│
├── notebooks/
│   ├── gpu_acceleration_demo.ipynb     # Main demo notebook
│
├── src/
│   ├── matrix_multiply_cpu.py          # CPU-based matrix multiplication
│   ├── matrix_multiply_gpu.py          # GPU-accelerated version using Numba
│
├── profiling/
│   ├── nsight_results.qdrep             # Nsight profiling report
│
├── requirements.txt
└── README.md

🧮 Benchmark Example
Implementation	Matrix Size	Avg Time (s)	Speedup
CPU (NumPy)	1024×1024	1.21 s	1×
GPU (CUDA JIT)	1024×1024	0.12 s	~10×
🧰 Requirements

Python 3.9+

Numba

NumPy

NVIDIA GPU (with CUDA Toolkit installed)

NVIDIA Nsight Systems (for profiling)

Install dependencies:

pip install -r requirements.txt

▶️ How to Run

Clone the repository:

git clone https://github.com/<your-username>/gpu-acceleration-demo.git
cd gpu-acceleration-demo


Run the CPU version:

python src/matrix_multiply_cpu.py


Run the GPU version:

python src/matrix_multiply_gpu.py


(Optional) Profile performance:

nsys profile python src/matrix_multiply_gpu.py

📊 Profiling Insights

Performance profiling was performed using NVIDIA Nsight Systems, which visualized:

GPU kernel execution timelines

Memory transfer throughput

CPU–GPU synchronization overhead

📘 Learning Outcomes

By completing this demo, you’ll:

Understand how CUDA enables Python code acceleration

Learn to use Numba for GPU-based parallel computation

Analyze and optimize GPU performance with Nsight Systems

Appreciate tradeoffs between CPU and GPU compute architectures
