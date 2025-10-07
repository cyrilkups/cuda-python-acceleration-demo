GPU Acceleration Demo — Fundamentals of Accelerated Computing with Python

This repository demonstrates the fundamentals of GPU acceleration in Python using CUDA and Numba.
It showcases how parallel computing and GPU optimization can drastically improve computation speed for large-scale mathematical operations.

Overview

The project implements a GPU-accelerated matrix multiplication using Numba’s CUDA JIT compiler.
It compares GPU and CPU execution performance to highlight the benefits of accelerated computing.
Performance profiling is conducted using NVIDIA Nsight Systems to analyze kernel execution time and memory throughput.

Key Concepts

CUDA programming and thread parallelism

GPU memory management (host-device transfers)

Numba’s @cuda.jit for just-in-time GPU compilation

Profiling GPU performance using NVIDIA Nsight Systems

Features

Implemented GPU-accelerated matrix multiplication using Numba’s CUDA JIT compiler

Achieved up to 10× speedup over standard CPU-based execution

Benchmarked and profiled performance using NVIDIA Nsight Systems

Includes reproducible code for CPU vs. GPU execution comparisons

Repository Structure
cuda-python-acceleration-demo/
│
├── notebooks/
│   ├── gpu_acceleration_demo.ipynb       # Main demonstration notebook
│
├── src/
│   ├── matrix_multiply_cpu.py            # CPU implementation
│   ├── matrix_multiply_gpu.py            # GPU implementation (Numba CUDA)
│
├── profiling/
│   ├── nsight_results.qdrep               # Nsight profiling report
│
├── requirements.txt
└── README.md

Benchmark Example
Implementation	Matrix Size	Average Time (s)	Speedup
CPU (NumPy)	1024×1024	1.21 s	1×
GPU (CUDA JIT)	1024×1024	0.12 s	~10×
Requirements

Python 3.9 or higher

NumPy

Numba

NVIDIA GPU with CUDA Toolkit installed

NVIDIA Nsight Systems (for profiling)

Install dependencies:

pip install -r requirements.txt

Running the Project

Clone the repository:

git clone https://github.com/CyrilOforiKupualor/cuda-python-acceleration-demo.git
cd cuda-python-acceleration-demo


Run the CPU version:

python src/matrix_multiply_cpu.py


Run the GPU version:

python src/matrix_multiply_gpu.py


(Optional) Profile the GPU performance:

nsys profile python src/matrix_multiply_gpu.py

Profiling Insights

Performance profiling using NVIDIA Nsight Systems provides visualizations for:

GPU kernel execution timelines

Memory transfer throughput

CPU–GPU synchronization latency

Learning Outcomes

Through this project, you will:

Understand GPU acceleration fundamentals with CUDA and Python

Learn how to apply Numba for GPU-based parallel computation

Perform GPU profiling and performance analysis

Compare and interpret CPU vs. GPU performance metrics

Author

Cyril Ofori Kupualor
Computer Science Major, Grambling State University
GitHub: github.com/CyrilOforiKupualor

LinkedIn: linkedin.com/in/cyriloforikupualor

Acknowledgments

This project is part of NVIDIA’s Fundamentals of Accelerated Computing with Python course.
Special thanks to the NVIDIA Deep Learning Institute (DLI) for educational materials and tooling support.
