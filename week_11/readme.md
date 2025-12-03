FlashMoE: Fast Distributed MoE Implementation

This project replaces the traditional, blocking NCCL AlltoAll communication with Non-blocking, Asynchronous One-sided RDMA using NVSHMEM.

Prerequisites

Hardware: NVIDIA GPUs (Pascal, Volta, Ampere, Hopper, or Ada Lovelace)

Software: CUDA Toolkit 

NVSHMEM Library (Must match your CUDA version)

1. Identify your CUDA Version

Run:

nvcc --version 

# Install the robust version compatible with CUDA 11 
sudo apt-get install nvshmem-cuda-11

Compilation

nvcc -rdc=true -arch=sm_80 flashMoE.cu -o flashMoE -lnvshmem -lcuda

Running the Verification

To verify the distributed logic, you must simulate multiple processes (ranks).

Using NVSHMEM Launcher:

# Simulates 2 GPUs
nvshmem-launch -np 2 ./flashMoE


Using MPI:

mpirun -np 2 ./flashMoE


Expected Output

FlashMoE Assignment: Single Kernel Implementation 
World Size (P): 2 GPUs
Launching Verification Kernel...
[PASS] GPU 0: Received correct RDMA from GPU 1
[PASS] GPU 1: Received correct RDMA from GPU 0
Test Completed Successfully.

