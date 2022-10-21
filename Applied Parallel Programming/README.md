# ECE408 Final Project for 2021 Fall
This is the personal record for UIUC Fall 2021 ECE408 course project.

## Introduction
In this final project, I was implementing and optimizing the forward-pass of a convolutional layer using CUDA. 

### Overall Learning Objectives:
* Demonstrating command of CUDA and optimization approaches by designing and implementing an optimized neural-network
 convolutional layer forward pass
* Obtaining practical experience in analyzing and fine tuning CUDA kernels through the use of profiling tools like Nsight Systems (```nsys```) 
 and Nsight-Compute (```nv-nsight-cu```)

### Three Main Parts:
* Milestone 1: Rai Installation, CPU convolution, Profiling
  * Testing Rai
  * Create a CPU Implementation
  * Specifying Batch Size
  * Use Gprof to profile CPU implementation
* Milestone 2: Baseline Convolutional Kernel
  * Create a GPU Implementation
  * Use Nsight-Systems and Nsight-Compute for initial Performance Results
* Milestone 3: GPU Convolution Kernel Optimizations
  * Add GPU Optimizations
  * Interpreting the timing output from rai
  * Performance Analysis with Nsight-Systems and Nsight-Compute
  
## Implemented Optimizations
1. Sweeping various parameters to find best values
2. Weight matrix (kernel values) in constant memory
3. Tuning with restrict and loop unrolling
4. Using Streams to overlap computation with data transfer
5. Multiple kernel implementations for different layer sizes

## Result:
*	Improved performance by 57% with above optimization steps.
