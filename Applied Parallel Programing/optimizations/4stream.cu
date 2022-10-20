#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_WIDTH 16
__constant__ float deviceKernel[4096];

#define nStreams 50
cudaStream_t stream[nStreams];

__global__ void conv_forward_kernel(float *__restrict y, const float *__restrict x, const int B, const int M, const int C, const int H, const int W, const int K, const int offset)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
	int W_grid = ceil(W_out/float(BLOCK_WIDTH));
	
	int n = blockIdx.x+offset;
	int m = blockIdx.y;
	int h_id = (blockIdx.z/W_grid)*BLOCK_WIDTH + threadIdx.y;
	int w_id = (blockIdx.z%W_grid)*BLOCK_WIDTH + threadIdx.x;

	if( w_id < W_out && h_id < H_out){
		float acc =0.0f;
		y4d(n, m, h_id, w_id) = 0;
		for(int c_id = 0; c_id < C; c_id++){ // sum over all input feature maps
			// KxK filter
			acc += x4d(n, c_id, h_id + 0, w_id + 0) * k4d(m, c_id, 0, 0);
			acc += x4d(n, c_id, h_id + 0, w_id + 1) * k4d(m, c_id, 0, 1);
			acc += x4d(n, c_id, h_id + 0, w_id + 2) * k4d(m, c_id, 0, 2);
			acc += x4d(n, c_id, h_id + 0, w_id + 3) * k4d(m, c_id, 0, 3);
			acc += x4d(n, c_id, h_id + 0, w_id + 4) * k4d(m, c_id, 0, 4);
			acc += x4d(n, c_id, h_id + 0, w_id + 5) * k4d(m, c_id, 0, 5);
			acc += x4d(n, c_id, h_id + 0, w_id + 6) * k4d(m, c_id, 0, 6);
			
			acc += x4d(n, c_id, h_id + 1, w_id + 0) * k4d(m, c_id, 1, 0);
			acc += x4d(n, c_id, h_id + 1, w_id + 1) * k4d(m, c_id, 1, 1);
			acc += x4d(n, c_id, h_id + 1, w_id + 2) * k4d(m, c_id, 1, 2);
			acc += x4d(n, c_id, h_id + 1, w_id + 3) * k4d(m, c_id, 1, 3);
			acc += x4d(n, c_id, h_id + 1, w_id + 4) * k4d(m, c_id, 1, 4);
			acc += x4d(n, c_id, h_id + 1, w_id + 5) * k4d(m, c_id, 1, 5);
			acc += x4d(n, c_id, h_id + 1, w_id + 6) * k4d(m, c_id, 1, 6);
			
			acc += x4d(n, c_id, h_id + 2, w_id + 0) * k4d(m, c_id, 2, 0);
			acc += x4d(n, c_id, h_id + 2, w_id + 1) * k4d(m, c_id, 2, 1);
			acc += x4d(n, c_id, h_id + 2, w_id + 2) * k4d(m, c_id, 2, 2);
			acc += x4d(n, c_id, h_id + 2, w_id + 3) * k4d(m, c_id, 2, 3);
			acc += x4d(n, c_id, h_id + 2, w_id + 4) * k4d(m, c_id, 2, 4);
			acc += x4d(n, c_id, h_id + 2, w_id + 5) * k4d(m, c_id, 2, 5);
			acc += x4d(n, c_id, h_id + 2, w_id + 6) * k4d(m, c_id, 2, 6);
			
			acc += x4d(n, c_id, h_id + 3, w_id + 0) * k4d(m, c_id, 3, 0);
			acc += x4d(n, c_id, h_id + 3, w_id + 1) * k4d(m, c_id, 3, 1);
			acc += x4d(n, c_id, h_id + 3, w_id + 2) * k4d(m, c_id, 3, 2);
			acc += x4d(n, c_id, h_id + 3, w_id + 3) * k4d(m, c_id, 3, 3);
			acc += x4d(n, c_id, h_id + 3, w_id + 4) * k4d(m, c_id, 3, 4);
			acc += x4d(n, c_id, h_id + 3, w_id + 5) * k4d(m, c_id, 3, 5);
			acc += x4d(n, c_id, h_id + 3, w_id + 6) * k4d(m, c_id, 3, 6);
			
			acc += x4d(n, c_id, h_id + 4, w_id + 0) * k4d(m, c_id, 4, 0);
			acc += x4d(n, c_id, h_id + 4, w_id + 1) * k4d(m, c_id, 4, 1);
			acc += x4d(n, c_id, h_id + 4, w_id + 2) * k4d(m, c_id, 4, 2);
			acc += x4d(n, c_id, h_id + 4, w_id + 3) * k4d(m, c_id, 4, 3);
			acc += x4d(n, c_id, h_id + 4, w_id + 4) * k4d(m, c_id, 4, 4);
			acc += x4d(n, c_id, h_id + 4, w_id + 5) * k4d(m, c_id, 4, 5);
			acc += x4d(n, c_id, h_id + 4, w_id + 6) * k4d(m, c_id, 4, 6);
			
			acc += x4d(n, c_id, h_id + 5, w_id + 0) * k4d(m, c_id, 5, 0);
			acc += x4d(n, c_id, h_id + 5, w_id + 1) * k4d(m, c_id, 5, 1);
			acc += x4d(n, c_id, h_id + 5, w_id + 2) * k4d(m, c_id, 5, 2);
			acc += x4d(n, c_id, h_id + 5, w_id + 3) * k4d(m, c_id, 5, 3);
			acc += x4d(n, c_id, h_id + 5, w_id + 4) * k4d(m, c_id, 5, 4);
			acc += x4d(n, c_id, h_id + 5, w_id + 5) * k4d(m, c_id, 5, 5);
			acc += x4d(n, c_id, h_id + 5, w_id + 6) * k4d(m, c_id, 5, 6);
			
			acc += x4d(n, c_id, h_id + 6, w_id + 0) * k4d(m, c_id, 6, 0);
			acc += x4d(n, c_id, h_id + 6, w_id + 1) * k4d(m, c_id, 6, 1);
			acc += x4d(n, c_id, h_id + 6, w_id + 2) * k4d(m, c_id, 6, 2);
			acc += x4d(n, c_id, h_id + 6, w_id + 3) * k4d(m, c_id, 6, 3);
			acc += x4d(n, c_id, h_id + 6, w_id + 4) * k4d(m, c_id, 6, 4);
			acc += x4d(n, c_id, h_id + 6, w_id + 5) * k4d(m, c_id, 6, 5);
			acc += x4d(n, c_id, h_id + 6, w_id + 6) * k4d(m, c_id, 6, 6);
			
		}
		y4d(n, m, h_id, w_id) = acc;
	}
#undef y4d
#undef x4d
#undef k4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *__restrict host_y, const float *__restrict host_x, const float *__restrict host_k, float **__restrict device_y_ptr, float **__restrict device_x_ptr, float **__restrict device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
	
	int size_k = M * C * K * K * sizeof(float);
	
	cudaMalloc((void **) device_y_ptr, B * M * (H - K + 1) * (W - K + 1) * sizeof(float));
	cudaMalloc((void **) device_x_ptr, B * C * H * W * sizeof(float));
	//cudaMalloc((void **) device_k_ptr, size_k);
	
	for(int i = 0; i < nStreams; ++i)
		cudaStreamCreate(&stream[i]);
	
	const int streamSize1 = B*M*(H-K+1)*(W-K+1)/nStreams;
	const int streamSize2 = B*C*H*W/nStreams;
	
	for (int i = 0; i < nStreams; ++i)
	{
		int offset1 = i * streamSize1;
		int offset2 = i * streamSize2;
		cudaMemcpyAsync(*device_y_ptr+offset1, (void *)(host_y+offset1), B*M*(H-K+1)*(W-K+1)*sizeof(float)/nStreams, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(*device_x_ptr+offset2, (void *)(host_x+offset2), B*C*H*W*sizeof(float)/nStreams, cudaMemcpyHostToDevice, stream[i]);
	}
	//cudaMemcpy(*device_k_ptr, (void *)host_k, size_k, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(deviceKernel, host_k, size_k, 0, cudaMemcpyHostToDevice);
	//cudaError_t error = cudaGetLastError();
    //if(error != cudaSuccess)
    //{
    //    std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //    exit(-1);
    //}
}


__host__ void GPUInterface::conv_forward_gpu(float *__restrict device_y, const float *__restrict device_x, const float *__restrict device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
	const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	const int H_grid = ceil(H_out/float(BLOCK_WIDTH));
	const int W_grid = ceil(W_out/float(BLOCK_WIDTH));
	
	const int streamSize = B/nStreams;
	dim3 dimGrid(B/nStreams, M, H_grid*W_grid);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	for (int i = 0; i < nStreams; ++i)
	{
		int offset = i * streamSize;
		conv_forward_kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(device_y, device_x, B, M, C, H, W, K, offset);
	}
	cudaDeviceSynchronize();
	//cudaError_t error = cudaGetLastError();
    //if(error != cudaSuccess)
    //{
    //    std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //    exit(-1);
    //}
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *__restrict host_y, float *__restrict device_y, float *__restrict device_x, float *__restrict device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
	const int streamSize = B*M*(H - K + 1)*(W - K + 1)/nStreams;
	for (int i = 0; i < nStreams; ++i)
	{
		int offset = i * streamSize;
		cudaMemcpyAsync(host_y+offset, device_y+offset, B*M*(H - K + 1)*(W - K + 1)*sizeof(float)/nStreams, cudaMemcpyDeviceToHost, stream[i]);
	}
    // Free device memory
	cudaFree(device_y);
	cudaFree(device_x);
	//cudaFree(device_k);
	//cudaError_t error = cudaGetLastError();
    //if(error != cudaSuccess)
    //{
    //    std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //    exit(-1);
    //}
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
