#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// I started with the SAXPY example then slowly added more and new functionality over serveral iterations as I began to understand them this #3 unified memory
// Moved array initialize into GPU this comes from the devblogs on unified memory

// function for checking the CUDA runtime API results.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

__global__
void init(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

}

__global__
void saxpy(int n, float a, float *x, float *y)
{
	// Using a grid stide loop that can cover an entire array regardless of size. The stride is also configured for the number of SM's in the gpu

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride)
	{
		y[i] = a * x[i] + y[i];
	}
}

int main(void)
{
	cudaDeviceProp		prop;
	cudaEvent_t			start, stop;
	float				elapsed;

	int		N = 1 << 25;
	int		numSM;
	float	*x, *y;

	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));
	checkCuda(cudaEventRecord(start, 0));

	checkCuda(cudaMallocManaged(&x, N * sizeof(float)));
	checkCuda(cudaMallocManaged(&y, N * sizeof(float)));

	checkCuda(cudaGetDeviceProperties(&prop, 0));
	numSM = prop.multiProcessorCount;

	init << <64 * numSM, 1024 >> > (N, x, y);
	
	//The number of symmetrical multiprocessors times 64 ensures we launch the maximum number of warps per sm and maximizes memory coalescing
	// and shows a simple method for handling arrays that are much larger than the maximum number of threads used or available even when using dim3
	saxpy << <64 * numSM, 1024 >> > (N, 2.0f, x, y);

	// Wait for gpu to finsih before allowing host to access the memory

	cudaDeviceSynchronize();

	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	checkCuda(cudaEventElapsedTime(&elapsed, start, stop));

	float maxError = 0.0f;

	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));

	printf("Max error: %f\n", maxError);
	printf("Time to calculate: %3.2fms, this does not include any cpu time.\n", elapsed);

	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

	cudaFree(x);
	cudaFree(y);

	return 0;

}