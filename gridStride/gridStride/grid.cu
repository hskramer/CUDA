#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
	// Using a grid stide loop that can cover an entire array regardless of size. The stride is also configured for the number of SM's in the gpu
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		y[i] = a * x[i] + y[i];
	}
}

int main(void)
{
	cudaDeviceProp		prop;
	int		numSM;

	int		N = 1 << 30;
	float	*x, *y, *d_x, *d_y;

	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));

	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaGetDeviceProperties(&prop, 0);
	numSM = prop.multiProcessorCount;

	//The number of symmetrical multiprocessors times 64 ensures we launch the maximum number of warps per sm and maximizes memory coalescing
	// and shows a simple method for handling arrays that are much larger than the maximum number of threads used or available when using dim3 x threads
	saxpy <<<64 * numSM,  1024>>> (N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;

	for(int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));

	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;

}