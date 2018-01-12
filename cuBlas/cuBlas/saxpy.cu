#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

int main(void)
{
	cublasStatus_t		stat;
	cublasHandle_t		handle;

	cudaDeviceProp		prop;
	cudaEvent_t			start, stop;

	int		N = 1 << 20;
	int		numSM;

	float	*d_x, *d_y, *x, *y;
	float	 elapsed;

	const float a = 2.0f;


	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));
	checkCuda(cudaEventRecord(start, 0));

	checkCuda(cudaMalloc(&d_x, N * sizeof(float)));
	checkCuda(cudaMalloc(&d_y, N * sizeof(float)));

	checkCuda(cudaHostAlloc(&x, N * sizeof(float), cudaHostAllocDefault));
	checkCuda(cudaHostAlloc(&y, N * sizeof(float), cudaHostAllocDefault));

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);


	checkCuda(cudaGetDeviceProperties(&prop, 0));
	numSM = prop.multiProcessorCount;

	init << <64 * numSM, 1024 >> > (N, x, y);

	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("CUBLAS initialization failed\n");
		return	EXIT_FAILURE;
	}

	cublasSetVector(N, sizeof(x[0]), x, 1, d_x, 1);
	cublasSetVector(N, sizeof(y[0]), y, 1, d_y, 1);

	stat = cublasSaxpy_v2(handle, N, &a, d_x, 1, d_y, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasSaxpy failed\n");
		return EXIT_FAILURE;
	}


	// Wait for gpu to finsih before allowing host to access the memory

	cudaDeviceSynchronize();

	cublasGetVector(N, sizeof(y[0]), d_y, 1, y, 1);


	checkCuda(cudaEventRecord(stop, 0));
	checkCuda(cudaEventSynchronize(stop));
	checkCuda(cudaEventElapsedTime(&elapsed, start, stop));

	float maxError = 0.0f;

	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));

	printf("Max error: %f\n", maxError);
	printf("Time to calculate: %3.2fms, this does not include the time calculating the error\n", elapsed);

	cudaFree(x);
	cudaFree(y);

	return 0;

}