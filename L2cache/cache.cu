#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <stdio.h>
#include <stdlib.h>

#define MEGABYTE 1048576

__global__ void kernel(const int *in, int *out)
{
	out[0 + threadIdx.x] = in[0 + threadIdx.x];
}

int main(int argc, char* argv[])
{

	int *dev_in = 0;
	int *dev_out = 0;

	int * in = (int*)malloc(sizeof(int) * MEGABYTE);
	int * out = (int*)malloc(sizeof(int) * MEGABYTE);

	for (int i = 0; i < MEGABYTE; i++)
		in[i] = i;

	cudaMalloc((void**)&dev_in, sizeof(int) * MEGABYTE);
	cudaMalloc((void**)&dev_out, sizeof(int) * MEGABYTE);

	cudaProfilerStart();

	cudaMemcpy(dev_in, in, sizeof(int) * MEGABYTE, cudaMemcpyHostToDevice);

	kernel << <1, 32 >> >(dev_in, dev_out);

	cudaMemcpy(out, dev_out, sizeof(int) * MEGABYTE, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaProfilerStop();


	free(in);
	free(out);
	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaDeviceReset();

	return 0;
}