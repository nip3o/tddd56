// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *c_gpu, float *result_gpu) 
{
	result_gpu[threadIdx.x] = sqrt(c_gpu[threadIdx.x]);
}

int main()
{
	const int size = N*sizeof(float);

	float *c = new float[N];
	float *result = new float[N];

	float *c_gpu, *result_gpu;

	for (int i = 0; i < N; i++)
		c[i] = i;
	
	cudaMalloc( (void**)&c_gpu, size );
	cudaMalloc( (void**)&result_gpu, size );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );

	cudaMemcpy( c_gpu, c, size, cudaMemcpyHostToDevice ); 

	simple<<<dimGrid, dimBlock>>>(c_gpu, result_gpu);
	cudaThreadSynchronize();

	cudaMemcpy( result, result_gpu, size, cudaMemcpyDeviceToHost ); 

	cudaFree( c_gpu );
	cudaFree( result );
	
	for (int i = 0; i < N; i++) {
		printf("%f ", result[i]);
		if(result[i] != sqrt(c[i])) {
			printf("Diff, %e", result[i] - sqrt(c[i]));
		}
		printf("\n");
	}

	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
