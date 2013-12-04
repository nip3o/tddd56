// Matrix addition, GPU version
// nvcc matrix_gpu.cu -L /usr/local/cuda/lib -lcudart -arch=sm_20 -o matrix_gpu 

#include <stdio.h>

const int GRIDSIZE = 1;
const int BLOCKSIZE = 16;


__global__
void multiply(float *a, float *b, float *c, int N) {
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    float sum = 0;
    //for (int i = 0; i < N; i++) {
    //    sum = sum + (a[row * N + i] * b[i * N + col]);
    //}
    sum = a[row*N + col] + b[row*N + col];
    c[N*row + col] = sum;
}

int main()
{
    const int N = 16;  // matrix size

    const int size = N*N*sizeof(float);

    float *a = new float[N*N];
    float *b = new float[N*N];
    float *c = new float[N*N];

    float *a_gpu, *b_gpu, *c_gpu;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i+j*N] = 10 + i;
            b[i+j*N] = (float)j / N;
        }
    }

    cudaMalloc((void**)&a_gpu, size);
    cudaMalloc((void**)&b_gpu, size);
    cudaMalloc((void**)&c_gpu, size);

    cudaMemcpy( a_gpu, a, size, cudaMemcpyHostToDevice ); 
    cudaMemcpy( b_gpu, b, size, cudaMemcpyHostToDevice ); 

    dim3 dimBlock( BLOCKSIZE, BLOCKSIZE );
    dim3 dimGrid( GRIDSIZE, GRIDSIZE );

    multiply<<<dimGrid, dimBlock>>>(a_gpu, b_gpu, c_gpu, N);
    cudaThreadSynchronize();

    cudaMemcpy( c, c_gpu, size, cudaMemcpyDeviceToHost ); 


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%0.2f ", c[i+j*N]);
        }
        printf("\n");
    }
}
