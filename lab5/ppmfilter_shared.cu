
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glut.h>
#endif

#define BLOCK_DIM 16
#define IMAGE_DIM 512
#define IMAGE_SIZE 3*((4 + IMAGE_DIM / BLOCK_DIM) * (4 + IMAGE_DIM / BLOCK_DIM))

__global__ void filter(unsigned char *image, unsigned char *out, int n, int m)
{
	__shared__ unsigned char shared[IMAGE_SIZE];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int shared_i = threadIdx.x;
	int shared_j = threadIdx.y;

	int sumx, sumy, sumz, k, l;

	shared[(shared_i * blockDim.x + shared_j)*3+0] = image[(i * n + j)*3+0];
	shared[(shared_i * blockDim.x + shared_j)*3+1] = image[(i * n + j)*3+1];
	shared[(shared_i * blockDim.x + shared_j)*3+2] = image[(i * n + j)*3+2];

	__syncthreads();

// printf is OK under --device-emulation
	

//		printf("%d %d %d %d\n", i, j, n, m);


	if (i > 1 && i < m-2 && j > 1 && j < n-2)
		{
			// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-2;k<3;k++)
				for(l=-2;l<3;l++)
				{
					if (shared_i > 1 && shared_i < blockDim.y-2 && shared_j > 1 && shared_j < blockDim.x-2) {
						sumx += shared[((shared_i+k)*blockDim.x + (shared_j+l))*3+0];
						sumy += shared[((shared_i+k)*blockDim.x + (shared_j+l))*3+1];
						sumz += shared[((shared_i+k)*blockDim.x + (shared_j+l))*3+2];
					} else {
						sumx += image[((i+k)*n+(j+l))*3+0];
						sumy += image[((i+k)*n+(j+l))*3+1];
						sumz += image[((i+k)*n+(j+l))*3+2];
					}

				}
			out[(i*n+j)*3+0] = sumx/25;
			out[(i*n+j)*3+1] = sumy/25;
			out[(i*n+j)*3+2] = sumz/25;
		} else {
			out[(i*n+j)*3+0] = shared[(shared_i * blockDim.x + shared_j)*3+0];
			out[(i*n+j)*3+1] = shared[(shared_i * blockDim.x + shared_j)*3+1];
			out[(i*n+j)*3+2] = shared[(shared_i * blockDim.x + shared_j)*3+2];	
		}
}


// Compute CUDA kernel and display image
void Draw()
{
	unsigned char *image, *out;
	int n, m;
	unsigned char *dev_image, *dev_out;

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	printf("Memory per block: %d B\n", prop.sharedMemPerBlock);
	
	image = readppm("maskros512.ppm", &n, &m);
	out = (unsigned char*) malloc(n*m*3);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

	
	cudaMalloc( (void**)&dev_image, n*m*3);
	cudaMalloc( (void**)&dev_out, n*m*3);
	cudaMemcpy( dev_image, image, n*m*3, cudaMemcpyHostToDevice);
	
	dim3 dimBlock( 16, 16 );
	dim3 dimGrid( 32, 32 );
	
	filter<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();
	
	cudaMemcpy( out, dev_out, n*m*3, cudaMemcpyDeviceToHost );


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float theTime;
    cudaEventElapsedTime(&theTime, start, stop);
    printf("Things took %f ms\n", theTime);
    

	cudaFree(dev_image);
	cudaFree(dev_out);
	
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glRasterPos2f(-1, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, image );
	glRasterPos2i(0, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, out );
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
	glutInitWindowSize( 1024, 512 );
	glutCreateWindow("CUDA on live GL");
	glutDisplayFunc(Draw);
	
	glutMainLoop();
}
