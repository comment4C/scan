#include<stdio.h>
#include<math.h>

#define N 10000
#define Block 100
#define thread 100

__global__ void inclusive_scan(int *d_in)
{
	__shared__ int temp_in[N];

	int i = threadIdx.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	temp_in[tid] = d_in[tid];
	__syncthreads();

	

	for(unsigned int s = 1; s <= N-1; s <<= 1)
	{
		if((i >= s) && (i < N)) {
			int a = temp_in[tid]; 
			int b = temp_in[tid-s];
			__syncthreads();
			int c = a + b;
			temp_in[tid] = c;
		}
		__syncthreads();
	}

	

	d_in[tid] = temp_in[tid];
	if (blockDim.x != 0) {
		d_in[tid] += blockIdx.x * blockDim.x;
	}
}

int main()
{
	int h_in[N];
	int h_out[N];

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i=0; i < N; i++)
		h_in[i] = 1;

	int *d_in;

	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);
	
	//Implementing kernel call
	cudaEventRecord(start);
	inclusive_scan<<<Block, thread>>>(d_in);
	cudaEventRecord(stop);


	cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_in);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	for(int i=0; i<N; i++)
		printf("out[%d] =  %d\n", i, h_out[i]); 
	printf("%f milliseconds\n", milliseconds);

	return -1;

}
