#include<stdio.h>
#include<math.h>

#define N 64
#define Block 2
#define thread 64

__global__ void exclusive_scan(int *d_in)
{
    //Phase 1 (Uptree)
    int s = 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(; s<=N-1; s<<=1)
    {
        int i = 2*s*(tid+1)-1;
        if((i-s >= 0) && (i<N)) {
            int a = d_in[i];
            int b = d_in[i-s];
            __syncthreads();
            d_in[i] = a+b;
            __syncthreads();

        }
        __syncthreads();
    }

    //Phase 2 (Downtree)
    if(tid == 0)
        d_in[N-1] = 0;
    
    for(s = s/2; s >= 1; s>>=1)
    {
        int i = 2*s*(tid+1)-1;
        if((i-s >= 0) && (i<N)) {
            int r = d_in[i];
            int l = d_in[i-s];
            __syncthreads();
            d_in[i] = l+r;
            d_in[i-s] = r;
            __syncthreads();
        }
        __syncthreads();
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
	exclusive_scan<<<Block, thread/2>>>(d_in);
    cudaEventRecord(stop);


    cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++)
		printf("out[%d] =  %d\n", i, h_out[i]);
    printf("%f milliseconds\n", milliseconds);

    cudaFree(d_in);


	return -1;

}
