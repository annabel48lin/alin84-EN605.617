#include <stdio.h> 

#define FILTER_RADIUS 5
__constant__ float filter_arr_gpu[2*FILTER_RADIUS + 1]; 

__global__ void filter_kernel_w_sharedmem(
    const float *input, 
    float* result, 
    int num_elem
) {
    extern __shared__ float tile[]; // dynamic shared memory. 
    // values defined here will use device register memory

    // each thread processes 1 elem
    int elem_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int idx_in_tile = threadIdx.x + FILTER_RADIUS;
    if (elem_idx < num_elem) {
        // put centered elem into shared memory. 
        // Since simd, effectively loads all elem in grid into tile
        tile[idx_in_tile] = input[elem_idx];

        // if leftmost elem needed is not in this section of grid, load it
        if (threadIdx.x < FILTER_RADIUS){
            if (elem_idx >= FILTER_RADIUS) {
                tile[idx_in_tile - FILTER_RADIUS] = 
                        input[elem_idx - FILTER_RADIUS];
            }
            else {
                // zero padding if filter exceeds input bounds
                tile[idx_in_tile - FILTER_RADIUS] = 0.0f; 
            }
        }

        // if rightmost elem needed is not in this section of grid, load it
        if (threadIdx.x >= blockDim.x - FILTER_RADIUS) {
            if (elem_idx + FILTER_RADIUS < num_elem) {
                tile[idx_in_tile + FILTER_RADIUS] = 
                        input[elem_idx + FILTER_RADIUS];
            } 
            else {
                // zero padding if filter exceeds input bounds
                tile[idx_in_tile + FILTER_RADIUS] = 0.0f; 
            }
        }

        __syncthreads();

        // Compute filter
        float computed = 0.0f;
        for (int fidx=-FILTER_RADIUS; fidx<=FILTER_RADIUS; fidx++) {
            computed += 
                filter_arr_gpu[FILTER_RADIUS + fidx] * tile[idx_in_tile + fidx];
        }
        result[elem_idx] = computed;
    }
}

__global__ void filter_kernel_no_sharedmem(
    const float* input, 
    float* result, 
    int num_elem
) {
    // values defined here will use device register memory
    // each thread processes 1 elem
    int elem_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (elem_idx < num_elem) {
        float computed = 0.0f;
        for (int fidx = -FILTER_RADIUS; fidx <= FILTER_RADIUS; fidx++) {
            int idx = elem_idx + fidx;
            float val = 0;
            if (idx >= 0 && idx < num_elem) {
                val = input[idx];
            }
            computed += filter_arr_gpu[FILTER_RADIUS + fidx] * val;
        }
        result[elem_idx] = computed;
    }
}


__global__ void warmup_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f; // simple operation to ensure loaded
    }
}

void warmup_flush_gpu_cache() {
    int N = (1<<20); // some arbitrary large number

    float* to_flush;
    cudaMalloc(&to_flush, N * sizeof(float));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    warmup_kernel<<<numBlocks, blockSize>>>(to_flush, N);
    cudaDeviceSynchronize();

    cudaFree(to_flush);
}



void memory_experiment(int blockSize, int numBlocks, int totalThreads) {
    // num elem = total threads. Each thread processes 1 elem
    int N = totalThreads; 

    // Host memory
    float *input_host = (float*) malloc(N * sizeof(float));
    float *result_shared_host = (float*) malloc(N * sizeof(float));
    float *result_global_host = (float*) malloc(N * sizeof(float));

    float filter_arr_host[2*FILTER_RADIUS+1] = {
        0.05f, 0.07f, 0.1f, 0.13f, 0.15f, 
        0.2f, 
        0.15f, 0.13f, 0.1f, 0.07f, 0.05f}; // radius 5


    srand(time(NULL));
    for (int i=0; i<N; i++) { // populate with random values
        input_host[i] = rand() / (float)RAND_MAX;
    }

    // Device memory 
    float *input_gpu, *result_shared_gpu, *result_global_gpu;
    cudaMalloc((void**) &input_gpu, N*sizeof(float));
    cudaMalloc((void**) &result_shared_gpu, N*sizeof(float));
    cudaMalloc((void**) &result_global_gpu, N*sizeof(float));

    // device global memory
    cudaMemcpy(input_gpu, input_host, N*sizeof(float), cudaMemcpyHostToDevice);
    // device constant memory
    cudaMemcpyToSymbol(filter_arr_gpu, filter_arr_host, 
            (2*FILTER_RADIUS+1)*sizeof(float));

    // device shared memory. Size: num elems in block + filter margins
    int sharedMemSize = (blockSize + 2*FILTER_RADIUS) * sizeof(float);

    // timing util
    cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

    // warm up and run version with shared memory
    warmup_flush_gpu_cache();
    cudaEventRecord(kernel_start, 0);
    filter_kernel_w_sharedmem<<<numBlocks, blockSize, sharedMemSize>>>(
        input_gpu, 
        result_shared_gpu, 
        N);
    cudaDeviceSynchronize();
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Launch error (during shared): %s\n", 
        cudaGetErrorString(err));

    float time_shared_ms = 0.0f;
    cudaEventElapsedTime(&time_shared_ms, kernel_start, kernel_stop);

    // warm up and run version without shared memory
    warmup_flush_gpu_cache();
    cudaEventRecord(kernel_start, 0);
    filter_kernel_no_sharedmem<<<numBlocks, blockSize>>>(input_gpu, 
        result_global_gpu, 
        N);

    cudaDeviceSynchronize();
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Launch error (during no shared): %s\n", 
        cudaGetErrorString(err));

    float time_global_ms = 0.0f;
    cudaEventElapsedTime(&time_global_ms, kernel_start, kernel_stop);


    // copy results back to host
    cudaMemcpy(result_shared_host, result_shared_gpu, 
        N*sizeof(float), 
        cudaMemcpyDeviceToHost);
    cudaMemcpy(result_global_host, result_global_gpu, 
        N*sizeof(float), 
        cudaMemcpyDeviceToHost);

    
    
    // print some results for sanity check. Comment out if not desired.
    int count = (N < 4) ? N : 4;  
    printf("Sanity Check: First/Last %d results:\n", count);
    for (int i = 0; i < count; i++) {
        printf("i=%d: shared=%f, global=%f \t(diff=%f)\n",
            i, result_shared_host[i], result_global_host[i],
            result_shared_host[i]-result_global_host[i]
        );
    }
    for (int i = N - count; i < N; i++) {
        printf("i=%d: shared=%f, global=%f \t(diff=%f)\n",
            i, result_shared_host[i], result_global_host[i], 
            result_shared_host[i]-result_global_host[i]
        );
    }

    printf("\nKernel execution times: shared=%.3f ms, no shared=%.3f ms\n", 
        time_shared_ms, time_global_ms);


    // cleanup
    cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);

    cudaFree(input_gpu);
    cudaFree(result_shared_gpu);
    cudaFree(result_global_gpu);

    free(input_host);
    free(result_shared_host);
    free(result_global_host);

}

int main(int argc, char**argv) {
    // read comand line arguments
    int totalThreads = (1<<20);
    int blockSize = 256;

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    int numBlocks = totalThreads/blockSize;

    if (blockSize < FILTER_RADIUS) {
        printf("Rounding blockSize up to filterRadius %d\n", FILTER_RADIUS);
        blockSize = FILTER_RADIUS;
    }

    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks*blockSize;
        printf("Warning: totalThreads is not evenly divisible by blockSize\n");
		printf("The totalThreads will be rounded up to %d\n", totalThreads);
    }

    printf("Run w totalThreads %d, blockSize %d, numBlocks %d, filterSize %d\n", 
        totalThreads, blockSize, numBlocks, (2*FILTER_RADIUS+1));

    memory_experiment(blockSize, numBlocks, totalThreads);
}