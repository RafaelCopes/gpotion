#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void lu_kernel1(float *A, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > k && i < n) {
        A[i * n + k] /= A[k * n + k];
    }
}

__global__ void lu_kernel2(float *A, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > k && j > k && i < n && j < n) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
    }
}

void initializeArray(float *A, int n) {
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i * n + j] = (rand() % 9) + 1;
            }
        }
        A[i * n + i] = n * 10;
    }
}

void printArray(float *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%5.2f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void luCuda(float *A, int n) {
    float *AGpu;
    cudaMalloc(&AGpu, n * n * sizeof(float));
    cudaMemcpy(AGpu, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int k = 0; k < n - 1; k++) {
        lu_kernel1<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(AGpu, n, k);
        cudaDeviceSynchronize();
        lu_kernel2<<<grid, block>>>(AGpu, n, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(A, AGpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(AGpu);
}

int main(int argc, char *argv[]) {
    if (argc != 2){
        printf("Please inform size!\n");
        exit(0);
    }

    int n = atoi(argv[1]);
   
    float *A = (float *)malloc(n * n * sizeof(float));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    A[0] = 30.00;  
    A[1] =2.00;
    A[2] =1.00;
    A[3] =4.00;
    A[4] =30.00;
    A[5] =3.00;
    A[6] =4.00;
    A[7] =6.00;
    A[8] =30.00;

    printf("A Matrix:\n");
    printArray(A, n);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    luCuda(A, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Solution Matrix:\n");
    printArray(A, n);

    printf("CUDA\t%d\t%3.1f\n", n, time);

    free(A);
    
    return 0;
}
