#include <iostream>
#include <cuda.h>

#define N 3 // Define the size of the matrix

__global__ void matrixAdd(int *A, int *B, int *C) {
    // Define shared memory for the block
    __shared__ int shared_A[N][N];
    __shared__ int shared_B[N][N];

    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (row < N && col < N) {
        shared_A[threadIdx.y][threadIdx.x] = A[row * N + col];
        shared_B[threadIdx.y][threadIdx.x] = B[row * N + col];
    }
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Perform matrix addition
    if (row < N && col < N) {
        C[row * N + col] = shared_A[threadIdx.y][threadIdx.x] + shared_B[threadIdx.y][threadIdx.x];
    }
}

int main() {
    int size = N * N * sizeof(int);
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
