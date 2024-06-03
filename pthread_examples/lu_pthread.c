#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define BLOCK_SIZE 16

typedef struct {
    float *A;
    int n;
    int k;
    int threadIdx_x;
    int threadIdx_y;
    int threadIdx_z;
    int blockIdx_x;
    int blockIdx_y;
    int blockIdx_z;
    int blockDim_x;
    int blockDim_y;
    int blockDim_z;
    int gridDim_x;
    int gridDim_y;
    int gridDim_z;
} ThreadData;


void *lu_kernel1(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float *A = data->A;
    int n = data->n;
    int k = data->k;
    int threadIdx_x = data->threadIdx_x;
    int blockIdx_x = data->blockIdx_x;
    int blockDim_x = data->blockDim_x;

    int i = blockIdx_x * blockDim_x + threadIdx_x;
    if (i > k && i < n) {
        A[i * n + k] /= A[k * n + k];
    }

    return NULL;
}

void *lu_kernel2(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float *A = data->A;
    int n = data->n;
    int k = data->k;
    int threadIdx_x = data->threadIdx_x;
    int threadIdx_y = data->threadIdx_y;
    int blockIdx_x = data->blockIdx_x;
    int blockIdx_y = data->blockIdx_y;
    int blockDim_x = data->blockDim_x;
    int blockDim_y = data->blockDim_y;

    int i = blockIdx_y * blockDim_y + threadIdx_y;
    int j = blockIdx_x * blockDim_x + threadIdx_x;
    if (i > k && j > k && i < n && j < n) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
    }

    return NULL;
}

void initializeArray(float *A, int n) {
    A[0] = 30.00;    
    A[1] = 2.00;    
    A[2] = 1.00;    
    A[3] = 1.00;    
    A[4] = 30.00;    
    A[5] = 8.00;    
    A[6] = 6.00;    
    A[7] = 8.00;    
    A[8] = 30.00;    
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

void luPthread(float *A, int n, int blockDim_x, int blockDim_y, int blockDim_z, int gridDim_x, int gridDim_y, int gridDim_z) {
    int numThreads = blockDim_x * blockDim_y * blockDim_z;
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];


    for (int k = 0; k < n - 1; k++) {
        // Launch lu_kernel1 threads
        int tid = 0;
        for (int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++) {
            for (int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++) {
                if (tid < numThreads) {
                    threadData[tid] = (ThreadData){A, n, k, threadIdx_x, 0, 0, blockIdx_x, 0, 0, blockDim_x, 1, 1, gridDim_x, 1, 1};
                    pthread_create(&threads[tid], NULL, lu_kernel1, &threadData[tid]);
                    tid++;
                }
            }
        }
        for (int i = 0; i < tid; i++) {
            pthread_join(threads[i], NULL);
        }


        // Launch lu_kernel2 threads
        tid = 0;
        for (int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++) {
            for (int blockIdx_y = 0; blockIdx_y < gridDim_y; blockIdx_y++) {
                for (int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++) {
                    for (int threadIdx_y = 0; threadIdx_y < blockDim_y; threadIdx_y++) {
                        if (tid < numThreads) {
                            threadData[tid] = (ThreadData){A, n, k, threadIdx_x, threadIdx_y, 0, blockIdx_x, blockIdx_y, 0, blockDim_x, blockDim_y, 1, gridDim_x, gridDim_y, 1};
                            pthread_create(&threads[tid], NULL, lu_kernel2, &threadData[tid]);
                            tid++;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < tid; i++) {
            pthread_join(threads[i], NULL);
        }

    }

}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Please inform size!\n");
        exit(0);
    }

    int n = atoi(argv[1]);

    float *A = (float *)malloc(n * n * sizeof(float));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    initializeArray(A, n);

    printf("A Matrix:\n");
    printArray(A, n);

    clock_t start, stop;
    start = clock();

    luPthread(A, n, BLOCK_SIZE, BLOCK_SIZE, 1, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    stop = clock();
    float time = (float)(stop - start) / CLOCKS_PER_SEC * 1000;

    printf("Solution Matrix:\n");
    printArray(A, n);

    printf("PTHREAD\t%d\t%3.1f\n", n, time);

    free(A);

    return 0;
}
