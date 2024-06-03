#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define BLOCK_SIZE 16

typedef struct {
    int x;
    int y;
    int z;
} dim3;

typedef struct {
    float *a;
    float *b;
    float *c;
    int n;
    dim3 gridDim;
    dim3 blockDim;
    dim3 blockIdx;
    dim3 threadIdx;
} ThreadData;

void *vectorAdd(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float *a = data->a;
    float *b = data->b;
    float *c = data->c;
    int n = data->n;
    dim3 threadIdx = data->threadIdx;
    dim3 blockIdx = data->blockIdx;
    dim3 blockDim = data->blockDim;
    dim3 gridDim = data->gridDim;

    int index = (threadIdx.x + (blockIdx.x * blockDim.x));
    int stride = (blockDim.x * gridDim.x);
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }

    printf("Thread %d: index=%d, stride=%d\n", (threadIdx.x + blockIdx.x * blockDim.x), index, stride);

    return NULL;
}

void initializeVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = i;
    }
}

void printVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        printf("%5.2f ", vec[i]);
    }
    printf("\n");
}

void vectorAddPthread(float *a, float *b, float *c, int n, dim3 gridDim, dim3 blockDim) {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;
    int totalThreads = gridDim.x * gridDim.y * gridDim.z * numThreads;

    pthread_t *threads = (pthread_t *)malloc(totalThreads * sizeof(pthread_t));
    if (threads == NULL) {
        fprintf(stderr, "Error allocating memory for threads\n");
        exit(1);
    }
    ThreadData *threadData = (ThreadData *)malloc(totalThreads * sizeof(ThreadData));
    if (threadData == NULL) {
        fprintf(stderr, "Error allocating memory for thread data\n");
        exit(1);
    }

    int tid = 0;
    for (int blockIdx_z = 0; blockIdx_z < gridDim.z; ++blockIdx_z) {
        for (int blockIdx_y = 0; blockIdx_y < gridDim.y; ++blockIdx_y) {
            for (int blockIdx_x = 0; blockIdx_x < gridDim.x; ++blockIdx_x) {
                for (int threadIdx_z = 0; threadIdx_z < blockDim.z; ++threadIdx_z) {
                    for (int threadIdx_y = 0; threadIdx_y < blockDim.y; ++threadIdx_y) {
                        for (int threadIdx_x = 0; threadIdx_x < blockDim.x; ++threadIdx_x) {
                            threadData[tid] = (ThreadData){a, b, c, n, gridDim, blockDim, {blockIdx_x, blockIdx_y, blockIdx_z}, {threadIdx_x, threadIdx_y, threadIdx_z}};
                            if (pthread_create(&threads[tid], NULL, vectorAdd, &threadData[tid]) != 0) {
                                fprintf(stderr, "Error creating thread %d\n", tid);
                                exit(1);
                            }
                            tid++;
                        }
                    }
                }

                // Join threads of the current block
                for (int i = tid - numThreads; i < tid; ++i) {
                    pthread_join(threads[i], NULL);
                }
            }
        }
    }

    free(threads);
    free(threadData);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Please inform size!\n");
        exit(0);
    }

    int n = atoi(argv[1]);

    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *c = (float *)malloc(n * sizeof(float));
    if (a == NULL || b == NULL || c == NULL) {
        printf("Memory allocation for vectors failed\n");
        exit(1);
    }

    initializeVector(a, n);
    initializeVector(b, n);

    int blockDim_x = BLOCK_SIZE;
    int gridDim_x = (n + blockDim_x - 1) / blockDim_x;

    dim3 gridDim = {gridDim_x, 1, 1};
    dim3 blockDim = {blockDim_x, 1, 1};

    clock_t start, stop;
    start = clock();

    vectorAddPthread(a, b, c, n, gridDim, blockDim);

    stop = clock();
    float time = (float)(stop - start) / CLOCKS_PER_SEC * 1000;

    printVector(a, n);
    printVector(b, n);
    printVector(c, n);

    printf("PTHREAD\t%d\t%3.1f\n", n, time);

    free(a);
    free(b);
    free(c);

    return 0;
}
