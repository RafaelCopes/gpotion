#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define imin(a, b) (a < b ? a : b)

const int threadsPerBlock = 256;
pthread_barrier_t barrier;

typedef struct {
    int x;
    int y;
    int z;
} dim3;

typedef struct {
    float *a;
    float *b;
    float *c;
    int N;
    dim3 gridDim;
    dim3 blockDim;
    dim3 blockIdx;
    dim3 threadIdx;
    float *cache;
} ThreadData;

void *thread_func(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float *a = data->a;
    float *b = data->b;
    float *c = data->c;
    int N = data->N;
    dim3 blockDim = data->blockDim;
    dim3 blockIdx = data->blockIdx;
    dim3 threadIdx = data->threadIdx;
    float *cache = data->cache;

    int threadIndexInBlock = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int blockIndexInGrid = blockIdx.x + blockIdx.y * data->gridDim.x + blockIdx.z * data->gridDim.x * data->gridDim.y;
    int globalThreadIndex = threadIndexInBlock + blockIndexInGrid * (blockDim.x * blockDim.y * blockDim.z);

    int cacheIndex = threadIndexInBlock;
    float temp = 0;
    while (globalThreadIndex < N) {
        temp += a[globalThreadIndex] * b[globalThreadIndex];
        globalThreadIndex += blockDim.x * blockDim.y * blockDim.z * data->gridDim.x * data->gridDim.y * data->gridDim.z;
    }

    cache[cacheIndex] = temp;

    printf("Thread (%d, %d, %d): globalThreadIndex = %d, temp = %f\n", blockIdx.x, threadIdx.x, cacheIndex, globalThreadIndex, temp);

    pthread_barrier_wait(&barrier);

    int i = blockDim.x * blockDim.y * blockDim.z / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }

        pthread_barrier_wait(&barrier);
        i /= 2;
    }

    if (cacheIndex == 0) {
        int blockId = blockIdx.x + blockIdx.y * data->gridDim.x + blockIdx.z * data->gridDim.x * data->gridDim.y;
        c[blockId] = cache[0];
        printf("Block (%d, %d, %d): cache[0] = %f\n", blockIdx.x, blockIdx.y, blockIdx.z, cache[0]);
    }

    return NULL;
}

void dot(float *a, float *b, float *c, dim3 gridDim, dim3 blockDim, int N) {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;
    if (pthread_barrier_init(&barrier, NULL, numThreads) != 0) {
        fprintf(stderr, "Error initializing barrier\n");
        exit(1);
    }

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
    float **cache = (float **)malloc(gridDim.x * gridDim.y * gridDim.z * sizeof(float *));
    if (cache == NULL) {
        fprintf(stderr, "Error allocating memory for cache\n");
        exit(1);
    }

    int tid = 0;
    for (int blockIdx_z = 0; blockIdx_z < gridDim.z; ++blockIdx_z) {
        for (int blockIdx_y = 0; blockIdx_y < gridDim.y; ++blockIdx_y) {
            for (int blockIdx_x = 0; blockIdx_x < gridDim.x; ++blockIdx_x) {
                int blockId = blockIdx_x + blockIdx_y * gridDim.x + blockIdx_z * gridDim.x * gridDim.y;
                cache[blockId] = (float *)malloc(numThreads * sizeof(float));
                if (cache[blockId] == NULL) {
                    fprintf(stderr, "Error allocating memory for cache[%d]\n", blockId);
                    exit(1);
                }
                for (int threadIdx_z = 0; threadIdx_z < blockDim.z; ++threadIdx_z) {
                    for (int threadIdx_y = 0; threadIdx_y < blockDim.y; ++threadIdx_y) {
                        for (int threadIdx_x = 0; threadIdx_x < blockDim.x; ++threadIdx_x) {
                            threadData[tid] = (ThreadData){a, b, c, N, gridDim, blockDim, {blockIdx_x, blockIdx_y, blockIdx_z},
                                                           {threadIdx_x, threadIdx_y, threadIdx_z}, cache[blockId]};
                            if (pthread_create(&threads[tid], NULL, thread_func, &threadData[tid]) != 0) {
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

    for (int blockIdx_z = 0; blockIdx_z < gridDim.z; ++blockIdx_z) {
        for (int blockIdx_y = 0; blockIdx_y < gridDim.y; ++blockIdx_y) {
            for (int blockIdx_x = 0; blockIdx_x < gridDim.x; ++blockIdx_x) {
                int blockId = blockIdx_x + blockIdx_y * gridDim.x + blockIdx_z * gridDim.x * gridDim.y;
                free(cache[blockId]);
            }
        }
    }

    free(threads);
    free(threadData);
    free(cache);

    pthread_barrier_destroy(&barrier);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    float *a, *b, *partial_c;
    int N = atoi(argv[1]);

    int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid * sizeof(float));
    if (a == NULL || b == NULL || partial_c == NULL) {
        printf("Memory allocation for vectors failed\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    dim3 gridDim = {blocksPerGrid, 1, 1};
    dim3 blockDim = {threadsPerBlock, 1, 1};

    dot(a, b, partial_c, gridDim, blockDim, N);

    float c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    printf("FINAL RESULT: %f\n", c);

    free(a);
    free(b);
    free(partial_c);

    return 0;
}
