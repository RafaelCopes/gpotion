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

  float *cache = data->cache;
  float *ref4 = data->c;
  float *a = data->a;
  float *b = data->b;
  int n = data->N;

  dim3 threadIdx = data->threadIdx;
  dim3 blockIdx = data->blockIdx;
  dim3 blockDim = data->blockDim;
  dim3 gridDim = data->gridDim;

  int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
  int cacheIndex = threadIdx.x;
  float temp = 0.0;
  while ((tid < n)) {
    temp = ((a[tid] * b[tid]) + temp);
    tid = ((blockDim.x * gridDim.x) + tid);
  }
  cache[cacheIndex] = temp;

  pthread_barrier_wait(&barrier);

  int i = (blockDim.x / 2);
  while ((i != 0)) {
    if ((cacheIndex < i)) {
      cache[cacheIndex] = (cache[(cacheIndex + i)] + cache[cacheIndex]);
    }

    pthread_barrier_wait(&barrier);

    i = (i / 2);
  }
  if ((cacheIndex == 0)) {
    ref4[blockIdx.x] = cache[0];
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


    int tid = 0;
    int bid = 0;

    for (int blockIdx_z = 0; blockIdx_z < gridDim.z; ++blockIdx_z) {
        for (int blockIdx_y = 0; blockIdx_y < gridDim.y; ++blockIdx_y) {
            for (int blockIdx_x = 0; blockIdx_x < gridDim.x; ++blockIdx_x) {
                float cache[gridDim.x * gridDim.y * gridDim.z][256];

                for (int threadIdx_z = 0; threadIdx_z < blockDim.z; ++threadIdx_z) {
                    for (int threadIdx_y = 0; threadIdx_y < blockDim.y; ++threadIdx_y) {
                        for (int threadIdx_x = 0; threadIdx_x < blockDim.x; ++threadIdx_x) {
                            threadData[tid] = (ThreadData){a, b, c, N, gridDim, blockDim, {blockIdx_x, blockIdx_y, blockIdx_z},
                                                           {threadIdx_x, threadIdx_y, threadIdx_z}, cache[bid]};
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

                bid++;
            }
        }
    }

    free(threads);
    free(threadData);

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
