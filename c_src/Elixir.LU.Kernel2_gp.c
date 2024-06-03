#include "erl_nif.h"
#include <pthread.h>

typedef struct dim3 {
  int x;
  int y;
  int z;
} Dim3;

typedef struct {
  float *a;
  int n;
  int k;
  Dim3 threadIdx;
  Dim3 blockIdx;
  Dim3 blockDim;
  Dim3 gridDim;
} ThreadData;

void *thread_kernel(void *arg) {
  ThreadData *data = (ThreadData *)arg;

  float *a = data->a;
  int n = data->n;
  int k = data->k;

  Dim3 threadIdx = data->threadIdx;
  Dim3 blockIdx = data->blockIdx;
  Dim3 blockDim = data->blockDim;
  Dim3 gridDim = data->gridDim;

  int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
  int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
  if (((((i > k) && (j > k)) && (i < n)) && (j < n))) {
    a[((i * n) + j)] =
        (a[((i * n) + j)] - (a[((i * n) + k)] * a[((k * n) + j)]));
  }

  return NULL;
}

void kernel2(float *a, int n, int k, Dim3 gridDim, Dim3 blockDim) {
  int numThreads = blockDim.x * blockDim.y;

  int totalThreads = gridDim.x * gridDim.y * numThreads;

  pthread_t *threads = (pthread_t *)malloc(totalThreads * sizeof(pthread_t));
  if (threads == NULL) {
    fprintf(stderr, "Error allocating memory for threads\n");
    exit(1);
  }

  ThreadData *threadData =
      (ThreadData *)malloc(totalThreads * sizeof(ThreadData));
  if (threadData == NULL) {
    fprintf(stderr, "Error allocating memory for thread data\n");
    exit(1);
  }

  Dim3 blockIdx;
  Dim3 threadIdx;

  int tid = 0;

  for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) {
    for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {

      for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
        for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {

          threadData[tid] = (ThreadData){

              a, n, k, threadIdx, blockIdx, blockDim, gridDim};

          if (pthread_create(&threads[tid], NULL, thread_kernel,
                             &threadData[tid]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", tid);
            exit(1);
          }

          tid++;
        }
      }

      for (int i = tid - numThreads; i < tid; ++i) {
        pthread_join(threads[i], NULL);
      }
    }
  }

  free(threads);
  free(threadData);
}

void kernel2_call(ErlNifEnv *env, const ERL_NIF_TERM argv[],
                  ErlNifResourceType *type) {

  ERL_NIF_TERM list;
  ERL_NIF_TERM head;
  ERL_NIF_TERM tail;
  float **array_res;

  const ERL_NIF_TERM *tuple_blocks;
  const ERL_NIF_TERM *tuple_threads;
  int arity;

  if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
    printf("spawn: blocks argument is not a tuple");
  }

  if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
    printf("spawn:threads argument is not a tuple");
  }
  int b1, b2, b3, t1, t2, t3;

  enif_get_int(env, tuple_blocks[0], &b1);
  enif_get_int(env, tuple_blocks[1], &b2);
  enif_get_int(env, tuple_blocks[2], &b3);
  enif_get_int(env, tuple_threads[0], &t1);
  enif_get_int(env, tuple_threads[1], &t2);
  enif_get_int(env, tuple_threads[2], &t3);

  list = argv[3];

  Dim3 gridDim;
  Dim3 blockDim;

  gridDim.x = b1;
  gridDim.y = b2;
  gridDim.z = b3;
  blockDim.x = t1;
  blockDim.y = t2;
  blockDim.z = t3;

  enif_get_list_cell(env, list, &head, &tail);
  enif_get_resource(env, head, type, (void **)&array_res);
  float *arg1 = *array_res;
  list = tail;

  enif_get_list_cell(env, list, &head, &tail);
  int arg2;
  enif_get_int(env, head, &arg2);
  list = tail;

  enif_get_list_cell(env, list, &head, &tail);
  int arg3;
  enif_get_int(env, head, &arg3);
  list = tail;

  kernel2(arg1, arg2, arg3, gridDim, blockDim);
}
