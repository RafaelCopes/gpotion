#include "erl_nif.h"

struct dim3 {
  int x;
  int y;
  int z;
};

void fan1(float *m, float *a, int size, int t, struct dim3 gridDim,
          struct dim3 blockDim) {
  struct dim3 blockIdx;
  struct dim3 threadIdx;

  for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {

    for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {

      int idx = (threadIdx.x + (blockIdx.x * blockDim.x));
      if ((idx >= ((size - 1) - t))) {
        continue;
      }

      m[((size * ((idx + t) + 1)) + t)] =
          (a[((size * ((idx + t) + 1)) + t)] / a[((size * t) + t)]);
    }
  }
}

void fan1_call(ErlNifEnv *env, const ERL_NIF_TERM argv[],
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

  struct dim3 gridDim;
  struct dim3 blockDim;

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
  enif_get_resource(env, head, type, (void **)&array_res);
  float *arg2 = *array_res;
  list = tail;

  enif_get_list_cell(env, list, &head, &tail);
  int arg3;
  enif_get_int(env, head, &arg3);
  list = tail;

  enif_get_list_cell(env, list, &head, &tail);
  int arg4;
  enif_get_int(env, head, &arg4);
  list = tail;

  fan1(arg1, arg2, arg3, arg4, gridDim, blockDim);
}
