#include "erl_nif.h"

struct dim3 {
  int x;
  int y;
  int z;
};

void gpu_nBodies(float *p, float dt, int n, float softening,
                 struct dim3 gridDim, struct dim3 blockDim) {
  struct dim3 blockIdx;
  struct dim3 threadIdx;

  for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) {

    for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) {

      for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {

        for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {

          for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {

            for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {

              int i = ((blockDim.x * blockIdx.x) + threadIdx.x);
              if ((i < n)) {
                float fx = 0.0;
                float fy = 0.0;
                float fz = 0.0;
                for (int j = 0; j < n; j++) {
                  float dx = (p[(6 * j)] - p[(6 * i)]);
                  float dy = (p[((6 * j) + 1)] - p[((6 * i) + 1)]);
                  float dz = (p[((6 * j) + 2)] - p[((6 * i) + 2)]);
                  float distSqr =
                      ((((dx * dx) + (dy * dy)) + (dz * dz)) + softening);
                  float invDist = (1.0 / sqrt(distSqr));
                  float invDist3 = ((invDist * invDist) * invDist);
                  fx = (fx + (dx * invDist3));
                  fy = (fy + (dy * invDist3));
                  fz = (fz + (dz * invDist3));
                }

                p[((6 * i) + 3)] = (p[((6 * i) + 3)] + (dt * fx));
                p[((6 * i) + 4)] = (p[((6 * i) + 4)] + (dt * fy));
                p[((6 * i) + 5)] = (p[((6 * i) + 5)] + (dt * fz));
              }
            }
          }
        }
      }
    }
  }
}

void gpu_nBodies_call(ErlNifEnv *env, const ERL_NIF_TERM argv[],
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
  double darg2;
  float arg2;
  enif_get_double(env, head, &darg2);
  arg2 = (float)darg2;
  list = tail;

  enif_get_list_cell(env, list, &head, &tail);
  int arg3;
  enif_get_int(env, head, &arg3);
  list = tail;

  enif_get_list_cell(env, list, &head, &tail);
  double darg4;
  float arg4;
  enif_get_double(env, head, &darg4);
  arg4 = (float)darg4;
  list = tail;

  gpu_nBodies(arg1, arg2, arg3, arg4, gridDim, blockDim);
}
