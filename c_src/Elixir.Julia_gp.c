#include "erl_nif.h"

struct dim3 {
  int x;
  int y;
  int z;
};

void julia_kernel(float *ptr, int dim, struct dim3 gridDim,
                  struct dim3 blockDim) {
  struct dim3 blockIdx;
  struct dim3 threadIdx;

  for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) {

    for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) {

      for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {

        for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {

          for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {

            for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {

              int x = blockIdx.x;
              int y = blockIdx.y;
              int offset = (x + (y * dim));
              int juliaValue = 1;
              float scale = 0.1;
              float jx = ((scale * (dim - x)) / dim);
              float jy = ((scale * (dim - y)) / dim);
              float cr = (-0.8);
              float ci = 0.156;
              float ar = jx;
              float ai = jy;
              for (int i = 0; i < 200; i++) {
                float nar = (((ar * ar) - (ai * ai)) + cr);
                float nai = (((ai * ar) + (ar * ai)) + ci);
                if ((((nar * nar) + (nai * nai)) > 1000)) {
                  juliaValue = 0;
                  break;
                }

                ar = nar;
                ai = nai;
              }

              ptr[((offset * 4) + 0)] = (255 * juliaValue);
              ptr[((offset * 4) + 1)] = 0;
              ptr[((offset * 4) + 2)] = 0;
              ptr[((offset * 4) + 3)] = 255;
            }
          }
        }
      }
    }
  }
}

void julia_kernel_call(ErlNifEnv *env, const ERL_NIF_TERM argv[],
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
  int arg2;
  enif_get_int(env, head, &arg2);
  list = tail;

  julia_kernel(arg1, arg2, gridDim, blockDim);
}
