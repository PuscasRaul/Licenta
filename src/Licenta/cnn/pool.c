#include "pool.h"

Pool_Layer *pool_init(
    Pool_Layer *pl,
    size_t filter_size,
    size_t f_stride,
    POOLING_TYPES type
    ) {
  if (!pl)
    return nullptr;

  *pl = (Pool_Layer) {
    .f_stride = f_stride, 
    .filter_size = filter_size,
    .type = type
  };
   
  return pl;
}

void pool_deinit(Pool_Layer *pl) {
  if (pl) 
    *pl = (Pool_Layer) {};
}

Tensor3D *downsample(Tensor3D *input, Pool_Layer *pl) {

}

/*
Tensor3D *downsample(Tensor3D *input, Pool_Layer *layer) {
  assert((input->size - input->size) % layer->stride != 0);

  size_t stride = layer->stride;
  size_t f_size = layer->filter_size;
  size_t dims = (input->size - f_size) / stride + 1;
  Tensor3D *result = tensor_init(dims, input->depth); //WARN:

  switch (layer->type) {
    case AVG_POOL: 
      for (size_t k = 0; k < input->depth; k++) {
        for (size_t i = 0; i < dims; i+=stride) {
          for (size_t j = 0; j < dims; j+=stride) {
            float avg = 0.0f;
            for (size_t di = 0; di < f_size; di++) {
              for (size_t dj = 0; dj < f_size; dj++) {
                avg += MAT_AT(input->maps[k], i + di, j + dj);
              }
            }
          MAT_AT(result->maps[k], i/stride, j/stride) = avg / (f_size * f_size);
          }
        }
      }
      break;
    case MAX_POOL:
      for (size_t k = 0; k < input->depth; k++) {
        for (size_t i = 0; i < dims; i+=stride) {
          for (size_t j = 0; j < dims; j+=stride) {
            float max = -INFINITY;
            for (size_t di = 0; di < f_size; di++) {
              for (size_t dj = 0; dj < f_size; dj++) {
                if (max < MAT_AT(input->maps[k], i + di, j + dj))
                  max = MAT_AT(input->maps[k], i + di, j + dj);
              }
            }
          MAT_AT(result->maps[k], i/stride, j/stride) = max;
          }
        }
      }
      break;
    default:
      break;
      }
  return result;
}
*/
