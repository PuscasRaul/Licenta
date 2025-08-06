#include "pool.h"

Tensor3D downsample(Pool_Layer *layer) {
  Tensor3D result = init_tensor(); //WARN:
  Tensor3D input = layer->input;

  assert((input.size - input.size) % layer->stride != 0);

  size_t stride = layer->stride;
  size_t f_size = layer->filter_size;
  size_t dims = (layer->input.size - f_size) / stride + 1;

  switch (layer->type) {
    case AVG_POOL: 
      for (size_t k = 0; k < input.depth; k++) {
        for (size_t i = 0; i < dims; i+=stride) {
          for (size_t j = 0; j < dims; j+=stride) {
            float avg = 0.0f;
            for (size_t di = 0; di < f_size; di++) {
              for (size_t dj = 0; dj < f_size; dj++) {
                avg += MAT_AT(input.maps[k], i + di, j + dj);
              }
            }
          MAT_AT(result.maps[k], i/stride, j/stride) = avg / (f_size * f_size);
          }
        }
      }
      break;
    case MAX_POOL:
      for (size_t k = 0; k < input.depth; k++) {
        for (size_t i = 0; i < dims; i+=stride) {
          for (size_t j = 0; j < dims; j+=stride) {
            float max = -INFINITY;
            for (size_t di = 0; di < f_size; di++) {
              for (size_t dj = 0; dj < f_size; dj++) {
                if (max < MAT_AT(input.maps[k], i + di, j + dj))
                  max = MAT_AT(input.maps[k], i + di, j + dj);
              }
            }
          MAT_AT(result.maps[k], i/stride, j/stride) = max;
          }
        }
      }
      break;
    default:
      break;
      }
  return result;
}
