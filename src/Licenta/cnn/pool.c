#include "pool.h"
#include <math.h>

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

/* TODO: most probably will be a bottle neck, optimize for better caching later
 */
Tensor3D *downsample(Tensor3D *input, Pool_Layer *pl) {
  size_t stride = pl->f_stride;
  size_t f_size = pl->filter_size;
  size_t dims = (input->size - f_size) / stride + 1;
  size_t depth = input->depth;
  size_t out_row = 0, out_col = 0;
  Tensor3D *result = tensor_new(dims, input->depth);
  if (!result)
    return nullptr;

  switch (pl->type) {
    case (MAX_POOL):
      double max = -1;
      for (size_t k = 0; k < depth; k++) {
        Mat current = input->maps[k];
        out_row = 0;
        for (size_t ih = 0; ih + stride - 1 < input->size; ih += stride) {
          out_col = 0;
          for (size_t jh = 0; jh + stride - 1 < input->size; jh += stride) {
            max = -INFINITY;
            for (size_t il = 0; il < f_size; il++) {
              for (size_t jl = 0; jl < f_size; jl++) {
                double elem = 
                  current.es[(ih + il) * current.stride + (jl + jh)];
                if (elem > max)
                  max = elem;
              }
            }
            result->maps[k].es[out_row * result->maps[k].stride + out_col] = max;
          }
          ++out_col;
        }
        ++out_row;
      }
      return result;
      break;
    case (AVG_POOL):
      double avg = 0.0f;
      for (size_t k = 0; k < depth; k++) {
        Mat current = input->maps[k];
        out_row = 0;
        for (size_t ih = 0; ih + stride - 1 < input->size; ih += stride) {
          out_col = 0;
          for (size_t jh = 0; jh + stride - 1 < input->size; jh += stride) {
            avg = 0.0f;
            for (size_t il = 0; il < f_size; il++) 
              for (size_t jl = 0; jl < f_size; jl++) 
                avg += current.es[(ih + il) * current.stride + (jl + jh)];
            result->maps[k].es[out_row * result->maps[k].stride + out_col] = 
              avg / (f_size * f_size); // (stride * stride) because of square fiters, so it's stride squared elements
          }
          ++out_col;
        }
        ++out_row;
      }
      return result;
      break;
    default:
      unreachable();
  }
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
