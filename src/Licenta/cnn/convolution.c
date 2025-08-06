#include "convolution.h"
#include "common.h"

static void activate_tensor(Tensor3D tensor, ACT_FUNC activation) {
  switch (activation) {
    case ACT_RELU:
      for (size_t k = 0; k < tensor.depth; k++) {
        for (size_t i = 0; i < tensor.size; i++) {
          for (size_t j = 0; j < tensor.size; j++) {
            float val = MAT_AT(tensor.maps[k], i, j);
            MAT_AT(tensor.maps[k], i, j) = reluf(val);
          }
        }
      }
      break;
    case ACT_SIG:
      for (size_t k = 0; k < tensor.depth; k++) {
        for (size_t i = 0; i < tensor.size; i++) {
          for (size_t j = 0; j < tensor.size; j++) {
            float val = MAT_AT(tensor.maps[k], i, j);
            MAT_AT(tensor.maps[k], i, j) = sigmoidf(val);
          }
        }
      }
      break;
    default:
      break;
  }
}

static inline Mat *convolve(Tensor3D input, Filter filter) {
  if (input.depth != filter.shape.depth)
    return NULL;

  // the output size is calculated as (N - F) / stride + 1
  // where (N - F) / stride must be whole
  if ((input.size - filter.shape.size) % filter.stride != 0)
    return NULL;

  size_t f_size = filter.shape.size;
  
  size_t res_rows = (input.size - filter.shape.size) / filter.stride + 1;
  size_t res_cols = (input.size - filter.shape.size) / filter.stride + 1;

  Mat *result = mat_create(res_rows, res_cols, FLOAT32);
  if (!result)
    return NULL;

  // go over rows x cols, and inside over depth
  for (size_t i = 0; i < res_rows; i+= filter.stride) {
    for (size_t j = 0; j < res_cols; j+= filter.stride) {
      for (size_t k = 0; k < filter.shape.depth; k++) {
        Mat *slice = mat_slice(input.maps[k], 
            i * filter.stride,
            j * filter.stride,
            f_size,
            f_size);

        if (!slice) 
          return NULL; // we panic
        
        float res;
        if (mat_dot(slice, filter.shape.maps[k], (void*) &res)) {
          mat_destroy(result);
          return NULL; // we panic
        }
        MAT_AT(result, i, j) += res;
      }
    }
  }
  return result;
}

Tensor3D get_activation_maps(Convolution_Layer *layer) {
  Tensor3D activation_maps = init_tensor(); //WARN:  
  activation_maps.depth = layer->n_filters;
  for (size_t i = 0; i < layer->n_filters; i++) {
    activation_maps.maps[i] = convolve(layer->input, layer->filters[i]);
  }
  
  activate_tensor(activation_maps, layer->activation);
  return activation_maps;
}

