#include "convolution.h"
#include "common.h"
#include "../utils/arena.h"

Filter *filter_init(
    size_t stride,
    size_t depth,
    size_t dims
) {
  Filter *filter = MALLOC(sizeof(Filter));
  if (!filter)
    return NULL;
  filter->stride = stride;
  filter->shape = tensor_init(dims, depth);
  if (!filter->shape) {
    FREE(filter);
    return NULL;
  }
  return filter;
}

void filter_deinit(Filter *filter) {
  if (filter->shape)
    tensor_deinit(filter->shape);

  FREE(filter);
}

Convolution_Layer *conv_init(
    size_t n_filters, 
    size_t f_size,
    size_t f_depth,
    size_t stride,
    ACT_FUNC act_func
){
  Convolution_Layer *cl = MALLOC(sizeof(Convolution_Layer));
  if (!cl)
    return NULL;

  cl->n_filters = n_filters;
  cl->activation = act_func;
  cl->filters = MALLOC(sizeof(Filter*) * n_filters);
  if (!cl->filters) {
    FREE(cl);
    return NULL;
  }

  for (size_t i = 0; i < n_filters; i++) {
    cl->filters[i] = filter_init(stride, f_depth, f_size);
    if (!cl->filters[i])
      goto fail;
  }

  return cl;

fail:
  for (size_t i = 0; i < n_filters; i++)
    if (cl->filters[i])
      FREE(cl->filters[i]);

  FREE(cl->filters);
  FREE(cl);
  return NULL;
}

void conv_deinit(Convolution_Layer *cl) {
  for (size_t i = 0; i < cl->n_filters; i++) {
    FREE(cl->filters[i]);
    cl->filters[i] = NULL;
  }

  FREE(cl->filters);
  cl->filters = NULL;

  FREE(cl);
}

static inline void activate_tensor(Tensor3D *tensor, ACT_FUNC activation) {
  switch (activation) {
    case ACT_RELU:
      for (size_t k = 0; k < tensor->depth; k++) {
        for (size_t i = 0; i < tensor->size; i++) {
          for (size_t j = 0; j < tensor->size; j++) {
            float val = MAT_AT(tensor->maps[k], i, j);
            MAT_AT(tensor->maps[k], i, j) = relu(val);
          }
        }
      }
      break;
    case ACT_SIG:
      for (size_t k = 0; k < tensor->depth; k++) {
        for (size_t i = 0; i < tensor->size; i++) {
          for (size_t j = 0; j < tensor->size; j++) {
            float val = MAT_AT(tensor->maps[k], i, j);
            MAT_AT(tensor->maps[k], i, j) = sigmoid(val);
          }
        }
      }
      break;
    default:
      break;
  }
}

static inline Mat *convolve(Tensor3D *input, Filter *filter) {
  if (input->depth != filter->shape->depth)
    return NULL;

  // the output size is calculated as (N - F) / stride + 1
  // where (N - F) / stride must be whole
  if ((input->size - filter->shape->size) % filter->stride != 0)
    return NULL;

  size_t f_size = filter->shape->size;
  
  size_t res_rows = (input->size - filter->shape->size) / filter->stride + 1;
  size_t res_cols = (input->size - filter->shape->size) / filter->stride + 1;

  Mat *result = mat_create(res_rows, res_cols);
  if (!result)
    return NULL;

  // go over rows x cols, and inside over depth
  for (size_t i = 0; i < res_rows; i += filter->stride) {
    for (size_t j = 0; j < res_cols; j += filter->stride) {
      for (size_t k = 0; k < filter->shape->depth; k++) {
        Mat *slice = mat_slice(input->maps[k], 
            i * filter->stride,
            j * filter->stride,
            f_size,
            f_size);

        if (!slice) 
          return NULL; // we panic
        
        float res;
        if (mat_dot(slice, filter->shape->maps[k], (void*) &res)) {
          mat_destroy(result);
          return NULL; // we panic
        }
        MAT_AT(result, i, j) += res;
      }
    }
  }
  return result;
}
