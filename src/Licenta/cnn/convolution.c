#include "convolution.h"
#include "../utils/arena.h"

[[nodiscard]]
Convolution_Layer *conv_init(
    Convolution_Layer *cl,
    size_t n_filters, 
    size_t f_size,
    size_t f_depth,
    size_t stride,
    ACT_FUNC act_func
){

  if (!cl)
    return nullptr;

  *cl = (Convolution_Layer) {
    .n_filters = n_filters,
    .activation = act_func,
    .filters = filter_vnew(n_filters, f_size, stride, f_depth)
  };

  if (!cl->filters)
    return nullptr;

  return cl;
}

void conv_deinit(Convolution_Layer *cl) {
  if (!cl)
    return;

  free(cl->filters);
  *cl = (Convolution_Layer) {};
}

[[nodiscard]]
static Mat *convolve(Tensor3D *input, Filter *filter) {
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
