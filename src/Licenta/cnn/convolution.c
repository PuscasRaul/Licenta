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
static Mat *convolve(Tensor3D *input, Filter *filter, Mat *result) {

  size_t f_size = filter->shape->size;
  size_t res_rows = (input->size - f_size) / filter->stride + 1;
  size_t res_cols = (input->size - f_size) / filter->stride + 1;
  double dot_result = 0;

  Mat *slice = mat_create(f_size, f_size);

  if (!result)
    goto fail;
  if (!slice) 
    goto fail;

  // go over rows x cols, and inside over depth
  for (size_t i = 0; i < res_rows; i += filter->stride) {
    for (size_t j = 0; j < res_cols; j += filter->stride) {
      for (size_t k = 0; k < filter->shape->depth; k++) {
        if (!mat_slice(slice, &input->maps[k],i, j, f_size,f_size))
          goto fail;
;
        if (mat_dot(slice, &filter->shape->maps[k], &dot_result)) 
          goto fail; // just panic
        
        *(mat_at(result, i, j)) += dot_result;
      }
    }
  }
  return result;

fail:
  mat_destroy(result);
  result = nullptr;
  if (slice) {
    mat_destroy(slice);
    slice = nullptr;
  }
  return nullptr;
}
  
Tensor3D *get_activation_maps(Convolution_Layer *layer, Tensor3D *input) {

  size_t f_size = layer->filters[0].shape->size;
  size_t f_stride = layer->filters[0].stride;
  size_t f_depth = layer->filters[0].shape->depth;
  size_t res_rows = (input->size - f_size) / f_stride + 1;

  if (input->depth != f_depth)
    return nullptr; 

  // the output size is calculated as (N - F) / stride + 1
  // where (N - F) / stride must be whole
  if ((input->size - f_size) % f_stride != 0)
    return nullptr;

  Tensor3D *act_map = tensor_new(res_rows, layer->n_filters);
  if (!act_map)
    return nullptr;

  for (size_t i = 0; i < layer->n_filters; i++) 
    if(!convolve(input, &layer->filters[i], &act_map->maps[i]))
      goto fail;

  return act_map;

fail:
  tensor_destroy(act_map);
  return nullptr;
}
