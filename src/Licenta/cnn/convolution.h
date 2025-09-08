#include "filter.h"

#ifndef CNN_CONVOLUTION_H
#define CNN_CONVOLUTION_H

typedef struct Convolution_Layer Convolution_Layer;

struct Convolution_Layer {
  size_t n_filters; // number of filters
  Filter *filters; // array of filters
  ACT_FUNC activation; // current layer activation function
};

[[nodiscard]]
Convolution_Layer *conv_init(
    Convolution_Layer *cl,
    size_t n_filters, 
    size_t f_size,
    size_t f_depth,
    size_t stride,
    ACT_FUNC act_func
);
void conv_deinit(Convolution_Layer *cl);

[[nodiscard]]
static inline Convolution_Layer *conv_new(
    size_t n_filters, 
    size_t f_size,
    size_t f_depth,
    size_t stride,
    ACT_FUNC act_func
) {
  return conv_init(
    malloc(sizeof(Convolution_Layer)), 
    n_filters, 
    f_size,
    f_depth,
    stride,
    act_func
  );
}

static inline void conv_destroy(Convolution_Layer *cl) {
  if (cl) {
    conv_deinit(cl);
    free(cl);
  }
}

Tensor3D *get_activation_maps(Convolution_Layer *layer, Tensor3D *input); 

#endif // CNN_CONVOLUTION_H
