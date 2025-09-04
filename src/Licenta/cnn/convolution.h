#include "../math/tensor3D.h"
#include "common.h"

#ifndef CNN_CONVOLUTION_H
#define CNN_CONVOLUTION_H

typedef struct Filter Filter;

typedef struct {
  size_t n_filters; // number of filters
  Filter *filters; // array of filters
  ACT_FUNC activation; // current layer activation function
} Convolution_Layer;


Convolution_Layer *conv_init(
    size_t n_filters, 
    size_t f_size,
    size_t f_depth,
    size_t stride,
    ACT_FUNC act_func
);
void conv_deinit(Convolution_Layer *cl);

Tensor3D get_activation_maps(Convolution_Layer *layer, Tensor3D input); 

#endif // CNN_CONVOLUTION_H
