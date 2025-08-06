#include "../math/tensor3D.h"
#include "common.h"

#ifndef CNN_CONVOLUTION_H
#define CNN_CONVOLUTION_H

typedef struct {
  size_t stride; // how much to skip, usually 1-2 
  Tensor3D shape;
} Filter;

typedef struct {
  size_t n_filters;
  Filter *filters; // array of filters
  ACT_FUNC activation;
} Convolution_Layer;

Convolution_Layer *CL_init(size_t filter_size, size_t nfilters);
void CL_deinit(Convolution_Layer *cl);

// TODO: add a bias to the whole operation?
Tensor3D get_activation_maps(Convolution_Layer *layer, Tensor3D input); 

#endif // CNN_CONVOLUTION_H
