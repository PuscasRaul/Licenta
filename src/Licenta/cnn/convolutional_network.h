#include <stddef.h>
#include <stdint.h>

#define MAT_IMPLEMENTATION
#include "../math/Mat.h"

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif // NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.1f
#endif // NN_RELU_PARAM

typedef enum {
  ACT_RELU,
  ACT_SIG,
} ACT_FUNC;

typedef struct {
  size_t size; // square tensors 
  uint8_t depth; 
  Mat **maps;  
} Tensor3D;

typedef struct {
  size_t stride; // how much to skip, usually 1-2 
  Tensor3D shape;
} Filter;

typedef struct {
  Tensor3D input;
  size_t n_filters;
  Filter *filters; // array of filters
  ACT_FUNC activation;
} Convolution_Layer;

typedef enum {
  MAX,
  AVG
} POOLING_TYPES;


// TODO: Implement and move to math module
// WARN: Mock function, used for defining the prototypes of CNN functions
Tensor3D init_tensor();

Mat *convolve(Tensor3D input, Filter filter);
Tensor3D get_activation_maps(Convolution_Layer *layer); 

