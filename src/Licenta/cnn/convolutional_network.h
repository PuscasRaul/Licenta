#include <stddef.h>
#include <stdint.h>

#define MAT_IMPLEMENTATION
#include "../math/Mat.h"

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
  MAX_POOL,
  AVG_POOL
} POOLING_TYPES;

typedef struct {
  Tensor3D input;
  POOLING_TYPES type;
  size_t filter_size;
  size_t stride;
} Pool_Layer;

union Layer {
  Convolution_Layer conv_layer;
  Pool_Layer pool_layer;
};

// TODO: Implement and move to math module
// WARN: Mock function, used for defining the prototypes of CNN functions
Tensor3D init_tensor();

// Convolution layer operations
// TODO: add a bias to the whole operation?
Mat *convolve(Tensor3D input, Filter filter);
Tensor3D get_activation_maps(Convolution_Layer *layer); 

// Pooling layer operations
Tensor3D downsample(Pool_Layer *layer);

