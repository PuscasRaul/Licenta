#include "../math/tensor3D.h"

#ifndef CNN_POOL_H
#define CNN_POOL_H

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


// Pooling layer operations
Tensor3D downsample(Pool_Layer *layer);

#endif // CNN_POOL_H
