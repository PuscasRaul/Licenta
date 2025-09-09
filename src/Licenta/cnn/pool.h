#include "../math/tensor3D.h"

#ifndef CNN_POOL_H
#define CNN_POOL_H

typedef enum {
  MAX_POOL,
  AVG_POOL
} POOLING_TYPES;

typedef struct {
  POOLING_TYPES type;
  size_t filter_size;
  size_t f_stride;
} Pool_Layer;


// Pooling layer operations
[[nodiscard]]
Pool_Layer *pool_init(
    Pool_Layer *pl,
    size_t filter_size,
    size_t f_stride,
    POOLING_TYPES type
);
void pool_deinit(Pool_Layer *pl);

[[nodiscard]]
static inline Pool_Layer *pool_new(
    size_t filter_size,
    size_t f_stride,
    POOLING_TYPES type
) {
  return pool_init(malloc(sizeof(Pool_Layer)), filter_size, f_stride, type);
}
static inline void pool_destroy(Pool_Layer *pl) {
  if (pl) {
    pool_deinit(pl);
    free(pl);
  }
}

Tensor3D *downsample(Tensor3D *input, Pool_Layer *layer);
#endif // CNN_POOL_H
