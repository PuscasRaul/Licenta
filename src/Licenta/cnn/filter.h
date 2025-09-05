#ifndef FILTER_H
#define FILTER_H

#include "../math/tensor3D.h"

typedef struct {
  size_t stride; // how much to skip, usually 1-2 
  Tensor3D *shape; // the actual shape of the filter
} Filter; 

Filter *filter_init(Filter *f, size_t stride, size_t depth, size_t dims);
void filter_deinit(Filter *f);

[[nodiscard]]
static inline Filter *filter_new(size_t stride, size_t depth, size_t dims) {
  return filter_init(malloc(sizeof(Filter)), stride, depth, dims);
}

static inline void filter_destroy(Filter *f) {
  if (f) {
    filter_deinit(f);
    free(f);
  }
}

[[nodiscard]]
static inline Filter *filter_vnew(
    size_t length,
    size_t size,
    size_t stride,
    size_t depth
) {
  if (!length)
    return nullptr;

  Filter *f = malloc(sizeof(Filter) * length);
  if (!f)
    return nullptr;

  for (size_t i = 0; i < length; i++)
    if (!filter_init(&f[i], stride, depth, size))
      goto fail;

  return f;

fail:
  for (size_t i = 0; i < length; i++) { 
    if (!f[i].stride)
      break;
    filter_deinit(&f[i]);
  }
  return nullptr;
}

Filter *filter_init(
    Filter *f,
    size_t stride,
    size_t depth,
    size_t dims
    ) {
  if (!f)
    return nullptr;

  *f = (Filter) {
    .stride = stride,
      .shape = tensor_new(dims, depth)
  };

  if (!f->shape)
    return nullptr;

  return f;
}

void filter_deinit(Filter *filter) {
  if (filter) {
    if (filter->shape)
      tensor_destroy(filter->shape);
    filter->shape = nullptr;
    *filter = (Filter) {0};
  }
}

#endif
