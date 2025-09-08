#include "filter.h"

[[nodiscard]]
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

Filter *filter_vnew(
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

void filter_vdestroy(size_t length, Filter vf[static length]) {
  for (size_t i = 0; i < length; i++) 
    filter_deinit(&vf[i]);
  
  free(vf);
}

