// TODO: IMPORTANT, CREATE FILTER VDESTROY
#ifndef FILTER_H
#define FILTER_H

#include "../math/tensor3D.h"

typedef struct {
  size_t stride; // how much to skip, usually 1-2 
  Tensor3D *shape; // the actual shape of the filter
} Filter; 

[[nodiscard]]
Filter *filter_init(Filter *f, size_t stride, size_t depth, size_t dims); 
void filter_deinit(Filter *f);

[[nodiscard]]
Filter *filter_vnew(size_t length, size_t size, size_t stride, size_t depth); 
void filter_vdestroy(size_t length, Filter vf[static length]);

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

#endif
