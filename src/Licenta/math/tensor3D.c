#include "tensor3D.h"
#include "utils.h"
#include "../utils/arena.h"
#include <stdlib.h>

Tensor3D *tensor_init(Tensor3D *tensor, size_t size, size_t depth) {
  if (!tensor)
    return nullptr;

  *tensor = (Tensor3D) {
    .size = size, 
    .depth = depth,
    .maps = malloc(sizeof(Mat*) * depth)
  };

  if (!tensor->maps) 
    goto fail;
  
  for (size_t i = 0; i < depth; i++) {
    tensor->maps[i] = mat_create(size, size);
    if (!tensor->maps[i])
      goto fail;
  }
  return tensor;

fail:
  if (tensor->maps) 
    for (size_t i = 0; i < depth; i++) {
      if (!tensor->maps[i])
        break;

      free(tensor->maps[i]);
      tensor->maps[i] = nullptr;
    }
   free(tensor->maps); 
  return nullptr;
}

void tensor_deinit(Tensor3D *tensor) {
  if (!tensor || !tensor->maps)
    return;

  for (size_t i = 0; i < tensor->depth; i++) 
    if (tensor->maps[i]) {
      free(tensor->maps[i]);
      tensor->maps[i] = nullptr;
  }

  free(tensor->maps);
  tensor->maps = nullptr;
  *tensor = (Tensor3D) {0};
}

[[nodiscard("pointer to tensor allocated data dropped")]]
__attribute__((malloc()))
inline Tensor3D *tensor_new(size_t size, size_t depth) {
  return tensor_init(malloc(sizeof(Tensor3D)), size, depth);
}

inline void tensor_destroy(Tensor3D *t) {
  if (t) {
    tensor_deinit(t);
    free(t);
  }
}
