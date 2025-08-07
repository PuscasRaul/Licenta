#include "tensor3D.h"
#include "utils.h"
#include "../utils/arena.h"

Tensor3D *tensor_init(size_t size, size_t depth) {
  Tensor3D *tensor = MALLOC(sizeof(Tensor3D));
  if (!tensor)
    return NULL;

  tensor->size = size;
  tensor->depth = depth;

  tensor->maps = MALLOC(sizeof(Mat*) * depth);
  if (!tensor->maps) 
    goto fail;
  
  for (size_t i = 0; i < depth; i++) {
    tensor->maps[i] = mat_create(size, size);
    if (!tensor->maps[i])
      goto fail;
  }

  return tensor;

fail:
  if (tensor->maps) {
    for (size_t i = 0; i < depth; i++) 
      if (tensor->maps[i]) {
        FREE(tensor->maps[i]);
        tensor->maps[i] = NULL;
      }
   FREE(tensor->maps); 
  }
  FREE(tensor);
  return NULL;
}

void tensor_deinit(Tensor3D *tensor) {
  if (!tensor)
    return;

  if (!tensor->maps) {
    FREE(tensor);
    return;
  }

  for (size_t i = 0; i < tensor->depth; i++) 
    if (tensor->maps[i]) {
      FREE(tensor->maps[i]);
      tensor->maps[i] = NULL;
    }

  FREE(tensor->maps);
  FREE(tensor);
}
