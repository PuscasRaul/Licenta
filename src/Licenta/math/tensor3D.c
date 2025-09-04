#include "tensor3D.h"
#include "utils.h"
#include "../utils/arena.h"
#include "Mat.h"

[[deprecated("Implementation")]]
Tensor3D *tensor_init(
    Tensor3D tensor[static 1],
    size_t size,
    size_t depth
){

  if (!size || !depth)
    return nullptr;

  *tensor = (Tensor3D) {
    .size = size, 
    .depth = depth,
    .maps = mat_create_array(depth)
  };

  if (!tensor->maps) 
    return nullptr;
  
  for (size_t i = 0; i < depth; i++) 
    if (!mat_init(&tensor->maps[i], size, size)) 
      goto fail;
  
  return tensor;

fail:
  for (size_t i = 0; i < depth; i++) {
    if (!tensor->maps[i].rows)
      break;
    mat_deinit(&tensor->maps[i]);
  }
  return nullptr; 
}

[[deprecated("Implementation")]]
void tensor_deinit(Tensor3D tensor[static 1]) {
  for (size_t i = 0; i < tensor->depth; i++) 
     mat_deinit(&tensor->maps[i]);

  free(tensor->maps);
  *tensor = (Tensor3D) {0};
}

