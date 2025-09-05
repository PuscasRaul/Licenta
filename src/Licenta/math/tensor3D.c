#include "tensor3D.h"
#include "utils.h"
#include "../utils/arena.h"
#include "Mat.h"

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
    .maps = mat_vcreate(depth, size)
  };

  if (!tensor->maps) 
    return nullptr;
  
  return tensor;
}

void tensor_deinit(Tensor3D tensor[static 1]) {
  for (size_t i = 0; i < tensor->depth; i++) 
     mat_deinit(&tensor->maps[i]);

  free(tensor->maps);
  *tensor = (Tensor3D) {0};
}

[[nodiscard]]
Tensor3D *tensor_new(size_t size, size_t depth) {
  return tensor_init(malloc(sizeof(Tensor3D)), size, depth);
}

void tensor_destroy(Tensor3D *t) {
  if (t) {
    tensor_deinit(t);
    free(t);
  }
}

void activate_tensor(Tensor3D *tensor, ACT_FUNC activation) {
  switch (activation) {
    case ACT_RELU:
      for (size_t k = 0; k < tensor->depth; k++) {
        for (size_t i = 0; i < tensor->size; i++) {
          for (size_t j = 0; j < tensor->size; j++) {
            float val = *(mat_at(&tensor->maps[k], i, j));
            *(mat_at(&tensor->maps[k], i, j)) = relu(val);
          }
        }
      }
      break;
    case ACT_SIG:
      for (size_t k = 0; k < tensor->depth; k++) {
        for (size_t i = 0; i < tensor->size; i++) {
          for (size_t j = 0; j < tensor->size; j++) {
            float val = *(mat_at(&tensor->maps[k], i, j));
            *(mat_at(&tensor->maps[k], i, j)) = sigmoid(val);
          }
        }
      }
      break;
    default:
      break;
  }
}
