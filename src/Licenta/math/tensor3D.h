#include <stdint.h>
#include "Mat.h"

#ifndef TENSOR3D_H
#define TENSOR3D_H

typedef struct {
  size_t size; // matrix size
  uint8_t depth; // number of matrixes
  Mat **maps; // data
} Tensor3D;

/* Initialize a tensor buffer t with depth matrixes of size X size
 * Only use this function on an unitialized tensor
 * Each tensor initialized with this must be destroyed with
 * tensor_deinit
 */
Tensor3D *tensor_init(Tensor3D *t, size_t size, size_t depth);

/* Destroy Tensor3D buffer t
 * t must have been initialized with tensor_init
 */
void tensor_deinit(Tensor3D *t);

/* Create tensor buffer with depth matrixes of size X size
 * Each tensor initialized with this must be destroyed with
 * tensor_destroy
 */
Tensor3D *tensor_new(size_t size, size_t depth);

/* Destroy Tensor3D buffer t
 * t must have been created with tensor_new
 */
void tensor_destroy(Tensor3D *t);

#endif // TENSOR3D_H
