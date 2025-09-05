#include <stdint.h>
#include <stdlib.h>
#include "Mat.h"
#include "activation.h"

#ifndef MATH_TENSOR3D_H
#define MATH_TENSOR3D_H

typedef struct Tensor3D Tensor3D;

struct Tensor3D{
  size_t size; // matrix size
  uint8_t depth; // number of matrixes
  Mat *maps; // data
};

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

[[nodiscard]]
Tensor3D *tensor_new(size_t size, size_t depth); 

/* Destroy Tensor3D buffer t
 * t must have been created with tensor_new
 */
void tensor_destroy(Tensor3D *t);

void activate_tensor(Tensor3D *tensor, ACT_FUNC activation);

#endif // MATH_TENSOR3D_H
