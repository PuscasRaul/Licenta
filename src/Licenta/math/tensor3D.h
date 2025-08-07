#include <stdint.h>
#include "Mat.h"

#ifndef TENSOR3D_H
#define TENSOR3D_H

typedef struct {
  size_t size; // square tensors 
  uint8_t depth; 
  Mat **maps;  
} Tensor3D;

// WARN: Mock function, used for defining the prototypes of CNN functions
Tensor3D *tensor_init(size_t size, size_t depth);
void tensor_deinit(Tensor3D *tensor);

#endif // TENSOR3D_H
