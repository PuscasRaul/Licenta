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
Tensor3D init_tensor();

#endif // TENSOR3D_H
