#include <stdio.h>
#include <assert.h>
#include "../Licenta/math/ndarray.h"

int main(void) {
  size_t dimensions[2] = {2, 3};
  ndarray *arr = ndarray_create(2, dimensions, FLOAT32); 
  assert(arr);
  ndarray_debugPrint(arr);
}
