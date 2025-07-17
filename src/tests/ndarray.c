#include <stdio.h>
#include <assert.h>
#include "../Licenta/math/ndarray.h"

void test_ndarray_create() {
  printf("--------->>CREATING TESTS<<---------");
  size_t dimensions[2] = {2, 3};
  // test for negative dimensions
  size_t dimensions1[64] = {2, 3};
  // test for max dims
  size_t dimensions2[128] = {2, 3};
  // test for single dimension 
  size_t dimensions3[1] = {4};
  // test for negative dimensions sizes
  size_t dimensions4[4] = {-1, 5, -3, 4};

  for (size_t i = 0; i < 128; i++) {
    if (i < 64)
      dimensions1[i] = i + 1;
    dimensions2[i] = 2 * i;
  }

  ndarray *arr = ndarray_create(2, dimensions, FLOAT32); 
  assert(arr);
  ndarray *arr1 = ndarray_create(64, dimensions1, FLOAT32); 
  assert(arr1);
  ndarray *arr2 = ndarray_create(128, dimensions2, FLOAT32); 
  assert(arr2);
  ndarray *arr3 = ndarray_create(1, dimensions3, FLOAT32); 
  assert(arr3);
  ndarray *arr4 = ndarray_create(4, dimensions4, FLOAT32); 
  assert(arr4);
  printf("----->>CREATION TEST PASSED<<------");
  ndarray_debugPrint(arr);
}

int main(void) {
  test_ndarray_create();
}
