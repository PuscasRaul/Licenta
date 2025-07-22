#include <stdio.h>
#include <assert.h>
#include "../Licenta/math/ndarray.h"

void test_ndarray_createDestroy() {
  printf("--------->>CREATING TESTS<<---------\n");
  size_t dimensions[2] = {2, 3};
  // test for negative dimensions
  size_t dimensions1[64];
  // test for max dims
  size_t dimensions2[128] = {2, 3};
  // test for single dimension 
  size_t dimensions3[1] = {4};
  // test for negative dimensions sizes
  size_t dimensions4[4] = {-1, 5, -3, 4};

  for (size_t i = 0; i < 128; i++) {
    if (i < 64)
      dimensions1[i] = 1; // anything more is just going to be huge number
    dimensions2[i] = 2; // even with a 2, it's 2^128 and like, hell nah

  }

  ndarray *arr = ndarray_create(2, dimensions, FLOAT32); 
  assert(arr);
  ndarray *arr1 = ndarray_create(64, dimensions1, FLOAT32); 
  assert(arr1);
  ndarray *arr2 = ndarray_create(128, dimensions2, FLOAT32); 
  assert(!arr2);
  ndarray *arr3 = ndarray_create(1, dimensions3, FLOAT32); 
  assert(arr3);
  ndarray *arr4 = ndarray_create(4, dimensions4, FLOAT32); 
  assert(!arr4);
  printf("----->>CREATION TEST PASSED<<------\n");
  ndarray_debugPrint(arr);

  printf("--------->>DESTRUCTION TESTS<<---------\n");
  ndarray_destroy(arr);
  assert(!arr->descr);
  assert(!arr->strides);
  assert(!arr->dimensions);
  ndarray_destroy(arr2);
  assert(1); // should be handled gracefully
  ndarray_destroy(arr1);
  ndarray_destroy(arr3);
  printf("--------->>DESTRUCTION TESTS PASSED<<---------\n");
}

void test_ndarray_reshape() {
  printf("--------->>RESHAPE TESTS<<---------\n");
  size_t dimensions[2] = {2, 3};
  size_t reshaped_dim[3] = {3, 2};

  ndarray *arr = ndarray_create(2, dimensions, FLOAT32);
  ndarray_debugPrint(arr);
  arr = reshape(arr, 2, reshaped_dim);
  ndarray_debugPrint(arr);
}

int main(void) {
  test_ndarray_createDestroy();
  test_ndarray_reshape();
}
