// UTILITAR SCREEN
#ifndef SRC_MATH_NDARRAY_H
#define SRC_MATH_NDARRAY_H

#include <stddef.h>
#include <stdlib.h>

#define NDARRAY_SIZE_MAX 10200 // make it a huge number, will tinker with it
#define NDARRAY_MAXDIMS 128

#ifndef NDARRAY_ALLOC
#define NDARRAY_ALLOC malloc
#endif //NDARRAY_ALLOC

#ifndef NDARRAY_REALLOC
#define NDARRAY_REALLOC realloc
#endif // NDARRAY_REALLOC

#ifndef NDARRAY_FREE
#define NDARRAY_FREE free
#endif //NDARRAY_free

typedef enum {
  FLOAT32,
  FLOAT64,
  INT32,
  INT64
} e_array_dtype;

typedef struct {
  /*
   * '>' (big), '<' (little), '|'
   * (not-applicable), or '=' (native).
   */
  char byteorder;
  /* element size (itemsize) for this type */
  e_array_dtype dtype;
  int owns_data;
  size_t item_size;
} array_description;

typedef struct _ndarray {
  /* Pointer to the raw data buffer */
  char *data;
  /* The number of dimensions, also called 'ndim' */
  size_t nd;
  /* The size in each dimension, also called 'shape' */
  size_t *dimensions; 
  /*
   * Number of bytes to jump to get to the
   * next element in each dimension
   */
  size_t *strides;
  array_description *descr;
} ndarray;

/*
 * Returns the size in number of elements
 */
size_t ndarray_size(ndarray *arr);

/* Returns a pointer to the created ndarray on success
 * NULL on failure
 * ndim - number of dimensions for the array
 * dimensions - array with the size of each dimension
 * type - the data type
 * NOTE: All the data is allocated through the NDARRAY_MALLOC
 *       define the macro if special behaviour is needed
 */
ndarray *ndarray_create(size_t ndim, size_t *dimensions, e_array_dtype type);

/* Free's the array, including all it's contents
 * NOTE: All the data is free'd through the NDARRAY_FREE
 * NOTE: Checks for null pointers before destroying a resource
 * NOTE: The ndarray must own the data for it to be free'd
 */
void ndarray_destroy(ndarray *arr);

/* TODO: Define the api, write the function properly
*/
ndarray *ndarray_dot(ndarray *left, ndarray *right);

/* Prints the arr and it's content, only essential information is being printed
 * Such as address, number of dimensions, dimensions, data type, data, strides
 * NOTE: Does not check if any pointer is null before attempting to dereference
 */
void ndarray_debugPrint(ndarray *arr);

/* Reshapes the ndarray
 * Will do a efficient no-copy reshape if possible, and return the same pointer
 * Otherwise it will return a new pointer
 * NULL on failure
 */
ndarray *reshape(ndarray *arr, size_t new_nds, size_t *new_dims);

/* Multiplies the 2 arrays
 * Assumes the multiplication is done over the first axis of a
 * And the last axis of b
 * Returns NULL if the 2 have different sizes, or on failure
 * a new array on success
 */
ndarray *ndarry_dot(ndarray *a, ndarray *b);

ndarray *ndarray_tensordot(
    ndarray *a, 
    ndarray *b, 
    size_t *axes_a,
    size_t *axes_b,
    size_t naxes
);

#endif //SRC_MATH_NDARRAY_H 
