#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef NDARRAY_H
#define NDARRAY_H 

#define MAXDIMS 128

typedef struct _ndarray ndarray;
typedef struct _desc descr;
typedef struct _arr_mem_allocator nd_mem_handler;

typedef enum {
  ND_FLOAT32,
  ND_FLOAT64,
  ND_INT32,
  ND_INT64
} e_dtype;

typedef enum {
  ND_NO_COPY,
  ND_COPY
} e_copy_mode;

struct _arr_mem_allocator {
  /* at the moment i am quite confused on why this is necessary, but
   * i need to keep this since i plan on creating a memory pool 
   */
  void *ctx; 
  void (*malloc)(void *ctx, size_t size);
  void* (*calloc) (void *ctx, size_t nelem, size_t elsize);
  void* (*realloc) (void *ctx, void *ptr, size_t new_size);
  void (*free) (void *ctx, void *ptr, size_t size);
};

struct _descr{
  e_dtype dtype;
  char byte_order;
  size_t item_size;
};

struct _ndarray {
  /* Pointer to raw data buffer */
  char *data;
  /* The number of dimensions */
  int nd;
  /* The size of each dimension, called 'shape' */
  size_t *dimensions;
  /* Number of bytes to jump to get to the next element in each dimension
   * With respect to the offset of the raw-data pointer 
   */
  size_t *stride;
  /* Properties of the tensor */
  descr *description;
  /* Used for defining malloc, free, calloc functions version 
   * A good way to avoid my macro definition of ND_ALLOC and such
   */
  nd_mem_handler *mem_hamdler;
  int flags;
};

/*
 * Means c-style contiguous (last index varies the fastest). The data
 * elements right after each other.
 *
 * This flag may be requested in constructor functions.
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_C_CONTIGUOUS    0x0001

/*
 * Set if array is a contiguous Fortran array: the first index varies
 * the fastest in memory (strides array is reverse of C-contiguous
 * array)
 *
 * This flag may be requested in constructor functions.
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_F_CONTIGUOUS    0x0002

/*
 * If set, the array owns the data: it will be free'd when the array
 * is deleted.
 *
 * This flag may be tested for in PyArray_FLAGS(arr).
 */
#define NPY_ARRAY_OWNDATA         0x0004

/*
  Compute the size of an array (in number of items)
*/
size_t ndarray_size(ndarray *ts); 

/* Check if the strides are correctly alligned
 * With the item size
 */
int ndarray_ElementStrides(ndarray *self); 

/* Allocates a new tensor
 * All the data passed will be copied, the caller must free it
 * By default, the tensor will own the data 
 */
void ndarray_DebugPrint(ndarray *self); 

#endif // NDARRAY_H
