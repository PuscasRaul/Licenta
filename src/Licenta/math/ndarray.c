// TODO: Switch every printf, assert with the Logger.log
// TODO: Fix stride calculations

#include "ndarray.h"
#include <string.h>
#include <stdio.h>

size_t ndarray_size(ndarray *arr) {
  if (!arr)
    return 0;

  size_t total_elems = 1;
  for (size_t i = 0; i < arr->nd; i++) {
    total_elems *= arr->dimensions[i];
  }

  return total_elems;
}

static inline int ndarray_ElementStrides(ndarray *arr) {
  size_t itemsize = arr->descr->item_size;
  size_t ndim = arr->nd;
  size_t *strides = arr->strides;

  for (size_t i = 0; i < ndim; i++) {
    if (strides[i] % itemsize != 0)
      return 0;
  }
  return 1;
}

ndarray *ndarray_create(
    size_t ndim, 
    size_t *dimensions, 
    e_array_dtype type
){
  if (ndim > NDARRAY_MAXDIMS)
    return NULL;

  ndarray *array = NDARRAY_ALLOC(sizeof(ndarray));
  if (!array)
    return NULL;
  
  array->descr = 0;
  array->strides = 0;
  array->dimensions = 0;
  array->data = 0;

  array->nd = ndim;
  array->descr = NDARRAY_ALLOC(sizeof(array_description));
  if (!array->descr) 
    goto fail;

  array->descr->owns_data = 1;
  array->descr->dtype = type;
  switch(type) {
    case FLOAT32:
      array->descr->item_size = 4;
      break;
    case FLOAT64:
      array->descr->item_size = 8;
      break;
    case INT32:
      array->descr->item_size = 4;
      break;
    case INT64:
      array->descr->item_size = 8;
      break;
    default:
      /* Unknown type */
      goto fail;
  }

  int n = 1;
  if (*(char *)&n == 1)
    array->descr->byteorder = '<';
  else 
    array->descr->byteorder = '>';

  array->dimensions = NDARRAY_ALLOC(ndim * sizeof(size_t));
  if (!array->dimensions) 
    goto fail;
  

  array->strides = NDARRAY_ALLOC(ndim * sizeof(size_t));
  if (!array->strides) 
    goto fail;
  
  
  for (size_t i = 0; i < ndim; i++) {
    array->dimensions[i] = dimensions[i];
  }

  /*
    the strides for each dimension
    represent the offset in bytes to get to 
    the very first element of the said dimension
    so the stride(layer[i]) is 
    stride(layer[i+1] + (dimension(layer[i+1]) * item_size))
    and it is the very next layer
    because we go outside-inwards
  */

  size_t item_size = array->descr->item_size;
  array->strides[ndim - 1] = item_size;
  for (int i = ndim - 2; i >= 0; i--) {
    array->strides[i] += array->strides[i + 1] * dimensions[i + 1]; 
  }

  if (!ndarray_ElementStrides(array)) 
    goto fail; 
  
  size_t data_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    if (dimensions[i] == 0) {
      data_size = 0;
      break;
    }

    if (data_size > NDARRAY_SIZE_MAX / dimensions[i]) 
      goto fail; 

    data_size *= dimensions[i];
  }

  size_t total_bytes;
  if (data_size > NDARRAY_SIZE_MAX / array->descr->item_size) 
    // Overflow
    goto fail;
  

  total_bytes = data_size * array->descr->item_size;
  array->data = NDARRAY_ALLOC(total_bytes);
  if (!array->data) 
    goto fail; 

  return array;

  fail:
    if (array->descr)
      NDARRAY_FREE(array->descr);
    if (array->dimensions)
      NDARRAY_FREE(array->dimensions);
    if (array->strides)
      NDARRAY_FREE(array->strides);
    if (array->data)
      NDARRAY_FREE(array->data);
    NDARRAY_FREE(array);
    return NULL;
}

inline void ndarray_destroy(ndarray *arr) {
  int owns = 0; // set to 0 by default
  if (!arr) 
    return;
  
  if (arr->descr) {
    owns = arr->descr->owns_data;
    NDARRAY_FREE(arr->descr);
    arr->descr = NULL;
  }
  if (arr->dimensions) {
    NDARRAY_FREE(arr->dimensions);
    arr->dimensions = NULL;
  }
  if (arr->strides) {
    NDARRAY_FREE(arr->strides);
    arr->strides = NULL;
  }
  if (arr->data && owns) {
    NDARRAY_FREE(arr->data);
    arr->data = NULL;
  }
  NDARRAY_FREE(arr);
  arr = NULL;
}

inline void ndarray_debugPrint(ndarray *arr) {
  printf("-------------------------------------------------------\n");
  printf(" Dump of Array at address %p\n", arr);
  if (arr == NULL) {
    printf(" It's NULL!\n");
    printf("-------------------------------------------------------\n");
    fflush(stdout);
    return;
  }

  printf(" ndim   : %ld\n", arr->nd);
  printf(" shape  :");
  for (size_t i = 0; i < arr->nd; ++i) {
    printf(" %ld," ,arr->dimensions[i]);
  }
  printf("\n");

  printf(" dtype  : ");
  switch(arr->descr->dtype) {
    case FLOAT32:
      printf("FLOAT32");
      break;
    case FLOAT64:
      printf("FLOAT64");
      break;
    case INT32:
      printf("INT32");
      break;
    case INT64: 
      printf("INT64");
      break;
  }
  printf("\n");
  printf(" data   : %p\n", arr->data);
  printf(" strides:");
  for (size_t i = 0; i < arr->nd; ++i) {
    printf(" %ld," , arr->strides[i]);
  }
  printf("\n");
  printf("-------------------------------------------------------\n");
  fflush(stdout);
}

static int attempt_nocopy_reshape(
    ndarray *self,
    size_t newnd,
    const size_t *new_dims,
    size_t *new_strides
) {
  int oldnd;
  size_t old_dims[NDARRAY_MAXDIMS];
  size_t old_strides[NDARRAY_MAXDIMS];
  size_t last_stride;
  int oi, oj, ok, ni, nj, nk;

  oldnd = 0;
  /*
   * Remove axes with dimension 1 from the old array. They have no effect
   * but would need special cases since their strides do not matter.
   */
  for (oi = 0; oi < self->nd; ++oi) {
    if (self->dimensions[oi] != 1) {
      old_dims[oldnd] = self->dimensions[oi];
      old_strides[oldnd] = self->strides[oi];
      ++oldnd;
    }
  }

  oi = 0;
  oj = 1;
  ni = 0;
  nj = 1;

  while (ni < newnd && oi < oldnd) {
    size_t np = new_dims[ni];
    size_t op = new_dims[oi];

    while (np != op) {
      if (np < op) 
        np *= new_dims[nj++];
      else 
        op *= old_dims[oj++];
    }

    for (ok = oi; ok < oj - 1; ok++) {
      if (old_strides[ok] != old_dims[ok + 1] * old_strides[ok + 1])
        /* not contiguous enough */
        return 0;
    }

    new_strides[nj - 1] = old_strides[oj - 1];
    for (nk = nj - 1; nk > ni; nk--) {
      new_strides[nk - 1] = new_strides[nk] * new_dims[nk];
    }

    ni = nj++;
    oi = oj++;
  }

  if (ni >= 1) 
    last_stride = new_strides[ni - 1];
  else
    last_stride = self->descr->item_size; 

  for (nk = ni; nk < newnd; nk++)
    new_strides[nk] = last_stride;

  return 1;
}

/* The current implementation does not check if the array
 * OWNS THE DATA
 * I do not see a need for it, yet, this function does a "deep-copy"
 */
static inline ndarray *deep_reshape(
    ndarray *arr, 
    size_t new_nd, 
    size_t *new_dims
){
  size_t old_nbytes, new_nbytes; 
  size_t old_size = 1, new_size = 1;
  int k, elsize;

  for (k = 0; k < arr->nd; k++) 
    old_size *= arr->dimensions[k];

  for (k = 0; k < new_nd; k++) {
    if (new_dims[k] == 0) {
      new_size = 0;
      break;
    }

    new_size *= new_dims[k];
  }

  if (old_size != new_size) {
    printf("ERROR: Size mismatch\n");
    return NULL;
  }

  // create the reshaped array
  ndarray *reshaped = ndarray_create(new_nd, new_dims, arr->descr->dtype);
  if (!reshaped)
    return NULL;

  // copy the data into it
  // strides should be the same after the ndarray_create, but out of safety
  // will recopy them
  // and the data will be copied in the same order through memcpy, 
  // just the view changed

  old_nbytes = old_size * arr->descr->item_size;
  new_nbytes = new_size * reshaped->descr->item_size;
  // feels a bit off
  // no checking can be done on it
  // but i will keep it as it is for now
  // if any problems arise
  // will move to a NDARRAY_REALLOC
  memcpy(reshaped->data, arr->data, old_nbytes);

  return reshaped;
}

static inline ndarray *view_reshape(
    ndarray *arr, 
    size_t new_nd, 
    size_t *new_dims
) {
  // temporary 
  int k;
  size_t item_size = arr->descr->item_size; 
  size_t old_size = 1, new_size = 1;
  size_t *new_strides;

  for (k = 0; k < arr->nd; k++) 
    old_size *= arr->dimensions[k];

  for (k = 0; k < new_nd; k++) {
    if (new_dims[k] == 0) {
      new_size = 0;
      break;
    }

    new_size *= new_dims[k];
  }

  if (old_size != new_size) {
    printf("ERROR: Size mismatch\n");
    return NULL;
  }

  new_strides = malloc(sizeof(size_t) * new_nd);
  if (!new_strides)
    return NULL;

  new_strides[new_nd - 1] = item_size; 
  for (k = new_nd - 2; k >= 0; k--) {
    new_strides[k] = new_strides[k + 1] * new_dims[k + 1];
  }

  if (!attempt_nocopy_reshape(arr, new_nd, new_dims, new_strides))
    return NULL;

  ndarray *view = NDARRAY_ALLOC(sizeof(ndarray));
  if (!view)
    goto fail;
  view->data = arr->data;

  // initialize the descriptor
  view->descr = NDARRAY_ALLOC(sizeof(array_description));
  if (!view->descr)
    goto fail;
  view->descr->item_size = item_size;
  view->descr->owns_data = 0;
  view->descr->dtype = arr->descr->dtype;
  view->descr->byteorder = arr->descr->byteorder;

  view->nd = new_nd;
  view->dimensions = NDARRAY_ALLOC(sizeof(size_t) * new_nd);
  if (!view->dimensions)
    goto fail;
  view->strides = NDARRAY_ALLOC(sizeof(size_t) * new_nd);
  if (!view->strides)
    goto fail;
  memcpy(view->strides, new_strides, sizeof(size_t) * new_nd);
  memcpy(view->dimensions, new_dims, sizeof(size_t) * new_nd);

  free(new_strides);
  return view;

fail:
  if (view->descr) {
    NDARRAY_FREE(view->descr);
    view->descr = NULL;
  }
  if (view->strides) {
    NDARRAY_FREE(view->strides);
    view->strides = NULL;
  }
  if (view->dimensions) {
    NDARRAY_FREE(view->dimensions);
    view->dimensions = NULL;
  }
  if (view) {
    NDARRAY_FREE(view);
    view = NULL;
  }
  free(new_strides);
  return NULL;
}

ndarray *reshape(ndarray *arr, size_t new_nd, size_t *new_dims) {
  ndarray *reshaped = view_reshape(arr, new_nd, new_dims);
  if (!reshaped)
    return deep_reshape(arr, new_nd, new_dims);

  return reshaped;
}


