#include "tensor.h"
#include <math.h>

size_t tensor_size(tensor *ts) {
  size_t total_size = 0;
  for (size_t i = 0; i < ts->nd; i++) {
    total_size += ts->dimensions[i] * ts->description->item_size;
  }
  return total_size;
}

int tensor_ElementStrides(tensor *self) {
  size_t itemsize = self->description->item_size;
  size_t ndim = self->nd;
  size_t *strides = self->stride;

  for (size_t i = 0; i < ndim; i++) {
    if (strides[i] % itemsize != 0)
      return 0;
  }
  return 1;
}

tensor *tensor_alloc(
    size_t ndim, 
    size_t *dimensions, 
    e_tensor_dtype type
){
  tensor *new_tensor = TENSOR_ALLOC(sizeof(tensor));
  if (!new_tensor)
    return NULL;
  
  new_tensor->description = 0;
  new_tensor->stride = 0;
  new_tensor->dimensions = 0;
  new_tensor->data = 0;

  new_tensor->nd = ndim;
  new_tensor->description = TENSOR_ALLOC(sizeof(tensor_descr));
  if (!new_tensor->description) 
    goto fail;

  new_tensor->description->owns_data = true;
  new_tensor->description->dtype = type;
  switch(type) {
    case 0:
      new_tensor->description->item_size = 4;
      break;
    case 1:
      new_tensor->description->item_size = 8;
      break;
    case 2:
      new_tensor->description->item_size = 4;
      break;
    case 3:
      new_tensor->description->item_size = 8;
      break;
    default:
      /* Unknown type */
      goto fail;
  }

  int n = 1;
  if (*(char *)&n == 1)
    new_tensor->description->byte_order = 'L';
  else 
    new_tensor->description->byte_order = 'B';

  new_tensor->dimensions = TENSOR_ALLOC(ndim * sizeof(size_t));
  if (!new_tensor->dimensions) 
    goto fail;

  new_tensor->stride = TENSOR_ALLOC(ndim * sizeof(size_t));
  if (!new_tensor->stride) 
    goto fail;
  
  for (size_t i = 0; i < ndim; i++) {
    new_tensor->dimensions[i] = dimensions[i];
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

  size_t item_size = new_tensor->description->item_size;
  new_tensor->stride[ndim - 1] = 0;
  for (size_t i = ndim - 2; i >= 0; i--) {
    new_tensor->stride[i] = new_tensor->stride[i + 1];
    new_tensor->stride[i] += new_tensor->dimensions[i + 1] * item_size;
  }

  if (!tensor_ElementStrides(new_tensor)) 
    goto fail; 
  
  size_t data_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    if (dimensions[i] == 0) {
      data_size = 0;
      break;
    }

    if (data_size > SIZE_MAX / dimensions[i]) 
      goto fail; 

    data_size *= dimensions[i];
  }

  size_t total_bytes;
  if (data_size > SIZE_MAX / new_tensor->description->item_size) 
    // Overflow
    goto fail;

  total_bytes = data_size * new_tensor->description->item_size;
  new_tensor->data = TENSOR_ALLOC(total_bytes);
  if (!new_tensor->data) 
    goto fail; 

  return new_tensor;

  fail:
    if (new_tensor->description)
      TENSOR_FREE(new_tensor->description);
    if (new_tensor->dimensions)
      TENSOR_FREE(new_tensor->dimensions);
    if (new_tensor->stride)
      TENSOR_FREE(new_tensor->stride);
    if (new_tensor->data)
      TENSOR_FREE(new_tensor->data);
    TENSOR_FREE(new_tensor);
    return NULL;
}

void tensor_dealloc(tensor *self) {
  if (self->description->owns_data) {
    TENSOR_FREE(self->data);
    self->data = 0;
  }

  TENSOR_FREE(self->description);
  TENSOR_FREE(self->dimensions);
  TENSOR_FREE(self->stride);
  TENSOR_FREE(self);
}

void tensor_DebugPrint(tensor *self) {
  printf("-------------------------------------------------------\n");
  printf(" Dump of Tensor  at address %p\n", self);
  if (self == NULL) {
    printf(" It's NULL!\n");
    printf("-------------------------------------------------------\n");
    fflush(stdout);
    return;
  }

  printf(" ndim   : %d\n", self->nd);
  printf(" shape  :");
  for (size_t i = 0; i < self->nd; ++i) {
    printf(" %ld," ,self->dimensions[i]);
  }
  printf("\n");

  printf(" dtype  : ");
  switch(self->description->dtype) {
    case 0:
      printf("FLOAT32");
      break;
    case 1:
      printf("FLOAT64");
    case 2:
      printf("INT32");
    case 3: 
      printf("INT64");
  }
  printf("\n");
  printf(" data   : %p\n", self->data);
  printf(" strides:");
  for (size_t i = 0; i < self->nd; ++i) {
    printf(" %ld," , self->stride[i]);
  }
  printf("\n");
  printf("-------------------------------------------------------\n");
  fflush(stdout);
}

/* 
 * this is so fucking memory error prone
 * it's insane
 * shall be used only for development i guess and debugging
 * WARNING: THIS SHIT HELLA BUGGY, NO CHECKS NO NOTHING
 * WARNING: CHECK THIS SHIT WELL WHEN USING IT
 * WARNING: VERY EASILY OUT OF BOUNDS
 */

void *tensor_at(tensor *self, size_t *coordinates) {
  #if PRODUCTION
  #else 
  size_t location = 0;

  if (!self || !self->dimensions || !self->stride)
    return NULL;

  for (size_t i = 0; i < self->nd; i++) {
    if (coordinates[i] > self->dimensions[i])
      return NULL; // this should be enough for now
    location += coordinates[i] * self->stride[i];
  }

  return self->data+location;
  #endif
}

/* Checks if the current tensor can be reshaped
 * just by changing the view over the data
 */

static int attempt_nocopy_reshape(
    tensor *self,
    size_t newnd,
    const size_t *new_dims,
    size_t *new_strides
) {
  int oldnd;
  size_t old_dims[MAXDIMS];
  size_t old_strides[MAXDIMS];
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
      old_strides[oldnd] = self->stride[oi];
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
    last_stride = self->description->item_size; 

  for (nk = ni; nk < newnd; nk++)
    new_strides[nk] = last_stride;

  return 1;
}

/* tensor dot
 * quite complicated
 * need to write a mapping about it
 * to know what i need and what not
 */
