#include "Mat.h"

static inline char *mat_at(Mat *m, size_t row, size_t col) {
  return m->es + row * m->stride + col; 
}

Mat *mat_create(size_t rows, size_t cols, data_type dtype) {
  if (rows <= 0 || cols <= 0)
    return NULL;

  Mat_description *descr = MAT_MALLOC(sizeof(Mat_description));
  if (!descr)
    return NULL;

  descr->owns_data = 1;
  descr->dtype = dtype;
  switch(dtype % 2) {
    case 0:
      descr->item_size = 4;
      break;
    case 1:
      descr->item_size = 8;
      break;
  }
  
  Mat *mat = MAT_MALLOC(sizeof(Mat));
  if (!mat)
    goto fail;

  char *es = MAT_MALLOC(descr->item_size * rows * cols);
  if (!es)
    goto fail;

  mat->rows = rows;
  mat->cols = cols;
  mat->stride = cols * descr->item_size;
  mat->descr = descr;
  mat->es = es;

  return mat;
  
  fail:
    if (descr)
      MAT_FREE(descr);
    if (mat) {
      if (mat->es)
        MAT_FREE(mat->es);
      MAT_FREE(mat);
    }
    return NULL;
}

void mat_destroy(Mat *m) {
  if (m->descr->owns_data)
    MAT_FREE(m->es);

  MAT_FREE(m->descr);
  MAT_FREE(m);
}

void mat_fill(Mat *m, float value) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) 
      MAT_AT(m, i, j) = value;
  }
}

Mat *mat_slice(Mat *m, size_t row, size_t col, size_t nrows, size_t ncols) {
  if (nrows <= 0 || ncols <= 0)
    return NULL;

  if (row < 0 || row >= m->rows)
    return NULL;

  if (col < 0 || col >= m->cols)
    return NULL;

  if (nrows + row >= m->rows || ncols + col >= m->cols)
    return 0;


  Mat *slice = MAT_MALLOC(sizeof(Mat));
  if (!slice)
    return NULL;

  slice->descr = MAT_MALLOC(sizeof(Mat_description));
  if (!slice->descr)
    goto fail;

  slice->descr->owns_data = 0;
  slice->descr->item_size = m->descr->item_size;
  slice->descr->dtype = m->descr->dtype;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  slice->stride = m->stride;
  slice->rows = nrows;
  slice->cols = ncols;
  slice->es = mat_at(m, row, col);

  return slice;

fail:
  if (slice->descr)
    MAT_FREE(slice->descr);

  MAT_FREE(slice);
  return NULL;
}

Mat *mat_row(Mat *m, size_t row) {
  if (row < 0 || row >= m->rows)
    return NULL;

  Mat *result = MAT_MALLOC(sizeof(Mat));
  if (!result)
    return NULL;

  result->descr = MAT_MALLOC(sizeof(Mat_description));
  if (!result->descr)
    goto fail;

  result->descr->owns_data = 0;
  result->descr->item_size = m->descr->item_size;
  result->descr->dtype = m->descr->dtype;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  result->stride = m->stride;
  result->rows = 1;
  result->cols = m->cols;
  result->es = mat_at(m, row, 0);

  return result;

fail:
  if (result->descr)
    MAT_FREE(result->descr);

  MAT_FREE(result);
  return NULL;
}

Mat *mat_col(Mat *m, size_t col) {
  if (col < 0 || col >= m->cols)
    return NULL;

  Mat *result = MAT_MALLOC(sizeof(Mat));
  if (!result)
    return NULL;

  result->descr = MAT_MALLOC(sizeof(Mat_description));
  if (!result->descr)
    goto fail;

  result->descr->owns_data = 0;
  result->descr->item_size = m->descr->item_size;
  result->descr->dtype = m->descr->dtype;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  result->stride = m->stride;
  result->rows = m->rows;
  result->cols = 1;
  result->es = mat_at(m, 0, col);

  return result;

fail:
  if (result->descr)
    MAT_FREE(result->descr);

  MAT_FREE(result);
  return NULL;
}

void mat_print(Mat m, char *name, size_t padding);

Mat *mat_multiply(Mat *left, Mat *right) {
  if (left->descr->dtype != right->descr->dtype)
    return NULL;
  Mat *result = mat_create(left->rows, right->cols, left->descr->dtype);

  for (size_t i = 0; i < left->rows; i++) {
    for (size_t j = 0; j < right->cols; j++) 
      for (size_t k = 0; k < left->cols; j++) 
        MAT_AT(result, i, j) += MAT_AT(left, i, k) * MAT_AT(right, k, j);
  }

  return result;
}

void mat_scalar(Mat *m, float value) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) 
      MAT_AT(m, i, j) *= value;
  }
}

int mat_sum(Mat *dst, Mat *a) {
  if (dst->rows != a->rows)
    return -1;

  if (dst->cols != a->cols)
    return -1;

  for (size_t i = 0; i < dst->rows; i++) {
    for (size_t j = 0; j < dst->cols; j++)
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
  }
  return 0;
}

int mat_dot(Mat *left, Mat *right, void *result) {
  if (left->cols != right->rows)
    return -1;

  if (left->descr->dtype != right->descr->dtype)
    return -1;

  switch (left->descr->dtype) {
    case FLOAT32: {
      float *casted = (float*) result;
      *casted = 0;
      for (size_t i = 0; i < left->rows; i++) {
        for (size_t j = 0; j < right->cols; j++) {
          for (size_t k = 0; k < left->cols; k++) {
            *casted += MAT_AT(left, i, j) * MAT_AT(right, i, j); 
          }
        }
      }
      break;
    }
    case FLOAT64: {
      double *casted = (double*) result;
      for (size_t i = 0; i < left->rows; i++) {
        for (size_t j = 0; j < right->cols; j++) {
          for (size_t k = 0; k < left->cols; k++) {
            *casted += MAT_AT(left, i, j) * MAT_AT(right, i, j); 
          }
        }
      }
      break;
    }
    case INT32: {
      int32_t *casted = (int32_t*) result;
      for (size_t i = 0; i < left->rows; i++) {
        for (size_t j = 0; j < right->cols; j++) {
          for (size_t k = 0; k < left->cols; k++) {
            *casted += MAT_AT(left, i, j) * MAT_AT(right, i, j); 
          }
        }
      }
      break;
    }
    case INT64: {
      int64_t *casted = (int64_t*) result;
      for (size_t i = 0; i < left->rows; i++) {
        for (size_t j = 0; j < right->cols; j++) {
          for (size_t k = 0; k < left->cols; k++) {
            *casted += MAT_AT(left, i, j) * MAT_AT(right, i, j); 
          }
        }
      }
      break;
    }
  }
  return 0;
}
