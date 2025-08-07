#include "Mat.h"

static inline double *mat_at(Mat *m, size_t row, size_t col) {
  return m->es + row * m->stride + col; 
}

Mat *mat_create(size_t rows, size_t cols) {
  if (rows <= 0 || cols <= 0)
    return NULL;

  Mat *mat = MAT_MALLOC(sizeof(Mat)); // add size of es here
  if (!mat)
    return NULL;

  mat->es = MAT_MALLOC(sizeof(double) * rows * cols);
  if (!mat->es)
    goto fail;

  mat->rows = rows;
  mat->cols = cols;
  mat->owns_data = 1;
  mat->stride = cols * sizeof(double);

  return mat;
  
  fail:
  if (mat->es)
    MAT_FREE(mat->es);
  MAT_FREE(mat);
  return NULL;
}

void mat_destroy(Mat *m) {
  if (m->owns_data)
    MAT_FREE(m->es);

  MAT_FREE(m);
}

void mat_fill(Mat *m, double value) {
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


  slice->owns_data = 0;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  slice->stride = m->stride;
  slice->rows = nrows;
  slice->cols = ncols;
  slice->es = mat_at(m, row, col);

  return slice;

fail:
  MAT_FREE(slice);
  return NULL;
}

Mat *mat_row(Mat *m, size_t row) {
  if (row < 0 || row >= m->rows)
    return NULL;

  Mat *result = MAT_MALLOC(sizeof(Mat));
  if (!result)
    return NULL;

  result->owns_data = 0;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  result->stride = m->stride;
  result->rows = 1;
  result->cols = m->cols;
  result->es = mat_at(m, row, 0);

  return result;

fail:
  MAT_FREE(result);
  return NULL;
}

Mat *mat_col(Mat *m, size_t col) {
  if (col < 0 || col >= m->cols)
    return NULL;

  Mat *result = MAT_MALLOC(sizeof(Mat));
  if (!result)
    return NULL;


  result->owns_data = 0;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  result->stride = m->stride;
  result->rows = m->rows;
  result->cols = 1;
  result->es = mat_at(m, 0, col);

  return result;

fail:
  MAT_FREE(result);
  return NULL;
}

void mat_print(Mat *m) {
    // Print matrix info header
    const char* dtype_names[] = {"FLOAT32", "FLOAT64", "INT32", "INT64"};
    printf("Matrix Info:\n");
    printf("  Shape: %zu x %zu\n", m->rows, m->cols);
    printf("  Stride: %zu\n", m->stride);
    printf("  Item Size: %zu bytes\n", sizeof(double));
    printf("  Owns Data: %s\n", m->owns_data ? "yes" : "no");
    printf("  Memory Address: %p\n", (void*)m->es);
    printf("\nMatrix Data:\n");
    
    // Handle null pointer
    if (m->es == NULL) {
        printf("  [NULL DATA]\n");
        return;
    }
    
    // Handle empty matrix
    if (m->rows == 0 || m->cols == 0) {
        printf("  [EMPTY MATRIX]\n");
        return;
    }
    
    // Print matrix data based on type
    for (size_t i = 0; i < m->rows; i++) {
        printf("[%2zu] ", i);
        
        for (size_t j = 0; j < m->cols; j++) {
          double val = *mat_at(m, i, j);
          printf("%f ", val);
        }
          printf("\n");
    }
    printf("\n");
}

Mat *mat_multiply(Mat *left, Mat *right) {
  if (left->cols != right->rows)
    return NULL;

  Mat *result = mat_create(left->rows, right->cols);
  if (!result)
    return NULL;

  mat_fill(result, 0.0);
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

  double *casted = (double*) result;
  for (size_t i = 0; i < left->rows; i++) {
    for (size_t j = 0; j < right->cols; j++) {
      for (size_t k = 0; k < left->cols; k++) {
        *casted += MAT_AT(left, i, j) * MAT_AT(right, i, j); 
      }
    }
  }

  return 0;
}
