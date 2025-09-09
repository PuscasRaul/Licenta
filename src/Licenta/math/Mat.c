#include "Mat.h"
#include <__stddef_unreachable.h>
#include <stdio.h>

// unsafe method, meant to be used only inside the library
#define MAT_AT(m, i, j) (m)->es[(i) * (m)->stride + (j)]

[[nodiscard]]
double *mat_at(
    const Mat *m,
    size_t row,
    size_t col
) {
  if (!m)
    return nullptr;

  if (row >= m->rows || col >= m->cols)
    return nullptr;

  return (m->es + row * m->stride + col); 
}

Mat *mat_init(Mat *m, size_t rows, size_t cols) {
  if (!m)
    return nullptr;

  if (!rows || !cols) {
    *m = (Mat) {};
    return nullptr;
  }
  
  *m = (Mat) {
    .cols = cols,
    .rows = rows,
    .stride = cols,
    .owns_data = 1,
    .es = MAT_MALLOC(sizeof(double) * rows * cols)
  };

  if (!m->es) { 
    *m = (Mat) {};
    return nullptr;
  }

  return m;
}

void mat_deinit(Mat m[static 1]) {
  if (!m)
    return;
  if (m->owns_data)  
    MAT_FREE(m->es);
  *m = (Mat) {};
}

Mat *mat_create(size_t rows, size_t cols) {
  return mat_init(malloc(sizeof(struct Mat)), rows, cols);
}

void mat_destroy(Mat *m) {
  if (m) {
    mat_deinit(m);
    MAT_FREE(m);
  }
}

[[nodiscard]]
Mat *mat_vcreate(size_t length, size_t m_size) {
  if (!length || !m_size)
    return nullptr;
  
  Mat *m = MAT_MALLOC(sizeof(Mat) * length);
  if (!m)
    return nullptr;

  for (size_t i = 0; i < length; i++)
    if (!mat_init(&m[i], m_size, m_size))
      goto fail;

  return m;

fail:
  for (size_t i = 0; i < length; i++) {
    if (!m[i].rows)
      break;
    mat_deinit(&m[i]);
  }

  return nullptr;
}

void mat_vdestroy(size_t length, Mat vmat[static length]) {
  for (size_t i = 0; i < length; i++) 
    mat_deinit(&vmat[i]);
   
  MAT_FREE(vmat);
}

void mat_fill(Mat *m, double value) {
  if (!m || !m->es) return;
  for (size_t i = 0; i < m->rows * m->cols; i++) {
    m->es[i] = value;
  }
}

Mat *mat_slice(
    Mat *out,
    const Mat *m,
    size_t row,
    size_t col,
    size_t nrows,
    size_t ncols
){
  
  if (!m) return nullptr;

  if (!m->cols || !m->rows) unreachable();

  if (row > m->rows || col > m->cols)
    return nullptr;

  if (nrows > m->rows - row || ncols > m->cols - col)
    return nullptr;

  if (!out)
    return nullptr;

  out->cols = ncols;
  out->rows = nrows;
  out->stride = m->stride;
  out->owns_data = 0;
  out->es = m->es + row * m->stride + col;
  return out;
}

Mat *mat_row(Mat *out, const Mat *m, size_t row) {
  if (row >= m->rows)
    return nullptr;

  if (!out) 
    if (!(out = malloc(sizeof(Mat))))
        return nullptr;

  out->owns_data = 0;

  // we do not need to modify the strides, since for the next row element
  // we need to skip the same amount of elements
  // and for the next column element, skip item_size
  out->stride = m->stride;
  out->rows = 1;
  out->cols = m->cols;
  out->es = m->es + (row * m->stride);

  return out;
}

Mat *mat_col(Mat *out, const Mat *m, size_t col) {
  if (!m) return nullptr;

  if (!m->cols || !m->rows) unreachable();

  if (col >= m->cols)
    return nullptr;

  if (!out)
    return nullptr;

  out->owns_data = 0;
  out->stride = m->stride;
  out->rows = m->rows;
  out->cols = 1;
  out->es = m->es + col;

  return out;
}

void mat_print(const Mat * const restrict m) {
  if (!m)
    return;

  // Print matrix info header
  printf("Matrix Info:\n");
  printf("  Shape: %zu x %zu\n", m->rows, m->cols);
  printf("  Stride: %zu\n", m->stride);
  printf("  Item Size: %zu bytes\n", sizeof(double));
  printf("  Owns Data: %s\n", m->owns_data ? "yes" : "no");
  printf("  Memory Address: %p\n", (void*)m->es);
  printf("\nMatrix Data:\n");

  // Handle null pointer
  if (m->es == nullptr) {
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
      double val = MAT_AT(m, i, j);
      printf("%f ", val);
    }
    printf("\n");
  }
  printf("\n");
}

// TODO: since no access to simd, maybe move this to a batch-oriented
// matrix multiplication
Mat *mat_multiply(
  Mat *out,
  const Mat *left,
  const Mat *right
) {
  if (!left || !right || !out)
    return nullptr;

  // might be redundant
  if (!out->es || !left->es || !right->es) unreachable(); 

  if (left->cols != right->rows)
    return nullptr;

  if (out->rows != left->rows || out->cols != right->cols)
    return nullptr;

  mat_fill(out, 0.0);
  for (size_t i = 0; i < left->rows; i++) {
    for (size_t k = 0; k < left->cols; k++) 
      for (size_t j = 0; j < right->cols; j++) 
        MAT_AT(out, i, j) += MAT_AT(left, i, k) * MAT_AT(right, k, j);
  }

  return out;
}

void mat_scalar(Mat *m, float value) {
  if (!m)
    return;

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) 
      MAT_AT(m, i, j) *= value;
  }
}

int mat_sum(Mat *dst, const Mat *a) {
  if (!dst || !a)
    return -1;

  if (dst->rows != a->rows)
    return -1;

  if (dst->cols != a->cols)
    return -1;

  for (size_t i = 0; i < dst->rows; i++) {
    for (size_t j = 0; j < dst->cols; j++)
      dst->es[i * dst->stride + j] += a->es[i * a->stride + j];
  }
  return 0;
}

int mat_dot(const Mat *left, const Mat *right, double *result) {
  if (!left || !right || !result)
    return -1;

  if (left->cols != right->cols || left->rows != right->rows)
    return -1;

  *result = 0;
  for (size_t i = 0; i < left->rows; i++) 
    for (size_t k = 0; k < left->cols; k++) 
      for (size_t j = 0; j < right->cols; j++) 
        *result += MAT_AT(left, i, k) * MAT_AT(right, k, j); 
    
  return 0;
}
