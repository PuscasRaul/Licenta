#include "Mat.h"
#include <__stddef_unreachable.h>

struct Mat {
  size_t rows;
  size_t cols;
  size_t stride;
  int_fast8_t owns_data;
  double *es;
};

static inline double *mat_at(const Mat *m, size_t row, size_t col) {
  assert(m->rows > row && m->cols > col);
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

void mat_deinit(Mat *m) {
  if (m) {
    MAT_FREE(m->es);
    *m = (Mat) {};
  }
}

[[nodiscard]]
inline Mat *mat_create(size_t rows, size_t cols) {
  return mat_init(malloc(sizeof(Mat)), rows, cols);
}

inline void mat_destroy(Mat *m) {
  if (m->owns_data)
    MAT_FREE(m->es);

  MAT_FREE(m);
}

void mat_fill(Mat *m, double value) {
  if (m->owns_data)
    memcpy(m->es, &value, sizeof(double) * m->rows * m->cols);
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
  out->es = mat_at(m, row, col);
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
  out->es = mat_at(m, row, 0);

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
  out->es = mat_at(m, 0, col);

  return out;
}

void mat_print(const Mat *m) {
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
      double val = *mat_at(m, i, j);
      printf("%f ", val);
    }
    printf("\n");
  }
  printf("\n");
}

Mat *mat_multiply(Mat *out, const Mat *left, const Mat *right) {
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
    for (size_t j = 0; j < right->cols; j++) 
      for (size_t k = 0; k < left->cols; j++) 
        *(mat_at(out, i, j)) += *(mat_at(left, i, k)) * *(mat_at(right, k, j));
  }

  return out;
}

void mat_scalar(Mat *m, float value) {
  if (!m)
    return;

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) 
      *(mat_at(m, i, j)) *= value;
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
      *(mat_at(dst, i, j)) += *(mat_at(a, i, j));
  }
  return 0;
}

int mat_dot(const Mat *left, const Mat *right, double *result) {
  if (!left || !right || !result)
    return -1;

  if (left->cols != right->rows)
    return -1;

  for (size_t i = 0; i < left->rows; i++) {
    for (size_t j = 0; j < right->cols; j++) {
      for (size_t k = 0; k < left->cols; k++) {
        *result += *(mat_at(left, i, j)) * *(mat_at(right, i, j)); 
      }
    }
  }

  return 0;
}
