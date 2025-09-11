#include "../Licenta/math/Mat.h"
#include <assert.h>

void creation_destruction_tests() {
  // basic initialization
  Mat m_stack;
  assert(mat_init(&m_stack, 1, 1));
  assert(m_stack.es);
  assert(m_stack.cols == m_stack.rows);

  // heap allocation
  Mat *m_heap = mat_create(1, 1);
  assert(m_heap);
  assert(m_heap->rows == m_heap->cols);

  // basic deinitialization
  mat_deinit(&m_stack);
  assert(!m_stack.cols);

  // heap destruction
  mat_destroy(m_heap);
  assert(m_heap); // should not be nullptr at the moment
  assert(!m_heap->rows);
  m_heap = nullptr;

  // CHECK FOR DOUBLE DESTROY BEHAVIOUR
  mat_deinit(m_heap);
  mat_destroy(m_heap);
  mat_deinit(&m_stack);

  // nullptr initialization, and 0 size initialization
  assert(!mat_init(nullptr, 1, 1));
  assert(!mat_init(&m_stack, 0, 0)); 
  assert(!mat_init(&m_stack, 1, 0)); 
  assert(!mat_init(&m_stack, 0, 1)); 
  assert(!mat_create(0, 0));
  assert(!mat_create(0, 1));
  assert(!mat_create(1, 0));
}

void slicing_tests() {
  Mat m1, m2;
  size_t row = 450, col = 450;
  size_t nrows = 300, ncols = 200;

  mat_init(&m1, 1000, 1000); // just a big chunck
  assert(mat_slice(&m2, &m1, row, col, nrows, ncols));
  assert(!m2.owns_data);
  assert(m2.stride == m1.stride);
  assert(m2.rows == nrows);
  assert(m2.cols == ncols);
  assert(m2.es == m1.es + row * m1.stride + col);

  *(mat_at(&m2, 0, 0)) = ncols + nrows;

  mat_deinit(&m2);
  assert(m1.es);
  assert(*(mat_at(&m1, row, col)) == ncols + nrows);

  // try going out of bounds when creating a slice
  assert(!mat_slice(&m2, &m1, row, col, nrows + 500, ncols + 500));
  assert(!mat_slice(&m2, &m1, row, col, nrows, ncols + 500));
  assert(!mat_slice(&m2, &m1, row, col, nrows + 500, ncols));
  assert(!mat_slice(&m2, &m1, row, col, -1, -1)); // should fail, since it will wrap to 2^32

  assert(!mat_slice(&m2, &m1, row + 1000, col + 1000, nrows, ncols));
  assert(!mat_slice(&m2, &m1, row, col + 1000, nrows, ncols));
  assert(!mat_slice(&m2, &m1, row + 1000, col, nrows, ncols));

  // create slice of size 0
  assert(mat_slice(&m2, &m1, row, col, 0, 0));

  // create just row, or just column
  assert(mat_slice(&m2, &m1, row, col, nrows, 0));
  assert(mat_slice(&m2, &m1, row, col, 0, ncols));
}

int main(void) {
  creation_destruction_tests(); 

  return 0;
}
