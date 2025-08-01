#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef MAT_MALLOC 
#define MAT_MALLOC malloc
#endif // MAT_MALLOC

#ifndef MAT_FREE
#define MAT_FREE free
#endif // MAT_FREE

#ifndef MAT_ASSERT
#define MAT_ASSERT assert
#endif // MAT_ASSERT

/**
 * @brief Access a matrix element at position (i, j)
 * @param m The matrix
 * @param i Row index
 * @param j Column index
 * @return The element at position (i, j)
 */
#define MAT_AT(m, i, j) (m)->es[(i)*(m)->stride + (j)]

typedef enum {
  FLOAT32,
  FLOAT64,
  INT32,
  INT64
} data_type;

typedef struct {
  int owns_data;
  data_type dtype;
  size_t item_size;
} Mat_description;

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  Mat_description *descr;
  char *es;
} Mat;

/**
 * @brief Create a new matrix with given dimensions
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to the created matrix or NULL on failure
 */
Mat *mat_create(size_t rows, size_t cols, data_type dtype); 

/**
 * @brief Destroy a matrix created with mat_create
 * @param m Pointer to the matrix to destroy
 * @return 0 on success, -1 on failure
 */
void mat_destroy(Mat *m);

/**
 * @brief Fill a matrix with a constant value
 * @param m Matrix to fill
 * @param value Value to fill the matrix with
 * @return 0 on success, -1 on failure
 */
void mat_fill(Mat *m, float value);

Mat *mat_slice(Mat *m, size_t row, size_t col, size_t nrows, size_t ncols); 

/**
 * @brief Copies a row from a matrix
 * @param dst Destination matrix (should be 1xN)
 * @param src Source matrix
 * @param row Row index to extract
 * @return 0 on success, -1 on failure
 */
Mat *mat_row(Mat *src, size_t row);

/**
 * @brief Extract a column from a matrix
 * @param dst Destination matrix (should be Mx1)
 * @param src Source matrix
 * @param col Column index to extract
 * @return 0 on success, -1 on failure
 */
Mat *mat_col(Mat *src, size_t col);

/**
 * @brief Print a matrix with a given name
 * @param m Matrix to print
 * @param name Name to display before the matrix
 */
void mat_print(Mat m, size_t padding);

/**
 * @brief Multiply two matrices
 * @param result Result matrix (must have proper dimensions)
 * @param left Left operand
 * @param right Right operand
 * @return 0 on success, -1 on failure
 * @note result must have dimensions left->rows x right->cols
 */
Mat *mat_multiply(Mat *left, Mat *right);

/**
 * @brief Multiply all elements of a matrix by a scalar
 * @param m Matrix to scale
 * @param value Scalar value
 * @return 0 on success, -1 on failure
 */
void mat_scalar(Mat * const m, float value);

/**
 * @brief Add another matrix to the destination matrix
 * @param dst Destination matrix (will be modified)
 * @param a Matrix to add
 * @return 0 on success, -1 on failure
 */
int mat_sum(Mat *dst, Mat *a);

#endif // MAT_H_

