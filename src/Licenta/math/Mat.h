#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef MAT_MALLOC 
#define MAT_MALLOC malloc
#endif // MAT_MALLOC

#ifndef MAT_FREE
#define MAT_FREE free
#endif // MAT_FREE

typedef struct Mat Mat;

/**
 * Initialize a matrix 
 * The matrix must later be de-initialized through a call to @ref mat_deinit
 *
 * @pre @m must be unitialized
 * @pre rows > 0 && cols > 0
 *
 * @param m: pointer to matrix structure
 * @param rows Number of rows
 * @param cols Number of cols
 * @return Pointer to the initialized @m, or nullptr on failure
 */
Mat *mat_init(Mat *m, size_t rows, size_t cols);

/**
 * Deinitialize a matrix.
 * If the `owns_data` flag is set, frees the internal buffer and sets its pointer to nullptr.
 * In all cases, resets all fields of `m` to zero.
 * This function does not free the `Mat` structure itself.
 * Function is safe to call with m == nullptr
 *
 * @pre @p m must have been initialized with @ref mat_init or @ref mat_slice.
 *
 * @param m Pointer to a Mat structure to deinitialize.
 */
void mat_deinit(Mat *m);

/**
 * Create a new matrix 
 * The matrix must then be destroyed through a call to @ref mat_destroy
 *
 * @pre rows > 0 && cols > 0
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to the created matrix or nullptr on failure
 */
[[nodiscard]]
inline Mat *mat_create(size_t rows, size_t cols); 

/**
 * Destroy a matrix 
 * Internally, the @ref mat_deinit is being called
 * Then the m pointer is free'd
 * Safe to call for m == nullptr
 *  
 * @pre @m must be created through @ref mat_create or @mat_slice
 *
 * @param m Pointer to the matrix to destroy
 */
inline void mat_destroy(Mat *m);

/**
 * Fill a matrix 
 * The function is safe to call with a nullptr
 *
 * @param m Matrix to fill
 * @param value Value to fill the matrix with
 */
void mat_fill(Mat *m, double value);

 /**
 * Create a submatrix (slice) of @p m and store it in @p out.
 * The slice starts at (row, col) and has dimensions nrows × ncols. Its
 * @es data pointer references a region within @p m->data; no new element
 * buffer is allocated.
 *
 * If @p out is NULL, a new Mat is allocated and returned. If @p out is not
 * NULL, it will be overwritten with the slice data. In both cases, the returned
 * value is the slice object.
 *
 * The slice does not own its data (an internal flag is set). It is safe to pass
 * the slice to @ref mat_deinit or @ref mat_destroy — these will reset the structure
 * but will not free the shared data. The source matrix @p m must remain valid for
 * the lifetime of the slice.
 *
 * @pre row < m->rows
 * @pre col < m->cols
 * @pre nrows > 0 && ncols > 0
 * @pre row + nrows <= m->rows
 * @pre col + ncols <= m->cols
 *
 * @param out    [out] Destination Mat object, or NULL to allocate a new one.
 * @param m      Source matrix.
 * @param row    Starting row index.
 * @param col    Starting column index.
 * @param nrows  Number of rows in the slice.
 * @param ncols  Number of columns in the slice.
 * @return       Pointer to the slice object, or NULL on failure.
 */
Mat *mat_slice(Mat *out, const Mat *m, size_t row, size_t col, size_t nrows, size_t ncols); 

/**
 * Create a view representing a single row from @p src and store it in @p out.
 *
 * The returned matrix has dimensions 1 × src->cols and its @c data pointer
 * references the row inside @p src->data; no new element buffer is allocated.
 *
 * If @p out is NULL, a new Mat is allocated and returned. If @p out is not
 * NULL, it will be overwritten with the row view. In both cases, the returned
 * value is the row object.
 *
 * The row view does not own its data (an internal flag is set). It is safe to
 * pass it to @ref mat_deinit or @ref mat_destroy — these will reset the structure
 * but will not free the shared data. The source matrix @p src must remain valid
 * for the lifetime of the row view.
 *
 * @pre row < src->rows
 *
 * @param out [out] Destination Mat object, or NULL to allocate a new one.
 * @param src Source matrix.
 * @param row Row index to extract.
 * @return    Pointer to the row view object, or NULL on failure.
 */
Mat *mat_row(Mat* out, const Mat *src, size_t row);

/**
 * Create a view representing a single column from @p src and store it in @p out.
 *
 * The returned matrix has dimensions src->rows × 1 and its @c data pointer
 * references the column inside @p src->data; no new element buffer is allocated.
 *
 * If @p out is NULL, a new Mat is allocated and returned. If @p out is not
 * NULL, it will be overwritten with the column view. In both cases, the returned
 * value is the column object.
 *
 * The column view does not own its data (an internal flag is set). It is safe to
 * pass it to @ref mat_deinit or @ref mat_destroy — these will reset the structure
 * but will not free the shared data. The source matrix @p src must remain valid
 * for the lifetime of the column view.
 *
 * @pre col < src->cols
 *
 * @param out [out] Destination Mat object, or NULL to allocate a new one.
 * @param src Source matrix.
 * @param col Column index to extract.
 * @return    Pointer to the column view object, or NULL on failure.
 */
Mat *mat_col(Mat *out, const Mat *src, size_t col);

/**
 * @brief Print a matrix with a given name
 * @param m Matrix to print
 * @param name Name to display before the matrix
 */
void mat_print(const Mat *m);

/**
 * Multiply matrix @p left by matrix @p right and store the result in @p left.
 *
 * Performs basic matrix multiplication: left = left × right.
 * The dimensions must be compatible:
 *   left->cols must equal right->rows.
 *
 * This function modifies @p out in-place.
 *
 * If @p left or @p right is NULL, the function returns NULL and no operation is performed.
 *
 * @pre left != NULL && right != NULL
 * @pre out must be properly initialized
 * @pre out->rows == left->rows && out->cols == right->cols 
 * @pre left->cols == right->rows
 *
 * @param out Matrix where the result is stored
 * @param left  Matrix to be multiplied
 * @param right Matrix to multiply with.
 * @return      Pointer to the updated @p left matrix, or NULL on error.
 */
Mat *mat_multiply(Mat *out, const Mat *left, const Mat *right);

/**
 * Multiply every element of matrix @p m by the scalar @p value.
 *
 * This operation modifies the matrix @p m in-place.
 *
 * @pre m != NULL
 *
 * @param m     Matrix to be scaled.
 * @param value Scalar multiplier.
 */
void mat_scalar(Mat *m, float value);

/**
 * Compute the element-wise sum of matrices @p dst and @p a, storing the result in @p dst.
 *
 * Both matrices must have the same dimensions.
 * The operation modifies @p dst in-place.
 *
 * @pre dst != NULL && a != NULL
 * @pre dst->rows == a->rows
 * @pre dst->cols == a->cols
 *
 * @param dst Destination matrix where the sum is stored.
 * @param a   Matrix to add to @p dst.
 * @return    0 on success, non-zero on error (e.g., dimension mismatch).
 */
[[nodiscard]]
int mat_sum(Mat *dst, const Mat *a);

/**
 * Compute the dot product of two matrices @p left and @p right.
 *
 * Both matrices must have the same dimensions.
 * The dot product is the sum of element-wise products and returns a single double value.
 *
 * @pre left != NULL && right != NULL
 * @pre left->rows == right->rows
 * @pre left->cols == right->cols
 *
 * @param left  First matrix.
 * @param right Second matrix.
 * @param result The variable in which the result is to be stored
 * @return 0 on success, non-zero on error (e.g dimension mismatch, invalid pointers) 
 */
[[nodiscard]]
int mat_dot(const Mat *left, const Mat *right, double *result); 

#endif // MAT_H_

