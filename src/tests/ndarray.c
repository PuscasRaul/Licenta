#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../Licenta/math/ndarray.h"

void test_ndarray_createDestroy() {
    printf("--------->>CREATING TESTS<<---------\n");
    
    // Basic 2D array creation
    size_t dimensions[2] = {2, 3};
    ndarray *arr = ndarray_create(2, dimensions, FLOAT32);
    assert(arr);
    assert(arr->nd == 2);
    assert(arr->dimensions[0] == 2);
    assert(arr->dimensions[1] == 3);
    assert(arr->descr->dtype == FLOAT32);
    assert(arr->descr->item_size == 4);
    assert(arr->descr->owns_data == 1);
    
    // Test maximum allowed dimensions (NDARRAY_MAXDIMS)
    size_t dimensions_max[64];
    for (size_t i = 0; i < 64; i++) {
        dimensions_max[i] = 1;
    }
    ndarray *arr_max = ndarray_create(64, dimensions_max, FLOAT32);
    assert(arr_max);
    
    // Test exceeding maximum dimensions
    size_t dimensions_overflow[128];
    for (size_t i = 0; i < 128; i++) {
        dimensions_overflow[i] = 2;
    }
    ndarray *arr_overflow = ndarray_create(128, dimensions_overflow, FLOAT32);
    assert(!arr_overflow); // Should fail
    
    // Test single dimension
    size_t dimensions_1d[1] = {10};
    ndarray *arr_1d = ndarray_create(1, dimensions_1d, INT64);
    assert(arr_1d);
    assert(arr_1d->nd == 1);
    assert(arr_1d->descr->dtype == INT64);
    assert(arr_1d->descr->item_size == 8);
    
    // Test zero dimensions in array
    size_t dimensions_zero[3] = {2, 0, 3};
    ndarray *arr_zero = ndarray_create(3, dimensions_zero, FLOAT64);
    assert(arr_zero);
    assert(arr_zero->dimensions[1] == 0);
    
    // Test all data types
    ndarray *arr_f32 = ndarray_create(1, dimensions_1d, FLOAT32);
    ndarray *arr_f64 = ndarray_create(1, dimensions_1d, FLOAT64);
    ndarray *arr_i32 = ndarray_create(1, dimensions_1d, INT32);
    ndarray *arr_i64 = ndarray_create(1, dimensions_1d, INT64);
    
    assert(arr_f32 && arr_f32->descr->item_size == 4);
    assert(arr_f64 && arr_f64->descr->item_size == 8);
    assert(arr_i32 && arr_i32->descr->item_size == 4);
    assert(arr_i64 && arr_i64->descr->item_size == 8);
    
    printf("----->>CREATION TEST PASSED<<------\n");
    ndarray_debugPrint(arr);
    
    printf("--------->>DESTRUCTION TESTS<<---------\n");
    
    // Test normal destruction
    ndarray_destroy(arr);
    ndarray_destroy(arr_max);
    ndarray_destroy(arr_1d);
    ndarray_destroy(arr_zero);
    ndarray_destroy(arr_f32);
    ndarray_destroy(arr_f64);
    ndarray_destroy(arr_i32);
    ndarray_destroy(arr_i64);
    
    // Test destroying NULL pointer (should handle gracefully)
    ndarray_destroy(NULL);
    
    // Test destroying already failed creation
    ndarray_destroy(arr_overflow);
    
    printf("--------->>DESTRUCTION TESTS PASSED<<---------\n");
}

void test_ndarray_size() {
    printf("--------->>SIZE TESTS<<---------\n");
    
    // Test NULL array
    assert(ndarray_size(NULL) == 0);
    
    // Test basic 2D array
    size_t dimensions_2d[2] = {3, 4};
    ndarray *arr_2d = ndarray_create(2, dimensions_2d, FLOAT32);
    assert(ndarray_size(arr_2d) == 12);
    
    // Test 1D array
    size_t dimensions_1d[1] = {5};
    ndarray *arr_1d = ndarray_create(1, dimensions_1d, INT32);
    assert(ndarray_size(arr_1d) == 5);
    
    // Test 3D array
    size_t dimensions_3d[3] = {2, 3, 4};
    ndarray *arr_3d = ndarray_create(3, dimensions_3d, FLOAT64);
    assert(ndarray_size(arr_3d) == 24);
    
    // Test array with zero dimension
    size_t dimensions_zero[2] = {5, 0};
    ndarray *arr_zero = ndarray_create(2, dimensions_zero, INT64);
    assert(ndarray_size(arr_zero) == 0);
    
    // Test single element array
    size_t dimensions_single[3] = {1, 1, 1};
    ndarray *arr_single = ndarray_create(3, dimensions_single, FLOAT32);
    assert(ndarray_size(arr_single) == 1);
    
    printf("----->>SIZE TESTS PASSED<<------\n");
    
    ndarray_destroy(arr_2d);
    ndarray_destroy(arr_1d);
    ndarray_destroy(arr_3d);
    ndarray_destroy(arr_zero);
    ndarray_destroy(arr_single);
}

void test_ndarray_reshape() {
    printf("--------->>RESHAPE TESTS<<---------\n");
    
    // Test basic reshape (should use view_reshape)
    size_t dimensions[2] = {2, 6};
    ndarray *arr = ndarray_create(2, dimensions, FLOAT32);
    assert(arr);
    
    printf("Original array:\n");
    ndarray_debugPrint(arr);
    
    // Reshape to 3x4
    size_t new_dims_1[2] = {3, 4};
    ndarray *reshaped_1 = reshape(arr, 2, new_dims_1);
    assert(reshaped_1);
    assert(reshaped_1->nd == 2);
    assert(reshaped_1->dimensions[0] == 3);
    assert(reshaped_1->dimensions[1] == 4);
    
    printf("Reshaped to 3x4:\n");
    ndarray_debugPrint(reshaped_1);
    
    // Reshape to 1D
    size_t new_dims_2[1] = {12};
    ndarray *reshaped_2 = reshape(arr, 1, new_dims_2);
    assert(reshaped_2);
    assert(reshaped_2->nd == 1);
    assert(reshaped_2->dimensions[0] == 12);
    
    printf("Reshaped to 1D (12):\n");
    ndarray_debugPrint(reshaped_2);
    
    // Reshape to 3D
    size_t new_dims_3[3] = {2, 2, 3};
    ndarray *reshaped_3 = reshape(arr, 3, new_dims_3);
    assert(reshaped_3);
    assert(reshaped_3->nd == 3);
    assert(reshaped_3->dimensions[0] == 2);
    assert(reshaped_3->dimensions[1] == 2);
    assert(reshaped_3->dimensions[2] == 3);
    
    printf("Reshaped to 3D (2x2x3):\n");
    ndarray_debugPrint(reshaped_3);
    
    // Test invalid reshape (size mismatch)
    size_t invalid_dims[2] = {3, 5}; // 15 elements vs 12 original
    ndarray *invalid_reshape = reshape(arr, 2, invalid_dims);
    assert(!invalid_reshape); // Should fail
    
    // Test reshape with zero dimension
    size_t zero_dims[3] = {2, 0, 6};
    ndarray *zero_reshape = reshape(arr, 3, zero_dims);
    assert(!zero_reshape); // Should fail due to size mismatch
    
    // Test reshape of array with zero elements
    size_t zero_orig[2] = {3, 0};
    ndarray *zero_arr = ndarray_create(2, zero_orig, FLOAT32);
    size_t zero_new[1] = {0};

    ndarray *zero_reshaped = reshape(zero_arr, 1, zero_new);
    assert(zero_reshaped);
    assert(ndarray_size(zero_reshaped) == 0);
    
    printf("----->>RESHAPE TESTS PASSED<<------\n");
    
    ndarray_destroy(arr);
    ndarray_destroy(reshaped_1);
    ndarray_destroy(reshaped_2);
    ndarray_destroy(reshaped_3);
    ndarray_destroy(zero_arr);
    /*
    ndarray_destroy(zero_reshaped);
    */
}

void test_ndarray_strides() {
    printf("--------->>STRIDES TESTS<<---------\n");
    
    // Test 2D array strides
    size_t dimensions_2d[2] = {3, 4};
    ndarray *arr_2d = ndarray_create(2, dimensions_2d, FLOAT32);
    assert(arr_2d);
    
    // For FLOAT32 (4 bytes), 3x4 array:
    // strides[1] = 4 (item_size)
    // strides[0] = 4 * 4 = 16 (stride[1] * dimensions[1])
    assert(arr_2d->strides[1] == 4);
    assert(arr_2d->strides[0] == 16);
    
    // Test 3D array strides
    size_t dimensions_3d[3] = {2, 3, 4};
    ndarray *arr_3d = ndarray_create(3, dimensions_3d, FLOAT64);
    assert(arr_3d);
    
    // For FLOAT64 (8 bytes), 2x3x4 array:
    // strides[2] = 8
    // strides[1] = 8 * 4 = 32
    // strides[0] = 32 * 3 = 96
    assert(arr_3d->strides[2] == 8);
    assert(arr_3d->strides[1] == 32);
    assert(arr_3d->strides[0] == 96);
    
    // Test 1D array strides
    size_t dimensions_1d[1] = {10};
    ndarray *arr_1d = ndarray_create(1, dimensions_1d, INT32);
    assert(arr_1d);
    assert(arr_1d->strides[0] == 4); // INT32 size
    
    printf("----->>STRIDES TESTS PASSED<<------\n");
    
    ndarray_destroy(arr_2d);
    ndarray_destroy(arr_3d);
    ndarray_destroy(arr_1d);
}

void test_ndarray_edge_cases() {
    printf("--------->>EDGE CASE TESTS<<---------\n");
    
    // Test very large single dimension (within limits)
    size_t large_dims[1] = {1000000};
    ndarray *large_arr = ndarray_create(1, large_dims, FLOAT32);
    if (large_arr) { // May fail due to memory constraints
        assert(ndarray_size(large_arr) == 1000000);
        ndarray_destroy(large_arr);
    }
    
    // Test array with dimension of 1
    size_t ones_dims[4] = {1, 1, 1, 1};
    ndarray *ones_arr = ndarray_create(4, ones_dims, INT64);
    assert(ones_arr);
    assert(ndarray_size(ones_arr) == 1);
    
    // Test reshape of 1-element array
    size_t new_ones[2] = {1, 1};
    ndarray *ones_reshaped = reshape(ones_arr, 2, new_ones);
    assert(ones_reshaped);
    assert(ndarray_size(ones_reshaped) == 1);
    
    // Test multiple reshapes
    size_t reshape_dims1[3] = {2, 3, 2};
    size_t reshape_dims2[2] = {4, 3};
    size_t reshape_dims3[1] = {12};
    
    ndarray *multi_arr = ndarray_create(3, reshape_dims1, FLOAT32);
    ndarray *multi_r1 = reshape(multi_arr, 2, reshape_dims2);
    ndarray *multi_r2 = reshape(multi_r1, 1, reshape_dims3);
    
    assert(multi_arr && multi_r1 && multi_r2);
    assert(ndarray_size(multi_arr) == ndarray_size(multi_r1));
    assert(ndarray_size(multi_r1) == ndarray_size(multi_r2));
    assert(ndarray_size(multi_r2) == 12);
    
    printf("----->>EDGE CASE TESTS PASSED<<------\n");
    
    ndarray_destroy(ones_arr);
    ndarray_destroy(ones_reshaped);
    ndarray_destroy(multi_arr);
    ndarray_destroy(multi_r1);
    ndarray_destroy(multi_r2);
}

int main(void) {
    printf("Starting NDArray Test Suite...\n\n");
    
    test_ndarray_createDestroy();
    printf("\n");
    
    test_ndarray_size();
    printf("\n");
    
    test_ndarray_strides();

    test_ndarray_reshape();
    printf("\n");
    
    test_ndarray_edge_cases();
    printf("\n");
    
    printf("All tests completed successfully!\n");
    return 0;
}
