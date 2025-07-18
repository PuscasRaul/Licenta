#define UTILS_ARENA_IMPLEMENTATION
#include "../Licenta/utils/arena.h"  

#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_arena_alloc_basic() {
  Arena a = {0};
  int *x = arena_alloc(&a, sizeof(int));
  assert(x != NULL);
  *x = 42;
  printf("test_arena_alloc_basic passed: %d\n", *x);
  arena_free(&a);
}

void test_arena_alloc_multiple() {
  Arena a = {0};
  const int N = 1000;
  int *nums[N];
  for (int i = 0; i < N; ++i) {
    nums[i] = arena_alloc(&a, sizeof(int));
    assert(nums[i] != NULL);
    *nums[i] = i;
  }
  for (int i = 0; i < N; ++i) {
    assert(*nums[i] == i);
  }
  printf("test_arena_alloc_multiple passed\n");
  arena_free(&a);
}

void test_arena_realloc() {
  Arena a = {0};
  char *msg = arena_alloc(&a, 6);
  strcpy(msg, "Hello");
  char *bigger = arena_realloc(&a, msg, 6, 12);
  strcat(bigger, ", AI!");
  assert(strcmp(bigger, "Hello, AI!") == 0);
  printf("test_arena_realloc passed: %s\n", bigger);
  arena_free(&a);
}

void test_arena_reset() {
  Arena a = {0};
  int *x = arena_alloc(&a, sizeof(int));
  *x = 1337;
  arena_reset(&a);
  int *y = arena_alloc(&a, sizeof(int));
  assert(y != NULL);
  *y = 2025;
  assert(*y == 2025);
  printf("test_arena_reset passed: %d\n", *y);
  arena_free(&a);
}

void test_zero_alloc() {
  Arena a = {0};
  void *ptr = arena_alloc(&a, 0);
  assert(ptr != NULL);  // We allow 0-byte allocs to succeed
  printf("test_zero_alloc passed\n");
  arena_free(&a);
}

void test_debug_dump() {
  Arena a = {0};
  for (int i = 0; i < 10; ++i)
    arena_alloc(&a, 1024);
  printf("arena_debug_dump output:\n");
  arena_debug_dump(&a);
  arena_free(&a);
}

int main(void) {
  test_arena_alloc_basic();
  test_arena_alloc_multiple();
  test_arena_realloc();
  test_arena_reset();
  test_zero_alloc();
  test_debug_dump();

  printf("All tests passed.\n");
  return 0;
}

