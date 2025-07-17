#ifndef UTILS_ARENA_H
#define UTILS_ARENA_H

#include <stddef.h>

#define ARENA_ALIGNOF(type) offsetof(struct {char c; type d}, d)

#ifndef ARENA_DEFAULT_ALIGNMENT
#define ARENA_DEFAULT_ALIGNMENT ARENA_ALIGNOF(size_t) 
#endif

typedef struct arena_allocation_s {
  size_t index;
  size_t size;
  char *pointer;
  struct arena_allocation_s *next;
} arena_allocation;

typedef struct {
  char *region;
  size_t index;
  size_t size;
} Arena;

#endif 
