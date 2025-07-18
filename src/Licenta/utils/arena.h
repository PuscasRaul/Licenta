// TODO: arena buffer pool with fragmentation control (for fun)
// most used, least used, most recent, different policies

#ifndef UTILS_ARENA_H
#define UTILS_ARENA_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct Region Region;

#ifndef DEFAULT_REGION_SIZE
#define DEFAULT_REGION_SIZE (16*1024)
#endif

struct Region {
  Region *next;
  size_t capacity;
  size_t offset;
  char *data; 
};

Region *region_alloc(size_t capacity);
void region_free(Region *r);

typedef struct {
  Region *head, *tail;
} Arena;

void *arena_alloc(Arena *a, size_t s_bytes);
void *arena_realloc(Arena *a, void *oldptr, size_t oldsz, size_t newsz);
void arena_reset(Arena *a);
void arena_free(Arena *a);
void arena_debug_dump(Arena *a);

#endif 

#ifdef UTILS_ARENA_IMPLEMENTATION

Region *region_alloc(size_t capacity) {
  if (capacity <= 0)
    return NULL;
  size_t bytes_size = sizeof(Region) + sizeof(char) * capacity;
  Region *r = malloc(bytes_size);
  if (!r)
    return NULL;

  r->next = NULL;
  r->capacity = capacity;
  r->offset = 0;
  r->data = (char *)r + sizeof(Region);

  return r;
}

void region_free(Region *r) {
  if (r)
    free(r);
}

void *arena_alloc(Arena *a, size_t s_bytes) {
  size_t size = (s_bytes + sizeof(char) - 1) / sizeof(char);

  if (a->tail == NULL) {
    if (a->head) return NULL; // the arena is not properly initialized
    size_t capacity = DEFAULT_REGION_SIZE;
    if (capacity < size) capacity = size;
    a->tail = region_alloc(capacity);
    a->head = a->tail;
  }

  // untill we find a region with enough memory
  while (a->tail->offset+size > a->tail->capacity && a->tail->next != NULL) {
    a->tail =  a->tail->next;
  }

  if (a->tail->offset + size > a->tail->capacity) {
    if (a->tail->next) return NULL;
    size_t capacity = DEFAULT_REGION_SIZE;
    if (capacity < size) capacity = size;
    a->tail->next = region_alloc(capacity);
    a->tail = a->tail->next;
  }

  void *result = &a->tail->data[a->tail->offset];
  a->tail->offset += size;
  return result;
}

void *arena_realloc(Arena *a, void *oldptr, size_t oldsz, size_t newsz) {
  if (newsz <= oldsz) return oldptr;
  void *newptr = arena_alloc(a, newsz);
  char *newptr_char = (char*)newptr;
  char *oldptr_char = (char*)oldptr;
  for (size_t i = 0; i < oldsz; ++i)
    newptr_char[i] = oldptr_char[i];

  return newptr;
}

void *arena_memcpy(void *dest, const void *src, size_t n) {
  char *dest_char = (char*) dest;
  char *src_char = (char*) src;
  for (; n; --n) *dest_char++ = *src_char++;
  return dest;
}

void arena_reset(Arena *a) {
  for (Region *r = a->head; r; r = r->next) {
    r->offset = 0;
  }
  a->tail = a->head;
}

void arena_free(Arena *a) {
  Region *r = a->head;
  Region *previous = a->head;
  while (r) {
    previous = r;
    r = r->next;
    region_free(previous);
  }
  a->head = NULL;
  a->tail = NULL;
}

void arena_debug_dump(Arena *a) {
  printf("-------------------------------------------------------\n");
  printf("Dump of arena at address %p\n", a);
  if (a == NULL) {
    printf(" It's NULL!\n");
    printf("-------------------------------------------------------\n");
    fflush(stdout);
    return;
  }

  printf("-------->REGIONS<-------\n");
  for (Region *r = a->head; r; r = r->next) {
    printf("Dump of region at address %p\n", r);
    printf("capacity     :    %ld\n", r->capacity);
    printf("size     :    %ld\n", r->offset);
    printf("------------\n");
  }
}

#endif
