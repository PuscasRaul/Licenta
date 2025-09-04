#include <math.h>

#ifndef CNN_COMMON_H
#define CNN_COMMON_H

typedef enum {
  ACT_RELU,
  ACT_SIG,
} ACT_FUNC;

static inline float sigmoidf(float x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

static inline double sigmoidd(double x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

static inline int sigmoidi(int x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

static inline long sigmoidli(long x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

static inline float reluf(float x) {
  return x > 0 ? x : x * 0.1f;
}

static inline double relud(double x) {
  return x > 0 ? x : x * 0.1f;
}

static inline int relui(int x) {
  return x > 0 ? x : x * 0.1f;
}

static inline long reluli(long x) {
  return x > 0 ? x : x * 0.1f;
}

#define sigmoid(X) \
_Generic((X), \
    float: sigmoidf, \
    double: sigmoidd, \
    int: sigmoidi, \
    long: sigmoidli, \
    default: sigmoidd \
  )(X)

#define relu(X) \
_Generic((X), \
    float: reluf, \
    double: relud, \
    int: relui, \
    long: reluli, \
    default: relud \
  )(X)

#endif // CNN_COMMON_h
