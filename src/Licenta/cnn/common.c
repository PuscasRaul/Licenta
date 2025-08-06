#include <math.h>

inline float sigmoidf(float x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

inline float reluf(float x) {
  return x > 0 ? x : x * 0.1f;
}
