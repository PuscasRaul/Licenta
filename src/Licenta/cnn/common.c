#include <math.h>

float sigmoidf(float x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

double sigmoidd(double x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

int sigmoidi(int x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

long sigmoidli(long x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

float reluf(float x) {
  return x > 0 ? x : x * 0.1f;
}

double relud(double x) {
  return x > 0 ? x : x * 0.1f;
}

int relui(int x) {
  return x > 0 ? x : x * 0.1f;
}

long reluli(long x) {
  return x > 0 ? x : x * 0.1f;
}
