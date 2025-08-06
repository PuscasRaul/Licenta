#ifndef CNN_COMMON_H
#define CNN_COMMON_H

typedef enum {
  ACT_RELU,
  ACT_SIG,
} ACT_FUNC;

float sigmoidf(float x);
float reluf(float x); 

#endif // CNN_COMMON_H
