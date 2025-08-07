#ifndef CNN_COMMON_H
#define CNN_COMMON_H

typedef enum {
  ACT_RELU,
  ACT_SIG,
} ACT_FUNC;

float sigmoidf(float x);
double sigmoidd(double x);
int sigmoidi(int x);
long sigmoidli(long x);

float reluf(float x); 
double relud(double x);
int relui(int x);
long reluli(long x);

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
