#include <cstddef>

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif // NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.1f
#endif // NN_RELU_PARAM

typedef enum {
  ACT_RELU,
  ACT_SIG,
  ACT_SIN,
  ACT_TANH
} Act;

typedef enum {
  LOSS_MSE,
  LOSS_BCE,
  LOSS_CE
} Loss;

typedef enum {
  LAYER_CONV,
  LAYER_POOL,
  LAYER_DENSE,
} layer_type;

typedef struct {
  size_t input_size;
  size_t output_size;
  Act act_func;
  Mat ws;
  Mat bs;
  Mat as;
} dense_layer;

typedef struct {
  size_t in_width, in_length, in_depth;
  size_t pool_size;
  size_t stride;
  size_t out_width, out_length, out_depth;
} pool_layer;

typedef struct {
  size_t in_width, in_length, in_depth; // input size
  size_t filter_size; // use square filters
  size_t filter_count;
  size_t stride;
  Mat *filters;
} conv_layer;

typedef struct {
  layer_type type;
  union {
    dense_layer dense;
    conv_layer conv;
    pool_layer pool;
  };
} CNN_layer;

typedef struct {
  size_t layer_count;
  CNN_layer *layers;
  float learning_rate;
  Loss loss_function;
} CNN;
