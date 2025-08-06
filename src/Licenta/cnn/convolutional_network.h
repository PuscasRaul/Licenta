#include <stddef.h>
#include <stdint.h>

#include "convolution.h"
#include "dense.h"
#include "pool.h"

#ifndef CNN_CONV_NETWORK_H
#define CNN_CONV_NETWORK_H

typedef enum {
  CONV_LAYER,
  POOL_LAYER,
  DENSE_LAYER
} Layer_Type;

typedef struct  {
  Layer_Type type;
  union {
    Convolution_Layer conv_layer;
    Pool_Layer pool_layer;
    Dense_Layer dense_layer;
  };
} Layer;

// WARN: First sketch, not meant to be a final version
typedef struct {
  size_t layer_count;
  Layer *layers; // i believe this is fine, i want the api similar to what
                 // tensorflow has, we create the Conv_Network object,
} Conv_Network;

#endif



