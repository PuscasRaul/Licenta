#include "convolutional_network.h"

// FUNCTION PROTOTYPE, just sketching it out
Mat *convolution(Mat *input, conv_layer *layer) {

  size_t f_size = layer->filter_size;
  size_t width = layer->in_width - f_size;
  size_t length = layer->in_length - f_size; // this ensures
                                             // no overflows 
  size_t depth = layer->in_depth;
  size_t filters = layer->filter_count;
  
  // at the end, each dot product becomes one cell over the said filter
  // are the results averaged?


  for (size_t f = 0; f < filters; f++) {
    // THIS IS FOR 1 FILTER, wrap it all up in a for over filters
    // go depth-wise first
      // we need to take slices of the input matrix, on the current channel
      // of the size filter_size x filter_size
      // so first of all the loop would be like:
      for (size_t w = 0; w < width; w++) {
        for (size_t l = 0; l < length; l++) {
          for (size_t d = 0; d < depth; d++) {
            Mat slice = mat_slice(&input[d], w, l, f_size, f_size);
            Mat result; 
            mat_init(&result, f_size, f_size);
            mat_multiply(&result, &slice, &layer->filters[f]);
        }
        // at the end, somehow add them all up to create 1 cell
        // which i believe means that the depth row, should be moved inwards
      }
    }
  }
}
