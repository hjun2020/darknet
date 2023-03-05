#ifndef ESPCN_LAYER_H
#define ESPCN_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer espcn_layer;
typedef layer convolutional_layer;

#ifdef GPU
void forward_espcn_layer_gpu(espcn_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

// void push_convolutional_layer(convolutional_layer layer);
// void pull_convolutional_layer(convolutional_layer layer);

// void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
// void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
// void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
// void cudnn_convolutional_setup(layer *l);
#endif
#endif

espcn_layer make_espcn_layer(int batch, int h, int w, int c, int n, int groups);
void forward_espcn_layer(const espcn_layer layer, network net);
void update_espcn_layer(espcn_layer layer, update_args a);
image *visualize_convolutional_layer(espcn_layer layer, char *window, image *prev_weights);


void backward_espcn_layer(espcn_layer layer, network net);

int espcn_out_height(espcn_layer layer);
int espcn_out_width(espcn_layer layer);

void espcn_cpu(float* data_im,
     int channels,  int scale, int height,  int width, float* data_col);
void reverse_espcn_cpu(float* data_im,
     int channels,  int scale, int height,  int width, float* data_col);


#endif