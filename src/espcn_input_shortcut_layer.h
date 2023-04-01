#ifndef SHORTCUT_INPUT_LAYER_H
#define SHORTCUT_INPUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shortcut_input_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_input_layer(const layer l, network net);
void backward_shortcut_input_layer(const layer l, network net);
void resize_shortcut_input_layer(layer *l, int w, int h);

#ifdef GPU
void forward_shortcut_input_layer_gpu(const layer l, network net);
void backward_shortcut_input_layer_gpu(const layer l, network net);
#endif

#endif
