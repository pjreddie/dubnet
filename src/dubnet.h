// Include guards and C++ compatibility
#ifndef DUBNET_H
#define DUBNET_H
#include <stdio.h>
#include "tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

// Layer and network definitions

// The kinds of activations our framework supports
typedef enum{LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX} ACTIVATION;

typedef struct layer {
    tensor x;

    // Weights
    tensor w;
    tensor dw;

    // Biases
    tensor b;
    tensor db;

    ACTIVATION activation;

    tensor  (*forward)  (struct layer *, struct tensor);
    tensor  (*backward) (struct layer *, struct tensor);
    void   (*update)   (struct layer *, float rate, float momentum, float decay);
} layer;

layer make_connected_layer(int inputs, int outputs);
layer make_activation_layer(ACTIVATION activation);
layer make_convolutional_layer(int filters, int size, int stride);
layer make_maxpool_layer(int size, int stride);
layer make_batchnorm_layer();

typedef struct {
    int n;
    layer *layers;
} net;

tensor forward_net(net m, tensor x);
void backward_net(net m, tensor d);
void update_net(net m, float rate, float momentum, float decay);
void free_layer(layer l);
void free_net(net n);


typedef struct{
    tensor x;
    tensor y;
} data;

data random_batch(data d, int n);
data load_image_classification_data(char *images, char *label_file);
void free_data(data d);
void train_image_classifier(net m, data d, int batch, int iters, float rate, float momentum, float decay);
float accuracy_net(net m, data d);

void tensor_write(tensor t, FILE *fp);
void tensor_read(tensor t, FILE *fp);
void tensor_save(tensor t, char *fname);
tensor tensor_load(char *fname);

#ifdef __cplusplus
}
#endif
#endif
