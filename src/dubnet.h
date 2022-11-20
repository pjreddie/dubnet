// Include guards and C++ compatibility
#ifndef DUBNET_H
#define DUBNET_H
#include <stdio.h>
#include "tensor.h"
#include "image.h"
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

    size_t size;
    size_t stride;
    size_t pad;

    tensor  (*forward)  (struct layer *, struct tensor);
    tensor  (*backward) (struct layer *, struct tensor);
    void   (*update)   (struct layer *, float rate, float momentum, float decay);
} layer;

layer make_connected_layer(int inputs, int outputs);
layer make_activation_layer(ACTIVATION activation);
layer make_convolutional_layer(size_t c, size_t n, size_t size, size_t stride, size_t pad);
layer make_maxpool_layer(size_t size, size_t stride);
layer make_batchnorm2d_layer(int c);

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
tensor image_to_tensor(image im);

tensor im2col(tensor im, size_t size_y, size_t size_x, size_t stride, size_t pad);
tensor col2im(tensor col, size_t c, size_t h, size_t w, size_t size_y, size_t size_x, size_t stride, size_t pad);


tensor mean2d(tensor x);
tensor variance2d(tensor x, tensor m);
tensor normalize2d(tensor x, tensor m, tensor v);
tensor delta_mean2d(tensor dy, tensor v);
tensor delta_variance2d(tensor dy, tensor x, tensor m, tensor v);
tensor delta_batchnorm2d(tensor dy, tensor dm, tensor dv, tensor m, tensor v, tensor x);

void tensor_write(tensor t, FILE *fp);
void tensor_read(tensor t, FILE *fp);
void tensor_save(tensor t, char *fname);
tensor tensor_load(char *fname);

#ifdef __cplusplus
}
#endif
#endif
