#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"

// Take mean of tensor x over rows and spatial dimension
// tensor x: tensor with data
// int groups: number of distinct means to take, usually equal to # outputs
// after connected layers or # channels after convolutional layers
// returns: (1 x groups) tensor with means
tensor mean2d(tensor x)
{
    int n = x.size[0];
    int c = x.size[1];
    int h = x.size[2];
    int w = x.size[3];
    tensor m = tensor_vmake(1, c);

    // TODO: 7.0 - Calculate mean - Already done!

    int i, j, k, b;
    for(b = 0; b < n; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    m.data[k] += x.data[i + w*(j + h*(k + b*c))];
                }
            }
        }
    }
    tensor_scale_(1.0/(n*h*w), m);
    return m;
}

// Take variance over tensor x given mean m
tensor variance2d(tensor x, tensor m)
{
    int n = x.size[0];
    int c = x.size[1];
    int h = x.size[2];
    int w = x.size[3];
    tensor v = tensor_vmake(1, c);

    // TODO: 7.1 - Calculate variance

    return v;
}

// Normalize x given mean m and variance v
// returns: y = (x-m)/sqrt(v + epsilon)
tensor normalize2d(tensor x, tensor m, tensor v)
{
    int n = x.size[0];
    int c = x.size[1];
    int h = x.size[2];
    int w = x.size[3];
    tensor y = tensor_make(x.n, x.size);

    // TODO: 7.2 - Normalize x

    return y;
}


// Run an batchnorm2d layer on input
// layer l: pointer to layer to run
// tensor x: input to layer
// returns: the result of running the layer y = (x - mu) / sigma
tensor forward_batchnorm2d_layer(layer *l, tensor x)
{
    assert(x.n == 4);
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);
    tensor rolling_mean = tensor_get_(l->w, 0);
    tensor rolling_variance = tensor_get_(l->w, 1);

    if(x.size[0] == 1){
        return normalize2d(x, rolling_mean, rolling_variance);
    }

    float s = 0.1;
    tensor m = mean2d(x);
    tensor v = variance2d(x, m);
    tensor y = normalize2d(x, m, v);

    tensor_scale_(1-s, rolling_mean);
    tensor_axpy_(s, m, rolling_mean);
    tensor_scale_(1-s, rolling_variance);
    tensor_axpy_(s, v, rolling_variance);

    tensor_free(m);
    tensor_free(v);

    return y;
}

tensor delta_mean2d(tensor dy, tensor v)
{
    int n = dy.size[0];
    int c = dy.size[1];
    int h = dy.size[2];
    int w = dy.size[3];
    tensor dm = tensor_vmake(1, c);

    // TODO: 7.3

    return dm;
}


tensor delta_variance2d(tensor dy, tensor x, tensor m, tensor v)
{
    int n = dy.size[0];
    int c = dy.size[1];
    int h = dy.size[2];
    int w = dy.size[3];
    tensor dv = tensor_vmake(1, c);

    // TODO 7.4 - Calculate dL/dv

    return dv;
}

tensor delta_batchnorm2d(tensor dy, tensor dm, tensor dv, tensor m, tensor v, tensor x)
{
    int n = dy.size[0];
    int c = dy.size[1];
    int h = dy.size[2];
    int w = dy.size[3];
    tensor dx = tensor_make(dy.n, dy.size);

    int num = n * h * w;

    // TODO 7.4 - Calculate dL/dv

    return dx;
}


// Run an batchnorm2d layer on input
// layer l: pointer to layer to run
// tensor dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
tensor backward_batchnorm2d_layer(layer *l, tensor dy)
{
    tensor x = l->x;

    tensor m = mean2d(x);
    tensor v = variance2d(x, m);

    tensor dm = delta_mean2d(dy, v);
    tensor dv = delta_variance2d(dy, x, m, v);
    tensor dx = delta_batchnorm2d(dy, dm, dv, m, v, x);

    tensor_free(m);
    tensor_free(v);
    tensor_free(dm);
    tensor_free(dv);

    return dx;
}

// Update batchnorm2d layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_batchnorm2d_layer(layer *l, float rate, float momentum, float decay){}

layer make_batchnorm2d_layer(int c)
{
    layer l = {0};

    l.w = tensor_vmake(2, 2, c);

    l.forward = forward_batchnorm2d_layer;
    l.backward = backward_batchnorm2d_layer;
    l.update = update_batchnorm2d_layer;
    return l;
}
