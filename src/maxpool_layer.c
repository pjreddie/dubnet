#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dubnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
tensor forward_maxpool_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    assert(x.n == 4);

    tensor y = tensor_vmake(4,
        x.size[0],  // same # data points and # of channels (N and C)
        x.size[1],
        (x.size[2]-1)/l->stride + 1, // H and W scaled based on stride
        (x.size[3]-1)/l->stride + 1);

    // This might be a useful offset...
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    return y;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
tensor backward_maxpool_layer(layer *l, tensor dy)
{
    tensor x    = l->x;
    tensor dx = tensor_make(x.n, x.size);
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer *l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(size_t size, size_t stride)
{
    layer l = {0};
    l.size = size;
    l.stride = stride;
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

