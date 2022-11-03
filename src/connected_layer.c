#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"
#include "matrix.h"

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = xw+b
tensor forward_connected_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    // turn x into matrix if it isn't (this is kind gross but has to be done)
    x = tensor_vview(x, 2, x.size[0], tensor_len(x)/x.size[0]);

    // TODO: 3.0 - run the network forward
    tensor y = tensor_make(0, 0);

    tensor_free(x);
    return y;
}

// Run a connected layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
tensor backward_connected_layer(layer *l, tensor dy)
{
    tensor x = tensor_vview(l->x, 2, l->x.size[0], tensor_len(l->x)/l->x.size[0]);

    // TODO: 3.1
    // Calculate the gradient dL/db for the bias terms using backward_bias
    // add this into any stored gradient info already in l.db


    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw


    // Calculate dL/dx and return it
    tensor dx = tensor_copy(l->x);

    
    // Don't remove this, just make sure your gradients are in `dx`
    // In the case that we flattened `x` in forward pass we have to
    // *unflatten* `dx` to be the same shape as `x`.
    tensor dxv = tensor_view(dx, l->x.n, l->x.size);
    tensor_free(x);
    tensor_free(dx);
    return dxv;
}

// Update weights and biases of connected layer
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_connected_layer(layer *l, float rate, float momentum, float decay)
{
    // TODO: 3.2
    // Apply our updates using our SGD update rule
    // assume  l.dw = dL/dw - momentum * update_prev
    // we want l.dw = dL/dw - momentum * update_prev + decay * w
    // then we update l.w = l.w - rate * l.dw
    // lastly, l.dw is the negative update (-update) but for the next iteration
    // we want it to be (-momentum * update) so we just need to scale it a little


    // Do the same for biases as well but no need to use weight decay on biases
}

layer make_connected_layer(int inputs, int outputs)
{
    layer l = {0};
    l.w  = tensor_vrandom(sqrtf(2.f/inputs), 2, inputs, outputs);
    l.dw = tensor_vmake(2, inputs, outputs);
    l.b  = tensor_vmake(2, 1, outputs);
    l.db = tensor_vmake(2, 1, outputs);
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

