#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "dubnet.h"
#include "tensor.h"

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    float max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

float accuracy_net(net m, data d)
{
    tensor p = forward_net(m, d.x);
    int i;
    int correct = 0;
    for (i = 0; i < d.y.size[0]; ++i) {
        tensor guess = tensor_get_(p, i);
        tensor truth = tensor_get(d.y, i);
        size_t len = tensor_len(guess);
        if (max_index(guess.data, len) == max_index(truth.data, len)) ++correct;
    }
    tensor_free(p);
    return (float)correct / d.y.size[0];
}

float cross_entropy_loss(tensor x, tensor y)
{
    assert(x.n == y.n);
    size_t i;
    for(i = 0; i < x.n; ++i) assert(x.size[i] == y.size[i]);
    size_t len = tensor_len(x);
    float sum = 0;
    for(i = 0; i < len; ++i){
        sum += -y.data[i]*log(x.data[i]);
    }
    return sum/y.size[0];
}

tensor cross_entropy_derivative(tensor x, tensor y)
{
    return tensor_sub(x, y);
}

void train_image_classifier(net m, data d, int batch, int iters, float rate, float momentum, float decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        tensor yhat = forward_net(m, b.x);
        float err = cross_entropy_loss(yhat, b.y);
        tensor dy = cross_entropy_derivative(yhat, b.y);
        fprintf(stderr, "%06d: Loss: %f\n", e, err);
        backward_net(m, dy);
        update_net(m, rate/batch, momentum, decay);
        free_data(b);
        tensor_free(yhat);
        tensor_free(dy);
    }
}
