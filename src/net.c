#include <stdlib.h>
#include <stdio.h>
#include "dubnet.h"

tensor forward_net(net m, tensor input)
{
    int i;
    tensor x = tensor_copy(input);
    for (i = 0; i < m.n; ++i) {
        layer *l = &m.layers[i];
        tensor y = l->forward(l, x);

        tensor_free(x);
        x = y;
    }
    return x;
}

void backward_net(net m, tensor d)
{
    tensor dy = tensor_copy(d);
    int i;
    for (i = m.n-1; i >= 0; --i) {
        layer *l = &m.layers[i];
        tensor dx = l->backward(l, dy);

        tensor_free(dy);
        dy = dx;
    }
    tensor_free(dy);
}

void update_net(net m, float rate, float momentum, float decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        layer *l = &m.layers[i];
        l->update(l, rate, momentum, decay);
    }
}

void free_layer(layer l)
{
    tensor_free(l.w);
    tensor_free(l.dw);
    tensor_free(l.b);
    tensor_free(l.db);
    tensor_free(l.x);
}

void free_net(net n)
{
    int i;
    for(i = 0; i < n.n; ++i){
        free_layer(n.layers[i]);
    }
    free(n.layers);
}

void file_error(char *filename)
{
    fprintf(stderr, "Couldn't open file %s\n", filename);
    exit(-1);
}

void save_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) tensor_write(l.b, fp);
        if(l.w.data) tensor_write(l.w, fp);
    }
    fclose(fp);
}

void load_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) tensor_read(l.b, fp);
        if(l.w.data) tensor_read(l.w, fp);
    }
    fclose(fp);
}
