// Include guards and C++ compatibility
#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensor {
    size_t n;
    size_t *size;
    float *data;
} tensor;

tensor tensor_make(const size_t n, const size_t *size);
tensor tensor_vmake(const size_t n, ...);

tensor tensor_copy(tensor t);

tensor tensor_scale(float s, tensor t);
void tensor_scale_(float s, tensor t);
void tensor_axpy_(float a, tensor x, tensor y);

tensor tensor_random(const float s, const size_t n, const size_t *size);
tensor tensor_vrandom(const float s, const size_t n, ...);
void   tensor_free(tensor t);

tensor tensor_get(const tensor t, const size_t e);
tensor tensor_get_(const tensor t, const size_t e);
size_t    tensor_len(const tensor t);

tensor tensor_view(tensor t, const size_t n, const size_t* size);
tensor tensor_vview(tensor t, const size_t n, ...);

void tensor_print(tensor t);

int tensor_broadcastable(tensor a, tensor b);
tensor tensor_add(tensor a, tensor b);
tensor tensor_sub(tensor a, tensor b);
tensor tensor_mul(tensor a, tensor b);
tensor tensor_div(tensor a, tensor b);

tensor tensor_sum_dim(tensor a, size_t dim);
float tensor_sum(tensor a);

void tensor_save(tensor t, char *fname);
tensor tensor_load(char *fname);
tensor matrix_load(char *fname);

#ifdef __cplusplus
}
#endif
#endif
