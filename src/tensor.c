#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include "tensor.h"

// Make a tensor with the specified dimension and size
// size_t n: number of dimensions
// size_t* size: size of each dimension
// returns: a tensor of specified size filled with zeroes
tensor tensor_make(const size_t n, const size_t *size)
{
    tensor t = {0};
    t.n = n;
    t.size = n > 0 ? calloc(n, sizeof(size_t)) : 0;
    size_t i;
    for(i = 0; i < n; ++i){
        t.size[i] = size[i];
    }
    size_t len = tensor_len(t);
    t.data = calloc(len, sizeof(float));
    return t;
}

// Variadic version of make, supply dimension and variable number of sizes
// size_t n: number of dimensions
// vargs: list of size_t sizes for dimensions
// returns: a tensor of specified size filled with zeroes
tensor tensor_vmake(const size_t n, ...)
{
    size_t *size = calloc(n, sizeof(size_t));
    va_list args;
    va_start(args, n);
    size_t i;
    for(i = 0; i < n; ++i){
        size[i] = va_arg(args, size_t);
    }
    tensor t  = tensor_make(n, size);
    free(size);
    return t;
}

// Total number of elements in tensor
// tensor t:
// returns: length of t
size_t tensor_len(const tensor t)
{
    size_t i;
    size_t len = 1;
    for(i = 0; i < t.n; ++i){
        len *= t.size[i];
    }
    return len;
}

// Copy a tensor
// tensor t: tensor to be copied
// returns: a copy of t
tensor tensor_copy(tensor t)
{
    // TODO 0.0: copy the tensor and return the copy
    tensor c = tensor_make(0, 0);
    return c;
}

// In-place scaling of tensor
// float s: scalar factor
// tensor t: tensor to scale in place
void tensor_scale_(float s, tensor t)
{
    // TODO 0.1: scale the tensor in place
}

// Scaling of tensor
// float s: scalar factor
// tensor t: tensor to scale
// returns: new tensor equal to s*t
tensor tensor_scale(float s, tensor t)
{
    tensor c = tensor_copy(t);
    tensor_scale_(s, c);
    return c;
}

// Perform computation y = ax + y (in-place)
// float a: scalar factor
// tensor x: tensor to be scaled
// tensor y: tensor to be added into
void tensor_axpy_(float a, tensor x, tensor y)
{
    assert(tensor_len(x) == tensor_len(y));
    // TODO 0.2: perform the elementwise, in-place computation
}

// Returns a new dimensionality view of a tensor
// input must have same total number of elements as reshaped tensor
// tensor t: tensor to reshape
// size_t n: new dimensionality
// size_t *sizes: new sizes for the dimensions
// returns: reshaped tensor
tensor tensor_view(tensor t, const size_t n, const size_t* size)
{
    tensor v = tensor_make(n, size);
    assert(tensor_len(t) == tensor_len(v));
    memcpy(v.data, t.data, tensor_len(t)*sizeof(float));
    return v;
}

// Variadic version of tensor_view
// tensor t: tensor to reshape
// size_t n: new dimensionality
// vargs: new sizes for the dimensions
// returns: reshaped tensor
tensor tensor_vview(tensor t, const size_t n, ...)
{
    size_t *size = calloc(n, sizeof(size_t));
    va_list args;
    va_start(args, n);
    size_t i;
    for(i = 0; i < n; ++i){
        size[i] = va_arg(args, size_t);
    }
    tensor v  = tensor_view(t, n, size);
    free(size);
    return v;
}

// Make a random tensor in interval [-s, s]
// float s: bounds for random generation
// size_t n: number of dimensions
// size_t* size: size of each dimension
// returns: a tensor filled with uniform distribution [-s, s]
tensor tensor_random(const float s, const size_t n, const size_t *size)
{
    tensor t = tensor_make(n, size);
    size_t len = tensor_len(t);
    size_t i;
    for(i = 0; i < len; ++i){
        t.data[i] = 2*s*((float)rand()/RAND_MAX) - s;
    }
    return t;
}

// Variadic version of tensor_random
// float s: bounds for random generation
// size_t n: number of dimensions
// size_t* size: size of each dimension
// returns: a tensor filled with uniform distribution [-s, s]
tensor tensor_vrandom(const float s, const size_t n, ...)
{
    size_t *size = calloc(n, sizeof(size_t));
    va_list args;
    va_start(args, n);
    size_t i;
    for(i = 0; i < n; ++i){
        size[i] = va_arg(args, size_t);
    }
    tensor t  = tensor_random(s, n, size);
    free(size);
    return t;
}

// Returns sub-tensor (in-place)
// tensor t: tensor to access
// size_t e: sub-tensor in zeroeth dimension to get
// returns: sub-tensor containing t[0]
tensor tensor_get_(const tensor t, const size_t e)
{
    assert (e >= 0 && (e == 0 || e < t.size[0]));
    if (t.n == 0) return t;
    tensor a = {0};
    a.n = t.n - 1;
    a.size = t.size + 1;
    size_t len = tensor_len(a);
    a.data = t.data + e*len;
    return a;
}

// Returns sub-tensor (copy)
// tensor t: tensor to access
// size_t e: sub-tensor in zeroeth dimension to get
// returns: copy of sub-tensor containing t[0]
tensor tensor_get(const tensor t, const size_t e)
{
    assert (e >= 0 && (e == 0 || e < t.size[0]));
    return tensor_copy(tensor_get_(t, e));
}

// Free storage of a tensor
void tensor_free(tensor t)
{
    if(t.size) free(t.size);
    if(t.data) free(t.data);
}

// This was so hard you don't even know...
void tensor_print_(tensor t, size_t d)
{
    size_t i, j;
    if(t.n == 0){
        printf("%6.3f", t.data[0]);
    } else {
        printf("[");
        for(i = 0; i < t.size[0]; ++i){
            if(i > 0 && t.n > 1){
                for(j = 0; j < t.n-1; ++j) printf("\n");
                for(j = 0; j <= d; ++j) printf(" ");
            }
            tensor g = tensor_get_(t, i);
            tensor_print_(g, d+1);
            if(i < t.size[0]-1) printf(",");
        }
        printf("]");
    }
}

void tensor_print(tensor t)
{
    size_t i;
    printf("Dim = %ld, Size = (", t.n);
    for(i = 0; i < t.n; ++i) printf("%ld%s", t.size[i], (i<t.n-1)?", ":"");
    printf(")\n");
    tensor_print_(t, 0);
    printf("\n");
}


int tensor_broadcastable(tensor a, tensor b)
{
    size_t ln = (a.n < b.n) ? a.n : b.n;
    size_t i;
    for(i = 0; i < ln; ++i){
        size_t sa = a.size[a.n - 1 - i];
        size_t sb = b.size[b.n - 1 - i];
        if (sa != 1 && sb != 1 && sa != sb) return 0;
    }
    return 1;
}

tensor tensor_broadcast(tensor a, tensor b)
{
    if (!tensor_broadcastable(a, b)){
        fprintf(stderr, "Can't broadcast tensors\n");
        tensor none = {0};
        return none;
    }
    if(a.n < b.n){
        tensor swap = a;
        a = b;
        b = swap;
    }

    size_t n  = a.n;
    size_t ln = b.n;

    size_t *size = calloc(n, sizeof(size_t));
    size_t i;
    for(i = 0; i < ln; ++i){
        size_t sa = a.size[a.n - 1 - i];
        size_t sb = b.size[b.n - 1 - i];
        size[n - 1 - i] = (sa > sb) ? sa : sb;
    }
    for(i = 0; i < n - ln; ++i){
        size[i] = a.size[i];
    }
    tensor t = tensor_make(n, size);
    free(size);
    return t;
}

void tensor_binary_op_(const tensor a, const tensor b, tensor t, float op (float, float))
{
    if(t.n == 0){
        t.data[0] = op(a.data[0], b.data[0]);
    } else {
        size_t i = 0;
        for(i = 0; i < t.size[0]; ++i){
            size_t inca = (a.n && a.size[0] == t.size[0]);
            size_t incb = (b.n && b.size[0] == t.size[0]);
            tensor suba = a;
            tensor subb = b;
            if(a.n == t.n){
                suba = tensor_get_(a, i*inca);
            }
            if(b.n == t.n){
                subb = tensor_get_(b, i*incb);
            }
            tensor subt = tensor_get_(t, i);
            tensor_binary_op_(suba, subb, subt, op);
        }
    }
}

tensor tensor_binary_op(tensor a, tensor b, float op (float, float))
{
    tensor t = tensor_broadcast(a, b);
    if(t.data == 0) return t;
    tensor_binary_op_(a, b, t, op);
    return t;
}

float tensor_add_op_(float a, float b)
{
    return a + b;
}

float tensor_sub_op_(float a, float b)
{
    return a - b;
}

float tensor_mul_op_(float a, float b)
{
    return a * b;
}

float tensor_div_op_(float a, float b)
{
    return a / b;
}

tensor tensor_add(tensor a, tensor b)
{
    return tensor_binary_op(a, b, tensor_add_op_);
}

tensor tensor_sub(tensor a, tensor b)
{
    return tensor_binary_op(a, b, tensor_sub_op_);
}

tensor tensor_mul(tensor a, tensor b)
{
    return tensor_binary_op(a, b, tensor_mul_op_);
}

tensor tensor_div(tensor a, tensor b)
{
    return tensor_binary_op(a, b, tensor_div_op_);
}

float tensor_sum(tensor a)
{
    float s = 0;
    size_t i;
    size_t len = tensor_len(a);
    for(i = 0; i < len; ++i){
        s += a.data[i];       
    }
    return s;
}

void tensor_sum_dim_(tensor a, size_t dim, tensor b)
{
    size_t i;
    if(dim == 0){
        for(i = 0; i < a.size[0]; ++i){
            tensor_binary_op_(tensor_get_(a,i), b, b, tensor_add_op_);
        }
    } else {
        for(i = 0; i < a.size[0]; ++i){
            tensor_sum_dim_(tensor_get_(a, i), dim-1, tensor_get_(b, i));
        }
    }
}

tensor tensor_sum_dim(tensor a, size_t dim)
{
    if(dim < 0 || dim >= a.n){
        fprintf(stderr, "Can't sum over dimension %ld\n", dim);
        tensor none = {0};
        return none;
    }
    size_t n = a.n - 1;
    size_t *size = calloc(n, sizeof(size_t));
    size_t i = 0;
    size_t ri = 0;
    for(i = 0; i < a.n; ++i){
        if (i == dim) continue;
        size[ri++] = a.size[i];
    }
    tensor result = tensor_make(n, size);
    tensor_sum_dim_(a, dim, result);
    return result;
}

void tensor_write(tensor t, FILE *fp)
{
    size_t len = tensor_len(t);
    fwrite(t.data, sizeof(float), len, fp);
}

void tensor_read(tensor t, FILE *fp)
{
    size_t len = tensor_len(t);
    assert(fread(t.data, sizeof(float), len, fp) == len);
}

void tensor_save(tensor t, char *fname)
{
    FILE *fp = fopen(fname, "wb");
    assert(fp != 0);
    size_t i;
    uint64_t n = t.n;
    fwrite(&n, sizeof(uint64_t), 1, fp);
    for(i = 0; i < t.n; ++i){
        uint64_t s = t.size[i];
        fwrite(&s, sizeof(uint64_t), 1, fp);
    }
    tensor_write(t, fp);
    fclose(fp);
}

tensor tensor_load(char *fname)
{
    uint64_t n;
    FILE *fp = fopen(fname, "rb");
    assert(fp != 0);
    assert(fread(&n, sizeof(uint64_t), 1, fp) == 1);
    size_t *size = calloc(n, sizeof(size_t));
    size_t i;
    for(i = 0; i < n; ++i){
        uint64_t s;
        assert(fread(&s, sizeof(uint64_t), 1, fp) == 1);
        size[i] = (size_t) s;
    }
    tensor t = tensor_make((size_t) n, size);
    tensor_read(t, fp);
    fclose(fp);
    free(size);
    return t;
}

tensor matrix_load(char *fname)
{
    uint64_t n = 2;
    FILE *fp = fopen(fname, "rb");
    assert(fp != 0);
    size_t *size = calloc(n, sizeof(size_t));
    size_t i;
    for(i = 0; i < n; ++i){
        int s;
        assert(fread(&s, sizeof(int), 1, fp) == 1);
        size[i] = (size_t) s;
    }
    tensor t = tensor_make((size_t) n, size);
    tensor_read(t, fp);
    fclose(fp);
    free(size);
    return t;
}

