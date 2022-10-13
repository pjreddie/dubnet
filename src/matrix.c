#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"

// Transpose a matrix
// tensor m: matrix to be transposed
// returns: tensor, result of transposition
tensor matrix_transpose(tensor a)
{
    assert(a.n == 2);
    // TODO 1.0: return a transposed version of a (don't modify a)
    tensor t = tensor_make(0, 0);

    return t;
}

// Perform matrix multiplication a*b, return result
// tensor a,b: operands
// returns: new tensor that is the result
tensor matrix_multiply(const tensor a, const tensor b)
{
    assert(a.n == 2);
    assert(b.n == 2);
    assert(a.size[1] == b.size[0]);
    // TODO 1.1: matrix multiplication! just use 3 for loops 
    tensor t = tensor_make(0, 0);

    return t;
}

// Used for matrix inversion
tensor matrix_augment(tensor m)
{
    assert(m.n == 2);
    size_t rows = m.size[0];
    size_t cols = m.size[1];
    size_t i,j;
    tensor c = tensor_vmake(2, rows, cols*2);
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            c.data[i*cols*2 + j] = m.data[i*cols + j];
        }
    }
    for(j = 0; j < rows; ++j){
        c.data[j*cols*2 + j+cols] = 1;
    }
    return c;
}

// Invert matrix m
tensor matrix_invert(tensor m)
{
    size_t i, j, k;
    assert(m.n == 2);
    assert(m.size[0] == m.size[1]);

    tensor c = matrix_augment(m);
    tensor none = {0};
    float **cdata = calloc(c.size[0], sizeof(float *));
    for(i = 0; i < c.size[0]; ++i){
        cdata[i] = c.data + i*c.size[1];
    }

    for(k = 0; k < c.size[0]; ++k){
        float p = 0.;
        size_t index = -1;
        for(i = k; i < c.size[0]; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            tensor_free(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.size[1]; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.size[0]; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.size[0]-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    tensor inv = tensor_make(2, m.size);
    for(i = 0; i < m.size[0]; ++i){
        for(j = 0; j < m.size[1]; ++j){
            inv.data[i*m.size[1] + j] = cdata[i][j+m.size[1]];
        }
    }
    tensor_free(c);
    free(cdata);
    return inv;
}

tensor solve_system(tensor M, tensor b)
{
    tensor none = {0};
    tensor Mt = matrix_transpose(M);
    tensor MtM = matrix_multiply(Mt, M);
    tensor MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    tensor Mdag = matrix_multiply(MtMinv, Mt);
    tensor a = matrix_multiply(Mdag, b);
    tensor_free(Mt);
    tensor_free(MtM);
    tensor_free(MtMinv);
    tensor_free(Mdag);
    return a;
}
