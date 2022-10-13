// Include guards and C++ compatibility
#ifndef MATRIX_H
#define MATRIX_H
#include "tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

tensor matrix_multiply(const tensor a, const tensor b);
tensor matrix_transpose(const tensor a);
tensor matrix_invert(tensor m);
tensor solve_system(tensor M, tensor b);


#ifdef __cplusplus
}
#endif
#endif
