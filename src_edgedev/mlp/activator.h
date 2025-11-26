#ifndef _activator
#define _activator

#include "../matrix/matrix_static.h"

enum ACT_TYPE{
    type_UNSUPPORTED,
    type_ReLU,
    type_ReLU6,
    type_Sigmoid,
    type_Tanh,
    type_Sigmoid_hard,
    type_Tanh_hard,
    type_Sign,
    type_Softmax,
    type_LeakyReLU
};

void ReLU(const matrix_bp input,matrix_bp output);
void ReLU6(const matrix_bp input,matrix_bp output);
void LeakyReLU(const matrix_bp input,matrix_bp output, qfix alpha);
void Sigmoid_Hard(const matrix_bp input,matrix_bp output);
void Tanh_Hard(const matrix_bp input,matrix_bp output);

void Sigmoid(const matrix_bp input,matrix_bp output);
void Tanh(const matrix_bp input,matrix_bp output);

void Sign(const matrix_bp input,matrix_bp output);
void Softmax(const matrix_bp input,matrix_bp output);


#endif