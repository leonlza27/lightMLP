#ifndef __mlp_act_d
#define __mlp_act_d

//dL[last]/dy = (dL[last]/da) * (da/dy)

#include "../matrix/matrix_static.h"
#include "activator.h"

void grad_ReLU(const matrix_qfix input, matrix_qfix output);

void grad_ReLU6(const matrix_qfix input, matrix_qfix output);

void grad_LeakyReLU(const matrix_qfix input, matrix_qfix output, qfix alpha);

void grad_Sigmoid(const matrix_qfix input, matrix_qfix output);

void grad_Sigmoid_Hard(const matrix_qfix input, matrix_qfix output);

void grad_Tanh(const matrix_qfix input, matrix_qfix output);

void grad_Tanh_Hard(const matrix_qfix input, matrix_qfix output);

void grad_Sign(const matrix_qfix input, matrix_qfix output);

//交叉熵优化
void grad_Softmax(const matrix_qfix input, const matrix_qfix standard, matrix_qfix output);

#endif