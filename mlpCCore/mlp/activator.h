#ifndef _mlp_activator
#define _mlp_activator

#include <stdint.h>
#include "../matrix/matrix_static.h"
#include "../tools/mult_proc.h"

enum ml_act_type{
    type_ReLU,
    type_ReLU6,
    type_LeakyReLU ,
    type_Sigmoid,
    type_Sigmoid_hard,
    type_Tanh,
    type_Tanh_hard,
    type_Sign,
    type_Softmax,
};

//预存变量

static const qfix Q_0 = float_to_qfix(0.0f);
static const qfix Q_1p0 = float_to_qfix(1.0f);
static const qfix Q_neg1p0 = float_to_qfix(-1.0f);
static const qfix Q_0p5 = float_to_qfix(0.5f);
static const qfix Q_0p2 = float_to_qfix(0.2f);
static const qfix Q_6p0 = float_to_qfix(6.0f);

//统一循环逻辑,默认循环变量i
#define _acLoop_process(tsknum, acexpr) {\
    for(uint32_t i = 0;i < tsknum; i++) acexpr;\
    }\

void ReLU(const matrix_qfix input, matrix_qfix output);

void ReLU6(const matrix_qfix input, matrix_qfix output);

void LeakyReLU(const matrix_qfix input,matrix_qfix output, qfix alpha);

void Sigmoid_Hard(const matrix_qfix input, matrix_qfix output);

void Tanh_Hard(const matrix_qfix input, matrix_qfix output);

void Sigmoid(const matrix_qfix input, matrix_qfix output);

void Tanh(const matrix_qfix input, matrix_qfix output);

void Sign(const matrix_qfix input, matrix_qfix output);

void Softmax(const matrix_qfix input, matrix_qfix output);

#endif