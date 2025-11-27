#include "activator.h"
#include "../qfix_ops/qfix_op.h"

//预存变量

static const qfix Q_ZERO = float_to_qfix(0.0f);
static const qfix Q_ONE = float_to_qfix(1.0f);
static const qfix Q_NEG_ONE = float_to_qfix(-1.0f);
static const qfix Q_HALF = float_to_qfix(0.5f);
static const qfix Q_POINT_TWO = float_to_qfix(0.2f);

//实现

void ReLU(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        out[i] = qfmax(Q_ZERO, in[i]);
    }
}

void ReLU6(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix six = float_to_qfix(6.0f);
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        out[i] = qfmin(six, qfmax(Q_ZERO, in[i]));
    }
}

void LeakyReLU(const matrix_qfix input,matrix_qfix output, qfix alpha){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        out[i] = qfmax(qfix_mul(in[i],alpha), in[i]);
    }
}

void Sigmoid_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        // Direct computation without temporary variable
        qfix val = Q_HALF + (qfix)((Q_POINT_TWO * (_tmp_larger)in[i]) >> QSHIFT); // Approximate multiply using shift
        out[i] = (val > Q_ONE) ? Q_ONE : (val < Q_ZERO ? Q_ZERO : val);
    }
}

void Tanh_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        // Direct comparison without function calls
        out[i] = (in[i] > Q_ONE) ? Q_ONE : (in[i] < Q_NEG_ONE ? Q_NEG_ONE : in[i]);
    }
}

void Sigmoid(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        out[i] = qfix_div(Q_ONE, Q_ONE + exp_qfix(-in[i]));
    }
}

void Tanh(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        qfix exp_x = exp_qfix(in[i]);
        qfix exp_negx = exp_qfix(-in[i]);
        out[i] = qfix_div(exp_x - exp_negx, exp_x + exp_negx);
    }
}

void Sign(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        out[i] = in[i] > 0 ? Q_ONE : Q_ZERO;
    }
}

void Softmax(const matrix_qfix input, matrix_qfix output){    
    qfix elem_exp_sum = 0;
    const uint32_t size = input->rows * input->cols;
    qfix elem_exp[size];
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        elem_exp[i] = exp_qfix(in[i]);
        elem_exp_sum += elem_exp[i];
    }
    
    for(uint32_t i = 0; i < size; i++){
        out[i] = qfix_div(elem_exp[i], elem_exp_sum);
    }
}
