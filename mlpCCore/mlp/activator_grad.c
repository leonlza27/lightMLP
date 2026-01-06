#include "activator_grad.h"

const static qfix Q_4p0 = float_to_qfix(4.0f);

#define _acLoop_process(tsknum, acexpr) {\
        for(uint32_t i = 0;i < tsknum; i++) acexpr;\
}\

void grad_ReLU(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = in[i] > 0? in[i] : 0;)
}

void grad_ReLU6(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = (in[i] > 0 || in[i] < Q_6p0)? in[i] : 0;)
}

void grad_LeakyReLU(const matrix_qfix input, matrix_qfix output, qfix alpha){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = in[i] > 0? in[i] : qfix_mul(alpha, in[i]);)
}

void grad_Sigmoid(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,{
        qfix ex = exp_qfix(in[i]);
        qfix ex_pu1 = ex + Q_1p0;
        qfix bottom = qfix_mul(ex_pu1, ex_pu1);
        out[i] = qfix_mul(in[i], qfix_div(ex, bottom));
    })
}

void grad_Sigmoid_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = (in[i] > 0 || in[i] < Q_1p0)? qfix_mul(Q_0p2, in[i]) : 0;)
}

void grad_Tanh(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,{
        qfix ex = exp_qfix(in[i]);
        qfix ex_sq2 = qfix_mul(ex, ex);
        qfix ex_sq2_pu1 = ex_sq2 + Q_1p0;
        qfix bottom = qfix_mul(ex_sq2_pu1, ex_sq2_pu1);
        out[i] = qfix_mul(in[i], qfix_div(qfix_mul(Q_4p0, ex_sq2), bottom));
    })
}

void grad_Tanh_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = (in[i] > Q_neg1p0 || in[i] < Q_1p0)? in[i] : 0;)
}

void grad_Sign(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size, out[i] = 0)
}

//交叉熵优化
void grad_Softmax(const matrix_qfix input, const matrix_qfix standard, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    const qfix* _std = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = in[i] - _std[i];)
}