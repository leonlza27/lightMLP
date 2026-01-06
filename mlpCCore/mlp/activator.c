#include "activator.h"


//统一循环逻辑,默认循环变量i
#define _acLoop_process(tsknum, acexpr) {\
        for(uint32_t i = 0;i < tsknum; i++) acexpr;\
}\

void ReLU(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size,out[i] = qfmax(Q_0, in[i]);)
}

void ReLU6(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;

    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size, out[i] = qfmin(Q_6p0, qfmax(Q_0, in[i]));)
}

void LeakyReLU(const matrix_qfix input,matrix_qfix output, qfix alpha){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size, out[i] = qfmax(qfix_mul(in[i],alpha), in[i]);)
}

void Sigmoid_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size,{
        // Direct computation without temporary variable
        qfix val = Q_0p5 + (qfix)((Q_0p2 * (_tmp_larger)in[i]) >> QSHIFT); // Approximate multiply using shift
        out[i] = (val > Q_1p0) ? Q_1p0 : (val < Q_0 ? Q_0 : val);
    })
}

void Tanh_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size, out[i] = (in[i] > Q_1p0) ? Q_1p0 : (in[i] < Q_neg1p0 ? Q_neg1p0 : in[i]);)
}

void Sigmoid(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size, out[i] = qfix_div(Q_1p0, Q_1p0 + exp_qfix(-in[i]));)
}

void Tanh(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size,{
        qfix exp_x = exp_qfix(in[i]);
        qfix exp_negx = exp_qfix(-in[i]);
        out[i] = qfix_div(exp_x - exp_negx, exp_x + exp_negx);
    })
}

void Sign(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;

    _acLoop_process(size, out[i] = in[i] > 0 ? Q_1p0 : Q_0;)
}

void Softmax(const matrix_qfix input, matrix_qfix output){    
    qfix elem_exp_sum = 0;
    const uint32_t size = input->rows * input->cols;
    qfix *elem_exp = (qfix*)malloc(sizeof(qfix) * size);
    const qfix* in = input->data;
    qfix* out = output->data;

        for(uint32_t i = 0; i < size; i++){
            elem_exp[i] = exp_qfix(in[i]);
            elem_exp_sum += elem_exp[i];
        }

    _acLoop_process(size, out[i] = qfix_div(elem_exp[i], elem_exp_sum);)

    free(elem_exp);
}