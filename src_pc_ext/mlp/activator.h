#ifndef _mlp_activator
#define _mlp_activator

#include <stdint.h>
#include "../matrix/matrix_gern.h"
#include "../tools/mult_proc.h"

//预存变量

static const qfix Q_0 = float_to_qfix(0.0f);
static const qfix Q_1p0 = float_to_qfix(1.0f);
static const qfix Q_neg1p0 = float_to_qfix(-1.0f);
static const qfix Q_0p5 = float_to_qfix(0.5f);
static const qfix Q_0p2 = float_to_qfix(0.2f);
static const qfix Q_6p0 = float_to_qfix(6.0f);

//实现

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

//统一循环逻辑,默认循环变量i
#define _acLoop_process(tsknum, acexpr) {\
    multi_process(tsknum,[=]wrapper_custom_start_end{\
        for(uint32_t i = start;i < end; i++) acexpr;\
    });\
}

#ifdef __cplusplus
extern "C"{
#endif

static inline void ReLU(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,out[i] = qfmax(Q_0, in[i]);)
}

static inline void ReLU6(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size, out[i] = qfmin(Q_6p0, qfmax(Q_0, in[i]));)
}

static inline void LeakyReLU(const matrix_qfix input,matrix_qfix output, qfix alpha){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size, out[i] = qfmax(qfix_mul(in[i],alpha), in[i]);)
}

static inline void Sigmoid_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,{
        // Direct computation without temporary variable
        qfix val = Q_0p5 + (qfix)((Q_0p2 * (_tmp_larger)in[i]) >> QSHIFT); // Approximate multiply using shift
        out[i] = (val > Q_1p0) ? Q_1p0 : (val < Q_0 ? Q_0 : val);
    })
}

static inline void Tanh_Hard(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size, out[i] = (in[i] > Q_1p0) ? Q_1p0 : (in[i] < Q_neg1p0 ? Q_neg1p0 : in[i]);)
}

static inline void Sigmoid(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size, out[i] = qfix_div(Q_1p0, Q_1p0 + exp_qfix(-in[i]));)
}

static inline void Tanh(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size,{
        qfix exp_x = exp_qfix(in[i]);
        qfix exp_negx = exp_qfix(-in[i]);
        out[i] = qfix_div(exp_x - exp_negx, exp_x + exp_negx);
    })
}

static inline void Sign(const matrix_qfix input, matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    _acLoop_process(size, out[i] = in[i] > 0 ? Q_1p0 : Q_0;)
}

static inline void Softmax(const matrix_qfix input, matrix_qfix output){    
    qfix elem_exp_sum = 0;
    const uint32_t size = input->rows * input->cols;
    qfix *elem_exp = (qfix*)malloc(sizeof(qfix) * size);
    const qfix* in = input->data;
    qfix* out = output->data;
    
    multi_process(size,[=,&elem_exp_sum]wrapper_custom_start_end{
        for(uint32_t i = start; i < end; i++){
            elem_exp[i] = exp_qfix(in[i]);
            elem_exp_sum += elem_exp[i];
        }
    });
    
    _acLoop_process(size, out[i] = qfix_div(elem_exp[i], elem_exp_sum);)

    free(elem_exp);
}

#ifdef __cplusplus
}
#endif

#endif