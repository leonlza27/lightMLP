#include "activator.h"

void mlp_mcu_act::ReLU(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat zero = float_to_qfloat(0.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::max(zero, input->data[i]);
    }
}

void mlp_mcu_act::ReLU6(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat zero = float_to_qfloat(0.0f);
    const qfloat six = float_to_qfloat(6.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::min(six, std::max(zero, input->data[i]));
    }
}

void mlp_mcu_act::LeakyReLU(const p_matrix_qfloat input,p_matrix_qfloat output, qfloat alpha){
    const uint32_t size = input->cols * input->rows;
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::max(qfloat_mul(input->data[i],alpha), input->data[i]);
    }
}

// Pre-computed constants
static const qfloat Q_ZERO = float_to_qfloat(0.0f);
static const qfloat Q_ONE = float_to_qfloat(1.0f);
static const qfloat Q_NEG_ONE = float_to_qfloat(-1.0f);
static const qfloat Q_HALF = float_to_qfloat(0.5f);
static const qfloat Q_POINT_TWO = float_to_qfloat(0.2f);

void mlp_mcu_act::Sigmoid_Hard(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat* in = input->data;
    qfloat* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        // Direct computation without temporary variable
        qfloat val = Q_HALF + (qfloat)((Q_POINT_TWO * (_tmp_larger)in[i]) >> QSHIFT); // Approximate multiply using shift
        out[i] = (val > Q_ONE) ? Q_ONE : (val < Q_ZERO ? Q_ZERO : val);
    }
}

void mlp_mcu_act::Tanh_Hard(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat* in = input->data;
    qfloat* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        // Direct comparison without function calls
        out[i] = (in[i] > Q_ONE) ? Q_ONE : (in[i] < Q_NEG_ONE ? Q_NEG_ONE : in[i]);
    }
}

void mlp_mcu_act::Sigmoid(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat one = float_to_qfloat(1.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = qfloat_div(one, one + exp_qfloat(-input->data[i]));
    }
}

void mlp_mcu_act::Tanh(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    
    for(uint32_t i = 0; i < size; i++){
        qfloat exp_x = exp_qfloat(input->data[i]);
        qfloat exp_negx = exp_qfloat(-input->data[i]);
        output->data[i] = qfloat_div(exp_x - exp_negx, exp_x + exp_negx);
    }
}

void mlp_mcu_act::Sign(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat one = float_to_qfloat(1.0f);
    const qfloat zero = 0;
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = input->data[i] > 0 ? one : zero;
    }
}

void mlp_mcu_act::Softmax(const p_matrix_qfloat input, p_matrix_qfloat output){    
    qfloat elem_exp_sum = 0;
    const uint32_t size = input->rows * input->cols;
    qfloat elem_exp[size];
    for(uint32_t i = 0; i < size; i++){
        elem_exp[i] = exp_qfloat(input->data[i]);
        elem_exp_sum += elem_exp[i];
    }
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = qfloat_div(elem_exp[i], elem_exp_sum);
    }
}

void get_act_func(ACT_TYPE type, const p_matrix_qfloat input, p_matrix_qfloat output, qfloat alpha) {
    switch(type) {
        case type_ReLU: mlp_mcu_act::ReLU(input, output); break;
        case type_ReLU6: mlp_mcu_act::ReLU6(input, output); break;
        case type_LeakyReLU: mlp_mcu_act::LeakyReLU(input, output, alpha); break;
        case type_Sigmoid: mlp_mcu_act::Sigmoid(input, output); break;
        case type_Tanh: mlp_mcu_act::Tanh(input, output); break;
        case type_Sigmoid_hard: mlp_mcu_act::Sigmoid_Hard(input, output); break;
        case type_Tanh_hard: mlp_mcu_act::Tanh_Hard(input, output); break;
        case type_Sign: mlp_mcu_act::Sign(input, output); break;
        case type_Softmax: mlp_mcu_act::Softmax(input, output); break;
        default: break; // 处理不支持的type
    }
}