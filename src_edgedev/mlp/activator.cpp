#include "activator.h"

void mlp_mcu_act::ReLU(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix zero = float_to_qfix(0.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::max(zero, input->data[i]);
    }
}

void mlp_mcu_act::ReLU6(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix zero = float_to_qfix(0.0f);
    const qfix six = float_to_qfix(6.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::min(six, std::max(zero, input->data[i]));
    }
}

void mlp_mcu_act::LeakyReLU(const p_matrix_qfix input,p_matrix_qfix output, qfix alpha){
    const uint32_t size = input->cols * input->rows;
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::max(qfix_mul(input->data[i],alpha), input->data[i]);
    }
}

// Pre-computed constants
static const qfix Q_ZERO = float_to_qfix(0.0f);
static const qfix Q_ONE = float_to_qfix(1.0f);
static const qfix Q_NEG_ONE = float_to_qfix(-1.0f);
static const qfix Q_HALF = float_to_qfix(0.5f);
static const qfix Q_POINT_TWO = float_to_qfix(0.2f);

void mlp_mcu_act::Sigmoid_Hard(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        // Direct computation without temporary variable
        qfix val = Q_HALF + (qfix)((Q_POINT_TWO * (_tmp_larger)in[i]) >> QSHIFT); // Approximate multiply using shift
        out[i] = (val > Q_ONE) ? Q_ONE : (val < Q_ZERO ? Q_ZERO : val);
    }
}

void mlp_mcu_act::Tanh_Hard(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix* in = input->data;
    qfix* out = output->data;
    
    for(uint32_t i = 0; i < size; i++){
        // Direct comparison without function calls
        out[i] = (in[i] > Q_ONE) ? Q_ONE : (in[i] < Q_NEG_ONE ? Q_NEG_ONE : in[i]);
    }
}

void mlp_mcu_act::Sigmoid(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix one = float_to_qfix(1.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = qfix_div(one, one + exp_qfix(-input->data[i]));
    }
}

void mlp_mcu_act::Tanh(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    
    for(uint32_t i = 0; i < size; i++){
        qfix exp_x = exp_qfix(input->data[i]);
        qfix exp_negx = exp_qfix(-input->data[i]);
        output->data[i] = qfix_div(exp_x - exp_negx, exp_x + exp_negx);
    }
}

void mlp_mcu_act::Sign(const p_matrix_qfix input, p_matrix_qfix output){
    const uint32_t size = input->cols * input->rows;
    const qfix one = float_to_qfix(1.0f);
    const qfix zero = 0;
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = input->data[i] > 0 ? one : zero;
    }
}

void mlp_mcu_act::Softmax(const p_matrix_qfix input, p_matrix_qfix output){    
    qfix elem_exp_sum = 0;
    const uint32_t size = input->rows * input->cols;
    qfix elem_exp[size];
    for(uint32_t i = 0; i < size; i++){
        elem_exp[i] = exp_qfix(input->data[i]);
        elem_exp_sum += elem_exp[i];
    }
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = qfix_div(elem_exp[i], elem_exp_sum);
    }
}

void get_act_func(ACT_TYPE type, const p_matrix_qfix input, p_matrix_qfix output, qfix alpha) {
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