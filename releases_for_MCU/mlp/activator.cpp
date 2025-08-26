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

void LeakyReLU(const p_matrix_qfloat input,p_matrix_qfloat output, qfloat alpha){
    const uint32_t size = input->cols * input->rows;
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::max(qfloat_mul(input->data[i],alpha), input->data[i]);
    }
}

void mlp_mcu_act::Sigmoid_Hard(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat zero = float_to_qfloat(0.0f);
    const qfloat one = float_to_qfloat(1.0f);
    const qfloat half = float_to_qfloat(0.5f);
    const qfloat point_two = float_to_qfloat(0.2f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = std::max(zero, std::min(one, half + qfloat_mul(point_two, input->data[i])));
    }
}

void mlp_mcu_act::Tanh_Hard(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat neg_one = float_to_qfloat(-1.0f);
    const qfloat one = float_to_qfloat(1.0f);
    
    for(uint32_t i = 0; i < size; i++){
        qfloat temp = input->data[i];
        if(temp < neg_one) output->data[i] = neg_one;
        else if(temp > one) output->data[i] = one;
        else output->data[i] = temp;
    }
}

void mlp_mcu_act::Sigmoid(const p_matrix_qfloat input, p_matrix_qfloat output){
    const uint32_t size = input->cols * input->rows;
    const qfloat one = float_to_qfloat(1.0f);
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = qfloat_div(one, one + exp_qfloat(input->data[i]));
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
    static qfloat elem_exp[1024]; // 静态变量替代栈数组
    qfloat elem_exp_sum = 0;
    const uint32_t size = input->rows * input->cols;
    
    for(uint32_t i = 0; i < size; i++){
        elem_exp[i] = exp_qfloat(input->data[i]);
        elem_exp_sum += elem_exp[i];
    }
    
    for(uint32_t i = 0; i < size; i++){
        output->data[i] = qfloat_div(elem_exp[i], elem_exp_sum);
    }
}
