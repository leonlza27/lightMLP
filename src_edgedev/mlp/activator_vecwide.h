//mlp_activator:仅用作前向激活
#ifndef _mlp_act_
#define _mlp_act_

#include "../matrix/matrix_static.h"
#include <algorithm>

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

namespace mlp_mcu_act{
    void ReLU(const p_matrix_qfloat input,p_matrix_qfloat output);
    void ReLU6(const p_matrix_qfloat input,p_matrix_qfloat output);
    void LeakyReLU(const p_matrix_qfloat input,p_matrix_qfloat output, qfloat alpha);
    void Sigmoid_Hard(const p_matrix_qfloat input,p_matrix_qfloat output);
    void Tanh_Hard(const p_matrix_qfloat input,p_matrix_qfloat output);
    
    void Sigmoid(const p_matrix_qfloat input,p_matrix_qfloat output);
    void Tanh(const p_matrix_qfloat input,p_matrix_qfloat output);
    
    void Sign(const p_matrix_qfloat input,p_matrix_qfloat output);

    void Softmax(const p_matrix_qfloat input,p_matrix_qfloat output);
}

void get_act_func(ACT_TYPE type, const p_matrix_qfloat input, p_matrix_qfloat output, qfloat alpha = 0);

#endif