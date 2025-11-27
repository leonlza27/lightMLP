#ifndef _activator
#define _activator

#include "../matrix/matrix_static.h"

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

inline void ReLU(const matrix_qfix input,matrix_qfix output);
inline void ReLU6(const matrix_qfix input,matrix_qfix output);
inline void LeakyReLU(const matrix_qfix input,matrix_qfix output, qfix alpha);
inline void Sigmoid_Hard(const matrix_qfix input,matrix_qfix output);
inline void Tanh_Hard(const matrix_qfix input,matrix_qfix output);

inline void Sigmoid(const matrix_qfix input,matrix_qfix output);
inline void Tanh(const matrix_qfix input,matrix_qfix output);

inline void Sign(const matrix_qfix input,matrix_qfix output);
inline void Softmax(const matrix_qfix input,matrix_qfix output);


#endif