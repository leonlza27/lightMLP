#ifndef _activator_
#define _activator_

#include "../matrix/matrix.h"
#include <math.h>

namespace MLP_ACT{
    matrix ReLU(const matrix& input);
    matrix Leak_ReLU(const matrix& input,double alpha = 0.01);
    matrix Sigmoid(const matrix& input);
    matrix Tanh(const matrix& input);
    matrix SiLU(const matrix& input);
    matrix ELU(const matrix& input,double alpha = 1.0);

    matrix Softmax(const matrix& input);
}

namespace MLP_ACT_DET{
    matrix d_ReLU(const matrix& input);
    matrix d_Leak_ReLU(const matrix& input,double alpha = 0.01);
    matrix d_Sigmoid(const matrix& input);
    matrix d_Tanh(const matrix& input);
    matrix d_SiLU(const matrix& input);
    matrix d_ELU(const matrix& input,double alpha = 1.0);

    matrix d_Softmax(const matrix& input,unsigned int curIndex);
}

#endif