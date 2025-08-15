#include "activator.h"

matrix MLP_ACT::ReLU(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        ret.directIndexData(i) = std::max(0.0,input.directIndexData(i));
    }
    return ret;
}

matrix MLP_ACT::Leak_ReLU(const matrix &input, double alpha){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        ret.directIndexData(i) = std::max(alpha,input.directIndexData(i));
    }
    return ret;
}

matrix MLP_ACT::Sigmoid(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        ret.directIndexData(i) = 1.0/(1.0+exp(-input.directIndexData(i)));
    }
    return ret;
}

matrix MLP_ACT::Tanh(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        ret.directIndexData(i) = tanh(input.directIndexData(i));
    }
    return ret;
}

matrix MLP_ACT::SiLU(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        ret.directIndexData(i) = input.directIndexData(i) * (1.0/(1.0+exp(-input.directIndexData(i))));
    }
    return ret;
}

matrix MLP_ACT::ELU(const matrix &input, double alpha){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = input.directIndexData(i);
        ret.directIndexData(i) = (tmp > 0? tmp : (exp(tmp) - 1.0) * alpha);
    }
    return ret;
}

matrix MLP_ACT::Softmax(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    double total = 0;
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        total += exp(input.directIndexData(i));
    }
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        ret.directIndexData(i) = exp(input.directIndexData(i))/total;
    }
    return ret;
}

matrix MLP_ACT_DET::d_ReLU(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = input.directIndexData(i);
        ret.directIndexData(i) = tmp > 0? 1.0 : 0.0;
    }
    return ret;
}

matrix MLP_ACT_DET::d_Leak_ReLU(const matrix &input, double alpha){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = input.directIndexData(i);
        ret.directIndexData(i) = tmp > 0? 1.0 : alpha;
    }
    return ret;
}

matrix MLP_ACT_DET::d_Sigmoid(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    matrix resu_from_sigmoid = MLP_ACT::Sigmoid(input);
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = resu_from_sigmoid.directIndexData(i);
        ret.directIndexData(i) = tmp * (1.0 - tmp);
    }
    return ret;
}

matrix MLP_ACT_DET::d_Tanh(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    matrix resu_from_tanh = MLP_ACT::Tanh(input);
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = resu_from_tanh.directIndexData(i);
        ret.directIndexData(i) = 1.0 - tmp * tmp;
    }
    return ret;
}

matrix MLP_ACT_DET::d_SiLU(const matrix &input){
    matrix ret(input.getCols(),input.getRows());
    matrix resu_from_sigmoid = MLP_ACT::Sigmoid(input);
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = resu_from_sigmoid.directIndexData(i);
        ret.directIndexData(i) = tmp * input.directIndexData(i) * (1.0 - tmp) + tmp;
    }
    return ret;
}

matrix MLP_ACT_DET::d_ELU(const matrix &input, double alpha){
    matrix ret(input.getCols(),input.getRows());
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = input.directIndexData(i);
        ret.directIndexData(i) = tmp > 0? 1.0 : alpha * exp(tmp);
    }
    return ret;
}

matrix MLP_ACT_DET::d_Softmax(const matrix &input,unsigned int curIndex){
    matrix ret(input.getCols(),input.getRows());
    matrix resu_from_softmax = MLP_ACT::Softmax(input);
    double curIndConst = resu_from_softmax.directIndexData(curIndex);
    for(unsigned int i = 0;i<input.getRows()*input.getCols();i++){
        double tmp = resu_from_softmax.directIndexData(i);
        ret.directIndexData(i) = i == curIndex? (tmp * (1.0 - tmp)) : -(tmp * curIndConst);
    }
    return ret;
}
