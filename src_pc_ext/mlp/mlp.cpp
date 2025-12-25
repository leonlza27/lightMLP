#include "mlp.h"
#include "activator_grad.h"

static inline void fwdCalc(const matrix_bp weights, const matrix_bp bias, const uint8_t ActiveType, const qfix _alpha  , const matrix_bp input, matrix_bp output){
    matrix_bp_mulpty(weights, input, output);
    matrix_bp_add(output, bias, output);

    switch(ActiveType){
        case type_ReLU:         ReLU(output, output); break;
        case type_ReLU6:        ReLU6(output, output); break;
        case type_LeakyReLU:    LeakyReLU(output, output, _alpha); break;
        case type_Sigmoid:      Sigmoid(output, output); break;
        case type_Sigmoid_hard: Sigmoid_Hard(output, output); break;
        case type_Tanh:         Tanh(output, output); break;
        case type_Tanh_hard:    Tanh_Hard(output, output); break;
        case type_Sign:         Sign(output, output); break;
        case type_Softmax:      Softmax(output, output); break;
        default: break;
    }
}



void mlpNetRef::init(uint16_t lyrnum,NetLyrConf *netstruct){
    lyrData = netstruct;
    netLyrCount = lyrnum;

    uint16_t maxSize = 0;
    for(uint16_t i = 0; i < lyrnum; i++){
        matrix_bp mcur = (matrix_bp)((char*)(lyrData + i) + sizeof(qfix) + sizeof(uint8_t));
        uint16_t curSize = *((uint16_t*)mcur) * *(((uint16_t*)mcur) + 1);
        maxSize = curSize > maxSize? curSize : maxSize;
    }

    fullConnDataMid = (matrix_bp)malloc(maxSize * sizeof(qfix) + 4);

}

void mlpNetRef::infer(matrix_bp input){
    qfix alpha = *(qfix*)((char*)lyrData + sizeof(uint8_t));
    uint8_t acTp = *(uint8_t*)lyrData;
    matrix_bp weights = *(matrix_bp*)((char*)lyrData + sizeof(uint8_t) + sizeof(qfix));
    matrix_bp bias = *(matrix_bp*)((char*)lyrData + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

    fwdCalc(weights, bias, acTp, alpha, input, fullConnDataMid);

    char *curAddr = 0;

    for(uint16_t i = 1; i < netLyrCount; i++){
        curAddr = (char*)(lyrData + i);
        alpha = *(qfix*)(curAddr + sizeof(uint8_t));
        acTp = *(uint8_t*)curAddr;
        weights = *(matrix_bp*)(curAddr + sizeof(uint8_t) + sizeof(qfix));
        bias = *(matrix_bp*)(curAddr + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

        fwdCalc(weights, bias, acTp, alpha, fullConnDataMid, fullConnDataMid);
    }    

}

inline void mlpNetTrainer::backCalc(uint16_t lyridx, matrix_bp grad, qfix lr){
    qfix _alpha = *(qfix*)((char*)&lyrData[lyridx] + sizeof(uint8_t));
    uint8_t acTp = *(uint8_t*)&lyrData[lyridx];
    matrix_bp weights = *(matrix_bp*)((char*)&lyrData[lyridx] + sizeof(uint8_t) + sizeof(qfix));
    matrix_bp bias = *(matrix_bp*)((char*)&lyrData[lyridx] + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

    matrix_bp dady_this = grad_this_det[lyridx];
    matrix_bp x_this = fullConnData[lyridx];
    uint16_t vec_y_size = bias->rows;
    uint16_t vec_x_size = weights->cols;

    matrix_bp grad_w_this = grad_weights[lyridx];

    switch(acTp){
        case type_ReLU:         grad_ReLU(grad, dady_this); break;
        case type_ReLU6:        grad_ReLU6(grad, dady_this); break;
        case type_LeakyReLU:    grad_LeakyReLU(grad, dady_this, _alpha); break;
        case type_Sigmoid:      grad_Sigmoid(grad, dady_this); break;
        case type_Sigmoid_hard: grad_Sigmoid_Hard(grad, dady_this); break;
        case type_Tanh:         grad_Tanh(grad, dady_this); break;
        case type_Tanh_hard:    grad_Tanh_Hard(grad, dady_this); break;
        case type_Sign:         grad_Sign(grad, dady_this); break;
        case type_Softmax:      for(uint16_t i = 0; i < vec_y_size; i++) dady_this[i] = grad[i]; break;
        default:                for(uint16_t i = 0; i < vec_y_size; i++) dady_this[i] = grad[i]; break;
    }

    x_this->cols = vec_x_size;
    x_this->rows = 1;

    matrix_bp_mulpty(dady_this, x_this, grad_w_this);
    matrix_bp_mulpty(weights_T[lyridx], dady_this, grad_to_last[lyridx]);

    for(uint16_t i = 0; i < vec_y_size; i++) bias->data[i] -= qfix_mul(lr, dady_this->data[i]);

    uint32_t size = vec_x_size * vec_y_size;
    for(uint32_t i = 0; i < size; i++) weights->data[i] -= qfix_mul(lr, grad_w_this->data[i]);

    x_this->rows = vec_x_size;
    x_this->cols = 1;    
}   

void mlpNetTrainer::init(uint16_t lyrnum, NetLyrConf *netstruct){
    lyrData = netstruct;
    netLyrCount = lyrnum;

    fullConnData = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    grad_to_last = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    grad_this_det = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    weights_T = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    grad_weights = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);

    uint16_t in_dim = 0, out_dim = 0;
    for(uint16_t i = 0; i < lyrnum; i++){
        in_dim = lyrData[i].existedWeightData->cols;
        out_dim = lyrData[i].existedWeightData->rows;
        grad_to_last[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + in_dim * sizeof(qfix));
        grad_this_det[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * sizeof(qfix));
        fullConnData[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * sizeof(qfix));
        weights_T[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * in_dim * sizeof(qfix));
        grad_weights[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * in_dim * sizeof(qfix));
        matrix_bp_transpose(lyrData[i].existedWeightData, weights_T[i]);
    }
}

void mlpNetTrainer::infer(matrix_bp input){
    qfix alpha = *(qfix*)((char*)lyrData + sizeof(uint8_t));
    uint8_t acTp = *(uint8_t*)lyrData;
    matrix_bp weights = *(matrix_bp*)((char*)lyrData + sizeof(uint8_t) + sizeof(qfix));
    matrix_bp bias = *(matrix_bp*)((char*)lyrData + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

    fwdCalc(weights, bias, acTp, alpha, input, fullConnData[0]);

    char *curAddr = 0;

    for(uint16_t i = 1; i < netLyrCount; i++){
        curAddr = (char*)(lyrData + i);
        alpha = *(qfix*)(curAddr + sizeof(uint8_t));
        acTp = *(uint8_t*)curAddr;
        weights = *(matrix_bp*)(curAddr + sizeof(uint8_t) + sizeof(qfix));
        bias = *(matrix_bp*)(curAddr + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

        fwdCalc(weights, bias, acTp, alpha, fullConnData[i-1], fullConnData[i]);
    }
}

void mlpNetTrainer::backward(matrix_bp grad_from_resu, qfix lr){

}
