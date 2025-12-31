#include "mlp.h"
#include <malloc.h>
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


void lmlp_setupRef(uint16_t lyrnum, NetLyrConf *netstruct, mlpNetRefInfo *dest){
    dest->lyrData = netstruct;
    dest->netLyrCount = lyrnum;

    uint16_t maxSize = 0;
    for(uint16_t i = 0; i < lyrnum; i++){
        matrix_bp mcur = (matrix_bp)((char*)(netstruct + i) + sizeof(qfix) + sizeof(uint8_t));
        uint16_t curSize = *((uint16_t*)mcur) * *(((uint16_t*)mcur) + 1);
        maxSize = curSize > maxSize? curSize : maxSize;
    }

    dest->fullConnDataMid[0] = (matrix_bp)malloc(maxSize * sizeof(qfix) + 4);
    dest->fullConnDataMid[1] = (matrix_bp)malloc(maxSize * sizeof(qfix) + 4);

}

void lmlp_ref_infer(mlpNetRefInfo net, matrix_bp input){
    qfix alpha = *(qfix*)((char*)net.lyrData + sizeof(uint8_t));
    uint8_t acTp = *(uint8_t*)net.lyrData;
    matrix_bp weights = *(matrix_bp*)((char*)net.lyrData + sizeof(uint8_t) + sizeof(qfix));
    matrix_bp bias = *(matrix_bp*)((char*)net.lyrData + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

    fwdCalc(weights, bias, acTp, alpha, input, net.fullConnDataMid[0]);

    char *curAddr = 0;

    for(uint16_t i = 1; i < net.netLyrCount; i++){
        curAddr = (char*)(net.lyrData + i);
        alpha = *(qfix*)(curAddr + sizeof(uint8_t));
        acTp = *(uint8_t*)curAddr;
        weights = *(matrix_bp*)(curAddr + sizeof(uint8_t) + sizeof(qfix));
        bias = *(matrix_bp*)(curAddr + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

        fwdCalc(weights, bias, acTp, alpha, net.fullConnDataMid[(i - 1) % 2], net.fullConnDataMid[i % 2]);
    }    


}

inline void backCalc(mlpNetTrainInfo net, uint16_t lyridx, matrix_bp grad, qfix lr){
    qfix _alpha = *(qfix*)((char*)&net.lyrData[lyridx] + sizeof(uint8_t));
    uint8_t acTp = *(uint8_t*)&net.lyrData[lyridx];
    matrix_bp weights = *(matrix_bp*)((char*)&net.lyrData[lyridx] + sizeof(uint8_t) + sizeof(qfix));
    matrix_bp bias = *(matrix_bp*)((char*)&net.lyrData[lyridx] + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

    matrix_bp dady_this = net.grad_this_det[lyridx];
    matrix_bp x_this = net.fullConnData[lyridx];
    uint16_t vec_y_size = bias->rows;
    uint16_t vec_x_size = weights->cols;

    matrix_bp grad_w_this = net.grad_weights[lyridx];

    switch(acTp){
        case type_ReLU:         grad_ReLU(grad, dady_this); break;
        case type_ReLU6:        grad_ReLU6(grad, dady_this); break;
        case type_LeakyReLU:    grad_LeakyReLU(grad, dady_this, _alpha); break;
        case type_Sigmoid:      grad_Sigmoid(grad, dady_this); break;
        case type_Sigmoid_hard: grad_Sigmoid_Hard(grad, dady_this); break;
        case type_Tanh:         grad_Tanh(grad, dady_this); break;
        case type_Tanh_hard:    grad_Tanh_Hard(grad, dady_this); break;
        case type_Sign:         grad_Sign(grad, dady_this); break;
        case type_Softmax:
        default:                {for(uint16_t i = 0; i < vec_y_size; i++) {dady_this->data[i] = grad->data[i];}} break;
    }

    x_this->cols = vec_x_size;
    x_this->rows = 1;

    matrix_bp_mulpty(dady_this, x_this, grad_w_this);
    matrix_bp_mulpty(net.weights_T[lyridx], dady_this, net.grad_to_last[lyridx]);

    for(uint16_t i = 0; i < vec_y_size; i++) bias->data[i] -= qfix_mul(lr, dady_this->data[i]);

    uint32_t size = vec_x_size * vec_y_size;
    for(uint32_t i = 0; i < size; i++) weights->data[i] -= qfix_mul(lr, grad_w_this->data[i]);

    matrix_bp_transpose(weights, net.weights_T[lyridx]);
    
    x_this->rows = vec_x_size;
    x_this->cols = 1;    
}   

void lmlp_setupTrainer(uint16_t lyrnum, NetLyrConf *netstruct, mlpNetTrainInfo *dest){
    dest->lyrData = netstruct;
    dest->netLyrCount = lyrnum;

    dest->fullConnData = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    dest->grad_to_last = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    dest->grad_this_det = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    dest->weights_T = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);
    dest->grad_weights = (matrix_bp*)malloc(sizeof(matrix_bp) * lyrnum);

    uint16_t in_dim = 0, out_dim = 0;
    for(uint16_t i = 0; i < lyrnum; i++){
        in_dim = netstruct[i].existedWeightData->cols;
        out_dim = netstruct[i].existedWeightData->rows;
        dest->grad_to_last[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + in_dim * sizeof(qfix));
        dest->grad_this_det[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * sizeof(qfix));
        dest->fullConnData[i + 1] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * sizeof(qfix));
        dest->weights_T[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * in_dim * sizeof(qfix));
        dest->grad_weights[i] = (matrix_bp)malloc(sizeof(matrix_bp_data) + out_dim * in_dim * sizeof(qfix));
        matrix_bp_transpose(netstruct[i].existedWeightData, dest->weights_T[i]);
        
        dest->grad_to_last[i]->rows = in_dim;
        dest->grad_to_last[i]->cols = 1;
        
        dest->grad_this_det[i]->rows = out_dim;
        dest->grad_this_det[i]->cols = 1;
    }
}

void lmlp_trainer_infer(mlpNetTrainInfo net, matrix_bp input){
    net.fullConnData[0] = input;

    char *curAddr = 0;

    for(uint16_t i = 0; i < net.netLyrCount; i++){
        curAddr = (char*)(net.lyrData + i);
        qfix alpha = *(qfix*)((char*)curAddr + sizeof(uint8_t));
        uint8_t acTp = *(uint8_t*)curAddr;
        matrix_bp weights = *(matrix_bp*)((char*)curAddr + sizeof(uint8_t) + sizeof(qfix));
        matrix_bp bias = *(matrix_bp*)((char*)curAddr + sizeof(uint8_t) + sizeof(qfix) + sizeof(matrix_bp));

        fwdCalc(weights, bias, acTp, alpha, net.fullConnData[i], net.fullConnData[i + 1]);
    }
}

void lmlp_trainer_backward(mlpNetTrainInfo net, matrix_bp grad_from_resu, qfix lr){
    backCalc(net, net.netLyrCount - 1, grad_from_resu, lr);

    for(uint16_t i = net.netLyrCount - 1; i > 0; i--){
        backCalc(net, i - 1, net.grad_to_last[i], lr);
    }
}
