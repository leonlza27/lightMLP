#include "mlp.h"
#include <stdlib.h>

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


