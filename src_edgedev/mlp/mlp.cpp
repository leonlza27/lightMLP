#include "mlp.h"

void mlp_calclyr::confLyr(NetLyrAllocator lyrConf){
    weights = new_matrix_bp();
    bias = new_matrix_bp();
    tmp = new_matrix_bp();

    matrix_bp_set(weights, lyrConf.inputs, lyrConf.outputs, lyrConf.existedWeightData);
    matrix_bp_set(bias, 1, lyrConf.inputs, lyrConf.existedBiasData);
    matrix_bp_set(tmp, 1, lyrConf.outputs, 0);

    _alpha = lyrConf.dataExtra;
}

void mlp_calclyr::fwdCalc(const matrix_bp input, matrix_bp output){
    matrix_bp_mulpty(weights, input, tmp);
    matrix_bp_add(tmp, bias, tmp);

    switch(ActiveType){
        case type_ReLU: ReLU(tmp, output); break;
        case type_ReLU6: ReLU6(tmp, output); break;
        case type_LeakyReLU: LeakyReLU(tmp, output, _alpha); break;
        case type_Sigmoid: Sigmoid(tmp, output); break;
        case type_Sigmoid_hard: Sigmoid_Hard(tmp, output); break;
        case type_Tanh: Tanh(tmp, output); break;
        case type_Tanh_hard: Tanh_Hard(tmp, output); break;
        case type_Sign: Sign(tmp, output); break;
        case type_Softmax: Softmax(tmp, output); break;
        default: break;
    }
}



mlpNetRef::mlpNetRef(uint16_t lyrnum,NetLyrAllocator *netstruct){
    NetLyrs = (mlp_calclyr*)malloc(sizeof(mlp_calclyr) * lyrnum);
    fullConnDataMid = (matrix_bp_data*)malloc(sizeof(matrix_bp_data) * 3);
    for(uint16_t i = 0; i < lyrnum; i++){
        NetLyrs[i].confLyr(netstruct[i]);
    }
    
    netLyrCount = lyrnum;
}

void mlpNetRef::infer(matrix_bp input){
    for(uint16_t i = 0; i < netLyrCount; i++){

        //NetLyrs[i].fwdCalc()
    }    

}