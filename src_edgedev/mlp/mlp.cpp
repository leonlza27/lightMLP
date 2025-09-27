#include "mlp.h"

void mlp_calclyr::confLyr(NetLyrAllocator lyrConf){
    weights = new_matrix_bp();
    bias = new_matrix_bp();
    tmp = new_matrix_bp();

    matrix_bp_set(weights, lyrConf.inputs, lyrConf.outputs, lyrConf.existedWeightData);
    matrix_bp_set(bias, 1, lyrConf.inputs, lyrConf.existedBiasData);
    matrix_bp_set(tmp, 1, lyrConf.outputs, 0);
}

void mlp_calclyr::fwdCalc(NetLyrAllocator lyrConf){
    
}

mlpNetRef::mlpNetRef(uint16_t lyrnum,NetLyrAllocator *netstruct){
    NetLyrs = (mlp_calclyr*)malloc(sizeof(mlp_calclyr) * lyrnum);
    fullConnData = (matrix_bp_data*)malloc(sizeof(matrix_bp_data) * lyrnum);
    for(uint16_t i = 0; i < lyrnum; i++){
        matrix_bp_init(fullConnData + i, 1, netstruct[i].outputs, 0);
        NetLyrs[i].confLyr(netstruct[i]);
    }
    
    netLyrCount = lyrnum;
}

void mlpNetRef::infer(p_matrix_qfix input){
    

}