#include "mlp.h"

void mlp_calclyr::confLyr(NetLyrAllocator lyrConf){
    weights = alloc_matrix_Q16_16();
    bias = alloc_matrix_Q16_16();
    tmp = alloc_matrix_Q16_16();

    matrix_Q16_16_init(weights,lyrConf.inputs,lyrConf.outputs,lyrConf.existedWeightData);
    matrix_Q16_16_init(bias,1,lyrConf.outputs,lyrConf.existedBiasData);
    matrix_Q16_16_init(tmp,1,lyrConf.outputs,0);
}