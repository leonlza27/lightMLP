//mlp:仅用作mcu上推理
#ifndef _mlp_gern_
#define _mlp_gern_

#include "../matrix/matrix_gern.h"
#include "activator.h"
#include <stdint.h>

#pragma pack(1)
typedef struct _netLyrConf{
    uint8_t Activetype;       //层激活类型
    qfix dataExtra;         
    matrix_bp existedWeightData;      //已有权重数据(matrix inLen * outLen)
    matrix_bp existedBiasData;        //已有偏置数据(matrix 1 * outLen)
}NetLyrConf;
#pragma pack(pop)

class mlpNetRef{
private:
    matrix_bp fullConnDataMid;      //全连接层
    NetLyrConf *lyrData;
    uint16_t netLyrCount;

public:
    void init(uint16_t lyrnum,NetLyrConf *netstruct);
    void infer(matrix_bp input);

    inline void resu_refAddr(matrix_bp output){output = fullConnDataMid;};
    inline void resu_copyied(matrix_bp output){
        uint16_t ElemSize = fullConnDataMid->cols * fullConnDataMid->rows;
        qfix *outDim = output->data;
        qfix *sourceDim = fullConnDataMid->data;
        for(uint16_t i = 0; i < ElemSize; i++) outDim[i] = sourceDim[i];
    };
};

class mlpNetTrainer{
private:
    matrix_bp *fullConnData;      //全连接层
    NetLyrConf *lyrData;
    uint16_t netLyrCount;

    matrix_bp *weights_T;

    matrix_bp *grad_weights;
    matrix_bp *grad_to_last;

public:
    void init(uint16_t lyrnum,NetLyrConf *netstruct);
    void infer(matrix_bp input);
    
    void backward(matrix_bp grad_from_resu, qfix lr);

    inline void resu_refAddr(matrix_bp output){output = fullConnData[netLyrCount - 1];};
    inline void resu_copyied(matrix_bp output){
        matrix_bp resu = fullConnData[netLyrCount - 1];
        uint16_t ElemSize = resu->cols * resu->rows;
        qfix *outDim = output->data;
        qfix *sourceDim = resu->data;
        for(uint16_t i = 0; i < ElemSize; i++) outDim[i] = sourceDim[i];
    };
};

#endif
