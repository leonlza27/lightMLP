#ifndef _mlp_purc_
#define _mlp_purc_

#include "../matrix/matrix_static.h"
#include "activator.h"
#include <stdint.h>

#pragma pack(1)
typedef struct _netLyrConf{
    uint8_t Activetype;       //层激活类型
    qfix dataExtra;         
    matrix_bp existedWeightData;      //已有权重数据(matrix out_dim * in_dim)
    matrix_bp existedBiasData;        //已有偏置数据(matrix out_dim * 1)
}NetLyrConf;
#pragma pack(pop)


typedef struct _data_mlpNetRef{
    matrix_bp fullConnDataMid[2];      //全连接层
    NetLyrConf *lyrData;
    uint16_t netLyrCount;
}mlpNetRefInfo;

typedef struct _data_mlpNetTrainer{
    matrix_bp *fullConnData;      //全连接层
    NetLyrConf *lyrData;
    uint16_t netLyrCount;

    //训练用缓存数据,每次被操作层以backCalc的lyridx为准

    matrix_bp *weights_T;       //层W^T缓存

    matrix_bp *grad_weights;    //W梯度(初始)缓存
    matrix_bp *grad_to_last;    //到上一层的初始梯度(dL[last]/da[last])
    matrix_bp *grad_this_det;   //层 da/dy
}mlpNetTrainInfo;

void lmlp_setupRef(uint16_t lyrnum, NetLyrConf *netstruct, mlpNetRefInfo *dest);
void lmlp_setupTrainer(uint16_t lyrnum, NetLyrConf *netstruct, mlpNetTrainInfo *dest);

void lmlp_ref_infer(mlpNetRefInfo net, matrix_bp input);
void lmlp_trainer_infer(mlpNetTrainInfo net, matrix_bp input);

void lmlp_trainer_backward(mlpNetTrainInfo net, matrix_bp grad_from_resu, qfix lr);

/*
class mlpNetRef{
private:
    matrix_bp fullConnDataMid[2];      //全连接层
    NetLyrConf *lyrData;
    uint16_t netLyrCount;

public:
    void init(uint16_t lyrnum,NetLyrConf *netstruct);
    void infer(matrix_bp input);

    inline matrix_bp resu_refAddr(){return fullConnDataMid[(netLyrCount - 1)%2];}
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

    //训练用缓存数据,每次被操作层以backCalc的lyridx为准

    matrix_bp *weights_T;       //层W^T缓存

    matrix_bp *grad_weights;    //W梯度(初始)缓存
    matrix_bp *grad_to_last;    //到上一层的初始梯度(dL[last]/da[last])
    matrix_bp *grad_this_det;   //层 da/dy

    inline void backCalc(uint16_t lyridx, matrix_bp grad, qfix lr);

public:
    void init(uint16_t lyrnum,NetLyrConf *netstruct);
    void infer(matrix_bp input);
    
    void backward(matrix_bp grad_from_resu, qfix lr);

    inline matrix_bp resu_refAddr(){return fullConnData[netLyrCount];}
    inline void resu_copyied(matrix_bp output){
        matrix_bp resu = fullConnData[netLyrCount];
        uint16_t ElemSize = resu->cols * resu->rows;
        qfix *outDim = output->data;
        qfix *sourceDim = resu->data;
        for(uint16_t i = 0; i < ElemSize; i++) outDim[i] = sourceDim[i];
    };
};
*/

#endif
