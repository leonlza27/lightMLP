//mlp:仅用作mcu上推理
#ifndef _mlp_static_
#define _mlp_static_

#include "../matrix/matrix_static.h"
#include "activator.h"
#include <stdint.h>

typedef struct _netLyrAllocator{
    qfix *existedWeightData;      //已有权重数据
    qfix *existedBiasData;        //已有偏置数据
    uint16_t inputs;                //层输入
    uint16_t outputs;               //层输出
    unsigned char Activetype;       //层激活类型
    qfix dataExtra;         
}NetLyrAllocator;

class mlp_calclyr{
private:
    matrix_bp tmp;        //激活前数据存储(防止运算动态分配)
    matrix_bp weights,bias;
    unsigned char ActiveType;   //激活类型
    qfix _alpha;

public:
    void confLyr(NetLyrAllocator lyrConf);
    void fwdCalc(const matrix_bp input, matrix_bp output);
};

class mlpNetRef{
private:
    matrix_bp_data *fullConnData;      //全连接层([<last>]:网络输出)
    mlp_calclyr *NetLyrs;
    uint16_t netLyrCount;
public:
    mlpNetRef(uint16_t lyrnum,NetLyrAllocator *netstruct);
    void infer(matrix_bp input);
    void getOutput(matrix_bp output);
};

#endif