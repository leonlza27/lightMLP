//mlp:仅用作mcu上推理
#ifndef _mlp_static_
#define _mlp_static_

#include "../matrix/matrix_static.h"
#include <stdint.h>

typedef struct _netLyrAllocator{
    uint16_t inputs;                //层输入
    uint16_t outputs;               //层输出
    unsigned char Activetype;       //层激活类型
    f_q16_16 *existedWeightData;    //已有权重数据
    f_q16_16 *existedBiasData;      //已有偏置数据
}NetLyrAllocator;

class mlp_calclyr{
private:
    p_matrix_Q16_16 weights,bias;
    unsigned char ActiveType;   //激活类型
    p_matrix_Q16_16 tmp;        //激活前数据存储(防止运算动态分配)

public:
    void confLyr(NetLyrAllocator lyrConf);
    void fwdCalc(const p_matrix_Q16_16 input,p_matrix_Q16_16 output);
};

class mlpNet_onMCU{
private:
    p_matrix_Q16_16 *fullConnData;      //全连接层([0]:网络输入 [<last>]:网络输出)
    mlp_calclyr *NetLyrs;
public:
    mlpNet_onMCU(uint16_t lyrnum,NetLyrAllocator *netstruct);
    void infer();
};

#endif