#ifndef _mlp_core
#define _mlp_core

#include "../CustomConf.h"
#include "builtin_ac_callback.h"

#ifdef CORE_DBG
#undef DLLEXPORT
#define DLLEXPORT
#endif

typedef struct netLyrConf{
    uint8_t acTp;
    uint16_t in_dim, out_dim;
    qfix *weights, *bias, dExtra;
}netLyrConf;

typedef struct _mlp_train_status{
    uint32_t calclyrs; 
    netLyrConf *modelsrc;
    qfix **fullConnData;    //0为梯度累积器
    qfix **va_wxb;          //激活(非线性)前原始数据用于反向,累计器同样为0
    qfix **w_grad;
    qfix **b_grad;
    qfix **lyrinput_grad;   //整层输入梯度,累计器时为0
}mlpTrainStatus;

typedef struct _mlp_exec_status{
    uint32_t calclyrs;
    netLyrConf *modelsrc;
    qfix *fullConnData_tmp[2];
}mlpExecStatus;

void DLLEXPORT mlptrain_setup(const uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void DLLEXPORT mlptrain_collector_setup(const uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void DLLEXPORT mlptrain_cleanup(mlpTrainStatus *net);
void DLLEXPORT mlptrain_execute(const mlpTrainStatus *model, const qfix *arrin);

void DLLEXPORT mlptrain_collector_collect(const mlpTrainStatus* net_or_gradcap,
                                               const mlpTrainStatus* gradscap_dest, const qfix lr_cur);

void DLLEXPORT mlptrain_backward(const mlpTrainStatus *model, const qfix *grad_inital, const qfix lr);
void DLLEXPORT mlptrain_update_from_collector(const mlpTrainStatus *model, const mlpTrainStatus *gradscap);

//给卷积用的

void DLLEXPORT mlptrain_convcalc_zerobia(const mlpTrainStatus *model);


void DLLEXPORT mlpexec_setup(const uint32_t calclyrs ,const netLyrConf *net, mlpExecStatus *dest);
void DLLEXPORT mlpexec_cleanup(mlpExecStatus *net);

void DLLEXPORT mlpexec_execute(const mlpExecStatus *model, const qfix *arrin);

//多线程计算拓展
void DLLEXPORT mlptrain_execute_mt(const mlpTrainStatus *model, const qfix *arrin);

void DLLEXPORT mlptrain_collector_collect_mt(const mlpTrainStatus* net_or_gradcap,
                                               const mlpTrainStatus* gradscap_dest, const qfix lr_cur);

void DLLEXPORT mlptrain_backward_mt(const mlpTrainStatus *model, const qfix *grad_inital, const qfix lr);
void DLLEXPORT mlptrain_update_from_collector_mt(const mlpTrainStatus *model, const mlpTrainStatus *gradscap);
void DLLEXPORT mlpexec_execute_mt(const mlpExecStatus *model, const qfix *arrin);

#define mlptrain_resu_ptr(_pmlp) (_pmlp)->fullConnData[(_pmlp)->calclyrs]
#define mlptrain_checkcap(_pmlp) (_pmlp)->fullConnData
#define mlpexec_resu_ptr(_pmlp) (_pmlp)->fullConnData_tmp[((_pmlp)->calclyrs + 1) % 2]

#endif
