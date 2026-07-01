#ifndef _mlp_core
#define _mlp_core

#include "../CustomConf.h"
#include "ac_callback.h"
#include <stdlib.h>

#ifdef CORE_DBG
#undef DLLEXPORT
#define DLLEXPORT
#endif

typedef struct _netlyrcnf{
    uint8_t acTp;
    uint16_t in_dim, out_dim;
    qfix *weights, *bias, dExtra;
}netLyrConf;

typedef struct _mlp_train_status{
    uint32_t calclyrs; 
    netLyrConf *modelsrc;
    qfix **fullConnData; //0为梯度累积器
    qfix **w_grad;
    qfix **lyrinput_grad;
}mlpTrainStatus;

typedef struct _mlp_exec_status{
    uint32_t calclyrs;
    netLyrConf *modelsrc;
    qfix *fullConnData_tmp[2];
}mlpExecStatus;

void DLLEXPORT mlptrainer_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void DLLEXPORT mlptrainer_totalgrads_cap_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void DLLEXPORT mlptrainer_cleanup(mlpTrainStatus *net);
void DLLEXPORT mlptrainer_execute(mlpTrainStatus *model, qfix *arrin);

//支持数据并行

void DLLEXPORT mlptrainer_totalgrads_savegrads(mlpTrainStatus *net_or_gradcap, mlpTrainStatus *gradscap_dest);

void DLLEXPORT mlptrainer_backward(mlpTrainStatus *model, qfix *grad_inital, qfix lr);
void DLLEXPORT mlptrainer_totalgrads_backward(mlpTrainStatus *model, mlpTrainStatus *gradscap, qfix lr);

//给卷积用的

void DLLEXPORT mlptrainer_convcalc_zerobia(mlpTrainStatus *model);

#define mlptrain_finalgrads_ref(mlptrain) mlptrain->lyrinput_grad[0]
#define mlptrain_resu_ref(mlptrain) mlptrain->fullConnData[mlptrain->calclyrs]

void DLLEXPORT mlpexec_setup(uint32_t calclyrs ,netLyrConf *net, mlpExecStatus *dest);
void DLLEXPORT mlpexec_cleanup(mlpExecStatus *net);

void DLLEXPORT mlpexec_execute(mlpExecStatus *model, qfix *arrin);

#define mlpexec_resu_ref(mlpexec) mlpexec->fullConnData_tmp[(mlpexec->calclyrs + 1) % 2]

#endif
