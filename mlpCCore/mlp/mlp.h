#ifndef _mlp_core
#define _mlp_core

#include "../CustomConf.h"
#include "ac_callback.h"

typedef struct _netlyrcnf{
    uint8_t acTp;
    uint16_t in_dim, out_dim;
    qfix *weights, *bias, dExtra;
}netLyrConf;

typedef struct _mlp_train_status{
    uint32_t calclyrs; 
    netLyrConf *modelsrc;
    qfix **fullConnData;
    qfix **w_grad;
    qfix **lyrinput_grad;
}mlpTrainStatus;

typedef struct _mlp_exec_status{
    uint32_t calclyrs;
    netLyrConf *modelsrc;
    qfix *fullConnData_tmp[2];
}mlpExecStatus;

void mlptrainer_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void mlptrainer_totalgrads_cap_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void mlptrainer_cleanup(mlpTrainStatus *net);

void mlptrainer_totalgrads_savegrads(mlpTrainStatus *net_or_gradcap, mlpTrainStatus *gradscap_dest);

void mlptrainer_backward(mlpTrainStatus *model, qfix *grad_inital, qfix lr);
void mlptrainer_totalgrads_backward(mlpTrainStatus *model, mlpTrainStatus *gradscap, qfix lr);
void mlptrainer_execute(mlpTrainStatus *model, qfix *arrin);

void mlpexec_setup(uint32_t calclyrs ,netLyrConf *net, mlpExecStatus *dest);
void mlpexec_cleanup(mlpExecStatus *net);

void mlpexec_execute(mlpExecStatus *model, qfix *arrin);


#endif
