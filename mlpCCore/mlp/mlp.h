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

void mlptrainer_setup(uint32_t calclyrs ,netLyrConf *net, mlpTrainStatus *dest);
void mlptrainer_cleanup(mlpTrainStatus *net);

void mlptrainer_backward(mlpTrainStatus *model, qfix *grad_inital, qfix lr);
void mlptrainer_execute(mlpTrainStatus *model, qfix *arrin);

#endif