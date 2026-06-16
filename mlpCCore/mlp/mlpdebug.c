#include "ac_callback.h"
#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <winnt.h>

//改为arena allocate
void mlptrainer_setup_1(uint32_t calclyrs, netLyrConf *net, mlpTrainStatus *dest){
    dest->calclyrs = calclyrs;
    dest->modelsrc = net;

    size_t all_size = (calclyrs * 3 + 2) * sizeof(qfix*);

    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
        all_size += (outdim + indim + outdim * indim) * sizeof(qfix);
    }

    
    HANDLE _heap = HeapCreate(HEAP_NO_SERIALIZE | HEAP_CREATE_ENABLE_EXECUTE, all_size, all_size);
    //char *fullmem = malloc(all_size);
    char *fullmem = HeapAlloc(_heap, HEAP_ZERO_MEMORY, all_size);
    int resu = VirtualLock(fullmem, all_size);//对windows的美好幻想但解决不了训练换页的问题
    printf("lockstat: %d\n", resu);
    
    //dest->fullConnData = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    //dest->lyrinput_grad = (qfix**)malloc(sizeof(qfix*) * (calclyrs + 1));
    //dest->w_grad = (qfix**)malloc(sizeof(qfix*) * calclyrs);

    dest->fullConnData = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->lyrinput_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * (calclyrs + 1);
    dest->w_grad = (qfix**)fullmem;
    fullmem += sizeof(qfix*) * calclyrs;

    for(uint32_t i = 0; i < calclyrs; i++){
        uint16_t indim = net[i].in_dim;
        uint16_t outdim = net[i].out_dim;
        //dest->fullConnData[i + 1] = (qfix*)malloc(sizeof(qfix) * outdim);
        //dest->lyrinput_grad[i] = (qfix*)malloc(sizeof(qfix) * indim);
        //dest->w_grad[i] = (qfix*)malloc(sizeof(qfix) * indim * outdim);
        dest->fullConnData[i + 1] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * outdim;
        dest->lyrinput_grad[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * indim;
        dest->w_grad[i] = (qfix*)fullmem;
        fullmem += sizeof(qfix) * indim * outdim;
    }
}

//这个证明windows在连续调用malloc会出现非0无效地址问题而代码逻辑并无问题<line:14>
int main(){
    qfix w0[6];
    qfix b0[3];
    qfix w1[3];
    qfix b1[1];
    netLyrConf net[] = {{ac_Tanh,2,3,w0,b0,0},{ac_Tanh,3,1,w1,b1,0}};

    mlpTrainStatus ntrain, ngradsave;
#ifndef manual
    mlptrainer_setup(2, net, &ntrain);
    mlptrainer_totalgrads_cap_setup(2, net, &ngradsave);
#else
    ntrain.calclyrs = 2;
    ntrain.modelsrc = net;

    qfix *fullconndatas[3];
    qfix fullconndata1[3];
    qfix fullconndata2[1];
    fullconndatas[1] = fullconndata1;
    fullconndatas[2] = fullconndata2;
    ntrain.fullConnData = fullconndatas;

    qfix *w_ds[2];
    qfix w_d1[6];
    qfix w_d2[3];
    w_ds[0] = w_d1;
    w_ds[1] = w_d2;
    ntrain.w_grad = w_ds;

    qfix *x_ls[2];
    qfix x_l1[2];
    qfix x_l2[3];
    x_ls[0] = x_l1;
    x_ls[1] = x_l2;
    ntrain.lyrinput_grad = x_ls;
#endif
    for(int i = 0; i < 500; i++){
        printf("simutlate train epoch %d: fullconndata: %p\nforwarding\n", i, ntrain.fullConnData);
        mlptrainer_execute(&ntrain, w0);
        qfix simresu = 3;
        printf("backwarding\n");
        mlptrainer_backward(&ntrain, &simresu, 0);
        printf("saving grads\n");
        mlptrainer_totalgrads_savegrads(&ntrain, &ngradsave);
    }
    mlptrainer_cleanup(&ntrain);
    mlptrainer_cleanup(&ngradsave);
    mlpExecStatus nexec;
    mlpexec_setup(2, net, &nexec);
    for(int i = 0; i < 500; i++){
        printf("simutlate exec epoch %d\n", i);
        mlpexec_execute(&nexec, w0);
    }
    mlpexec_cleanup(&nexec);
}