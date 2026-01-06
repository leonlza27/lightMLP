#include <stdio.h>
#include "../mlp/mlp.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(){
    int rndfd = open("/dev/urandom", O_RDONLY);
    if (rndfd == -1) {
        perror("open");
        exit(1);
    }
    
    matrix_bp w = (matrix_bp)malloc(sizeof(matrix_bp_data) + 2 * sizeof(qfix));
    matrix_bp b = (matrix_bp)malloc(sizeof(matrix_bp_data) + 1 * sizeof(qfix));
    
    w->rows = 1;
    w->cols = 2;
    w->data[0] = 346;
    w->data[1] = 368;
    
    b->rows = 1;
    b->cols = 1;
    b->data[0] = 5000;
    
    matrix_bp in = (matrix_bp)malloc(sizeof(matrix_bp_data) + 2 * sizeof(qfix));
    in->rows = 2;
    in->cols = 1;
    
    matrix_bp out = (matrix_bp)malloc(sizeof(matrix_bp_data) + 1 * sizeof(qfix));
    out->rows = 1;
    out->cols = 1;
    
    matrix_bp loss = (matrix_bp)malloc(sizeof(matrix_bp_data) + 1 * sizeof(qfix));
    loss->rows = 1;
    loss->cols = 1;
    
    NetLyrConf _orProblem = {type_Tanh, 0, w, b};
    
    mlpNetTrainInfo _model;
    
    lmlp_setupTrainer(1, &_orProblem, &_model);
    
    uint8_t correct = 0;
    
    for(int i = 0; i < 1000; i++){
        uint8_t rnd1;
        uint8_t rnd2;
        read(rndfd, &rnd1, sizeof(uint8_t));
        in->data[0] = (rnd1%16 - 8) << QSHIFT;
        read(rndfd, &rnd2, sizeof(uint8_t));
        in->data[1] = (rnd2%16 - 8) << QSHIFT;
        out->data[0] = ((in->data[0] > 0 && in->data[1] > 0)? 1 : -1) << QSHIFT;
        lmlp_trainer_infer(_model, in);
        matrix_bp_sub(out ,_model.fullConnData[1], loss);
        //printf("ep %d loss %d (datain %d(%d) %d(%d), exc %d)\n",i , loss->data[0], rnd1%16 - 8, in->data[0], rnd2%16 - 8, in->data[1], out->data[0]);
        lmlp_trainer_backward(_model, loss, float_to_qfix(0.3f));
    }
    
    printf("\n");
    
    for(int i = 0; i < 200; i++){
        uint8_t rnd1;
        uint8_t rnd2;
        read(rndfd, &rnd1, sizeof(uint8_t));
        in->data[0] = (rnd1%16 - 8) << QSHIFT;
        read(rndfd, &rnd2, sizeof(uint8_t));
        in->data[1] = (rnd2%16 - 8) << QSHIFT;
        lmlp_trainer_infer(_model, in);
        printf("%f (datain %d(%d) %d(%d))\n",qfix_to_float(_model.fullConnData[1]->data[0]), rnd1%16 - 8, in->data[0], rnd2%16 - 8, in->data[1]);
        correct += ((in->data[0] > 0 && in->data[1] > 0) == _model.fullConnData[1]->data[0] > 0);
        
    }
    
    printf("\ncorrect:%.4f\n", (double)correct/200.0f);
    
    close(rndfd);
}
