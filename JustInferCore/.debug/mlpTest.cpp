#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include "../mlp/mlp.h"
#include <chrono>
#include <stdlib.h>

#define time4(expr) {\
   auto start = std::chrono::high_resolution_clock::now();\
   expr;\
   auto end = std::chrono::high_resolution_clock::now();\
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);\
   printf("elapsed time in chrono = %ld ns\n" , duration.count());\
}

mlpNetRef test;

int main(){
    uint16_t in = 200;
    uint16_t out = 30;

    matrix_bp W = (matrix_bp)malloc(sizeof(matrix_bp_data) + sizeof(qfix) * in * out);
    matrix_bp B = (matrix_bp)malloc(sizeof(matrix_bp_data) + sizeof(qfix) * out);
    matrix_bp in_sim = (matrix_bp)malloc(sizeof(matrix_bp_data) + sizeof(qfix) * in);

    for(uint16_t i = 0; i < out * in; i++){
       W->data[i] = float_to_qfix(1.0f);
    }

    for(uint16_t i = 0; i < out; i++){
       B->data[i] = float_to_qfix(1.0f);
    }

    for(uint16_t i = 0; i < in; i++){
       in_sim->data[i] = float_to_qfix(4.0f);
    }


    W->rows = out;
    W->cols = in;

    B->rows = out;
    B->cols = 1;

    in_sim->rows = in;
    in_sim->cols = 1;

    NetLyrConf cnf;

    cnf.Activetype = type_ReLU;
    cnf.existedWeightData = W;
    cnf.existedBiasData = B;

    test.init(1, &cnf);

    time4(test.infer(in_sim));

/*
    matrix_bp out_sim = (matrix_bp)malloc(sizeof(matrix_bp_data) + sizeof(qfix) * out);

    test.resu_copyied(out_sim);

    for(uint16_t i = 0; i < out; i++){
        printf("%f\t",qfix_to_float(out_sim->data[i]));
    }

    printf("\n");
    */

}