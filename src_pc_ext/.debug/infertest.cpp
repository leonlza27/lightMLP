#include <stdio.h>
#include "../mlp/mlp.h"

#define allocMatrix(m, n) (matrix_bp)malloc(sizeof(matrix_bp_data) + sizeof(qfix) * m * n);



int main(){
    mlpNetTrainer test;

    int in_dim = 2;
    int out_dim = 2;

    matrix_bp W = allocMatrix(in_dim, out_dim);
    matrix_bp B = allocMatrix(out_dim, 1);

    W->rows = out_dim;
    W->cols = in_dim;

    B->cols = 1;
    B->rows = out_dim;

    matrix_bp datain = allocMatrix(in_dim, 1);
    matrix_bp datares = allocMatrix(out_dim, 1);
    matrix_bp loss = allocMatrix(out_dim, 1);

    datain->rows = in_dim;
    datain->cols = 1;

    datares->rows = out_dim;
    datares->cols = 1;

    loss->rows = out_dim;
    loss->cols = 1;

    datain->data[0] = float_to_qfix(1.f);
    datain->data[1] = float_to_qfix(4.f);

    datares->data[0] = float_to_qfix(2.f);
    datares->data[1] = float_to_qfix(2.f);

    NetLyrConf testStruct;

    testStruct.Activetype = -1;
    testStruct.existedBiasData = B;
    testStruct.existedWeightData = W;

    test.init(1, &testStruct);

    for(int i = 0; i < 100; i++){
        test.infer(datain);
        matrix_bp_sub(test.resu_refAddr(), datares, loss);
        test.backward(loss, float_to_qfix(.1f));
        
    }
    formatMatrix2f(test.resu_refAddr())
    formatMatrix2f(W)
    formatMatrix2f(B)

}