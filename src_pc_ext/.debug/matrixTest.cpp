#include <stdlib.h>
#include <time.h>
#include "../matrix/matrix_gern.h"
#include <stdio.h>
#define time4(expr) {\
   auto start = std::chrono::high_resolution_clock::now();\
   expr;\
   auto end = std::chrono::high_resolution_clock::now();\
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);\
   printf("elapsed time in chrono = %f ms\n" , duration.count()/1000000.0f);\
}

void matrix_bp_mulpty_raw(const matrix_bp_data *mmul1, const matrix_bp_data *mmul2, matrix_bp_data *resu) {
    resu->cols = mmul2->cols;
    resu->rows = mmul1->rows;
    const uint16_t _k = mmul1->cols,
                    out_rows = mmul1->rows,
                    out_cols = mmul2->cols;

    bp *in1_base = (bp*)mmul1->data,
           *in2_base = (bp*)mmul2->data,
           *out_base = resu->data;
           
    for(uint16_t i = 0; i < out_rows; i++){
        for(uint16_t j = 0; j < out_cols; j++){
            _tmp_larger sum = 0;
            for(uint16_t k = 0; k < _k; k++){
                sum += ((_tmp_larger)in1_base[i * _k + k] * in2_base[k * out_cols + j]);
            }
            out_base[i * out_cols + j] = (bp)((sum + (1LL << (QSHIFT - 1))) >> QSHIFT);
        }
    }
}

int main(){
    //matrix_bp m1;
    //matrix_bp m2;
    matrix_bp m3;
    matrix_bp mr;

    int m = 500, k = 3, n = 500;

    srand(time(0));

    //qfix *cap1 = (qfix*)malloc(sizeof(qfix) * m * k + sizeof(matrix_bp_data));
    //qfix *cap2 = (qfix*)malloc(sizeof(qfix) * k * n + sizeof(matrix_bp_data));
    qfix *cap3 = (qfix*)malloc(sizeof(qfix) * m * n + sizeof(matrix_bp_data));
    qfix *capr = (qfix*)malloc(sizeof(qfix) * m * n + sizeof(matrix_bp_data));


    //m1 = (matrix_bp)cap1;
    //m1->cols = k;
    //m1->rows = m;

    //m2 = (matrix_bp)cap2;
    //m2->cols = n;
    //m2->rows = k;

    m3 = (matrix_bp)cap3;
    m3->cols = n;
    m3->rows = m;

    mr = (matrix_bp)capr;
    mr->cols = m;
    mr->rows = n;

    //for(int i = 0; i < m * k; i++) m1->data[i] = rand() % (1 << 19);
    //for(int i = 0; i < n * k; i++) m2->data[i] = rand() % (1 << 19);
    

    /*
    m1->data[0] = float_to_qfix(1.0f);
    m1->data[1] = float_to_qfix(2.0f);
    m1->data[2] = float_to_qfix(3.0f);
    m1->data[3] = float_to_qfix(4.0f);
    
    m2->data[0] = float_to_qfix(1.0f);
    m2->data[1] = float_to_qfix(2.0f);
    */

    time4(matrix_bp_transpose(m3, mr);)

    //for(int i = 0; i < m3->rows; i++){
    //    for(int j = 0; j < m3->cols; j++){
    //        printf("%d\t",m3->data[i * m3->cols + j]);
    //    }
    //    putc('\n',stdout);
    //}
//
    //for(int i = 0; i < mr->rows; i++){
    //    for(int j = 0; j < mr->cols; j++){
    //        printf("%d\t",mr->data[i * mr->cols + j]);
    //    }
    //    putc('\n',stdout);
    //}

}