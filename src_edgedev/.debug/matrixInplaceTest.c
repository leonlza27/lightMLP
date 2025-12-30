#include <stdio.h>
#include "../matrix/matrix_static.h"
#include "../qfix_ops/qfix_op.h"
#include <stdlib.h>
#include <time.h>

int main(){
    matrix_bp m1;
    matrix_bp m2;
    matrix_bp m3;
    matrix_bp mr;

    int m = 2, k = 28, n = 4;

    srand(time(0));

    qfix cap1[64] = {};
    qfix cap2[64] = {};
    qfix cap3[64] = {};
    qfix capr[64] = {};

    for(int i = 0; i < 64; i++) cap1[i] = rand() % (1 << 17);
    for(int i = 0; i < 64; i++) cap2[i] = rand() % (1 << 17);

    m1 = (matrix_bp)cap1;
    m1->cols = k;
    m1->rows = m;

    m2 = (matrix_bp)cap2;
    m2->cols = n;
    m2->rows = k;

    m3 = (matrix_bp)cap3;
    m3->cols = n;
    m3->rows = m;
    
    mr = (matrix_bp)capr;
    mr->cols = n;
    mr->rows = m;

    matrix_bp_mulpty(m1, m2, mr);

    matrix_bp_mulpty(m1, m2, m1);

    
    for(uint16_t i = 0; i < m * n; i++){
        if(m1->data[i] != mr->data[i]) return 1;
    }
}
