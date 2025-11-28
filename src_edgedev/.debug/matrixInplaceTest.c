#include <stdio.h>
#include "../matrix/matrix_static.h"
#include "../qfix_ops/qfix_op.h"

int main(){
    matrix_bp m1 = new_matrix_bp();
    matrix_bp m2 = new_matrix_bp();
    matrix_bp m3 = new_matrix_bp();

    qfix cap1[] = {float_to_qfix(1.0f),float_to_qfix(2.0f),float_to_qfix(1.0f),float_to_qfix(2.0f)};
    qfix cap2[] = {float_to_qfix(1.0f),float_to_qfix(2.0f),float_to_qfix(1.0f),float_to_qfix(2.0f)};
    qfix cap3[4] = {};

    m1->cols = 2;
    m1->rows = 2;
    m1->data = cap1;

    m2->cols = 2;
    m2->rows = 2;
    m2->data = cap2;

    m3->cols = 2;
    m3->rows = 2;
    m3->data = cap3;

    matrix_bp_mulpty(m1, m2, m3);
    matrix_bp_mulpty(m1, m2, m1);

    int is_eq = 1;
    for (int i = 0; i < 2; i++){
        if(m1->data[i] != m3->data[i]){
            is_eq = 0;
            break;
        }
    }

    printf("%d\n", is_eq);
}