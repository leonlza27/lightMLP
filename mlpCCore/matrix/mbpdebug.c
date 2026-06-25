#include "matrix_static.h"
#include <stdio.h>


int main(){
    matrix_bp m1 = alloc_matrix_bp(20,30),m2 = alloc_matrix_bp(20,30);
    for(int i = 0; i < 100; i++){
        matrix_bp m3 = alloc_matrix_bp(20,30);
        printf("loop %d:", i);
        matrix_bp_sub(m1, m2, m3);
        for(int j = 0; j < 60; j ++) printf("[%d]=%d ", j, m3->data[j]);
        free(m3);
    }
}