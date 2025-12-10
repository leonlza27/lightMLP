#include <stdlib.h>
#include <time.h>
#include "../matrix/matrix_gern.h"
#include <stdio.h>
#define time4(expr) {\
   auto start = std::chrono::high_resolution_clock::now();\
   expr;\
   auto end = std::chrono::high_resolution_clock::now();\
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);\
   printf("elapsed time in chrono = %ld ns\n" , duration.count());\
}

int main(){
    matrix_bp m1;
    matrix_bp m2;
    matrix_bp m3;
    matrix_bp mr;

    int m = 2000, k = 280, n = 40;

    srand(time(0));

    qfix *cap1 = (qfix*)malloc(sizeof(qfix) * m * k + sizeof(matrix_bp_data));
    qfix *cap2 = (qfix*)malloc(sizeof(qfix) * k * n + sizeof(matrix_bp_data));
    qfix *cap3 = (qfix*)malloc(sizeof(qfix) * m * n + sizeof(matrix_bp_data));

    m1 = (matrix_bp)cap1;
    m1->cols = k;
    m1->rows = m;

    m2 = (matrix_bp)cap2;
    m2->cols = n;
    m2->rows = k;

    m3 = (matrix_bp)cap3;
    m3->cols = n;
    m3->rows = m;

    time4(matrix_bp_mulpty(m1,m2,m3));

}