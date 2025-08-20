#include <stdio.h>

#include "matrix/matrix_static.h"


int main(){

    matrix_Q16_16_data *mat1 = alloc_matrix_Q16_16(),*mat2 = alloc_matrix_Q16_16();

    f_q16_16 data[2]= {float_to_q16_16(1.0f),float_to_q16_16(2.0f)};

    matrix_Q16_16_init(mat1,1,2,data);
    matrix_Q16_16_init(mat2,2,1,data);

    DbgPrint_Q16_16_matrix(mat1,"mat1");
    putc('\n',stdout);
    DbgPrint_Q16_16_matrix(mat2,"mat2");
    putc('\n',stdout);

    matrix_Q16_16_data *resu = alloc_matrix_Q16_16();
    f_q16_16 resucast[4] = {0,0,0,0};
    matrix_Q16_16_init(resu,2,2,resucast);

    matrix_Q16_16_mulpty(mat1,mat2,resu);

    DbgPrint_Q16_16_matrix(resu,"resu");

    return 0;
}