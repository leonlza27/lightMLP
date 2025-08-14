#include <stdio.h>
#include "matrix/matrix.h"

int main(){

    matrix test(3,2),add1(3,2);
    double data1[] = {1.0,0.0,2.0,-1.0,3.0,1.0};
    double data2[] = {3.0,1.0,2.0,1.0,1.0,0.0};
    test = data1;
    add1 = data2;

    test.opt("test");
    putc('\n',stdout);
    add1.opt("adder");
    putc('\n',stdout);

    matrix resu1 = test + add1;
    resu1.opt("op +");
    putc('\n',stdout);

    matrix resu2 = test * 2.0;
    resu2.opt("op * num");
    putc('\n',stdout);

    matrix resu3 = test.HadamardWith(add1);
    resu3.opt("op hadamard *");

    
    return 0;
}