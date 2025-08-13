#include <stdio.h>
#include "matrix/matrix.h"

int main(){

    matrix test(3,2),add1(2,3);
    double data1[] = {1.0,0.0,2.0,-1.0,3.0,1.0};
    double data2[] = {3.0,1.0,2.0,1.0,1.0,0.0};
    test = data1;
    add1 = data2;

    test.opt();
    putc('\n',stdout);
    add1.opt();
    putc('\n',stdout);

    matrix resu = test * add1;
    resu.opt();

    
    return 0;
}