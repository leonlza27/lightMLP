#ifndef _matrix_
#define _matrix_

#include <malloc.h>
#include "../wrapper/wrapper.h"
#include <cstddef>

#ifdef __ON_DBG
#include <stdio.h>
#endif

class matrix{
private:
    float* data;
    size_t rows,cols;//行,列
public:
    //初始化:列,行
    matrix(size_t colsize,size_t rowsize);
    matrix(const matrix& other);

    double& indexVal(size_t row,size_t col);
    double indexVal(size_t row,size_t col) const;
    double& directIndexData(size_t index);
    double directIndexData(size_t index) const;

    void operator=(const double *datainput);
    void operator=(const matrix& datainput);   

    matrix Transpose() const;

    matrix operator+(const matrix& madd) const;

    matrix operator*(const double mulpnum) const;
    matrix operator*(const matrix& mulpmtx) const;

    matrix HadamardWith(const matrix& mulpmtx) const; 


    ~matrix();

    size_t getRows() const;
    size_t getCols() const;

    #ifdef __ON_DBG

    void opt(const char* mark = "");

    #endif
};

#endif
