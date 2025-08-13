#ifndef _matrix_
#define _matrix_

#include <malloc.h>

#ifdef __ON_DBG
#include <stdio.h>
#endif

class matrix{
private:
    double* data;
    unsigned int rows,cols;//行,列
public:
    //初始化:列,行
    matrix(unsigned int colsize,unsigned int rowsize);
    matrix(const matrix& other);

    double& indexVal(unsigned int row,unsigned int col);
    double indexVal(unsigned int row,unsigned int col) const;
    double& directIndexData(unsigned int index);
    double directIndexData(unsigned int index) const;

    void operator=(const double *datainput);
    void operator=(const matrix& datainput);   

    matrix Transpose() const;

    matrix operator+(const matrix& madd) const;

    matrix operator*(const double mulpnum) const;
    matrix operator*(const matrix& mulpmtx) const;

    matrix HadamardWith(const matrix& mulpmtx) const; 


    ~matrix();

    unsigned int getRows() const;
    unsigned int getCols() const;

    #ifdef __ON_DBG

    void opt();

    #endif
};

#endif
