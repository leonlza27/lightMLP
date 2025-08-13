#include "matrix.h"

matrix::matrix(unsigned int colsize,unsigned int rowsize):rows(rowsize),cols(colsize){
    data= (double*)malloc(sizeof(double)*rows*cols);
    for(int i =0; i<rows*cols;i++){
		data[i] = 0;
	}
}

void matrix::operator=(const double *datainput){
    for(unsigned int i =0; i<rows*cols;i++){
		data[i] = datainput[i];
	}
	
}

void matrix::operator=(const matrix &datainput){
    if(rows*cols!=datainput.cols*datainput.rows){
    free(data);
    data = (double*)malloc(sizeof(double)*rows*cols);
    }

    rows = datainput.rows;
    cols = datainput.cols;


    for(int i =0; i<rows*cols;i++){
		data[i] = datainput.data[i];
	}
}

matrix::matrix(const matrix& other) : rows(other.rows), cols(other.cols) {
    data = (double*)malloc(sizeof(double) * rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        data[i] = other.data[i];
    }
}

double &matrix::indexVal(unsigned int row,unsigned int col){
    return data[row*cols+col];
}

double matrix::indexVal(unsigned int row,unsigned int col) const{
    return data[row*cols+col];
}

double &matrix::directIndexData(unsigned int index){
    return data[index];
}

double matrix::directIndexData(unsigned int index) const{
    return data[index];
}

matrix matrix::Transpose() const{
    matrix ret(rows,cols);

    for(int i =0; i<rows; i++){
        for(int j =0; j<cols; j++){
            ret.indexVal(j,i) = this->indexVal(i,j);
        }
    }

    return ret;
}

matrix matrix::operator+(const matrix& madd) const{
	matrix ret(cols,rows);
    
    for(int i =0;i<cols*rows;i++){
        ret.data[i] = data[i] + madd.data[i];
    }

	return ret;
}



matrix matrix::operator*(const double mulpnum) const{
    matrix ret(cols,rows);

    for(int i =0;i<cols*rows;i++){
        ret.data[i] = data[i] * mulpnum;
    }

	return ret;
}



matrix matrix::operator*(const matrix &mulpmtx) const{
    matrix ret(rows,mulpmtx.cols);

    for(int i = 0;i<ret.rows;i++){
        for(int j = 0;j<ret.cols;j++){
            double temp = 0;
            for(int k = 0;k<cols;k++){
                temp += this->indexVal(i,k) * mulpmtx.indexVal(k,j);
            }
            ret.indexVal(i,j) = temp;
        }
    }

    return ret;
}

matrix matrix::HadamardWith(const matrix &mulpmtx) const{
    matrix ret(cols,rows);

    for(int i =0;i<cols*rows;i++){
        ret.data[i] = data[i] * mulpmtx.data[i];
    }

	return ret;
}

matrix::~matrix(){
    free(data);
}

unsigned int matrix::getRows() const{
    return rows;
}

unsigned int matrix::getCols() const{
    return cols;
}

#ifdef __ON_DBG
void matrix::opt(){
    for(int i =0; i<rows*cols;i++){
        printf("%.2f\t",data[i]);
        if(i%cols==cols-1) putc('\n',stdout);
    }
}
#endif
