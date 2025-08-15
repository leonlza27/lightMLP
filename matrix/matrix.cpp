#include "matrix.h"

matrix::matrix(size_t colsize,size_t rowsize):rows(rowsize),cols(colsize){
    data= (float*)malloc(sizeof(float)*rows*cols);
    for(size_t i =0; i<rows*cols;i++){
		data[i] = 0;
	}
}

void matrix::operator=(const float *datainput){
    for(size_t i =0; i<rows*cols;i++){
		data[i] = datainput[i];
	}
	
}

void matrix::operator=(const matrix &datainput){
    if(rows*cols!=datainput.cols*datainput.rows){
    free(data);
    data = (float*)malloc(sizeof(float)*rows*cols);
    }

    rows = datainput.rows;
    cols = datainput.cols;


    for(size_t i =0; i<rows*cols;i++){
		data[i] = datainput.data[i];
	}
}

matrix::matrix(const matrix& other) : rows(other.rows), cols(other.cols) {
    data = (float*)malloc(sizeof(float) * rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = other.data[i];
    }
}

float &matrix::indexVal(size_t row,size_t col){
    return data[row*cols+col];
}

float matrix::indexVal(size_t row,size_t col) const{
    return data[row*cols+col];
}

float &matrix::directIndexData(size_t index){
    return data[index];
}

float matrix::directIndexData(size_t index) const{
    return data[index];
}

matrix matrix::Transpose() const{
    matrix ret(rows,cols);

    for(size_t i =0; i<rows; i++){
        for(size_t j =0; j<cols; j++){
            ret.indexVal(j,i) = this->indexVal(i,j);
        }
    }

    return ret;
}

matrix matrix::operator+(const matrix& madd) const{
	matrix ret(cols,rows);

    for(size_t i =0;i < rows * cols;i++) ret.data[i] = data[i] + madd.data[i];
    

	return ret;
}



matrix matrix::operator*(const float mulpnum) const{
    matrix ret(cols,rows);

    for(size_t i =0;i < rows * cols;i++)
        ret.data[i] = data[i] * mulpnum;

	return ret;
}



matrix matrix::operator*(const matrix &mulpmtx) const{
    matrix ret(rows,mulpmtx.cols);

    for(size_t i = 0;i<ret.rows;i++){
        for(size_t j = 0;j<ret.cols;j++){
            float temp = 0;
            for(size_t k = 0;k<cols;k++){
                temp += this->indexVal(i,k) * mulpmtx.indexVal(k,j);
            }
            ret.indexVal(i,j) = temp;
        }
    }

    return ret;
}

matrix matrix::HadamardWith(const matrix &mulpmtx) const{
    matrix ret(cols,rows);

    for(size_t i =0;i < rows * cols;i++)
        ret.data[i] = data[i] * mulpmtx.data[i];

	return ret;
}

matrix::~matrix(){
    free(data);
}

size_t matrix::getRows() const{
    return rows;
}

size_t matrix::getCols() const{
    return cols;
}

#ifdef __ON_DBG
void matrix::opt(const char* mark){
    if(mark[0]!=0){
        printf("%s\n",mark);
    }
    for(size_t i =0; i<rows*cols;i++){
        printf("%.2f\t",data[i]);
        if(i%cols==cols-1) putc('\n',stdout);
    }
}
#endif
