#ifndef _mlp_
#define _mlp_

#include "../matrix/matrix.h"

typedef struct{
    unsigned int inputNum;
    unsigned int outputNum;
    int ActiveType;
}NetLyrAllocator;

class mlpCalcLyr {
private:
    int ActiveType;
public:
    matrix* weights,*bias;
	matrix* OriSum;

    mlpCalcLyr(const NetLyrAllocator thisLyrConf);
    void updateConf(const NetLyrAllocator thisLyrConf);

    ~mlpCalcLyr() = default;

    matrix calcsum(const matrix& inputdata);
};  

class mlpNet{
private:
    matrix *fullConnData;
    mlpCalcLyr *SigleLyrNutronSets;    
	
public:
	mlpNet(unsigned int arrLen,const NetLyrAllocator *netStruct);
    ~mlpNet();

    void setInput(const matrix &inputdata);

    void forward();
    #ifdef __ON_TRAINING
    void backward(const matrix &excepedResult);
    #endif

    matrix getOutput();
};

#endif