#include "mlp.h"


mlpNet::mlpNet(unsigned int arrLen,const NetLyrAllocator *netStruct){
    fullConnData = (matrix*)malloc(sizeof(matrix)*(arrLen+1));
    SigleLyrNutronSets = (mlpCalcLyr*)malloc(sizeof(mlpCalcLyr)*arrLen);

    for(unsigned int i = 0;i < arrLen;i++){
        fullConnData[i] = matrix(netStruct[i].inputNum,1);
        SigleLyrNutronSets[i].updateConf(netStruct[i]);
    }

    fullConnData[arrLen] = matrix(netStruct[arrLen-1].outputNum,1);
}

mlpCalcLyr::mlpCalcLyr(const NetLyrAllocator thisLyrConf){
    updateConf(thisLyrConf);
}

void mlpCalcLyr::updateConf(const NetLyrAllocator thisLyrConf){
    weights = new matrix(thisLyrConf.inputNum,thisLyrConf.outputNum);
    bias = new matrix(1,thisLyrConf.outputNum);
    OriSum = new matrix(1,thisLyrConf.outputNum);
}

matrix mlpCalcLyr::calcsum(const matrix &inputdata){
    (*OriSum) = (*weights) * inputdata + (*bias);
}
