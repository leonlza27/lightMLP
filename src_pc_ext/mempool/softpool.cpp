#include "softpool.h"

inline size_t NormPool::matchBitmapFreePart(size_t objsize){
    size_t loopnum = memLength - objsize;
    for(size_t i = 0; i<loopnum; i++){
        if(bitmap[i / 8]&(1 <<(i%8)) == 0){
            size_t matchend = i + objsize;
            for(size_t j = i + 1; j < matchend; j++){
                if(bitmap[j / 8]&(1 <<(j%8)) == 1){
                    i = j;
                    continue;
                }
            }
            return i;
        }
    }

    return -1;
}

inline void NormPool::changeBitmapMark(size_t start, size_t end, char bitmark){
    if(bitmark){
        for(size_t i = start; i<end; i++){
            bitmap[i / 8]|=bitmark<<(i%8);
        }
    }else{
        for(size_t i = start; i<end; i++){
            bitmap[i / 8]&= ~(bitmark<<(i%8));
        }
    }
    
}

NormPool::NormPool(int units){
    poolmem = alloc_8kbpages(units);
    memLength = units * _8KB_PAGE_UNIT;
    bitmap = (char*)malloc(memLength / 8);
    memset(bitmap, 0, memLength / 8);
}

NormPool::~NormPool(){
    freepage(poolmem, memLength);
    free(bitmap);
    poolmem = 0;
    bitmap = 0;
    memLength= 0;
}

void *NormPool::poolmalloc(size_t size){
    size_t inpoolOffset = matchBitmapFreePart(size);

    if(inpoolOffset == -1) return 0;

    changeBitmapMark(inpoolOffset, inpoolOffset+size, 1);

    return poolmem + inpoolOffset;
}

void NormPool::poolfree(void *ptr, size_t free_size){
    size_t inpoolOffset = 0;
    char *calcLoopStart = (char*)poolmem;
    while(calcLoopStart<ptr){
        calcLoopStart++;
        inpoolOffset++;
    }
    changeBitmapMark(inpoolOffset, inpoolOffset + free_size, 0);
}
