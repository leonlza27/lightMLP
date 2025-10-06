#include "DynPool.h"

inline size_t NormPool::matchBitmapFreePart(size_t objsize){
   if (memLength < objsize) return -1;
    
    size_t left = 0;   // 窗口左边界
    size_t right = 0;  // 窗口右边界
    
    // 滑动窗口查找连续的0
    while (right < memLength) {
        // 如果right位置是1，则重置窗口
        size_t right_byte_index = right / 8;
        size_t right_bit_index = right % 8;
        if ((bitmap[right_byte_index] & (1 << right_bit_index)) != 0) {
            // 遇到1，重置窗口
            left = right + 1;
        } else {
            // right位置是0
            // 检查窗口大小是否达到要求
            if (right - left + 1 == objsize) {
                return left;  // 返回起始位置
            }
        }
        right++;  // 扩展右边界
    }
    
    return -1;  // 未找到
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

NormPool::NormPool(){
    poolmem = alloc_8kbpages(1);
    memLength = _8KB_PAGE_UNIT;
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
