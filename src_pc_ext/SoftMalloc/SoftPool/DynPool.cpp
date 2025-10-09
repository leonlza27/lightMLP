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
            bitmap[i / 8]|=1<<(i%8);
        }
    }else{
        for(size_t i = start; i<end; i++){
            bitmap[i / 8]&= ~(1<<(i%8));
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

char NormPool::poolAllocate(SfMetaData *ret, size_t size){
    std::lock_guard<std::mutex> allocGuard(MetaWriteLock);

    //查找可用
    size_t offset = matchBitmapFreePart(size);
    if(offset = -1){
        ret->blockUsed = 0;
        return -1;
    }

    //添加元数据
    size_t LenSpread = lstSpreadPos.size();

    if(LenSpread){
        ret->MetaIdx = LenSpread - 1;
        sfptr_lst[LenSpread - 1] = ret;
        lstSpreadPos.pop_back();
        
    }else{
        ret->MetaIdx = sfptr_lst.size();
        sfptr_lst.push_back(ret);
    }
    changeBitmapMark(offset, offset+size, 1);

    //元数据绑定
    ret->block0_Offset = offset;
    ret->blockUsed = size;
    ret->poolPending = this;
    ret->PoolGCLockRef = &GCLock;
    ret->actualPoolType = Norm;

    return 0;
}

void NormPool::poolfree(SfMetaData *metadata){
    std::lock_guard<std::mutex> freeGuard(MetaWriteLock);

    //池标记清除
    short DelTg = metadata->MetaIdx;
    sfptr_lst[DelTg] = 0;
    lstSpreadPos.push_back(DelTg);
    short offsetStart = metadata->block0_Offset;
    changeBitmapMark(offsetStart, offsetStart + metadata->blockUsed, 0);

    //元数据解绑
    metadata->block0_Offset = -1;
    metadata->blockUsed = 0;
    metadata->poolPending = 0;
    metadata->PoolGCLockRef = 0;
}

inline void *NormPool::inferRawAddress(short block0_offset) const{
    return poolmem + block0_offset;
}
