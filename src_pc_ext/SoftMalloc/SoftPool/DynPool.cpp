#include "DynPool.h"

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
    std::lock_guard<std::mutex> allocLock(MetaWriteLock);

    short lastEnd = 0;
    for(auto part : objDatas){
         if((part->block0_Offset - lastEnd + 1) <= size){
             ret->block0_Offset = lastEnd;
             break;
         }
         lastEnd = part->block0_Offset + part->blockUsed;
    }
    
    if((lastEnd + size) > 8193){
        //触发池压缩GC
    }else{
        ret->block0_Offset = lastEnd;
    }


    //元数据绑定
    ret->blockUsed = size;
    ret->poolPending = this;
    ret->PoolGCLockRef = &GCLock;
    ret->actualPoolType = Norm;

    objDatas.insert(ret);

    return 0;
}

void NormPool::poolfree(SfMetaData *metadata){
    std::lock_guard<std::mutex> freeGuard(MetaWriteLock);

    

    //元数据解绑
    metadata->block0_Offset = -1;
    metadata->blockUsed = 0;
    metadata->poolPending = 0;
    metadata->PoolGCLockRef = 0;
}

inline void *NormPool::inferRawAddress(short block0_offset) const{
    return poolmem + block0_offset;
}
