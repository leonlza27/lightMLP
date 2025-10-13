#ifndef _softmalloc_dynpool_
#define _softmalloc_dynpool_

#include "softpool.h"
#include <set>

//小对象池,2页&8b padding
class NormPool{
private:
    char *bitmap;   //位图索引 
    void *poolmem;  //8kb总pool内存
    size_t memLength;   // 总可用字节数

    std::mutex MetaWriteLock;   //元数据写入锁
    std::mutex GCLock;          //GC锁,处理时池全局锁定

    std::set<SfMetaData*, SfMetaPtrIdxCmp> objDatas;

public:
    NormPool();
    ~NormPool();

    char poolAllocate(SfMetaData *ret, size_t size);
    void poolfree(SfMetaData *metadata);

    inline void* inferRawAddress(short block0_offset) const;
};

//中对象池,4页&32b padding
class MediumPool{
    private:
    char *bitmap;   //位图索引 
    void *poolmem;  //16kb总pool内存
    size_t memLength;   // 总可用字节数

public:
    MediumPool();
    ~MediumPool();

};

#endif
