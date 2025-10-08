#ifndef _softmalloc_dynpool_
#define _softmalloc_dynpool_

#include "softpool.h"
#include <vector>

//小对象池,2页&8B padding
class NormPool{
private:
    char *bitmap;   //位图索引 
    void *poolmem;  //8kb总pool内存
    size_t memLength;   // 总可用字节数

    std::mutex AllocationLock;  //分配锁
    std::mutex GCLock;          //GC锁,处理时池全局锁定

    std::vector<SfMetaData*> sfptr_lst;
    std::vector<short> lstSpreadPos;

    inline size_t matchBitmapFreePart(size_t objsize);
    inline void changeBitmapMark(size_t start, size_t end, char bitmark);
public:
    NormPool();
    ~NormPool();

    char poolAllocate(SfMetaData *ret, size_t size);
    void poolfree(SfMetaData *metadata);
};

//中对象池,4页&32B padding
class MediumPool{
    private:
    char *bitmap;   //位图索引 
    void *poolmem;  //16kb总pool内存
    size_t memLength;   // 总可用字节数

    inline size_t matchBitmapFreePart(size_t objsize);
    inline void changeBitmapMark(size_t start, size_t end, char bitmark);
public:
    MediumPool();
    ~MediumPool();

};

#endif
