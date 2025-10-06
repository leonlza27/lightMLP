#ifndef _softmalloc_dynpool_
#define _softmalloc_dynpool_

#include "softpool.h"

//小对象池,2页&8B padding
class NormPool{
private:
    char *bitmap;   //位图索引 
    void *poolmem;  //8kb总pool内存
    size_t memLength;   // 总可用字节数

    inline size_t matchBitmapFreePart(size_t objsize);
    inline void changeBitmapMark(size_t start, size_t end, char bitmark);
public:
    NormPool();
    ~NormPool();

    void *poolmalloc(size_t size);
    void poolfree(void *ptr, size_t free_size);
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