#ifndef __softmalloc__
#define __softmalloc__

#include "SoftPool/softpool.h"

enum SubPooltype{
    Norm,
    Medium,
    LSO
};

//全局默认池
static SoftPool _GlobSoftAllocator;

struct SfMetaData{
    void *rawptr;                   //得到的原始地址
    void *poolPending;              //依赖池地址
    SubPooltype actualPoolType;
    std::mutex &PoolGCLockRef;
    short MetaIdx;
};
    
//sfptr<T> 实现

template<typename T, SoftPool &SfAllocator = _GlobSoftAllocator>
class sfptr{
private:
    T *addr;
public:
    sfptr(size_t elemlength);
    ~sfptr();
};

#include "SoftPool/sfptr.inl"

#endif
