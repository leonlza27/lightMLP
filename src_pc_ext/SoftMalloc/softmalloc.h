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
    void *poolPending;              //依赖池地址
    std::mutex *PoolGCLockRef;
    SubPooltype actualPoolType;
    short block0_Offset;            //位图地址0标记池偏移
    short blockUsed;                //位图已用块数
    short MetaIdx;
};
    
//sfptr<T> 实现

template<typename T, SoftPool &SfAllocator = _GlobSoftAllocator>
class sfptr{
private:
    SfMetaData metathis;
    T *addr;
public:
    sfptr(size_t elemlength);
    ~sfptr();
};

#include "SoftPool/sfptr.inl"

#endif
