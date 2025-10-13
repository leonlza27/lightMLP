#ifndef __softmalloc__
#define __softmalloc__

#include "SoftPool/softpool.h"

enum SubPooltype{
    _none,
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
    short block0_Offset;            //起始相对池位移
    short blockUsed;                //已用大小(按padding记)
};

class SfMetaPtrIdxCmp{
    bool operator()(const SfMetaData *dat1, const SfMetaData *dat2){
        return dat1->block0_Offset < dat2->block0_Offset;
    }
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
