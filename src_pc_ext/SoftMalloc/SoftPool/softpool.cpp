#include "softpool.h"

SoftPool::SoftPool(){
    NormPool *poolInit = new NormPool();
    NormPools.push_front(poolInit);
}
