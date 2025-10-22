#include <stdio.h>
#include <set>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <time.h>
#include <stack>

//模拟元数据分配

struct TestData{
    short ind0;
    short length;
};

//区间[start,end)
struct DataStorage{
    short start;
    short end;
};

class DataStorageCmp{
public:
    bool operator()(const DataStorage& _cmpl, const DataStorage& _cmpr)const{
        return _cmpl.start < _cmpr.start;
    }
};

//模拟单例对象池
template<typename T>
class SimSigInfMemPool{
private:
    T *mem;
    std::stack<T*> ava_space_start;
public:
    SimSigInfMemPool(size_t capacity){
        init(capacity);
    }

    void init(size_t capacity){
        mem = (T*)malloc(capacity * sizeof(T));
        ava_space_start.push(mem);
    }

    T* palloc(){
        T* ret = ava_space_start.top();
        ava_space_start.pop();
        if(ava_space_start.empty()){
            ava_space_start.push(ret + 1);
        }
        return ret;
    }

    void pfree(T* ptr){
        ava_space_start.push(ptr);
    }

};

template<typename T>
class PooledSigleAllocator{
private:
    SimSigInfMemPool<T> _pool;
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<class U>
    struct rebind{ 
        using other = PooledSigleAllocator<U>; 
    }; 

    PooledSigleAllocator()noexcept : _pool(4097){}
    ~PooledSigleAllocator()noexcept{}

    template<typename U>
    PooledSigleAllocator(const PooledSigleAllocator<U>&)noexcept{}
    
    T *allocate(size_t size){
        return static_cast<T*>(_pool.palloc());
    }

    void deallocate(T *ptr, size_t size)noexcept{
        _pool.pfree(ptr);
    }    

};

std::mutex SituateMallocLock;
std::set<DataStorage,DataStorageCmp,PooledSigleAllocator<DataStorage>> plset;

int main(){
    srand((unsigned int)time(0));
    std::thread testthreads[100];
    

    plset.insert({0,0});

    return 0;
}
