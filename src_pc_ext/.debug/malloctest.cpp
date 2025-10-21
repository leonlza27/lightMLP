#include <stdio.h>
#include <set>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <time.h>

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

template<typename T>
class PooledSigleAllocator{
private:
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


    PooledSigleAllocator()noexcept{}
    ~PooledSigleAllocator()noexcept{}

    template<typename U>
    PooledSigleAllocator(const PooledSigleAllocator<U>&)noexcept{}
    
    T *allocate(size_t size){
        printf("Allocate num:%lld\n",size);
        return static_cast<T*>(::operator new(sizeof(T) * size));
    }

    void deallocate(T *ptr, size_t size)noexcept{
        ::operator delete(ptr);

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
