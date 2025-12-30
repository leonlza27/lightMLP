#include <stdio.h>
#include <set>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <time.h>
#include <stack>

//模拟元数据分配

//元数据结构体,区间[start,end)
struct DataStorage{
    short start;
    short end;

    DataStorage() : start(-1), end(-1){}
};

class DataStorageCmp{
public:
    bool operator()(const DataStorage& _cmpl, const DataStorage& _cmpr)const{
        return _cmpl.start < _cmpr.start;
    }
};

class DataStoragePtrCmp{
public:     
    bool operator()(const DataStorage* _cmpl, const DataStorage* _cmpr)const{
        return _cmpl->start < _cmpr->start;   
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
        for(size_t i = 0; i < capacity; i++)
            ava_space_start.push(mem + i);
    }

    T* palloc(){
        if(ava_space_start.empty()){
            return 0;
        }
        T* ret = ava_space_start.top();
        ava_space_start.pop();
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

    PooledSigleAllocator()noexcept : _pool(8192){}
    ~PooledSigleAllocator()noexcept{}

    template<typename U>
    PooledSigleAllocator(const PooledSigleAllocator<U>&)noexcept{}
    
    T *allocate(size_t size){
        if(size > 1) return 0;
        return static_cast<T*>(_pool.palloc());
    }

    void deallocate(T *ptr, size_t size)noexcept{
        _pool.pfree(ptr);
    }    

};
std::mutex SituateMallocLock;
std::set<DataStorage*,DataStoragePtrCmp,PooledSigleAllocator<DataStorage*>> plset;

FILE *memLayout;

void printMemLayout(){
    fprintf(memLayout, ">");
    for(auto i : plset){
        fprintf(memLayout, "|[%4d~%4d)\t",i->start,i->end);
    }
    fprintf(memLayout, "|\n");
}

//main test fns

bool SimAlloc(size_t size, DataStorage *dsret){
    std::lock_guard<std::mutex> malloclock(SituateMallocLock);
    short LastEnd = 0;
    for(auto i : plset){
        if(i->start - LastEnd >= size) break;
        LastEnd = i->end;
    }
    if(LastEnd + size > 8192) return 0;

    dsret->start = LastEnd;
    dsret->end = LastEnd + size;
    plset.insert(dsret);
    printMemLayout();
    return 1;
}

void SimFree(DataStorage *dptr){
    std::lock_guard<std::mutex> freelock(SituateMallocLock);
    plset.erase(dptr);
    dptr->start = -1;
    dptr->end = -1;
    printMemLayout();
}

//test thread fn
void TestHandler(){
    DataStorage _dataThis;
    size_t size = rand() % 255 + 1;
    if(!SimAlloc(size, &_dataThis)){
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 999 + 1));
    SimFree(&_dataThis);
}

int main(){
    srand((unsigned int)time(0));
    std::thread testthreads[100];
    memLayout = fopen("layout.txt", "w");

    for(auto &i : testthreads){
        i = std::thread(TestHandler);
    }
    for(auto &i : testthreads){
        i.join();
    }

    fclose(memLayout);

    return 0;
}
