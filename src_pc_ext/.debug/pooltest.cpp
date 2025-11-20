#include <stdio.h>
#include <sys/mman.h>
#include <vector>
#include <map>

#include "pooleg.h"

class cntHolder{
private:
    std::atomic<int8_t> &_counter;
public:
    cntHolder(std::atomic<int8_t> &counter):_counter(counter){
        _counter.fetch_add(1,std::memory_order_acquire);
    }

    ~cntHolder(){
        _counter.fetch_add(-1,std::memory_order::memory_order_acquire);
    }

};

class plswitcher{
private:
    std::atomic<int8_t> tnum;
    uint8_t UsableIdx;
    spin_lock SetAttrLock;
    std::atomic<int8_t> FullQueueFlag;
    mempool *pools[15];
public:
    plswitcher():UsableIdx(1){ 
        for (auto &p : pools) {
            p = 0;
        }
        pools[0] = CreateMempool();
        tnum = 0;
        FullQueueFlag = 0;
    }

    ~plswitcher(){
        for(auto &p : pools){
            munmap(p, 8192);
        }
    }

    void *alllocate(size_t size){
        cntHolder hld(tnum);

        //FullQueueFlag.store((tnum > 15)||(tnum > 5 && FullQueueFlag.load(std::memory_order_acquire)),std::memory_order_acquire);
        
        bool curFlag = FullQueueFlag.load(std::memory_order_relaxed);
        int curT = tnum.load(std::memory_order_relaxed);

        // 锁死条件
        bool lockCond = (curT > 15);

        // 解锁条件
        bool unlockCond = (curT <= 5);

        // 状态机
        bool nextFlag = (curFlag & !unlockCond) | (!curFlag & lockCond);

        FullQueueFlag.store(nextFlag, std::memory_order_release);
        
        while(SetAttrLock.is_locked()){
            std::this_thread::yield();
        }
        
        void* retAddr;
        for(auto &p : pools){
            if(p == 0) break;
            if(!p->isOperating()){
                retAddr = p->pAlloc(size);
                if(retAddr) return retAddr;
            }
        }

        if(UsableIdx < 15){
            std::lock_guard<spin_lock> attrLock(SetAttrLock);
            mempool *pNew = CreateMempool();
            pools[UsableIdx] = pNew;
            UsableIdx++;
            return pNew->pAlloc(size);
        }

        printf("pool: full\n");

        return 0;
    }

    void free(void* ptr){
        char* freePtr = (char*)ptr - 8;
        DataNode *freeNode = (DataNode*)(freePtr);
        mempool *parentPool = (mempool*)(freePtr - freeNode->thisStart - 3);
        parentPool->pFree(ptr);
    }

    bool isQueueAvaliable() {return FullQueueFlag.load(std::memory_order_relaxed);}    

};

plswitcher *switcher;
spin_lock releaseThread;
spin_lock reportLock;
std::map<size_t,int> addrReport;

void testHandler(){
     while(releaseThread.is_locked()){
            std::this_thread::yield();
        }
    size_t _size = rand()%225 + 1;
    char *testmem = (char*)switcher->alllocate(_size);
    if(testmem == 0) {
        printf("thread: got a null addr\n");
        return;
    }
    int __handleTime = rand()%9999 + 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(__handleTime));
    {
        std::lock_guard<spin_lock> repo(reportLock);
        if(!addrReport[(size_t)testmem]){
            addrReport[(size_t)testmem] = 0;           
        }
        addrReport[(size_t)testmem]++;
    }
    switcher->free(testmem);
}

int main(){ 
    switcher = (plswitcher*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    switcher = new(switcher) plswitcher;

    std::vector<std::thread> workers;
    workers.reserve(1024);
    releaseThread.lock();
    for(int i = 0; i< 64; i++)
        workers.emplace_back(testHandler);

    releaseThread.unlock();
    for(auto &t : workers) t.join();   

    size_t TotalCalls = 0;
    for(auto &i : addrReport){
        TotalCalls+=i.second;
    }

    printf("Total allocatins :%lu\n",TotalCalls);

    return 0;
}
