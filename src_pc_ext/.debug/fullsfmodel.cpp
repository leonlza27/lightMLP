#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <mutex>
#include <thread>
#include <stdlib.h>
#include <time.h>

struct PoolHead{
    uint16_t fullsize;
    uint16_t padding;
};

//元数据头节点 空闲[end, startNext)
struct DataNode{
    int16_t end;
    int16_t startNext;
    int16_t prev;
};

class mempool{
private:
    char *mem;
    char *dataStart;

    std::mutex mallocLock;
public:
    mempool(){
        mem = (char*)mmap(0, 8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        memset(mem, 0, 8192);
        PoolHead *head = (PoolHead*)mem;
        head->fullsize = 8192;
        head->padding = 4;
        
        dataStart = mem + 4;
        DataNode *lkhead = (DataNode*)dataStart;
        DataNode *lktail = (DataNode*)(dataStart + 8180);
        
        lkhead->end = 8;
        lkhead->startNext = 8180;
        lkhead->prev = -1;

        lktail->startNext = -1;
        lktail->end = 8188;
        lktail->prev = 0;

        //total 8172B
    }
    
    ~mempool(){
        dataStart = 0;
        munmap(mem, 8192);
    }

    void *pAlloc(uint16_t size){
        std::lock_guard<std::mutex> allocG(mallocLock);
        size_t total_size = size + 4;
        size_t _needed = (total_size + 3) & (~3); 

        DataNode *curnode;
        int16_t _idxnxt = 0;
        int16_t _idxcur = 0;
        int16_t ava0 = 0;        //可用起始偏移
        while (_idxcur != -1){
            curnode = (DataNode*)(dataStart + _idxcur);
            _idxnxt = curnode->startNext;
            ava0 = curnode->end;
            if(_idxnxt - ava0 >= _needed) break;
            _idxcur = _idxnxt;
        }
        if(_idxnxt == -1) return 0;
        
        DataNode *nodeNew = (DataNode*)(dataStart + ava0);
        nodeNew->end = ava0 + _needed;
        nodeNew->prev = _idxcur;
        nodeNew->startNext = _idxnxt;

        DataNode *nnewNext = (DataNode*)(dataStart + _idxnxt);

        curnode->startNext = ava0;
        nnewNext->prev = ava0;
        
        return dataStart + ava0 + 6;
    }

    void pFree(void* ptr){
        std::lock_guard<std::mutex> freeG(mallocLock);
        if(ptr == 0) return;
        size_t delidx = ((size_t)ptr) - ((size_t)dataStart) - 6;
        if(delidx >= 8172 || delidx < 8) return;

        DataNode *delNode = (DataNode*)(dataStart + delidx);
        if(delNode->end == 0) return;
        DataNode *prev_of_del = (DataNode*)(dataStart + delNode->prev);
        DataNode *next_of_del = (DataNode*)(dataStart + delNode->startNext);
        
        next_of_del->prev = delNode->prev;
        prev_of_del->startNext = delNode->startNext;
        delNode->end = 0;
    }
};

mempool _mempool;

void testHandler(int _no){
    size_t _size = rand()%225 + 1;
    char *testmem = (char*)_mempool.pAlloc(_size);
    printf("thread %d: try get %lu B...\n", _no, _size);
    if(testmem == 0) {
        printf("thread %d: fail\n", _no);
        return;
    }
    int __handleTime = rand()%999 + 1;
    printf("thread %d: succeed, hold for %d ms\n", _no, __handleTime);
    std::this_thread::sleep_for(std::chrono::milliseconds(__handleTime));
    _mempool.pFree(testmem);
    printf("thread %d: mem returned\n", _no);
}

int main(){
    srand(time(0));

    std::thread testThreads[256];
    int nu = 0;
    for(auto &t : testThreads){
        t = std::thread(testHandler, nu);
        nu++;
    }

    for(auto &t : testThreads){
        t.join();
    }
    
    return 0;
}
