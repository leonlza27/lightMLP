#include <sys/mman.h>
#include <stdint.h>
#include <stdlib.h>
#include <mutex>

#include "spin_lock.h"

//元数据头节点 空闲[end, startNext)
struct DataNode{
    int16_t end;
    int16_t startNext;
    int16_t prev;
    int16_t fullsize;
};

class mempool{
private:
    uint16_t objnum;
    alignas(2) spin_lock mallocLock;
    char dataStart[8188];
public:
    mempool(){
        DataNode *lkhead = (DataNode*)dataStart;
        DataNode *lktail = (DataNode*)(dataStart + 8180);
        
        lkhead->end = 8;
        lkhead->startNext = 8180;
        lkhead->prev = -1;
        lkhead->fullsize = 8;

        lktail->startNext = -1;
        lktail->end = 8188;
        lktail->prev = 0;
        lktail->fullsize = 8;

        //total 8172B
    }
    
    ~mempool(){
        
    }

    void *pAlloc(uint16_t size){
        std::lock_guard<spin_lock> alloclk(mallocLock);
        size_t total_size = size + 8;
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
        nodeNew->fullsize = _needed;

        DataNode *nnewNext = (DataNode*)(dataStart + _idxnxt);

        curnode->startNext = ava0;
        nnewNext->prev = ava0;

        objnum++;
        return dataStart + ava0 + 8;
    }

    void pFree(void* ptr){
        std::lock_guard<spin_lock> freelk(mallocLock);
        if(ptr == 0) return;
        size_t delidx = ((size_t)ptr) - ((size_t)dataStart) - 8;
        if(delidx >= 8172 || delidx < 8) return;

        DataNode *delNode = (DataNode*)(dataStart + delidx);
        if(delNode->fullsize == 0) return;
        DataNode *prev_of_del = (DataNode*)(dataStart + delNode->prev);
        DataNode *next_of_del = (DataNode*)(dataStart + delNode->startNext);
        
        next_of_del->prev = delNode->prev;
        prev_of_del->startNext = delNode->startNext;
        delNode->fullsize = 0;

        objnum -= 1 * !(objnum < 0);
    }

    int16_t GetObjNum() const { return objnum;}
    bool isOperating() { return mallocLock.is_locked();}
};

mempool *CreateMempool(){
    void *mem = mmap(0,8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    return new(mem) mempool;
}