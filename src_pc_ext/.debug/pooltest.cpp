#include <stdio.h>
#include <sys/mman.h>

#include "pooleg.h"

//4096 total

class pllstNode{
private:
    alignas(8) uint16_t usableIdx;
    mempool *pools[510];
    pllstNode *next; 
public:
    pllstNode() : usableIdx(0), next(0){
        for(auto &i : pools) i = 0;
    }

    ~pllstNode(){
        for(auto &i : pools){
            if(i){
                i->~mempool();
                munmap(i, 8192);
            }
        } 
    }

    mempool* GetPool(uint16_t idx) const{
        return pools[idx];
    }

    pllstNode *Next() const{
        return next;
    }
    
    //添加池: 成功返回1, 失败则自动添加next&&返回0
    bool appendPool(){
        if(usableIdx < 510){            
            void *pnew = mmap(0, 8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
            pools[usableIdx] = new(pnew) mempool;
            usableIdx++;
            return 1;
        }
        next = (pllstNode*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);  
        next = new(next) pllstNode;
        return 0;
    }
    
};

class plswitcher{
private:
    pllstNode *begin;
    mempool *hotpools[10]; 
public:
    plswitcher(){
        begin = (pllstNode*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        begin = new(begin) pllstNode;
        begin->appendPool();
        for (auto &p : hotpools) {
            p = 0;
        }
        hotpools[0] = begin->GetPool(0);
    }

    ~plswitcher(){
        pllstNode *pndefree = begin;
        while(pndefree){
            pllstNode *curfree = pndefree;
            pndefree = curfree->Next();
            curfree->~pllstNode();
            munmap(curfree, 4096);
        }
    }
};

int main(){ 


}
