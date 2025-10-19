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

std::mutex SituateMallocLock;
std::set<TestData*> plset;

void SimAllocate(short size, TestData *ret){
    std::lock_guard<std::mutex> AllocateLock(SituateMallocLock);
    short lastEnd = -1;
    for(auto part : plset){
         if((part->ind0 - lastEnd) >= size){
             break;
         }
         lastEnd = part->ind0 + part->length - 1;
    }
    if(lastEnd + size > 8192){
        ret->ind0 = -1;
        ret->length = 0;
        return;
    }

    ret->ind0 = lastEnd + 1;
    ret->length = size;

    plset.insert(ret);
}

void SimFree(TestData *delObj){
    std::lock_guard<std::mutex> FreeLock(SituateMallocLock);
    plset.erase(delObj);
    delObj->length = 0;
    delObj->ind0 = -1;
}

void TestHandler(){
    std::thread::id _tid = std::this_thread::get_id();
    TestData privHandle;
    short _size = rand()%9000 + 1;
    printf("%ld: try allocate %d B\n",_tid,_size);
    SimAllocate(_size,&privHandle);
    if(privHandle.length == 0){
        printf("%ld: malloc failed\n",_tid);
        return;
    }
    short _holdtime = rand()%490 + 10;
    printf("%ld: handle succeed(for about %d ms)\n",_tid,_holdtime);

    std::this_thread::sleep_for(std::chrono::milliseconds(_holdtime));
    SimFree(&privHandle);
    printf("%ld: handle free\n",_tid);
}

    

int main(){
    srand((unsigned int)time(0));
    TestHandler();

    return 0;
}
