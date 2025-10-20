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

class TestDataCmp{
public:
    bool operator()(const TestData* _cmpl, const TestData* _cmpr)const{
        return _cmpl->ind0 < _cmpr->ind0;
    }
};

std::mutex SituateMallocLock;
std::set<TestData*, TestDataCmp> plset;

FILE *_log;

int cycles = 0;

inline void ReportSpace(){
    fprintf(_log, "%4d: |", ++cycles);
    for(auto p : plset){
        fprintf(_log, " %4d~%4d |", p->ind0, p->length + p->ind0 -1);
    }
    fprintf(_log, "\n");
}

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
        ReportSpace();
        return;
    }

    ret->ind0 = lastEnd + 1;
    ret->length = size;

    plset.insert(ret);
    ReportSpace();
}

void SimFree(TestData *delObj){
    std::lock_guard<std::mutex> FreeLock(SituateMallocLock);
    plset.erase(delObj);
    delObj->length = 0;
    delObj->ind0 = -1;
    ReportSpace();
}

void TestHandler(){
    std::thread::id _tid = std::this_thread::get_id();
    TestData privHandle;
    short _size = rand()%255 + 1;
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
    _log = fopen("mallocrep.txt", "w");
    std::thread testthreads[100];
    for(auto &i : testthreads){
        i = std::thread(TestHandler);
    }
    for(auto &i : testthreads){
        i.join();
    }
    TestHandler();

    return 0;
}
