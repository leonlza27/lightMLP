#include <string.h>
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <winerror.h>

//共享资源模型原型
//用于处理在c无threadlocal特性支持下的临时分配/算力资源统一管理


//实现原型

typedef struct resource{
    int* info;
    uint8_t ishold : 1;
    uint8_t free_cntdwn : 4;
}resource;

resource sim_resource[32];
uint16_t curidx;

uint8_t onrun;

HANDLE cleanprocess;

int32_t findspare(){
    uint8_t cycles = 2;
    while(cycles){
        if(!sim_resource[curidx].ishold){
            sim_resource[curidx].ishold = 1;
            return curidx;
        }
        curidx++;
        if(curidx >= 32){
            curidx = 0;
            cycles--;
        }
    }
    return -1;
}

void release(uint16_t spaceidx){
    sim_resource[spaceidx].ishold = 0;
    sim_resource[spaceidx].free_cntdwn = 8;
}

DWORD WINAPI backend_check(void *param){
    while(onrun){
        for(uint16_t i = 0; i < 32; i++){
            if(!sim_resource[i].ishold &&sim_resource[i].free_cntdwn) sim_resource[i].free_cntdwn--;
            if(!sim_resource[i].free_cntdwn){
                free(sim_resource[i].info);
                sim_resource[i].info = 0;
            }
        }
        Sleep(5);
    }
    return 0;
}

//测试用例

DWORD WINAPI subprocess(void *param){
    uint16_t thread_no = (uint16_t)param;
    for(int i = 0; i < 10; i++){
        int32_t idxget = findspare();
        if( idxget == -1){
            printf("thread %d<attmpt %d/10>: not get resource\n", thread_no, i + 1);
            continue;
        }
        size_t hold_ms = rand()%2000;
        printf("thread %d<attmpt %d/10>: hold for %llu ms\n", thread_no, i + 1, hold_ms);
        Sleep(hold_ms);
        release(idxget);
    }
    return 0;
}

int main(){
    srand(time(0));
    curidx = 0;
    for(uint8_t i = 0; i < 32; i++){
        sim_resource[i].free_cntdwn = 0;
        sim_resource[i].ishold = 0;
        sim_resource[i].info = 0;
    }
    cleanprocess = CreateThread(0, 0, backend_check, 0, 0, 0);
    HANDLE usagethreads[128];
    for(uint16_t i = 0; i < 128; i++){
        usagethreads[i] = CreateThread(0, 0, subprocess, (void*)i, 0, 0);
    }
    DWORD waitstat = WAIT_TIMEOUT;
    while(WAIT_TIMEOUT == waitstat){
        waitstat = WaitForMultipleObjects(128, usagethreads, 1, INFINITE);
    }
    onrun = 0;
    WaitForSingleObject(cleanprocess, INFINITE);
    return 0;
}