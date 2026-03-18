#include <stdio.h>
#include <pthread.h>
#include <stdint.h>
#include <malloc.h>
#include <unistd.h>

// an idea to build a simple gc
// for jni mlpnet space reuse
// because java vm would make dataload slower and may cause memory manangement problems caused by detaching with c core

typedef struct obcap_int{
    uint16_t rcnt;
    int value;
}obcap_int;

pthread_mutex_t lstop_lk;
int16_t recentusable = 0;
uint16_t keeploop = 1;

#define maxnum 32
obcap_int *globlst;

void checkarrive(){
    pthread_mutex_lock(&lstop_lk);

    for(int i = 0; i < maxnum; i++){
        obcap_int *cur = &globlst[i];
        if(!cur) continue;
        if(cur->rcnt){
            cur->rcnt--;
            continue;
        }
    }
    pthread_mutex_unlock(&lstop_lk);
}

void *bkcheck_caller(void *param){
    while(keeploop){
        printf("gc call\n");
        checkarrive();
        printf("gc end\n");
        sleep(10);
    }
}

int16_t pushvalue(int v){
    pthread_mutex_lock(&lstop_lk);
    uint16_t ret = recentusable;
    globlst[ret].value = v;
    globlst[ret].rcnt = 1;
    uint16_t avaloops = 2;
    while(avaloops){
        if(!globlst[recentusable].rcnt) break;
        recentusable++;
        if(recentusable >=maxnum){
            avaloops--;
            recentusable = 0;
        }
    }
    pthread_mutex_unlock(&lstop_lk);
    return avaloops? ret : -1;
}

int16_t checkidx(int idx){
    obcap_int *watch = &globlst[idx];
    if(!watch->rcnt) return -1;
    watch->rcnt++;
    return watch->value;
}

int main(){
    globlst = (obcap_int*)malloc(sizeof(obcap_int) * maxnum);
    for(int i = 0; i < maxnum; i++) globlst[i].rcnt = 0;
    pthread_t bkcall;
    pthread_create(&bkcall, 0, bkcheck_caller, 0);
    char c, buf[8];
    int16_t vop;
    while(1){
        fgets(buf, 8, stdin); 
        sscanf(buf ,"%c %d",&c, &vop);
        switch(c){
            case 'i':
                printf("push value %d, at idx %d\n", vop, pushvalue(vop));
                break;
            case 'w':
                printf("watch idx %d get value %d\n", vop, checkidx(vop));
                break;
            case 'q':
                goto _out;
        }
    }
_out:
    keeploop = 0;
    pthread_join(bkcall, 0);
    free(globlst);
}


