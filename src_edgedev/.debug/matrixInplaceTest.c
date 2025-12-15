#include <stdio.h>
#include "../matrix/matrix_static.h"
#include "../qfix_ops/qfix_op.h"
#include <stdlib.h>
#include <time.h>
#include <linux/time.h>

// Linux/macOS: 纳秒级精度
uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// 或者使用CLOCK_MONOTONIC_RAW（不受NTP调整影响）
uint64_t get_time_ns_raw() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(){
    matrix_bp m1;
    matrix_bp m2;
    matrix_bp m3;
    matrix_bp mr;

    int m = 2, k = 28, n = 4;

    srand(time(0));

    qfix cap1[64] = {};
    qfix cap2[64] = {};
    qfix cap3[64] = {};
    qfix capr[64] = {};

    for(int i = 0; i < 64; i++) cap1[i] = rand() % 2 >> 17;
    for(int i = 0; i < 64; i++) cap2[i] = rand() % 2 >> 17;

    m1 = (matrix_bp)cap1;
    m1->cols = k;
    m1->rows = m;

    m2 = (matrix_bp)cap2;
    m2->cols = n;
    m2->rows = k;

    m3 = (matrix_bp)cap3;
    m3->cols = n;
    m3->rows = m;
    
    mr = (matrix_bp)capr;
    mr->cols = n;
    mr->rows = m;

    uint64_t start = get_time_ns();
    matrix_bp_mulpty_optimized(m1, m2, m3);
    uint64_t end = get_time_ns();

    printf("opt %d\n", end - start);

    start = get_time_ns();
    matrix_bp_mulpty_raw(m1, m2, mr);
    end = get_time_ns();

    printf("raw %d\n", end - start);
    
    for(uint16_t i = 0; i < m * n; i++){
        if(m3->data[i] != mr->data[i]) return 1;
    }
}
