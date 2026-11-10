#include <stddef.h>   /* size_t(Linux/glibc 的 stdint.h 不提供 size_t) */
#include <stdint.h>

struct threadpool;
typedef struct threadpool threadpool;

uint8_t public_pool_reconfig(uint16_t mananged_count, uint16_t delay_per_flush, uint8_t flush_cycles_to_free);

threadpool *pubpools_getpool(uint16_t idx);
int32_t pubpools_append_task(void (*callback)(void*), void *params, size_t count, size_t param_size);
uint8_t pubpools_checkprocessing(uint16_t idx);
void pubpools_release(uint16_t idx);
