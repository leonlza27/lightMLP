#include "mt_tpool_recofig.h"
#include "mlpCCore/threadpool/sharedpool.h"

//threadpool_reconfig(new_tpools_count: int, new_flush_delay_ms: int, new_cooldown_cycles: int)
PyObject *mt_tpool_reconfig(PyObject *_rtime, PyObject *args){
    uint16_t new_tpools_count, new_flush_delay_ms;
    uint8_t new_cooldown_cycles;
    if(!PyArg_ParseTuple(args, "HHb", &new_tpools_count, &new_flush_delay_ms, &new_cooldown_cycles)) return 0;

    public_pool_reconfig(new_tpools_count, new_flush_delay_ms, new_cooldown_cycles);

    Py_RETURN_NONE;
}

PyMethodDef pool_reconfig[] = {
    {"threadpool_recofig", mt_tpool_reconfig, METH_VARARGS, 0},
    {0,0,0,0}
};