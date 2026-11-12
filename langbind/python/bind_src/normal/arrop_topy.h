#ifndef arrop_topy_norm
#define arrop_topy_norm

#include "../PythonBind_CustomConf.h"
#include "moduleobject.h"

#include <stddef.h>

typedef struct arr_pyhandle{
    PyObject_HEAD;
    uint32_t size;          //总容量
    qfix *data;             //数据
}arr_pyhandle;

// 序列/映射协议: 支持 len() 和 [] 操作
Py_ssize_t arr_handle_len(PyObject *self);
PyObject *arr_handle_getitem(PyObject *self, PyObject *key);
int arr_handle_setitem(PyObject *self, PyObject *key, PyObject *val);

//在pyinclude_entry里拉起库时使用
extern PyModuleDef arr_norm_topy;
extern PyModuleDef arr_mt_topy;
extern PyTypeObject arr_handle_def;

#endif