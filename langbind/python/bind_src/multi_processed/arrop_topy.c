#include "../PythonBind_CustomConf.h"
#include "../normal/arrop_topy.h"
#include "mlpCCore/arrop/arrop.h"


DLLEXPORT PyObject* arr_mt_py_add(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_mt_py_sub(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_mt_py_scale(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_mt_py_emul(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_mt_py_mmul(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_mt_py_mtpose(PyObject *_rtime, PyObject *args);


PyMethodDef arr_mt_funcs[] = {
    {"vadd", arr_mt_py_add, METH_VARARGS, 0},
    {"vsub", arr_mt_py_sub, METH_VARARGS, 0},
    {"vscale", arr_mt_py_scale, METH_VARARGS, 0},
    {"vemul", arr_mt_py_emul, METH_VARARGS, 0},
    {"mmul", arr_mt_py_mmul, METH_VARARGS, 0},
    {"mtpose", arr_mt_py_mtpose, METH_VARARGS, 0},
    {0, 0, 0, 0},
};

PyModuleDef arr_mt_topy = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_methods = arr_mt_funcs,
    .m_name = "core.arrop.mt",
    .m_size = -1,
    .m_doc = 0,
};

DLLEXPORT PyObject* arr_mt_py_add(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    if(!PyArg_ParseTuple(args, "O!O!O!", &arr_handle_def, &in1, &arr_handle_def, &in2, &arr_handle_def, &ret)) return 0;
    arr_add_mt(in1->data, in2->data, in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_mt_py_sub(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    if(!PyArg_ParseTuple(args, "O!O!O!", &arr_handle_def, &in1, &arr_handle_def, &in2, &arr_handle_def, &ret)) return 0;
    arr_sub_mt(in1->data, in2->data, in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_mt_py_scale(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *ret; 
    double scaler;
    if(!PyArg_ParseTuple(args, "O!dO!", &arr_handle_def, &in1, &scaler, &arr_handle_def, &ret)) return 0;
    arr_scale_mt(in1->data, float_to_qfix(scaler), in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_mt_py_emul(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    if(!PyArg_ParseTuple(args, "O!O!O!", &arr_handle_def, &in1, &arr_handle_def, &in2, &arr_handle_def, &ret)) return 0;
    arr_emul_mt(in1->data, in2->data, in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_mt_py_mmul(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    uint16_t m, k, n;
    if(!PyArg_ParseTuple(args, "O!O!HHHO!", &arr_handle_def, &in1, &arr_handle_def, &in2, &m, &k, &n, &arr_handle_def, &ret)) return 0;
    arr_mmul_mt(in1->data, in2->data,m, k, n, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_mt_py_mtpose(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *ret; 
    uint16_t m, n;
    if(!PyArg_ParseTuple(args, "O!HHO!", &arr_handle_def, &in1, &m, &n, &arr_handle_def, &ret)) return 0;
    arr_mtpose_mt(in1->data, m, n, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}