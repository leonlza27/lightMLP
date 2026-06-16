#ifndef _mbpbuffer
#define _mbpbuffer

#include <Python.h>
#include "../../mlpCCore/matrix/matrix_static.h"

typedef struct _mbp_topy{
    PyObject_HEAD
    matrix_bp info;
}matrixbp_py;

DLLEXPORT PyObject* mbp_new(PyTypeObject *tp, PyObject *args, PyObject *kwargs);
DLLEXPORT void mbp_dealloc(PyObject *self);

DLLEXPORT PyObject *mbp_repr(PyObject *self);

DLLEXPORT PyObject *mbp_fromlist(PyObject *self, PyObject *args);
DLLEXPORT PyObject *mbp_fromrand(PyObject *self, PyObject *args);

DLLEXPORT PyObject *mbp_tolist(PyObject *self, PyObject *args, PyObject *args_dict);

static PyMethodDef fn_mbp_py[] = {
    {"fromlist", mbp_fromlist, METH_VARARGS, "initlize from a list"},
    {"fromrand", mbp_fromrand, METH_VARARGS, "initlize with a random value"},
    {"tolist", (PyCFunction)(PyCFunctionWithKeywords)mbp_tolist, METH_VARARGS | METH_KEYWORDS, "dump a matrixbp to a python list in 1d or 2d"},
    {0,0,0,0},
};

static PyTypeObject mbp_py_tpdef = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "libmbp16d.matrixbp",
    .tp_basicsize = sizeof(matrixbp_py),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = mbp_new,
    .tp_dealloc = mbp_dealloc,
    .tp_repr = mbp_repr,
    .tp_methods = fn_mbp_py,
};

#endif