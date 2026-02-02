#ifndef _matrixbp2py
#define _matrixbp2py
#include <python3.12/Python.h>
#include "mlpCCore/matrix/matrix_static.h"


typedef struct _mbp_topy{
    PyObject_HEAD
    matrix_bp info;
}matrixbp_py;

static PyObject* mbp_new(PyTypeObject *tp, PyObject *args, PyObject *kwargs);
static int mbp_init(PyObject *self, PyObject *args, PyObject *kwargs);
static void mbp_dealloc(PyObject *self);

static PyObject *mbp_repr(PyObject *self);

static PyObject *mbp_fromlist(PyObject *self, PyObject *args);
static PyObject *mbp_fromrand(PyObject *self, PyObject *args);

static PyObject *mbp_tolist(PyObject *self, PyObject *args, PyObject *args_dict);

static PyMethodDef fn_mbp_py[] = {
    {"fromlist", mbp_fromlist, METH_VARARGS, "initlize from a list"},
    {"fromrand", mbp_fromrand, METH_VARARGS, "initlize with a random value"},
    {"tolist", mbp_tolist, METH_VARARGS | METH_KEYWORDS, "dump a matrixbp to a python list in 1d or 2d"},
    {0}
};

static PyTypeObject mbp_py_tpdef = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "bp16p16matrix.matrixbp",
    .tp_basicsize = sizeof(matrixbp_py),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = mbp_new,
    .tp_init = mbp_init,
    .tp_dealloc = mbp_dealloc,
    .tp_repr = mbp_repr,
    .tp_methods = fn_mbp_py,
};


static PyObject *mbp_add(PyObject *self, PyObject *args);
static PyObject *mbp_sub(PyObject *self, PyObject *args);
static PyObject *mbp_scale(PyObject *self, PyObject *args);
static PyObject *mbp_mulByElem(PyObject *self, PyObject *args);
static PyObject *mbp_mul(PyObject *self, PyObject *args);
static PyObject *mbp_transpose(PyObject *self, PyObject *args);

static PyMethodDef moudlefns[] = {
    {"madd", mbp_add, METH_VARARGS, 0},
    {"msub", mbp_sub, METH_VARARGS, 0},
    {"mscale", mbp_scale, METH_VARARGS, 0},
    {"mmulElem", mbp_mulByElem, METH_VARARGS, 0},
    {"mmul", mbp_mul, METH_VARARGS, 0},
    {"mtpose", mbp_transpose, METH_VARARGS, 0},
    {0, 0, 0, 0}
};

static struct PyModuleDef matrixbp_topy_root = {
    PyModuleDef_HEAD_INIT, "bp16p16matrix", 0, 0,
    .m_methods = moudlefns
};

PyMODINIT_FUNC PyInit_bp16p16matrix();

#endif