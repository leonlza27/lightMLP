#ifndef _mlpcore2py
#define _mlpcore2py
#include <python3.12/Python.h>
#include "mlpCCore/mlp/mlp.h"
#include "mlpCCore/matrix/matrix_static.h"
#include "matrixbp2py.h"

typedef struct _netstruct_py{
    PyObject_HEAD;
    NetLyrConf *netsrc;
    uint16_t size;
//src:list[tuple[indim, outdim, ac, dexa<optional>]], 用于引用计数控制
}netstruct_py;

static PyObject *netstructpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
static void netstructpy_empty(PyObject *self);

static PyObject *netstructpy_swnet(PyObject *self);

static PyObject *constructnet(PyObject *self, PyObject *args);

static PyTypeObject netstructpy_tpdef = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "core.netdef",
    .tp_basicsize = sizeof(netstruct_py),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = netstructpy_new,
    .tp_dealloc = netstructpy_empty,
    .tp_repr = netstructpy_swnet,
};

typedef struct _mlptrain2py{
    PyObject_HEAD;
    mlpNetTrainInfo info;
    netstruct_py *datasrc_py;   //原始数据,用于引用计数控制
}mlptrain2py;

static PyObject *mlptrainerpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
static void mlptrainerpy_empty(PyObject *self);

static PyObject *mlptrainerpy_execute(PyObject *self, PyObject *args);
static PyObject *mlptrainerpy_backward(PyObject *self, PyObject *args);

static PyMethodDef mlptrainerpy_memberfns[] = {
    {"execute", mlptrainerpy_execute, METH_VARARGS, "run a full mlpnet with a matrixbp input"},
    {"backward", mlptrainerpy_backward, METH_VARARGS, "upgrade a net through backeard with a inital grad"},
    {0,0,0,0},
};

static PyTypeObject mlptrainerpy_tpdef = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "core.mlptrainer",
    .tp_basicsize = sizeof(mlptrain2py),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = mlptrainerpy_new,
    .tp_dealloc = mlptrainerpy_empty,
    .tp_methods = mlptrainerpy_memberfns,
};

typedef struct _mlpexec2py{
    PyObject_HEAD;
    mlpNetRefInfo info;
    netstruct_py *datasrc_py;   //原始数据,用于引用计数控制
}mlpexec2py;

static PyMethodDef core_modulefns[] = {
    {"constructnet", constructnet, METH_VARARGS, 0},
    {0,0,0,0},
};

static struct PyModuleDef lightmlp_topy_root = {
    PyModuleDef_HEAD_INIT, "core", 0, 0,
    .m_methods = core_modulefns,
};

PyMODINIT_FUNC PyInit_core();

#endif