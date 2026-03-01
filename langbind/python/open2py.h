#ifndef _mlpccore_pyapi
#define _mlpccore_pyapi

#include <python3.13/Python.h>
#include "../../mlpCCore/mlp/mlp.h"
#include "matrixbp2py.h"

typedef struct netdefpy{
    PyObject_HEAD;
    netLyrConf *nstruct;
    uint32_t lyrcnt;
}netdefpy;

PyObject *netdefpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
void netdefpy_dealloc(PyObject *self);

PyObject *buildnet(PyObject *_rtime, PyObject *args);

//dict contains kwd: filetype, exposemrk
//norm_args: modelsrc, filename
//bin as default
PyObject *dumpmodel_frompy(PyObject *_rtime, PyObject *args, PyObject *args_dict);

PyObject *load_frombin(PyObject *_rtime, PyObject *args);

PyTypeObject netdefpy_tpdef = {
    PyVarObject_HEAD_INIT(0, 0)
    .tp_basicsize = sizeof(netdefpy),
    .tp_itemsize = 0,
    .tp_name = "lightmlpcore_py.netdef",
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = netdefpy_new,
    .tp_dealloc = netdefpy_dealloc,
};

typedef struct mlpTrainStatPy{
    PyObject_HEAD;
    mlpTrainStatus statloc;
    netdefpy *modelsrc;
}mlpTrainStatPy;

//in python: mlptrain.__init__(netsrc: netdef)
//same for exec_only one
PyObject *mlptrainpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
void mlptrainpy_dealloc(PyObject *self);

//in python: mlptrain.execute(vecin: matrixbp)
//same for exec_only one
PyObject *mlptrainpy_mexecute(PyObject *self, PyObject *args);
//in python: mlptrain.backward(grad: matrixbp, lr: Float)
PyObject *mlptrainpy_mbackward(PyObject *self, PyObject *args);

static PyMethodDef mlptrainpy_memberfns[] = {
    {"execute", mlptrainpy_mexecute, METH_VARARGS, 0},
    {"backward", mlptrainpy_mbackward, METH_VARARGS, 0},
    {0,0,0,0},
};

PyTypeObject mlptrainpy_tpdef = {
    PyVarObject_HEAD_INIT(0, 0)
    .tp_basicsize = sizeof(mlpTrainStatPy),
    .tp_itemsize = 0,
    .tp_name = "lightmlpcore_py.mlptrain",
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = mlptrainpy_new,
    .tp_dealloc = mlptrainpy_dealloc,
    .tp_methods = mlptrainpy_memberfns,
};


typedef struct mlpExecStatPy{
    PyObject_HEAD;
    mlpExecStatus statloc;
    netdefpy *modelsrc;
}mlpExecStatPy;

PyObject *mlpexecpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
void mlpexecpy_dealloc(PyObject *self);

PyObject *mlpexecpy_mexecute(PyObject *self, PyObject *args);
//use mlpexec.__call__() to call the upper func
PyObject *mlpexecpy_mexecute_opcall(PyObject *self, PyObject *args, PyObject *args_dict);

static PyMethodDef mlpexecpy_memberfns[] = {
    {"execute", mlpexecpy_mexecute, METH_VARARGS, 0},
    {0,0,0,0},
};

PyTypeObject mlpexecpy_tpdef = {
    PyVarObject_HEAD_INIT(0,0)
    .tp_basicsize =sizeof(mlpExecStatPy),
    .tp_itemsize = 0,
    .tp_name = "lightmlpcore_py.mlpexec",
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = mlpexecpy_new,
    .tp_dealloc = mlpexecpy_dealloc,
    .tp_methods = mlpexecpy_memberfns,
    .tp_call = mlpexecpy_mexecute_opcall,
};

static PyMethodDef libcorepy_modulefns[] = {
    {"buildnet", buildnet, METH_VARARGS, "build a net descrption from 0"},
    {0,0,0,0},
};

static struct PyModuleDef lmlpcore = {
    PyModuleDef_HEAD_INIT, "libcorepy", 0, 0,
    .m_methods = libcorepy_modulefns,
};

PyMODINIT_FUNC PyInit_libcorepy();


#endif
