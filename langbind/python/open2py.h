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

PyTypeObject netdefpy_tpdef = {
    PyVarObject_HEAD_INIT(0, 0)
    .tp_basicsize = sizeof(netdefpy),
    .tp_itemsize = 0,
    .tp_name = "libcore.netdef",
    .tp_new = netdefpy_new,
    .tp_dealloc = netdefpy_dealloc,
};


typedef struct mlpTrainStatPy{
    PyObject_HEAD;
    mlpTrainStatus statloc;
    netdefpy *modelsrc;
}mlpTrainStatPy;

PyObject *mlptrainpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
void mlptrainpy_dealloc(PyObject *self);

//in python: mlptrain.execute(vecin: matrixbp)
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
    .tp_name = "mlptrain",
    .tp_new = mlptrainpy_new,
    .tp_dealloc = mlptrainpy_dealloc,
    .tp_methods = mlptrainpy_memberfns,
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
