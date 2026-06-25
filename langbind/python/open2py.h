#ifndef _mlpccore_pyapi
#define _mlpccore_pyapi

#include <Python.h>
#include "methodobject.h"
#include "mlpCCore/mlp/mlp.h"
#include "pytypedefs.h"

//module corepy

struct _mbp_topy;
typedef struct _mbp_topy matrixbp_py;

//[EXPOSE] class netdef
typedef struct netdefpy{
    PyObject_HEAD
    netLyrConf *nstruct;
    uint32_t lyrcnt;
}netdefpy;

DLLEXPORT PyObject *netdefpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
DLLEXPORT void netdefpy_dealloc(PyObject *self);

//[EXPOSE]python: buildnet(netdef: List[Tuple[int , int, int<enum core.actp>, float]]) -> None
DLLEXPORT PyObject *buildnet(PyObject *_rtime, PyObject *args);

//[EXPOSE]python: savemodel(modelsrc: netdef, filename: str, **kwargs[filetype: str = "bin", exposemrk: str = ""]) -> None
DLLEXPORT PyObject *dumpmodel_frompy(PyObject *_rtime, PyObject *args, PyObject *args_dict);

//[EXPOSE]python: loadmodel(filepath: str) -> netdef
DLLEXPORT PyObject *load_frombin(PyObject *_rtime, PyObject *args);

PyTypeObject netdefpy_tpdef = {
    PyVarObject_HEAD_INIT(0, 0)
    .tp_basicsize = sizeof(netdefpy),
    .tp_itemsize = 0,
    .tp_name = "core_py.netdef",
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = netdefpy_new,
    .tp_dealloc = netdefpy_dealloc,
};

//[EXPOSE] class mlptrain
typedef struct mlpTrainStatPy{
    PyObject_HEAD
    mlpTrainStatus statloc;
    netdefpy *modelsrc;
}mlpTrainStatPy;

//[EXPOSE]python: mlptrain.__init__(netsrc: netdef, **kwargs[<optional> totalgrad_cap: bool | int])
//same for exec_only one, without kwargs
DLLEXPORT PyObject *mlptrainpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
DLLEXPORT void mlptrainpy_dealloc(PyObject *self);

//[EXPOSE]python: mlptrain.execute(vecin: mbp16dpy.matrixbp, <optional>vecout: mbp16dpy.matrixbp) -> mbp16dpy.matrixbp
//same for exec_only one
//if vecout passed, it will return it after modifying, else create a new one
DLLEXPORT PyObject *mlptrainpy_mexecute(PyObject *self, PyObject *args);
//[EXPOSE]python: mlptrain.backward(grad: matrixbp, lr: Float) -> None
DLLEXPORT PyObject *mlptrainpy_mbackward(PyObject *self, PyObject *args);
//[EXPOSE]python: mlptrain.finalgrads(<optional>vecgrad0: mbp16dpy.matrixbp) -> mbp16dpy.matrixbp
//if vecgrad0 passed, it will return it after modifying, else create a new one
DLLEXPORT PyObject *mlptrainpy_mgetfinalgrads(PyObject *self, PyObject *args);
//[EXPOSE]python: mlptrain.isgradsaver() -> bool
DLLEXPORT PyObject *mlptrainpy_mcheckgradsaver(PyObject *self, PyObject *args);

static PyMethodDef mlptrainpy_memberfns[] = {
    {"execute", mlptrainpy_mexecute, METH_VARARGS, 0},
    {"backward", mlptrainpy_mbackward, METH_VARARGS, 0},
    {"finalgrads", mlptrainpy_mgetfinalgrads, METH_VARARGS, 0},
    {"isgradsaver", mlptrainpy_mcheckgradsaver, METH_VARARGS, 0},
    {0,0,0,0},
};

//[EXPOSE]python: savegrads(model_or_totalgrads: mlptrain, dest_totalgrads_cap: mlptrain)
DLLEXPORT PyObject *mlptrainpy_totalgrads_savegrads(PyObject *_rtime, PyObject *args);

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


//[EXPOSE] class mlpexec
typedef struct mlpExecStatPy{
    PyObject_HEAD
    mlpExecStatus statloc;
    netdefpy *modelsrc;
}mlpExecStatPy;

//[EXPOSE]python: mlpexec.__init__(netsrc: netdef)
DLLEXPORT PyObject *mlpexecpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict);
DLLEXPORT void mlpexecpy_dealloc(PyObject *self);

//[EXPOSE]python: mlpexec.execute(vecin: mbp16dpy.matrixbp, <optional>vecout: mbp16dpy.matrixbp) -> mbp16dpy.matrixbp
DLLEXPORT PyObject *mlpexecpy_mexecute(PyObject *self, PyObject *args);
//[EXPOSE]python: mlpexec.__call__(vecin: mbp16dpy.matrixbp, <optional>vecout: mbp16dpy.matrixbp) -> mbp16dpy.matrixbp
//<NOTE: wrapper of mlpexec.execute>
DLLEXPORT PyObject *mlpexecpy_mexecute_opcall(PyObject *self, PyObject *args, PyObject *args_dict);

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
    {"savemodel", (PyCFunction)(PyCFunctionWithKeywords)dumpmodel_frompy, METH_VARARGS | METH_KEYWORDS, 0},
    {"loadmodel", load_frombin, METH_VARARGS, 0},
    {"savegrads", mlptrainpy_totalgrads_savegrads, METH_VARARGS, "save totalgrads for data parral training, with a mlptrain object to store total grads"},
    {0,0,0,0},
};

static struct PyModuleDef lmlpcore = {
    PyModuleDef_HEAD_INIT, "libcorepy", 0, 0,
    .m_methods = libcorepy_modulefns,
};

//[EXPOSE] module corepy contains enum actp(as submodule)
//strats with 0: [ReLU, ReLU6, LeakyReLU, Sigmoid, Sigmoid_hard, Tanh, Tanh_hard, Sign, Pass]

#endif
