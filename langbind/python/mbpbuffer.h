#ifndef _mbpbuffer
#define _mbpbuffer

#include <Python.h>
#include "../../mlpCCore/matrix/matrix_static.h"

typedef struct mbp_topy{
    PyObject_HEAD
    uint32_t orisize;
    matrix_bp info;
}matrixbp_py;

//module mbp16dpy

//[EXPOSE]python: matrixbp.__init__(rows:int, cols:int)
DLLEXPORT PyObject* mbp_new(PyTypeObject *tp, PyObject *args, PyObject *kwargs);
DLLEXPORT void mbp_dealloc(PyObject *self);

//[EXPOSE]python: matrixbp.__repr__()
DLLEXPORT PyObject *mbp_repr(PyObject *self);

//[EXPOSE]python: matrixbp.fromlist(lstin: List[int|float]) -> None
DLLEXPORT PyObject *mbp_fromlist(PyObject *self, PyObject *args);
//[EXPOSE]python: matrixbp.fromrand() -> None
DLLEXPORT PyObject *mbp_fromrand(PyObject *self, PyObject *args);

//[EXPOSE]python: matrixbp.tolist(**kwargs[<optional>arr2d: bool|int = 0]) -> List[float] | List[List[float]]
DLLEXPORT PyObject *mbp_tolist(PyObject *self, PyObject *args, PyObject *args_dict);

//[EXPOSE]py: matrixbp.shape() -> Tuple[int, int]
DLLEXPORT PyObject *mbp_shape(PyObject *self, PyObject *args);
//[EXPOSE]py: matrixbp.capacity() -> int
DLLEXPORT PyObject *mbp_capacity(PyObject *self, PyObject *args);
//[EXPOSE]py: matrixbp.__getitem__(index: Tuple[int, int] | int) -> float
DLLEXPORT PyObject *mbp_index(PyObject *self, PyObject *args);

static PyMethodDef fn_mbp_py[] = {
    {"shape", mbp_shape, METH_VARARGS,0},
    {"capacity", mbp_capacity, METH_VARARGS,0},
    {"fromlist", mbp_fromlist, METH_VARARGS, "initlize from a list"},
    {"fromrand", mbp_fromrand, METH_VARARGS, "initlize with a random value"},
    {"tolist", (PyCFunction)(PyCFunctionWithKeywords)mbp_tolist, METH_VARARGS | METH_KEYWORDS, "dump a matrixbp to a python list in 1d or 2d"},
    {0,0,0,0},
};

//PyObject::operator[]
static PyMappingMethods mbp_opidx = {
    .mp_subscript = mbp_index
};

static PyTypeObject mbp_py_tpdef = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "mbp16dpy.matrixbp",
    .tp_basicsize = sizeof(matrixbp_py),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = mbp_new,
    .tp_dealloc = mbp_dealloc,
    .tp_repr = mbp_repr,
    .tp_methods = fn_mbp_py,
    .tp_as_mapping = &mbp_opidx,
};

#endif