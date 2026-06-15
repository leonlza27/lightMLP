#include <Python.h>
#include <stdio.h>
#include "../../mlpCCore/matrix/matrix_static.h"
#include "mbpbuffer.h"

DLLEXPORT PyObject *mbp_add(PyObject *self, PyObject *args){
    printf("func called\n");
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!",&mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    printf("entering core\n");
    matrix_bp_add(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

DLLEXPORT PyObject *mbp_sub(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_sub(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

DLLEXPORT PyObject *mbp_scale(PyObject *self, PyObject *args){
    matrixbp_py *m1, *mr;
    PyObject *sc_o;
    if(!PyArg_ParseTuple(args, "O!OO!", &mbp_py_tpdef, &m1, &sc_o, &mbp_py_tpdef, &mr)) return 0;
    qfix sc_v;
    if(PyLong_Check(sc_o)){
        sc_v = float_to_qfix(PyLong_AsLong(sc_o));
    }else if(PyFloat_Check(sc_o)){
        sc_v = float_to_qfix(PyFloat_AsDouble(sc_o));
    }else{
        PyErr_SetString(PyExc_TypeError,"arg \"sc_v\" unexcepted type: not a int or float");
        return 0;
    }

    matrix_bp_scale(m1->info, sc_v, mr->info);
    Py_RETURN_NONE;
}

DLLEXPORT PyObject *mbp_mulByElem(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_mulptyByElem(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

DLLEXPORT PyObject *mbp_mul(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_mulpty(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

DLLEXPORT PyObject *mbp_transpose(PyObject *self, PyObject *args){
    matrixbp_py *m1, *mr;
    if(!PyArg_ParseTuple(args, "O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_transpose(m1->info, mr->info);
    Py_RETURN_NONE;
}

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
    PyModuleDef_HEAD_INIT, "libmbp16d", 0, 0,
    .m_methods = moudlefns
};

PyMODINIT_FUNC PyInit_libmbp16d(){
    PyObject *m;
    if(PyType_Ready(&mbp_py_tpdef)) return 0;
    m = PyModule_Create(&matrixbp_topy_root);
    if(!m) return 0;
    if(PyModule_AddObject(m, "matrixbp", (PyObject*)&mbp_py_tpdef)){
        Py_DecRef(m);
        return 0;
    }
    return m;
}
