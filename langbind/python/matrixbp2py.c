#include "matrixbp2py.h"
#include <stdlib.h>
#include <time.h>

PyObject *mbp_new(PyTypeObject *tp, PyObject *args, PyObject *kwargs){
    uint16_t rows, cols;
    if(!PyArg_ParseTuple(args, "HH", &rows, &cols)) return 0;
    matrixbp_py *obnew = (matrixbp_py*)tp->tp_alloc(tp, 0);
    matrix_bp mnew = alloc_matrix_bp(rows, cols);
    if(!obnew || !mnew) return 0;
    obnew->info = mnew;
    return (PyObject*) obnew;
}

void mbp_dealloc(PyObject *self){
    matrixbp_py *obj = (matrixbp_py*)self;
    if(obj->info) free(obj->info);
    Py_TYPE(obj)->tp_free(obj);
}

PyObject *mbp_repr(PyObject *self){
    matrixbp_py *obj = (matrixbp_py*)self;
    uint16_t rows, cols;
    rows = obj->info->rows;
    cols = obj->info->cols;
    qfix *data = obj->info->data;
    printf("[matrixbp at shape %d rows * %d cols]\n", rows, cols);
    for(uint16_t i = 0; i < rows; i++){
        putc('|', stdout);
        putc(' ', stdout);
        qfix* r_this = data + i * cols;
        for(uint16_t j = 0; j < cols; j++){
            printf("%.4f, ", qfix_to_float64(r_this[j]));
        }
        putc('|', stdout);
        putc('\n', stdout);
    }
    return PyUnicode_FromString("");
}

PyObject *mbp_fromlist(PyObject *self, PyObject *args){
    PyObject *lstin;
    if(!PyArg_ParseTuple(args, "O", &lstin)){
        PyErr_SetString(PyExc_TypeError, "input must a list[int | float]\n");
        return 0;
    }
    if(!PyList_Check(lstin)){
        PyErr_SetString(PyExc_TypeError, "input must a list[int | float]\n");
        return 0;
    }

    matrix_bp tg = ((matrixbp_py*)self)->info;
    uint32_t size = tg->cols * tg->rows;

    if(size > PyList_GET_SIZE(lstin)){
        PyErr_SetString(PyExc_TypeError, "size of list isnot equal or larger than capacity of matrix\n");
        return 0;
    }

    qfix *dtg = tg->data;
    PyObject *cur;

    for(uint32_t i = 0; i < size; i++){
        cur = PyList_GetItem(lstin, i);
        if(PyFloat_Check(cur)){
            dtg[i] = float_to_qfix(PyFloat_AsDouble(cur));
            continue;
        }
        if(PyLong_Check(cur)){
            dtg[i] = float_to_qfix(PyLong_AsLong(cur));
            continue;
        }
        PyErr_SetString(PyExc_TypeError, "list contains a non_number(float or int) value\n");
        return 0;
    }
    Py_RETURN_NONE;
}

PyObject *mbp_fromrand(PyObject *self, PyObject *args){
    matrixbp_py *obj = (matrixbp_py*)self;
    uint32_t size = obj->info->rows * obj->info->cols;
    qfix *data = obj->info->data;
    srand(time(0));
    
    for(uint32_t i = 0; i < size; i++) data[i] = ((rand() % (1 << 16)) - (1 << 15)) << 8;
    
    Py_RETURN_NONE;
}

PyObject *mbp_tolist(PyObject *self, PyObject *args, PyObject *args_dict){
    matrixbp_py *obj = (matrixbp_py*)self;
    PyObject *is2d = 0;
    static char *kwds[] = {"arr2d", 0};
    if(!PyArg_ParseTupleAndKeywords(args, args_dict, "|O", kwds, &is2d)) return 0;
    uint16_t rows = obj->info->rows;
    uint16_t cols = obj->info->cols;
    qfix *data = obj->info->data;
    PyObject *retlst;
    if(!is2d) goto _ret_normlst;
    uint8_t is2d_b = PyBool_Check(is2d);
    uint8_t is2d_i =  PyLong_Check(is2d); 
    if(!(is2d_b || is2d_i)){
        PyErr_SetString(PyExc_TypeError, "arg \"arr2d\" unexcepted type: not a bool or int");
        return 0;
    }
    if((is2d_b && (is2d == Py_True)) || (is2d_i && PyLong_AsLong(is2d))){
        retlst = PyList_New(rows);
        for(uint16_t i = 0; i < rows; i++){
            PyObject *rcolcur = PyList_New(cols);
            qfix *dcolcur = data + i * cols;
            for(uint16_t j = 0; j < cols; j++){
                PyObject *vacur = PyFloat_FromDouble(qfix_to_float64(dcolcur[j]));
                PyList_SetItem(rcolcur, j, vacur);
            }
            PyList_SetItem(retlst, i, rcolcur);
        }

        return retlst;
    }

    _ret_normlst:

    uint32_t size = cols * rows;
    retlst = PyList_New(size);
    for(uint32_t i = 0; i < size; i++){
        PyObject *vacur = PyFloat_FromDouble(qfix_to_float64(data[i]));
        PyList_SetItem(retlst, i, vacur);
    }
    return retlst;
}

PyObject *mbp_add(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!",&mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_add(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_sub(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_sub(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_scale(PyObject *self, PyObject *args){
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

PyObject *mbp_mulByElem(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_mulptyByElem(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_mul(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "O!O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &m2, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_mulpty(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_transpose(PyObject *self, PyObject *args){
    matrixbp_py *m1, *mr;
    if(!PyArg_ParseTuple(args, "O!O!", &mbp_py_tpdef, &m1, &mbp_py_tpdef, &mr)) return 0;
    matrix_bp_transpose(m1->info, mr->info);
    Py_RETURN_NONE;
}

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
