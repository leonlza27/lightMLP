#include "matrixbp2py.h"

PyObject *mbp_new(PyTypeObject *tp, PyObject *args, PyObject *kwargs){
    matrixbp_py *obnew = (matrixbp_py*)tp->tp_alloc(tp, 0);
    if(!obnew) return 0;
    obnew->info = 0;
    return (PyObject*) obnew;
}

int mbp_init(PyObject *self, PyObject *args, PyObject *kwargs){
    uint16_t rows, cols;
    if(!PyArg_ParseTuple(args, "HH", &rows, &cols)) return -1;
    matrixbp_py *obj = (matrixbp_py*)self;
    obj->info = alloc_matrix_bp(rows, cols);
    if(!obj->info){
        PyErr_SetString(PyExc_MemoryError, "cannot alloc a matrix\n");
        return -1;
    }
    return 0;
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

PyObject *mbp_add(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "OOO", &m1, &m2, &mr)) return 0;
    matrix_bp_add(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_sub(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "OOO", &m1, &m2, &mr)) return 0;
    matrix_bp_sub(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_scale(PyObject *self, PyObject *args){
    matrixbp_py *m1, *mr;
    PyObject *sc_o;
    if(!PyArg_ParseTuple(args, "OOO", &m1, &sc_o, &mr)) return 0;
    qfix sc_v;
    if(PyLong_Check(sc_o)){
        sc_v = float_to_qfix(PyLong_AsLong(sc_o));
    }else if(PyFloat_Check(sc_o)){
        sc_v = float_to_qfix(PyFloat_AsDouble(sc_o));
    }else{
        PyErr_SetString(PyExc_TypeError,"arg \"sc_v'\" unexcepted type: not a int or float");
    }

    matrix_bp_scale(m1->info, sc_v, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_mulByElem(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "OOO", &m1, &m2, &mr)) return 0;
    matrix_bp_mulptyByElem(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_mul(PyObject *self, PyObject *args){
    matrixbp_py *m1, *m2, *mr;
    if(!PyArg_ParseTuple(args, "OOO", &m1, &m2, &mr)) return 0;
    matrix_bp_mulpty(m1->info, m2->info, mr->info);
    Py_RETURN_NONE;
}

PyObject *mbp_transpose(PyObject *self, PyObject *args){
    matrixbp_py *m1, *mr;
    if(!PyArg_ParseTuple(args, "OO", &m1, &mr)) return 0;
    matrix_bp_transpose(m1->info, mr->info);
    Py_RETURN_NONE;
}

PyMODINIT_FUNC PyInit_bp16p16matrix(){
    PyObject *m;
    if(PyType_Ready(&mbp_py_tpdef)) return 0;
    m = PyModule_Create(&matrixbp_topy_root);
    if(!m) return 0;
    if(PyModule_AddObjectRef(m, "matrixbp", (PyObject*)&mbp_py_tpdef)){
        Py_DecRef(m);
        return 0;
    }
    return m;
}
