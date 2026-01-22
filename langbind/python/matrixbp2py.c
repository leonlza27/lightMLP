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
        putc('\t', stdout);
        qfix* r_this = data + i * cols;
        for(uint16_t j = 0; j < cols; j++){
            printf("%4d\t", r_this[j]);
        }
        putc('|', stdout);
        putc('\n', stdout);
    }
    return PyUnicode_FromString("");
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
