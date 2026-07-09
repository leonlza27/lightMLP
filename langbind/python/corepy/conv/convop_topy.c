#include "convop_topy.h"
#include "../../mbpbuffer_topy/mbpbuffer.h"

extern PyTypeObject *mbp_tpdef_ref;

PyObject* convoppy_averpool(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0;
    PyObject *poolsize_tuple, *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    if (!PyArg_ParseTuple(args, "O!OOO!", mbp_tpdef_ref, &graphin, &poolsize_tuple, &step_tuple, mbp_tpdef_ref,
                          &graphout)) return 0;
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &pool_c, &pool_r)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;

    conv_averpool(graphin->info->data, graphin->info->rows, graphin->info->cols,
                    pool_c, pool_r, stepr, stepc, graphout->info->data);

    Py_RETURN_NONE;
}

PyObject* convoppy_maxpool(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0, *maxscap = 0;
    PyObject *poolsize_tuple, *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    if (!PyArg_ParseTuple(args, "O!OOO!O!", mbp_tpdef_ref, &graphin, &poolsize_tuple, &step_tuple, mbp_tpdef_ref,
                          &graphout, mbp_tpdef_ref, &maxscap)) return 0;
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &pool_c, &pool_r)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;

    conv_maxpool(graphin->info->data, graphin->info->rows, graphin->info->cols,
                    pool_c, pool_r, stepr, stepc, graphout->info->data, (uint16_t*)maxscap->info->data);

    Py_RETURN_NONE;
}

PyObject* convoppy_averpool_backward(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0;
    PyObject *poolsize_tuple, *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    if (!PyArg_ParseTuple(args, "O!OOO!", mbp_tpdef_ref, &graphin, &poolsize_tuple, &step_tuple, mbp_tpdef_ref,
                          &graphout)) return 0;
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &pool_c, &pool_r)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;

    conv_averpool_backward(graphin->info->data, graphout->info->rows, graphout->info->cols,
                    pool_c, pool_r, stepr, stepc, graphout->info->data);

    Py_RETURN_NONE;
}

PyObject* convoppy_maxpool_backward(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0, *maxscap = 0;
    PyObject *poolsize_tuple, *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    if (!PyArg_ParseTuple(args, "O!O!OOO!", mbp_tpdef_ref, &graphin, mbp_tpdef_ref, &maxscap, &poolsize_tuple, &step_tuple, mbp_tpdef_ref,
                          &graphout)) return 0;
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &pool_c, &pool_r)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;

    conv_maxpool_backward(graphin->info->data, (uint16_t*)maxscap->info->data,
                    graphin->info->rows, graphin->info->cols,
                    pool_c, pool_r, stepr, stepc, graphout->info->data);

    Py_RETURN_NONE;
}

PyObject* convoppy_padding(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0;
    uint16_t r_extend, c_extend;
    if (!PyArg_ParseTuple(args, "O!iiO!", mbp_tpdef_ref, &graphin, &r_extend, &c_extend, mbp_tpdef_ref,
                          &graphout)) return 0;

    conv_padding(graphin->info->data, graphin->info->rows, graphin->info->cols,
                    r_extend, c_extend, graphout->info->data);
    Py_RETURN_NONE;
}

PyObject* convoppy_cutedges(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0;
    uint16_t r_sub, c_sub;
    if (!PyArg_ParseTuple(args, "O!iiO!", mbp_tpdef_ref, &graphin, &r_sub, &c_sub, mbp_tpdef_ref,
                          &graphout)) return 0;

    conv_cutedges(graphin->info->data, graphin->info->rows, graphin->info->cols,
                    r_sub, c_sub, graphout->info->data);
    Py_RETURN_NONE;
}

PyObject* convoppy_divelem(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin = 0;
    PyObject *dest_list, *poolsize_tuple, *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    uint16_t expand;
    Py_ssize_t dest_size;
    qfix **arrouts;
    matrixbp_py **dest_objs;
    
    if (!PyArg_ParseTuple(args, "O!OOiO", mbp_tpdef_ref, &graphin, &poolsize_tuple, &step_tuple, &expand, &dest_list)) return 0;
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &pool_c, &pool_r)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;
    
    if (!PyList_Check(dest_list)) {
        PyErr_SetString(PyExc_TypeError, "dest must be a list");
        return 0;
    }
    
    dest_size = PyList_Size(dest_list);
    arrouts = (qfix**)malloc(dest_size * sizeof(qfix*));
    dest_objs = (matrixbp_py**)malloc(dest_size * sizeof(matrixbp_py*));
    
    for (Py_ssize_t i = 0; i < dest_size; i++) {
        PyObject *item = PyList_GetItem(dest_list, i);
        if (!PyObject_TypeCheck(item, mbp_tpdef_ref)) {
            free(arrouts);
            free(dest_objs);
            PyErr_SetString(PyExc_TypeError, "All dest items must be matrixbp objects");
            return 0;
        }
        dest_objs[i] = (matrixbp_py*)item;
        arrouts[i] = dest_objs[i]->info->data;
    }
    
    conv_divelem(graphin->info->data, graphin->info->rows, graphin->info->cols,
                 pool_r, pool_c, expand, stepr, stepc, arrouts);
    
    free(arrouts);
    free(dest_objs);
    Py_RETURN_NONE;
}

PyObject* convoppy_mergeelem(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphout = 0;
    PyObject *input_list, *poolsize_tuple, *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    uint16_t expand;
    Py_ssize_t input_size;
    const qfix **arrin;
    matrixbp_py **input_objs;
    
    if (!PyArg_ParseTuple(args, "OOiiO!O", &input_list, &poolsize_tuple, &expand, &step_tuple, mbp_tpdef_ref, &graphout)) return 0;
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &pool_c, &pool_r)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;
    
    if (!PyList_Check(input_list)) {
        PyErr_SetString(PyExc_TypeError, "input must be a list");
        return 0;
    }
    
    input_size = PyList_Size(input_list);
    arrin = (const qfix**)malloc(input_size * sizeof(qfix*));
    input_objs = (matrixbp_py**)malloc(input_size * sizeof(matrixbp_py*));
    
    for (Py_ssize_t i = 0; i < input_size; i++) {
        PyObject *item = PyList_GetItem(input_list, i);
        if (!PyObject_TypeCheck(item, mbp_tpdef_ref)) {
            free((void*)arrin);
            free(input_objs);
            PyErr_SetString(PyExc_TypeError, "All input items must be matrixbp objects");
            return 0;
        }
        input_objs[i] = (matrixbp_py*)item;
        arrin[i] = input_objs[i]->info->data;
    }
    
    conv_mergeelem(arrin, graphout->info->rows, graphout->info->cols,
                   pool_r, pool_c, expand, stepr, stepc, graphout->info->data);
    
    free((void*)arrin);
    free(input_objs);
    Py_RETURN_NONE;
}

PyObject* convoppy_fwdcalc(PyObject* _rtime, PyObject* args){
    matrixbp_py *input = 0, *weights = 0, *dest = 0;
    PyObject *step_tuple;
    uint16_t stepr, stepc;
    uint16_t pool_c, pool_r;
    uint16_t expand;
    
    if (!PyArg_ParseTuple(args, "O!O!iOO!", mbp_tpdef_ref, &input, mbp_tpdef_ref, &weights, &expand, &step_tuple, mbp_tpdef_ref, &dest)) return 0;
    if (!PyArg_ParseTuple(step_tuple, "ii", &stepr, &stepc)) return 0;
    
    // For conv operation, poolsize is the kernel size from weights
    pool_r = weights->info->rows;
    pool_c = weights->info->cols;
    
    conv_fwdcalc(input->info->data, weights->info->data, input->info->rows, input->info->cols,
                 pool_r, pool_c, expand, stepr, stepc, dest->info->data);
    
    Py_RETURN_NONE;
}

PyMethodDef convops[] = {
    {"averpool", convoppy_averpool, METH_VARARGS, 0},
    {"maxpool", convoppy_maxpool, METH_VARARGS, 0},
    {"averpool_backward", convoppy_averpool_backward, METH_VARARGS, 0},
    {"maxpool_backward", convoppy_maxpool_backward, METH_VARARGS, 0},
    {"padding", convoppy_padding, METH_VARARGS, 0},
    {"cutedges", convoppy_cutedges, METH_VARARGS, 0},
    {"divelem", convoppy_divelem, METH_VARARGS, 0},
    {"mergeelem", convoppy_mergeelem, METH_VARARGS, 0},
    {"conv", convoppy_fwdcalc, METH_VARARGS, 0},
    {0, 0, 0, 0},
};

struct PyModuleDef corepy_conv = {
    PyModuleDef_HEAD_INIT, "corepy.convop", 0, 0,
    .m_methods = convops,
};