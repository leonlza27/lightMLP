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
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &stepr, &stepc)) return 0;

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
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &stepr, &stepc)) return 0;

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
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &stepr, &stepc)) return 0;

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
    if (!PyArg_ParseTuple(poolsize_tuple, "ii", &stepr, &stepc)) return 0;

    conv_maxpool_backward(graphin->info->data, (uint16_t*)maxscap->info->data,
                    graphin->info->rows, graphin->info->cols,
                    pool_c, pool_r, stepr, stepc, graphout->info->data);

    Py_RETURN_NONE;
}

PyObject* convoppy_padding(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0;
    uint16_t pool_c, pool_r;
    if (!PyArg_ParseTuple(args, "O!iiO!", mbp_tpdef_ref, &graphin, &pool_r, &pool_c, mbp_tpdef_ref,
                          &graphout)) return 0;

    conv_padding(graphin->info->data, graphin->info->rows, graphin->info->cols,
                    pool_r, pool_c,  graphout->info->data);
    Py_RETURN_NONE;
}

PyObject* convoppy_cutedges(PyObject* _rtime, PyObject* args){
    matrixbp_py *graphin, *graphout = 0;
    PyObject *poolsize_tuple;
    uint16_t pool_c, pool_r;
    if (!PyArg_ParseTuple(args, "O!iiO!", mbp_tpdef_ref, &graphin, &pool_c, &pool_r, mbp_tpdef_ref,
                          &graphout)) return 0;

    conv_cutedges(graphin->info->data, graphin->info->rows, graphin->info->cols,
                    pool_r, pool_c,  graphout->info->data);
    Py_RETURN_NONE;
}

PyMethodDef convops[] = {
    {"averpool", convoppy_averpool, METH_VARARGS, 0},
    {"maxpool", convoppy_maxpool, METH_VARARGS, 0},
    {"averpool_backward", convoppy_averpool_backward, METH_VARARGS, 0},
    {"maxpool_backward", convoppy_maxpool_backward, METH_VARARGS, 0},
    {"padding", convoppy_padding, METH_VARARGS, 0},
    {"cutedges", convoppy_cutedges, METH_VARARGS, 0},
    {0, 0, 0, 0},
};

struct PyModuleDef corepy_conv = {
    PyModuleDef_HEAD_INIT, "corepy.convop", 0, 0,
    .m_methods = convops,
};