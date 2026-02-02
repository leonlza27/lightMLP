#include "open2py.h"

PyObject *netstructpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict){
    netstruct_py *obnew = (netstruct_py*)tp->tp_alloc(tp, 0);
    if(!obnew) return 0;
    obnew->netsrc = 0;
    obnew->size = 0;
    return (PyObject*)obnew;
}

int netstructpy_init(PyObject *self, PyObject *args, PyObject *args_dict){
    netstruct_py *obj = (netstruct_py*)self;
    return 0;
}

void netstructpy_free(PyObject *self){
    netstruct_py *obj = (netstruct_py*)self;
    if(obj->netsrc) free(obj->netsrc);
    
}

PyObject *constructnet(PyObject *self, PyObject *args){
    PyObject *lstin;
    if(!PyArg_ParseTuple(args,"O!",&PyList_Type, &lstin)) return 0;
    size_t size = PyList_GET_SIZE(lstin);
    if(!size || size > UINT16_MAX){
        PyErr_SetString(PyExc_ValueError,"incorrect list size: 0 or beyound 65535(max value of unsigned int16)");
        return 0;
    }
    for(size_t i = 0; i < size; i++){
        PyObject *lyrtuple_oncheck = PyList_GetItem(lstin, i);
        int indi, outdi, actp, extra;
        if(!PyArg_ParseTuple(lyrtuple_oncheck, "iiii", &indi, &outdi, &actp, &extra)) return 0;
        if(indi < 1 || indi > UINT16_MAX){
            PyErr_SetString(PyExc_ValueError,"incorrect in_dim(at tuple[0]) value: lower than 1 or beyound 65535(max value of unsigned int16)");
            return 0;
        }
        if(outdi < 1 || outdi > UINT16_MAX){
            PyErr_SetString(PyExc_ValueError,"incorrect out_dim(at tuple[1]) value: lower than 1 or beyound 65535(max value of unsigned int16)");
            return 0;
        }
    }


    netstruct_py *netret = (netstruct_py*)netstructpy_tpdef.tp_alloc(&netstructpy_tpdef, 0);

    return (PyObject*)netret;
}
