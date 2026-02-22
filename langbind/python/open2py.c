#include "open2py.h"

PyObject *netdefpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict){
    netdefpy *obret = (netdefpy*)tp->tp_alloc(tp, 0);
    if(!obret) return 0;
    obret->nstruct = 0;
    return (PyObject*)obret;
}

void netdefpy_dealloc(PyObject *self){
    netdefpy *obj = (netdefpy*)self;
    if(obj->nstruct){
        netLyrConf *ndealloc = obj->nstruct;
        uint32_t lyrcnt = obj->lyrcnt;
        for(uint32_t i = 0; i < lyrcnt; i++){
            free(ndealloc[i].weights);
            free(ndealloc[i].bias);
        }
        free(ndealloc);
    }
}

PyObject* buildnet(PyObject *_rtime, PyObject *args){
    PyObject *lstin;
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &lstin)) return 0;
    size_t netsize_p = PyList_GET_SIZE(lstin);
    if(netsize_p >UINT32_MAX){
        PyErr_SetString(PyExc_RuntimeError, "Len of list is larger than max of uint32 value");
        return 0;
    }
    uint32_t netsize = netsize_p;
    //check: input=tuple(indim, outdim, actp, dexa) 
    int indim_c, outdim_c, outdim_c_last = 0, actp;
    double dexa;
    for(uint32_t i = 0; i < netsize; i++){
        PyObject *cur = PyList_GetItem(lstin, i);
        if(!PyArg_ParseTuple(cur, "iiid" , &indim_c, &outdim_c, &actp, &dexa)) return 0;
        if(indim_c < 0 || indim_c > UINT16_MAX){
            PyErr_Format(PyExc_ValueError,"lstin[%d] indim at tuple[0] requires a uint16 value", i);
            return 0;
        }
        if(outdim_c < 0 || outdim_c > UINT16_MAX){
            PyErr_Format(PyExc_ValueError,"lstin[%d] outdim at tuple[1] requires a uint16 value", i);
            return 0;
        }
        if(outdim_c_last && indim_c != outdim_c_last){
            PyErr_Format(PyExc_ValueError,"lstin[%d] indim at tuple[0] not equal to outdim of last", i);
            return 0;
        }
        outdim_c_last = outdim_c;
    }

    // actual initization
    
    netdefpy *ret = (netdefpy*)netdefpy_tpdef.tp_alloc(&netdefpy_tpdef, 0);
    netLyrConf *rdata = (netLyrConf*)malloc(sizeof(netLyrConf) * netsize);
    uint16_t indim, outdim;
    for(uint32_t i = 0; i < netsize; i++){
        PyObject *cur = PyList_GetItem(lstin, i);
        if(!PyArg_ParseTuple(cur, "HHid" , &indim_c, &outdim_c, &actp, &dexa)) return 0;
        rdata[i].in_dim = indim;
        rdata[i].out_dim = outdim;
        rdata[i].dExtra = float_to_qfix(dexa);
        rdata[i].acTp = actp;
        rdata[i].weights = (qfix*)malloc(sizeof(qfix) * indim * outdim);
        rdata[i].bias = (qfix*)malloc(sizeof(qfix) * outdim);
    }
    
    ret->lyrcnt = netsize;
    ret->nstruct = rdata;
    return (PyObject*)ret;
}

PyMODINIT_FUNC PyInit_libcorepy(){
    PyObject *retmodule = 0;
    if(0 > PyType_Ready(&netdefpy_tpdef)) return 0;
    
    retmodule = PyModule_Create(&lmlpcore);
    if(!retmodule) return 0;

    if(PyModule_AddObject(retmodule, "netstruct", (PyObject*)&netdefpy_tpdef) < 0){
        Py_DECREF(retmodule);
        return 0;
    }

    return retmodule;
}

