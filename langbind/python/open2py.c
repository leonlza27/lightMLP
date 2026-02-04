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

void netstructpy_empty(PyObject *self){
    netstruct_py *obj = (netstruct_py*)self;
    if(obj->netsrc) {
        NetLyrConf *dfree = obj->netsrc;
        uint32_t size = obj->size;
        for(uint16_t i = 0; i < size; i++){
            free(dfree[i].existedBiasData);
            free(dfree[i].existedWeightData);
        }
        free(dfree);
    }
    obj->netsrc = 0;
    obj->size = 0;
    Py_TYPE(obj)->tp_free(obj);
}

PyObject *netstructpy_swnet(PyObject *self){
    netstruct_py *obj = (netstruct_py*)self;
    NetLyrConf *netprint = obj->netsrc;
    //format: lyr_no. in out acTp dextra
    uint32_t size = obj->size;
    char *opt = (char*)malloc((size + 1) * 52);
    char curlyr[52];
    opt[0] = 0;
    sprintf(opt, "lyr_no.|   in  |  out  | acTp | dextra\n");
    for(uint32_t i = 0; i < size; i++){
        NetLyrConf *curp = &netprint[i];
        sprintf(curlyr, "%6d  %6d  %6d  %6d  %.4f\n", i, curp->existedWeightData->cols, curp->existedWeightData->rows, curp->Activetype, qfix_to_float64(curp->dataExtra));
        strcat(opt, curlyr);
    }
    return PyUnicode_FromString(opt);
}

PyObject *constructnet(PyObject *self, PyObject *args){
    PyObject *lstin;
    if(!PyArg_ParseTuple(args,"O!",&PyList_Type, &lstin)) return 0;
    size_t size = PyList_GET_SIZE(lstin);
    if(!size || size > UINT16_MAX){
        PyErr_SetString(PyExc_ValueError,"incorrect list size: 0 or beyound 65535(max value of unsigned int16)");
        return 0;
    }
    //网络验证
    for(size_t i = 0; i < size; i++){
        PyObject *lyrtuple_oncheck = PyList_GetItem(lstin, i);
        int indi, outdi, actp, extra, outdilast = 0;
        if(!PyArg_ParseTuple(lyrtuple_oncheck, "iiii", &indi, &outdi, &actp, &extra)) return 0;
        if(indi < 1 || indi > UINT16_MAX){
            PyErr_SetString(PyExc_ValueError,"incorrect in_dim(at tuple[0]) value: lower than 1 or beyound 65535(max value of unsigned int16)");
            return 0;
        }
        if(outdilast && indi != outdilast){
            PyErr_SetString(PyExc_ValueError, "input dim is not equal to the last's output and it is required");
            return 0;    
        }
        if(outdi < 1 || outdi > UINT16_MAX){
            PyErr_SetString(PyExc_ValueError,"incorrect out_dim(at tuple[1]) value: lower than 1 or beyound 65535(max value of unsigned int16)");
            return 0;
        }
    }

    //actual allocation
    netstruct_py *netret = PyObject_New(netstruct_py, &netstructpy_tpdef);
    netret->size = size;
    NetLyrConf *netret_data = (NetLyrConf*)malloc(sizeof(NetLyrConf) * size);
    for(uint16_t i = 0; i < size; i++){
        PyObject *lyrtuple = PyList_GetItem(lstin, i);
        int indi, outdi, actp, extra;
        PyArg_ParseTuple(lyrtuple, "iiii", &indi, &outdi, &actp, &extra);
        NetLyrConf *_this = &netret_data[i];
        _this->Activetype = actp;
        _this->dataExtra = extra;
        _this->existedBiasData = alloc_matrix_bp(outdi, 1);
        _this->existedWeightData = alloc_matrix_bp(outdi, indi);
    }
    netret->netsrc = netret_data;
    return (PyObject*)netret;
}

PyMODINIT_FUNC PyInit_core(){
    PyObject *module;
    if(PyType_Ready(&netstructpy_tpdef) < 0) return 0;
    module = PyModule_Create(&lightmlp_topy_root);
    if(module == 0) return 0;
    if(PyModule_AddObject(module, "netdef", &netstructpy_tpdef) < 0) goto _ret_err;

    return module;

    _ret_err:
    Py_DECREF(module);
    return 0;
}
