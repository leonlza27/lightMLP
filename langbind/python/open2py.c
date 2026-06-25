#include "open2py.h"
#include "boolobject.h"
#include "mlpCCore/mlp/filedump.h"
#include "import.h"
#include "mbpbuffer.h"
#include "object.h"
#include "pyerrors.h"
#include "pytypedefs.h"

PyTypeObject *_mbp_tpdef;

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
            netLyrConf *curlyr = &ndealloc[i];
            if(curlyr->weights) free(curlyr->weights);
            if(curlyr->bias) free(curlyr->bias);
            curlyr->weights = 0;
            curlyr->bias = 0;
        }
        free(ndealloc);
        obj->nstruct = 0;
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
        if(!PyArg_ParseTuple(cur, "HHid" , &indim, &outdim, &actp, &dexa)) return 0;
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

PyObject *dumpmodel_frompy(PyObject *_rtime, PyObject *args, PyObject *args_dict){
    char *dicargs[] = {"modelsrc", "filename", "filetype", "exposemrk", 0};
    char *filetype = 0, *exposemrk = 0;
    netdefpy *modelsrc;
    char *filename;
    int ret;
    if(!PyArg_ParseTupleAndKeywords(args, args_dict, "O!s|ss", dicargs, &netdefpy_tpdef, &modelsrc, &filename, &filetype, &exposemrk)) return 0;
    if(filetype) goto _export_othftype;

_save_as_bin:
    ret = savemodel(filename, modelsrc->lyrcnt, modelsrc->nstruct);
    goto _io_stat_check;
_export_othftype:
    if(strcmp(filetype, "bin") == 0) goto _save_as_bin;
    if(!exposemrk || exposemrk[0] == 0){
        PyErr_SetString(PyExc_ValueError, "arg \"exposemrk\" must a non-empty string");
        return 0;
    }
    //check exposemrk
    //for dump as any source code
    
    if(exposemrk[0] >= '0' && exposemrk[0] <= '9') goto _str_not_suit_for_srccode_value;

    char exposemrk_cverify = exposemrk[1];
    size_t i = 1;
    do{
        if((exposemrk_cverify == '_') || (exposemrk_cverify >= '0' && exposemrk_cverify <= '9') || (exposemrk_cverify >= 'A' && exposemrk_cverify <= 'Z') || (exposemrk_cverify >= 'a' || exposemrk_cverify <= 'z')){
            i++;
            exposemrk_cverify = exposemrk[i];
        }
        else{
        _str_not_suit_for_srccode_value:
            PyErr_SetString(PyExc_ValueError, "arg \"exposemrk\" string not suitable for naming a value of model exposed to any type of source code");
            return 0;
        }
    }while (exposemrk_cverify);

    if(strcmp(filetype, "csrc") == 0){
        ret = dump_asCHeader(filename, exposemrk, modelsrc->lyrcnt, modelsrc->nstruct);
        goto _io_stat_check;
    }

    PyErr_SetString(PyExc_ValueError, "arg \"filetype\" not supported type");
    return 0;

_io_stat_check:
    switch(ret){
        case lmlp_DUMPBINHEADER_ERR:
            PyErr_SetString(PyExc_RuntimeError, "Tried to write bin header but failed");
            return 0;
        case lmlp_FILENOTEXIST:
            PyErr_SetString(PyExc_OSError, "Destnaion file cannot open or create");
            return 0;
        case lmlp_DUMPINFOD_ERR:
            PyErr_SetString(PyExc_RuntimeError, "Cannot save lyrinfo of some layer, please check and re-try");
            return 0;
        case lmlp_LOADINFOW_ERR:
            PyErr_SetString(PyExc_RuntimeError, "Cannot save weights of some layer, please check and re-try");
            return 0;
        case lmlp_DUMPINFOB_ERR:
            PyErr_SetString(PyExc_RuntimeError, "Cannot save bias of some layer, please check and re-try");
            return 0;
        case lmlp_WRITESRCCODEVDEF_ERR:
            PyErr_SetString(PyExc_RuntimeError, "Cannot generate final 'netLyrConf' array defination in source code");
            return 0;
        case lmlp_WRITESRCCODESTR_ERR:
            PyErr_SetString(PyExc_RuntimeError, "Cannot dump str(or additional info) to source code");
            return 0;
        default:
            Py_RETURN_NONE;
    }
}

PyObject *load_frombin(PyObject *_rtime, PyObject *args){
    int flag = 0;
    netLyrConf *netnew;
    char *filepath;
    if(!PyArg_ParseTuple(args, "s", &filepath)) return 0;
    flag = loadmodel(filepath ,&netnew);
    switch (flag) {
        case lmlp_FILENOTEXIST:
            PyErr_SetString(PyExc_OSError, "Source bin file not exist");
            return 0;
        case lmlp_FILEVERIFY_FAIL:
            PyErr_SetString(PyExc_OSError, "file is not the bin format of lightmlp");
            return 0;
        case lmlp_LOADINFOD_ERR:
            PyErr_SetString(PyExc_OSError, "failed to load lyrinfo of some layer");
            return 0;
        case lmlp_LOADINFOB_ERR:
            PyErr_SetString(PyExc_OSError, "failed to load bias of some layer");
            return 0;
        case lmlp_LOADINFOW_ERR:
            PyErr_SetString(PyExc_OSError, "failed to load weights of some layer");
            return 0;
        default: break;
    }
    netdefpy *ret = PyObject_NEW(netdefpy, &netdefpy_tpdef);
    if(!ret){
        PyErr_SetString(PyExc_MemoryError, "try to malloc \"netdef\" but failed");
        return 0;
    }
    ret->lyrcnt = flag;
    ret->nstruct = netnew;
    return (PyObject*)ret;
}

PyObject *mlptrainpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict){
    netdefpy *src;
    PyObject *flag_gradcontiner = 0;
    char *kwds[] = {"modelsrc", "totalgrad_cap", 0};
    mlpTrainStatPy *ret = (mlpTrainStatPy*)tp->tp_alloc(tp, 0);
    if(!PyArg_ParseTupleAndKeywords(args, args_dict, "O!|O", kwds, &netdefpy_tpdef, &src, &flag_gradcontiner)) return 0;
    if(!flag_gradcontiner) mlptrainer_setup(src->lyrcnt, src->nstruct, &ret->statloc);
    else if((PyBool_Check(flag_gradcontiner) && flag_gradcontiner == Py_True) || (PyLong_Check(flag_gradcontiner) && PyLong_AsLong(flag_gradcontiner)))
        mlptrainer_totalgrads_cap_setup(src->lyrcnt, src->nstruct, &ret->statloc);
    else{
        PyErr_SetString(PyExc_TypeError, "arg \"totalgrad_cap\" not a bool (or int)");
        return 0;
    }
    Py_INCREF(src);
    ret->modelsrc = src;
    return (PyObject*)ret;
}

void mlptrainpy_dealloc(PyObject *self){
    mlpTrainStatPy *obj = (mlpTrainStatPy*)self;
    mlptrainer_cleanup(&obj->statloc);
    Py_DECREF(obj->modelsrc);
}

PyObject *mlptrainpy_mexecute(PyObject *self, PyObject *args){    
    mlpTrainStatPy *obj = (mlpTrainStatPy*)self;
    matrixbp_py *vecin, *ret = 0;
    if(!PyArg_ParseTuple(args, "O!|O!", _mbp_tpdef, &vecin, _mbp_tpdef, &ret)) goto _err_ret;
    uint16_t outdim = obj->modelsrc->nstruct[obj->modelsrc->lyrcnt - 1].out_dim;
    if(!ret) goto _allocate_mbppy_if_0;
    Py_INCREF(ret);
    goto _actual_exec;

_allocate_mbppy_if_0:
    ret = PyObject_NEW(matrixbp_py, _mbp_tpdef);
    ret->info = alloc_matrix_bp(outdim, 1);

_actual_exec:
    mlptrainer_execute(&obj->statloc, vecin->info->data);
    qfix *resusrc = obj->statloc.fullConnData[obj->modelsrc->lyrcnt], *dest = ret->info->data;
    for(uint16_t i = 0; i < outdim; i++) dest[i] = resusrc[i];
    return (PyObject*)ret;

_err_ret:
    return 0;
}

PyObject *mlptrainpy_mbackward(PyObject *self, PyObject *args){
    mlpTrainStatPy *obj = (mlpTrainStatPy*)self;
    matrixbp_py *grad0;
    mlpTrainStatPy *gradscap;
    double lr;
    if(!PyArg_ParseTuple(args, "Od", &grad0, &lr)) goto _err_ret;
    if(!Py_IS_TYPE(grad0, _mbp_tpdef)) goto _total_grad_backward;

    mlptrainer_backward(&obj->statloc, grad0->info->data, float_to_qfix(lr));
    goto _ret;
_total_grad_backward:
    gradscap = (mlpTrainStatPy*)grad0;
    if(!Py_IS_TYPE(grad0, &mlptrainpy_tpdef) || gradscap->statloc.fullConnData){
        PyErr_SetString(PyExc_ValueError, "arg \"grad0\" not a matrixbp with inital grad or a mlptrain object for capping grad info");
        goto _err_ret;
    }
    mlptrainer_totalgrads_backward(&obj->statloc, &gradscap->statloc, float_to_qfix(lr));

_ret:
    Py_RETURN_NONE;

_err_ret:
    return 0;
}

PyObject *mlptrainpy_mgetfinalgrads(PyObject *self, PyObject *args){
    mlpTrainStatPy *obj = (mlpTrainStatPy*)self;
    matrixbp_py *ret = 0;
    qfix *datadst, *grad_final;
    if(!PyArg_ParseTuple(args, "|O!", _mbp_tpdef, &ret)) goto _err_ret;
    uint16_t indim = obj->modelsrc->nstruct[0].in_dim;
    if(!ret) goto _allocate_mbppy_if_0;
    Py_INCREF(ret);
    goto _actual_exec;

_allocate_mbppy_if_0:
    ret = PyObject_NEW(matrixbp_py, _mbp_tpdef);
    ret->info = alloc_matrix_bp(indim, 1);

_actual_exec:
    datadst = ret->info->data;
    grad_final = obj->statloc.lyrinput_grad[0];
    for(uint16_t i = 0; i < indim; i++) datadst[i] = grad_final[i];
    return (PyObject*)ret;

_err_ret:
    return 0;

}

DLLEXPORT PyObject *mlptrainpy_mcheckgradsaver(PyObject *self, PyObject *args){
    mlpTrainStatPy *obj = (mlpTrainStatPy*)self;
    if(obj->statloc.fullConnData) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject *mlptrainpy_totalgrads_savegrads(PyObject *_rtime, PyObject *args){
    mlpTrainStatPy *from, *gradto;
    if(!PyArg_ParseTuple(args, "O!O!", &mlptrainpy_tpdef, &from, &mlptrainpy_tpdef, &gradto)) return 0;
    mlptrainer_totalgrads_savegrads(&from->statloc ,&gradto->statloc);
    Py_RETURN_NONE;
}

PyObject *mlpexecpy_new(PyTypeObject *tp, PyObject *args, PyObject *args_dict){
    netdefpy *src;
    if(!PyArg_ParseTuple(args, "O!", &netdefpy_tpdef, &src)) return 0;
    mlpExecStatPy *ret = (mlpExecStatPy*)tp->tp_alloc(tp, 0);
    mlpexec_setup(src->lyrcnt, src->nstruct, &ret->statloc);
    Py_INCREF(src);
    ret->modelsrc = src;
    return (PyObject*)ret;
}

void mlpexecpy_dealloc(PyObject *self){
    mlpExecStatPy *obj = (mlpExecStatPy*)self;
    mlpexec_cleanup(&obj->statloc);
    Py_DECREF(obj->modelsrc);
}

PyObject *mlpexecpy_mexecute(PyObject *self, PyObject *args){    
    mlpExecStatPy *obj = (mlpExecStatPy*)self;
    matrixbp_py *vecin, *ret = 0;
    if(!PyArg_ParseTuple(args, "O!|O!", _mbp_tpdef, &vecin, _mbp_tpdef, &ret)) goto _err_ret;
    uint16_t outdim = obj->modelsrc->nstruct[obj->modelsrc->lyrcnt - 1].out_dim;
    if(!ret) goto _allocate_mbppy_if_0;
    Py_INCREF(ret);
    goto _actual_exec;

_allocate_mbppy_if_0:
    ret = PyObject_NEW(matrixbp_py, _mbp_tpdef);
    ret->info = alloc_matrix_bp(outdim, 1);

_actual_exec:
    mlpexec_execute(&obj->statloc, vecin->info->data);
    qfix *resusrc = obj->statloc.fullConnData_tmp[(obj->modelsrc->lyrcnt + 1)%2], *dest = ret->info->data;
    for(uint16_t i = 0; i < outdim; i++) dest[i] = resusrc[i];
    return (PyObject*)ret;

_err_ret:
    return 0;
}

PyObject *mlpexecpy_mexecute_opcall(PyObject *self, PyObject *args, PyObject *args_dict){
    return mlpexecpy_mexecute(self, args);
}

PyMODINIT_FUNC PyInit_corepy(){
    PyObject *retmodule = 0;

    PyObject *libmbp16 = PyImport_ImportModule("mbp16dpy");
    if(!libmbp16){
        PyErr_SetString(PyExc_ImportError, "cannot get libmbp16d matrixbp buffer type");
        return 0;
    }

    _mbp_tpdef = (PyTypeObject*)PyObject_GetAttrString(libmbp16, "matrixbp");
    Py_DECREF(libmbp16);

    if(0 > PyType_Ready(&netdefpy_tpdef)) return 0;
    if(0 > PyType_Ready(&mlptrainpy_tpdef)) return 0;
    if(0 > PyType_Ready(&mlpexecpy_tpdef)) return 0;
    
    retmodule = PyModule_Create(&lmlpcore);
    if(!retmodule) return 0;

    if(PyModule_AddObject(retmodule, "netstruct", (PyObject*)&netdefpy_tpdef) < 0) goto _err_init;
    if(PyModule_AddObject(retmodule, "mlptrain", (PyObject*)&mlptrainpy_tpdef) < 0) goto _err_init;
    if(PyModule_AddObject(retmodule, "mlpexec", (PyObject*)&mlpexecpy_tpdef) < 0) goto _err_init;
    
    // constants for register activation type
    PyObject *actpenum_topy = PyModule_New("actp");
    if(!actpenum_topy) goto _err_init;

    //add enum fields
    if(PyModule_AddObject(actpenum_topy, "ReLU", PyLong_FromLong(ac_ReLU)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "ReLU6", PyLong_FromLong(ac_ReLU6)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "LeakyReLU", PyLong_FromLong(ac_LeakyReLU)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Sigmoid", PyLong_FromLong(ac_Sigmoid)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Sigmoid_hard", PyLong_FromLong(ac_Sigmoid_hard)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Tanh", PyLong_FromLong(ac_Tanh)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Tanh_hard", PyLong_FromLong(ac_Tanh_hard)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Sign", PyLong_FromLong(ac_Sign)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Pass", PyLong_FromLong(ac_pass)) < 0) goto _err_init_add_actpenum;
    
    //link the enums
    if(PyModule_AddObject(retmodule, "actp", actpenum_topy) < 0) goto _err_init_add_actpenum;
    return retmodule;

_err_init_add_actpenum:
    Py_DECREF(actpenum_topy);
_err_init:
    Py_DECREF(retmodule);
    return 0;
}

