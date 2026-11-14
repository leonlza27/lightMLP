#include "mlp_modelop.h"
#include "methodobject.h"
#include "mlpCCore/mlp/filedump.h"
#include "moduleobject.h"
#include "pytypedefs.h"
#include "refcount.h"

PyTypeObject modeldef_handle_def = {
    .ob_base = PyVarObject_HEAD_INIT(0,0)
    .tp_name = "core.mlp.netdef",
    .tp_basicsize = sizeof(modeldef_pyhandle),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = modeldef_construct,
    .tp_dealloc = modeldef_destruct,
};

PyObject *modeldef_construct(PyTypeObject *tp, PyObject *args, PyObject *args_kw){
    modeldef_pyhandle *mdefnew = 0;

    mdefnew = (modeldef_pyhandle*)tp->tp_alloc(tp,0);
    if(!mdefnew) return 0;
    mdefnew->lyrscnt = 0;
    mdefnew->modeldef = 0;
    return (PyObject*)mdefnew;
}

void modeldef_destruct(PyObject *_this){
    modeldef_pyhandle *this = (modeldef_pyhandle*)_this;
    for(uint32_t i = 0; i < this->lyrscnt; i++){
        netLyrConf *cur = &this->modeldef[i];
        if(cur->bias) free(cur->bias);
        if(cur->weights) free(cur->weights);
    }
    if(this->modeldef) free(this->modeldef);
    this->modeldef = 0;
    this->lyrscnt = 0;
    Py_TYPE(_this)->tp_free(_this);
}

//constructnet(List[tuple(int, int, int, float)]) -> modeldef  
//输入格式[(i(<uint16_t> indim),i(<uint16_t> outdim),i(<uint8_t> actp),f(<qfix> dexa))]
PyObject *constructnet_to_pyhandle(PyObject *_rtime, PyObject *args){
    PyObject *input_lst;
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &input_lst)) return 0;

    modeldef_pyhandle *ret;
    ret = PyObject_New(modeldef_pyhandle, &modeldef_handle_def);
    if(!ret) return 0;

    uint32_t len = PyList_GET_SIZE(input_lst);
    if(!len) goto _ret_err;

    ret->lyrscnt = len;
    ret->modeldef = (netLyrConf*)malloc(sizeof(netLyrConf) * len);
    if(!ret->modeldef) goto _ret_err;

    for(uint32_t i = 0; i < len; i++){
        netLyrConf *cur = &ret->modeldef[i];
        cur->bias = 0;
        cur->weights = 0;
    }

    uint32_t outdi_last = 0;
    for(uint32_t i = 0; i < len; i++){
        netLyrConf *curly = &ret->modeldef[i];
        PyObject *cur;
        cur = PyList_GetItem(input_lst, i);
        if(!cur) goto _ret_err;
        if(!Py_IS_TYPE(cur, &PyTuple_Type)) goto _ret_err;
        //注意: "H" 只写 2 字节, 这里必须是 uint16_t(与注释里的接口约定一致),
        //用 uint32_t 会让高 2 字节保留未初始化值, 使下面的层间维度校验误判
        uint16_t indim, outdim;
        uint8_t actp;
        double dexa_orig;
        if(!PyArg_ParseTuple(cur, "HHbd", &indim, &outdim, &actp, &dexa_orig)) goto _ret_err;
        if(i && indim != outdi_last) goto _ret_err;
        //权重含 in_dim*out_dim 个元素: 原写法 sizeof(qfix*)*out_dim 在 in_dim>2 时under-allocate,
        //in_dim<2 时又超分配, 都会破坏堆
        qfix *wnew = (qfix*)malloc(sizeof(qfix) * indim * outdim);
        if (!wnew) goto _ret_err;
        curly->weights = wnew;
        qfix *bnew = (qfix*)malloc(sizeof(qfix) * outdim);
        if(!bnew) goto _ret_err;
        curly->bias = bnew;
        curly->in_dim = indim;
        curly->out_dim = outdim;
        curly->acTp = actp;
        curly->dExtra = float_to_qfix(dexa_orig);
        outdi_last = outdim;
    }

    return (PyObject*)ret;

    _ret_err:
    //上面的解析/分配失败可能没设置异常, 此时直接返回0会让CPython报 SystemError
    if(!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "modeldef: bad layer config or out of memory");
    Py_DECREF(ret);
    return 0;
};

//setwb(netdef : modeldef, List[Tuple[List,List]]) -> None
PyObject *write_model_wb(PyObject *_rtime, PyObject *args){
    PyObject *input_lst; modeldef_pyhandle *modeldef_src;
    if(!PyArg_ParseTuple(args, "O!O!", &modeldef_handle_def, &modeldef_src, &PyList_Type, &input_lst)) return 0;

    uint32_t lenit = PyList_GET_SIZE(input_lst);
    lenit = lenit < modeldef_src->lyrscnt? lenit : modeldef_src->lyrscnt;

    for(uint32_t i = 0; i < lenit; i++){
        netLyrConf *curly = &modeldef_src->modeldef[i];
        PyObject *cur;
        cur = PyList_GetItem(input_lst, i);
        if(!cur) return 0;
        if(!Py_IS_TYPE(cur, &PyTuple_Type)) return 0;
        PyObject *wlst, *blst;
        if(!PyArg_ParseTuple(cur, "O!O!", &PyList_Type, &wlst, &PyList_Type, &blst)) return 0;
        uint32_t lenWlst_it = PyList_GET_SIZE(wlst);
        uint32_t lenBlst_it = PyList_GET_SIZE(blst);
        lenWlst_it = lenWlst_it < curly->in_dim * curly->out_dim? lenWlst_it : curly->in_dim * curly->out_dim;
        lenBlst_it = lenBlst_it < curly->out_dim? lenBlst_it : curly->out_dim;
        for(uint32_t ii = 0; ii < lenWlst_it; ii++){
            PyObject *curnum = PyList_GetItem(wlst, ii);
            if(!curnum) return 0;
            if(PyLong_Check(curnum)) curly->weights[ii] = float_to_qfix(PyLong_AsLong(curnum));
            else if(PyFloat_Check(curnum)) curly->weights[ii] = float_to_qfix(PyFloat_AsDouble(curnum));
            else return 0;
        }
        for(uint32_t ii = 0; ii < lenBlst_it; ii++){
            PyObject *curnum = PyList_GetItem(blst, ii);
            if(!curnum) return 0;
            if(PyLong_Check(curnum)) curly->bias[ii] = float_to_qfix(PyLong_AsLong(curnum));
            else if(PyFloat_Check(curnum)) curly->bias[ii] = float_to_qfix(PyFloat_AsDouble(curnum));
            else return 0;
        }
    }

    Py_RETURN_NONE;
}

//dumpmodel(netdef : modeldef, path:str,dump_font:str, dump_name: str) -> None
PyObject *model_dump(PyObject *_rtime, PyObject *args){
    char *path, *font = 0, *mark = 0;
    modeldef_pyhandle *modeldef_src;
    if(!PyArg_ParseTuple(args, "O!s|ss", &modeldef_handle_def, &modeldef_src, &path, &font, &mark)) return 0;
    if(!font){
        int resu = savemodel(path, modeldef_src->lyrscnt, modeldef_src->modeldef);
        if(resu < 0) return 0;
    }else if(mark && font){
        if(strcmp(font,"csrc") == 0){
            int resu = dump_asCHeader(path, mark,modeldef_src->lyrscnt, modeldef_src->modeldef);
            if(resu < 0) return 0;
        }
    }else return 0;

    Py_RETURN_NONE;
}

//loadmodel(binpath:str) -> modeldef
PyObject *loadmodel_py_frombin(PyObject *rtime, PyObject *args){
    char *path;
    if(!PyArg_ParseTuple(args, "s", &path)) return 0;
    modeldef_pyhandle *ret;
    ret = PyObject_New(modeldef_pyhandle, &modeldef_handle_def);
    if(!ret) return 0;   

    int resu = loadmodel(path, &ret->modeldef);
    if(resu < 0){
        Py_DECREF(ret);
        return 0;
    }

    ret->lyrscnt = resu;

    return (PyObject*)ret;
}

PyMethodDef model_operations[] = {
    {"constructmodel",constructnet_to_pyhandle, METH_VARARGS, 0},
    {"set_wb",write_model_wb, METH_VARARGS, 0},
    {"dumpmodel",model_dump, METH_VARARGS, 0},
    {"loadmodel",loadmodel_py_frombin, METH_VARARGS, 0},
    {0,0,0,0},
};

PyModuleDef model_ops = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.mlp.ioops",
    .m_methods = model_operations,
    .m_size = -1,
    .m_doc = 0,
};