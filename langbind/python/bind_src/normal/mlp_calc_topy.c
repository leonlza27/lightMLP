#include "mlp_calc_topy.h"
#include "arrop_topy.h"


//mlptrain.__init__(netdef : modeldef, ascollector: bool | int)
PyObject *mlptrain_py_init(PyTypeObject *tp, PyObject *args, PyObject *args_kw){
    modeldef_pyhandle *modeldef_src;
    PyObject *gradcap_flag = 0;
    if(!PyArg_ParseTuple(args, "O!|O",&modeldef_handle_def, &modeldef_src, &gradcap_flag)) return 0;
    if(gradcap_flag && (PyBool_Check(gradcap_flag) || PyLong_Check(gradcap_flag))) return 0;
    mlptrain_pyhandle *trainnew = (mlptrain_pyhandle*)tp->tp_alloc(tp,0);
    if(!trainnew) return 0;
    if(gradcap_flag && (PyBool_Check(gradcap_flag) && gradcap_flag == Py_True) && (PyLong_Check(gradcap_flag) && PyLong_AsLong(gradcap_flag))){
        mlptrain_collector_setup(modeldef_src->lyrscnt, modeldef_src->modeldef, trainnew->model);
    }else{
        mlptrain_setup(modeldef_src->lyrscnt, modeldef_src->modeldef, trainnew->model);
    }
    return (PyObject*)trainnew;
}

void mlptrain_py_free(PyObject *_this){
    mlptrain_pyhandle *this = (mlptrain_pyhandle*)_this;
    mlptrain_cleanup(this->model);
    Py_TYPE(_this)->tp_free(_this);
}

//mlptrain.iscollector() -> bool
PyObject *mlptrain_py_check_collector(PyObject *_this, PyObject *args){
    mlptrain_pyhandle *this = (mlptrain_pyhandle*)_this;
    if(mlptrain_checkcap(this->model)) Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

PyMethodDef mlptrain_py_memfuncs[] = {
    {"iscollector", mlptrain_py_check_collector, METH_NOARGS, 0},
    {0,0,0,0},
};

PyTypeObject mlptrain_pyhandle_def = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "core.mlp.mlptrain",
    .tp_doc = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_basicsize = sizeof(mlptrain_pyhandle),
    .tp_itemsize = 0,
    .tp_new = mlptrain_py_init,
    .tp_dealloc = mlptrain_py_free,
    .tp_methods = mlptrain_py_memfuncs,
};


//mlptexec.__init__(netdef : modeldef)
PyObject *mlpexec_py_init(PyTypeObject *tp, PyObject *args, PyObject *args_kw){
    modeldef_pyhandle *modeldef_src;
    if(!PyArg_ParseTuple(args, "O!",&modeldef_handle_def, &modeldef_src)) return 0;
    mlpexec_pyhandle *execnew = (mlpexec_pyhandle*)tp->tp_alloc(tp,0);
    if(!execnew) return 0;
    mlpexec_setup(modeldef_src->lyrscnt, modeldef_src->modeldef, execnew->model);
    return (PyObject*)execnew;
}

void mlpexec_py_free(PyObject *_this){
    mlptrain_pyhandle *this = (mlptrain_pyhandle*)_this;
    mlptrain_cleanup(this->model);
    Py_TYPE(_this)->tp_free(_this);
}

PyTypeObject mlpexec_pyhandle_def = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "core.mlp.mlpexec",
    .tp_doc = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_basicsize = sizeof(mlpexec_pyhandle),
    .tp_itemsize = 0,
    .tp_new = mlpexec_py_init,
    .tp_dealloc = mlpexec_py_free,
};


//execute(model: mlptrain | mlpexec, arrin: core.arrop.arr, arrout: core.arrop.arr<optional>) -> core.arrop.arr
PyObject *mlpxxx_pynorm_execute(PyObject* _rtime, PyObject *args){
    PyObject *model;
    arr_pyhandle *in, *out = 0;
    if(!PyArg_ParseTuple(args, "OO!|O!", &model, &arr_handle_def, &in, &arr_handle_def, &out)) return 0;
    uint32_t outlen = 0;
    if(Py_IS_TYPE(model, &mlpexec_pyhandle_def)){
        mlpexec_pyhandle *mrun = (mlpexec_pyhandle*)model;
        outlen = mrun->model->modelsrc[mrun->model->calclyrs - 1].out_dim;
        if(!out){
            out = PyObject_New(arr_pyhandle, &arr_handle_def);
            if(!out) return 0;
            out->data = (qfix*)malloc(sizeof(qfix) * outlen);
            if(!out->data){
                Py_DECREF(out);
                return 0;
            }
            out->size = outlen;
        }else{
            if(out->size < outlen) return 0;
            Py_INCREF(out);
        }
        mlpexec_execute(mrun->model, in->data);
        qfix *resu = mlpexec_resu_ptr(mrun->model);
        for(uint32_t i = 0; i < outlen; i++){
            out->data[i] = resu[i];
        }
        return (PyObject*)out;
    }else if(Py_IS_TYPE(model, &mlptrain_pyhandle_def)){
        mlptrain_pyhandle *mrun = (mlptrain_pyhandle*)model;
        if(mlptrain_checkcap(mrun->model)) return 0;
        outlen = mrun->model->modelsrc[mrun->model->calclyrs - 1].out_dim;
        if(!out){
            out = PyObject_New(arr_pyhandle, &arr_handle_def);
            if(!out) return 0;
            out->data = (qfix*)malloc(sizeof(qfix) * outlen);
            if(!out->data){
                Py_DECREF(out);
                return 0;
            }
            out->size = outlen;
        }else{
            if(out->size < outlen) return 0;
            Py_INCREF(out);
        }
        mlptrain_execute(mrun->model, in->data);
        qfix *resu = mlptrain_resu_ptr(mrun->model);
        for(uint32_t i = 0; i < outlen; i++){
            out->data[i] = resu[i];
        }
        return (PyObject*)out;
    }else return 0;
}

//backward(model: mlptrain, arrin: core.arrop.arr, lr: float) -> None
PyObject *mlptrain_pynorm_backward(PyObject* _rtime, PyObject *args){
    mlptrain_pyhandle *model;
    arr_pyhandle *in;
    double lr_ori;
    if(!PyArg_ParseTuple(args, "O!O!d", &mlptrain_pyhandle_def, &model, &arr_handle_def, &in, &lr_ori)) return 0;
    if(in->size < model->model->modelsrc[model->model->calclyrs - 1].out_dim) return 0;
    if(mlptrain_checkcap(model->model)) return 0;
    mlptrain_backward(model->model, in->data, float_to_qfix(lr_ori));
    Py_RETURN_NONE;
}

//collect_grad(src: mlptrain, dest: mlptrain<collector>, lr: float) -> None
PyObject *mlptrain_pynorm_collect(PyObject* _rtime, PyObject *args){
    mlptrain_pyhandle *src, *dest;
    double lr_ori;
    if(!PyArg_ParseTuple(args, "O!O!d", &mlptrain_pyhandle_def, &src, &mlptrain_pyhandle_def, &dest, &lr_ori)) return 0;
    if(!mlptrain_checkcap(dest->model)) return 0;
    mlptrain_collector_collect(src->model, dest->model, float_to_qfix(lr_ori));
    Py_RETURN_NONE;
}

//upgrade_from_collected(src: mlptrain<collector>, dest: mlptrain) -> None
PyObject *mlptrain_pynorm_upgrade_from_collected(PyObject* _rtime, PyObject *args){
    mlptrain_pyhandle *src, *dest;
    if(!PyArg_ParseTuple(args, "O!O!", &mlptrain_pyhandle_def, &src, &mlptrain_pyhandle_def, &dest)) return 0;
    if(mlptrain_checkcap(dest->model) || !mlptrain_checkcap(src->model)) return 0;
    mlptrain_update_from_collector(src->model, dest->model);
    Py_RETURN_NONE;
}

PyMethodDef mlp_calc_norm_ops[] = {
    {"execute", mlpxxx_pynorm_execute, METH_VARARGS, 0},
    {"backward", mlptrain_pynorm_backward, METH_VARARGS, 0},
    {"collect_grad", mlptrain_pynorm_collect, METH_VARARGS, 0},
    {"upgrade_from_collected", mlptrain_pynorm_upgrade_from_collected, METH_VARARGS, 0},
    {0,0,0,0},
};

PyModuleDef mlp_norm_topy = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.mlp.norm",
    .m_size = -1,
    .m_doc = 0,
    .m_methods = mlp_calc_norm_ops
};