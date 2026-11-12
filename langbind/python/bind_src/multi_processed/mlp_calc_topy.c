#include "../normal/mlp_calc_topy.h"
#include "../normal/arrop_topy.h"

//execute(model: mlptrain | mlpexec, arrin: core.arrop.arr, arrout: core.arrop.arr<optional>) -> core.arrop.arr
PyObject *mlpxxx_pymt_execute(PyObject* _rtime, PyObject *args){
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
        mlpexec_execute_mt(mrun->model, in->data);
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
        mlptrain_execute_mt(mrun->model, in->data);
        qfix *resu = mlptrain_resu_ptr(mrun->model);
        for(uint32_t i = 0; i < outlen; i++){
            out->data[i] = resu[i];
        }
        return (PyObject*)out;
    }else return 0;
}

//backward(model: mlptrain, arrin: core.arrop.arr, lr: float) -> None
PyObject *mlptrain_pymt_backward(PyObject* _rtime, PyObject *args){
    mlptrain_pyhandle *model;
    arr_pyhandle *in;
    double lr_ori;
    if(!PyArg_ParseTuple(args, "O!O!d", &mlptrain_pyhandle_def, &model, &arr_handle_def, &in, &lr_ori)) return 0;
    if(in->size < model->model->modelsrc[model->model->calclyrs - 1].out_dim) return 0;
    if(mlptrain_checkcap(model->model)) return 0;
    mlptrain_backward_mt(model->model, in->data, float_to_qfix(lr_ori));
    Py_RETURN_NONE;
}

//collect_grad(src: mlptrain, dest: mlptrain<collector>, lr: float) -> None
PyObject *mlptrain_pymt_collect(PyObject* _rtime, PyObject *args){
    mlptrain_pyhandle *src, *dest;
    double lr_ori;
    if(!PyArg_ParseTuple(args, "O!O!d", &mlptrain_pyhandle_def, &src, &mlptrain_pyhandle_def, &dest, &lr_ori)) return 0;
    if(!mlptrain_checkcap(dest->model)) return 0;
    mlptrain_collector_collect_mt(src->model, dest->model, float_to_qfix(lr_ori));
    Py_RETURN_NONE;
}

//upgrade_from_collected(src: mlptrain<collector>, dest: mlptrain) -> None
PyObject *mlptrain_pymt_upgrade_from_collected(PyObject* _rtime, PyObject *args){
    mlptrain_pyhandle *src, *dest;
    if(!PyArg_ParseTuple(args, "O!O!", &mlptrain_pyhandle_def, &src, &mlptrain_pyhandle_def, &dest)) return 0;
    if(mlptrain_checkcap(dest->model) || !mlptrain_checkcap(src->model)) return 0;
    mlptrain_update_from_collector_mt(src->model, dest->model);
    Py_RETURN_NONE;
}

PyMethodDef mlp_calc_mt_ops[] = {
    {"execute", mlpxxx_pymt_execute, METH_VARARGS, 0},
    {"backward", mlptrain_pymt_backward, METH_VARARGS, 0},
    {"collect_grad", mlptrain_pymt_collect, METH_VARARGS, 0},
    {"upgrade_from_collected", mlptrain_pymt_upgrade_from_collected, METH_VARARGS, 0},
    {0,0,0,0},
};

PyModuleDef mlp_mt_topy = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.mlp.mt",
    .m_size = -1,
    .m_doc = 0,
    .m_methods = mlp_calc_mt_ops
};