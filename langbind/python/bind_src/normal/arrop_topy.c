#include "arrop_topy.h"
#include "mlpCCore/arrop/arrop.h"


PyObject *arr_handle_construct(PyTypeObject *tp, PyObject *args, PyObject *args_kw);
void arr_handle_destruct(PyObject *_this);

PyObject *arr_handle_from_list(PyObject *_this, PyObject *args);
PyObject *arr_handle_to_list(PyObject *_this, PyObject *args);

DLLEXPORT PyObject* arr_norm_py_add(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_norm_py_sub(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_norm_py_scale(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_norm_py_emul(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_norm_py_mmul(PyObject *_rtime, PyObject *args);
DLLEXPORT PyObject* arr_norm_py_mtpose(PyObject *_rtime, PyObject *args);

static PySequenceMethods arr_handle_as_sequence = {
    .sq_length = arr_handle_len,
};

static PyMappingMethods arr_handle_as_mapping = {
    .mp_subscript   = arr_handle_getitem,
    .mp_ass_subscript = arr_handle_setitem,
};

PyMethodDef arr_handle_memfuncs[] = {
    {"fromlist", arr_handle_from_list, METH_VARARGS, 0},
    {"tolist", arr_handle_to_list, METH_VARARGS, 0},
    {0,0,0,0},
};

PyTypeObject arr_handle_def = {
    .ob_base = PyVarObject_HEAD_INIT(0,0)
    .tp_name = "core.arrop.arr",
    .tp_basicsize = sizeof(arr_pyhandle),
    .tp_itemsize = 0,
    .tp_new = arr_handle_construct,
    .tp_dealloc = arr_handle_destruct,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = arr_handle_memfuncs,
    .tp_as_sequence = &arr_handle_as_sequence,
    .tp_as_mapping = &arr_handle_as_mapping,
};

PyObject *arr_handle_construct(PyTypeObject *tp, PyObject *args, PyObject *args_kw){
    uint32_t size_init = 0;
    if(!PyArg_ParseTuple(args, "I", &size_init)) return 0;
    if(!size_init) return 0;

    arr_pyhandle *this_construct = 0;
    this_construct = (arr_pyhandle*)tp->tp_alloc(tp,0);
    if(!this_construct) return 0;
    
    this_construct->data = (qfix*)malloc(sizeof(qfix) * size_init);
    if(!this_construct->data){
        Py_DECREF(this_construct);
        return 0;
    }
    
    this_construct->size = size_init;

    return (PyObject*)this_construct;
}

void arr_handle_destruct(PyObject *_this){
    arr_pyhandle *obfree = (arr_pyhandle*)_this;
    free(obfree->data);
    Py_TYPE(obfree)->tp_free(_this);
}

PyObject *arr_handle_from_list(PyObject *_this, PyObject *args){
    PyObject *arr_in;
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &arr_in)) return 0;
    arr_pyhandle *ob = (arr_pyhandle*)_this;
    uint32_t len = (uint32_t)PyList_Size(arr_in);
    if(len > ob->size){
        PyErr_SetString(PyExc_ValueError, "arr: list longer than array capacity");
        return 0;
    }
    PyObject *cur;
    for(uint32_t i = 0; i < len; i++){
        cur = PyList_GetItem(arr_in, i);
        if(PyFloat_Check(cur)) ob->data[i] = float_to_qfix(PyFloat_AsDouble(cur));
        else if(PyLong_Check(cur)) ob->data[i] = float_to_qfix(PyLong_AsLong(cur));
        else{
            PyErr_SetString(PyExc_TypeError, "arr: list items must be numbers");
            return 0;
        }
    }

    Py_RETURN_NONE;
}

PyObject *arr_handle_to_list(PyObject *_this, PyObject *args){
    arr_pyhandle *ob = (arr_pyhandle*)_this;
    PyObject *lst_new = PyList_New(ob->size);
    for(uint32_t i = 0; i < ob->size; i++){
        PyList_SetItem(lst_new, i, PyFloat_FromDouble(qfix_to_float64(ob->data[i])));
    }

    return lst_new;
}

//========== 序列/映射协议实现: len() 和 [] 重载 ==========

Py_ssize_t arr_handle_len(PyObject *self){
    arr_pyhandle *ob = (arr_pyhandle*)self;
    return (Py_ssize_t)ob->size;
}

PyObject *arr_handle_getitem(PyObject *self, PyObject *key){
    arr_pyhandle *ob = (arr_pyhandle*)self;

    // 处理整数索引: obj[i]
    if(PyLong_Check(key)){
        Py_ssize_t idx = PyLong_AsSsize_t(key);
        if(idx < 0) idx += (Py_ssize_t)ob->size;   // 支持负索引
        if(idx < 0 || (uint32_t)idx >= ob->size){
            PyErr_SetString(PyExc_IndexError, "arr: index out of range");
            return 0;
        }
        return PyFloat_FromDouble(qfix_to_float64(ob->data[idx]));
    }

    // 处理切片: obj[start:stop:step]
    if(PySlice_Check(key)){
        Py_ssize_t start, stop, step, slicelen;
        if(PySlice_Unpack(key, &start, &stop, &step) < 0) return 0;
        slicelen = PySlice_AdjustIndices((Py_ssize_t)ob->size, &start, &stop, step);

        PyObject *lst = PyList_New(slicelen);
        if(!lst) return 0;
        for(Py_ssize_t i = 0; i < slicelen; i++){
            PyList_SET_ITEM(lst, i, PyFloat_FromDouble(
                qfix_to_float64(ob->data[(uint32_t)(start + i * step)])));
        }
        return lst;
    }

    PyErr_SetString(PyExc_TypeError, "arr indices must be integers or slices");
    return 0;
}

int arr_handle_setitem(PyObject *self, PyObject *key, PyObject *val){
    arr_pyhandle *ob = (arr_pyhandle*)self;

    // 删除操作不支持
    if(val == 0){
        PyErr_SetString(PyExc_TypeError, "arr does not support item deletion");
        return -1;
    }

    if(PyLong_Check(key)){
        Py_ssize_t idx = PyLong_AsSsize_t(key);
        if(idx < 0) idx += (Py_ssize_t)ob->size;   // 支持负索引
        if(idx < 0 || (uint32_t)idx >= ob->size){
            PyErr_SetString(PyExc_IndexError, "arr: index out of range");
            return -1;
        }

        if(PyFloat_Check(val)){
            ob->data[idx] = float_to_qfix(PyFloat_AS_DOUBLE(val));
            return 0;
        }
        if(PyLong_Check(val)){
            ob->data[idx] = float_to_qfix((double)PyLong_AS_LONG(val));
            return 0;
        }
        // 回退: 尝试用数字转换
        PyObject *f = PyNumber_Float(val);
        if(!f){
            PyErr_SetString(PyExc_TypeError, "arr value must be a number");
            return -1;
        }
        ob->data[idx] = float_to_qfix(PyFloat_AS_DOUBLE(f));
        Py_DECREF(f);
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "arr indices must be integers");
    return -1;
}

//========================================================

PyMethodDef arr_norm_funcs[] = {
    {"vadd", arr_norm_py_add, METH_VARARGS, 0},
    {"vsub", arr_norm_py_sub, METH_VARARGS, 0},
    {"vscale", arr_norm_py_scale, METH_VARARGS, 0},
    {"vemul", arr_norm_py_emul, METH_VARARGS, 0},
    {"mmul", arr_norm_py_mmul, METH_VARARGS, 0},
    {"mtpose", arr_norm_py_mtpose, METH_VARARGS, 0},
    {0, 0, 0, 0},
};

PyModuleDef arr_norm_topy = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_methods = arr_norm_funcs,
    .m_name = "core.arrop.norm",
    .m_size = -1,
    .m_doc = 0,
};

DLLEXPORT PyObject* arr_norm_py_add(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    if(!PyArg_ParseTuple(args, "O!O!O!", &arr_handle_def, &in1, &arr_handle_def, &in2, &arr_handle_def, &ret)) return 0;
    arr_add(in1->data, in2->data, in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_norm_py_sub(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    if(!PyArg_ParseTuple(args, "O!O!O!", &arr_handle_def, &in1, &arr_handle_def, &in2, &arr_handle_def, &ret)) return 0;
    arr_sub(in1->data, in2->data, in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_norm_py_scale(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *ret; 
    double scaler;
    if(!PyArg_ParseTuple(args, "O!dO!", &arr_handle_def, &in1, &scaler, &arr_handle_def, &ret)) return 0;
    arr_scale(in1->data, float_to_qfix(scaler), in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_norm_py_emul(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    if(!PyArg_ParseTuple(args, "O!O!O!", &arr_handle_def, &in1, &arr_handle_def, &in2, &arr_handle_def, &ret)) return 0;
    arr_emul(in1->data, in2->data, in1->size, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_norm_py_mmul(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *in2, *ret; 
    uint16_t m, k, n;
    if(!PyArg_ParseTuple(args, "O!O!HHHO!", &arr_handle_def, &in1, &arr_handle_def, &in2, &m, &k, &n, &arr_handle_def, &ret)) return 0;
    arr_mmul(in1->data, in2->data,m, k, n, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}

DLLEXPORT PyObject* arr_norm_py_mtpose(PyObject *_rtime, PyObject *args){
    arr_pyhandle *in1, *ret; 
    uint16_t m, n;
    if(!PyArg_ParseTuple(args, "O!HHO!", &arr_handle_def, &in1, &m, &n, &arr_handle_def, &ret)) return 0;
    arr_mtpose(in1->data, m, n, ret->data);

    Py_INCREF(ret);
    return (PyObject*)ret;
}