#include "convop_topy.h"
#include "mlpCCore/mlp/convop.h"
#include "arrop_topy.h"

#include <string.h>

/*
core.convops: 卷积/池化算子绑定
算子本体在 mlpCCore/mlp/convop.c, 本文件只做参数解析与必要的安全校验:
核心按"裸指针 + 尺寸"直写内存, 既不校验缓冲容量, 也不防除零/无符号回绕,
非法入参若原样下传, 后果是越界写堆或除零崩溃, 故在绑定层先行拦截。
*/

//==================== 公共校验 ====================

//核尺寸与步长非零校验: step 参与整除, poolsize 作为除数, 取 0 会触发除零
static int convop_check_pool(uint16_t rows_pool, uint16_t cols_pool, uint16_t stepr, uint16_t stepc){
    if(!rows_pool || !cols_pool || !stepr || !stepc){
        PyErr_SetString(PyExc_ValueError, "convop: pool size and step must be >= 1");
        return 0;
    }
    return 1;
}

//缓冲容量校验: 核心不做边界检查, 容量不足即越界写堆
static int convop_check_cap(const arr_pyhandle *arr, uint32_t need, const char *which){
    if(arr->size < need){
        PyErr_Format(PyExc_ValueError, "convop: %s buffer too small: need %u, got %u",
                     which, (unsigned)need, (unsigned)arr->size);
        return 0;
    }
    return 1;
}

//averpool(in: core.arrop.arr, input_size: tuple[int, int], pool_size: tuple[int, int], step_rows: int, step_cols: int, out: core.arrop.arr) -> None
PyObject *convpy_averpool(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input, *output;
    PyObject *tuple_insize, *tuple_poolsize;
    uint16_t rows_in, cols_in, rows_pool, cols_pool, stepr, stepc;
    if(!PyArg_ParseTuple(args,"O!O!O!HHO!",&arr_handle_def, &input, &PyTuple_Type, &tuple_insize, &PyTuple_Type, &tuple_poolsize, &stepr, &stepc, &arr_handle_def, &output)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;
    if(!PyArg_ParseTuple(tuple_poolsize, "HH", &rows_pool, &cols_pool)) return 0;

    if(!convop_check_pool(rows_pool, cols_pool, stepr, stepc)) return 0;
    //(rows_in - rows_pool) 为无符号运算, 核大于输入会回绕成极大值, 必须先行拦截
    if(rows_pool > rows_in || cols_pool > cols_in){
        PyErr_SetString(PyExc_ValueError, "convop: pool size must not exceed input size");
        return 0;
    }

    const uint32_t rows_out = (uint32_t)(rows_in - rows_pool) / stepr + 1;
    const uint32_t cols_out = (uint32_t)(cols_in - cols_pool) / stepc + 1;
    const uint32_t size_out = rows_out * cols_out;

    if(!convop_check_cap(input, (uint32_t)rows_in * cols_in, "in")) return 0;
    if(!convop_check_cap(output, size_out, "out")) return 0;

    //核心 conv_averpool 先累加再除核面积, 要求输出区进入时已清零(参考 .debug/testconv.c)
    memset(output->data, 0, sizeof(qfix) * size_out);

    conv_averpool(input->data, rows_in, cols_in, rows_pool, cols_pool, stepr, stepc, output->data);
    Py_RETURN_NONE;
}

//maxpool(in: core.arrop.arr, input_size: tuple[int, int], pool_size: tuple[int, int], step_rows: int, step_cols: int, out: core.arrop.arr, idx_max: core.arrop.arr) -> None
PyObject *convpy_maxpool(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input, *output, *idx_max;
    PyObject *tuple_insize, *tuple_poolsize;
    uint16_t rows_in, cols_in, rows_pool, cols_pool, stepr, stepc;
    if(!PyArg_ParseTuple(args,"O!O!O!HHO!O!",&arr_handle_def, &input, &PyTuple_Type, &tuple_insize, &PyTuple_Type, &tuple_poolsize, &stepr, &stepc, &arr_handle_def, &output, &arr_handle_def, &idx_max)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;
    if(!PyArg_ParseTuple(tuple_poolsize, "HH", &rows_pool, &cols_pool)) return 0;

    if(!convop_check_pool(rows_pool, cols_pool, stepr, stepc)) return 0;
    if(rows_pool > rows_in || cols_pool > cols_in){
        PyErr_SetString(PyExc_ValueError, "convop: pool size must not exceed input size");
        return 0;
    }

    const uint32_t rows_out = (uint32_t)(rows_in - rows_pool) / stepr + 1;
    const uint32_t cols_out = (uint32_t)(cols_in - cols_pool) / stepc + 1;
    const uint32_t size_out = rows_out * cols_out;

    if(!convop_check_cap(input, (uint32_t)rows_in * cols_in, "in")) return 0;
    if(!convop_check_cap(output, size_out, "out")) return 0;
    if(!convop_check_cap(idx_max, size_out, "idx_max")) return 0;

    //核心以 uint16_t 回写各池窗最大值在核内的线性下标, 再由绑定层转定点存入 idx_max
    uint16_t *idx_max_tmp = (uint16_t*)malloc(sizeof(uint16_t) * size_out);
    if(!idx_max_tmp) return PyErr_NoMemory();

    conv_maxpool(input->data, rows_in, cols_in, rows_pool, cols_pool, stepr, stepc, output->data, idx_max_tmp);

    for(uint32_t i = 0; i < size_out; i++){
        idx_max->data[i] = float_to_qfix(idx_max_tmp[i]);
    }
    free(idx_max_tmp);

    Py_RETURN_NONE;
}

//averpool_grad(in: core.arrop.arr, input_size: tuple[int, int], pool_size: tuple[int, int], step_rows: int, step_cols: int, out: core.arrop.arr) -> None
//注: input_size 指前向池化时的原输入尺寸; in 为按池化输出尺寸的梯度, out 为按原输入尺寸的梯度
PyObject *convpy_averpool_grad(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input, *output;
    PyObject *tuple_insize, *tuple_poolsize;
    uint16_t rows_in, cols_in, rows_pool, cols_pool, stepr, stepc;
    if(!PyArg_ParseTuple(args,"O!O!O!HHO!",&arr_handle_def, &input, &PyTuple_Type, &tuple_insize, &PyTuple_Type, &tuple_poolsize, &stepr, &stepc, &arr_handle_def, &output)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;
    if(!PyArg_ParseTuple(tuple_poolsize, "HH", &rows_pool, &cols_pool)) return 0;

    if(!convop_check_pool(rows_pool, cols_pool, stepr, stepc)) return 0;
    if(rows_pool > rows_in || cols_pool > cols_in){
        PyErr_SetString(PyExc_ValueError, "convop: pool size must not exceed input size");
        return 0;
    }

    const uint32_t rows_out = (uint32_t)(rows_in - rows_pool) / stepr + 1;
    const uint32_t cols_out = (uint32_t)(cols_in - cols_pool) / stepc + 1;

    //核心内部会先清零 gradout, 再按池窗把梯度平摊回去: in 需容纳池化输出, out 需容纳原输入
    if(!convop_check_cap(input, rows_out * cols_out, "in")) return 0;
    if(!convop_check_cap(output, (uint32_t)rows_in * cols_in, "out")) return 0;

    conv_averpool_backward(input->data, rows_in, cols_in, rows_pool, cols_pool, stepr, stepc, output->data);
    Py_RETURN_NONE;
}

//maxpool_grad(in: core.arrop.arr, idx_max: core.arrop.arr, output_size: tuple[int, int], pool_size: tuple[int, int], step_rows: int, step_cols: int, out: core.arrop.arr) -> None
PyObject *convpy_maxpool_grad(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input, *output, *idx_max;
    PyObject *tuple_insize, *tuple_poolsize;
    uint16_t rows_in, cols_in, rows_pool, cols_pool, stepr, stepc;
    if(!PyArg_ParseTuple(args,"O!O!O!O!HHO!",&arr_handle_def, &input, &arr_handle_def, &idx_max, &PyTuple_Type, &tuple_insize, &PyTuple_Type, &tuple_poolsize, &stepr, &stepc, &arr_handle_def, &output)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;
    if(!PyArg_ParseTuple(tuple_poolsize, "HH", &rows_pool, &cols_pool)) return 0;

    if(!convop_check_pool(rows_pool, cols_pool, stepr, stepc)) return 0;
    if(rows_pool > rows_in || cols_pool > cols_in){
        PyErr_SetString(PyExc_ValueError, "convop: pool size must not exceed input size");
        return 0;
    }

    const uint32_t rows_out = (uint32_t)(rows_in - rows_pool) / stepr + 1;
    const uint32_t cols_out = (uint32_t)(cols_in - cols_pool) / stepc + 1;
    const uint32_t size_out = rows_out * cols_out;

    if(!convop_check_cap(input, size_out, "in")) return 0;
    if(!convop_check_cap(idx_max, size_out, "idx_max")) return 0;
    if(!convop_check_cap(output, (uint32_t)rows_in * cols_in, "out")) return 0;

    uint16_t *idx_max_tmp = (uint16_t*)malloc(sizeof(uint16_t) * size_out);
    if(!idx_max_tmp) return PyErr_NoMemory();

    const uint32_t pool_cnt = (uint32_t)rows_pool * cols_pool;
    for(uint32_t i = 0; i < size_out; i++){
        const qfix v = idx_max->data[i];
        //核心把它当核内线性下标直接索引梯度输出, 越界值会造成堆越界写, 必须拦截
        if(v < 0 || (uint32_t)qfix_to_int(v) >= pool_cnt){
            free(idx_max_tmp);
            PyErr_SetString(PyExc_ValueError, "convop: idx_max out of range");
            return 0;
        }
        idx_max_tmp[i] = (uint16_t)qfix_to_int(v);
    }

    conv_maxpool_backward(input->data, idx_max_tmp, rows_in, cols_in, rows_pool, cols_pool, stepr, stepc, output->data);

    free(idx_max_tmp);

    Py_RETURN_NONE;
}

//padding(in: core.arrop.arr, input_size: tuple[int,int], r_extend: int, c_extend: int, out: core.arrop.arr) ->None
PyObject* convpy_padding(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input, *output;
    PyObject *tuple_insize;
    uint16_t rows_in, cols_in, r_extend, c_extend;
    if(!PyArg_ParseTuple(args, "O!O!HHO!", &arr_handle_def, &input, &PyTuple_Type, &tuple_insize, &r_extend, &c_extend, &arr_handle_def, &output)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;

    //核心按 uint16_t 计算 (rows_in + 2*r_extend), 这里跟随其位宽核对输出容量
    const uint16_t rows_out = rows_in + r_extend * 2;
    const uint16_t cols_out = cols_in + c_extend * 2;
    const uint32_t size_out = (uint32_t)rows_out * cols_out;

    if(!convop_check_cap(input, (uint32_t)rows_in * cols_in, "in")) return 0;
    if(!convop_check_cap(output, size_out, "out")) return 0;

    conv_padding(input->data, rows_in, cols_in, r_extend, c_extend, output->data);

    Py_RETURN_NONE;
}

//cutedges(in: core.arrop.arr, input_size: tuple[int,int], r_cut: int, c_cut: int, out: core.arrop.arr) ->None
PyObject* convpy_cutedge(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input, *output;
    PyObject *tuple_insize;
    uint16_t rows_in, cols_in, r_cut, c_cut;
    if(!PyArg_ParseTuple(args, "O!O!HHO!", &arr_handle_def, &input, &PyTuple_Type, &tuple_insize, &r_cut, &c_cut, &arr_handle_def, &output)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;

    //(rows_in - 2*r_cut) 为无符号运算, 裁边超过一半会回绕成极大值, 先行拦截
    if((uint32_t)r_cut * 2 > rows_in || (uint32_t)c_cut * 2 > cols_in){
        PyErr_SetString(PyExc_ValueError, "convop: cut size must not exceed half input size");
        return 0;
    }

    const uint16_t rows_out = rows_in - r_cut * 2;
    const uint16_t cols_out = cols_in - c_cut * 2;
    const uint32_t size_out = (uint32_t)rows_out * cols_out;

    if(!convop_check_cap(input, (uint32_t)rows_in * cols_in, "in")) return 0;
    if(!convop_check_cap(output, size_out, "out")) return 0;

    conv_cutedges(input->data, rows_in, cols_in, r_cut, c_cut, output->data);

    Py_RETURN_NONE;
}

//divelem(in: core.arrop.arr, in_size: tuple[int,int], pool_size: tuple[int,int], step_rows: int, step_cols: int, expand: int, outs: List[core.arrop.arr]) -> None
PyObject* convpy_divelem(PyObject *_rtime, PyObject *args){
    arr_pyhandle *input;
    PyObject *tuple_insize, *tuple_poolsize, *lst_outputs;
    uint16_t rows_in, cols_in, rows_pool, cols_pool, expand, stepr, stepc;
    if(!PyArg_ParseTuple(args, "O!O!O!HHHO!", &arr_handle_def, &input, &PyTuple_Type, &tuple_insize, &PyTuple_Type, &tuple_poolsize, &stepr, &stepc, &expand, &PyList_Type, &lst_outputs)) return 0;
    if(!PyArg_ParseTuple(tuple_insize, "HH", &rows_in, &cols_in)) return 0;
    if(!PyArg_ParseTuple(tuple_poolsize, "HH", &rows_pool, &cols_pool)) return 0;

    if(!convop_check_pool(rows_pool, cols_pool, stepr, stepc)) return 0;

    const uint16_t _scale = expand + 1;
    //分块数与核心 conv_divelem 计算式保持一致, 且须保证被减数非负(否则无符号回绕)
    const uint32_t cover_r = (uint32_t)rows_pool * _scale;
    const uint32_t cover_c = (uint32_t)cols_pool * _scale;
    if(cover_r > (uint32_t)rows_in + expand || cover_c > (uint32_t)cols_in + expand){
        PyErr_SetString(PyExc_ValueError, "convop: pool cover must not exceed input size");
        return 0;
    }

    const uint16_t arrsizer = (rows_in - rows_pool * _scale + expand) / stepr + 1;
    const uint16_t arrsizec = (cols_in - cols_pool * _scale + expand) / stepc + 1;
    const uint32_t arr_size = (uint32_t)arrsizer * arrsizec;

    if(!convop_check_cap(input, (uint32_t)rows_in * cols_in, "in")) return 0;

    if(PyList_Size(lst_outputs) < (Py_ssize_t)arr_size){
        PyErr_SetString(PyExc_ValueError, "convop: outputs list shorter than the number of blocks");
        return 0;
    }

    //每个输出块都会被写入 pool_rows*pool_cols 个元素, 逐块核对类型与容量
    const uint32_t blk_size = (uint32_t)rows_pool * cols_pool;
    for(uint32_t i = 0; i < arr_size; i++){
        PyObject *cur = PyList_GetItem(lst_outputs, i);
        if(!cur) return 0;
        if(!Py_IS_TYPE(cur, &arr_handle_def)){
            PyErr_SetString(PyExc_TypeError, "convop: outputs must be core.arrop.arr");
            return 0;
        }
        if(!convop_check_cap((arr_pyhandle*)cur, blk_size, "outputs")) return 0;
    }

    qfix **arrout_tmp = (qfix**)malloc(sizeof(qfix*) * arr_size);
    if(!arrout_tmp) return PyErr_NoMemory();

    for(uint32_t i = 0; i < arr_size; i++){
        arr_pyhandle *cur = (arr_pyhandle*)PyList_GetItem(lst_outputs, i);
        arrout_tmp[i] = cur->data;
    }

    conv_divelem(input->data, rows_in, cols_in, rows_pool, cols_pool, expand, stepr, stepc, arrout_tmp);

    free(arrout_tmp);

    Py_RETURN_NONE;
}

//mergeelem(ins: List[core.arrop.arr], out_size: tuple[int,int], pool_size: tuple[int,int], step_rows: int, step_cols: int, expand: int, out: core.arrop.arr) -> None
PyObject* convpy_mergeelem(PyObject *_rtime, PyObject *args){
    arr_pyhandle *output;
    PyObject *tuple_outsize, *tuple_poolsize, *lst_inputs;
    uint16_t rows_to, cols_to, rows_pool, cols_pool, expand, stepr, stepc;
    if(!PyArg_ParseTuple(args, "O!O!O!HHHO!", &PyList_Type, &lst_inputs, &PyTuple_Type, &tuple_outsize, &PyTuple_Type, &tuple_poolsize, &stepr, &stepc, &expand, &arr_handle_def, &output)) return 0;
    if(!PyArg_ParseTuple(tuple_outsize, "HH", &rows_to, &cols_to)) return 0;
    if(!PyArg_ParseTuple(tuple_poolsize, "HH", &rows_pool, &cols_pool)) return 0;

    if(!convop_check_pool(rows_pool, cols_pool, stepr, stepc)) return 0;

    const uint16_t _scale = expand + 1;
    const uint32_t cover_r = (uint32_t)rows_pool * _scale;
    const uint32_t cover_c = (uint32_t)cols_pool * _scale;
    if(cover_r > (uint32_t)rows_to + expand || cover_c > (uint32_t)cols_to + expand){
        PyErr_SetString(PyExc_ValueError, "convop: pool cover must not exceed output size");
        return 0;
    }

    const uint16_t arrsizer = (rows_to - rows_pool * _scale + expand) / stepr + 1;
    const uint16_t arrsizec = (cols_to - cols_pool * _scale + expand) / stepc + 1;
    const uint32_t arr_size = (uint32_t)arrsizer * arrsizec;

    if(!convop_check_cap(output, (uint32_t)rows_to * cols_to, "out")) return 0;

    if(PyList_Size(lst_inputs) < (Py_ssize_t)arr_size){
        PyErr_SetString(PyExc_ValueError, "convop: inputs list shorter than the number of blocks");
        return 0;
    }

    const uint32_t blk_size = (uint32_t)rows_pool * cols_pool;
    for(uint32_t i = 0; i < arr_size; i++){
        PyObject *cur = PyList_GetItem(lst_inputs, i);
        if(!cur) return 0;
        if(!Py_IS_TYPE(cur, &arr_handle_def)){
            PyErr_SetString(PyExc_TypeError, "convop: inputs must be core.arrop.arr");
            return 0;
        }
        if(!convop_check_cap((arr_pyhandle*)cur, blk_size, "inputs")) return 0;
    }

    const qfix **arrin_tmp = (const qfix**)malloc(sizeof(qfix*) * arr_size);
    if(!arrin_tmp) return PyErr_NoMemory();

    for(uint32_t i = 0; i < arr_size; i++){
        arr_pyhandle *cur = (arr_pyhandle*)PyList_GetItem(lst_inputs, i);
        arrin_tmp[i] = cur->data;
    }

    //核心 conv_mergeelem 是累加写输出, 要求输出区进入时已清零
    memset(output->data, 0, sizeof(qfix) * (uint32_t)rows_to * cols_to);

    conv_mergeelem(arrin_tmp, rows_to, cols_to, rows_pool, cols_pool, expand, stepr, stepc, output->data);

    free(arrin_tmp);

    Py_RETURN_NONE;
}

PyMethodDef convop_fns[] = {
    {"averpool",convpy_averpool, METH_VARARGS, 0},
    {"maxpool", convpy_maxpool, METH_VARARGS, 0},
    {"averpool_grad", convpy_averpool_grad, METH_VARARGS, 0},
    {"maxpool_grad", convpy_maxpool_grad, METH_VARARGS, 0},
    {"padding", convpy_padding, METH_VARARGS, 0},
    {"cutedges", convpy_cutedge, METH_VARARGS, 0},
    {"divelem", convpy_divelem, METH_VARARGS, 0},
    {"mergeelem", convpy_mergeelem, METH_VARARGS, 0},
    {0,0,0,0},
};

PyModuleDef convops_topy = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.convops",
    .m_doc = 0,
    .m_size = -1,
    .m_methods = convop_fns
};
