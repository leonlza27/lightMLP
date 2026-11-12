#include "normal/arrop_topy.h"


//core 包本身与 core.arrop 子模块的模块定义
static PyModuleDef core_moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core",
    .m_size = -1,
    .m_doc = 0,
};

static PyModuleDef arrop_moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.arrop",
    .m_size = -1,
    .m_doc = 0,
};

PyMODINIT_FUNC PyInit_core(){
    //arr_handle_def 为静态类型对象, 加入模块前必须先 PyType_Ready
    //否则其 ob_type 为 NULL, 被 GC 遍历时会访问空指针而崩溃
    if(PyType_Ready(&arr_handle_def) < 0) return 0;

    PyObject *ob_create = PyModule_Create(&core_moduledef);
    if(!ob_create) return 0;

    PyObject *sub_arrop = PyModule_Create(&arrop_moduledef);
    if(!sub_arrop){
        Py_DECREF(ob_create);
        return 0;
    }

    PyObject *sub_arrop_v_norm = PyModule_Create(&arr_norm_topy);
    if(!sub_arrop_v_norm){
        Py_DECREF(sub_arrop);
        Py_DECREF(ob_create);
        return 0;
    }
    PyModule_AddObject(sub_arrop, "norm", sub_arrop_v_norm);
    #ifdef MULTI_PROCESS
    PyObject *sub_arrop_v_mt = PyModule_Create(&arr_mt_topy);
    if(!sub_arrop_v_mt){
        Py_DECREF(sub_arrop);
        Py_DECREF(ob_create);
        return 0;
    }
    PyModule_AddObject(sub_arrop, "mt", sub_arrop_v_mt);
    #endif

    //静态类型对象加入模块需先补引用, 避免模块析构时释放静态对象
    Py_INCREF(&arr_handle_def);
    PyModule_AddObject(sub_arrop, "arr", (PyObject*)&arr_handle_def);

    //将 core.arrop 挂到 core 上
    PyModule_AddObject(ob_create, "arrop", sub_arrop);

    return ob_create;
}
