#include "normal/arrop_topy.h"      //arr_norm_topy / arr_mt_topy / arr_handle_def
#include "normal/mlp_modelop.h"     //model_ops / modeldef_handle_def
#include "normal/mlp_calc_topy.h"   //mlptrain_/mlpexec_ 类型对象与 mlp_norm_topy / mlp_mt_topy

/*
模块层次: 与各绑定源文件内 .m_name / tp_name 的约定一一对应
  core                      本文件 PyInit_core
  |-- arrop                 core.arrop            容器模块
  |   |-- arr               core.arrop.arr        PyTypeObject arr_handle_def
  |   |-- norm              core.arrop.norm       arr_norm_topy   单线程算子
  |   `-- mt                core.arrop.mt         arr_mt_topy     仅 -DMULTI_PROCESS
  `-- mlp                   core.mlp              容器模块
      |-- modeldef          core.mlp.netdef       PyTypeObject modeldef_handle_def
      |-- mlptrain          core.mlp.mlptrain     PyTypeObject mlptrain_pyhandle_def
      |-- mlpexec           core.mlp.mlpexec      PyTypeObject mlpexec_pyhandle_def
      |-- ioops             core.mlp.ioops        model_ops       模型定义与读写
      |-- norm              core.mlp.norm         mlp_norm_topy   单线程前向/反向
      `-- mt                core.mlp.mt           mlp_mt_topy     仅 -DMULTI_PROCESS

注: core 为扩展模块(没有 __path__), 子模块除挂到父模块属性外, 还必须登记进
    sys.modules, 否则 "import core.arrop.norm" 这类点号导入会因父模块不是包
    而抛 ModuleNotFoundError(仅挂属性时只有 from core.arrop import norm 可用)
*/

//core 包本身与两个容器子模块的模块定义
static PyModuleDef core_moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core",
    .m_doc = "lightMLP core: core.arrop(定点数组与算子) / core.mlp(网络定义/训练/推理)",
    .m_size = -1,
};

static PyModuleDef arrop_moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.arrop",
    .m_doc = "core.arrop: 定点数组 core.arrop.arr 与线性代数算子(core.arrop.norm / core.arrop.mt)",
    .m_size = -1,
};

static PyModuleDef mlp_moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "core.mlp",
    .m_doc = "core.mlp: 网络定义 modeldef, 训练 mlptrain, 推理 mlpexec 与模型读写 ioops",
    .m_size = -1,
};

//初始化失败时需从 sys.modules 撤销的子模块, 顺序与登记顺序相反(先撤子后撤父)
static const char *const submod_names[] = {
    "core.mlp.mt",
    "core.mlp.norm",
    "core.mlp.ioops",
    "core.mlp",
    "core.arrop.mt",
    "core.arrop.norm",
    "core.arrop",
};
#define SUBMOD_CNT (sizeof(submod_names) / sizeof(submod_names[0]))

//把 value 挂到 mod 的 name 属性上
//各python版本下 PyModule_AddObject 对失败时是否偷引用的细节不一致, 这里先补
//一份引用再交给它, 失败时自己收回, 保证任何版本下引用计数都平衡
static int mod_set_attr(PyObject *mod, const char *name, PyObject *value){
    Py_INCREF(value);
    if(PyModule_AddObject(mod, name, value) < 0){
        Py_DECREF(value);
        return -1;
    }
    return 0;
}

//把静态类型对象加入模块
//必须先 PyType_Ready: 否则其 ob_type 为 NULL, 被 GC 遍历时会访问空指针而崩溃;
//另需补引用: 静态对象生命周期长于模块, 不能让模块析构时把它释放掉
static int mod_add_type(PyObject *mod, const char *name, PyTypeObject *tp){
    if(PyType_Ready(tp) < 0) return -1;
    return mod_set_attr(mod, name, (PyObject*)tp);
}

//创建一个子模块, 同时登记进 sys.modules 并挂到 parent 的 attr 属性上
//成功返回该子模块(借用引用, 由 sys.modules 与 parent 属性共同持有), 失败返回 0
//失败时本步已产生的引用全部回收; sys.modules 中的残留由 submod_rollback 兜底
static PyObject *submod_attach(PyObject *parent, const char *qname, const char *attr, PyModuleDef *moddef){
    PyObject *sub = PyModule_Create(moddef);
    if(!sub) return 0;

    //登记 sys.modules, 供 import core.xxx / import core.xxx.yyy 解析(字典自行持有引用)
    if(PyDict_SetItemString(PyImport_GetModuleDict(), qname, sub) < 0){
        Py_DECREF(sub);
        return 0;
    }

    //挂到父模块属性上, 供 core.xxx.yyy 形式访问
    if(mod_set_attr(parent, attr, sub) < 0){
        PyDict_DelItemString(PyImport_GetModuleDict(), qname);
        Py_DECREF(sub);
        return 0;
    }

    //让出创建引用: 此后 sub 仅由 sys.modules 与 parent 属性持有
    Py_DECREF(sub);
    return sub;
}

//撤销 sys.modules 中已登记的子模块(仅初始化失败时调用)
//core 本身由 import 机制在 PyInit_core 成功返回后才登记, 这里不用管
//删除动作会连带释放字典持有的那份引用, 故这里不能再 Py_DECREF
static void submod_rollback(void){
    PyObject *exc_type = 0, *exc_val = 0, *exc_tb = 0;
    PyObject *modules = PyImport_GetModuleDict();
    //暂存当前异常: 字典操作内部可能设置或清除异常指示器
    PyErr_Fetch(&exc_type, &exc_val, &exc_tb);

    for(size_t i = 0; i < SUBMOD_CNT; i++){
        //键不存在时返回 NULL 且不设置异常
        if(!PyDict_GetItemString(modules, submod_names[i])) continue;
        //键刚确认存在, 删除不会失败, 此处仅作防御
        if(PyDict_DelItemString(modules, submod_names[i]) < 0) PyErr_Clear();
    }

    PyErr_Restore(exc_type, exc_val, exc_tb);
}

PyMODINIT_FUNC PyInit_core(){
    PyObject *ob_core = 0;
    PyObject *ob_arrop = 0, *ob_mlp = 0;

    ob_core = PyModule_Create(&core_moduledef);
    if(!ob_core) return 0;

    //---- core.arrop: arr(类型) / norm / mt ----
    ob_arrop = submod_attach(ob_core, "core.arrop", "arrop", &arrop_moduledef);
    if(!ob_arrop) goto _init_fail;

    if(!submod_attach(ob_arrop, "core.arrop.norm", "norm", &arr_norm_topy)) goto _init_fail;
    #ifdef MULTI_PROCESS
    if(!submod_attach(ob_arrop, "core.arrop.mt", "mt", &arr_mt_topy)) goto _init_fail;
    #endif
    if(mod_add_type(ob_arrop, "arr", &arr_handle_def) < 0) goto _init_fail;

    //---- core.mlp: modeldef/mlptrain/mlpexec(类型) 与 ioops / norm / mt ----
    ob_mlp = submod_attach(ob_core, "core.mlp", "mlp", &mlp_moduledef);
    if(!ob_mlp) goto _init_fail;

    if(!submod_attach(ob_mlp, "core.mlp.ioops", "ioops", &model_ops)) goto _init_fail;
    if(!submod_attach(ob_mlp, "core.mlp.norm", "norm", &mlp_norm_topy)) goto _init_fail;
    #ifdef MULTI_PROCESS
    if(!submod_attach(ob_mlp, "core.mlp.mt", "mt", &mlp_mt_topy)) goto _init_fail;
    #endif

    if(mod_add_type(ob_mlp, "modeldef", &modeldef_handle_def) < 0) goto _init_fail;
    if(mod_add_type(ob_mlp, "mlptrain", &mlptrain_pyhandle_def) < 0) goto _init_fail;
    if(mod_add_type(ob_mlp, "mlpexec", &mlpexec_pyhandle_def) < 0) goto _init_fail;

    return ob_core;

    _init_fail:
    //子模块已登记进 sys.modules, 需先撤下; 其余由 ob_core 析构时随父模块属性一并回收
    //ob_arrop/ob_mlp 为借用引用(sys.modules 与 ob_core 持有), 此处不可 Py_DECREF
    submod_rollback();
    Py_XDECREF(ob_core);
    return 0;
}
