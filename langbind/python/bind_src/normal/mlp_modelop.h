#ifndef pybind_modelop
#define pybind_modelop
#include "../PythonBind_CustomConf.h"
#include "mlpCCore/mlp/mlp.h"

typedef struct modeldef_pyhandle{
    PyObject_HEAD;
    uint32_t lyrscnt;
    netLyrConf *modeldef;
}modeldef_pyhandle;

//在pyinclude_entry里拉起库时使用
extern PyModuleDef model_ops;
extern PyTypeObject modeldef_handle_def;

PyObject *modeldef_construct(PyTypeObject *tp, PyObject *args, PyObject *args_kw);
void modeldef_destruct(PyObject *_this);
#endif