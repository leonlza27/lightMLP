#ifndef pybind_mlpcalc
#define pybind_mlpcalc

#include "../PythonBind_CustomConf.h"
#include "mlpCCore/mlp/mlp.h"
#include "mlp_modelop.h"

extern PyTypeObject modeldef_handle_def;

typedef struct mlptrain_pyhandle{
    PyObject_HEAD;
    PyObject *modeldef;
    mlpTrainStatus *model;
}mlptrain_pyhandle;

typedef struct mlpexec_pyhandle{
    PyObject_HEAD;
    PyObject *modeldef;
    mlpExecStatus *model;
}mlpexec_pyhandle;

extern PyTypeObject mlptrain_pyhandle_def;
extern PyTypeObject mlpexec_pyhandle_def;

extern PyModuleDef mlp_norm_topy;
extern PyModuleDef mlp_mt_topy;
#endif