#ifndef pybind_mlpcalc
#define pybind_mlpcalc

#include "../PythonBind_CustomConf.h"
#include "mlpCCore/mlp/mlp.h"
#include "mlp_modelop.h"

extern PyTypeObject modeldef_handle_def;

typedef struct mlptrain_pyhandle{
    PyObject_HEAD;
    mlpTrainStatus *model;
}mlptrain_pyhandle;

typedef struct mlpexec_pyhandle{
    PyObject_HEAD;
    mlpExecStatus *model;
}mlpexec_pyhandle;

extern PyTypeObject mlptrain_pyhandle_def;
extern PyTypeObject mlpexec_pyhandle_def;
#endif