#include "mlp/open2py.h"
#include "conv/convop_topy.h"

PyTypeObject *mbp_tpdef_ref;

extern PyTypeObject netdefpy_tpdef;
extern PyTypeObject mlptrainpy_tpdef;
extern PyTypeObject mlpexecpy_tpdef;

PyMODINIT_FUNC PyInit_corepy(){
    PyObject *retmodule = 0;

    PyObject *libmbp16 = PyImport_ImportModule("mbp16dpy");
    if(!libmbp16){
        PyErr_SetString(PyExc_ImportError, "cannot get libmbp16d matrixbp buffer type");
        return 0;
    }

    mbp_tpdef_ref = (PyTypeObject*)PyObject_GetAttrString(libmbp16, "matrixbp");
    Py_DECREF(libmbp16);

    if(0 > PyType_Ready(&netdefpy_tpdef)) return 0;
    if(0 > PyType_Ready(&mlptrainpy_tpdef)) return 0;
    if(0 > PyType_Ready(&mlpexecpy_tpdef)) return 0;

    retmodule = PyModule_Create(&lmlpcore);
    if(!retmodule) return 0;

    if(PyModule_AddObject(retmodule, "netstruct", (PyObject*)&netdefpy_tpdef) < 0) goto _err_init;
    if(PyModule_AddObject(retmodule, "mlptrain", (PyObject*)&mlptrainpy_tpdef) < 0) goto _err_init;
    if(PyModule_AddObject(retmodule, "mlpexec", (PyObject*)&mlpexecpy_tpdef) < 0) goto _err_init;

    // constants for register activation type
    PyObject *actpenum_topy = PyModule_New("corepy.actp");
    if(!actpenum_topy) goto _err_init;

    //add enum fields
    if(PyModule_AddObject(actpenum_topy, "ReLU", PyLong_FromLong(ac_ReLU)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "ReLU6", PyLong_FromLong(ac_ReLU6)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "LeakyReLU", PyLong_FromLong(ac_LeakyReLU)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Sigmoid", PyLong_FromLong(ac_Sigmoid)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Sigmoid_hard", PyLong_FromLong(ac_Sigmoid_hard)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Tanh", PyLong_FromLong(ac_Tanh)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Tanh_hard", PyLong_FromLong(ac_Tanh_hard)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Sign", PyLong_FromLong(ac_Sign)) < 0) goto _err_init_add_actpenum;
    if(PyModule_AddObject(actpenum_topy, "Pass", PyLong_FromLong(ac_pass)) < 0) goto _err_init_add_actpenum;

    //link the enums
    if(PyModule_AddObject(retmodule, "actp", actpenum_topy) < 0) goto _err_init_add_actpenum;

    PyObject *sub_convops = PyModule_Create(&corepy_conv);
    if (!sub_convops) goto _err_init_add_actpenum;
    if (PyModule_AddObject(retmodule, "convop", sub_convops) < 0) goto _err_link_convop;;

    return retmodule;
_err_link_convop:
    Py_DECREF(sub_convops);
_err_init_add_actpenum:
    Py_DECREF(actpenum_topy);
_err_init:
    Py_DECREF(retmodule);
    return 0;
}
