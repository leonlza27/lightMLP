#define PY_SSIZE_T_CLEAN
#include <python3.12/Python.h>
#include <time.h>


typedef struct _time_to_py{
    PyObject_HEAD
    struct tm date;
}DateObject;

static PyObject* Date_new(PyTypeObject *tp, PyObject *args, PyObject *kwargs){
    DateObject *datanew =(DateObject*)tp->tp_alloc(tp, 0);
    if(datanew) return (PyObject*) datanew;
    return 0;
}

static int Date_init(PyObject *self, PyObject *args, PyObject *kwargs){
    DateObject *obj = (DateObject*)self;
    return 0;
}

static void Date_dealloc(PyObject *self){
    DateObject *obj = (DateObject*)self;
    Py_TYPE(obj)->tp_free(obj);
}

static PyObject *whattime(PyObject *self, PyObject *args){
    return PyLong_FromLong(time(0));
}

static PyObject* Date_fromshot(PyObject *self, PyObject *args){
    time_t tshot;
    struct tm *_date;
    DateObject *obj = (DateObject*)self;
    if(!PyArg_ParseTuple(args, "l", &tshot)) return -1;
    _date = localtime(&tshot);
    obj->date = *_date;
    return Py_None;
}

static PyObject *Date_repr(PyObject *op){
    DateObject *obj = (DateObject*) op;
    struct tm _date = obj->date;
    return PyUnicode_FromFormat("%d/%d/%d",_date.tm_year + 1900, _date.tm_mon, _date.tm_mday);
}

static PyMethodDef Date_methods[] = {
    {"fromshot", (PyCFunction)Date_fromshot, METH_VARARGS, "convert a timeshot to local time expression"},
    {0, 0, 0, 0}
};

static PyMethodDef exampleMoudles[] = {
    {"whattime", whattime, METH_VARARGS, "get a timeshot"},
    {0, 0, 0, 0}
};

static PyTypeObject DateType = {
    .ob_base = PyVarObject_HEAD_INIT(0,0)
    .tp_name = "exampletime.Date",
    .tp_basicsize = sizeof(DateObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Date_new,
    .tp_init = Date_init,
    .tp_dealloc = Date_dealloc,
    .tp_methods = Date_methods,
    .tp_repr = Date_repr
};

static int date_setup(PyObject *m){
    if(PyType_Ready(&DateType) < 0) return -1;
    if(PyModule_AddObject(m, "Date", (PyObject*)&DateType) < 0) return -1;
    return 0;
}

static struct PyModuleDef _moudle = {
    PyModuleDef_HEAD_INIT, "exampletime", 0, 0, 
    .m_methods = exampleMoudles
};

PyMODINIT_FUNC PyInit_exampletime(){
    PyObject *m;
    if (PyType_Ready(&DateType) < 0)
        return NULL;

    m = PyModule_Create(&_moudle);
    if (m == NULL)
        return NULL;
    
    
    if (PyModule_AddObject(m, "Date", (PyObject *) &DateType) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;

}