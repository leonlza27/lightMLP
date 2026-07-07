#ifndef LIGHTMLP_CONVOP_TOPY_H
#define LIGHTMLP_CONVOP_TOPY_H

#include <Python.h>
#include "mlpCCore/mlp/convop.h"

typedef struct mbp_topy matrixbp_py;

//module corepy.conv

//[EXPOSE]python: averpool(graphin: matrixbp, poolsize: Tuple[int, int], step: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_averpool(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: maxpool(graphin: matrixbp, poolsize: Tuple[int, int], step: int, graphout: matrixbp, idx_of_maxs: matrixbp)
DLLEXPORT PyObject *convoppy_maxpool(PyObject *_rtime, PyObject *args);

//[EXPOSE]python: averpool_backward(graphin: matrixbp, poolsize: Tuple[int, int], step: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_averpool_backward(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: amaxpool_backward(graphin: matrixbp, idx_of_maxs: matrixbp, poolsize: Tuple[int, int], step: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_maxpool_backward(PyObject *_rtime, PyObject *args);

//[EXPOSE]python: padding(graphin: matrixbp, r_extend: int, c_extend: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_padding(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: cutedges(graphin: matrixbp, r_sub: int, c_sub: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_cutedges(PyObject *_rtime, PyObject *args);

extern PyMethodDef convops[];
extern PyModuleDef corepy_conv;

#endif //LIGHTMLP_CONVOP_TOPY_H