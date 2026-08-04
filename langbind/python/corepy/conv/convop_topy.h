#ifndef LIGHTMLP_CONVOP_TOPY_H
#define LIGHTMLP_CONVOP_TOPY_H

#include <Python.h>
#include "mlpCCore/mlp/convop.h"

typedef struct mbp_topy matrixbp_py;

//module corepy.conv

//step统一为Tuple[stepr(行/y步数), stepc(列/x步数)]

//[EXPOSE]python: averpool(graphin: matrixbp, poolsize: Tuple[int, int], step: Tuple[int, int], graphout: matrixbp)
DLLEXPORT PyObject *convoppy_averpool(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: maxpool(graphin: matrixbp, poolsize: Tuple[int, int], step: Tuple[int, int], graphout: matrixbp, idx_of_maxs: matrixbp)
DLLEXPORT PyObject *convoppy_maxpool(PyObject *_rtime, PyObject *args);

//[EXPOSE]python: averpool_backward(graphin: matrixbp, poolsize: Tuple[int, int], step: Tuple[int, int], graphout: matrixbp)
DLLEXPORT PyObject *convoppy_averpool_backward(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: amaxpool_backward(graphin: matrixbp, idx_of_maxs: matrixbp, poolsize: Tuple[int, int], step: Tuple[int, int], graphout: matrixbp)
DLLEXPORT PyObject *convoppy_maxpool_backward(PyObject *_rtime, PyObject *args);

//[EXPOSE]python: padding(graphin: matrixbp, r_extend: int, c_extend: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_padding(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: cutedges(graphin: matrixbp, r_sub: int, c_sub: int, graphout: matrixbp)
DLLEXPORT PyObject *convoppy_cutedges(PyObject *_rtime, PyObject *args);

//[EXPOSE]python: divelem(input: matrixbp, poolsize: Tuple[int,int], expand :int, step: Tuple[int,int], dest: List[matrixbp])
DLLEXPORT PyObject *convoppy_divelem(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: mergeelem(input: List[matrixbp], poolsize: Tuple[int,int], expand :int, step: Tuple[int,int], dest: matrixbp)
DLLEXPORT PyObject *convoppy_mergeelem(PyObject *_rtime, PyObject *args);
//[EXPOSE]python: conv(input: matrixbp, weights: matrixbp, expand :int, step: Tuple[int,int], dest: matrixbp)
DLLEXPORT PyObject *convoppy_fwdcalc(PyObject *_rtime, PyObject *args);

extern PyMethodDef convops[];
extern PyModuleDef corepy_conv;

#endif //LIGHTMLP_CONVOP_TOPY_H