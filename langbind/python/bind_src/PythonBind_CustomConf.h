#ifndef pythonbind_customconf
#define pythonbind_customconf

#include <Python.h>
#include "mlpCCore/CustomConf.h"

//防止不必要的链接,宏来自mlpCCore/qfix_ops/qfix_op.h

#define float_to_qfix(value) (qfix)((value) * (1 << QSHIFT))
#define qfix_to_float(qfvalue) ((float)(qfvalue) / (1 << QSHIFT))
#define qfix_to_float64(qfvalue) ((double)(qfvalue) / (1 << QSHIFT))
#define qfix_to_int(qfvalue) ((qfvalue) >> QSHIFT)

#endif