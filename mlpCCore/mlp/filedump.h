#ifndef _lmlp_fdump
#define _lmlp_fdump

#define lmlpBINMARK "lightmlp_model"
//in mlp.h, for shared library object independence
struct _netlyrcnf;
typedef struct _netlyrcnf netLyrConf;
#include <stdint.h>
#if defined(_WIN32) && defined(_MSC_VER)
#define DLLEXPORT _declspec(dllexport)
#else
#define DLLEXPORT
#endif

#define lmlp_DUMPBINHEADER_ERR -1
#define lmlp_DUMPINFOW_ERR -2       //Weights
#define lmlp_DUMPINFOB_ERR -3       //Bias
#define lmlp_DUMPINFOD_ERR -4       //Description(struct lmlp_model_binlyrheaderdef)
//EOF for file closed unexcepted
#define lmlp_LOADINFOW_ERR -5       //Weights
#define lmlp_LOADINFOB_ERR -6       //Bias
#define lmlp_LOADINFOD_ERR -7       //Description(struct lmlp_model_binlyrheaderdef)
#define lmlp_FILENOTEXIST -8        // contains file not exist in read-only mode or file creation failed with write mode
#define lmlp_FILEVERIFY_FAIL -9
#define lmlp_LOADFILEINFO_ERR -10
#define lmlp_WRITESRCCODESTR_ERR -11
#define lmlp_WRITESRCCODEVDEF_ERR -12

int DLLEXPORT savemodel(const char *fname ,uint32_t lyrcnt ,const netLyrConf *modelsrc);

int DLLEXPORT loadmodel(const char *fname, netLyrConf* *dstnet);

int DLLEXPORT dump_asCHeader(const char *fname, const char *modelname ,uint32_t lyrcnt ,const netLyrConf *modelsrc);

#endif 
