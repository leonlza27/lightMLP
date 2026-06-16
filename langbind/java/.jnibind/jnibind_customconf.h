#ifndef _jnibind_customconf
#define _jnibind_customconf

#define jnienvcall (*env)
#define idx_javaclass(logpath) jnienvcall->FindClass(env, logpath)

#define _reinterpt(type, value) *(type*)&value

#define javaclasspath_mbp "mlib/matrixbp"

#define javaclasspath_netdef "core/netdef"
#define javaclasspath_netlyr "core/netlyr"
#define javaclasspath_mlptrain "core/mlptrain"
#define javaclasspath_mlpexec "core/mlpexec"

#endif