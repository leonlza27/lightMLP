import libcorepy as core
import libmbp16d as mlib
import random

print("checkpoint")

ndef = core.buildnet([(2,3,core.actp.Tanh,0),(3,1,core.actp.Tanh,0)])
ntrain = core.mlptrain(ndef)

ntrain_cap = core.mlptrain(ndef, totalgrad_cap=1)

print("checkpoint")

vecin = mlib.matrixbp(2,1)
vecexc = mlib.matrixbp(1,1)
vecgrad = mlib.matrixbp(1,1)
for i in range(2000):
    _in = [random.randint(-5,6), random.randint(-5,6)]
    excp = [int(_in[0]>0 and _in[1]>0)]
    vecin.fromlist(_in)
    vecexc.fromlist(excp)
    resu = ntrain.execute(vecin)
    mlib.msub(vecexc,resu,vecgrad)
    print("backwarding")
    ntrain.backward(vecgrad, 0.015)
    print("saving grads")
    core.savegrads(ntrain, ntrain_cap)

print("checkpoint")
corr = 0
corr_prral = 0
total_verify = 100

ntrain_cap_verify = core.mlptrain(ndef)
ntrain_cap_verify.backward(ntrain_cap, 0.015)

print("checkpoint")
for i in range(total_verify):
    _in = [random.randint(-5,6), random.randint(-5,6)]
    excp = [int(_in[0]>0 and _in[1]>0)]
    vecin.fromlist(_in)
    
    resu = ntrain.execute(vecin)
    if(int(resu.tolist()[0] > 0) == excp[0]): 
        corr+=1

    resu = ntrain_cap_verify.execute(vecin)
    if(int(resu.tolist()[0] > 0) == excp[0]):
        corr_prral+=1 

print(f"orig:{corr} cap:{corr_prral}")
