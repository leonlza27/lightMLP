import corepy as core
import mbp16dpy as mlib
import random

ndef = core.buildnet([(2,5,core.actp.ReLU,0),(5,1,core.actp.Tanh,0)])
ntrain = core.mlptrain(ndef)

vecin = mlib.matrixbp(2,1)
vecexc = mlib.matrixbp(1,1)
vecgrad = mlib.matrixbp(1,1)
for i in range(2000):
    _in = [random.randint(-5,6), random.randint(-5,6)]
    excp = [int(_in[0]>0 and _in[1]>0)]
    vecin.fromlist(_in)
    vecexc.fromlist(excp)
    resu = ntrain.execute(vecin)
    #mlib.msub(vecexc,resu,vecgrad)
    vecgrad.fromlist([excp[0] - resu.tolist()[0]])
   
    ntrain.backward(vecgrad, 0.15)

corr = 0
total_verify = 100

for i in range(total_verify):
    _in = [random.randint(-5,6), random.randint(-5,6)]
    excp = [int(_in[0]>0 and _in[1]>0)]
    vecin.fromlist(_in)
    
    resu = ntrain.execute(vecin)
    if(int(resu.tolist()[0] > 0) == excp[0]): 
        corr+=1

print(corr)
#test exec
nexec = core.mlpexec(ndef)
nexec(vecin)

#filedump
core.savemodel(ndef,"test.bin")
core.savemodel(ndef,"testexport.c",filetype="csrc",exposemrk="mymodel")

#load
ndef1 = core.loadmodel("test.bin")
