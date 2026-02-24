import sys
sys.path.append("..")
import random
import libcore as mlplib

ndef = mlplib.constructnet([(2,5,2,int(0.2 * (1<<16))),(5,1,6,0)])

ntrain = mlplib.mlptrainer(ndef)

print(ndef)

ocap = mlplib.matrixbp16d.matrixbp(1,1)
rcap = mlplib.matrixbp16d.matrixbp(1,1)
dcap = mlplib.matrixbp16d.matrixbp(1,1)

for i in range(2000):
    inputs = [random.randint(-7,6), random.randint(-7,6)]
    rcap.fromlist([-10 if inputs[0] < 0 or inputs[1] < 0 else 10])
    icap = mlplib.matrixbp16d.matrixbp(2,1)
    icap.fromlist(inputs)
    ntrain.execute(icap, ocap)
    mlplib.matrixbp16d.msub(rcap, ocap, dcap)
    ntrain.backward(dcap,-0.3)

corr = 0

for i in range(200):
    inputs = [random.randint(-7,6), random.randint(-7,6)]
    rcap.fromlist([-1 if inputs[0] < 0 or inputs[1] < 0 else 1])
    icap = mlplib.matrixbp16d.matrixbp(2,1)
    icap.fromlist(inputs)
    ntrain.execute(icap, ocap)
    res = (ocap.tolist())[0]
    if(((res > 0 or res == 0) and not(inputs[0] < 0 or inputs[1] < 0)) or (res < 0 and (inputs[0] < 0 or inputs[1] < 0))):
        corr += 1

print(corr/200)