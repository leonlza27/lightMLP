import libcorepy as core
import libmbp16d as mlib

ndef = core.buildnet([(1,2,0,0),(2,4,0,0)])
ntrain = core.mlptrain(ndef)

vecin = mlib.matrixbp(1,1)

print(vecin)

ntrain.execute(vecin)

vecgrad = mlib.matrixbp(4,1)
ntrain.backward(vecgrad, 0.3)
