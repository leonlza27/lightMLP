#mlp训练一层backward参考
import numpy as np

tg = np.array([[3],[4]])

w = np.array([[3, 4],[4 , 5]])
b = np.array([[3],[4]])

in1 = np.array([[1],[4]])

lr = 0.1

for i in range(1000):
    res = w @ in1 + b
    dl = tg - res
    w = w + dl @ in1.T * lr
    b = b + dl * lr

print("final w:",w)
print("final b:", b)
print("resu:", w @ in1 + b)