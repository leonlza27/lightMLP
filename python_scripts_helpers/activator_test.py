import numpy as np
import random
import json

# 将输出至目标json中测试数据
optdata = {
    "RawData" : [],
    "VectorSize" : 100,
    "allowedEpsion" : 0.001,
    "extra_data" : {
        "alpha_of_LeakyReLU" : 0.1,
    },
    "excepteds" : {
        "ReLU" : [],
        "ReLU6" : [],
        "LeakyReLU" : [],
        "Sigmoid_Hard" : [],
        "Tanh_Hard" : [],
        "Sigmoid" : [],
        "Tanh" : [],
        "Sign" : [],
        "Softmax" : []
    }
}

def float_to_q1616(va : float) -> int:
    return int(va * (1 << 16))

def q1616_to_float(va : int) -> float:
    return va / (1 << 16)

def relu(input):
    """ReLU activation function"""
    return np.maximum(0, input)

def relu6(input):
    """ReLU6 activation function"""
    return np.minimum(6, np.maximum(0, input))

def leaky_relu(input, alpha=0.1):
    """LeakyReLU activation function"""
    return np.where(input > 0, input, input * alpha)

def sigmoid_hard(input):
    """Hard Sigmoid activation function"""
    return np.minimum(1, np.maximum(0, 0.5 + 0.2 * input))

def tanh_hard(input):
    """Hard Tanh activation function"""
    return np.minimum(1, np.maximum(-1, input))

def sigmoid(input):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-input))

def tanh(input):
    """Tanh activation function"""
    return np.tanh(input)

def sign(input):
    """Sign activation function"""
    return (input > 0).astype(float)

def softmax(input):
    """Softmax activation function"""
    exp_input = np.exp(input - np.max(input, axis=-1, keepdims=True))
    return exp_input / np.sum(exp_input, axis=-1, keepdims=True)

if __name__ == "__main__":
    # 生成测试数据

    vectorSize = 20
    test_input = np.array([random.uniform(-8,8) for i in range(vectorSize)])

    optdata["VectorSize"] = vectorSize
    
    optdata["RawData"]= np.vectorize(float_to_q1616)(test_input).tolist()
    optdata["excepteds"]["ReLU"] = np.vectorize(float_to_q1616)(relu(test_input)).tolist()
    optdata["excepteds"]["ReLU6"] = np.vectorize(float_to_q1616)(relu6(test_input)).tolist()
    optdata["excepteds"]["LeakyReLU"] = np.vectorize(float_to_q1616)(leaky_relu(test_input, alpha=0.1)).tolist()
    optdata["excepteds"]["Sigmoid_Hard"] = np.vectorize(float_to_q1616)(sigmoid_hard(test_input)).tolist()
    optdata["excepteds"]["Tanh_Hard"] = np.vectorize(float_to_q1616)(tanh_hard(test_input)).tolist()
    optdata["excepteds"]["Sigmoid"] = np.vectorize(float_to_q1616)(sigmoid(test_input)).tolist()
    optdata["excepteds"]["Tanh"] = np.vectorize(float_to_q1616)(tanh(test_input)).tolist()
    optdata["excepteds"]["Sign"] = np.vectorize(float_to_q1616)(sign(test_input)).tolist()
    optdata["excepteds"]["Softmax"] = np.vectorize(float_to_q1616)(softmax(test_input)).tolist()
    
    with open("activator_test.json", "w") as f:
        json.dump(optdata, f, indent=4)
        print("activator_test.json has been generated.")