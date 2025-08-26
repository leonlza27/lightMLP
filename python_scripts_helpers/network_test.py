#PyTorch网络测试用例,数据用于优化releases_for_MCU/mlp/activator.cpp内激活函数及其LUT
import torch
import torch.nn as nn

#被hook示例
class NetExampleForHook(nn.Module):
    def __init__(self):
        super(NetExampleForHook,self).__init__()
        self.fc1 = nn.Linear(10,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,16)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)

        return x

lyrInputData = {}

def hooker(name):
    def hook(module, input, output):
        lyrInputData[name] = input[0].detach().cpu().numpy()

    return hook


if __name__ == "__main__" :
    net = NetExampleForHook()

    net.fc1.register_forward_hook(hooker("fc1"))
    net.fc2.register_forward_hook(hooker("fc2"))
    net.fc3.register_forward_hook(hooker("fc3"))

    input = torch.randn(2000,10)
    opt = net(input)

    for name, data in lyrInputData.items():
        print(f"{name} > min:{data.min()}, max:{data.max()}, mean:{data.mean()}")