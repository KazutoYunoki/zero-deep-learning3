import numpy as np


# Dezeroの変数をVariableクラスとして実装
class Variable:
    def __init__(self, data):
        self.data = data


# Functionクラスの実装
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


# 入力された値を2乗するクラスを実装
class Square(Function):
    def forward(self, x):
        return x ** 2


# exp関数の実装
class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# 数値微分の実装（中心差分近似）
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)
    print(dy)
