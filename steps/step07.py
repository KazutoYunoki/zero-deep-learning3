import numpy as np


# Dezeroの変数をVariableクラスとして実装
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. 関数を取得
        if f is not None:
            x = f.input  # 2. 関数の入力を取得
            x.grad = f.backward(self.grad)  # 3. 関数のbackwardメソッドを呼ぶ
            x.backward()  # 自分より1つ前の変数のbackwardメソッドを呼ぶ（再帰）


# Functionクラスの実装
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 出力変数に海の親を覚えさせる
        self.input = input  # 入力された変数を覚える
        self.output = output  # 出力も覚える
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


# 入力された値を2乗するクラスを実装
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


# exp関数の実装
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


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
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 逆伝播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
