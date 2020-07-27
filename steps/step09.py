import numpy as np


# Dezeroの変数をVariableクラスとして実装
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not suported".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 関数を取得
            x, y = f.input, f.output  # 　関数の入出力を取得
            x.grad = f.backward(y.grad)  # backward　メソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)  # 1つ前の関数をリストに追加


# Functionクラスの実装
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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


# pythonの関数として使えるように
def square(x):
    return Square()(x)  # 1行でまとめて書くことが可能


def exp(x):
    return Exp()(x)


# スカラー量をndarray変換する関数　numpyの仕様で出力結果がndarray以外の型になってしまうため
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


if __name__ == "__main__":
    # x = Variable(np.array(1.0))
    # x = Variable(None)
    x = Variable(1.0)
