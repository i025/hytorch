import numpy as np
import weakref
import contextlib


class Config:              # 有的时候不需要反向传播，只需要正向传递，这时候修改配置configuration
    enable_backprop = True


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)             # 要允许后面as_variable函数允许传入int和float类型，书上没提？或者改为as_array?
        self.data = data  # data的类型是ndarray，属性有shape,nidm,size,dtype,itemsize等
        self.name = name  # 命名变量
        self.grad = None
        self.creator = None
        self.generation = 0

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n',
                                   '\n' + ' ' * 9)  # ndarray变成字符串，自带换行：把换行替换成换行加 9 空格，让数组的每一行（除首行）向右缩进，variable是9个字母
        return 'variable(' + p + ')'

    def __len__(self):  # 对ndarray使用len，会返回第一个维度的元素数量
        return len(self.data)

    @property  # shape方法可以作为实例变量被访问，即用x.shape替代x.shape()
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):  # 维度
        return self.data.ndim

    @property
    def size(self):  # 元素数
        return self.data.size

    @property
    def dtype(self):  # 数据类型
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    '''类内定义实现运算符重载
    def __mul__(self,other):
        return mul(self,other)
    def __rmul__......
    '''

    def cleargrad(self):
        self.grad = None

    def backward1(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

    def backward(self, retain_grad=False):  # 不保留不必要的grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]   # output是weak连接，要用（）
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = gx + x.grad  # 不能用+=

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y是weakref

    __array_priority__ = 200  # 解决左项为ndarray的问题，Variable的优先级高于ndarray，此时调用rmul或radd


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]  # Variable和ndarray一起使用

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 需要反向传播：设置“辈分”
            self.generation = max([x.generation for x in inputs])  # generation要在set_creator前面
            for output in outputs:  # 设置“连接”
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0+x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)   # 实现Variable和int，float混用的问题
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0*x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1*gy, x0*gy


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)   # 式子中只有一个量，那么一定是Variable


class Sub(Function):
    def forward(self, x0, x1):
        y = x0-x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):         # 如2.0-x，x->self(x0),2.0->other(x1)
    x1 = as_array(x1)
    return Sub()(x1, x0)  # 非对称的，不回退，想支持 scalar - var 就必须自己写 __rsub__。


class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c*x**(c-1)*gy
        return gx


def pow(x, c):
    return Pow(c)(x)


@contextlib.contextmanager       # 包装成上下文管理器
def using_config(name, value):
    old_value = getattr(Config, name)  # 把 Config 类（或对象）里当前 name 对应的值保存起来，以便以后恢复。
    setattr(Config, name, value)    # 把同一属性改成新值 value，进入「临时模式」
    try:
        yield                       # 交出执行权，with 语句体就在这行开始执行。
    finally:
        setattr(Config, name, old_value)  # 无论 with 体内是否抛异常，都保证把属性恢复成旧值。


def no_grad():
    return using_config('enable_backprop', False)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def setup_variable():
    Variable.__radd__ = add
    Variable.__add__ = add  # 类外赋值实现运算符重载,radd解决int或float在运算符左边的问题，即在self位置上
    Variable.__rmul__ = mul
    Variable.__mul__ = mul  # 类外赋值实现运算符重载,rmul解决int或float在运算符左边的问题，即在self位置上
    Variable.__neg__ = neg
    Variable.__rsub__ = rsub
    Variable.__sub__ = sub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
