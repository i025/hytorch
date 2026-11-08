import numpy as np
import weakref
import contextlib
import hytorch
# import hytorch.cuda 会和cuda重复导入


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True     # 有的时候不需要反向传播，只需要正向传递，这时候修改配置configuration
    train = True               # 在使用dropout的情况下，需要区分训练阶段和测试阶段


@contextlib.contextmanager       # 包装成上下文管理器
def using_config(name, value):
    old_value = getattr(Config, name)  # 把 Config 类（或对象）里当前 name 对应的值保存起来，以便以后恢复。
    setattr(Config, name, value)    # 把同一属性改成新值 value，进入「临时模式」
    try:
        yield                       # 交出执行权，with 语句体就在这行开始执行。
    finally:
        setattr(Config, name, old_value)  # 无论 with 体内是否抛异常，都保证把属性恢复成旧值。


def test_mode():
    return using_config('train', False)


def no_grad():
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):      # 加入GPU，将ndarray改为array_types
                raise TypeError('{} is not supported'.format(type(data)))
                # def的function用了as_array,Function中as_variable传入的是array
        self.data = data  # data的类型是ndarray/cparray，属性有shape,nidm,size,dtype,itemsize等
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

    @property
    def T(self):
        # from hytorch.functions import transpose     ?为什么前面的import dezero不行,此时的__init__里面还没有functions
        return hytorch.functions.transpose(self)

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

    def reshape(self, *shape):   # 实现从Variable对象中调用reshape
        # 个 if 是为了让你可以像 NumPy 一样，既可以直接传多个参数，也可以传一个元组/列表，
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        # from hytorch.functions import reshape
        return hytorch.functions.reshape(self, shape)

    def sum(self, axis=None, keepdims=False):   # 实现从Variable对象中调用sum
        from hytorch.functions import sum         # ?
        return sum(self, axis, keepdims)

    def to_cpu(self):   # 实现数据从gpu传输到cpu
        if self.data is not None:
            self.data = hytorch.cuda.as_numpy(self.data)

    def to_gpu(self):   # 实现数据从cpu传输到gpu
        if self.data is not None:
            self.data = hytorch.cuda.as_cupy(self.data)

    def backward1(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

    def backward(self, retain_grad=False, create_graph=False):  # 不保留不必要的grad, 默认不要反向传播的计算图
        if self.grad is None:
            xp = hytorch.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))    # 计算高阶导数，Variable类只修改一处，将grad也变为Variable，可以构建计算图
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
            gys = [output().grad for output in f.outputs]     # output是weak连接，要用（）
            # create_backprop为True时，反向传播时Function里面才设置辈分和连接，结束后恢复False
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = gx + x.grad  # 不能用+=，gx 和 x.grad 指向同一块内存，导致 += 变成“自己加自己

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y是weakref

    __array_priority__ = 200  # 解决左项为ndarray的问题，Variable的优先级高于ndarray，此时调用rmul或radd


class Parameter(Variable):   # 实现参数的自动化管理,类Variable
    pass


def as_array(x, array_module=np):    # cpu->gpu
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]  # Variable和ndarray一起使用

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 需要反向传播：设置“辈分” ,默认为True
            self.generation = max([x.generation for x in inputs])  # generation要在set_creator前面
            for output in outputs:  # 设置“连接”
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):            # forward用的是ndarray，可以自动广播
        raise NotImplementedError()

    def backward(self, gy):           # 如果进行了广播，反向传播要手动sum_to
        raise NotImplementedError()


'''
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = np.exp(x.data)*gy
        return gx


def exp(x):
    f = Exp()
    return f(x)
'''


# =============================================================================
# 四則演算
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0+x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = hytorch.functions.sum_to(gx0, self.x0_shape)     # 如果在正向传播中进行了广播，此时应该进行广播的反向传播
            gx1 = hytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    # 实现Variable和int，float混用的问题,后者先变成array，重复array和Variable混用的问题，在Function里面变成Variable
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0*x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = x1*gy, x0*gy
        if self.x0_shape != self.x1_shape:
            gx0 = hytorch.functions.sum_to(gx0, self.x0_shape)     # 如果在正向传播中进行了广播，此时应该进行广播的反向传播
            gx1 = hytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)      # 式子中只有一个量，那么一定是Variable


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0-x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = hytorch.functions.sum_to(gx0, self.x0_shape)     # 如果在正向传播中进行了广播，此时应该进行广播的反向传播
            gx1 = hytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):         # 如2.0-x，x->self(x0),2.0->other(x1)
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)  # 非对称的，不回退，想支持 scalar - var 就必须自己写 __rsub__。


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0/x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        if self.x0_shape != self.x1_shape:
            gx0 = hytorch.functions.sum_to(gx0, self.x0_shape)     # 如果在正向传播中进行了广播，此时应该进行广播的反向传播
            gx1 = hytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, hytorch.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c*x**(c-1)*gy
        return gx


def pow(x, c):
    return Pow(c)(x)


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
    Variable.__getitem__ = hytorch.functions.get_item

    Variable.matmul = hytorch.functions.matmul
    Variable.dot = hytorch.functions.matmul
    Variable.max = hytorch.functions.max
    Variable.min = hytorch.functions.min
