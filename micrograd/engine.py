import math
import numpy as np
import matplotlib.pyplot as plt

"""
Notes:

definition of derivative:
limit of (f(x+h) - f(x)) / h as h approaches 0
aka the slope

the chain rule: 
if var z depends on var y which depends on var x, then
the derivative of z wrt x is the product of the following 2 rates of change
dz/dx = (dz/dy) * (dy/dx)

multivariable case of chain rule:
we need to accumulate gradients, aka add them, so use +=
this is needed to prevent overwriting gradients

micrograd is roughly modeled after PyTorch
micrograd is a scalar valued engine, scalar values like 2.0
"""

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        # grad represents the derivative of the output with respect to this value
        self.grad = 0.0  # zero by default

        # internal variables used for autograd graph construction
        self._backward = lambda: None # this is a function that will perform the chain rule, by default does nothing, say for a leaf node
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"
    
    def __neg__(self):  # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __add__(self, other):  # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad  # 1.0 * out.grad
            other.grad += out.grad  # 1.0 * out.grad

        out._backward = _backward  # store this in a closure
        return out

    def __radd__(self, other): # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):  # other * self
        """A fallback, if python cannot do 2 * a, 2.__mul__(a),
        then will try a.__rmul__(2)
        """
        return self * other
    
    def __pow__(self, other):  # raise a value to some constant
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):  # self / other
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self, ), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            # derivative of e^x is e^x
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        # topological order all of the children in the graph
        # topological order is laying out the nodes so that all the edges go only one way from left to right
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


if __name__ == "__main__":
    def f(x):
        return 3*x**2 - 4*x + 5
    
    # xs = np.arange(-5, 5, 0.25)  # -5 to 5 (not including 5) in steps of 0.25
    # ys = f(xs)
    # plt.plot(xs, ys)
    # plt.show()

    # a = Value(3.0, label='a')
    # b = a + a; b.label = 'b'
    # b.backward()
    # print(a.grad, b.grad)

    a = Value(2.0)
    b = Value(4.0)
    a / b
    print(2 * a)

    c = Value(2.0)
    print(c.exp())
