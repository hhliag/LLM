import math


class Value:
    def __init__(self, data, children=(), _op='', label = ''):
        self.data = data
        self._prev = set(children)
        self._backward = lambda:None
        self._op = _op
        self.label = label
        self.grad = 0
    def __repr__(self):
        return f"Value(data={self.data})"
    def __neg__(self):
        return self*(-1)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __sub__(self, other):
        return self+(-1*other)
    def __rmul__(self, other):
        return self*other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other), '**')
        def _backward():
            self.grad += other.data*(self.data**(other.data-1))*out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self*(other**-1)
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1- t**2) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad+= out.data * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def topological_order(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    topological_order(c)
                topo.append(v)
        topological_order(self)

        self.grad = 1.0
        for c in reversed(topo):
            c._backward()
    def __radd__(self, other):
        return self + other


    # def backprop(self, parent= None):
    #     if parent!= None: 
    #         self.grad *= parent.grad
    #         if parent._op=='+':
    #             self.grad*=1
    #         elif parent._op=='*':
    #             for child in parent._prev:
    #                 self.grad*= child.data
    #             self.grad/= self.data
    #         elif parent._op == 'tanh':
    #             self.grad = 1 - parent.data**2
    #     for child in self._prev:
    #         child.backprop(self)

        
