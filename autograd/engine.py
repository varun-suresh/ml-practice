from collections import deque
import math

class Value:
    def __init__(self, data, _children=(), _op='') -> None:
        self.data = data
        self.grad = 0
        # Internal variables for the value class
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
     
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other),'*')

        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(pow(self.data, other.data), (self, other),'pow')

        def _backward():
            self.grad += out.grad * other.data * pow(self.data, other.data-1)
            other.grad += out.grad * out.data * math.log(other.data)
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self),'relu')

        def _backward(self):
            self.grad += out.grad if out.data > 0 else 0
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Value( 1 / (1 + pow(math.exp, -self.data)), (self),'sigmoid')
    
        def _backward(self):
            self.grad += out.grad * out.data * (1 - out.data)

        out._backward = _backward
        return out

    def reverse_topological_sort(self):
        topo = deque()
        visited = set()
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                topo.appendleft(node)
                 
        dfs(self)
        return topo

    def backward(self):
        """
        Run topological sort, reverse it. Calculate the gradients
        """
        self.grad = 1
        topo = self.reverse_topological_sort()
        for child in topo:
            child._backward()

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return  self * other
    
    def __truediv__(self, other):
        return self * pow(other, -1)
    
    def __rtruediv__(self, other):
        return other * pow(self, -1)
    
if __name__ == "__main__":
    a = Value(2)
    b = Value(3)
    c = Value(8)
    d = Value(2)
    f = a + b
    g = f * c
    h = pow(g,d)

    h.backward()
    print(f.grad)