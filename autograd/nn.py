from abc import ABC, abstractmethod

class Module(ABC):
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0
    
    @abstractmethod
    def parameters(self):
        pass