import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = 1 / (1 + np.exp(-Z))
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A - self.A**2
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.tanh(Z)
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - self.A**2
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = np.maximum(Z, 0)
        
        return self.A
    
    def backward(self):
    
        dAdZ = (self.A > 0).astype(float)

        return dAdZ
        
        
