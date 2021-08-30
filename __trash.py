import numpy as np
import torch
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[10,11,12]])
a = torch.tensor(a)
b = torch.tensor(b)
print(a*b)
