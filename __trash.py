import torch
a = torch.arange(0.,24)
a = a.unsqueeze(-1)
b = a
print(a.shape)
print(torch.cat((a,b),dim=1))
