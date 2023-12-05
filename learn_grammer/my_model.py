import torch
import torch.nn as nn

def cross_entropy_loss(x, y):
    h, _ = x.shape
    x = torch.softmax(x, dim=-1)
    x = -torch.log(x[torch.arange(h), y])
    return x.sum() / h

B = 64
C = 100
loss_func = nn.CrossEntropyLoss()
x = torch.rand((B, C))
y = torch.randint(C, (B,), dtype=torch.long)
print(y)

print(cross_entropy_loss(x, y))
print(loss_func(x, y))