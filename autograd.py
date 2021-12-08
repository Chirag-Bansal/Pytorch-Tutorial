import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
print(z)

q = z.mean()
print(q)

q.backward()
print(x.grad)

y = x.detach()
print(y)

weights = torch.ones(4,requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()

    print(weights.grad)
    weights.grad.zero_()
