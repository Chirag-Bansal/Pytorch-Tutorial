import torch
import numpy as np

x = torch.empty(1)
print(x)
# torch.empty() is a function that fills with unintialized data and when we print gives us weird random values

x = torch.empty(2,3,4)
print(x)

x = torch.rand(3)
print(x)

x = torch.zeros(3)
print(x)

x = torch.ones(3)
print(x)

x = torch.ones(2,2, dtype=torch.float16)
print(x.size())
print(x)
print(x.dtype)

x = torch.tensor([2.5,0.1])
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x,y)
z = x + y # or torch,add(x,y)
print(z)

y.add_(x)
print(y) #All the functions with trailing underscore does an inplace addition

x = torch.rand(2)
y = torch.rand(2)
print(x,y)
print(torch.mul(x,y))

x = torch.rand(1)
print(x.item())

x = torch.rand(4,4)
print(x)
y = x.view(16)
# The above function adds every row after each other
print(y)

y2 = x.view(-1,8)
print(y2)


a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

print(type(a), type(b))

#If they are on the cpu then both have the same memory location
a.add_(1)
print(a,b)
#Both were modified in the above code

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a +=1
print(a,b)

x = torch.ones(5, requires_grad=True)
print(x)
