import autograd.numpy as np
from autograd import grad

#for example, let f(x)=x1+2*x2+x3^2
def f(x):
    return x[0]+2*x[1]+x[2]**2

x = np.array([0.0, 1.0, 1.0])
print(f(x))
autodiff = grad(f)
print(autodiff(x))

