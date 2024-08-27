import numpy as np
import matplotlib.pyplot as plt

#real data function
def g(x):
    return (0.5 + np.tan(np.pi * x / 2))**2

#relative error c=0.05
c=0.05

#number of mesh points N and mesh x
N=10

x=np.linspace(0,2/np.pi,N)
print('x:')
print(x)
#sigma_eta=1,eta~N(0,1),\bar{eta}=0
sigma_eta=1

#generate eta
eta=np.random.normal(0,sigma_eta,N)

#generate y=g(1+c*eta)
y=g(x)*(1+c*eta)
print('y:')
print(y)

#variance,sigma_i=c*y_i
sigma=c*y
print('sigma:')
print(sigma)

plt.plot(x,g(x))
plt.show()