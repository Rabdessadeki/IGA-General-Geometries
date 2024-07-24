# coding: utf-8
from sympy.abc import x,y, t, epsilon
from sympy import diff
from sympy import sin, pi, exp, log, sqrt, tanh, cosh, sinh, cos
from sympy import Symbol
from sympy.utilities import lambdify

import numpy as np
from matplotlib import pyplot as plt

dx   = lambda u: diff(u,x)
dy   = lambda u: diff(u,y)
dt   = lambda u: diff(u,t) 
grad = lambda u: (dx(u), dy(u))
Jac  = lambda u : (dx(dx(u)), dy(dy(u)), dx(dy(u)) )
laplace = lambda u : dx(dx(u) ) +dy(dy(u) )

eps = Symbol('eps')
a1  = Symbol('a1')
a2 = Symbol('a2')
b1 = Symbol('b1')
b2 = Symbol('b2')
c1 = Symbol('c1')
c2 = Symbol('c2')
sigma = Symbol('sigma')

#phi =  x*y*sin(pi*(x**2+y**2))
#f = lambdify((x,y,eps), grad(phi))

#phi = (x**2+y**2)/2 + eps*exp(-((x-c1)**2+(y-c2)**2)/(2*sigma))
#f = lambdify((x,y,eps,c1,c2,sigma), grad(phi))           

#phi = (x**2+y**2)/2 + eps*exp(-((x-c1)**2+(y-c2)**2)/(2*sigma)) + eps*exp(-((x-a1)**2+(y-a2)**2)/(2*sigma)) + eps*exp(-((x-b1)**2+(y-b2)**2)/(2*sigma))
#f = lambdify((x,y,eps,a1,a2,b1,b2,c1,c2,sigma), grad(phi))


Sol =  (1-x**2-y**2)*(2-x**2-y**2)#tanh( (sqrt(0.25**2-2.*t)-sqrt((x-0.5)**2+ (y-0.5)**2))/(sqrt(2)*c1))
print(dt(Sol),'\n\n\n', dx(Sol),'\n\n',dy(Sol) ,'\n\n', - dx(dx(Sol)) - dy(dy(Sol)) ) 

'''
psi = 5.*(x**2+y**2)

print(dx(Sol),'\n',dy(Sol) ,'\n',dt(Sol),'\n\n',dt(Sol) -0.01*dx(dx(Sol)) - 0.01*dy(dy(Sol)) + dx(psi)*dy(Sol) - dy(psi)*dx(Sol)) 

psi = 1.+5.*exp(-50*((x-0.5)**2+(y-0.5)**2 - 0.2)**2)

print('\n\n\n',dx(Sol),'\n',dy(Sol) ,'\n',dt(Sol),'\n\n',dt(Sol) -dx(dx(Sol)) - dy(dy(Sol)) + dx(psi)*dy(Sol) - dy(psi)*dx(Sol)) 


#.. Tools of CH equation
theta   = 3/2.
alpha   = 1000.
print('\n\n cahn-halliard =', dx( Sol*(1-Sol)*( (3*alpha/(2*theta*Sol*(1.-Sol)) - 6.* alpha)*dx(Sol) - dx(laplace(Sol))) ) + dy( Sol*(1-Sol)*( (3*alpha/(2*theta*Sol*(1.-Sol)) - 6.* alpha)*dy(Sol) - dy(laplace(Sol))) )  ) 

#print(grad(phi))
#print(Jac(phi))
N = 64
xs = np.linspace(0,1,N)
ys = np.linspace(0,1,N)

X = np.zeros((N,N))
Y = np.zeros((N,N))

eps = 0.005
c1  = 0.05
c2  = 0.05
sigma = 0.01

def F1(x,y):
 return 0.005*(2*0.5 - 2*x)*exp((-(-0.5 + x)**2 - (-0.5 + y)**2)/(2*0.01))/(2*0.01) + x


def F2(x,y):
 return 0.005*(2*0.5 - 2*y)*exp((-(-0.5 + x)**2 - (-0.5 + y)**2)/(2*0.01))/(2*0.01) + y

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
#        a = f(x,y,0.005)
        a = [F1(x,y), F2(x,y)]#f(x,y,0.005,0.5,0.5,0.01)
#        a = f(x,y,0.005,0.5,0.5,0.25,0.25,0.75,0.75,0.01)
        X[i,j] = a[0]
        Y[i,j] = a[1]

for k in range(N):
    plt.plot(X[:,k], Y[:,k], '-b')
    plt.plot(X[k,:], Y[k,:], '-b')

plt.show()
'''
