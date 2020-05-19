from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh
R = 1. # radius
N = 20 # mesh resolution
domain = Circle(Point(0., 0.), R)
mesh = generate_mesh(domain, N)
plot(mesh,linewidth=0.3)
plt.show()

#define function space
degreeElements = 1
FS = FunctionSpace(mesh, 'Lagrange', degreeElements)

#impose Dirichlet boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(FS, Constant(0.), boundary)

#define function u and test function v
u = Function(FS)
v = TestFunction(FS)

# weak formulation of the problem
Res = -dot(grad(u), grad(v))*dx + v*dx

# solve the problem
solve(Res == 0, u, bc)

# plot solution
c = plot(u,mode='color',title='$u$')
plt.colorbar(c)
plot(mesh,linewidth=0.3)
plt.show()

# exact solution
uExact=Expression('(1-x[0]*x[0]-x[1]*x[1])/4',degree=2)

# Compute error (L2 norm)
error_L2 = errornorm(uExact, u, 'L2')
print("Error = ",error_L2)

# evaluate function at several points
print("u(0,0) = ",u(0,0))
print("u(0.5,0.5) = ",u(0.5,0.5))
#print("u(2,0) = ",u(2,0)) #point outside domain
print("u(1,0) = ",u(1,0))
# print("u(0,1) = ",u(0,1)) #point outside the discretized domain


# plot solution
tol=1e-2;
x=np.linspace(-1+tol,1-tol,40)
points = [(x_, 0) for x_ in x]
u_line = np.array([u(point) for point in points])
plt.plot(x,u_line,'k-')
plt.plot(x,(1-x*x)/4,'r--')
plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.legend(['FEM','exact'])
plt.show()
