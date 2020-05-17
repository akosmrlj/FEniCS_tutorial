from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh
R = 1. # radius
N = 20 # mesh density
domain = Circle(Point(0., 0.), R)
mesh = generate_mesh(domain, N)
#plot(mesh,linewidth=0.3)

#define function space, function u, and test function v
V = FunctionSpace(mesh, 'Lagrange', 1)
u = Function(V)
v = TestFunction(V)

# weak formulation of the problem
Res = dot(grad(u), grad(v))*dx - v*dx

#impose Dirichlet boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.), boundary)

# solve the problem
solve(Res == 0, u, bc)

# plot solution
c = plot(u,mode='color',title='$u$')
plt.colorbar(c)
plot(mesh,linewidth=0.3)
plt.show()

# exact solution
uExact=Expression('(1-x[0]*x[0]-x[1]*x[1])/4',degree=1)

# Compute error (L2 norm)
error_L2 = errornorm(uExact, u, 'L2')
print("Error = ",error_L2)

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
