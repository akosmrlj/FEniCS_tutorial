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
f = Constant(0.)
#f = Expression('t', degree=1, t=0.)
Res = dot(grad(u), grad(v))*dx - f*v*dx

#impose Dirichlet boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.), boundary)

for t in range(0,3):
    f.assign(Constant((t)))
    #f.t=t
    # solve the problem
    solve(Res == 0, u, bc)

    # plot solution
    c = plot(u,mode='color',title='time = '+str(t),vmin=0,vmax=0.5)
    plt.colorbar(c)
    plot(mesh,linewidth=0.3)
    plt.show()
