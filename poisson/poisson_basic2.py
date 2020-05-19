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
f = Expression('t', degree=1, t=0.)
#f = Constant(0.)
Res = -dot(grad(u), grad(v))*dx + f*v*dx

for t in range(0,3):
    # update the value of the source term
    f.t=t
#    f.assign(Constant((t)))

    # solve the problem
    solve(Res == 0, u, bc)

    # plot solution
    c = plot(u,mode='color',title='t = '+str(t),vmin=0,vmax=0.5)
    plt.colorbar(c)
    plot(mesh,linewidth=0.3)
    plt.show()
