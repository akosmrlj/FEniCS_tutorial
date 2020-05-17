from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt


# Create rectangular mesh with circular hole
N=50
L=1
R=0.2
domain = Rectangle(Point(-L/2,-L/2),Point(L/2,L/2)) - Circle(Point(0.,0.), R)
mesh=generate_mesh(domain, N)

# define elastic constants
E=1
nu=0.4
mu=E/2/(1+nu)
Lambda=E*nu/(1-nu*nu)

Delta=0.2*L # displacement of the clamped ends
d=2 #dimensionality of the system

#define vector function space, function u, and test function v
VFS = VectorFunctionSpace(mesh, 'P', 1)
u  = Function(VFS)
v  = TestFunction(VFS)


#impose clamped boundary conditions
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0],-L/2);
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0],L/2);

bc_left_X  = DirichletBC(VFS.sub(0), Constant(-Delta/2), left_boundary)
bc_right_X = DirichletBC(VFS.sub(0), Constant(+Delta/2), right_boundary)
bc_left_Y  = DirichletBC(VFS.sub(1), Constant(0.), left_boundary)
bc_right_Y = DirichletBC(VFS.sub(1), Constant(0.), right_boundary)

bc = [bc_left_X, bc_right_X, bc_left_Y, bc_right_Y]


# define strain and stress
def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return Lambda*tr(epsilon(u))*Identity(d) + 2*mu*epsilon(u)

# elastic energy
Energy = 1/2*inner(sigma(u),epsilon(u))*dx

# minimize elastic energy
Res = derivative(Energy, u, v)
solve(Res == 0, u, bc)

# calculate elastic energy
print("Energy = ",assemble(Energy))

# export displacements
u.rename("displacements","")
fileD = File("data/clamped_displacement.pvd");
fileD << u;

# calculate and export von Mises stress
FS = FunctionSpace(mesh, 'Lagrange', 1)
devStress = sigma(u) - (1./d)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = project(sqrt(3./2*inner(devStress, devStress)), FS)
von_Mises.rename("von Mises","")
fileS = File("data/clamped_vonMises_stress.pvd");
fileS << von_Mises;

# calculate and export stress component sigma_xx
sigma_xx = project(sigma(u)[0,0], FS)
sigma_xx.rename("sigma_xx","")
fileS = File("data/clamped_sigma_xx.pvd");
fileS << sigma_xx;

# calculate and export stress component sigma_yy
sigma_yy = project(sigma(u)[1,1], FS)
sigma_yy.rename("sigma_yy","")
fileS = File("data/clamped_sigma_yy.pvd");
fileS << sigma_yy;

# calculate and export stress component sigma_xy
sigma_xy = project(sigma(u)[0,1], FS)
sigma_xy.rename("sigma_xy","")
fileS = File("data/clamped_sigma_xy.pvd");
fileS << sigma_xy;
