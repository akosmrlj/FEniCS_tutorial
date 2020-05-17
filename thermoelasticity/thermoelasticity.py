from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt


# create square mesh
N=30
L=1
domain = Rectangle(Point(0.,0.),Point(L,L))
mesh=generate_mesh(domain, N)

d=2

# elastic constants
E=1
nu=0.4
mu=E/2/(1+nu)
Lambda=E*nu/(1-nu*nu)

# thermal expansion coefficient
alpha = 0.1
# material conductivity
k = 1

# temperatures at the boundaries
T0 = 0
T1 = 1

#degree of finite elements
degreeElements = 1


#define left, right, top, bottom boundaries
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0],0);
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0],L);
def top_boundary(x, on_boundary):
    return on_boundary and near(x[1],L);
def bottom_boundary(x, on_boundary):
    return on_boundary and near(x[1],0);


#define function space, temperature function T, and test function dT
FS = FunctionSpace(mesh, 'Lagrange', degreeElements)
T  = Function(FS)
dT = TestFunction(FS)

# impose Dirichlet boundary conditions for temperature
bc_T_left   = DirichletBC(FS, Constant(T0), left_boundary)
bc_T_right  = DirichletBC(FS, Constant(T0), right_boundary)
bc_T_top    = DirichletBC(FS, Constant(T1), top_boundary)
bc_T_bottom = DirichletBC(FS, Constant(T1), bottom_boundary)

bc_T = [bc_T_left, bc_T_right, bc_T_top, bc_T_bottom]


# solve for the temperature field
Res_T = k*dot(grad(T),grad(dT))*dx
solve(Res_T == 0, T, bc_T)

# plot temperature field
c = plot(T,mode='color',title='$T$')
plt.colorbar(c)
plot(mesh,linewidth=0.3)
plt.show()

# save temperature field
T.rename("temperature","")
fileT = File("data/temperature.pvd");
fileT << T;

#define vector function space, displacements function u, and test function v
VFS = VectorFunctionSpace(mesh, 'Lagrange', degreeElements)
u  = Function(VFS)
v  = TestFunction(VFS)

# define total strain
def epsilon_tot(u):
    return sym(grad(u))
# define elastic strain
def epsilon_elastic(u,T):
    return epsilon_tot(u) - alpha*T*Identity(d)
# define stress
def sigma(u,T):
    return Lambda*tr(epsilon_elastic(u,T))*Identity(d) + 2*mu*epsilon_elastic(u,T)


#clamped boundary conditions on the left and right boundaries
bc_u_left  = DirichletBC(VFS, Constant((0.,0.)), left_boundary)
bc_u_right = DirichletBC(VFS, Constant((0.,0.)), right_boundary)

bc_u = [bc_u_left, bc_u_right]


# elastic energy functional
Energy = 1/2*inner(sigma(u,T),epsilon_elastic(u,T))*dx

# solve for the displacement field
Res_u = derivative(Energy, u, v)
solve(Res_u == 0, u, bc_u)

# calculate elastic energy
print("Energy = ",assemble(Energy))

# export displacements
u.rename("displacements","")
fileD = File("data/displacements.pvd");
fileD << u;

# calculate and export von Mises stress
devStress = sigma(u,T) - (1./d)*tr(sigma(u,T))*Identity(d)  # deviatoric stress
von_Mises = project(sqrt(3./2*inner(devStress, devStress)), FS)
von_Mises.rename("von Mises","")
fileS = File("data/vonMises_stress.pvd");
fileS << von_Mises;
