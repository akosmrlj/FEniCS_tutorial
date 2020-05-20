from __future__ import print_function
import random
from fenics import *
from dolfin import *
from mshr import *

    
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT
        # on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                (not ((near(x[0], 0) and near(x[1], 1)) or
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.
            


# Create mesh and define function space with periodic boundary conditions
mesh = UnitSquareMesh.create(96, 96, CellType.Type.quadrilateral)
P = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement([P,P]),constrained_domain=PeriodicBoundary())

# Define functions
u_new = Function(MFS)  # solution for the next step
u_old  = Function(MFS)  # solution from previous step
u_new.rename("fields","")
# Split mixed functions
c_new, mu_new  = split(u_new)
c_old, mu_old = split(u_old)

# Define test functions
tf = TestFunction(MFS)
q, v  = split(tf)

# Define test functions
tf = TestFunction(MFS)
q, v  = split(tf)


# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random()) # concentration
        values[1] = 0.0 # chemical potential
    def value_shape(self):
        return (2,)
    

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u_new.interpolate(u_init)
u_old.interpolate(u_init)



# Compute the chemical potential df/dc
c_new = variable(c_new)
f    = 100*c_new**2*(1-c_new)**2
dfdc = diff(f, c_new)


lmbda  = 1.0e-02  # surface parameter
dt     = 5.0e-06  # time step
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# mu_(n+theta)
mu_mid = (1.0-theta)*mu_old + theta*mu_new

# Weak statement of the equations
Res_0 = (c_new - c_old)/dt*q*dx + dot(grad(mu_mid), grad(q))*dx
Res_1 = mu_new*v*dx - dfdc*v*dx - lmbda*dot(grad(c_new), grad(v))*dx
Res = Res_0 + Res_1


# Output files for concentration and chemical potential
fileC = File("data/concentration.pvd", "compressed")
fileM = File("data/chem_potential.pvd", "compressed")


# Step in time
t = 0.0
T = 50*dt
while (t < T):
    t += dt
    print(t)
    u_old.vector()[:] = u_new.vector()
    solve(Res == 0, u_new)
    fileC << (u_new.split()[0], t)
    fileM << (u_new.split()[1], t)
