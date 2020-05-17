import random
from dolfin import *
from mshr import *


# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)
    
    
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
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
            

lmbda  = 1.0e-02  # surface parameter
dt     = 5.0e-06  # time step
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson, theta=0 -> forward Euler

# Create mesh and build function space
mesh = UnitSquareMesh.create(96, 96, CellType.Type.quadrilateral)
P = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement([P,P]),constrained_domain=PeriodicBoundary())

# Define functions
u     = Function(MFS)  # current solution
u_old = Function(MFS)  # solution from previous converged step
# Split mixed functions
c,  mu  = split(u)
c_old, mu_old = split(u_old)

# Define test functions
tf = TestFunction(MFS)
q, v  = split(tf)



u.rename("fields","")

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u_old.interpolate(u_init)

# Compute the chemical potential df/dc
c = variable(c)
f    = 100*c**2*(1-c)**2
dfdc = diff(f, c)

# mu_(n+theta)
mu_mid = (1.0-theta)*mu_old + theta*mu

# Weak statement of the equations
Res_0 = c*q*dx - c_old*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
Res_1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
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
    u_old.vector()[:] = u.vector()
    solve(Res == 0, u)
    fileC << (u.split()[0], t)
    fileM << (u.split()[1], t)
