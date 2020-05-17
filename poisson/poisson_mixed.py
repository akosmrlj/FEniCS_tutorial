from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# Create mesh
R = 1. # radius
N = 20 # mesh density
domain = Circle(Point(0., 0.), R)
mesh = generate_mesh(domain, N)
#plot(mesh,linewidth=0.3)


#define function space with mixed finite elements
degreeElements = 1
FE_u     = FiniteElement('Lagrange', mesh.ufl_cell(), degreeElements)
FE_sigma = FiniteElement('Lagrange', mesh.ufl_cell(), degreeElements)
FS = FunctionSpace(mesh, MixedElement([FE_u, FE_sigma*FE_sigma]))

# define function and split it into u and sigma
F  = Function(FS)           # center line
u,sigma = split(F)

# define test function and split it into v and tau
TF = TestFunction(FS)
v,tau = split(TF)


# define functions a and f
a  = Expression('1-0.5*(x[0]*x[0]+x[1]*x[1])', degree=degreeElements)
f  = 1

# define function g
class G(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval_cell(self, value, x, ufc_cell):
        value[0]=x[1]

g  = G(degree=degreeElements)


#impose Dirichlet boundary conditions for u
def right_boundary(x, on_boundary):
    return on_boundary and x[0]>=0

uD = Expression('x[0]',degree=degreeElements)
bc = DirichletBC(FS.sub(0), uD, right_boundary)

#weak formulation of the problem
Res_1 = (dot(sigma,grad(v)) - f*v)*dx - v*g*ds
Res_2 = (dot(sigma, tau) - dot(a*grad(u), tau))*dx
Res   = Res_1 + Res_2

# solve the problem and store solution in F
solve(Res == 0, F, bc)

# plot solution for u
c = plot(u,mode='color',title='$u$')
plt.colorbar(c)
plot(mesh,linewidth=0.3)
plt.show()

# plot solution for sigma
c=plot(sigma,title='$\sigma=a \\nabla u$',width=.008)
plt.colorbar(c)
plot(mesh,linewidth=0.3)
plt.show()

V = FunctionSpace(mesh, 'Lagrange', degreeElements)
divSigma = project(div(sigma), V)
c=plot(divSigma,mode='color',title='$\\nabla \cdot \sigma=\\nabla \cdot (a \\nabla u)$',vmin=-1.1,vmax=-0.9)
plt.colorbar(c)
plot(mesh,linewidth=0.3)
plt.show()

