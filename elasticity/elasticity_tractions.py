from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt

# Create rectangular mesh with two circular inclusions
N=100
L=1
R_2=0.05
R_3=0.08
domain = Rectangle(Point(-L/2,-L/2),Point(L/2,L/2))
# mark subdomains with markers 1, 2, 3
domain.set_subdomain(1, Rectangle(Point(-L/2,-L/2),Point(L/2,L/2)))
domain.set_subdomain(2, Circle(Point(0.,0.43), R_2))
domain.set_subdomain(3, Circle(Point(-0.15,0.35), R_3))
mesh = generate_mesh(domain, N)
d = mesh.topology().dim() # dimensionality of the problem
markers = MeshFunction("size_t", mesh, d , mesh.domains())


# define boundary subdomains
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -L/2)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], +L/2)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], L/2)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], -L/2)

left = Left()
right = Right()
top = Top()
bottom = Bottom()

# mark boundary subdomains with markers 1, 2, 3, 4
boundaries = MeshFunction("size_t", mesh, d-1, 0)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)


# elastic constants of the matrix and two circular inclusions
E_1=1
E_2=10
E_3=0.1
nu_1=0.3
nu_2=0.2
nu_3=0.1



# define class for calculating the Young's modulus over the whole domain
class E_class(UserExpression):
    def __init__(self, **kwargs):
        self.markers = markers
        super().__init__(**kwargs)
    def eval_cell(self, value, x, ufc_cell):
        if markers[ufc_cell.index] == 1:
            value[0] = E_1
        elif markers[ufc_cell.index] == 2:
            value[0] = E_2
        else:
            value[0] = E_3

# define class for calculating the Poisson's ratio over the whole domain
class nu_class(UserExpression):
    def __init__(self, **kwargs):
        self.markers = markers
        super().__init__(**kwargs)
    def eval_cell(self, value, x, ufc_cell):
        if markers[ufc_cell.index] == 1:
            value[0] = nu_1
        elif markers[ufc_cell.index] == 2:
            value[0] = nu_2
        else:
            value[0] = nu_3

# functions of elastic constants on the whole domain
E = E_class(degree=1)
nu = nu_class(degree=1)
mu=E/2/(1+nu)
Lambda=E*nu/(1-nu*nu)




#define function space with mixed finite elements (displacements + 3 Lagrange multipliers)
degreeElements = 1
P1 = FiniteElement('Lagrange', mesh.ufl_cell(), degreeElements)
R = FiniteElement('Real', mesh.ufl_cell(), 0)
MFS = FunctionSpace(mesh, MixedElement([(P1*P1),(R*R),R]))

#define function and split it into displacements u and Lagrange multipliers
f  = Function(MFS)
u, c_trans, c_rot = split(f)
#define test function
tf = TestFunction(MFS)


# define strain and stress
def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return 2*mu*epsilon(u) + Lambda*tr(epsilon(u))*Identity(d)

#external load
sigma_xx = 0.2*E_1
sigma_xy = 0
sigma_yy = 0
sigma_0 = Constant(((sigma_xx,sigma_xy),(sigma_xy,sigma_yy)))

#unit normal vector to the boundary
n = FacetNormal(mesh)
                   
#tractions on boundaries
t = sigma_0 * n

#calculate elastic energy
elastic_energy = 1/2*inner(sigma(u),epsilon(u))*dx
#calculate work of external tractions
work = dot(t,u)*ds
#Lagrange multipliers to prevent rigid body motions
r=Expression(('x[0]','x[1]'),degree=1)
constraints = dot(c_trans,u)*dx + c_rot*(r[0]*u[1]-r[1]*u[0])*dx
#total free energy
free_energy =  elastic_energy - work + constraints


#minimize total free energy
Res = derivative(free_energy, f, tf)
solve(Res == 0, f)

#calculate total free energy
print("Tot Free Energy = ",assemble(free_energy))
print("Elastic Energy = ",assemble(elastic_energy))
print("Work = ",assemble(work))
print("Constraints = ",assemble(constraints))


# export displacements
VFS = VectorFunctionSpace(mesh, 'Lagrange', 1)
disp=project(u, VFS)
disp.rename("displacements","")
fileD = File("data/tractions_displacement.pvd");
fileD << disp;

# calculate and export von Mises stress
FS = FunctionSpace(mesh, 'Lagrange', 1)
devStress = sigma(u) - (1./d)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = project(sqrt(3./2*inner(devStress, devStress)), FS)
von_Mises.rename("von Mises","")
fileS = File("data/tractions_vonMises_stress.pvd");
fileS << von_Mises;

# calculate and export stress component sigma_xx
sigma_xx = project(sigma(u)[0,0], FS)
sigma_xx.rename("sigma_xx","")
fileS = File("data/tractions_sigma_xx.pvd");
fileS << sigma_xx;

# calculate and export stress component sigma_yy
sigma_yy = project(sigma(u)[1,1], FS)
sigma_yy.rename("sigma_yy","")
fileS = File("data/tractions_sigma_yy.pvd");
fileS << sigma_yy;

# calculate and export stress component sigma_xy
sigma_xy = project(sigma(u)[0,1], FS)
sigma_xy.rename("sigma_xy","")
fileS = File("data/tractions_sigma_xy.pvd");
fileS << sigma_xy;

# export Young's modulus
young = project(E, FS)
young.rename("Young's modulus","")
fileS = File("data/tractions_young.pvd");
fileS << young;
