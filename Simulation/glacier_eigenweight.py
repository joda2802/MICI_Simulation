import dolfinx as dlf
import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
import sys

# Ausgabe − Level festlegen (ERROR, INFO, OFF, WARNING)
dlf.log.set_log_level(dlf.log.LogLevel.INFO)

# Import mesh
# input und output Dateien
filename = 'Mesh_Initial_MICI.xdmf'


# Stoppuhr einrichten und starten
stopwatch = dlf.common.Timer()
stopwatch.start()

# MPI-Umgebung einrichten und Status
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# Netz einlesen, Dimension und Randbedigungstags
with dlf.io.XDMFFile(comm, filename, 'r') as mesh_inp:
    region = mesh_inp.read_mesh()
    dim = region.topology.dim 
    fdim = dim-1
    region.topology.create_connectivity(fdim, dim)
    facet_tags = mesh_inp.read_meshtags(region, name='Facet tags')



# Dimension des Raums/der Berandungen
Emod = 10e9  # E-Modul in [Pa] = [N/m^2]
nu = 0.325  # Querkontraktionszahl [-]
rho = 918.0  # Dichte in [kg/m^3]
g = 9.81  # Erdbeschleunigung in [m/s^2]

# Berechne Lame-Konstanten (einfacher für Spannungsberechnung)
lam = dlf.fem.Constant(region, Emod * nu / ((1 - 2 * nu) * (1 + nu)))
mu = dlf.fem.Constant(region, Emod / (2 * (1 + nu)))

# Volumenlast aus Eigengewicht
f = dlf.fem.Constant(region, (0, -rho * g))


# Element-Typ (3D Verschiebungselement) und Funktionenraum erzeugen
element = basix.ufl.element('CG', region.topology.cell_name(), 1, shape=(dim,))
V = dlf.fem.functionspace(region, element)

# Randbedingungen
buttom_dofs_y = dlf.fem.locate_dofs_topological(V.sub(1), fdim, facet_tags.find(2))
inflow_dofs_x = dlf.fem.locate_dofs_topological(V.sub(0), fdim, facet_tags.find(5))




BC_inflow = dlf.fem.dirichletbc(0.0, inflow_dofs_x, V.sub(0))
BC_bottom = dlf.fem.dirichletbc(0.0, buttom_dofs_y, V.sub(1))

# Sammeln der Randbedingungen
bcs = [BC_inflow, BC_bottom]

# Convert to a FEniCSx function
def pressure_function(x):
    return rho * g * x[1]  # Pressure = ρgh (x[1] is the y-coordinate)


# Create function space for pressure (scalar field)
P0 = dlf.fem.functionspace(region, ("DG", 0))

# Create function and interpolate
pressure = dlf.fem.Function(P0)
pressure.interpolate(pressure_function)

pressure_boundary_tag = 3

# Define the normal vector on the boundary
n = ufl.FacetNormal(region)

# Verzerrungen eps als symmetrischer Verschiebungsgradient
def eps(u):
    return ufl.sym(ufl.grad(u))

# Spannung sig über isotropes Stoffgesetz
def sig(u):
    return lam * ufl.tr(eps(u)) * ufl.Identity(dim) + 2 * mu * eps(u)

# Definiere Ansatz-/Test-/FEM-Funktionen
u = ufl.TrialFunction(V)
du = ufl.TestFunction(V)
uh = dlf.fem.Function(V)

# Schwache Form des Gleichgewichts; 
a = ufl.inner(sig(u), eps(du)) * ufl.dx
# Assuming the pressure acts on the x-component (update if needed)
L = ufl.dot(f, du) * ufl.dx - pressure * ufl.dot(n, du) * ufl.ds(pressure_boundary_tag)

problem = dlf.fem.petsc.LinearProblem(a, L, bcs, petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
uh = problem.solve()

with dlf.io.XDMFFile(region.comm, "output.xdmf", "w") as xdmfout:
    xdmfout.write_mesh(region)
    uh.name = "displacement u"
    xdmfout.write_function(uh)

