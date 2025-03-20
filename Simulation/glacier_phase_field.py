  
import dolfinx as dlf

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import *
import ufl 
import basix.ufl
import numpy as np
import sys

# Ausgabe-Level festlegen (EROOR, INFO, OFF, WARNING)
dlf.log.set_log_level(dlf.log.LogLevel.INFO)

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
filename = '../Mesh/Mesh_Initial_MICI.xdmf'
with dlf.io.XDMFFile(comm, filename, 'r') as mesh_inp:
    region = mesh_inp.read_mesh()
    dim = region.topology.dim 
    fdim = dim-1
    region.topology.create_connectivity(fdim, dim)
    facet_tags = mesh_inp.read_meshtags(region, name='Facet tags')

# Dimension des Raums/der Berandungen 
dim = region.topology.dim 
fdim = dim-1

# Zeitschrittsteuerung
tend = 1.0
max_iters = 8
min_iters = 4
dt_scale_down = 0.5
dt_scale_up = 1.0
dt = dlf.fem.Constant(region, 1e-6)



# Eis-Parameter
Emod = 1e9  # E-Modul in [Pa] = [N/m^2]
nu = 0.325  # Querkontraktionszahl [-]
rho_ice = 918.0  # Dichte in [kg/m^3]
g = 9.81  # Erdbeschleunigung in [m/s^2]
rho_water = 1024.0  # Dichte des Wassers in [kg/m^3]
p_luft = 101325 # Dichte der  Luft in [N/m^2]

# Volumenlast aus Eigengewicht
f = dlf.fem.Constant(region, (0, -rho_ice * g))

# Berechne Lame-Konstanten (einfacher für Spannungsberechnung)
lam = dlf.fem.Constant(region, Emod * nu / ((1 - 2 * nu) * (1 + nu)))
mu = dlf.fem.Constant(region, Emod / (2 * (1 + nu)))
K = dlf.fem.Constant(region, lam.value+(2/3)*mu.value)

# Helperfunktion kleinstes Element
def all(x):
   val = np.full(np.shape(x)[1], True)
   return val

all_elements = dlf.mesh.locate_entities(region, dim, all)
hmin = comm.allreduce(np.min(region.h(dim, all_elements)), op=MPI.MIN)  

# Phasenfeld-Parameter
K_Ic = dlf.fem.Constant(region, 95.0e3) # Critical fracture toughness [Pa/m^(1/2)]
G_c = dlf.fem.Constant(region, (K_Ic.value)**2/Emod*10000)
epsilon = dlf.fem.Constant(region, 3*hmin) # Reguarisierungsparameter
eta = dlf.fem.Constant(region, 0.001)
print(G_c.value)

Mob = dlf.fem.Constant(region, 10.0)
iMob = dlf.fem.Constant(region, 1.0/Mob.value)

# Define pressure
x = ufl.SpatialCoordinate(region)
p_front = -rho_water * g * ufl.conditional(le(x[1],0),x[1],0)
pressure_boundary_tag = 3
# Define the normal vector on the boundary
n = ufl.FacetNormal(region)
ds_bc = ufl.Measure("ds", subdomain_data=facet_tags)



# Gemsichten Funktionenraum (u,s)
Ue = element("Lagrange", region.basix_cell(), 1, shape=(dim,)) 
Se = element("Lagrange", region.basix_cell(), 1) 
W = dlf.fem.functionspace(region, mixed_element([Ue, Se]))

# Mapping zwischen Freiheitsgraden des gemischten Funktionenraums
U, Umap = W.sub(0).collapse()
S, Smap = W.sub(1).collapse()

# Oberer/unterer Rand
def top(x):
    return np.isclose(x[1], 1.0)

def bottom(x):
    return np.isclose(x[1], 0.0)

# Anfangsriss
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.5), x[0]<0.1)


# Maximale Verschiebung oben/unten
#umax = 0.4
#utop = dlf.fem.Constant(region, umax)
#ubottom = dlf.fem.Constant(region, -umax)

"""
# Obere/Untere/Riss Randbedingungen
topfacets = dlf.mesh.locate_entities_boundary(region, fdim, top)
bottomfacets = dlf.mesh.locate_entities_boundary(region, fdim, bottom)
crackfacets = dlf.mesh.locate_entities(region, fdim, crack)

topdofs_x = dlf.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, topfacets)
topdofs_y = dlf.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, topfacets)
bottomdofs_x = dlf.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, bottomfacets)
bottomdofs_y = dlf.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, bottomfacets)
crackdofs = dlf.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)

bctop_x = dlf.fem.dirichletbc(0.0, topdofs_x, W.sub(0).sub(0))
bctop_y = dlf.fem.dirichletbc(utop, topdofs_y, W.sub(0).sub(1))
bcbottom_x = dlf.fem.dirichletbc(0.0, bottomdofs_x, W.sub(0).sub(0))
bcbottom_y = dlf.fem.dirichletbc(ubottom, bottomdofs_y, W.sub(0).sub(1))
bccrack = dlf.fem.dirichletbc(0.0, crackdofs, W.sub(1))
bcs = [bctop_x, bctop_y, bcbottom_x, bcbottom_y, bccrack]
"""

#crackfacets = dlf.mesh.locate_entities(region, fdim, crack)
#crackdofs = dlf.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
#bccrack = dlf.fem.dirichletbc(1, crackdofs, W.sub(1))
# Element-Typ (3D Verschiebungselement) und Funktionenraum erzeugen
# CG := Continuous Galerkin
element = basix.ufl.element('CG', region.topology.cell_name(), 1, shape=(dim,))
V = dlf.fem.functionspace(region, element)

# Da sigma in der Ableitung von u steckt, soll der Ansatzraum in einer Dimension niedriger sein.
# DG
# Element-Typ (Spannungstensor) und Funktionenraum erzeugen
element_tensor = basix.ufl.element('DG', region.topology.cell_name(), 0, shape=(dim,dim,)) # Das zweite Komma muss dort hin
T = dlf.fem.functionspace(region, element_tensor)


def links(x):
    return x[0] < 0


# Randbedingungen
buttom_dofs_y = dlf.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, facet_tags.find(2))
inflow_dofs_x = dlf.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, facet_tags.find(5))
inflow_cells = dlf.mesh.locate_entities(region, fdim, links)
inflow_dofs_s = dlf.fem.locate_dofs_topological(W.sub(1), fdim, inflow_cells)

BC_inflow = dlf.fem.dirichletbc(0.0, inflow_dofs_x, W.sub(0).sub(0))
BC_bottom = dlf.fem.dirichletbc(0.0, buttom_dofs_y, W.sub(0).sub(1))
BC_inflow_s = dlf.fem.dirichletbc(1.0, inflow_dofs_s, W.sub(1))



# Sammeln der Randbedingungen
bcs = [BC_inflow, BC_bottom, BC_inflow_s]

# Degradationsfunktion
def degrad(s):
    degrad = s**2+eta
    return degrad

# Verzerrungsenergie

# Helperfunktionen für Verzerrungsenergie
def positive(val):
    return 0.5*(np.abs(val)+val)

def negative(val):
    return 0.5*(np.abs(val)-val)

def eps(u):
    eps = ufl.sym(ufl.grad(u))
    return eps

# Verzerrungsenergie Zug -> relevant für Rissbildung
def psielP(eps):
    epsD = ufl.dev(eps)
    psiel = 0.5*K*positive(ufl.tr(eps))**2 + mu*ufl.inner(epsD, epsD)
    return psiel

# Verzerrungsenergie Druck
def psielM(eps):
    psielM = 0.5*K*negative(ufl.tr(eps))**2
    return psielM

def stressf(eps, s):
    eps_var = ufl.variable(eps)
    s_var = ufl.variable(s)
    str = ufl.diff(degrad(s_var) * psielP(eps_var) + psielM(eps_var), eps_var)
    

# Bruchenergie
def psifrac(s):
    psifrac = G_c*(((1-s)**2)/(4*epsilon)+epsilon*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return psifrac

# Loesung (aktuell, alt), Restartloesung, Testfunktion
w =  dlf.fem.Function(W)
wm1 =  dlf.fem.Function(W)
wrestart =  dlf.fem.Function(W)
dw = ufl.TestFunction(W)
#strain function
strain = dlf.fem.Function(T)
stress = dlf.fem.Function(T)

# Aufspalten in Verschiebung und Bruchfeld
u, s = ufl.split(w)
um1, sm1 = ufl.split(wm1)
du, ds = ufl.split(dw)


# Potential, Gleichgeeicht, Triebkraft, Rate, Residuum
pot = (degrad(s)*psielP(eps(u))+ psielM(eps(u)) + psifrac(s))*ufl.dx
equi = ufl.derivative(pot, u, du)
sdrive = ufl.derivative(pot, s, ds)
rate = (s-sm1)/dt*ds*ufl.dx
L =  - ufl.dot(p_front * n, du) * ds_bc(pressure_boundary_tag) + ufl.dot(f*degrad(s), du) * ufl.dx
res = iMob*rate+sdrive+equi - L

# Nichtlineares Problem und Loeser
nl_problem = dlf.fem.petsc.NonlinearProblem(res, w, bcs)
nl_solver = dlf.nls.petsc.NewtonSolver(comm, nl_problem)
nl_solver.max_it = max_iters
nl_solver.convergence_criterion = "incremental"

# Konfigurieren des linearen Gleichungsloesers 
lin_solver = nl_solver.krylov_solver
opts = PETSc.Options()
option_prefix = lin_solver.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly" 
opts[f"{option_prefix}pc_type"] = "lu" 
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
lin_solver.setFromOptions()

# Initialisiere s=1
wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
wrestart.x.array[:] = wm1.x.array[:]

print("s init:", wm1.sub(1).x.array[:].min(), wm1.sub(1).x.array[:].max())
u = wm1.sub(0).collapse()
s = wm1.sub(1).collapse()
s.name = "s"
print(s)
# xdmf-Ausgabe vorbereiten
with dlf.io.XDMFFile(comm, 'output.xdmf', 'w') as xdmfout:
    xdmfout.write_mesh(region)
    #xdmfout.write_function(s, 0)

u, s = ufl.split(w)
# Zeit initialisieren
t = 0.0
trestart = 0.0

# Berechnungsschleife
while t<tend:

    # Stoppuhr fuer Zeitschritt
    stopwatch_step = stopwatch.elapsed()[0]

    # update obere/untere Randbedingung  
    #utop.value = umax*t
    #ubottom.value = -umax*t

    # Aktuelle Loesungsinfos
    if rank == 0:
        print('')
        print('Computing solution at t = {0:.4e}'.format(t))
        print('Current time step dt = {0:.7e}'.format(dt.value))
        sys.stdout.flush()

    # Steure Zeitschritt adaptiv
    converged = False
    iters = 0
    try:
        (iters, converged) = nl_solver.solve(w)
    except RuntimeError:
        dt.value = dt_scale_down*dt.value
         
    if converged and iters < min_iters and t > np.finfo(float).eps:
        dt.value = dt_scale_up*dt.value
        if rank == 0:
            print('!FAST CONVERGENCE => dt = {0:.4e}'.format(dt.value))
            sys.stdout.flush()
    
    if converged:
        # Aufspalten in Verschiebungs- und Bruchfeld
        u = w.sub(0).collapse()
        s = w.sub(1).collapse()
        sm1 = wm1.sub(1).collapse()
        u.name = 'u'
        s.name = 's'
        
        strain_expr = dlf.fem.Expression(eps(u), T.element.interpolation_points())
        #stress_expr = dlf.fem.Expression(stressf(eps(u),s), T.element.interpolation_points())

        strain.interpolate(strain_expr)
        #stress.interpolate(stress_expr)
        strain.name = 'strain'
        #stress.name = 'stress'

        # xdmf-Ausgabe fortfuehren
        with dlf.io.XDMFFile(comm, 'output.xdmf', 'a') as xdmfout:
            xdmfout.write_function(u, t)
            xdmfout.write_function(s, t)
            xdmfout.write_function(strain, t)
            #xdmfout.write_function(stress, t)

        
        # update Loesung und Restart-Loesung
        wm1.x.array[Umap] = u.x.array[:]
        wm1.x.array[Smap] = np.minimum(s.x.array[:], sm1.x.array[:])
        wrestart.x.array[:] = wm1.x.array[:] # change w to wm1
        trestart = t
        t = t+dt.value
    else:
        # Setze auf Restart-Loesung
        if rank == 0:
            print('!NO CONVERGENCE  =>  dt = {0:.4e}'.format(dt.value))
            sys.stdout.flush()
        t = trestart+dt.value
        w.x.array[:] = wrestart.x.array[:]
        w.x.scatter_forward()

    # Tensorraum für stress, strain
   

    #stress = dlf.fem.Function(T)

    #stress_expr = dlf.fem.Expression(stress(eps(u), s), T.element.interpolation_points())
    #stress.interpolate(stress_expr)
    #stress.name = 'stress'
    


   

    # Bericht aktuelle Loesung
    if rank == 0:    
        print('No. of iterations:', iters)
        print('Converged:        ', converged)
        print('Runtime step/total {0:.2e}/{1:.2e}'.format(stopwatch.elapsed()[0]-stopwatch_step, stopwatch.elapsed()[0]))
        sys.stdout.flush()

# Stoppuhr anhalten
stopwatch.stop()



