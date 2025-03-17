import numpy as np
import dolfinx as dlfx
import gmsh
from mpi4py import MPI
import sys
import csv

'''
Das Skript erstellt einen Gmsh mesh aus einer geordneten Liste von Punkten

Vorgehen: Passe Schritt 1-3 an um gmsh zu erstellen 

Schritt1: Passe mesh_name an 
Schritt2: Passe den Pfad zum csv-file an 
Schritt3: Die Tags müssen angpasst werden, 
          tags=range(1,8) bedeudet, dass Linie 1 bis 7 enthalten sind, 
          damit sind Punkt 1 bis 8 verbunden 
Schritt4: Überprüfe in Paraview ob die Tags und das Mesh sinnvoll erstellt wurden. 

'''

mesh_name = "Mesh_Geometry_Brunt2_Refined"
gdim = 2 #Dimension


with open('/root/share/MICI_Simulation/Mesh/CSV/Mesh.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

'Auslesen der CSV'
gmsh.initialize()
gmsh.clear()
gmsh.model.add(mesh_name)

#Geometrie erstellen
points = [gmsh.model.occ.add_point(x=float(data[i][1]), y=float(data[i][2]), z=0) for i in range(1,len(data)) ]
avg_point = gmsh.model.occ.add_point(x=np.mean([float(data[i][1]) for i in range(1,9)]), y=np.mean([float(data[i][2]) for i in range(1,9)]), z=0)
lines = [gmsh.model.occ.add_line(points[i], points[i+1]) for i in range(0,len(points)-1)]
lines.append(gmsh.model.occ.add_line(points[len(points)-1], points[0]))
loop = gmsh.model.occ.add_curve_loop(lines)
area = gmsh.model.occ.add_plane_surface([loop])
gmsh.model.occ.synchronize()

#Tags
'''
#no crevasse
outflow =[i for i in range(8,43)]+[i for i in range(46,56)]
gmsh.model.addPhysicalGroup(dim=gdim, tags=[area], tag=1)   #Netz
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(1,8), tag=2) #icerise
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=outflow, tag=3)    #outflow
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(43,46), tag=4)   #inflow
'''
#crevasse
outflow =[i for i in range(8,39)]+[i for i in range(50,52)]+[i for i in range(61,71)]
gmsh.model.addPhysicalGroup(dim=gdim, tags=[area], tag=1)   #Netz
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(1,8), tag=2) #icerise
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=outflow, tag=3)    #outflow
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(58,61), tag=4)   #inflow
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(39,50), tag=5)   #crack 1
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(52,58), tag=6)   #crack 2

'''
#crevasse2
outflow =[i for i in range(8,39)]+[i for i in range(62,72)]

gmsh.model.addPhysicalGroup(dim=gdim, tags=[area], tag=1)   #Netz
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(1,8), tag=2) #icerise
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=outflow, tag=3)    #outflow
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(59,62), tag=4)   #inflow
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(39,59), tag=5)   #Riss
'''

#Netz
h_size = 1000 #Elemengröße
h_size_fine = 200 #verfeinerte Elementgröße
'''
#globale Elemeltgröße
gmsh.option.setNumber('Mesh.CharacteristicLengthFactor',h_size)
gmsh.model.mesh.generate(dim=gdim)
'''

#variable Elemetgröße
fine_points = [avg_point] #Liste von Punkten, um die verfeinert werden soll
#fine_lines = [] #Liste von Linien, um die verfeinert werden soll
dist_min = 5000
dist_max = 10000


gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", fine_points)
#gmsh.model.mesh.field.setNumbers(1, "CurvesList", fine_lines)
#gmsh.model.mesh.field.setNumber(1,"Sampling",100)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", h_size_fine)
gmsh.model.mesh.field.setNumber(2, "SizeMax", h_size)
gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)

gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.mesh.generate(dim=gdim)



mesh, cell_tags, facet_tags = dlfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=gdim)
#Abspeicern als XDMF
with dlfx.io.XDMFFile(MPI.COMM_WORLD, mesh_name+'.xdmf', 'w') as xdmfout:
    xdmfout.write_mesh(mesh)
    xdmfout.write_meshtags(cell_tags, mesh.geometry) 
    xdmfout.write_meshtags(facet_tags, mesh.geometry) 