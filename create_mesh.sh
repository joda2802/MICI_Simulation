#!/bin/sh

#create rectangular mesh with refinement
cd Mesh
python3 CSV/create_mesh_csv.py
python3 Create_GMSH_from_CSV_with_Refinement.py