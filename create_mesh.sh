#!/bin/sh
python3 Mesh/CSV/create_mesh_csv.py
python3 Mesh/Create_GMSH_from_CSV_with_Refinement.py