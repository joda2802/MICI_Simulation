#!/bin/sh

#create rectangular mesh with refinement
cd Mesh/CSV

read -p "Trapezoid-Shaped mesh? (y/n)" answer
if [ "$answer" = "y" ];
then
  python3 create_mesh_csv.py
else
  python3 create_mesh_simple_csv.py
fi


cd ..
python3 Create_GMSH_from_CSV_with_Refinement.py