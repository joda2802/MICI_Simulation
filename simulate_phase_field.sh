#!/bin/sh

#execute phase-field simulation with precreated mesh
cd Simulation
touch output.h5
touch output.xdmf
python3 glacier_eigenweight.py