#!/bin/sh
start=3
end=5

trapezoid="y"
L=5000 #Length of Ice
T=1000 #Thickness
h=100 #height above Water
bslope=1
islope=-1
for i in $(seq $start $end);
do
   echo "$trapezoid\n$L\n$T\n$h\n$bslope\n$islope" | ./create_mesh.sh
done

#echo "$trapezoid\n$L\n$T\n$h\n$bslope\n$islope"

