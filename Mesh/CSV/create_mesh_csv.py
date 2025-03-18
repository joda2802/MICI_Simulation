import csv
#creates a 2D-trapezoid shaped mesh 
#Input Values

H = float(input("Enter Ice Thickness at the Cliff:").strip() or "1000")
hcliff = float(input("Enter Height above Water:").strip() or "100")
bed_slope =  float(input("Enter Slope of Bedrock:").strip() or "0")
ice_slope =  float(input("Enter Slope of Ice Surface:").strip() or "0")

L = 2000

data = [
    {'id': "1", 'xcoord': 0, 'ycoord': -(H-hcliff)-L*bed_slope},
    {'id': "2", 'xcoord': L, 'ycoord': -(H-hcliff)},
    {'id': "3", 'xcoord': L, 'ycoord': 0},
    {'id': "4", 'xcoord': L, 'ycoord': hcliff},
    {'id': "5", 'xcoord': 0, 'ycoord': hcliff-L*ice_slope},
]

with open('Mesh.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'xcoord', 'ycoord']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)


