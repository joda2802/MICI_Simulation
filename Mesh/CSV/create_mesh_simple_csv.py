import csv
#creates a simple rectangular mesh
#Values for ice thickness and height above water level

H = float(input("Enter Ice Thickness:").strip() or "1000")
hcliff = float(input("Enter Height above Water:").strip() or "100")

L = 2000

data = [
    {'id': "1", 'xcoord': 0, 'ycoord': -(H-hcliff)},
    {'id': "2", 'xcoord': L, 'ycoord': -(H-hcliff)},
    {'id': "3", 'xcoord': L, 'ycoord': 0},
    {'id': "4", 'xcoord': L, 'ycoord': hcliff},
    {'id': "5", 'xcoord': 0, 'ycoord': hcliff},
]

with open('Mesh.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'xcoord', 'ycoord']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)


