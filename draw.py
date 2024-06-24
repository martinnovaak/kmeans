import matplotlib.pyplot as plt
import numpy as np

# File paths
data_file = 'data.txt'
assignments_file = 'assignments.txt'
centers_file = 'final_centers.txt'

# Number of points to visualize
N = 500

# Function to read data points
def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split())
            points.append((x, y))
    return points

# Function to read cluster assignments
def read_assignments(file_path):
    assignments = []
    with open(file_path, 'r') as file:
        for line in file:
            assignments.append(int(line.strip()))
    return assignments

# Function to read cluster centers
def read_centers(file_path):
    centers = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            centers.append((x, y))
    return centers

# Read data
points = read_points(data_file)
assignments = read_assignments(assignments_file)
centers = read_centers(centers_file)

# Limit to first N points
points = points[:N]
assignments = assignments[:N]

# Convert to numpy arrays for easy slicing
points = np.array(points)
assignments = np.array(assignments)
centers = np.array(centers)

# Create a scatter plot
plt.figure(figsize=(10, 8))

# Plot points with colors according to their cluster
for i, point in enumerate(points):
    plt.scatter(point[0], point[1], color=f'C{assignments[i]}', s=10)

# Plot centroids
for i, center in enumerate(centers):
    plt.scatter(center[0], center[1], color=f'C{i}', edgecolors='black', marker='X', s=200)

# Draw lines from points to their centroids
for i, point in enumerate(points):
    plt.plot([point[0], centers[assignments[i]][0]], [point[1], centers[assignments[i]][1]], 
             color=f'C{assignments[i]}', linestyle='-', linewidth=0.05)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Assignments for First 500 Points')
plt.grid(True)
plt.show()
