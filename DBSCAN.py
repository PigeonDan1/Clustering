import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
import os
import shutil

# Step 1: Read and process the data
data = pd.read_csv('students003.txt', sep='\s+', header=None, names=['time_step', 'ID', 'X', 'Y'])

# Create a directory to save the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Step 2: Perform DBSCAN clustering for each time_step and visualize the results
unique_time_steps = data['time_step'].unique()
frame_count = 0

for time_step in unique_time_steps:
    plt.figure(figsize=(10, 8))
    time_step_data = data[data['time_step'] == time_step]

    X = time_step_data[['X', 'Y']].values

    # DBSCAN clustering
    dbscan = DBSCAN(eps=1.0, min_samples=2)  # Adjust eps and min_samples as needed
    labels = dbscan.fit_predict(X)
    unique_labels = np.unique(labels)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # Plot each cluster
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points
            cluster_points = X[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='k', label='Noise')
        else:
            cluster_points = X[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i + 1}')

    plt.legend()
    plt.title(f'Time Step: {time_step}')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the plot to a file
    plt.savefig(f'frames/frame_{frame_count:04d}.png')
    plt.close()
    frame_count += 1

# Step 3: Create a video from the frames
frame_files = [f'frames/frame_{i:04d}.png' for i in range(frame_count)]
frame = cv2.imread(frame_files[0])
height, width, layers = frame.shape
video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

for file in frame_files:
    video.write(cv2.imread(file))

cv2.destroyAllWindows()
video.release()

# Clean up the frames directory

shutil.rmtree('frames')
