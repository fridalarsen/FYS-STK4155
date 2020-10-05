import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import data
terrain = imread("SRTM_data_Norway_2.tif")

# visualize terrain
plt.imshow(terrain, cmap="terrain")
plt.title("Terrain", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.savefig("Figures/terrain.png", dpi=300)
plt.show()

x = np.linspace(0, terrain.shape[0], terrain.shape[0])
y = np.linspace(0, terrain.shape[1], terrain.shape[1])
x, y = np.meshgrid(x, y)
z = terrain.T

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(x, y, z, cmap="terrain")
ax.set_title("Terrain", fontsize=15)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_zlabel("z", fontsize=12)
plt.savefig("Figures/terrain3D.png", dpi=300)
plt.show()
