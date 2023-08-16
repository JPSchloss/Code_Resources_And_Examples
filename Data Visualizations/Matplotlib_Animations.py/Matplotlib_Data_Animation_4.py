# Heat Map Visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap
colors = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0)]
cmap_name = 'custom_div_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Set up the figure, the axis
fig, ax = plt.subplots()
ax.set_facecolor('black')
fig.set_facecolor('black')

# Initial heatmap data
data = np.zeros((10, 10))
rows, cols = data.shape

# Display initial data
cax = ax.imshow(data, cmap=cm, interpolation='nearest')

# Add colorbar for reference
cbar = fig.colorbar(cax)

def init():
    global data
    data = np.zeros((10, 10))
    cax.set_data(data)
    return cax,

def update(frame):
    global data
    
    # Adding sinusoidal waves to data beacuase it looks pretty cool.
    for i in range(rows):
        for j in range(cols):
            data[i, j] += 0.005 * np.sin(i + frame * 0.1) * np.sin(j + frame * 0.1)
    
    # Clamp the values to be between 0 and 1
    data = np.clip(data, 0, 1)
    
    cax.set_data(data)
    return cax,

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50)

plt.show()
