# Scatter Plot Visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure, the axis
fig, ax = plt.subplots()
ax.set_facecolor('black')  # Set the plotting area to black
fig.set_facecolor('black')
ax.set_xlim(0, 10)  # Setting x-axis limits
ax.set_ylim(0, 10)  # Setting y-axis limits

# Number of points per color group and their colors
n_points = 10
colors = ['red', 'blue', 'green', 'yellow']

# Initial positions randomly scattered
x_values = np.random.rand(4 * n_points) * 10
y_values = np.random.rand(4 * n_points) * 10
c_values = np.array(colors * n_points)

scatter = ax.scatter(x_values, y_values, color=c_values, s=50)

# Define target positions for each color group initially
def redefine_targets():
    return {
        'red': (np.random.rand() * 8 + 1, np.random.rand() * 8 + 1),
        'blue': (np.random.rand() * 8 + 1, np.random.rand() * 8 + 1),
        'green': (np.random.rand() * 8 + 1, np.random.rand() * 8 + 1),
        'yellow': (np.random.rand() * 8 + 1, np.random.rand() * 8 + 1)
    }

targets = redefine_targets()

def init():
    global x_values, y_values, targets
    x_values = np.random.rand(4 * n_points) * 10
    y_values = np.random.rand(4 * n_points) * 10
    targets = redefine_targets()
    scatter.set_offsets(np.c_[x_values, y_values])
    return scatter,

def update(frame):
    # If it's the beginning of the animation, reinitialize positions and targets
    if frame == 0:
        return init()
    
    # Move each point a fraction of the distance towards its group's target
    for i, color in enumerate(c_values):
        target_x, target_y = targets[color]
        x_values[i] += (target_x - x_values[i]) * 0.05
        y_values[i] += (target_y - y_values[i]) * 0.05

    scatter.set_offsets(np.c_[x_values, y_values])
    return scatter,

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50, repeat=True)

plt.show()
