# Line plot visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure, the axis, and the plot element
fig, ax = plt.subplots()
ax.set_facecolor('black')  
fig.set_facecolor('black')

# Setting colors for the lines in the plot. 
colors = ['red', 'blue', 'green', 'yellow', 'cyan']
lines = [plt.plot([], [], color=c, animated=True)[0] for c in colors]
xdata, ydata = [], []

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return lines

def update(frame):
    if frame == 0:  # Reset data at the start of the animation
        xdata.clear()
        ydata.clear()

    xdata.append(frame)
    ydata.append(np.sin(frame))
    
    # For adding variation to other lines
    shifts = [0, 0.5, 1, 1.5, 2]

    if len(xdata) > 50:  # Limit to the last 50 data points for display
        del xdata[0]
        del ydata[0]

    for i, ln in enumerate(lines):
        ln.set_data(xdata, [np.sin(x + shifts[i]) for x in xdata])
    return lines

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True, interval=50)

plt.show()
