import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure, the axis
fig, ax = plt.subplots()
ax.set_facecolor('black')  # Set the plotting area to black
fig.set_facecolor('black')
ax.set_ylim(0, 100)  # Our data will vary between 0 and 100

# Number of bars we want to display
n_bars = 5
x = np.arange(n_bars)
bar_width = 0.8
data_values = [10 for _ in range(n_bars)]  # Starting value for our data
colors = ['red', 'blue', 'green', 'yellow', 'cyan']
bars = ax.bar(x, data_values, bar_width, color=colors)

def init():
    return bars

def update(frame):
    # Randomly vary the growth rate for each bar in each frame
    growth_rates = np.random.randint(1, 6, n_bars)  # Growth rates between 1 and 5 inclusive
    
    # Increment each bar's height based on its growth rate
    data_values[:] = [min(data + growth, 100) for data, growth in zip(data_values, growth_rates)]
    
    for bar, height in zip(bars, data_values):
        bar.set_height(height)
    return bars

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=500)

plt.show()
