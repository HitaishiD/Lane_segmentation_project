import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Your Data Goes Here ---
# Put your list of 0s and 1s here
data_list =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
interval_sec = 2  # Each value lasts for 5 seconds
# -----------------------------

# 1. Prepare the data for a step plot
# We need time points for the *start* of each interval
# We add one extra time point for the end of the last interval
x_times = np.arange(0, (len(data_list) + 1) * interval_sec, interval_sec)

# For a step plot, we need to repeat the last value to make it draw
# the final horizontal line correctly.
y_values = np.append(data_list, data_list[-1])

# 2. Set up the plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, x_times[-1])  # Set X-axis to the total time
ax.set_ylim(-0.2, 1.2)      # Set Y-axis for 0 and 1
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Overtaking")
ax.set_title("Overtaking status over time")

# Set Y-ticks to only 0 and 1
ax.set_yticks([0, 1])
# Add horizontal grid lines (as you requested)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
# Add vertical grid lines for each interval
ax.set_xticks(x_times)
ax.grid(True, axis='x', linestyle=':', alpha=0.5)

# Initialize an empty line object. 'where='post'' creates the step plot.
line, = ax.step([], [], where='post', linewidth=2.5)

# 3. Define the animation function
def update(frame):
    """
    This function is called for each frame of the animation.
    'frame' will go from 0 up to len(data_list) - 1.
    """
    # We need 'frame + 2' points to draw 'frame + 1' segments
    x_segment = x_times[0 : frame + 2]
    y_segment = y_values[0 : frame + 2]
    
    # Update the line data
    line.set_data(x_segment, y_segment)
    return line,

# 4. Create and save the animation
# interval=5000 means 5000 milliseconds (5 seconds) per frame
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(data_list),  # Number of frames = number of data points
    interval=5000,          # 5 seconds between frames
    blit=True,              # Optimizes rendering
    repeat=False            # Don't loop the animation in the file
)

# Save the animation as a GIF
# The 'writer='pillow'' part requires the 'pillow' library
ani.save("signal_animation.gif", writer='pillow', fps=1/interval_sec)

print("✅ Animation saved as 'signal_animation.gif'")

# Uncomment the line below if you want to see the animation pop up
# plt.show()