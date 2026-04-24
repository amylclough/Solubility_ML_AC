import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 1. Defining the data and boundaries
X_MIN = -12
X_MAX = 2
BOUNDARY_TICKS = [X_MIN, -4, -2, 0, X_MAX]
# Position for the labels above the bar, relative to the axis transform 
LABEL_Y_POS = 1.08 

# Solubility classes and their boundary points
solubility_classes = [
    ("Insoluble", -12, -4),
    ("Slightly Soluble", -4, -2),
    ("Soluble", -2, 0),
    ("Highly Soluble", 0, 2)
]

# Defining the colors and their normalized positions (0.0 to 1.0) for a smooth gradient
colors_points = [
    (0.0, "#e74c3c"),              
    (( -4 + 12) / 14.0, "#f39c12"), 
    (( -2 + 12) / 14.0, "#3498db"), 
    (( 0 + 12) / 14.0, "#27ae60"),  
    (1.0, "#27ae60")                
]

# Creating the custom colormap
custom_cmap = LinearSegmentedColormap.from_list("solubility_gradient", colors_points)

# 2. Setting up the figure and axes

fig, ax = plt.subplots(figsize=(12, 1.8)) 


ax.set_yticks([]) # Hiding the y-axis ticks and labels

# 3. Plotting the continuous gradient bar using pcolormesh
x = np.linspace(X_MIN, X_MAX, 1000) # Fine resolution for a smooth gradient
y = np.array([0, 1])
X, Y = np.meshgrid(x, y)

# Z is the scalar field that maps to the colormap. Z is the normalized x-value.
Z = (X - X_MIN) / (X_MAX - X_MIN)
ax.pcolormesh(X, Y, Z, cmap=custom_cmap, shading='auto')

# 4. Adding the vertical boundary lines (clear lines)
for boundary in [-4, -2, 0]:
    # Ploting the vertical line from y=0 to y=1 for clarity
    ax.axvline(boundary, color='black', linewidth=1, linestyle='-', alpha=0.6)

# 5. Adding the text labels at the top

for name, lower_bound, upper_bound in solubility_classes:
    center_x = (lower_bound + upper_bound) / 2
    
    # Placing the label high above the bar using transform=ax.get_xaxis_transform()
    ax.text(center_x, LABEL_Y_POS, name,
            ha='center', va='bottom',
            fontsize=12, color='black', 
            transform=ax.get_xaxis_transform(), 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'))

# 6. Final adjustments to the axes
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(0, 1) 
ax.set_xlabel('LogS Value', fontsize=14, labelpad=15) 

# Setting the x-ticks, which include -12 and 2
ax.set_xticks(BOUNDARY_TICKS)
ax.tick_params(axis='x', labelsize=12) 

# Manually adjusting bottom margin to ensure x-label fits
plt.subplots_adjust(bottom=0.4)
plt.subplots_adjust(top=0.8)
plt.show()