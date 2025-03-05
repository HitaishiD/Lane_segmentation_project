# This dictionary maps RGB colors to class labels, 
# according to the colors and classes used in the KITTI dataset
# Key = (R, G, B) tuple
# Value = corresponding class label

color_map = {
    (128, 64, 128): 1,    # Road
    (35,142,107): 2,      # Sidewalk
    (70, 70, 70): 3,      # Building
    (60,20,220): 4,       # Wall
    (153, 153, 153): 5,   # Fence
    (153, 153, 190): 6,   # Vegetation
    (0,220,220): 7,       # Terrain
    (142,0,0): 8,         # Sky
    (100, 100, 150): 9,   # Person
    (152, 251, 152): 10,  # Car
    (180, 130, 70): 11,   # Bicycle
    (232, 35, 244): 12,   # Motorcycle
    (0,0,0): 0,           # Background
}