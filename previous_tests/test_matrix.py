import numpy as np

F0 = np.array([[-22.915, 0, 0],
                [0, -0.00969, 0.00264],
                [0, 0.00264, -0.00264]])

# Add another dimension (3D matrix)
F0_3d = np.array([F0, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

# how to extract the matrix
# MAT[page, row, col]

st = 1
