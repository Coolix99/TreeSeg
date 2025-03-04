import numpy as np
from numba import njit, prange

@njit
def find(parent, label):
    """Finds the root of the label with path compression."""
    if parent[label] != label:
        parent[label] = find(parent, parent[label])
    return parent[label]

@njit
def union(parent, rank, label1, label2):
    """Unites two labels using union by rank."""
    root1, root2 = find(parent, label1), find(parent, label2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1

@njit
def convert_to_direction_6(vx, vy, vz):
    """Converts vector field values to a directional step for 6-connectivity (3D)."""
    # Determine the axis with the largest movement
    abs_vx, abs_vy, abs_vz = abs(vx), abs(vy), abs(vz)
    
    if abs_vx >= abs_vy and abs_vx >= abs_vz:
        return (1, 0, 0) if vx > 0 else (-1, 0, 0)
    elif abs_vy >= abs_vx and abs_vy >= abs_vz:
        return (0, 1, 0) if vy > 0 else (0, -1, 0)
    else:
        return (0, 0, 1) if vz > 0 else (0, 0, -1)

@njit(parallel=False)
def initial_labeling_pass_6(mask, vector_field, labels, parent, rank):
    """Initial pass for 3D labeling using 6-connectivity."""
    for i in prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k] == 1:
                    if labels[i, j, k] == 0:
                        labels[i, j, k] = mask.shape[0] * mask.shape[1] * mask.shape[2]  # Initial label assignment

                    vx, vy, vz = vector_field[0, i, j, k], vector_field[1, i, j, k], vector_field[2, i, j, k]
                    dx, dy, dz = convert_to_direction_6(vx, vy, vz)
                    nx, ny, nz = i + dx, j + dy, k + dz

                    # Check if neighbor is within bounds and also in the mask
                    if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nz < mask.shape[2] and mask[nx, ny, nz] == 1:
                        if labels[nx, ny, nz] == 0:
                            labels[nx, ny, nz] = mask.shape[0] * mask.shape[1] * mask.shape[2]
                        label1 = i * mask.shape[2] * mask.shape[1] + j * mask.shape[2] + k
                        label2 = nx * mask.shape[2] * mask.shape[1] + ny * mask.shape[2] + nz
                        union(parent, rank, label1, label2)

@njit(parallel=True)
def resolve_labels_3D(labels, parent,shape):
    """Resolve labels in 3D after the initial parallel pass."""
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if labels[i, j, k] != 0:
                    labels[i, j, k] = find(parent, i * shape[1] * shape[2] + j * shape[2] + k) + 1

def connected_components_3D(mask, vector_field):
    """
    3D version of connected components with Union-Find.

    Parameters:
        mask (np.ndarray): Binary image (3D array).
        vector_field (np.ndarray): 3D array (3, N1, N2, N3) for vector directions.
        
    Returns:
        labels (np.ndarray): Labeled 3D image where each component has a unique label.
    """
    labels = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.int32)
    parent = np.arange(mask.shape[0] * mask.shape[1] * mask.shape[2], dtype=np.int32)
    rank = np.zeros(mask.shape[0] * mask.shape[1] * mask.shape[2], dtype=np.int32)



    initial_labeling_pass_6(mask, vector_field, labels, parent, rank)
    resolve_labels_3D(labels, parent, mask.shape)
    return labels



@njit
def trilinear_interpolation(vf, x, y, z):
    """Performs trilinear interpolation for the vector field at floating-point position (x, y, z)."""
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = min(x0 + 1, vf.shape[1] - 1), min(y0 + 1, vf.shape[2] - 1), min(z0 + 1, vf.shape[3] - 1)

    xd, yd, zd = x - x0, y - y0, z - z0

    # Interpolate vector field components separately
    interp_vector = np.zeros(3)
    for c in range(3):
        v000 = vf[c, x0, y0, z0]
        v100 = vf[c, x1, y0, z0]
        v010 = vf[c, x0, y1, z0]
        v001 = vf[c, x0, y0, z1]
        v101 = vf[c, x1, y0, z1]
        v011 = vf[c, x0, y1, z1]
        v110 = vf[c, x1, y1, z0]
        v111 = vf[c, x1, y1, z1]

        c00 = v000 * (1 - xd) + v100 * xd
        c01 = v001 * (1 - xd) + v101 * xd
        c10 = v010 * (1 - xd) + v110 * xd
        c11 = v011 * (1 - xd) + v111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        interp_vector[c] = c0 * (1 - zd) + c1 * zd

    return interp_vector


@njit(parallel=True)
def euler_connected_components(mask, vector_field, labels, parent, rank, step_size=0.3, N_steps=30):
    """Euler integration to determine connections in the 3D mask."""
    for i in prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k] == 1:
                    # Assign an initial unique label if not already set
                    if labels[i, j, k] == 0:
                        labels[i, j, k] = mask.shape[0] * mask.shape[1] * mask.shape[2]

                    x, y, z = float(i), float(j), float(k)

                    # Perform Euler integration along the flow
                    for _ in range(N_steps):
                        vx, vy, vz = trilinear_interpolation(vector_field, x, y, z)
                        x += step_size * vx
                        y += step_size * vy
                        z += step_size * vz

                        # Ensure position is inside bounds
                        x = max(0, min(mask.shape[0] - 1, x))
                        y = max(0, min(mask.shape[1] - 1, y))
                        z = max(0, min(mask.shape[2] - 1, z))

                    # Round and clip the final position
                    nx = min(mask.shape[0] - 1, max(0, int(round(x))))
                    ny = min(mask.shape[1] - 1, max(0, int(round(y))))
                    nz = min(mask.shape[2] - 1, max(0, int(round(z))))
                    
                    # Ensure the final pixel is in the mask
                    if mask[nx, ny, nz] == 1:
                        if labels[nx, ny, nz] == 0:
                            labels[nx, ny, nz] = mask.shape[0] * mask.shape[1] * mask.shape[2]

                        label1 = i * mask.shape[2] * mask.shape[1] + j * mask.shape[2] + k
                        label2 = nx * mask.shape[2] * mask.shape[1] + ny * mask.shape[2] + nz
                        
                        union(parent, rank, label1, label2)



def euler_connected_components_3D(mask, vector_field, step_size=0.3, N_steps=30):
    """
    3D connected components using Euler integration along the flow field.

    Parameters:
        mask (np.ndarray): Binary 3D image (shape: [D, H, W]).
        vector_field (np.ndarray): 3D flow field (shape: [3, D, H, W]).
        step_size (float): Step size for Euler integration.
        N_steps (int): Number of Euler integration steps.

    Returns:
        labels (np.ndarray): Labeled 3D image where each component has a unique label.
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    parent = np.arange(mask.shape[0] * mask.shape[1] * mask.shape[2], dtype=np.int32)
    rank = np.zeros(mask.shape[0] * mask.shape[1] * mask.shape[2], dtype=np.int32)

    euler_connected_components(mask, vector_field,labels, parent, rank, step_size, N_steps)
    resolve_labels_3D(labels, parent, mask.shape)
    
    return labels
