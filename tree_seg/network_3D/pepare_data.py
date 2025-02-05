from scipy.ndimage import  find_objects
import torch
import numpy as np

import torch
import torch.nn.functional as F


def _extend_centers_gpu_3d(neighbors, meds, isneighbor, shape, n_iter=200,
                        device=torch.device("cuda")):
    """Runs diffusion on GPU to generate flows for training images or quality control.

    Args:
        neighbors (torch.Tensor): 9 x pixels in masks.
        meds (torch.Tensor): Mask centers.
        isneighbor (torch.Tensor): Valid neighbor boolean 9 x pixels.
        shape (tuple): Shape of the tensor.
        n_iter (int, optional): Number of iterations. Defaults to 200.
        device (torch.device, optional): Device to run the computation on. Defaults to torch.device("cuda").

    Returns:
        torch.Tensor: Generated flows.

    """
    if device is None:
        device = torch.device("cuda")

    T = torch.zeros(shape, dtype=torch.double, device=device)
    for i in range(n_iter):
        T[tuple(meds.T)] += 1
        Tneigh = T[tuple(neighbors)]
        Tneigh *= isneighbor
        T[tuple(neighbors[:, 0])] = Tneigh.mean(axis=0)
    del meds, isneighbor, Tneigh

    
    grads = T[tuple(neighbors[:,1:])]
    del neighbors
    dz = grads[0] - grads[1]
    dy = grads[2] - grads[3]
    dx = grads[4] - grads[5]
    del grads
    mu_torch = np.stack(
        (dz.cpu().squeeze(0), dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
    return mu_torch

def masks_to_flows_gpu_3d(masks, device=None):
    """Convert masks to flows using diffusion from center pixel.

    Args:
        masks (3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.

    Returns:

        mu0 (float, 4D array): Flows 

    """
    if device is None:
        device = torch.device("cuda")

    Lz0, Ly0, Lx0 = masks.shape
    #Lz, Ly, Lx = Lz0 + 2, Ly0 + 2, Lx0 + 2

    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1, 1, 1, 1, 1, 1))
    
    # get mask pixel neighbors
    z, y, x = torch.nonzero(masks_padded).T
    neighborsZ = torch.stack((z, z + 1, z - 1, z, z, z, z))
    neighborsY = torch.stack((y, y, y, y + 1, y - 1, y, y), axis=0)
    neighborsX = torch.stack((x, x, x, x, x, x + 1, x - 1), axis=0)

    neighbors = torch.stack((neighborsZ, neighborsY, neighborsX), axis=0)
    
    # get mask centers
    slices = find_objects(masks)
    if len(slices)<1:
        return  np.zeros((3, Lz0, Ly0, Lx0))
    centers = np.zeros((masks.max(), 3), "int")
    ext=[]
    for i, si in enumerate(slices):
        if si is not None:
            sz, sy, sx = si
            #lz, ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            zi, yi, xi = np.nonzero(masks[sz, sy, sx] == (i + 1))
            zi = zi.astype(np.int32) + 1  # add padding
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            zmed = np.mean(zi)
            ymed = np.mean(yi)
            xmed = np.mean(xi)
            imin = np.argmin((zi - zmed)**2 + (xi - xmed)**2 + (yi - ymed)**2)
            zmed = zi[imin]
            ymed = yi[imin]
            xmed = xi[imin]
            centers[i, 0] = zmed + sz.start
            centers[i, 1] = ymed + sy.start
            centers[i, 2] = xmed + sx.start

            ext.append([sz.stop - sz.start + 1, sy.stop - sy.start + 1, sx.stop - sx.start + 1])

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[tuple(neighbors)]
    isneighbor = neighbor_masks == neighbor_masks[0]

    ext = np.array(ext)
    n_iter = 6 * (ext.sum(axis=1)).max()

    # run diffusion
    shape = masks_padded.shape
    mu = _extend_centers_gpu_3d(neighbors, centers, isneighbor, shape, n_iter=n_iter,
                             device=device)
    # normalize
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((3, Lz0, Ly0, Lx0))
    mu0[:, z.cpu().numpy() - 1, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

    return mu0

def calculateFlow(masks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res=masks_to_flows_gpu_3d(masks,device)

    # flow_vector_field = res.transpose(1, 2, 3, 0)
    # viewer = napari.Viewer()
    # viewer.add_labels(masks, name='3D Labels')
    # z, y, x = np.nonzero(masks)
    # origins = np.stack((z, y, x), axis=-1)
    # vectors = flow_vector_field[z, y, x]
    # vector_data = np.stack((origins, vectors), axis=1)
    # viewer.add_image(np.linalg.norm(flow_vector_field, axis=3), name='norm 3D Flow Field')
    # viewer.add_vectors(vector_data, name='3D Flow Field', edge_width=0.2, length=5, ndim=3)
    # napari.run()

    return res

def calculateNeighborConnection(mask):
    """
    Calculates neighbor connectivity in a 3D mask.
    
    Args:
        mask (torch.Tensor): 3D tensor of shape (D, H, W) representing labeled masks.
        
    Returns:
        torch.Tensor: 6-channel binary tensor (6, D, H, W) indicating connectivity.
    """
    device = mask.device
    D, H, W = mask.shape

    # Initialize empty tensor for neighbor connectivity (6 directions)
    connectivity = torch.zeros((6, D, H, W), dtype=torch.uint8, device=device)

    # Define shifts in 6 directions (z, y, x) -> (depth, height, width)
    shifts = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    # Iterate over shifts
    for i, (dz, dy, dx) in enumerate(shifts):
        shifted_mask = torch.roll(mask, shifts=(dz, dy, dx), dims=(0, 1, 2))
        
        # Check if the voxel has the same label as its shifted neighbor
        same_object = (mask == shifted_mask) & (mask > 0)  # Ignore background (0)
        
        # Set false connectivity where shifting caused out-of-bounds artifacts
        if dz == -1: same_object[0, :, :] = 0
        if dz == 1:  same_object[-1, :, :] = 0
        if dy == -1: same_object[:, 0, :] = 0
        if dy == 1:  same_object[:, -1, :] = 0
        if dx == -1: same_object[:, :, 0] = 0
        if dx == 1:  same_object[:, :, -1] = 0
        
        # Store result in the corresponding channel
        connectivity[i] = same_object

    return connectivity
