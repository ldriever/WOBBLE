import numpy as np
import akantu as aka

mesh_file = 'mesh_files/beam_lcr_0_4.msh'

save_path = 'force_files/case_study_lcr_0_4.txt'
header = 'Impact on front upper left middle of 100 N in x, 10 N in z for 10 ms'# with displacment in x, then y, then x'

footer = "force_times = [0.0, 1e-2], x=0, 0.3 < y, 4.1 < z"

mesh = aka.Mesh(3)
mesh.read(mesh_file)
mesh_num_nodes = mesh.getNbNodes()
mesh_nodes = mesh.getNodes()

B_mask = (mesh_nodes[:, 0] == 0) | (mesh_nodes[:, 0] == 1) | (mesh_nodes[:, 1] == 0) | (mesh_nodes[:, 1] == 1) | (mesh_nodes[:, 2] == 0) | (mesh_nodes[:, 2] == 5)
np.savetxt('beam_boundary_mask_lcr_0_4.txt', ((np.ones((mesh_num_nodes, 3))*B_mask.reshape(-1,1))==1).flatten())

mesh_nodes = mesh_nodes[B_mask]

# Create the fundamental masks for selecting the faces of the beam
'''
MASK BUILDING BLOCKS
mask_front = mesh_nodes[:, 0] == 0  X
mask_back = mesh_nodes[:, 0] == 1
mask_lower = mesh_nodes[:, 1] == 0  Y
mask_upper = mesh_nodes[:, 1] == 1
mask_left = mesh_nodes[:, 2] == 0   Z
mask_right = mesh_nodes[:, 2] == 5
'''

position_mask = (mesh_nodes[:, 0] == 0) & (mesh_nodes[:, 1] > 0.3) & (mesh_nodes[:, 2] > 4.1)

N_forced_nodes = np.sum(position_mask)

x_mask = [True, False, False]
y_mask = [False, True, False]
z_mask = [False, False, True]

mask_x = (position_mask.reshape(-1, 1) * x_mask).flatten()
mask_z = (position_mask.reshape(-1, 1) * z_mask).flatten()

# Flatten the node positions into a column vector
mesh_nodes = mesh_nodes.reshape(len(mesh_nodes) * 3)

# Set the times at which forces are to apply
force_times = [0.0, 1e-2]

forces = np.zeros((len(mesh_nodes) + 1, len(force_times)))
forces[0, :] = force_times

# Specify the forces at each time step:

forces[1:, 0][mask_x] = 100 / N_forced_nodes
forces[1:, 0][mask_z] = 10 / N_forced_nodes

np.savetxt(save_path, forces, delimiter=" ", header=header, footer=footer)
