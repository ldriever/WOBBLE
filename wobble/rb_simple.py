import numpy as np
from .rb_fundamentals import RBFundamentals
from .ma_exceptions import *

from scipy.spatial.transform import Rotation as rot


class SimpleRB(RBFundamentals):
    """
    Class implementing the SimpleRB algorithm. Inherits from RBFundamentals.
    
    Methods:
        solve_system
    """
    def __init__(self,
                 object_name,
                 dimensions,
                 timestep,
                 T,
                 **kwargs):
        ref = kwargs.pop('rigid_body_motion', None)
        del ref
        super().__init__(object_name, dimensions, timestep, T, rigid_body_motion=True, **kwargs)

        self.moi = None
        self.moi_inv = None

    def solve_system(self, **kwargs):
        """
        Solves the coupled system using the SimpleRB algorithm.
        """
        self.moi, self.moi_inv = self.find_moi(self.mesh_nodes)
        
        # Doing the modal analysis part of the solution
        # can be done entirely separately of RBM
        # due to SimpleRB decoupling assumption
        self.solve_step_loading(**kwargs)
        self.get_r_and_r_dot(**kwargs)
        self.get_displacement_vectors(**kwargs)
        self.get_velocity_vectors(**kwargs)
        
        # Doing the rigid body rotation
        self.angle_mtrx = np.zeros((3, 3, len(self.time_array)))
        self.angle_mtrx[:, :, 0] = self.angle_mtrx_0
        self.rb_omega = np.zeros((self.ndim, len(self.time_array)))
        self.rb_alpha = np.zeros((self.ndim, len(self.time_array)))
        self.rb_omega[:, 0] = self.rb_omega_0

        torque = np.zeros((self.ndim, len(self.time_array)))
        if self.boundary_mask is None:
            relative_node_positions = self.mesh_nodes.flatten() - self.cm_repeated
        else:
            relative_node_positions = (self.mesh_nodes.flatten() - self.cm_repeated)[self.boundary_mask]

        for i in range(len(self.force_times) + 1):
            if i == 0:
                lower = 0
                local_torque = np.zeros(3)
            else:
                lower = self.force_times[i - 1]
                local_torque = self.find_torque(self.unprojected_force[:, i - 1], relative_node_positions)

            if i == len(self.force_times):
                higher = self.T
            else:
                higher = self.force_times[i]

            torque[:, (lower <= self.time_array) & (self.time_array <= higher)] = local_torque.reshape(-1, 1)

        for t in range(1, len(self.time_array)):
            # If the timestep is not constant, compute it here
            self.rb_alpha[:, t] = self.get_angular_acceleration(torque[:, t], self.moi_inv, self.rb_omega[:, t-1])

            self.rb_omega[:, t] = self.rb_omega[:, t - 1] + self.rb_alpha[:, t] * self.timestep
            
            angle_increment = self.rb_omega[:, t - 1] * self.timestep + .5 * self.rb_alpha[:, t] * self.timestep ** 2
            self.angle_mtrx[:, :, t] = rot.from_rotvec(angle_increment).as_matrix() @ self.angle_mtrx[:, :, t - 1]

        self.evolve_position()
