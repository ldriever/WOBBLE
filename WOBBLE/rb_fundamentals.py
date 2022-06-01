import numpy as np
from .ma_fundamentals import MAFundamentals
from scipy.spatial.transform import Rotation as rot
from abc import ABC, abstractmethod
from .ma_exceptions import *

class RBFundamentals(MAFundamentals, ABC):
    def __init__(self, object_name, dimensions, timestep, T, **kwargs):
        ref = kwargs.pop('rigid_body_motion', None)
        del ref
        super().__init__(object_name, dimensions, rigid_body_motion=True, **kwargs)

        self.timestep = timestep
        self.T = T

        self.r_cm = None
        self.r_cm_0 = np.zeros(3)
        self.r_dot_cm = None
        self.r_dot_cm_0 = np.zeros(3)
        self.angle_mtrx = None
        self.angle_mtrx_0 = np.eye(3)
        self.angle_vect = None
        self.rb_omega = None
        self.rb_omega_0 = np.zeros(3)
        self.rb_alpha = None
        self.total_displacement_vectors = None
        self.cm = None
        self.cm_repeated = None
        self.m_lumped = None
        self.moi = None
        self.m_tot = None
        self.total_force = None
        self.pe = None
        self.ke = None

        self.time_array = np.arange(0, self.T+self.timestep, self.timestep)
        
    def solve(self, get_rotation_angles=False, save_physical_displacement=False, **kwargs):
        self.set_up(**kwargs)
        self.set_initial_conditions(**kwargs)
        self.handle_forces(**kwargs)
        self.solve_system(**kwargs)
        self.get_total_displacement_vectors(**kwargs)
        
        if get_rotation_angles:
            self.get_rotation_angles(**kwargs)
        
        if save_physical_displacement:
            self.save_physical_displacement(**kwargs)

    def set_up(self, m_lumped=None, cm=None, **kwargs):
        self.initialize_model(**kwargs)
        self.assemble_stiffness(**kwargs)
        self.assemble_mass(**kwargs)
        
        if m_lumped is not None:
            self.m_lumped = m_lumped
        else:
            self.m_lumped = np.array(np.sum(self.m_star[::3, :], axis=1)).flatten()
        self.m_tot = np.sum(self.m_lumped)

        if cm is not None:
            self.cm = cm
        else:
            self.find_cm()
        self.cm_repeated = np.tile(self.cm, self.mesh_num_nodes)

    def set_initial_conditions(self, r_0=None, r_dot_0=None, angle_mtrx_0=None, rb_omega_0=None, **kwargs):
        if r_0 is not None:
            self.r_cm_0 = r_0
        elif self.r_cm_0 is None:
            self.r_cm_0 = self.cm

        if r_dot_0 is not None:
            self.r_dot_cm_0 = r_dot_0

        if angle_mtrx_0 is not None:
            self.angle_mtrx_0 = angle_mtrx_0

        if rb_omega_0 is not None:
            self.rb_omega_0 = rb_omega_0

        self.project_initial_displacement(**kwargs)
        self.project_initial_velocity(**kwargs)

    def handle_forces(self, force_path=None, **kwargs):
        if force_path is None:
            force_path = self.force_path
        container = np.loadtxt(force_path)
        self.force_times = container[0, :]
        self.unprojected_force = container[1:, :]
        self.total_force = np.zeros((self.ndim, len(self.time_array)))
        
        if self.boundary_mask is None:
            num_considered_modes = self.mesh_num_nodes
        else:
            num_considered_modes = self.boundary_mask.sum() // 3
        
        for i in range(len(self.force_times) + 1):
            if i == 0:
                lower = 0
                total_force = 0
            else:
                lower = self.force_times[i - 1]
                total_force = self.unprojected_force[:, i-1].reshape(-1, 3).sum(axis=0).reshape(-1, 1)

            if i == len(self.force_times):
                higher = self.T
            else:
                higher = self.force_times[i]

            self.total_force[:, (lower <= self.time_array) & (self.time_array <= higher)] = total_force

        self.project_force(unprojected_force_array=self.unprojected_force, force_times=self.force_times, **kwargs)
        
        if len(self.projected_initial_disp) != self.num_modes:
            self.projected_initial_disp = self.projected_initial_disp[:self.num_modes]
            self.projected_initial_vel = self.projected_initial_vel[:self.num_modes]
            

    def evolve_position(self, **kwargs):

        self.r_cm = np.zeros((self.ndim, len(self.time_array)))
        self.r_cm[:, 0] = self.r_cm_0
        self.r_dot_cm = np.zeros((self.ndim, len(self.time_array)))
        self.r_dot_cm[:, 0] = self.r_dot_cm_0

        force_in_inertial = []
        for t in range(len(self.time_array) - 1):
            force_in_inertial.append(self.rotate(self.angle_mtrx[:, :, t], self.total_force[:, t])) 
            
        force_in_inertial = np.array(force_in_inertial).T
        dt = self.time_array[1:] - self.time_array[:-1]
        acceleration = force_in_inertial / self.m_tot
        vel_increments = acceleration * dt
        self.storage1 = force_in_inertial
        self.r_dot_cm[:, 1:] = np.cumsum(vel_increments, axis=1)

        pos_acceleration_increments = .5 * acceleration * np.power(dt, 2)

        self.r_cm[:, 1:] = np.cumsum(pos_acceleration_increments, axis=1) + np.cumsum(self.r_dot_cm[:, :-1] * dt, axis=1)

    @abstractmethod
    def solve_system(self):
        # function should give self.displacement_vectors, self.r_cm, and self.angle_vec
        ...

    def get_total_displacement_vectors(self, scale_factor=1, **kwargs):
        self.total_displacement_vectors = []
    
        if scale_factor == 'auto':
            if np.max(self.displacement_vectors) == 0:
                scale_factor = 0
            else:
                scale_factor = np.max(self.mesh_nodes) / np.max(self.displacement_vectors) / 10
        

        if self.boundary_mask is None:
            relative_displacements = self.displacement_vectors * scale_factor + self.mesh_nodes.flatten().reshape(-1, 1) - self.cm_repeated.reshape(-1,1)
            num_nodes = self.mesh_num_nodes
            offset = - self.mesh_nodes.flatten() + self.cm_repeated
        
        else:
            relative_displacements = self.displacement_vectors * scale_factor + (self.mesh_nodes.flatten().reshape(-1, 1) - self.cm_repeated.reshape(-1,1))[self.boundary_mask]
            num_nodes =  self.boundary_mask.sum() // 3
            offset = (- self.mesh_nodes.flatten() + self.cm_repeated)[self.boundary_mask]
        
        for t in range(len(self.time_array)):
            self.total_displacement_vectors.append(np.tile(self.r_cm[:, t], num_nodes) + self.rotate(self.angle_mtrx[:, :, t], relative_displacements[:, t]) + offset)

        self.total_displacement_vectors = np.array(self.total_displacement_vectors).T

    def save_physical_displacement(self, file_base_name=None, freq=1, **kwargs):
        if self.total_displacement_vectors is None:
            self.get_total_displacement_vectors(**kwargs)

        self.node_displacement = self.model.getDisplacement()
        self.model.addDumpFieldVector("displacement")

        if file_base_name:
            self.model.setBaseName(file_base_name)
        else:
            self.model.setBaseName(f"{self.name}_total_motion")

        for t in range(0, len(self.time_array), freq):
            self.node_displacement.reshape(self.mesh_num_nodes * self.ndim)[
                self.blocked_dof_mask] = self.total_displacement_vectors[:, t]
            self.model.dump()

    def get_rotation_angles(self, **kwargs):
        if self.angle_mtrx is None:
            raise SolutionError('There are no body rotation matrices to convert. Please first solve the system')

        self.angle_vect = np.zeros((3, len(self.time_array)))
        for i in range(len(self.time_array)):
            self.angle_vect[:, i] = rot.from_matrix(self.angle_mtrx[:, :, i]).as_rotvec()

    def get_energy(self, **kwargs):

        if self.vib_ke is None or self.vib_pe is None:
            self.get_vib_energy(**kwargs)

        self.pe = self.vib_pe

        self.ke = self.vib_ke

        moi_dim = len(self.moi.shape)
        for i in range(len(self.time_array)):
            if moi_dim == 3:
                moi = self.moi[:, :, i]

            else:
                moi = self.moi

            self.ke[i] += .5 * self.m_tot * (self.r_dot_cm[:, i].T @ self.r_dot_cm[:, i]) + .5* self.rb_omega[:, i].T @ moi @ self.rb_omega[:, i]

    def find_cm(self):
        self.cm = (self.m_lumped@self.mesh_nodes)/(self.m_lumped.sum())
    
    def find_moi(self, node_positions):
        moi = np.zeros([3,3])
        rel_pos = (node_positions - self.cm)

        moi[0,0] = (rel_pos[:,1]**2 + rel_pos[:,2]**2)@self.m_lumped
        moi[1,1] = (rel_pos[:,0]**2 + rel_pos[:,2]**2)@self.m_lumped
        moi[2,2] = (rel_pos[:,1]**2 + rel_pos[:,0]**2)@self.m_lumped
        moi[0,1] = -(rel_pos[:,0]*rel_pos[:,1])@self.m_lumped
        moi[0,2] = -(rel_pos[:,0]*rel_pos[:,2])@self.m_lumped
        moi[1,2] = -(rel_pos[:,1]*rel_pos[:,2])@self.m_lumped
        moi[1,0] = moi[0,1]
        moi[2,1] = moi[1,2]
        moi[2,0] = moi[0,2]

        return moi, np.linalg.inv(moi)

    def get_angular_acceleration(self, torque, moi_inv, omega):

        return moi_inv @ torque - np.cross(moi_inv @ omega, omega)
    
    def find_torque(self, forces, node_position):
        # Cross product position(relative to cm) x force
        product = np.zeros((int(len(forces) / self.ndim), self.ndim))

        product[:, 0] = node_position[1::3] * forces[2::3] - node_position[2::3] * forces[1::3]
        product[:, 1] = node_position[2::3] * forces[::3] - node_position[::3] * forces[2::3]
        product[:, 2] = node_position[::3] * forces[1::3] - node_position[1::3] * forces[0::3]
        
        return product.sum(axis=0)
    
    def rotate(self, angle, vect):
        # Rptation from body to inertial frame
        rotated = np.zeros(vect.shape)
        R = angle
        rotated[::3]=vect[::3]*R[0,0]+vect[1::3]*R[0,1]+vect[2::3]*R[0,2]
        rotated[1::3]=vect[::3]*R[1,0]+vect[1::3]*R[1,1]+vect[2::3]*R[1,2]
        rotated[2::3]=vect[::3]*R[2,0]+vect[1::3]*R[2,1]+vect[2::3]*R[2,2]
        return rotated