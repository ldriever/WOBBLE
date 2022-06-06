"""
This file contains a class designed to deal with rigid body mechanics.
It is built on top of the modal analysis solver.
"""

__author__ = "Oisín Morrison, Leonhard Driever"
__credits__ = [
    "Oisín Morrison <oisin.morrison@epfl.ch>",
    "Leonhard Driever <leonhard.driever@epfl.ch>"
]

import numpy as np
from .ma_fundamentals import MAFundamentals
from scipy.spatial.transform import Rotation as rot
from abc import ABC, abstractmethod
from .ma_exceptions import *

class RBFundamentals(MAFundamentals, ABC):
    """
    Abstract class implementing main methods required for RB solvers. Inherits from MAFundamentals.
    
    Methods:
        solve
        
        set_up
        
        set_initial_conditions
        
        handle_forces
        
        evolve_position
        
        solve_system
        
        get_total_displacement_vectors
        
        save_physical_displacement
        
        get_rotation_angles
        
        get_energy
        
        find_cm
        
        find_moi
        
        get_angular_acceleration
        
        find_torque
        
        rotate
        
        get_decoupling_region
    """
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
        """
        Shorthand solve function which sets up the system and calls a solve_system method to solve it.
        Solve system is implemented either in SimpleRB or CoupledRB classes.
        """
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
        """
        Sets up the system for solving by creating the mesh and computing the eigenmodes.
        """
        self.initialize_model(**kwargs)
        self.assemble_stiffness(**kwargs)
        self.assemble_mass(**kwargs)
        
        #get lumped mass matrix
        if m_lumped is not None:
            self.m_lumped = m_lumped
        else:
            self.m_lumped = np.array(np.sum(self.m_star[::3, :], axis=1)).flatten()
        self.m_tot = np.sum(self.m_lumped)

        #get centre of mass
        if cm is not None:
            self.cm = cm
        else:
            self.find_cm()
        #as a repeated vector for ease of use later
        self.cm_repeated = np.tile(self.cm, self.mesh_num_nodes)

    def set_initial_conditions(self, r_0=None, r_dot_0=None, angle_mtrx_0=None, rb_omega_0=None, **kwargs):
        """
        Sets the initial conditions for the solver.
        """
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
        """
        Computes the projected forces for the solver.
        """
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
        #make force for each time in time_array
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
        """
        Computes the new position of the body centre of mass.
        """
        self.r_cm = np.zeros((self.ndim, len(self.time_array)))
        self.r_cm[:, 0] = self.r_cm_0
        self.r_dot_cm = np.zeros((self.ndim, len(self.time_array)))
        self.r_dot_cm[:, 0] = self.r_dot_cm_0

        force_in_inertial = []
        for t in range(len(self.time_array) - 1):
            force_in_inertial.append(self.rotate(self.angle_mtrx[:, :, t], self.total_force[:, t])) 
        
        #find r_cm
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
        """
        Function giving displacement vectors, centre of mass and angle vectors.
        """
        # function should give self.displacement_vectors, self.r_cm, and self.angle_vec
        ...

    def get_total_displacement_vectors(self, scale_factor=1, **kwargs):
        """
        Obtains the total displacement of each node relative to the starting position of the body.
        :param scale_factor:   factor determining scaling of vibrational behaviour
        """
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
        """
        Saves the physical displacements.
        :param freq:   save every freq-th step
        """
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
        """
        Computes the rotation angles from the rotation matrix using a logmap.
        Note: bugs can occur as angles approach pi.
        """
        if self.angle_mtrx is None:
            raise SolutionError('There are no body rotation matrices to convert. Please first solve the system')

        self.angle_vect = np.zeros((3, len(self.time_array)))
        for i in range(len(self.time_array)):
            #method from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
            self.angle_vect[:, i] = rot.from_matrix(self.angle_mtrx[:, :, i]).as_rotvec()

    def get_energy(self, **kwargs):
        """
        Obtains the total energies (vibrational+inertial energies)
        """
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
            #add rotational energy to KE: 0.5 omega^T I_hat omega
            self.ke[i] += .5 * self.m_tot * (self.r_dot_cm[:, i].T @ self.r_dot_cm[:, i]) + .5* self.rb_omega[:, i].T @ moi @ self.rb_omega[:, i]

    def find_cm(self):
        """
        Computes the centre of mass of the body.
        """
        self.cm = (self.m_lumped@self.mesh_nodes)/(self.m_lumped.sum())
    
    def find_moi(self, node_positions):
        """
        Computes the moment of inertia of the body.
        :param node_positions:   positions of nodes in body
        """
        moi = np.zeros([3,3])
        rel_pos = (node_positions - self.cm)
        #find moment of inertia in discrete case
        #standard formulae e.g. from https://farside.ph.utexas.edu/teaching/336k/Newton/node64.html
        moi[0,0] = (rel_pos[:,1]**2 + rel_pos[:,2]**2)@self.m_lumped
        moi[1,1] = (rel_pos[:,0]**2 + rel_pos[:,2]**2)@self.m_lumped
        moi[2,2] = (rel_pos[:,1]**2 + rel_pos[:,0]**2)@self.m_lumped
        moi[0,1] = -(rel_pos[:,0]*rel_pos[:,1])@self.m_lumped
        moi[0,2] = -(rel_pos[:,0]*rel_pos[:,2])@self.m_lumped
        moi[1,2] = -(rel_pos[:,1]*rel_pos[:,2])@self.m_lumped
        #moi is symmetric
        moi[1,0] = moi[0,1]
        moi[2,1] = moi[1,2]
        moi[2,0] = moi[0,2]

        return moi, np.linalg.inv(moi)

    def get_angular_acceleration(self, torque, moi_inv, omega):
        """
        Computes the angular acceleration of the body.
        :param torque:   torque acting on the body
        :param moi_inv:   inverse of the moment of inertia
        :param omega:   angular velocity
        """
        return moi_inv @ torque - np.cross(moi_inv @ omega, omega)
    
    def find_torque(self, forces, node_position):
        """
        Computes the torque acting on the body.
        :param forces:   forces acting on the body
        :param node_position:   positions of nodes
        """
        # Cross product position(relative to cm) x force
        product = np.zeros((int(len(forces) / self.ndim), self.ndim))

        product[:, 0] = node_position[1::3] * forces[2::3] - node_position[2::3] * forces[1::3]
        product[:, 1] = node_position[2::3] * forces[::3] - node_position[::3] * forces[2::3]
        product[:, 2] = node_position[::3] * forces[1::3] - node_position[1::3] * forces[0::3]
        
        return product.sum(axis=0)
    
    def rotate(self, angle, vect):
        """
        Rotates a vector about an angle.
        :param angle:   angle matrix to rotate about
        :param vect:   vector to rotate
        """
        # Rotation from body to inertial frame
        # NB: this process is not commutative
        # normal scalar approximations e.g. Forward Euler
        # can not be applied
        # Instead, we use Rodrigues formula
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        rotated = np.zeros(vect.shape)
        R = angle
        rotated[::3]=vect[::3]*R[0,0]+vect[1::3]*R[0,1]+vect[2::3]*R[0,2]
        rotated[1::3]=vect[::3]*R[1,0]+vect[1::3]*R[1,1]+vect[2::3]*R[1,2]
        rotated[2::3]=vect[::3]*R[2,0]+vect[1::3]*R[2,1]+vect[2::3]*R[2,2]
        return rotated
    
    def get_decoupling_region(self, d_min=5*10**-2, d_max=10**-6, f_min=None, f_max=None, save=False, **kwargs):
        """
        Plots the regions suitable for SimpleRB and CoupledRB simulations
        :param d_min:   maximum displacement allowed for SimpleRB
        :param d_max:   maximum displacement allowed for CoupledRB
        :param f_min:   maximum force allowed for SimpleRB
        :param f_max:   maximum force allowed for CoupleRB
        """
        #get values of bounding curves
        if (f_min is not None) and (f_max is not None):
            denom = np.max((np.ones((ma.mesh_num_nodes, 3)) * (np.linalg.norm(ma.mesh_nodes-ma.cm, axis=1) * ma.m_lumped).reshape(-1, 1)).flatten())
            K1 = f_min/denom
            K2 = f_max/denom
        elif (d_min is not None) and (d_max is not None):
            bm=(np.ones((ma.mesh_num_nodes, 3)) * (np.linalg.norm(ma.mesh_nodes-ma.cm, axis=1) * ma.m_lumped).reshape(-1, 1)).flatten()
            denom = np.max((ma.eigenvectors @ ((ma.eigenvectors.T @ bm ) / ma.eigenvalues)) )
            K1 = d_min/denom
            K2 = d_max/denom
        else:
            raise ValueError("Did not pass in displacements or force limits")
        
        #make the plot
        x = np.logspace(-2, 4, 1000)
        y1 = K1 - x**2
        y2 = K2 - x**2

        fig,ax=plt.subplots(figsize=(7,7))
        ax.fill_between(x, y1, color='green', label=r"SimpleRB Region")
        ax.fill_between(x, y1, y2, color='yellow', label=r"CoupledRB Region")
        ax.fill_between(x, y2, K2*100000, color='red', label=r"Nonlinear Region")
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.legend()
        plt.ylim(10**-2.1,10**9)
        ax.set_ylabel(r'$|\alpha|$')
        ax.set_xlabel(r'$|\Omega|$')
        if save:
            fig.savefig('neglection_region.png', bbox_inches='tight')
        plt.show()