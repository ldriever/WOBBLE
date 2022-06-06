"""
This file contains a class implementing a coupled rigid body and modal analysis solver
"""

__author__ = "Oisín Morrison, Leonhard Driever"
__credits__ = [
    "Oisín Morrison <oisin.morrison@epfl.ch>",
    "Leonhard Driever <leonhard.driever@epfl.ch>"
]

import numpy as np
from .rb_fundamentals import RBFundamentals
from .ma_exceptions import *

from scipy.spatial.transform import Rotation as rot
from .ma_fundamentals import get_alpha_beta


class CoupledRB(RBFundamentals):
    """
    Class implementing the CoupledRB algorithm. Inherits from RBFundamentals.
    
    Methods:
        get_f_fict
    
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
        self.f_fict=None
        
    def get_f_fict(self, rb_alpha, rb_omega, t):
        """
        Computes the fictitious forces given a timestep, angular acceleration and angular velocity.
        :param rb_alpha:   angular acceleration of body
        :param rb_omega:   angular velocity of body
        :param t:   time in simulation
        """
        #compute f_fict with
        # f_fict = −m α×b − 2 m Ω×b − m Ω×(Ω×b)
        disp = self.mesh_nodes.flatten()-self.cm_repeated+self.displacement_vectors[:, t]
        disp=disp.reshape([self.mesh_num_nodes, 3])
        m_extended = (self.m_lumped*np.ones([3, self.mesh_num_nodes])).T
        return (m_extended*(-np.cross(rb_alpha, disp)-2*np.cross(rb_omega, self.velocity_vectors[:, t].reshape([self.mesh_num_nodes, 3])) -np.cross(rb_omega, np.cross(rb_omega, disp)))).flatten()

    def solve_system(self):
        """
        Solves the coupled system using the CoupledRB algorithm.
        """
        if self.time_array is None:
            raise SolutionError('No time array at which to calculate the solution.\nSet a time array for self.time_array or pass a suitable array as argument to this function.')
        
        #set up variables
        self.angle_mtrx = np.zeros((3, 3, len(self.time_array)))
        self.angle_mtrx[:, :, 0] = self.angle_mtrx_0
        self.rb_omega = np.zeros((self.ndim, len(self.time_array)))

        self.rb_alpha = np.zeros((self.ndim, len(self.time_array)))
        self.rb_omega[:, 0] = self.rb_omega_0

        self.r=np.zeros((self.num_modes, len(self.time_array)))
        self.r_dot=np.zeros((self.num_modes, len(self.time_array)))
        
        self.displacement_vectors=np.zeros([3*self.mesh_num_nodes, len(self.time_array)])
        self.velocity_vectors=np.zeros([3*self.mesh_num_nodes, len(self.time_array)])
        
        self.f_fict=np.zeros([self.mesh_num_nodes*3, len(self.time_array)])
        self.f=np.zeros([self.mesh_num_nodes*3, len(self.time_array)])

        force = np.zeros([self.mesh_num_nodes*3, len(self.time_array)])
                
        time_shift = 1
        if self.force_times[0] == 0:
            time_shift = 0

        self.alphas = np.zeros((self.num_modes, len(self.time_array) + time_shift))
        self.betas = np.zeros((self.num_modes, len(self.time_array) + time_shift))
        self.offset = np.zeros((self.num_modes, len(self.time_array) + time_shift))
        
        #deal with situation where forcing doesn't start at 0
        if time_shift == 1:
            self.force_times = np.hstack(([0], self.force_times))
            self.projected_force = np.hstack((np.zeros([self.num_modes, 1]), self.projected_force))
            if self.unprojected_force is not None:
                self.unprojected_force = np.hstack((np.zeros([len(self.unprojected_force), 1]), self.unprojected_force))
        
        # compute modal analysis for first step
        magnitude_0 = np.sqrt(np.power(self.projected_initial_vel[:] / self.omega[:], 2) + np.power(self.projected_initial_disp[:], 2))
        if 0 in magnitude_0:
            angle_0 = np.zeros(self.num_modes)
            mask = magnitude_0 != 0
            angle_0[mask] = np.arcsin(self.projected_initial_disp[:][mask] / magnitude_0[mask])
        else:
            angle_0 = np.arcsin(self.projected_initial_disp[:] / magnitude_0)

        if time_shift == 0:
            magnitude_0, angle_0 = get_alpha_beta(magnitude_0, -self.projected_force[:, 0] / self.eigenvalues[:], angle_0, np.zeros(self.num_modes))  # theta is 0 as this is only executed if the first t'=0
            self.offset[:, 0] = self.projected_force[:, 0] / self.eigenvalues[:]
                    
        self.moi=np.zeros([3,3,len(self.time_array)])
        #make force for each time in time_array
        for i in range(len(self.force_times) + 1):
            if i == 0:
                lower = 0
            else:
                lower = self.force_times[i - 1]

            if i == len(self.force_times):
                higher = self.T
            else:
                higher = self.force_times[i]
            force[:, (lower <= self.time_array) & (self.time_array <= higher)] = self.unprojected_force[:, i - 1].reshape(-1, 1)
                
        self.alphas[:, 0] = magnitude_0
        self.betas[:, 0] = angle_0
        
        self.moi[:, :, 0], self.moi_inv = self.find_moi(self.mesh_nodes - self.displacement_vectors[:, 0].reshape(self.mesh_num_nodes, 3))

        #apply CoupledRB algorithm
        for t in range(1,len(self.time_array)):
            
            #do RBM stuff
            torque = self.find_torque(force[:, t], self.mesh_nodes.flatten()-self.cm_repeated+self.displacement_vectors[:, t-1])

            self.rb_alpha[:,t] = self.get_angular_acceleration(torque, self.moi_inv, self.rb_omega[:, t-1])
            self.rb_omega[:, t] = self.rb_omega[:, t - 1] + self.rb_alpha[:,t] * self.timestep
            angle_increment = self.rb_omega[:, t - 1] * self.timestep + .5 * self.rb_alpha[:,t] * self.timestep ** 2
            self.angle_mtrx[:, :, t] = rot.from_rotvec(angle_increment).as_matrix() @ self.angle_mtrx[:, :, t - 1]
                
            self.f_fict[:, t] = self.get_f_fict(self.rb_alpha[:,t], self.rb_omega[:, t], t-1)
            
            #do MA stuff
            if t==1:
                f_prev=force[:,t-1]
            else:
                f_prev=force[:,t-1]+self.f_fict[:,t-1]
            self.f[:,t]=self.f_fict[:,t]+force[:,t]
            delta_f=(self.eigenvectors.T)@(self.f[:,t]-f_prev)
                
            self.alphas[:, t], self.betas[:, t] = get_alpha_beta(self.alphas[:, t - 1], -delta_f / self.eigenvalues[:], self.betas[:, t - 1], -self.omega[:] * self.time_array[t])
            self.offset[:, t] = self.eigenvectors.T@self.f[:,t] / self.eigenvalues[:]
            
            self.r[:, t] = (self.alphas[:, t].reshape(-1, 1) * np.sin(self.omega.reshape(-1, 1) * self.time_array[t] + self.betas[:, t].reshape(-1, 1)) + self.offset[:, t].reshape(-1, 1)).flatten()
            
            self.r_dot[:, t] = (self.alphas[:, t].reshape(-1, 1) * self.omega.reshape(-1, 1) * np.cos(self.omega.reshape(-1, 1) * self.time_array[t] + self.betas[:, t].reshape(-1, 1))).flatten()
            
            self.displacement_vectors[:, t]=(self.eigenvectors @ self.r[:, t]).flatten()
            self.velocity_vectors[:, t]=(self.eigenvectors @ self.r_dot[:, t]).flatten()
            
            self.moi[:, :, t], self.moi_inv = self.find_moi(self.mesh_nodes - self.displacement_vectors[:, t].reshape(self.mesh_num_nodes, 3))
            
        self.evolve_position()
