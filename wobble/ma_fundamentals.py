"""
This file contains a class designed to handle and perform modal analysis of structures.
It is based in parts on the software package Akantu
"""

__author__ = "Oisín Morrison, Leonhard Driever"
__credits__ = [
    "Oisín Morrison <oisin.morrison@epfl.ch>",
    "Leonhard Driever <leonhard.driever@epfl.ch>"
]

import numpy as np
import akantu as aka
import matplotlib.pyplot as plt
import subprocess

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from abc import ABC, abstractmethod
from .ma_exceptions import *


class MAFundamentals(ABC):
    """
    Abstract class implementing the main methods required for all solvers
    
    Methods:
        create_mesh
        
        initialise_model
        
        apply_boundary_conditions
        
        assemble_stiffness
        
        assemble_mass
        
        find_eigenmodes
        
        project_force
        
        project_initial_displacement
        
        project_initial_velocity
        
        solve_step_loading
        
        save_alpha_beta_offset
        
        get_r_and_r_dot
        
        get_displacement_vectors
        
        get_velocity_vectors
        
        save_modes
        
        save_physical_vibrations
        
        get_vib_energy
        
        plot_vibrational_energy
        
    """
    def __init__(self,
                 object_name,
                 dimensions,
                 geometry_file=None,  # Not essential if a mesh file path is specified elsewhere
                 material_file=None,
                 num_modes=None,  # use 'all' to use all of the modes, if rigid_body_motion == True, it does not usually make sense for this to be less than 6
                 rigid_body_motion=False,  # can alternatively be set to 'simple' or 'coupled'. Any other words are treated like false
                 zero_mode_tolerance='auto',
                 force_path=None,
                 projected_force_path=None,  # This and all parameters above are essential. If not specified here they have to be specified in the functions
                 mesh_file=None,  # This and all parameters below are non-essential and the program can run without them being specified anywhere
                 bc_dicts=None,
                 time_array=None,
                 k_path=None,
                 k_star_path=None,
                 m_path=None,
                 m_star_path=None,
                 initial_displacement_path=None,
                 projected_initial_displacement_path=None,
                 initial_velocity_path=None,
                 projected_initial_velocity_path=None,
                 eigenmode_path=None,
                 boundary_mask=None):
        # Store input arguments:
        self.name = object_name
        self.ndim = dimensions
        self.geometry_file = geometry_file
        self.material_file = material_file
        self.num_modes = num_modes
        self.rigid_body_motion = rigid_body_motion
        self.zero_mode_tolerance = zero_mode_tolerance  # Only makes sense to specify this when rigid_body_motion == True
        self.force_path = force_path 
        self.projected_force_path = projected_force_path 
        self.mesh_file = mesh_file
        self.bc_dicts = bc_dicts
        self.time_array = time_array
        self.k_path = k_path
        self.k_star_path = k_star_path
        self.m_path = m_path
        self.m_star_path = m_star_path
        self.initial_displacement_path = initial_displacement_path
        self.projected_initial_displacement_path = projected_initial_displacement_path
        self.initial_velocity_path = initial_velocity_path
        self.projected_initial_velocity_path = projected_initial_velocity_path
        self.eigenmode_path = eigenmode_path
        self.boundary_mask = boundary_mask  # Specifying a boundary mask (Boolean array indicating which modes are on the boundary) will automatically mean that the boundary approach is used

        # Additional set up actions:
        self.mesh = None
        self.model = None
        self.mesh_num_nodes = 0
        self.mesh_nodes = None  # Position of the undisplaced nodes in the mesh
        self.blocked_dof_mask = None
        self.k_star = None
        self.m_star = None
        self.proj_matrix = None # P^T M
        self.eigenvalues = None
        self.omega = None
        self.eigenvectors = None
        self.eigenvectors_boundary = None
        self.projected_force = None
        self.unprojected_force = None
        self.force_times = None
        self.projected_initial_disp = None
        self.projected_initial_vel = None
        self.alphas = None
        self.betas = None
        self.offset = None
        self.r = None
        self.r_dot = None
        self.displacement_vectors = None
        self.velocity_vectors = None
        self.node_displacement = None
        self.node_velocity = None
        self.vib_pe = None
        self.vib_ke = None
        self.force_energy_correction = None

    # Methods for setting up and solving the system:
    def create_mesh(self, overload_geometry_file=None, mesh_save_path=None, **kwargs):
        """
        Given a geometry file, creates a mesh using gmsh. Note: mesh is of order 2.
        """
        if overload_geometry_file:
            geometry_file_path = overload_geometry_file
        elif self.geometry_file:
            geometry_file_path = self.geometry_file
        else:
            raise FileError("No geometry file specified.\nSet a file path for self.geometry_file or pass it as overload_geometry_file to this function.")

        if not mesh_save_path:
            mesh_save_path = str(self.name) + '.msh'
        
        #way to create gmsh from commandline (https://gmsh.info/doc/texinfo/gmsh.html#Command_002dline-options)
        subprocess.call(['gmsh', f'-{self.ndim}', geometry_file_path, '-order', '2', '-o', mesh_save_path])
        self.mesh_file = mesh_save_path

    def initialize_model(self, overload_mesh_file=None, overload_material_file=None, **kwargs):
        """
        Initialises the attributes associated with the mesh.
        """
        if overload_mesh_file:
            mesh_file_path = overload_mesh_file
        elif self.mesh_file:
            mesh_file_path = self.mesh_file
        else:
            self.create_mesh(**kwargs)
            mesh_file_path = self.mesh_file

        if overload_material_file:
            material_file_path = overload_material_file
        elif self.material_file:
            material_file_path = self.material_file
        else:
            raise FileError("No material file specified.\nSet a file path for self.material_file or pass it as overload_material_file to this function.")
        
        aka.parseInput(material_file_path)

        if self.mesh is None:
            self.mesh = aka.Mesh(self.ndim)
            self.mesh.read(mesh_file_path)
            self.mesh_num_nodes = self.mesh.getNbNodes()
            self.mesh_nodes = self.mesh.getNodes()

        if self.model is None:
            # initialise model with implicit dynamics model (as per akantu examples: https://gitlab.com/akantu/akantu/-/blob/master/examples/python/eigen_modes/eigen_modes.py)
            self.model = aka.SolidMechanicsModel(self.mesh)
            self.model.initFull(aka._implicit_dynamic)

        if self.num_modes == 'all':
            self.num_modes = self.ndim * self.mesh_num_nodes

    def apply_boundary_conditions(self, overload_bc_dicts=None, **kwargs):
        """
        Apply given boundary conditions (follows convention of akantu).
        
        Example boundary conditions:
        bc_dicts = [{'type': 'FixedValue', 'value': 0.0, 'axis': 'x', 'group': 'left'},
                    {'type': 'IncrementValue', 'value': 1.5, 'axis': 'y', 'group': 'top'},
                    {'type': 'FromTraction', 'traction': surface_traction, 'group': 'front'}]
        """
        if overload_bc_dicts:
            bc_dicts = overload_bc_dicts
        elif self.bc_dicts:
            bc_dicts = self.bc_dicts
        else:
            bc_dicts = [] # No boundary conditions are applied

        if len(bc_dicts) != 0 and self.rigid_body_motion:
            print('WARNING: Applying boundary conditions while computing the system rigid body motion may have unintended effects')

        if self.model is None:
            self.initialize_model(**kwargs)

        for i in range(len(bc_dicts)):
            #types of boundary conditions possible in akantu (https://akantu.gitlab.io/akantu/manual/solidmechanicsmodel.html)
            if bc_dicts[i]['type'] == 'FixedValue':
                self.model.applyBC(aka.FixedValue(bc_dicts[i]['value'], get_aka_axis(bc_dicts[i]['axis'])), bc_dicts[i]['group'])
            elif bc_dicts[i]['type'] == 'IncrementValue':
                self.model.applyBC(aka.IncrementValue(bc_dicts[i]['value'], get_aka_axis(bc_dicts[i]['axis'])), bc_dicts[i]['group'])
            elif bc_dicts[i]['type'] == 'FromTraction':
                self.model.applyBC(aka.FromTraction(bc_dicts[i]['traction']), bc_dicts[i]['group'])
            elif bc_dicts[i]['type'] == 'FromStress':
                self.model.applyBC(aka.FromStress(bc_dicts[i]['stress']), bc_dicts[i]['group'])
            else:
                raise BoundaryConditionError(f'The type of boundary condition specified in entry {i} of bc_dicts is not a valid boundary condition type.\nAllowed boundary condition types are FixedValue, IncrementValue, FromTraction and FromStress.')

        #get blocked degrees of freedom mask
        self.blocked_dof_mask = np.invert(self.model.getBlockedDOFs().flatten())

        if self.num_modes > np.sum(self.blocked_dof_mask):
            self.num_modes = np.sum(self.blocked_dof_mask)

    def assemble_stiffness(self, overload_k_path=None, overload_k_star_path=None, **kwargs):
        """
        Assembles the stiffness matrix K for the mesh.
        """
        k = None
        if overload_k_star_path:
            self.k_star = np.loadtxt(overload_k_star_path)
        elif overload_k_path:
            k = np.loadtxt(overload_k_path)
        elif self.k_star_path:
            self.k_star = np.loadtxt(self.k_star_path)
        elif self.k_path:
            k = np.loadtxt(self.k_path)
        else:
            self.model.assembleStiffnessMatrix()
            #get stiffness matrix from mesh
            k = self.model.getDOFManager().getMatrix('K')
            k = aka.AkantuSparseMatrix(k).toarray()

        if k is not None:
            if self.blocked_dof_mask is None:
                self.apply_boundary_conditions(**kwargs)
            self.k_star = csr_matrix(k[self.blocked_dof_mask, :][:, self.blocked_dof_mask].copy())

    def assemble_mass(self, overload_m_path=None, overload_m_star_path=None, **kwargs):
        """
        Assembles the mass matrix M for the mesh.
        """
        m = None
        if overload_m_star_path:
            self.m_star = np.loadtxt(overload_m_star_path)
        elif overload_m_path:
            m = np.loadtxt(overload_m_path)
        elif self.m_star_path:
            self.m_star = np.loadtxt(self.m_star_path)
        elif self.m_path:
            m = np.loadtxt(self.m_path)
        else:
            self.model.assembleMass()
            #get mass matrix from mesh
            m = self.model.getDOFManager().getMatrix('M')
            m = aka.AkantuSparseMatrix(m).toarray()

        if m is not None:
            if self.blocked_dof_mask is None:
                self.apply_boundary_conditions(**kwargs)
            self.m_star = csr_matrix(m[self.blocked_dof_mask, :][:, self.blocked_dof_mask].copy())

    def find_eigenmodes(self, eigenmode_path=None, overload_num_modes=None, save_modes=False, **kwargs):
        """
        Computes the eigenmodes of the system i.e. the eigenvalues and eigenvectors of M^-1 K. 
        
        Rigid body modes are already known, so these are explicitly included when rigid body dynamics solvers are used.
        """
        if eigenmode_path:
            self.eigenmode_path = eigenmode_path
        
        if self.eigenmode_path:
            if self.rigid_body_motion and self.num_modes > self.mesh_num_nodes * 3 - 6:
                self.num_modes = self.mesh_num_nodes * 3 - 6
            try:
                storage = np.loadtxt(self.eigenmode_path, usecols=np.arange(self.num_modes), delimiter=',')
            except:
                raise FileError('The file containing the eigenmodes could not be opened or contains less modes than desired. Lower self.num_modes or use a different file.')
                
            self.eigenvalues = storage[0]
            if self.num_modes==1:
                self.eigenvalues = np.array([self.eigenvalues])
            self.omega = np.sqrt(self.eigenvalues)
            self.eigenvectors = storage[1:]
            self.eigenvectors=self.eigenvectors.reshape(self.blocked_dof_mask.sum(), self.num_modes)

        else:
            if self.k_star is None:
                self.assemble_stiffness(**kwargs)
            if self.m_star is None:
                self.assemble_mass(**kwargs)

            if overload_num_modes:
                considered_modes = overload_num_modes
            elif self.num_modes:
                considered_modes = self.num_modes
            else:
                raise ModeError(
                    'The number of modes to be considered is not specified.\nSet the number of modes as self.num_modes or pass it as overload_num_modes to this function.')

            if considered_modes == 'all':
                considered_modes = self.mesh_num_nodes * self.ndim
                
            #rigid body modes require special attention
            if self.rigid_body_motion:
                # Depending on the number of modes with eigenvalue zero that the algorithm computes, between zero and five modes too many must be computed (one zero mode always appears to be computed)
                considered_modes = considered_modes + 6

            if considered_modes >= np.sum(self.blocked_dof_mask):
                #get all modes
                vals, vects = eigh(self.k_star.toarray(), b=self.m_star.toarray())
            else:
                #get subset of modes (slower algorithm)
                vals, vects = eigsh(self.k_star, M=self.m_star, which='SM', k=considered_modes)
                
            storage = np.vstack((np.real(vals).reshape(1, -1), np.real(vects)))
            storage = storage[:, storage[0].argsort()]

            if self.rigid_body_motion:
                #deal with zero value modes (within numerical tolerance)
                if self.zero_mode_tolerance == 'auto':
                    first_mode = np.where(np.invert(abs(storage[0, 1:] / storage[0, :-1]) < 100))[0][
                                     0] + 1
                else:
                    first_mode = np.sum(storage[0, :] < self.zero_mode_tolerance)

                # Remove the computed zero-modes and make sure to oblige to the specified number of considered modes
                storage = storage[:, first_mode : first_mode + considered_modes - 6]

            self.eigenvalues = storage[0, :]
            if self.num_modes==1:
                self.eigenvalues = np.array([self.eigenvalues])
            self.omega = np.sqrt(self.eigenvalues)
            self.eigenvectors = storage[1:, :]
            self.eigenvectors=self.eigenvectors.reshape(self.blocked_dof_mask.sum(), self.num_modes)
            
            #normalise eigenvectors so that P^T M P = I
            self.eigenvectors = self.eigenvectors / np.sqrt(
                np.diag(self.eigenvectors.T @ self.m_star @ self.eigenvectors))

            if save_modes:
                self.save_modes(**kwargs)
                
        if len(self.eigenvalues) < self.num_modes:
            self.num_modes = len(self.eigenvalues)
            
        if self.boundary_mask is not None:
            self.eigenvectors_boundary = self.eigenvectors[self.boundary_mask[self.blocked_dof_mask]]

    def project_force(self, overload_force_path=None, overload_projected_force_path=None, unprojected_force_array=None, force_array=None, force_times=None, **kwargs):
        """
        Computes the projected force by either reading from a file or projecting a physical force into modal space.
        """
        if force_array is not None and force_times is not None:
            self.projected_force = force_array
            self.force_times = force_times
            return

        # First row of file should give the corrresponding times at which the steps are activated
        projected = True
        if unprojected_force_array is not None and force_times is not None:
            container = np.vstack((force_times, unprojected_force_array))
            projected = False
        elif overload_projected_force_path:
            container = np.loadtxt(overload_projected_force_path)
        elif overload_force_path:
            container = np.loadtxt(overload_force_path)
            projected = False
        elif self.projected_force_path:
            container = np.loadtxt(self.projected_force_path)
        elif self.force_path:
            container = np.loadtxt(self.force_path)
            projected = False
        else:
            raise ModeError(
                'No force to project onto modal basis.\nSet a file path for self.projected_force_path or self.force_path or pass one of these paths as argument to this function.')

        self.force_times = container[0, :]
        
        #compute projected force F = P^T f_ext
        if projected:
            self.projected_force = container[1:, :]
        else:
            self.unprojected_force = container[1:, :]
            if self.eigenvectors is None:
                self.find_eigenmodes(**kwargs)

            if self.boundary_mask is None:
                self.projected_force = self.eigenvectors.T @ container[1:, :][self.blocked_dof_mask, :]
            else:
                self.projected_force = self.eigenvectors_boundary.T @ container[1:, :][self.blocked_dof_mask[ self.boundary_mask], :]

    def project_initial_displacement(self, overload_initial_displacement_path=None, overload_projected_initial_displacement_path=None, **kwargs):
        """
        Computes the projected initial displacement by either reading from a file or projecting the initial displacement into modal space.
        """
        projected = False
        if overload_projected_initial_displacement_path:
            self.projected_initial_disp = np.loadtxt(overload_projected_initial_displacement_path)
            projected = True
        elif overload_initial_displacement_path:
            displacement = np.loadtxt(overload_initial_displacement_path)
        elif self.projected_initial_displacement_path:
            self.projected_initial_disp = np.loadtxt(self.projected_initial_displacement_path)
            projected = True
        elif self.initial_displacement_path:
            displacement = np.loadtxt(self.initial_displacement_path)
        else:
            self.projected_initial_disp = np.zeros(self.num_modes)
            projected = True
        
        #compute r_0 = P^T M u_0
        if not projected:
            if self.eigenvectors is None:
                self.find_eigenmodes(**kwargs)

            if self.proj_matrix is None:
                self.proj_matrix = self.eigenvectors.T @ self.m_star
            self.projected_initial_disp = self.proj_matrix @ displacement

    def project_initial_velocity(self, overload_initial_velocity_path=None, overload_projected_initial_velocity_path=None, **kwargs):
        """
        Computes the projected initial velocity by either reading from a file or projecting the initial velocity into modal space.
        """
        projected = False
        if overload_projected_initial_velocity_path:
            self.projected_initial_vel = np.loadtxt(overload_projected_initial_velocity_path)
            projected = True
        elif overload_initial_velocity_path:
            velocity = np.loadtxt(overload_initial_velocity_path)
        elif self.projected_initial_velocity_path:
            self.projected_initial_vel = np.loadtxt(self.projected_initial_velocity_path)
            projected = True
        elif self.initial_velocity_path:
            velocity = np.loadtxt(self.initial_velocity_path)
        else:
            self.projected_initial_vel = np.zeros(self.num_modes)
            projected = True
            
        #compute r_dot_0 = P^T M u_dot_0
        if not projected:
            if self.eigenvectors is None:
                self.find_eigenmodes(**kwargs)
            if self.proj_matrix is None:
                self.proj_matrix = self.eigenvectors.T @ self.m_star
            self.projected_initial_vel = self.proj_matrix @ velocity

    def solve_step_loading(self, save_solution=None, **kwargs):
        """
        Computes the alphas and betas.
        """ 
        if self.projected_force is None:
            self.project_force(**kwargs)
        if self.projected_initial_disp is None:
            self.project_initial_displacement(**kwargs)
        if self.projected_initial_vel is None:
            self.project_initial_velocity(**kwargs)

        time_shift = 1
        if self.force_times[0] == 0:
            time_shift = 0

        self.alphas = np.zeros((self.num_modes, len(self.force_times) + time_shift))
        self.betas = np.zeros((self.num_modes, len(self.force_times) + time_shift))
        self.offset = np.zeros((self.num_modes, len(self.force_times) + time_shift))
        
        if time_shift == 1:
            self.force_times = np.hstack(([0], self.force_times))
            self.projected_force = np.hstack((np.zeros([self.num_modes, 1]), self.projected_force))
            if self.unprojected_force is not None:
                self.unprojected_force = np.hstack((np.zeros([len(self.unprojected_force), 1]), self.unprojected_force))

        # Compute the solution for the modes with non-zero frequency
        # The solution is of the form alpha * sin(w t + beta)

        #set initial values
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

        self.alphas[:, 0] = magnitude_0
        self.betas[:, 0] = angle_0

        #iterate and compute the next solution for the next heaviside step function
        for t in range(1, len(self.force_times)):
            delta_f = self.projected_force[:, t] - self.projected_force[:, t - 1]

            self.alphas[:, t], self.betas[:, t] = get_alpha_beta(self.alphas[:, t - 1], -delta_f / self.eigenvalues[:], self.betas[:, t - 1], -self.omega[:] * self.force_times[t])

            self.offset[:, t] = self.projected_force[:, t] / self.eigenvalues[:]

        if save_solution:
            self.save_alpha_beta_offset(**kwargs)

    def save_alpha_beta_offset(self, ab_save_path=None, **kwargs):
        """
        Function that saves the computed solution for an applied sequence of step loadings
        :param ab_save_path:   file path to which the information is to be saved. Should have mime type csv
        :return:            None
        """
        if not self.force_times or not self.alphas or not self.betas or not self.offset:
            raise ModeError('Cannot save solution to step loading because there is no such solution')

        if not ab_save_path:
            ab_save_path = self.name + '_alpha_beta_offset.csv'

        np.savetxt(ab_save_path, np.vstack((self.force_times, self.alphas, self.betas, self.offset)), delimiter=",")

    def get_r_and_r_dot(self, time_array=None, **kwargs):
        """
        Computes the modal solution.
        :param time_array: array of times to compute solution at
        """
        if time_array is not None:
            self.time_array = time_array

        if self.time_array is None:
            raise SolutionError('No time array at which to calculate the solution.\nSet a time array for self.time_array or pass a suitable array as argument to this function.')

        if self.alphas is None or self.betas is None or self.offset is None:
            self.solve_step_loading(**kwargs)

        max_index = sum(self.force_times <= self.time_array[-1])  # <= to also include the boundary case

        #construct r and r_dot from the alphas and betas
        self.r = np.zeros((self.num_modes, len(self.time_array)))  # Coordinate in modal space
        self.r_dot = np.zeros((self.num_modes, len(self.time_array)))  # time derivative of r
        for t in range(max_index):
            if t == max_index - 1:
                mask = self.force_times[t] <= self.time_array
            else:
                mask = (self.force_times[t] <= self.time_array) & (self.time_array < self.force_times[t + 1])

            ones = np.ones((self.num_modes, sum(mask)))
            self.r[:, mask] = self.alphas[:, t].reshape(-1, 1) * np.sin(self.omega[:].reshape(-1, 1) * self.time_array[mask] + self.betas[:, t].reshape(-1, 1) * ones) + self.offset[:, t].reshape(-1, 1) * ones
            self.r_dot[:, mask] = self.alphas[:, t].reshape(-1, 1) * self.omega[:].reshape(-1, 1) * np.cos(self.omega[:].reshape(-1, 1) * self.time_array[mask] + self.betas[:, t].reshape(-1, 1) * ones)

    # Methods for retrieving primary information:
    def get_displacement_vectors(self, displacement_save_path=None, **kwargs):
        """
        Computes the physical displacements from the modal solution.
        """
        if self.r is None:
            self.get_r_and_r_dot(**kwargs)
        
        #find displacements u = P r
        if self.boundary_mask is None:
            self.displacement_vectors = self.eigenvectors @ self.r
        else:
            self.displacement_vectors = self.eigenvectors_boundary @ self.r

        if displacement_save_path is not None:
            np.savetxt(displacement_save_path, self.displacement_vectors)

    def get_velocity_vectors(self, velocity_save_path=None, **kwargs):
        """
        Computes the physical velocities from the modal solution.
        """
        if self.r_dot is None:
            self.get_r_and_r_dot(**kwargs)
        
        #find displacements u_dot = P r_dot
        if self.boundary_mask is None:
            self.velocity_vectors = self.eigenvectors @ self.r_dot
        else:
            self.velocity_vectors = self.eigenvectors_boundary @ self.r_dot

        if velocity_save_path is not None:
            np.savetxt(velocity_save_path, self.velocity_vectors)

    def save_modes(self, modes_save_path=None, **kwargs):
        """
        Function that saves the computed eigenvalues and vectors to a csv file
        :param modes_save_path:   file path to which the information is to be saved. Should have mime type csv
        :return:            None
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ModeError('Cannote save modes because there are no eigenvalues or eigenvectors or both')

        if not modes_save_path:
            modes_save_path = self.name + '_modes.csv'

        np.savetxt(modes_save_path, np.vstack((self.eigenvalues.reshape(1, -1), self.eigenvectors)), delimiter=",")

    def save_physical_vibrations(self, file_base_name=None, freq=1, **kwargs):
        """
        Function that saves the physical vibrations of the system in paraview format.
        :param freq:   save every freq-th displacement
        """
        if self.displacement_vectors is None:
            self.get_displacement_vectors(**kwargs)

        self.node_displacement = self.model.getDisplacement()
        self.model.addDumpFieldVector("displacement")

        if file_base_name:
            self.model.setBaseName(file_base_name)
        else:
            self.model.setBaseName(f"{self.name}_vibrations")

        if self.boundary_mask is None:
            for t in range(0, len(self.time_array), freq):
                self.node_displacement.reshape(self.mesh_num_nodes * self.ndim)[self.blocked_dof_mask] = self.displacement_vectors[:, t]
                self.model.dump()
        else:
            print('NOTE: you are projecting the results of the boundary computations onto the full mesh. Make sure to ignore mesh nodes within the body when visualising the displacement.')
            for t in range(0, len(self.time_array), freq):
                self.node_displacement.reshape(self.mesh_num_nodes * self.ndim)[np.logical_and(self.boundary_mask, self.blocked_dof_mask)] = self.displacement_vectors[:, t]
                self.model.dump()
            
    def get_vib_energy(self, **kwargs):
        """
        Function that computes the vibrational energy of the system using the modal solution.
        """
        if self.r is None or self.r_dot is None:
            raise SolutionError('No solution in modal space (r and r_dot) that can be used to compute the vibrational energy')
        
        # equivalent to KE = 1/2 u_dot^T M u_dot
        # and PE = 1/2 u^T K u and cheaper to compute
        self.vib_ke = .5 * np.sum(self.r_dot * self.r_dot, axis=0)
        self.vib_pe = .5 * np.sum(self.r * self.r * self.eigenvalues.reshape(-1, 1), axis=0)
        
    def get_force_energy_correction(self, **kwargs):
        """
        Computes the force energy correction f_ext dot u, useful for checking the behaviour of the energy.
        """
        if self.displacement_vectors is None:
            self.get_displacement_vectors(**kwargs)
        if self.velocity_vectors is None:
            self.get_velocity_vectors(**kwargs)
        if self.time_array is None:
            raise SolutionError('No time array at which to calculate the solution.\nSet a time array for self.time_array and rerun this function')

        forces = (self.m_star @ self.eigenvectors) @ self.projected_force  # Un-projected projected force, described as forces acting on the nodes

        max_index = np.sum(self.force_times <= self.time_array[-1])  # <= to also include the boundary case
        
        #find f_ext dot u
        self.force_energy_correction = np.zeros(len(self.time_array))
        for t in range(max_index):
            if t == max_index - 1:
                mask = self.force_times[t] <= self.time_array
            else:
                mask = (self.force_times[t] <= self.time_array) & (self.time_array < self.force_times[t + 1])

            self.force_energy_correction[mask] = self.displacement_vectors[:, mask].T @ forces[:,t]
    
    def plot_vibrational_energy(self, force_energy_correction=False, **kwargs):
        """
        Plots the vibrational energy for visualisation purposes.
        """
        if force_energy_correction and self.force_energy_correction is None:
            raise SolutionError('There is no force correction available')
        if self.vib_pe is None or self.vib_ke is None:
            self.get_vib_energy(**kwargs)
            
        plt.plot(self.time_array, self.vib_pe, 'g-', linewidth=.5, label='Potential energy')
        plt.plot(self.time_array, self.vib_ke, 'r-', linewidth=.5, label='Kinetic energy')

        if force_energy_correction:
            plt.plot(self.time_array, self.vib_ke + self.vib_pe - self.force_energy_correction, 'b-', label='Total force-corrected energy')
        else:
            plt.plot(self.time_array, self.vib_ke + self.vib_pe, 'b-', label='Total energy')

        plt.title('Energy of the modelled system')
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.legend()
        plt.show()


# Auxilliary functions:
def get_aka_axis(label):
    """
    Helper function to get the akantu axis label.
    """
    if label in ['x', 'X', '_x', '_X']:
        return aka._x
    if label in ['y', 'Y', '_y', '_Y']:
        return aka._y
    if label in ['z', 'Z', '_z', '_Z']:
        return aka._z
    else:
        raise AxisError('The specified label does not correspond to a known axis label')


def get_alpha_beta(a, b, phi, theta):
    """
    Helper function to compute alpha and beta at a given solution step.
    
    Params as given in report.
    """
    c = a * np.sin(phi) + b * np.cos(theta)
    d = a * np.cos(phi) - b * np.sin(theta)

    alpha = np.sqrt(np.power(c, 2) + np.power(d, 2))
    beta = np.arctan2(c, d)

    return alpha, beta
