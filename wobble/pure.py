import numpy as np
from .ma_fundamentals import MAFundamentals
from .ma_exceptions import *


class PureMA(MAFundamentals):
    """
    Class implementing pure modal analysis solver. Inherits from MAFundamentals.
    
    Methods:
        solve
    
        get_energy
    
        plot_energy
    """
    def __init__(self,
                 object_name,
                 dimensions,
                 **kwargs):
        ref = kwargs.pop('rigid_body_motion', None)
        del ref
        super().__init__(object_name, dimensions, rigid_body_motion=False, **kwargs)

        self.ke = None
        self.pe = None

    def solve(self, save_physical_vibrations=False, **kwargs):
        """
        Shorthand function which deals with mesh creation, finding eigenmodes, and solving the system to get physical displacements.
        """
        self.initialize_model(**kwargs)
        self.assemble_stiffness(**kwargs)
        self.assemble_mass(**kwargs)
        self.find_eigenmodes(**kwargs)
        self.project_initial_displacement(**kwargs)
        self.project_initial_velocity(**kwargs)
        self.project_force(**kwargs)
        self.solve_step_loading(**kwargs)
        self.get_r_and_r_dot(**kwargs)
        self.get_displacement_vectors(**kwargs)

        if save_physical_vibrations:
            self.save_physical_vibrations(**kwargs)

    def get_energy(self, **kwargs):
        """
        Computes the energy for modal analysis.
        """
        super().get_vib_energy(**kwargs)
        self.ke = self.vib_ke
        self.pe = self.vib_pe

    def plot_energy(self, force_energy_correction=False, **kwargs):
        """
        Plots the energy for modal analysis.
        """
        super().plot_vibrational_energy(force_energy_correction=force_energy_correction, **kwargs)
